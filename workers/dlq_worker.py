#!/usr/bin/env python3
"""
DLQ Consumer — writes 'dead-lettered' status to service_dispatch for messages that
exhausted retries in primary queues.

Polls all {queue_name}.dlq queues in a round-robin. For each message:
  1. Parses image_id from the message body.
  2. Derives service name from the DLQ name.
  3. Writes 'dead-lettered' status (with failure reason from x-death headers) to
     service_dispatch.
  4. Acks the message — removes it from the DLQ permanently.

This completes the job lifecycle event chain:
  pending        → written by api.py / producer on job submission
  complete       → written by primary workers on success
  failed         → written by primary workers on DB-error nack
  dead-lettered  → written here for messages that exhausted all retries

Without this worker, exhausted-retry messages leave service_dispatch rows stuck at
'pending' permanently and the stale detection system in api.py cannot distinguish
them from slow-but-still-running jobs.

Managed by windmill.sh like any other worker. Runs on one machine only — DLQ writes
are idempotent (UPDATE WHERE status = 'pending'), so running on multiple is safe but
wasteful.
"""
import os
import sys
import time
import json
import socket
import signal
import logging
import threading
import psycopg2
import pika
import pika.exceptions
from dotenv import load_dotenv

# Workers run from the project root via windmill.sh
if not load_dotenv('.env'):
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    if not load_dotenv(env_path):
        raise ValueError("Could not load .env file")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from service_config import get_service_config

SERVICE_NAME = 'dlq_consumer'
WORKER_ID    = f"worker_{SERVICE_NAME}_{int(time.time())}"
HOST         = socket.gethostname()

HEARTBEAT_INTERVAL = int(os.getenv('WORKER_HEARTBEAT_INTERVAL', '30'))
POLL_INTERVAL      = float(os.getenv('DLQ_POLL_INTERVAL', '5.0'))  # seconds between empty rounds

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper())
)
logger = logging.getLogger(SERVICE_NAME)

_stop_event = threading.Event()


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _db_connect(autocommit=True):
    kwargs = dict(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        connect_timeout=10,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=3,
    )
    sslmode = os.getenv('DB_SSLMODE')
    if sslmode:
        kwargs['sslmode'] = sslmode
    conn = psycopg2.connect(**kwargs)
    conn.autocommit = autocommit
    return conn


def _register(conn):
    cur = conn.cursor()
    cur.execute("""
        UPDATE worker_registry
        SET status = 'offline', offline_at = last_heartbeat
        WHERE service = %s AND host = %s AND status = 'online'
    """, (SERVICE_NAME, HOST))
    cur.execute("""
        INSERT INTO worker_registry (worker_id, service, host, started_at, last_heartbeat, status)
        VALUES (%s, %s, %s, NOW(), NOW(), 'online')
    """, (WORKER_ID, SERVICE_NAME, HOST))
    cur.close()
    logger.info(f"Registered in worker registry ({HOST})")


def _heartbeat_loop(conn):
    try:
        while not _stop_event.wait(HEARTBEAT_INTERVAL):
            try:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE worker_registry SET last_heartbeat = NOW() WHERE worker_id = %s",
                    (WORKER_ID,)
                )
                cur.close()
                logger.debug("Heartbeat sent")
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}. Reconnecting...")
                try:
                    conn.close()
                    conn = _db_connect(autocommit=True)
                except Exception as reconnect_e:
                    logger.error(f"Heartbeat reconnect failed: {reconnect_e}")

        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE worker_registry SET status = 'offline', offline_at = NOW() WHERE worker_id = %s",
                (WORKER_ID,)
            )
            cur.close()
            logger.info("Marked offline in worker registry")
        except Exception as e:
            logger.warning(f"Failed to mark offline: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Heartbeat thread crashed: {e}", exc_info=True)


def _write_dead_lettered(db_conn, image_id, service, reason):
    """Best-effort UPDATE — swallows errors so a bad row never stalls the consumer."""
    try:
        cur = db_conn.cursor()
        cur.execute(
            """UPDATE service_dispatch
               SET status = 'dead-lettered', failed_reason = %s
               WHERE image_id = %s AND service = %s AND status = 'pending'""",
            (reason, image_id, service),
        )
        cur.close()
        logger.info(f"Marked dead-lettered: {service}/{image_id}  reason={reason!r}")
    except Exception as e:
        logger.warning(f"Failed to write dead-lettered for {service}/{image_id}: {e}")


def _write_dead_lettered_by_dispatch_id(db_conn, dispatch_id, reason):
    """Best-effort targeted UPDATE for multi-dispatch downstream rows."""
    try:
        cur = db_conn.cursor()
        cur.execute(
            """UPDATE service_dispatch
               SET status = 'dead-lettered', failed_reason = %s
               WHERE dispatch_id = %s AND status = 'pending'""",
            (reason, dispatch_id),
        )
        cur.close()
        logger.info(f"Marked dead-lettered dispatch_id={dispatch_id} reason={reason!r}")
    except Exception as e:
        logger.warning(f"Failed to write dead-lettered for dispatch_id={dispatch_id}: {e}")


# ---------------------------------------------------------------------------
# Queue helpers
# ---------------------------------------------------------------------------

def _build_queue_params():
    host     = os.getenv('QUEUE_HOST', 'localhost')
    port     = int(os.getenv('QUEUE_PORT', '5672'))
    user     = os.getenv('QUEUE_USER', 'guest')
    password = os.getenv('QUEUE_PASSWORD', 'guest')
    use_ssl  = os.getenv('QUEUE_SSL', 'true').lower() not in ('false', '0', 'no')

    creds = pika.PlainCredentials(user, password)
    if use_ssl and port != 5672:
        import ssl
        ctx = ssl.create_default_context()
        return pika.ConnectionParameters(
            host=host, port=port, credentials=creds,
            ssl_options=pika.SSLOptions(ctx),
            heartbeat=60, blocked_connection_timeout=30,
        )
    return pika.ConnectionParameters(
        host=host, port=port, credentials=creds,
        heartbeat=60, blocked_connection_timeout=30,
    )


def _get_all_dlq_names():
    """Return list of (dlq_name, service_name) pairs for every service in service_config."""
    config = get_service_config()
    pairs = []
    seen_queues = set()
    for category, services in config.raw_config['services'].items():
        if not services:
            continue
        for service_name, service_cfg in services.items():
            if not service_cfg:
                continue
            queue_name = service_cfg.get('queue_name', service_name)
            if queue_name in seen_queues:
                continue
            seen_queues.add(queue_name)
            pairs.append((f"{queue_name}.dlq", queue_name))
    return pairs


def _extract_reason(properties):
    """Build a human-readable failure reason from RabbitMQ x-death headers."""
    try:
        headers = properties.headers or {}
        x_death = headers.get('x-death', [])
        if x_death:
            entry = x_death[0]
            reason  = entry.get('reason', 'unknown')
            count   = entry.get('count', '?')
            queue   = entry.get('queue', '?')
            return f"{reason} after {count} attempt(s) in {queue}"
    except Exception:
        pass
    return "dead-lettered (reason unknown)"


def _poll_once(ch, db_conn, dlq_pairs):
    """One full round: basic_get from every DLQ. Returns count processed."""
    processed = 0
    for dlq_name, service_name in dlq_pairs:
        if _stop_event.is_set():
            break
        try:
            method, properties, body = ch.basic_get(dlq_name, auto_ack=False)
        except Exception as e:
            logger.warning(f"basic_get failed for {dlq_name}: {e}")
            continue

        if method is None:
            continue  # queue empty

        try:
            message = json.loads(body)
            image_id = message.get('image_id')
            if image_id is None:
                logger.warning(f"DLQ message on {dlq_name} has no image_id — acking without status write")
            else:
                reason = _extract_reason(properties)
                _write_dead_lettered(db_conn, image_id, service_name, reason)
            ch.basic_ack(method.delivery_tag)
            processed += 1
        except Exception as e:
            logger.error(f"Error processing DLQ message from {dlq_name}: {e}", exc_info=True)
            try:
                ch.basic_nack(method.delivery_tag, requeue=True)
            except Exception:
                pass

    return processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _handle_sigterm(signum, frame):
    raise KeyboardInterrupt("SIGTERM received")


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)

    logger.info(f"Starting DLQ consumer ({WORKER_ID})")
    logger.info(f"Poll interval: {POLL_INTERVAL}s")

    # DB connection
    try:
        db_conn = _db_connect(autocommit=True)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

    _register(db_conn)

    hb_conn = _db_connect(autocommit=True)
    hb_thread = threading.Thread(
        target=_heartbeat_loop, args=(hb_conn,),
        daemon=True, name="heartbeat"
    )
    hb_thread.start()

    # Queue connection
    try:
        queue_conn = pika.BlockingConnection(_build_queue_params())
        ch = queue_conn.channel()
        ch.basic_qos(prefetch_count=1)
    except Exception as e:
        logger.error(f"Failed to connect to queue: {e}")
        _stop_event.set()
        hb_thread.join(timeout=5)
        sys.exit(1)

    # Discover all DLQs and declare them (idempotent — safe to call even if they exist)
    dlq_pairs = _get_all_dlq_names()
    logger.info(f"Monitoring {len(dlq_pairs)} DLQs: {[d for d, _ in dlq_pairs]}")
    for dlq_name, _ in dlq_pairs:
        try:
            ch.queue_declare(queue=dlq_name, durable=True)
        except Exception as e:
            logger.warning(f"Could not declare {dlq_name}: {e}")

    logger.info("Running. Press CTRL+C to exit.")
    try:
        while not _stop_event.is_set():
            try:
                processed = _poll_once(ch, db_conn, dlq_pairs)
                if processed:
                    logger.debug(f"Processed {processed} DLQ message(s)")
            except (pika.exceptions.AMQPConnectionError,
                    pika.exceptions.AMQPChannelError,
                    pika.exceptions.StreamLostError) as e:
                logger.warning(f"Queue connection lost: {e}. Reconnecting...")
                time.sleep(5)
                try:
                    queue_conn = pika.BlockingConnection(_build_queue_params())
                    ch = queue_conn.channel()
                    ch.basic_qos(prefetch_count=1)
                    for dlq_name, _ in dlq_pairs:
                        try:
                            ch.queue_declare(queue=dlq_name, durable=True)
                        except Exception:
                            pass
                    logger.info("Queue connection restored")
                except Exception as reconnect_e:
                    logger.error(f"Queue reconnect failed: {reconnect_e}")
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                logger.warning(f"DB connection lost: {e}. Reconnecting...")
                time.sleep(5)
                try:
                    db_conn = _db_connect(autocommit=True)
                    logger.info("DB connection restored")
                except Exception as reconnect_e:
                    logger.error(f"DB reconnect failed: {reconnect_e}")

            _stop_event.wait(POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        _stop_event.set()
        hb_thread.join(timeout=5)
        try:
            queue_conn.close()
        except Exception:
            pass
        try:
            db_conn.close()
        except Exception:
            pass
        logger.info("DLQ consumer stopped")


if __name__ == '__main__':
    main()
