#!/usr/bin/env python3
"""
Registry Sweeper — admin process that evicts stale worker_registry rows on a schedule.

Runs independently of all ML workers. No RabbitMQ dependency — DB only.
Marks online rows whose last_heartbeat is older than STALE_THRESHOLD as offline.
Runs every REGISTRY_SWEEP_INTERVAL seconds (default 60).

Managed by windmill.sh like any other worker.
"""
import os
import sys
import time
import socket
import signal
import logging
import threading
import psycopg2
from dotenv import load_dotenv

# Workers run from the project root via windmill.sh
if not load_dotenv('.env'):
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    if not load_dotenv(env_path):
        raise ValueError("Could not load .env file")

SERVICE_NAME = 'registry_sweeper'
WORKER_ID    = f"worker_{SERVICE_NAME}_{int(time.time())}"
HOST         = socket.gethostname()

HEARTBEAT_INTERVAL = int(os.getenv('WORKER_HEARTBEAT_INTERVAL', '30'))
SWEEP_INTERVAL     = int(os.getenv('REGISTRY_SWEEP_INTERVAL', '60'))
STALE_THRESHOLD    = HEARTBEAT_INTERVAL * 3  # seconds

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper())
)
logger = logging.getLogger(SERVICE_NAME)

_stop_event = threading.Event()


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
    """Mark previous row for this (service, host) offline, run initial stale sweep, insert fresh row."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE worker_registry
        SET status = 'offline', offline_at = last_heartbeat
        WHERE service = %s AND host = %s AND status = 'online'
    """, (SERVICE_NAME, HOST))
    cur.execute("""
        UPDATE worker_registry
        SET status = 'offline', offline_at = last_heartbeat
        WHERE status = 'online'
          AND worker_id <> %s
          AND last_heartbeat < NOW() - INTERVAL '%s seconds'
    """, (WORKER_ID, STALE_THRESHOLD))
    cur.execute("""
        INSERT INTO worker_registry (worker_id, service, host, started_at, last_heartbeat, status)
        VALUES (%s, %s, %s, NOW(), NOW(), 'online')
    """, (WORKER_ID, SERVICE_NAME, HOST))
    cur.close()
    logger.info(f"Registered in worker registry ({HOST})")


def _heartbeat_loop(conn):
    """Background thread: keeps this sweeper's own registry row fresh."""
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

        # Clean shutdown — mark offline with precise timestamp
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
        logger.error(f"Heartbeat thread crashed unexpectedly: {e}", exc_info=True)


def _sweep(conn):
    """Mark stale online rows as offline. Returns count swept."""
    cur = conn.cursor()
    cur.execute("""
        UPDATE worker_registry
        SET status = 'offline', offline_at = last_heartbeat
        WHERE status = 'online'
          AND worker_id <> %s
          AND last_heartbeat < NOW() - INTERVAL '%s seconds'
        RETURNING worker_id, service, host
    """, (WORKER_ID, STALE_THRESHOLD))
    swept = cur.fetchall()
    cur.close()
    for worker_id, service, host in swept:
        logger.info(f"Swept stale worker offline: {service} on {host} ({worker_id})")
    return len(swept)


def _handle_sigterm(signum, frame):
    raise KeyboardInterrupt("SIGTERM received")


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)

    logger.info(f"Starting registry sweeper ({WORKER_ID})")
    logger.info(f"Sweep interval: {SWEEP_INTERVAL}s  Stale threshold: {STALE_THRESHOLD}s")

    try:
        conn = _db_connect(autocommit=True)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

    _register(conn)

    hb_conn = _db_connect(autocommit=True)
    hb_thread = threading.Thread(
        target=_heartbeat_loop, args=(hb_conn,),
        daemon=True, name="heartbeat"
    )
    hb_thread.start()

    logger.info(f"Running. Press CTRL+C to exit.")
    try:
        while not _stop_event.is_set():
            swept = _sweep(conn)
            if swept:
                logger.info(f"Sweep complete: {swept} stale worker(s) evicted")
            else:
                logger.debug("Sweep complete: all workers healthy")
            _stop_event.wait(SWEEP_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        _stop_event.set()
        hb_thread.join(timeout=5)
        try:
            conn.close()
        except Exception:
            pass
        logger.info("Registry sweeper stopped")


if __name__ == '__main__':
    main()
