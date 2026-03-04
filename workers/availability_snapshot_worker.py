#!/usr/bin/env python3
"""
Availability Snapshot Worker — records per-service availability once per minute.

Reads all known customer-facing services from service_config.yaml, checks worker_registry for at least
one online worker with a fresh heartbeat, and writes one row per service to
worker_availability_log. Runs every SNAPSHOT_INTERVAL seconds (default 60).

Managed by windmill.sh like any other worker. No RabbitMQ dependency — DB only.
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
from service_config import get_service_config

if not load_dotenv('.env'):
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    if not load_dotenv(env_path):
        raise ValueError("Could not load .env file")

SERVICE_NAME = 'availability_snapshot'
WORKER_ID    = f"worker_{SERVICE_NAME}_{int(time.time())}"
HOST         = socket.gethostname()

HEARTBEAT_INTERVAL = int(os.getenv('WORKER_HEARTBEAT_INTERVAL', '30'))
SNAPSHOT_INTERVAL  = int(os.getenv('AVAILABILITY_SNAPSHOT_INTERVAL', '60'))
HEARTBEAT_WINDOW   = HEARTBEAT_INTERVAL * 3  # seconds — freshness threshold for online workers


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


def _known_services():
    return get_service_config().get_all_tiered_service_names()


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
        logger.error(f"Heartbeat thread crashed unexpectedly: {e}", exc_info=True)


def _snapshot(conn, services):
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT service
        FROM worker_registry
        WHERE status = 'online'
          AND last_heartbeat >= NOW() - INTERVAL '%s seconds'
          AND worker_id <> %s
    """, (HEARTBEAT_WINDOW, WORKER_ID))
    online = {row[0] for row in cur.fetchall()}

    rows = [(svc, svc in online) for svc in services]
    cur.executemany(
        "INSERT INTO worker_availability_log (service, is_available) VALUES (%s, %s)",
        rows,
    )
    cur.close()

    unavailable = [svc for svc, avail in rows if not avail]
    if unavailable:
        logger.warning(f"Unavailable: {', '.join(unavailable)}")
    else:
        logger.debug(f"All {len(rows)} services available")
    return len(unavailable)


def _handle_sigterm(signum, frame):
    raise KeyboardInterrupt("SIGTERM received")


def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)

    logger.info(f"Starting availability snapshot worker ({WORKER_ID})")
    logger.info(f"Snapshot interval: {SNAPSHOT_INTERVAL}s  Heartbeat window: {HEARTBEAT_WINDOW}s")

    try:
        services = _known_services()
        logger.info(f"Tracking {len(services)} services: {', '.join(services)}")
    except Exception as e:
        logger.error(f"Failed to load service config: {e}")
        sys.exit(1)

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

    logger.info("Running. Press CTRL+C to exit.")
    try:
        while not _stop_event.is_set():
            try:
                _snapshot(conn, services)
            except Exception as e:
                logger.error(f"Snapshot failed: {e}", exc_info=True)
                try:
                    conn.close()
                    conn = _db_connect(autocommit=True)
                except Exception:
                    pass
            _stop_event.wait(SNAPSHOT_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        _stop_event.set()
        hb_thread.join(timeout=5)
        try:
            conn.close()
        except Exception:
            pass
        logger.info("Availability snapshot worker stopped")


if __name__ == '__main__':
    main()
