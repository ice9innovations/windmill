#!/usr/bin/env python3
"""
Availability Snapshot Worker — records per-service availability once per minute.

Reads all known customer-facing services from service_config.yaml, checks worker_registry for at least
one online worker with a fresh heartbeat, and writes one row per service to
worker_availability_log. Runs every SNAPSHOT_INTERVAL seconds (default 60).

Managed by windmill.sh like any other worker. No RabbitMQ dependency — DB only.
"""
from service_config import get_service_config
import os

from db_worker import DbWorker

def _known_services():
    return get_service_config().get_all_tiered_service_names()

class AvailabilitySnapshotWorker(DbWorker):
    def __init__(self):
        self.snapshot_interval = int(os.getenv('AVAILABILITY_SNAPSHOT_INTERVAL', '60'))
        super().__init__(
            'availability_snapshot',
            interval_seconds=self.snapshot_interval,
        )
        self.heartbeat_window = self.heartbeat_interval * 3
        self.services = []

    def on_startup(self):
        self.logger.info(f"Starting availability snapshot worker ({self.worker_id})")
        self.logger.info(
            f"Snapshot interval: {self.snapshot_interval}s  Heartbeat window: {self.heartbeat_window}s"
        )
        try:
            self.services = _known_services()
        except Exception as e:
            self.logger.error(f"Failed to load service config: {e}")
            raise SystemExit(1)
        self.logger.info(f"Tracking {len(self.services)} services: {', '.join(self.services)}")

    def run_iteration(self, conn):
        _snapshot(conn, self.services, self.heartbeat_window, self.worker_id, self.logger)

    def on_shutdown(self):
        self.logger.info("Availability snapshot worker stopped")


def _snapshot(conn, services, heartbeat_window, worker_id, logger):
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT service
        FROM worker_registry
        WHERE status = 'online'
          AND last_heartbeat >= NOW() - INTERVAL '%s seconds'
          AND worker_id <> %s
    """, (heartbeat_window, worker_id))
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


if __name__ == '__main__':
    AvailabilitySnapshotWorker().run()
