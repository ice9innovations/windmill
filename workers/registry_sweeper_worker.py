#!/usr/bin/env python3
"""
Registry Sweeper — admin process that evicts stale worker_registry rows on a schedule.

Runs independently of all ML workers. No RabbitMQ dependency — DB only.
Marks online rows whose last_heartbeat is older than STALE_THRESHOLD as offline.
Runs every REGISTRY_SWEEP_INTERVAL seconds (default 60). Each iteration also
prunes a bounded batch of expired service_events so operational telemetry has
a finite lifetime without creating a large delete/WAL spike.

Managed by windmill.sh like any other worker.
"""
import os

from db_worker import DbWorker


class RegistrySweeperWorker(DbWorker):
    def __init__(self):
        self.sweep_interval = int(os.getenv('REGISTRY_SWEEP_INTERVAL', '60'))
        self.service_events_retention_days = int(
            os.getenv('SERVICE_EVENTS_RETENTION_DAYS', '14')
        )
        self.service_events_retention_batch_size = int(
            os.getenv('SERVICE_EVENTS_RETENTION_BATCH_SIZE', '10000')
        )
        super().__init__(
            'registry_sweeper',
            interval_seconds=self.sweep_interval,
        )
        self.stale_threshold = self.heartbeat_interval * 3

    def registry_stale_threshold(self):
        return self.heartbeat_interval * 3

    def on_startup(self):
        self.logger.info(f"Starting registry sweeper ({self.worker_id})")
        self.logger.info(
            f"Sweep interval: {self.sweep_interval}s  Stale threshold: {self.stale_threshold}s"
        )
        self.logger.info(
            "Service event retention: "
            f"{self.service_events_retention_days} days, "
            f"{self.service_events_retention_batch_size} rows/iteration"
        )

    def run_iteration(self, conn):
        swept = _sweep(conn, self.registry, self.logger)
        if swept:
            self.logger.info(f"Sweep complete: {swept} stale worker(s) evicted")
        else:
            self.logger.debug("Sweep complete: all workers healthy")

        pruned = _prune_service_events(
            conn,
            retention_days=self.service_events_retention_days,
            batch_size=self.service_events_retention_batch_size,
        )
        if pruned:
            self.logger.info(f"Pruned {pruned} expired service event(s)")
        else:
            self.logger.debug("Service event retention: nothing expired")

    def on_shutdown(self):
        self.logger.info("Registry sweeper stopped")


def _sweep(conn, registry, logger):
    """Mark stale online rows as offline. Returns count swept."""
    swept = registry.sweep_stale(conn, return_rows=True)
    for worker_id, service, host in swept:
        logger.info(f"Swept stale worker offline: {service} on {host} ({worker_id})")
    return len(swept)


def _prune_service_events(conn, *, retention_days, batch_size):
    """Delete one bounded batch of expired service events."""
    if retention_days < 1:
        raise ValueError("SERVICE_EVENTS_RETENTION_DAYS must be at least 1")
    if batch_size < 1:
        raise ValueError("SERVICE_EVENTS_RETENTION_BATCH_SIZE must be at least 1")

    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            WITH expired AS (
                SELECT event_id
                FROM service_events
                WHERE created_at < NOW() - make_interval(days => %s)
                ORDER BY created_at, event_id
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            )
            DELETE FROM service_events AS service_event
            USING expired
            WHERE service_event.event_id = expired.event_id
            """,
            (retention_days, batch_size),
        )
        return cursor.rowcount
    finally:
        cursor.close()


if __name__ == '__main__':
    RegistrySweeperWorker().run()
