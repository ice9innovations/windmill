#!/usr/bin/env python3
"""
Registry Sweeper — admin process that evicts stale worker_registry rows on a schedule.

Runs independently of all ML workers. No RabbitMQ dependency — DB only.
Marks online rows whose last_heartbeat is older than STALE_THRESHOLD as offline.
Runs every REGISTRY_SWEEP_INTERVAL seconds (default 60).

Managed by windmill.sh like any other worker.
"""
import os

from db_worker import DbWorker


class RegistrySweeperWorker(DbWorker):
    def __init__(self):
        self.sweep_interval = int(os.getenv('REGISTRY_SWEEP_INTERVAL', '60'))
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

    def run_iteration(self, conn):
        swept = _sweep(conn, self.registry, self.logger)
        if swept:
            self.logger.info(f"Sweep complete: {swept} stale worker(s) evicted")
        else:
            self.logger.debug("Sweep complete: all workers healthy")

    def on_shutdown(self):
        self.logger.info("Registry sweeper stopped")


def _sweep(conn, registry, logger):
    """Mark stale online rows as offline. Returns count swept."""
    swept = registry.sweep_stale(conn, return_rows=True)
    for worker_id, service, host in swept:
        logger.info(f"Swept stale worker offline: {service} on {host} ({worker_id})")
    return len(swept)


if __name__ == '__main__':
    RegistrySweeperWorker().run()
