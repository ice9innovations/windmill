#!/usr/bin/env python3
"""
Shared worker-registry lifecycle for Windmill processes.

This owns one concern only:
- register a worker row
- keep its heartbeat fresh
- mark it offline on clean shutdown
- optionally sweep stale online rows
"""

import socket
import threading
from typing import Optional

from core.postgres_connection import close_quietly


class ManagedWorkerRegistry:
    """Own worker_registry registration and heartbeat lifecycle."""

    def __init__(
        self,
        *,
        connection_factory,
        logger,
        worker_id: str,
        service: str,
        heartbeat_interval: int,
        host: Optional[str] = None,
        stale_threshold: Optional[int] = None,
    ):
        self.connection_factory = connection_factory
        self.logger = logger
        self.worker_id = worker_id
        self.service = service
        self.host = host or socket.gethostname()
        self.heartbeat_interval = heartbeat_interval
        self.stale_threshold = stale_threshold
        self._stop_event = threading.Event()
        self._thread = None

    def register(self, conn):
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                UPDATE worker_registry
                SET status = 'offline', offline_at = last_heartbeat
                WHERE service = %s AND host = %s AND status = 'online'
                """,
                (self.service, self.host),
            )
            if self.stale_threshold:
                cursor.execute(
                    """
                    UPDATE worker_registry
                    SET status = 'offline', offline_at = last_heartbeat
                    WHERE status = 'online'
                      AND last_heartbeat < NOW() - INTERVAL '%s seconds'
                    """,
                    (self.stale_threshold,),
                )
            cursor.execute(
                """
                INSERT INTO worker_registry (worker_id, service, host, started_at, last_heartbeat, status)
                VALUES (%s, %s, %s, NOW(), NOW(), 'online')
                """,
                (self.worker_id, self.service, self.host),
            )
        finally:
            close_quietly(cursor)

    def sweep_stale(self, conn, *, return_rows: bool = False):
        if not self.stale_threshold:
            return [] if return_rows else 0
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                UPDATE worker_registry
                SET status = 'offline', offline_at = last_heartbeat
                WHERE status = 'online'
                  AND worker_id <> %s
                  AND last_heartbeat < NOW() - INTERVAL '%s seconds'
                RETURNING worker_id, service, host
                """,
                (self.worker_id, self.stale_threshold),
            )
            rows = cursor.fetchall()
            return rows if return_rows else len(rows)
        finally:
            close_quietly(cursor)

    def mark_offline(self, conn):
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE worker_registry SET status = 'offline', offline_at = NOW() WHERE worker_id = %s",
                (self.worker_id,),
            )
        finally:
            close_quietly(cursor)

    def _heartbeat_loop(self):
        conn = None
        try:
            conn = self.connection_factory(autocommit=True)
        except Exception as e:
            self.logger.warning(f"Heartbeat thread failed to connect to DB: {e}")
            return

        try:
            while not self._stop_event.wait(self.heartbeat_interval):
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE worker_registry SET last_heartbeat = NOW() WHERE worker_id = %s",
                        (self.worker_id,),
                    )
                    close_quietly(cursor)
                    self.logger.debug("Heartbeat sent")
                except Exception as e:
                    self.logger.warning(f"Heartbeat failed: {e}. Reconnecting...")
                    close_quietly(conn)
                    conn = None
                    try:
                        conn = self.connection_factory(autocommit=True)
                    except Exception as reconnect_e:
                        self.logger.error(f"Heartbeat reconnect failed: {reconnect_e}")

            if conn is not None:
                try:
                    self.mark_offline(conn)
                    self.logger.info("Marked offline in worker registry")
                except Exception as e:
                    self.logger.warning(f"Failed to mark offline in worker registry: {e}")
        except Exception as e:
            self.logger.error(f"Heartbeat thread crashed unexpectedly: {e}", exc_info=True)
        finally:
            close_quietly(conn)

    def start(self, conn):
        self._stop_event.clear()
        self.register(conn)
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"{self.service}_heartbeat",
        )
        self._thread.start()

    def stop(self, join_timeout: float = 5):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=join_timeout)
