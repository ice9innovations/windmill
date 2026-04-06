#!/usr/bin/env python3
"""
Thin runtime scaffold for DB-only Windmill workers.

Use this for operational workers that need:
- .env loading
- logging
- a managed Postgres connection
- worker_registry registration / heartbeat
- SIGTERM handling
- a recurring loop

This is intentionally narrow. It is not a second BaseWorker.
"""

import logging
import os
import signal
import socket
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.postgres_connection import PostgresConnectionConfig, create_connection
from core.worker_registry import ManagedWorkerRegistry


class DbWorker(ABC):
    def __init__(
        self,
        service_name: str,
        *,
        interval_seconds: int,
        heartbeat_interval: Optional[int] = None,
        env_file: str = '.env',
    ):
        self.service_name = service_name
        self.worker_id = f"worker_{service_name}_{int(time.time())}"
        self.host = socket.gethostname()
        self.interval_seconds = interval_seconds
        self.heartbeat_interval = heartbeat_interval or int(os.getenv('WORKER_HEARTBEAT_INTERVAL', '30'))

        self._load_env(env_file)
        self._setup_logging()
        self.stop_event = threading.Event()

        self.db_config = PostgresConnectionConfig(
            host=self._get_required('DB_HOST'),
            database=self._get_required('DB_NAME'),
            user=self._get_required('DB_USER'),
            password=self._get_required('DB_PASSWORD'),
            sslmode=os.getenv('DB_SSLMODE'),
        )
        self.registry = ManagedWorkerRegistry(
            connection_factory=self._db_connect,
            logger=self.logger,
            worker_id=self.worker_id,
            service=self.service_name,
            heartbeat_interval=self.heartbeat_interval,
            host=self.host,
            stale_threshold=self.registry_stale_threshold(),
        )

    def _load_env(self, env_file):
        if load_dotenv(env_file):
            return
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', env_file)
        if not load_dotenv(env_path):
            raise ValueError(f"Could not load {env_file} file")

    def _setup_logging(self):
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper()),
        )
        self.logger = logging.getLogger(self.service_name)

    def _get_required(self, key):
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value

    def _db_connect(self, autocommit=True):
        return create_connection(self.db_config, autocommit=autocommit)

    def registry_stale_threshold(self):
        return None

    def on_startup(self):
        pass

    def on_connected(self, conn):
        pass

    @abstractmethod
    def run_iteration(self, conn):
        raise NotImplementedError

    def on_iteration_error(self, exc):
        self.logger.error(f"Iteration failed: {exc}", exc_info=True)

    def on_shutdown(self):
        pass

    def _handle_sigterm(self, signum, frame):
        raise KeyboardInterrupt("SIGTERM received")

    def run(self):
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        self.on_startup()

        try:
            conn = self._db_connect(autocommit=True)
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            sys.exit(1)

        try:
            self.registry.start(conn)
            self.logger.info(f"Registered in worker registry ({self.host})")
        except Exception as e:
            self.logger.error(f"Failed to register in worker registry: {e}")
            sys.exit(1)

        try:
            self.on_connected(conn)
            self.logger.info("Running. Press CTRL+C to exit.")
            while not self.stop_event.is_set():
                try:
                    self.run_iteration(conn)
                except Exception as e:
                    self.on_iteration_error(e)
                    try:
                        conn.close()
                        conn = self._db_connect(autocommit=True)
                    except Exception:
                        pass
                self.stop_event.wait(self.interval_seconds)
        except KeyboardInterrupt:
            self.logger.info("Stopping...")
        finally:
            self.stop_event.set()
            self.registry.stop(join_timeout=5)
            try:
                conn.close()
            except Exception:
                pass
            self.on_shutdown()
