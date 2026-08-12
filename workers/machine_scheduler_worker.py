#!/usr/bin/env python3
"""
MachineSchedulerWorker - per-machine RabbitMQ capacity controller.

This runner lets one box be eligible for many service queues while accepting
only WINDMILL_WORKER_CAPACITY messages total. It is intended for constrained
edge devices; leaving WINDMILL_WORKER_CAPACITY blank keeps the existing
one-process-per-worker behavior.
"""
import importlib
import inspect
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pika
from dotenv import load_dotenv

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from base_worker import BaseWorker
from core.rabbitmq_connection import declare_queue
from core.postgres_connection import close_quietly


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = REPO_ROOT / ".windmill_state"
SCHEDULER_STATUS_FILE = REPO_ROOT / ".windmill_scheduler_status"


def _split_names(value):
    if not value:
        return []
    names = []
    for part in value.replace(",", " ").split():
        part = part.strip()
        if part:
            names.append(part)
    return names


def _state_worker_names():
    if not STATE_FILE.exists():
        return []
    return [
        line.strip()
        for line in STATE_FILE.read_text().splitlines()
        if line.strip()
    ]


def _worker_module_name(name):
    aliases = {
        "caption_score": "caption_score_worker",
        "colors_post": "colors_post_worker",
        "face": "face_worker",
        "pose": "pose_worker",
    }
    name = aliases.get(name, name)
    if name.endswith(".py"):
        name = name[:-3]
    if not name.endswith("_worker"):
        name = f"{name}_worker"
    return name


def _load_worker(name):
    module_name = _worker_module_name(name)
    module = importlib.import_module(module_name)
    candidates = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is BaseWorker:
            continue
        if issubclass(obj, BaseWorker) and obj.__module__ == module.__name__:
            candidates.append(obj)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one BaseWorker subclass in {module_name}, found {len(candidates)}"
        )
    return candidates[0]()


class MachineScheduler:
    def __init__(self):
        load_dotenv(REPO_ROOT / ".env")
        self.capacity = self._load_capacity()
        self.poll_interval = float(os.getenv("WINDMILL_SCHEDULER_POLL_INTERVAL", "0.02"))
        self.enabled_names = self._load_enabled_names()
        self.queue_priority = _split_names(os.getenv("WINDMILL_SCHEDULER_QUEUE_PRIORITY", ""))
        self.logger = self._setup_logging()
        self.slots = []
        self.running = True

    def _setup_logging(self):
        logger = logging.getLogger("machine_scheduler")
        logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()))
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(handler)
        return logger

    def _load_capacity(self):
        raw = os.getenv("WINDMILL_WORKER_CAPACITY", "").strip()
        if not raw:
            raise ValueError("WINDMILL_WORKER_CAPACITY is blank; use normal workers instead")
        capacity = int(raw)
        if capacity < 1:
            raise ValueError("WINDMILL_WORKER_CAPACITY must be a positive integer")
        return capacity

    def _load_enabled_names(self):
        names = _split_names(os.getenv("WINDMILL_ENABLED_WORKERS", ""))
        if names:
            return names
        return _state_worker_names()

    def _handle_signal(self, _signum, _frame):
        self.running = False

    def _startup_worker(self, name, slot_id, register):
        worker = _load_worker(name)
        worker.worker_id = f"{worker.worker_id}_slot{slot_id}"
        worker._registry.worker_id = worker.worker_id
        worker._scheduler_registered = False
        if not worker.connect_to_database():
            raise RuntimeError(f"{name}: database connection failed")
        if not worker.warm_image_store_connection():
            raise RuntimeError(f"{name}: image store connection failed")
        worker._running = True
        worker._async_publisher.start()
        if register:
            worker._start_registry()
            worker._scheduler_registered = True
        return worker

    def startup(self):
        if not self.enabled_names:
            raise RuntimeError(
                "No enabled workers. Set WINDMILL_ENABLED_WORKERS or enable workers with windmill.sh."
            )
        self.logger.info(
            "Starting machine scheduler capacity=%s enabled=%s",
            self.capacity,
            ",".join(self.enabled_names),
        )
        self.slots = [
            SchedulerSlot(
                slot_id=slot_id,
                workers=[
                    self._startup_worker(name, slot_id, register=(slot_id == 0))
                    for name in self.enabled_names
                ],
                poll_interval=self.poll_interval,
                queue_priority=self.queue_priority,
                logger=self.logger,
            )
            for slot_id in range(self.capacity)
        ]
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def run(self):
        self.startup()
        for slot in self.slots:
            slot.start()
        while self.running:
            self._write_status("running")
            failed_slots = [slot for slot in self.slots if slot.failed_exception is not None]
            if failed_slots:
                for slot in failed_slots:
                    self.logger.error(
                        "Slot %s stopped unexpectedly: %s",
                        slot.slot_id,
                        slot.failed_exception,
                    )
                self.running = False
                break
            time.sleep(0.5)
        self.shutdown()

    def _write_status(self, state):
        workers = []
        if self.slots:
            workers = [
                {
                    "worker": _worker_module_name(worker.service_name.split(".", 1)[-1]),
                    "service": worker.service_name,
                    "queue": worker.queue_name,
                }
                for worker in self.slots[0].workers
            ]
        payload = {
            "state": state,
            "capacity": self.capacity,
            "updated_at_epoch": time.time(),
            "enabled_workers": self.enabled_names,
            "managed": workers,
            "slots": [slot.status() for slot in self.slots],
        }
        tmp_path = SCHEDULER_STATUS_FILE.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            tmp_path.replace(SCHEDULER_STATUS_FILE)
        except Exception as exc:
            self.logger.warning("Failed to write scheduler status: %s", exc)

    def shutdown(self):
        self.logger.info("Stopping machine scheduler")
        self._write_status("stopping")
        for slot in self.slots:
            slot.stop()
        for slot in self.slots:
            slot.join(timeout=10)
        for worker in [worker for slot in self.slots for worker in slot.workers]:
            worker._running = False
            worker._async_publisher.stop(join_timeout=10)
            if getattr(worker, "_scheduler_registered", False):
                worker._stop_registry()
            worker._cleanup()
            close_quietly(getattr(worker, "read_db_conn", None))
            close_quietly(worker.db_conn)
            worker._sync_publish_queue.close()
        self._write_status("stopped")
        self.logger.info("Machine scheduler stopped")


class SchedulerSlot:
    def __init__(self, *, slot_id, workers, poll_interval, queue_priority, logger):
        self.slot_id = slot_id
        self.workers = self._prioritize_workers(workers, queue_priority)
        self.poll_interval = poll_interval
        self.logger = logger
        self.running = threading.Event()
        self.connection = None
        self.channel = None
        self.failed_exception = None
        self.current_job = None
        self.status_lock = threading.Lock()
        self.thread = threading.Thread(
            target=self.run,
            daemon=True,
            name=f"machine_scheduler_slot_{slot_id}",
        )

    def start(self):
        if self.thread.is_alive():
            return
        self.running.set()
        self.thread.start()

    def stop(self):
        self.running.clear()

    def join(self, timeout=None):
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def is_alive(self):
        return self.thread.is_alive()

    def status(self):
        with self.status_lock:
            current_job = dict(self.current_job) if self.current_job else None
        return {
            "slot": self.slot_id,
            "state": "active" if current_job else "idle",
            "current_job": current_job,
            "alive": self.thread.is_alive(),
            "failed": str(self.failed_exception) if self.failed_exception else None,
        }

    def _set_current_job(self, job):
        with self.status_lock:
            self.current_job = job

    def _prioritize_workers(self, workers, queue_priority):
        if not queue_priority:
            return list(workers)

        priority = {}
        for index, name in enumerate(queue_priority):
            module_name = _worker_module_name(name)
            short_name = module_name[:-7] if module_name.endswith("_worker") else module_name
            priority[name] = index
            priority[module_name] = index
            priority[short_name] = index

        def sort_key(worker):
            module_name = _worker_module_name(worker.service_name.split(".", 1)[-1])
            short_name = module_name[:-7] if module_name.endswith("_worker") else module_name
            return min(
                priority.get(worker.queue_name, 10_000),
                priority.get(module_name, 10_000),
                priority.get(short_name, 10_000),
            )

        return sorted(workers, key=sort_key)

    def _connect_queue(self):
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                pass
        connection_owner = self.workers[0]
        self.connection, self.channel = connection_owner._consume_queue.connect()
        for worker in self.workers:
            worker.connection = self.connection
            worker.channel = self.channel
            declare_queue(
                self.channel,
                worker.queue_name,
                ttl_ms=worker._queue_message_ttl_ms(),
            )
            self.logger.info(
                "Slot %s eligible queue: %s (%s)",
                self.slot_id,
                worker.queue_name,
                worker.service_name,
            )

    def _poll_once(self):
        for worker in self.workers:
            method, properties, body = self.channel.basic_get(
                queue=worker.queue_name,
                auto_ack=False,
            )
            if method is None:
                continue

            self.logger.info(
                "Slot %s dispatching queue=%s service=%s delivery_tag=%s",
                self.slot_id,
                worker.queue_name,
                worker.service_name,
                method.delivery_tag,
            )
            worker.channel = self.channel
            self._set_current_job({
                "service": worker.service_name,
                "queue": worker.queue_name,
                "delivery_tag": method.delivery_tag,
                "started_at_epoch": time.time(),
            })
            try:
                worker.process_message(self.channel, method, properties, body)
            finally:
                self._set_current_job(None)
            return True
        return False

    def _poll_forever(self):
        self._connect_queue()
        self.logger.info(
            "Slot %s polling RabbitMQ queues in order: %s",
            self.slot_id,
            ",".join(worker.queue_name for worker in self.workers),
        )
        while self.running.is_set():
            if not self._poll_once():
                time.sleep(self.poll_interval)

    def run(self):
        try:
            while self.running.is_set():
                try:
                    self._poll_forever()
                    if self.running.is_set():
                        self.logger.warning("Slot %s polling loop stopped unexpectedly; reconnecting", self.slot_id)
                        time.sleep(2)
                except (pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError,
                        pika.exceptions.StreamLostError) as exc:
                    if self.running.is_set():
                        self.logger.warning(
                            "Slot %s queue connection lost: %s. Reconnecting...",
                            self.slot_id,
                            exc,
                        )
                        time.sleep(2)
                        self._connect_queue()
        except Exception as exc:
            self.failed_exception = exc
            self.logger.error("Slot %s crashed", self.slot_id, exc_info=True)
        finally:
            if self.connection is not None:
                try:
                    self.connection.close()
                except Exception:
                    pass


if __name__ == "__main__":
    scheduler = MachineScheduler()
    scheduler.run()
