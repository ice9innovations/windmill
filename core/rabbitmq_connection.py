#!/usr/bin/env python3
"""Shared RabbitMQ connection parameter policy for Windmill workers."""

from dataclasses import dataclass
from collections import deque
from datetime import datetime
from functools import partial
import ssl
import threading
import time
from typing import Optional

import pika


@dataclass(frozen=True)
class RabbitMQConnectionConfig:
    host: str
    port: int
    user: str
    password: str
    use_ssl: bool = False
    server_hostname: Optional[str] = None

    def build_params(
        self,
        *,
        heartbeat: int = 300,
        blocked_connection_timeout: int = 300,
        connection_attempts: int = 3,
        retry_delay: int = 5,
        socket_timeout: int = 10,
        tcp_options: Optional[dict] = None,
        **overrides,
    ):
        credentials = pika.PlainCredentials(self.user, self.password)
        kwargs = dict(
            host=self.host,
            port=self.port,
            credentials=credentials,
            heartbeat=heartbeat,
            blocked_connection_timeout=blocked_connection_timeout,
            connection_attempts=connection_attempts,
            retry_delay=retry_delay,
            socket_timeout=socket_timeout,
            tcp_options=tcp_options or None,
        )
        if self.use_ssl:
            ssl_context = ssl.create_default_context()
            kwargs['ssl_options'] = pika.SSLOptions(
                ssl_context,
                self.server_hostname or self.host,
            )
        kwargs.update(overrides)
        return pika.ConnectionParameters(**kwargs)


def declare_queue(channel, queue_name: str, *, ttl_ms: Optional[int] = None):
    dlq_name = f"{queue_name}.dlq"
    channel.queue_declare(queue=dlq_name, durable=True)
    args = {
        'x-dead-letter-exchange': '',
        'x-dead-letter-routing-key': dlq_name,
    }
    if ttl_ms and ttl_ms > 0:
        args['x-message-ttl'] = ttl_ms
    channel.queue_declare(queue=queue_name, durable=True, arguments=args)


class ManagedRabbitMQBlockingConnection:
    """Small lifecycle wrapper around a pika BlockingConnection."""

    def __init__(
        self,
        config: Optional[RabbitMQConnectionConfig] = None,
        *,
        params_factory=None,
        logger=None,
        label: str = "RabbitMQ",
        **param_overrides,
    ):
        if config is None and params_factory is None:
            raise ValueError("ManagedRabbitMQBlockingConnection requires config or params_factory")
        self.config = config
        self.params_factory = params_factory
        self.param_overrides = param_overrides
        self.logger = logger
        self.label = label
        self.connection = None
        self.channel = None

    def _build_params(self):
        if self.params_factory is not None:
            return self.params_factory()
        return self.config.build_params(**self.param_overrides)

    def close(self):
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                pass
        self.connection = None
        self.channel = None

    def connect(self):
        self.close()
        self.connection = pika.BlockingConnection(self._build_params())
        self.channel = self.connection.channel()
        return self.connection, self.channel

    def reconnect(self):
        return self.connect()

    def is_open(self) -> bool:
        return (
            self.connection is not None
            and not self.connection.is_closed
            and self.channel is not None
            and not self.channel.is_closed
        )

    def ensure_channel(self):
        if not self.is_open():
            self.connect()
        return self.channel


class ManagedRabbitMQAsyncPublisher:
    """Owns one async RabbitMQ publisher connection and its local buffer."""

    def __init__(
        self,
        *,
        params_factory,
        declaration_provider,
        prepare_body,
        publish_timing_logger,
        logger,
        retry_delay: int = 5,
        label: str = "RabbitMQ async publisher",
    ):
        self.params_factory = params_factory
        self.declaration_provider = declaration_provider
        self.prepare_body = prepare_body
        self.publish_timing_logger = publish_timing_logger
        self.logger = logger
        self.retry_delay = retry_delay
        self.label = label

        self.connection = None
        self.channel = None
        self.ready = threading.Event()
        self.pending = deque()
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=self.label.replace(" ", "_"),
        )
        self.thread.start()

    def stop(self, join_timeout: float = 10):
        self.running = False
        if self.connection is not None:
            try:
                self.connection.ioloop.add_callback_threadsafe(self.connection.ioloop.stop)
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=join_timeout)

    def publish(self, routing_key, body):
        body = self.prepare_body(
            body,
            _publisher_enqueued_at=self._now_iso(),
        )
        local_enqueued_at = time.monotonic()
        if (
            self.connection is not None
            and self.channel is not None
            and self.ready.is_set()
        ):
            try:
                self.connection.ioloop.add_callback_threadsafe(
                    partial(self._publish_message, routing_key, body, local_enqueued_at)
                )
                return
            except Exception as e:
                self.logger.warning(f"Async publish handoff failed, buffering locally: {e}")

        with self.lock:
            self.pending.append((routing_key, body, local_enqueued_at))

    def _run(self):
        while self.running:
            try:
                self.ready.clear()
                self.connection = pika.SelectConnection(
                    self.params_factory(),
                    on_open_callback=self._on_connection_open,
                    on_open_error_callback=self._on_connection_open_error,
                    on_close_callback=self._on_connection_closed,
                )
                self.connection.ioloop.start()
            except Exception as e:
                if self.running:
                    self.logger.error(f"Async publish loop failed: {e}")
            finally:
                self.ready.clear()
                self.channel = None
                self.connection = None

            if self.running:
                time.sleep(self.retry_delay)

        self.logger.info("Async publish thread stopped")

    def _on_connection_open(self, connection):
        connection.channel(on_open_callback=self._on_channel_open)

    def _on_connection_open_error(self, connection, error):
        self.logger.warning(f"Async publish connection open failed: {error}")
        connection.ioloop.stop()

    def _on_connection_closed(self, connection, reason):
        self.ready.clear()
        self.channel = None
        if self.running:
            self.logger.warning(f"Async publish connection closed: {reason}")
        connection.ioloop.stop()

    def _on_channel_open(self, channel):
        self.channel = channel
        self._declare_queue_at_index(self.declaration_provider(), 0)

    def _declare_queue_at_index(self, declarations, index):
        if self.channel is None:
            return
        if index >= len(declarations):
            self.ready.set()
            self.logger.info("Async publisher connected to RabbitMQ")
            self._flush_pending()
            return

        queue_name, arguments = declarations[index]
        self.channel.queue_declare(
            queue=queue_name,
            durable=True,
            arguments=arguments,
            callback=lambda _frame: self._declare_queue_at_index(declarations, index + 1),
        )

    def _flush_pending(self):
        while True:
            with self.lock:
                if not self.pending:
                    return
                routing_key, body, local_enqueued_at = self.pending.popleft()
            self._publish_message(routing_key, body, local_enqueued_at)

    def _publish_message(self, routing_key, body, local_enqueued_at):
        if self.channel is None or self.channel.is_closed:
            with self.lock:
                self.pending.appendleft((routing_key, body, local_enqueued_at))
            return

        local_queue_wait = time.monotonic() - local_enqueued_at
        body = self.prepare_body(
            body,
            _publisher_started_at=self._now_iso(),
        )
        publish_started_at = time.time()
        self.channel.basic_publish(
            exchange='',
            routing_key=routing_key,
            body=body,
            properties=pika.BasicProperties(delivery_mode=2),
        )
        publish_duration = time.time() - publish_started_at
        self.publish_timing_logger(
            routing_key,
            body,
            local_queue_wait,
            publish_duration,
        )

    def _now_iso(self):
        return datetime.now().isoformat()
