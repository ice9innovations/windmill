#!/usr/bin/env python3
"""
Predeclare Windmill RabbitMQ queues from service_config.yaml.

This is intended for install/bootstrap time so fresh deployments do not depend
on hot-path worker code to create downstream queues.
"""

import os
import sys

from dotenv import load_dotenv
import pika

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "workers"))

from core.rabbitmq_connection import RabbitMQConnectionConfig, declare_queue
from workers.service_config import ServiceConfig


def build_queue_config() -> RabbitMQConnectionConfig:
    host = os.getenv("QUEUE_HOST")
    user = os.getenv("QUEUE_USER")
    password = os.getenv("QUEUE_PASSWORD")
    if not host or not user or not password:
        raise ValueError("Missing QUEUE_HOST/QUEUE_USER/QUEUE_PASSWORD")

    port = int(os.getenv("QUEUE_PORT", "5672"))
    use_ssl = os.getenv("QUEUE_USE_SSL", "false").lower() in {"1", "true", "yes", "on"}
    server_hostname = os.getenv("QUEUE_SSL_SERVER_HOSTNAME") or None
    return RabbitMQConnectionConfig(
        host=host,
        port=port,
        user=user,
        password=password,
        use_ssl=use_ssl,
        server_hostname=server_hostname,
    )


def queue_message_ttl_ms():
    ttl_env = os.getenv("QUEUE_MESSAGE_TTL_MS")
    if ttl_env and ttl_env.isdigit():
        ttl_ms = int(ttl_env)
        if ttl_ms > 0:
            return ttl_ms
    return None


def gather_queue_names(config: ServiceConfig):
    queue_names = set()
    for category, services in config.raw_config.get("services", {}).items():
        for service_name, service_config in services.items():
            if not service_config:
                continue
            queue_names.add(service_config.get("queue_name", service_name))
    return sorted(queue_names)


def main() -> int:
    load_dotenv()
    config = ServiceConfig(os.path.join(ROOT, "service_config.yaml"))
    queue_config = build_queue_config()
    params = queue_config.build_params()

    ttl_ms = queue_message_ttl_ms()
    queue_names = gather_queue_names(config)
    declared = 0

    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    try:
        for queue_name in queue_names:
            declare_queue(channel, queue_name, ttl_ms=ttl_ms)
            declared += 1
        print(f"Declared {declared} queues (and matching DLQs).")
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
