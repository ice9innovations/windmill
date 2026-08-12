#!/usr/bin/env python3
"""
Measure RabbitMQ publish latency against one or more queues.

This is intended to isolate broker/network publish cost from worker logic.

WARNING:
  This publishes synthetic messages directly to the named queues. Do not target
  live worker queues unless you intend to inject non-worker payloads there.

Examples:
  python utils/rabbitmq_publish_latency_probe.py --queues harmony --count 25
  python utils/rabbitmq_publish_latency_probe.py --queues pose --count 50 --confirm
  python utils/rabbitmq_publish_latency_probe.py --queues face pose --count 25 --confirm --size 262144
"""
import argparse
import json
import os
import ssl
import statistics
import sys
import time
from datetime import datetime

import pika
from dotenv import load_dotenv


DEFAULT_QUEUES = ["harmony"]


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Required environment variable {key} not set")
    return value


def build_queue_params() -> pika.ConnectionParameters:
    credentials = pika.PlainCredentials(
        _require_env("QUEUE_USER"),
        _require_env("QUEUE_PASSWORD"),
    )
    queue_host = _require_env("QUEUE_HOST")
    queue_port = int(os.getenv("QUEUE_PORT", "5672"))
    queue_ssl = os.getenv("QUEUE_SSL", "").lower() in ("true", "1", "yes")

    kwargs = dict(
        host=queue_host,
        port=queue_port,
        credentials=credentials,
        heartbeat=60,
        blocked_connection_timeout=300,
        connection_attempts=10,
        retry_delay=5,
        socket_timeout=10,
    )
    if queue_ssl:
        ssl_context = ssl.create_default_context()
        kwargs["ssl_options"] = pika.SSLOptions(ssl_context, queue_host)
    return pika.ConnectionParameters(**kwargs)


def percentile(sorted_values, fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def format_seconds(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms"


def build_payload(queue_name: str, size: int, sequence: int) -> bytes:
    base = {
        "probe": "rabbitmq_publish_latency",
        "queue": queue_name,
        "sequence": sequence,
        "sent_at": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
    }
    body = json.dumps(base, separators=(",", ":")).encode("utf-8")
    if len(body) >= size:
        return body

    padding = b"x" * (size - len(body))
    return body + padding


def verify_queue_exists(channel, queue_name: str) -> None:
    channel.queue_declare(queue=queue_name, durable=True, passive=True)


def measure_queue(queue_name: str, count: int, payload_size: int, confirm: bool):
    connection = pika.BlockingConnection(build_queue_params())
    try:
        channel = connection.channel()
        verify_queue_exists(channel, queue_name)
        if confirm:
            channel.confirm_delivery()

        latencies = []
        for sequence in range(count):
            body = build_payload(queue_name, payload_size, sequence)
            started_at = time.perf_counter()
            published = channel.basic_publish(
                exchange="",
                routing_key=queue_name,
                body=body,
                properties=pika.BasicProperties(delivery_mode=2),
                mandatory=True,
            )
            elapsed = time.perf_counter() - started_at
            if confirm and published is False:
                raise RuntimeError(f"Broker did not confirm publish to {queue_name}")
            latencies.append(elapsed)

        latencies_sorted = sorted(latencies)
        return {
            "queue": queue_name,
            "count": count,
            "confirm": confirm,
            "payload_size": payload_size,
            "min": min(latencies_sorted),
            "avg": statistics.mean(latencies_sorted),
            "p50": percentile(latencies_sorted, 0.50),
            "p95": percentile(latencies_sorted, 0.95),
            "max": max(latencies_sorted),
        }
    finally:
        connection.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Measure RabbitMQ publish latency by queue")
    parser.add_argument(
        "--queues",
        nargs="+",
        default=DEFAULT_QUEUES,
        help=f"Queues to test (default: {' '.join(DEFAULT_QUEUES)})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=25,
        help="Number of publishes per queue (default: 25)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Payload size in bytes (default: 1024)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Enable broker publish confirms on the channel",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    if args.count <= 0:
        print("--count must be > 0")
        return 1
    if args.size <= 0:
        print("--size must be > 0")
        return 1

    try:
        results = []
        for queue_name in args.queues:
            result = measure_queue(
                queue_name=queue_name,
                count=args.count,
                payload_size=args.size,
                confirm=args.confirm,
            )
            results.append(result)
    except Exception as exc:
        print(f"Probe failed: {exc}")
        return 1

    print(
        f"RabbitMQ publish latency probe "
        f"(count={args.count}, size={args.size}, confirm={args.confirm})"
    )
    for result in results:
        print(
            f"{result['queue']:10} "
            f"min={format_seconds(result['min'])} "
            f"avg={format_seconds(result['avg'])} "
            f"p50={format_seconds(result['p50'])} "
            f"p95={format_seconds(result['p95'])} "
            f"max={format_seconds(result['max'])}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
