#!/usr/bin/env python3
"""
Requeue messages from a Dead Letter Queue (DLQ) back to the original queue.

Usage:
  QUEUE_HOST=... QUEUE_USER=... QUEUE_PASSWORD=... \
  python utils/requeue_from_dlq.py <queue_name> [--limit N]

Example:
  python utils/requeue_from_dlq.py queue_harmony --limit 500
"""

import os
import sys
import argparse
import pika
from dotenv import load_dotenv


def requeue_from_dlq(queue_name: str, limit: int) -> int:
    load_dotenv()

    queue_host = os.getenv("QUEUE_HOST")
    queue_user = os.getenv("QUEUE_USER")
    queue_password = os.getenv("QUEUE_PASSWORD")

    if not queue_host or not queue_user or not queue_password:
        print("Missing QUEUE_HOST/QUEUE_USER/QUEUE_PASSWORD in environment")
        return 1

    creds = pika.PlainCredentials(queue_user, queue_password)
    conn = pika.BlockingConnection(pika.ConnectionParameters(queue_host, credentials=creds))
    ch = conn.channel()

    dlq = f"{queue_name}.dlq"

    moved = 0
    try:
        for _ in range(limit):
            method, props, body = ch.basic_get(dlq, auto_ack=False)
            if not method:
                break
            ch.basic_publish('', queue_name, body, pika.BasicProperties(delivery_mode=2))
            ch.basic_ack(method.delivery_tag)
            moved += 1
    finally:
        conn.close()

    print(f"Requeued {moved} messages from {dlq} to {queue_name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Requeue messages from <queue>.dlq back to <queue>")
    parser.add_argument('queue', help='Original queue name (without .dlq)')
    parser.add_argument('--limit', type=int, default=100, help='Maximum messages to requeue (default: 100)')
    args = parser.parse_args()

    return requeue_from_dlq(args.queue, args.limit)


if __name__ == '__main__':
    sys.exit(main())


