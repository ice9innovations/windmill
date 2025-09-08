#!/usr/bin/env python3
"""
List all Dead Letter Queues (<queue>.dlq) with message counts.

Usage:
  QUEUE_HOST=... QUEUE_USER=... QUEUE_PASSWORD=... python utils/list_dlqs.py
"""

import os
import sys
import json
import pika
from dotenv import load_dotenv


def list_queues(channel):
    # Use passive declare over a known set if needed. Here we query via the management API if available
    # but to keep dependencies minimal, we iterate a guess list from environment or fallback.
    # Best-effort: read a comma-separated list of known queues from QUEUE_CATALOG.
    catalog = os.getenv('QUEUE_CATALOG')
    if catalog:
        names = [q.strip() for q in catalog.split(',') if q.strip()]
    else:
        # Minimal set; users can set QUEUE_CATALOG for completeness.
        names = [
            'queue_harmony', 'consensus',
            'blip', 'clip', 'colors', 'detectron', 'metadata', 'ocr', 'nsfw', 'ollama', 'rtdetr', 'yolo_v8', 'yolo_365', 'yolo_oi7',
            'colors_post', 'face', 'pose'
        ]
    result = []
    seen = set()
    for q in names:
        for name in (q, f"{q}.dlq"):
            if name in seen:
                continue
            seen.add(name)
            try:
                m = channel.queue_declare(name, passive=True)
                result.append({
                    'name': name,
                    'messages': m.method.message_count,
                    'consumers': m.method.consumer_count
                })
            except Exception:
                # queue may not exist
                pass
    return sorted(result, key=lambda x: (-x['messages'], x['name']))


def main() -> int:
    load_dotenv()
    host = os.getenv('QUEUE_HOST')
    user = os.getenv('QUEUE_USER')
    pwd = os.getenv('QUEUE_PASSWORD')
    if not host or not user or not pwd:
        print('Missing QUEUE_HOST/QUEUE_USER/QUEUE_PASSWORD')
        return 1

    creds = pika.PlainCredentials(user, pwd)
    conn = pika.BlockingConnection(pika.ConnectionParameters(host, credentials=creds))
    ch = conn.channel()
    try:
        rows = list_queues(ch)
        if not rows:
            print('No queues found (set QUEUE_CATALOG for better coverage).')
            return 0
        print('\nDLQ overview (top by messages):')
        for r in rows:
            suffix = '  <-- DLQ' if r['name'].endswith('.dlq') else ''
            print(f"{r['name']:30} messages={r['messages']:6} consumers={r['consumers']:3}{suffix}")
        return 0
    finally:
        conn.close()


if __name__ == '__main__':
    sys.exit(main())


