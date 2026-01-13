#!/usr/bin/env python3
"""
Retrigger harmony and consensus for images that have results but missing harmonization.
This is useful when workers weren't running during initial processing.
"""
import os
import sys
import json
import pika
import psycopg2
import base64
import requests
from datetime import datetime
from dotenv import load_dotenv

def main():
    load_dotenv()

    # Get image group from command line
    if len(sys.argv) < 2:
        print("Usage: python retrigger_harmony.py <image_group> [--limit N]")
        print("Example: python retrigger_harmony.py nudenet_test --limit 100")
        sys.exit(1)

    image_group = sys.argv[1]
    limit = None

    if '--limit' in sys.argv:
        limit_idx = sys.argv.index('--limit')
        if limit_idx + 1 < len(sys.argv):
            limit = int(sys.argv[limit_idx + 1])

    # Connect to database
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    cursor = conn.cursor()

    # Get images from the group
    query = """
        SELECT DISTINCT i.image_id, i.image_filename, i.image_url
        FROM images i
        JOIN results r ON i.image_id = r.image_id
        WHERE i.image_group = %s
        AND r.service = 'nudenet'
        AND r.status = 'success'
        ORDER BY i.image_id
    """

    params = [image_group]
    if limit:
        query += " LIMIT %s"
        params.append(limit)

    cursor.execute(query, params)
    images = cursor.fetchall()

    print(f"Found {len(images)} images to reprocess for harmony/consensus")

    if len(images) == 0:
        print("No images found. Exiting.")
        cursor.close()
        conn.close()
        return

    # Connect to RabbitMQ
    credentials = pika.PlainCredentials(
        os.getenv('QUEUE_USER'),
        os.getenv('QUEUE_PASSWORD')
    )
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=os.getenv('QUEUE_HOST'),
            credentials=credentials
        )
    )
    channel = connection.channel()

    # Queues should already exist - just use them passively
    # (They were created by workers with DLQ configuration)
    try:
        channel.queue_declare(queue='harmony', durable=True, passive=True)
        channel.queue_declare(queue='consensus', durable=True, passive=True)
    except Exception as e:
        print(f"Warning: Could not verify queues exist: {e}")
        print("Continuing anyway - queues should exist from workers")

    # Process each image
    harmony_published = 0
    consensus_published = 0

    for image_id, image_filename, image_url in images:
        try:
            # Fetch image data
            response = requests.get(image_url, timeout=10)
            if response.status_code != 200:
                print(f"  ⚠️  Failed to fetch {image_filename}: HTTP {response.status_code}")
                continue

            image_data_b64 = base64.b64encode(response.content).decode('utf-8')

            # Publish to harmony queue
            harmony_message = {
                'image_id': image_id,
                'image_filename': image_filename,
                'image_data': image_data_b64,
                'service': 'nudenet',
                'worker_id': 'retrigger_script',
                'processed_at': datetime.now().isoformat()
            }

            channel.basic_publish(
                exchange='',
                routing_key='harmony',
                body=json.dumps(harmony_message),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            harmony_published += 1

            # Publish to consensus queue
            consensus_message = {
                'image_id': image_id,
                'image_filename': image_filename,
                'image_data': image_data_b64,
                'service': 'nudenet',
                'worker_id': 'retrigger_script',
                'processed_at': datetime.now().isoformat()
            }

            channel.basic_publish(
                exchange='',
                routing_key='consensus',
                body=json.dumps(consensus_message),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            consensus_published += 1

            if (harmony_published % 100) == 0:
                print(f"  Published {harmony_published} messages...")

        except Exception as e:
            print(f"  ❌ Error processing {image_filename}: {e}")
            continue

    print(f"\n✅ Published {harmony_published} harmony messages")
    print(f"✅ Published {consensus_published} consensus messages")

    # Cleanup
    connection.close()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
