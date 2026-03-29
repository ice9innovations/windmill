#!/usr/bin/env python3
"""
Re-run NudeNet Service

Deletes existing NudeNet results and triggers re-processing for all images.
After NudeNet completes, triggers consensus recalculation.

Usage:
    python utils/rerun_nudenet.py --group nudenet_test
    python utils/rerun_nudenet.py --group nudenet_test --limit 1000
"""
import os
import sys
import json
import argparse
import base64
import pika
import psycopg2
import requests
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import io

from core.image_store import is_valkey_image_store_enabled, put_image

load_dotenv()


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )


def get_queue_connection():
    credentials = pika.PlainCredentials(
        os.getenv('QUEUE_USER'),
        os.getenv('QUEUE_PASSWORD')
    )
    return pika.BlockingConnection(
        pika.ConnectionParameters(
            host=os.getenv('QUEUE_HOST'),
            credentials=credentials
        )
    )


def get_images(cursor, limit=None, group=None):
    """Get images to process, optionally filtered by group"""
    if group:
        query = """
            SELECT image_id, image_filename, image_url
            FROM images
            WHERE image_group = %s
            ORDER BY image_id
        """
        if limit:
            query += f" LIMIT {limit}"
        cursor.execute(query, (group,))
    else:
        query = """
            SELECT image_id, image_filename, image_url
            FROM images
            ORDER BY image_id
        """
        if limit:
            query += f" LIMIT {limit}"
        cursor.execute(query)

    return cursor.fetchall()


def delete_nudenet_results(cursor, image_ids=None):
    """Delete existing NudeNet results"""
    if image_ids:
        cursor.execute("""
            DELETE FROM results
            WHERE service = 'nudenet' AND image_id = ANY(%s)
        """, (image_ids,))
    else:
        cursor.execute("DELETE FROM results WHERE service = 'nudenet'")

    return cursor.rowcount


def fetch_image_data(image_url, image_filename):
    """Fetch image from URL and return transport payload with dimensions."""
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image_bytes = response.content

        # Extract dimensions
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size

        return {
            **(
                {'image_ref': put_image(image_bytes)}
                if is_valkey_image_store_enabled()
                else {'image_data': base64.b64encode(image_bytes).decode('utf-8')}
            ),
            'image_width': width,
            'image_height': height
        }
    except Exception as e:
        print(f"  Failed to fetch {image_filename}: {e}")
        return None


def publish_nudenet_messages(channel, images):
    """Publish messages to nudenet queue, fetching image data for each"""
    success_count = 0
    fail_count = 0

    for i, (image_id, image_filename, image_url) in enumerate(images):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i + 1}/{len(images)}... ({success_count} success, {fail_count} failed)")

        # Fetch and encode image
        image_info = fetch_image_data(image_url, image_filename)
        if not image_info:
            fail_count += 1
            continue

        message = {
            'image_id': image_id,
            'image_filename': image_filename,
            'image_width': image_info['image_width'],
            'image_height': image_info['image_height'],
            'submitted_at': datetime.now().isoformat(),
            **(
                {'image_ref': image_info['image_ref']}
                if 'image_ref' in image_info
                else {'image_data': image_info['image_data']}
            ),
        }
        channel.basic_publish(
            exchange='',
            routing_key='nudenet',
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        success_count += 1

    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(description='Re-run NudeNet processing')
    parser.add_argument('--limit', type=int, help='Limit number of images')
    parser.add_argument('--group', type=str, help='Only process images from this group (e.g., nudenet_test)')
    parser.add_argument('--skip-delete', action='store_true',
                        help='Skip deletion (for testing, will create duplicates)')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt')
    args = parser.parse_args()

    # Connect to database
    print("Connecting to database...")
    db_conn = get_db_connection()
    cursor = db_conn.cursor()

    # Get images
    if args.group:
        print(f"Fetching images from group '{args.group}'...")
    else:
        print("Fetching all images...")
    images = get_images(cursor, args.limit, args.group)

    if not images:
        print("No images found")
        return

    print(f"Found {len(images):,} images to process")

    # Confirm before deletion
    if not args.skip_delete and not args.yes:
        # Check current count
        cursor.execute("SELECT COUNT(*) FROM results WHERE service = 'nudenet'")
        current_count = cursor.fetchone()[0]

        print(f"\nThis will DELETE {current_count:,} existing NudeNet results")
        print(f"and queue {len(images):,} images for re-processing.")
        response = input("\nProceed? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted")
            return

    # Delete existing results
    if not args.skip_delete:
        print("Deleting existing NudeNet results...")
        if args.limit:
            image_ids = [img[0] for img in images]
            deleted = delete_nudenet_results(cursor, image_ids)
        else:
            deleted = delete_nudenet_results(cursor)
        db_conn.commit()
        print(f"Deleted {deleted:,} results")

    cursor.close()
    db_conn.close()

    # Publish messages
    print("Connecting to RabbitMQ...")
    queue_conn = get_queue_connection()
    channel = queue_conn.channel()

    print("Fetching images and publishing to nudenet queue...")
    print("(This may take a while - each image must be fetched from URL)")
    success, failed = publish_nudenet_messages(channel, images)

    queue_conn.close()
    print(f"\nPublished {success:,} messages ({failed:,} failed to fetch)")
    print("NudeNet processing will trigger harmony and consensus automatically.")


if __name__ == '__main__':
    main()
