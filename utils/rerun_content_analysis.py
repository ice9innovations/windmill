#!/usr/bin/env python3
"""
Re-run Content Analysis

Triggers content analysis re-processing for all images with consensus data.
Use this after updating spatial_analysis.py or content_analysis_worker.py.

Usage:
    python utils/rerun_content_analysis.py [--limit N] [--reason "description"]
"""
import os
import sys
import json
import argparse
import pika
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )


def get_queue_connection():
    """Create RabbitMQ connection"""
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


def get_images_to_process(cursor, limit=None):
    """Get all images with consensus data"""
    query = """
        SELECT DISTINCT c.image_id, i.image_filename
        FROM consensus c
        JOIN images i ON c.image_id = i.image_id
        ORDER BY c.image_id
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    return cursor.fetchall()


def publish_messages(channel, images, reason):
    """Publish re-analysis messages to queue"""
    for image_id, image_filename in images:
        message = {
            'image_id': image_id,
            'image_filename': image_filename,
            'triggered_by': reason,
            'triggered_at': datetime.now().isoformat()
        }
        channel.basic_publish(
            exchange='',
            routing_key='content_analysis',
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=2)
        )


def main():
    parser = argparse.ArgumentParser(description='Re-run content analysis')
    parser.add_argument('--limit', type=int, help='Limit number of images to process')
    parser.add_argument('--reason', type=str, default='manual_rerun',
                        help='Reason for re-run (logged in message)')
    args = parser.parse_args()

    print("Connecting to database...")
    db_conn = get_db_connection()
    cursor = db_conn.cursor()

    print("Fetching images with consensus data...")
    images = get_images_to_process(cursor, args.limit)
    cursor.close()
    db_conn.close()

    if not images:
        print("No images found to process")
        return

    print(f"Found {len(images):,} images to re-process")

    print("Connecting to RabbitMQ...")
    queue_conn = get_queue_connection()
    channel = queue_conn.channel()

    print(f"Publishing messages (reason: {args.reason})...")
    publish_messages(channel, images, args.reason)

    queue_conn.close()
    print(f"Published {len(images):,} messages to content_analysis queue")


if __name__ == '__main__':
    main()
