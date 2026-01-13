#!/usr/bin/env python3
"""
Trigger content analysis for existing images (re-analysis)
Useful for testing or re-processing after algorithm updates
"""
import os
import sys
import json
import pika
import argparse
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

# Load environment
load_dotenv('../.env')

def get_images_to_process(image_group=None, limit=10):
    """Get images that have consensus but no content_analysis yet"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        cursor = conn.cursor()

        query = """
            SELECT DISTINCT c.image_id, i.image_filename
            FROM consensus c
            JOIN images i ON c.image_id = i.image_id
            LEFT JOIN content_analysis ca ON c.image_id = ca.image_id
            WHERE ca.image_id IS NULL
        """

        params = []
        if image_group:
            query += " AND i.image_group = %s"
            params.append(image_group)

        query += " ORDER BY c.image_id LIMIT %s"
        params.append(limit)

        cursor.execute(query, params)
        images = cursor.fetchall()

        cursor.close()
        conn.close()

        return images

    except Exception as e:
        print(f"Error fetching images: {e}")
        return []


def trigger_content_analysis(images):
    """Publish content_analysis messages to RabbitMQ"""
    try:
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

        # Note: Queue will be created by worker with proper DLQ configuration
        # We just publish messages here

        published = 0
        for image_id, image_filename in images:
            message = {
                'image_id': image_id,
                'image_filename': image_filename,
                'triggered_by': 'manual_retrigger',
                'triggered_at': datetime.now().isoformat()
            }

            channel.basic_publish(
                exchange='',
                routing_key='content_analysis',
                body=json.dumps(message),
                properties=pika.BasicProperties(delivery_mode=2)
            )

            published += 1
            print(f"Triggered content analysis for {image_filename} (ID: {image_id})")

        connection.close()
        print(f"\nPublished {published} messages to content_analysis")

    except Exception as e:
        print(f"Error publishing messages: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Trigger content analysis for images with consensus'
    )
    parser.add_argument(
        '--group', '-g',
        help='Image group to process (e.g., nudenet_test)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=10,
        help='Number of images to process (default: 10)'
    )

    args = parser.parse_args()

    print(f"Finding images to process...")
    if args.group:
        print(f"Filtering by group: {args.group}")
    print(f"Limit: {args.limit}\n")

    images = get_images_to_process(args.group, args.limit)

    if not images:
        print("No images found needing content analysis")
        return 0

    print(f"Found {len(images)} images\n")

    trigger_content_analysis(images)

    print("\nContent analysis messages published!")
    print("Start the content_analysis_worker to process them:")
    print("  cd /home/sd/windmill/workers")
    print("  python content_analysis_worker.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
