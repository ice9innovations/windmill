#!/usr/bin/env python3
"""
Refinement Producer - Creates refinement jobs from consensus results
Runs as a batch job to queue refinement work for images with completed consensus
"""
import os
import sys
import json
import time
import logging
import argparse
import base64
import io
import psycopg2
import pika
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from PIL import Image

class RefinementProducer:
    """Produces refinement jobs from consensus results"""

    def __init__(self, env_file='.env'):
        self.load_config(env_file)
        self.setup_logging()
        self.db_conn = None
        self.queue_connection = None
        self.queue_channel = None

        # Expansion parameters
        self.expansion_pixels = int(os.getenv('REFINEMENT_EXPANSION_PIXELS', '10'))

    def load_config(self, env_file):
        """Load configuration from .env file"""
        if not load_dotenv(env_file):
            raise ValueError(f"Could not load {env_file} file. Copy .env.example to .env and configure.")

        # Database configuration
        self.db_host = self._get_required('DB_HOST')
        self.db_name = self._get_required('DB_NAME')
        self.db_user = self._get_required('DB_USER')
        self.db_password = self._get_required('DB_PASSWORD')

        # Queue configuration
        self.queue_host = self._get_required('QUEUE_HOST')
        self.queue_user = self._get_required('QUEUE_USER')
        self.queue_password = self._get_required('QUEUE_PASSWORD')
        self.queue_name = 'refinement'

        # Request configuration
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '30'))

    def _get_required(self, key):
        """Get required environment variable or raise error"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value

    def setup_logging(self):
        """Set up logging configuration"""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def connect_to_database(self):
        """Connect to PostgreSQL database"""
        try:
            self.db_conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            self.db_conn.autocommit = True
            self.logger.info("Connected to PostgreSQL")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False

    def connect_to_queue(self):
        """Connect to RabbitMQ"""
        try:
            credentials = pika.PlainCredentials(self.queue_user, self.queue_password)
            parameters = pika.ConnectionParameters(host=self.queue_host, credentials=credentials)
            self.queue_connection = pika.BlockingConnection(parameters)
            self.queue_channel = self.queue_connection.channel()

            # Declare the refinement queue with DLQ pattern (matching worker declaration exactly)
            def declare_with_dlq(channel, queue_name):
                """Declare queue with dead letter queue configuration"""
                dlq_name = f"{queue_name}.dlq"
                channel.queue_declare(queue=dlq_name, durable=True)
                args = {
                    'x-dead-letter-exchange': '',
                    'x-dead-letter-routing-key': dlq_name,
                    'x-max-length': int(os.getenv('QUEUE_MAX_LENGTH', '100000'))
                }
                # Only add TTL if environment variable is set and positive
                ttl_env = os.getenv('QUEUE_MESSAGE_TTL_MS')
                if ttl_env and ttl_env.isdigit() and int(ttl_env) > 0:
                    args['x-message-ttl'] = int(ttl_env)
                channel.queue_declare(queue=queue_name, durable=True, arguments=args)

            declare_with_dlq(self.queue_channel, self.queue_name)

            self.logger.info("Connected to RabbitMQ")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False

    def get_consensus_results_needing_refinement(self, limit=None, hours_back=24):
        """Get consensus results that need refinement processing"""
        try:
            cursor = self.db_conn.cursor()

            # Find images with consensus but no refinement in postprocessing
            cutoff_time = datetime.now() - timedelta(hours=hours_back)

            query = """
                SELECT DISTINCT c.image_id, c.consensus_data, i.image_url, i.image_filename, c.consensus_created
                FROM consensus c
                JOIN images i ON c.image_id = i.image_id
                LEFT JOIN postprocessing p ON c.image_id = p.image_id AND p.service = 'refinement'
                WHERE c.consensus_created >= %s
                AND p.post_id IS NULL
                AND c.consensus_data IS NOT NULL
                AND c.consensus_data != 'null'
                ORDER BY c.consensus_created DESC
            """

            params = [cutoff_time]
            if limit:
                query += " LIMIT %s"
                params.append(limit)

            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()

            self.logger.info(f"Found {len(results)} images needing refinement")
            return results

        except Exception as e:
            self.logger.error(f"Error querying consensus results: {e}")
            return []

    def download_image(self, image_url):
        """Download image from URL and return PIL Image object"""
        try:
            response = requests.get(image_url, timeout=self.request_timeout)
            response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))
            return image

        except Exception as e:
            self.logger.error(f"Error downloading image from {image_url}: {e}")
            return None

    def expand_bbox(self, bbox, image_width, image_height):
        """Expand bounding box by expansion_pixels, clamped to image dimensions"""
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

        # Expand by pixels in all directions
        new_x = max(0, x - self.expansion_pixels)
        new_y = max(0, y - self.expansion_pixels)
        new_w = min(image_width - new_x, w + (2 * self.expansion_pixels))
        new_h = min(image_height - new_y, h + (2 * self.expansion_pixels))

        return {
            'x': new_x,
            'y': new_y,
            'width': new_w,
            'height': new_h
        }

    def crop_and_encode_image(self, image, bbox):
        """Crop image using bounding box and return base64 encoded result"""
        try:
            # Crop using bounding box coordinates
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            cropped = image.crop((x, y, x + w, y + h))

            # Convert to base64
            buffer = io.BytesIO()
            cropped.save(buffer, format='JPEG')
            cropped_base64 = base64.b64encode(buffer.getvalue()).decode()

            return cropped_base64

        except Exception as e:
            self.logger.error(f"Error cropping and encoding image: {e}")
            return None

    def publish_refinement_job(self, image_id, image_filename, box_index, bbox_data, cropped_base64, crop_info):
        """Publish a single refinement job to the queue"""
        try:
            refinement_job = {
                'image_id': image_id,
                'image_filename': image_filename,
                'box_index': box_index,
                'original_bbox': {
                    'x': bbox_data['x'],
                    'y': bbox_data['y'],
                    'width': bbox_data['width'],
                    'height': bbox_data['height']
                },
                'emoji': bbox_data['emoji'],
                'confidence': bbox_data.get('confidence', 0.5),
                'cropped_image_data': cropped_base64,
                'crop_info': crop_info,  # For coordinate re-scaling
                'expansion_pixels': self.expansion_pixels,
                'produced_at': datetime.now().isoformat()
            }

            self.queue_channel.basic_publish(
                exchange='',
                routing_key=self.queue_name,
                body=json.dumps(refinement_job),
                properties=pika.BasicProperties(delivery_mode=2)  # Persistent message
            )

            return True

        except Exception as e:
            self.logger.error(f"Error publishing refinement job: {e}")
            return False

    def process_consensus_result(self, image_id, consensus_data, image_url, image_filename):
        """Process a single consensus result and create refinement jobs"""
        try:
            self.logger.info(f"Processing consensus for {image_filename} (ID: {image_id})")

            # Download the image
            image = self.download_image(image_url)
            if not image:
                self.logger.error(f"Failed to download image: {image_url}")
                return 0

            image_width, image_height = image.size


            # Extract consensus boxes from votes.consensus structure
            consensus_results = []
            if isinstance(consensus_data, dict) and 'votes' in consensus_data and 'consensus' in consensus_data['votes']:
                consensus_results = consensus_data['votes']['consensus']

            if not consensus_results:
                self.logger.warning(f"No consensus results found for image {image_id}. Available keys: {list(consensus_data.keys()) if isinstance(consensus_data, dict) else 'not a dict'}")
                return 0


            jobs_created = 0

            # Process each consensus box
            for box_index, box in enumerate(consensus_results):
                if not isinstance(box, dict) or 'emoji' not in box or 'bounding_boxes' not in box:
                    self.logger.warning(f"Invalid box format in consensus data: {box}")
                    continue

                # Extract bbox from bounding_boxes[0]['merged_bbox']
                if not box['bounding_boxes'] or 'merged_bbox' not in box['bounding_boxes'][0]:
                    self.logger.warning(f"No merged_bbox found in consensus result: {box}")
                    continue

                merged_bbox = box['bounding_boxes'][0]['merged_bbox']
                original_bbox = {
                    'x': merged_bbox.get('x', 0),
                    'y': merged_bbox.get('y', 0),
                    'width': merged_bbox.get('width', 0),
                    'height': merged_bbox.get('height', 0)
                }

                # Skip boxes that are too small or invalid
                if original_bbox['width'] <= 0 or original_bbox['height'] <= 0:
                    self.logger.warning(f"Skipping invalid bbox: {original_bbox}")
                    continue

                # Expand the bounding box
                expanded_bbox = self.expand_bbox(original_bbox, image_width, image_height)

                # Crop and encode the image
                cropped_base64 = self.crop_and_encode_image(image, expanded_bbox)
                if not cropped_base64:
                    self.logger.error(f"Failed to crop image for box {box_index}")
                    continue

                # Create crop info for coordinate re-scaling
                crop_info = {
                    'original_image_size': [image_width, image_height],
                    'crop_bbox': expanded_bbox,
                    'original_bbox_in_crop': {
                        'x': original_bbox['x'] - expanded_bbox['x'],
                        'y': original_bbox['y'] - expanded_bbox['y'],
                        'width': original_bbox['width'],
                        'height': original_bbox['height']
                    }
                }

                # Create box data for refinement job
                box_data = {
                    'emoji': box['emoji'],
                    'confidence': box.get('democratic_confidence', 0.5),  # Use democratic_confidence or default
                    'x': original_bbox['x'],
                    'y': original_bbox['y'],
                    'width': original_bbox['width'],
                    'height': original_bbox['height']
                }

                # Publish the refinement job
                if self.publish_refinement_job(image_id, image_filename, box_index, box_data, cropped_base64, crop_info):
                    jobs_created += 1
                else:
                    self.logger.error(f"Failed to publish refinement job for box {box_index}")

            self.logger.info(f"Created {jobs_created} refinement jobs for {image_filename}")
            return jobs_created

        except Exception as e:
            self.logger.error(f"Error processing consensus result for image {image_id}: {e}")
            return 0

    def run(self, limit=None, hours_back=24):
        """Main processing loop"""
        self.logger.info("Starting refinement producer")

        if not self.connect_to_database():
            return 1

        if not self.connect_to_queue():
            return 1

        # Get consensus results that need refinement
        consensus_results = self.get_consensus_results_needing_refinement(limit, hours_back)

        if not consensus_results:
            self.logger.info("No consensus results need refinement processing")
            return 0

        total_jobs = 0
        processed_images = 0

        for image_id, consensus_data, image_url, image_filename, consensus_created in consensus_results:
            jobs_created = self.process_consensus_result(image_id, consensus_data, image_url, image_filename)
            total_jobs += jobs_created
            if jobs_created > 0:
                processed_images += 1

        self.logger.info(f"Refinement producer completed: {processed_images} images processed, {total_jobs} jobs created")

        # Close connections
        if self.queue_connection:
            self.queue_connection.close()
        if self.db_conn:
            self.db_conn.close()

        return 0

def main():
    parser = argparse.ArgumentParser(description='Refinement Producer - Create refinement jobs from consensus results')
    parser.add_argument('--limit', type=int, help='Limit number of images to process')
    parser.add_argument('--hours-back', type=int, default=24, help='Hours back to look for consensus results (default: 24)')
    parser.add_argument('--env-file', default='.env', help='Environment file path (default: .env)')

    args = parser.parse_args()

    try:
        producer = RefinementProducer(args.env_file)
        exit_code = producer.run(args.limit, args.hours_back)
        sys.exit(exit_code)

    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()