#!/usr/bin/env python3
"""
Refinement Worker - Refines bounding boxes using SAM2 segmentation
Processes refinement requests queued after consensus completes
"""
import os
import json
import time
import logging
import base64
import io
import psycopg2
import pika
import requests
from datetime import datetime
from base_worker import BaseWorker
from PIL import Image, ImageDraw

class RefinementWorker(BaseWorker):
    """Refines bounding boxes using SAM2 segmentation service"""

    def __init__(self):
        # Initialize with segmentation service type
        super().__init__('system.segmentation')

        # Refinement worker needs separate read connection for queries
        self.read_db_conn = None

    def connect_to_database(self):
        """Connect to PostgreSQL database with dual connections"""
        # Call parent to set up main connection
        if not super().connect_to_database():
            return False

        try:
            # Set up transaction mode for main connection
            self.db_conn.autocommit = False

            # Create separate read connection for queries
            self.read_db_conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            self.read_db_conn.autocommit = True  # Auto-commit for read queries

            return True

        except Exception as e:
            self.logger.error(f"Failed to set up dual database connections: {e}")
            return False

    def ensure_database_connection(self):
        """Ensure database connections are healthy, reconnect if needed"""
        reconnect_needed = False

        # Check main connection
        try:
            if not self.db_conn or self.db_conn.closed:
                reconnect_needed = True
            else:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
        except Exception as e:
            self.logger.warning(f"Main database connection unhealthy: {e}")
            reconnect_needed = True

        # Check read connection
        try:
            if not self.read_db_conn or self.read_db_conn.closed:
                reconnect_needed = True
            else:
                cursor = self.read_db_conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
        except Exception as e:
            self.logger.warning(f"Read database connection unhealthy: {e}")
            reconnect_needed = True

        if reconnect_needed:
            self.logger.info("Reconnecting to database...")
            return self.connect_to_database()

        return True


    def call_segmentation_service(self, cropped_base64, emoji_prompt):
        """Call SAM2 segmentation service with emoji prompt"""
        try:
            # Convert base64 to bytes for multipart upload
            import base64
            image_bytes = base64.b64decode(cropped_base64)

            # Prepare multipart form data (SAM2 doesn't use prompt/format params)
            files = {
                'file': ('crop.jpg', image_bytes, 'image/jpeg')
            }

            # Make request to segmentation service
            url = f"http://{self.service_host}:{self.service_port}{self.service_endpoint}"

            response = requests.post(
                url,
                files=files,
                timeout=self.request_timeout
            )
            response.raise_for_status()

            result = response.json()
            self.logger.debug(f"Segmentation service response: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error calling segmentation service: {e}")
            return None

    def store_refinement_result(self, image_id, consensus_box_id, original_bbox, refined_bbox, confidence, emoji):
        """Store refinement result in postprocessing table"""
        try:
            if not self.ensure_database_connection():
                raise Exception("Failed to ensure database connection")

            cursor = self.db_conn.cursor()

            refinement_data = {
                'original_bbox': original_bbox,
                'refined_bbox': refined_bbox,
                'confidence': confidence,
                'emoji': emoji,
                'worker_id': self.worker_id,
                'processed_at': datetime.now().isoformat()
            }

            cursor.execute("""
                INSERT INTO postprocessing
                (image_id, merged_box_id, service, data, status, result_created, processing_time)
                VALUES (%s, %s, %s, %s, %s, NOW(), %s)
            """, (
                image_id,
                None,  # NULL for consensus-box-level analysis (not merged_box)
                'refinement',
                json.dumps(refinement_data),
                'success',
                0  # TODO: Track actual processing time
            ))

            self.db_conn.commit()
            cursor.close()

            self.logger.info(f"Stored refinement result for image {image_id}, consensus box {consensus_box_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error storing refinement result: {e}")
            if self.db_conn:
                self.db_conn.rollback()
            return False

    def rescale_bbox_to_full_image(self, refined_bbox_in_crop, crop_info):
        """Re-scale refined bbox coordinates from cropped image back to full image coordinates"""
        try:
            # Get crop information
            crop_bbox = crop_info['crop_bbox']

            # Re-scale coordinates from cropped space to full image space
            full_image_bbox = {
                'x': crop_bbox['x'] + refined_bbox_in_crop['x'],
                'y': crop_bbox['y'] + refined_bbox_in_crop['y'],
                'width': refined_bbox_in_crop['width'],
                'height': refined_bbox_in_crop['height']
            }

            return full_image_bbox

        except Exception as e:
            self.logger.error(f"Error re-scaling bbox coordinates: {e}")
            return None

    def process_refinement_request(self, message):
        """Process a single refinement request from producer"""
        try:
            image_id = message['image_id']
            image_filename = message.get('image_filename', f'image_{image_id}')
            box_index = message['box_index']
            original_bbox = message['original_bbox']
            emoji = message['emoji']
            confidence = message.get('confidence', 0.5)
            cropped_base64 = message['cropped_image_data']
            crop_info = message['crop_info']

            self.logger.info(f"Processing refinement for {image_filename} box {box_index} (emoji: {emoji})")

            # Call segmentation service with emoji prompt on cropped image
            segmentation_result = self.call_segmentation_service(cropped_base64, emoji)
            if not segmentation_result:
                self.logger.error(f"Segmentation failed for box {box_index} with emoji {emoji}")
                return False

            # Extract refined bounding box from segmentation result
            if 'predictions' in segmentation_result and segmentation_result['predictions'] and 'bbox' in segmentation_result['predictions'][0]:
                # Get refined bbox relative to cropped image
                refined_bbox_in_crop = segmentation_result['predictions'][0]['bbox']

                # Re-scale coordinates back to full image space
                refined_bbox_full_image = self.rescale_bbox_to_full_image(refined_bbox_in_crop, crop_info)
                if not refined_bbox_full_image:
                    self.logger.error(f"Failed to re-scale bbox coordinates for box {box_index}")
                    return False

                # Use confidence from segmentation result if available
                refined_confidence = segmentation_result['predictions'][0].get('confidence', confidence)

                # Store the refinement result
                if self.store_refinement_result(image_id, box_index, original_bbox, refined_bbox_full_image, refined_confidence, emoji):
                    self.logger.info(f"Successfully refined box {box_index} for {image_filename}")
                    return True
                else:
                    self.logger.error(f"Failed to store refinement result for box {box_index}")
                    return False
            else:
                self.logger.error(f"No bbox in segmentation result for box {box_index}")
                return False

        except Exception as e:
            self.logger.error(f"Error processing refinement request: {e}")
            return False

    def process_queue_message(self, ch, method, properties, body):
        """Process a message from the refinement queue"""
        try:
            message = json.loads(body)

            if self.process_refinement_request(message):
                # Acknowledge successful processing
                ch.basic_ack(delivery_tag=method.delivery_tag)
                self.jobs_completed += 1
            else:
                # Reject and requeue for retry
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                self.jobs_failed += 1

        except Exception as e:
            self.logger.error(f"Error processing refinement queue message: {e}")
            # Reject and requeue for retry
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            self.jobs_failed += 1

    def run(self):
        """Main entry point - pure queue-based processing"""

        if not self.connect_to_database():
            return

        if not self.connect_to_queue():
            return

        self.logger.info(f"Listening on queue: {self.queue_name}")

        try:
            # Set up message consumption
            self.channel.basic_qos(prefetch_count=self.worker_prefetch_count)
            self.channel.basic_consume(
                queue=self.queue_name,
                on_message_callback=self.process_queue_message
            )

            # Start consuming
            self.channel.start_consuming()

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, shutting down...")
            self.channel.stop_consuming()
        except Exception as e:
            self.logger.error(f"Error in main processing loop: {e}")
        finally:
            self.close_connections()

if __name__ == "__main__":
    worker = RefinementWorker()
    worker.run()