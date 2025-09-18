#!/usr/bin/env python3
"""
Reset Harmony and Consensus Data

This script clears harmony (merged_boxes) and consensus data from the database
and retriggers processing by sending messages to the harmony queue.

DANGER: This will delete ALL harmony and consensus data!
"""
import os
import json
import logging
import psycopg2
import pika
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging - suppress noisy Pika logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('pika').setLevel(logging.WARNING)  # Suppress Pika connection chatter
logger = logging.getLogger(__name__)

class HarmonyConsensusReset:
    def __init__(self):
        # Database configuration
        self.db_host = os.getenv('DB_HOST')
        self.db_name = os.getenv('DB_NAME')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        
        # Queue configuration
        self.queue_host = os.getenv('QUEUE_HOST')
        self.queue_user = os.getenv('QUEUE_USER')
        self.queue_password = os.getenv('QUEUE_PASSWORD')
        
        # Connections
        self.db_conn = None
        self.queue_connection = None
        self.queue_channel = None

    def connect_to_database(self):
        """Connect to PostgreSQL database"""
        try:
            self.db_conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            self.db_conn.autocommit = False  # Use transactions
            logger.info(f"Connected to database: {self.db_host}/{self.db_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def connect_to_queue(self):
        """Connect to RabbitMQ"""
        try:
            credentials = pika.PlainCredentials(self.queue_user, self.queue_password)
            self.queue_connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=self.queue_host,
                    credentials=credentials,
                    heartbeat=60,
                    blocked_connection_timeout=300
                )
            )
            self.queue_channel = self.queue_connection.channel()
            
            # Don't declare queue - harmony queue should already exist with proper config
            
            logger.info(f"Connected to RabbitMQ: {self.queue_host}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False

    def get_images_with_bbox_results(self):
        """Get all images that have bbox service results and their image data"""
        try:
            cursor = self.db_conn.cursor()
            
            # Get images that have bbox service results (spatial services)
            query = """
                SELECT DISTINCT i.image_id, i.image_filename, i.image_url, i.image_path
                FROM images i
                JOIN results r ON i.image_id = r.image_id
                WHERE r.service IN ('yolo_v8', 'yolo_365', 'yolo_oi7', 'detectron2', 'rtdetr')
                AND r.status = 'success'
                ORDER BY i.image_id
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            
            logger.info(f"Found {len(results)} images with bbox results")
            return results
            
        except Exception as e:
            logger.error(f"Error getting images: {e}")
            return []

    def clear_harmony_consensus_data(self, image_ids=None):
        """Clear harmony and consensus data, optionally for specific image IDs"""
        try:
            cursor = self.db_conn.cursor()
            
            if image_ids:
                # Clear specific images
                image_id_list = tuple(image_ids)
                logger.info(f"Clearing data for {len(image_ids)} specific images")
                
                # Step 1: Delete postprocessing data
                cursor.execute("""
                    DELETE FROM postprocessing 
                    WHERE merged_box_id IN (
                        SELECT merged_id FROM merged_boxes WHERE image_id IN %s
                    )
                """, (image_id_list,))
                postprocessing_deleted = cursor.rowcount
                
                # Step 2: Delete consensus data
                cursor.execute("DELETE FROM consensus WHERE image_id IN %s", (image_id_list,))
                consensus_deleted = cursor.rowcount
                
                # Step 3: Delete merged_boxes data
                cursor.execute("DELETE FROM merged_boxes WHERE image_id IN %s", (image_id_list,))
                merged_boxes_deleted = cursor.rowcount
                
            else:
                # Clear ALL data
                logger.warning("Clearing ALL harmony and consensus data!")
                
                # Step 1: Delete all postprocessing
                cursor.execute("DELETE FROM postprocessing")
                postprocessing_deleted = cursor.rowcount
                
                # Step 2: Delete all consensus
                cursor.execute("DELETE FROM consensus")
                consensus_deleted = cursor.rowcount
                
                # Step 3: Delete all merged_boxes
                cursor.execute("DELETE FROM merged_boxes")
                merged_boxes_deleted = cursor.rowcount
            
            # Commit the transaction
            self.db_conn.commit()
            cursor.close()
            
            logger.info(f"Deleted: {postprocessing_deleted} postprocessing, {consensus_deleted} consensus, {merged_boxes_deleted} merged_boxes records")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            try:
                self.db_conn.rollback()
            except:
                pass
            return False

    def fetch_image_data(self, image_url, image_path):
        """Fetch image data - simplified version, you may need to adapt this"""
        import base64
        import requests
        from PIL import Image
        import io
        
        try:
            if image_url:
                # Download from URL
                response = requests.get(image_url, timeout=30)
                response.raise_for_status()
                image_data = base64.b64encode(response.content)
            elif image_path:
                # Read from local file
                with open(image_path, 'rb') as f:
                    image_data = base64.b64encode(f.read())
            else:
                return None
                
            return image_data.decode('latin-1')  # For JSON transport
            
        except Exception as e:
            logger.warning(f"Failed to fetch image data: {e}")
            return None

    def retrigger_harmony_processing(self, images):
        """Send messages to harmony queue to retrigger processing"""
        try:
            success_count = 0
            failed_count = 0
            
            for image_id, image_filename, image_url, image_path in images:
                try:
                    # Fetch image data (required for postprocessing)
                    image_data = self.fetch_image_data(image_url, image_path)
                    
                    if not image_data:
                        logger.warning(f"Skipping {image_filename} - could not fetch image data")
                        failed_count += 1
                        continue
                    
                    # Create harmony message
                    message = {
                        'image_id': image_id,
                        'image_filename': image_filename or f'image_{image_id}',
                        'image_data': image_data,
                        'service': 'harmony_reset',
                        'worker_id': 'reset_script',
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    # Send to harmony queue
                    self.queue_channel.basic_publish(
                        exchange='',
                        routing_key='harmony',
                        body=json.dumps(message),
                        properties=pika.BasicProperties(delivery_mode=2)  # Persistent
                    )
                    
                    success_count += 1
                    if success_count % 100 == 0:
                        logger.info(f"Queued {success_count} harmony messages...")
                        
                except Exception as e:
                    logger.warning(f"Failed to queue {image_filename}: {e}")
                    failed_count += 1
            
            logger.info(f"Queued {success_count} harmony messages, {failed_count} failed")
            return success_count
            
        except Exception as e:
            logger.error(f"Error retriggering harmony processing: {e}")
            return 0

    def run_reset(self, image_ids=None, dry_run=False):
        """Main reset workflow"""
        logger.info("Starting harmony and consensus reset...")
        
        # Connect to services
        if not self.connect_to_database():
            return False
        if not self.connect_to_queue():
            return False
        
        try:
            # Get images that need processing
            images = self.get_images_with_bbox_results()
            if not images:
                logger.warning("No images with bbox results found")
                return False
            
            # Filter to specific image IDs if provided
            if image_ids:
                images = [img for img in images if img[0] in image_ids]
                logger.info(f"Filtered to {len(images)} specific images")
            
            if dry_run:
                logger.info(f"DRY RUN: Would process {len(images)} images")
                return True
            
            # Clear existing data
            if not self.clear_harmony_consensus_data([img[0] for img in images]):
                logger.error("Failed to clear data")
                return False
            
            # Retrigger processing
            queued_count = self.retrigger_harmony_processing(images)
            if queued_count == 0:
                logger.error("Failed to queue any messages")
                return False
            
            logger.info(f"Reset complete: {queued_count} images queued for reprocessing")
            return True
            
        finally:
            # Cleanup connections
            if self.db_conn:
                self.db_conn.close()
            if self.queue_connection and not self.queue_connection.is_closed:
                self.queue_connection.close()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Reset harmony and consensus data')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--image-ids', type=str, help='Comma-separated list of image IDs to reset (default: all)')
    parser.add_argument('--confirm', action='store_true', help='Required to actually run the reset')
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.confirm:
        print("ERROR: This script will delete harmony and consensus data!")
        print("Use --dry-run to see what would happen, or --confirm to actually run it")
        return 1
    
    # Parse image IDs if provided
    image_ids = None
    if args.image_ids:
        try:
            image_ids = [int(x.strip()) for x in args.image_ids.split(',')]
            logger.info(f"Will reset specific image IDs: {image_ids}")
        except ValueError:
            logger.error("Invalid image IDs format. Use comma-separated integers.")
            return 1
    
    # Run the reset
    reset_tool = HarmonyConsensusReset()
    success = reset_tool.run_reset(image_ids=image_ids, dry_run=args.dry_run)
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())