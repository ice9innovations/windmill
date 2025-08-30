#!/usr/bin/env python3
"""
Spatial Enrichment Worker - Post-processes harmonized bounding boxes
Scans merged_boxes and adds face + pose data for person boxes, colors for all boxes
"""
import os
import json
import time
import logging
import socket
import psycopg2
import mysql.connector
import pika
import requests
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import io

class SpatialEnrichmentWorker:
    """Post-processing worker for spatial enrichment of harmonized bounding boxes"""
    
    def __init__(self):
        # Load configuration
        if not load_dotenv():
            raise ValueError("Could not load .env file")
        
        # Load service definitions
        with open('service_config.json', 'r') as f:
            self.service_definitions = json.load(f)['services']
        
        # Database configuration
        self.db_host = self._get_required('DB_HOST')
        self.db_name = self._get_required('DB_NAME')
        self.db_user = self._get_required('DB_USER')
        self.db_password = self._get_required('DB_PASSWORD')
        
        # Services for enrichment (from service_config.json)
        face_config = self.service_definitions.get('face', {})
        pose_config = self.service_definitions.get('pose', {})
        colors_config = self.service_definitions.get('colors', {})
        
        self.face_service_url = f"http://{face_config.get('host', 'localhost')}:{face_config.get('port', 7772)}{face_config.get('endpoint', '/analyze')}"
        self.pose_service_url = f"http://{pose_config.get('host', 'localhost')}:{pose_config.get('port', 7786)}{pose_config.get('endpoint', '/analyze')}"
        self.colors_service_url = f"http://{colors_config.get('host', 'localhost')}:{colors_config.get('port', 7770)}{colors_config.get('endpoint', '/analyze')}"
        
        # Worker configuration
        self.worker_id = os.getenv('WORKER_ID', f'spatial_enrichment_{int(time.time())}')
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '15'))
        
        # Queue configuration
        self.queue_name = os.getenv('SPATIAL_QUEUE_NAME', 'queue_spatial_enrichment')
        self.queue_host = self._get_required('QUEUE_HOST')
        self.queue_user = self._get_required('QUEUE_USER')
        self.queue_password = self._get_required('QUEUE_PASSWORD')
        self.downstream_queue = os.getenv('CONSENSUS_QUEUE', 'queue_consensus')
        
        # Person detection labels that should get face/pose enrichment
        self.person_labels = ['person', 'human', 'people', 'man', 'woman', 'child', 'boy', 'girl']
        self.person_emojis = ['🧑', '🙂', '👤', '👥']
        
        # Logging
        self.setup_logging()
        self.db_conn = None
        self.read_db_conn = None
        
    def setup_logging(self):
        """Configure logging"""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('spatial_enrichment')
    
    def _get_required(self, key):
        """Get required environment variable with no fallback"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value
    
    def connect_to_database(self):
        """Connect to PostgreSQL database"""
        try:
            # Main connection for transactions (write operations)
            self.db_conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            self.db_conn.autocommit = False
            
            # Read-only connection for queries (prevents idle transactions)
            self.read_db_conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            self.read_db_conn.autocommit = True  # Auto-commit for read queries
            
            self.logger.info(f"Connected to PostgreSQL at {self.db_host}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def connect_to_rabbitmq(self):
        """Connect to RabbitMQ for queue-based processing"""
        try:
            credentials = pika.PlainCredentials(self.queue_user, self.queue_password)
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=self.queue_host,
                    credentials=credentials
                )
            )
            self.channel = self.connection.channel()
            
            # Declare the queues
            self.channel.queue_declare(queue=self.queue_name, durable=True)
            self.channel.queue_declare(queue=self.downstream_queue, durable=True)
            
            # Set prefetch count for fair distribution
            self.channel.basic_qos(prefetch_count=1)
            
            self.logger.info(f"Connected to RabbitMQ at {self.queue_host}, queue: {self.queue_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False
    
    def publish_to_consensus_queue(self, image_id, image_filename):
        """Publish message to consensus queue after spatial enrichment completion"""
        try:
            message = {
                'image_id': image_id,
                'image_filename': image_filename,
                'service': 'spatial_enrichment',
                'worker_id': self.worker_id,
                'processed_at': datetime.now().isoformat()
            }
            
            self.channel.basic_publish(
                exchange='',
                routing_key=self.downstream_queue,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2  # Make message persistent
                )
            )
            
            self.logger.debug(f"Published to {self.downstream_queue}: {message}")
            
        except Exception as e:
            self.logger.error(f"Failed to publish to consensus queue: {e}")
    
    
    def extract_bbox_instances(self, merged_data):
        """Extract individual bbox instances from harmonized data"""
        instances = []
        
        if not isinstance(merged_data, dict):
            return instances
        
        # Each merged_boxes row now contains exactly one merged box
        instances.append({
            'cluster_id': merged_data.get('cluster_id', ''),
            'emoji': merged_data.get('emoji', ''),
            'label': merged_data.get('label', ''),
            'merged_bbox': merged_data.get('merged_bbox', {}),
            'detection_count': merged_data.get('detection_count', 0),
            'avg_confidence': merged_data.get('avg_confidence', 0.0)
        })
        
        return instances
    
    def is_person_box(self, instance):
        """Check if a bounding box represents a person"""
        label = instance.get('label', '').lower()
        emoji = instance.get('emoji', '')
        
        # Check label
        if any(person_label in label for person_label in self.person_labels):
            return True
        
        # Check emoji
        if emoji in self.person_emojis:
            return True
        
        return False
    
    def crop_bbox_from_image(self, image_path, bbox):
        """Crop bbox region from image and return as bytes for POST request"""
        try:
            # Load image
            with Image.open(image_path) as img:
                # Extract bbox coordinates
                x = bbox['x']
                y = bbox['y'] 
                width = bbox['width']
                height = bbox['height']
                
                # Crop the bbox region
                crop_box = (x, y, x + width, y + height)
                cropped_img = img.crop(crop_box)
                
                # Convert to bytes for POST request
                img_buffer = io.BytesIO()
                cropped_img.save(img_buffer, format='JPEG', quality=90)
                img_buffer.seek(0)
                
                return img_buffer.getvalue()
                
        except Exception as e:
            self.logger.error(f"Failed to crop bbox from {image_path}: {e}")
            return None
    
    def enrich_person_bbox(self, image_path, instance):
        """Add face and pose data for person bounding boxes"""
        enrichment_data = {
            'face_data': None,
            'pose_data': None,
            'enrichment_type': 'person_analysis'
        }
        
        # Crop bbox region in memory
        cropped_image_data = self.crop_bbox_from_image(image_path, instance['merged_bbox'])
        if not cropped_image_data:
            return enrichment_data
        
        # Get face data via POST with cropped image
        try:
            files = {'file': ('bbox_crop.jpg', io.BytesIO(cropped_image_data), 'image/jpeg')}
            face_response = requests.post(
                self.face_service_url,
                files=files,
                timeout=self.request_timeout
            )
            
            if face_response.status_code == 200:
                face_data = face_response.json()
                if face_data.get('status') == 'success' and face_data.get('predictions'):
                    enrichment_data['face_data'] = face_data
                    self.logger.debug(f"Added face data for {instance['cluster_id']}")
            
        except Exception as e:
            self.logger.warning(f"Failed to get face data for {instance['cluster_id']}: {e}")
        
        # Get pose data via POST with cropped image
        try:
            files = {'file': ('bbox_crop.jpg', io.BytesIO(cropped_image_data), 'image/jpeg')}
            pose_response = requests.post(
                self.pose_service_url,
                files=files,
                timeout=self.request_timeout
            )
            
            if pose_response.status_code == 200:
                pose_data = pose_response.json()
                if pose_data.get('status') == 'success' and pose_data.get('predictions'):
                    enrichment_data['pose_data'] = pose_data
                    self.logger.debug(f"Added pose data for {instance['cluster_id']}")
            
        except Exception as e:
            self.logger.warning(f"Failed to get pose data for {instance['cluster_id']}: {e}")
        
        return enrichment_data
    
    def enrich_bbox_colors(self, image_path, instance):
        """Add color analysis for any bounding box"""
        enrichment_data = {
            'color_data': None,
            'enrichment_type': 'color_analysis'
        }
        
        # Crop bbox region in memory
        cropped_image_data = self.crop_bbox_from_image(image_path, instance['merged_bbox'])
        if not cropped_image_data:
            return enrichment_data
        
        # Get color data via POST with cropped image
        try:
            files = {'file': ('bbox_crop.jpg', io.BytesIO(cropped_image_data), 'image/jpeg')}
            colors_response = requests.post(
                self.colors_service_url,
                files=files,
                timeout=self.request_timeout
            )
            
            if colors_response.status_code == 200:
                colors_data = colors_response.json()
                if colors_data.get('status') == 'success' and colors_data.get('predictions'):
                    enrichment_data['color_data'] = colors_data
                    self.logger.debug(f"Added color data for {instance['cluster_id']}")
            
        except Exception as e:
            self.logger.warning(f"Failed to get color data for {instance['cluster_id']}: {e}")
        
        return enrichment_data
    
    def process_merged_box_enrichment(self, image_id, merged_id, image_filename, image_path, merged_data):
        """Process spatial enrichment for a single merged box"""
        
        try:
            # Extract bbox instances from harmonized data
            instances = self.extract_bbox_instances(merged_data)
            if not instances:
                self.logger.debug(f"No instances found in merged box {merged_id}")
                return False
            
            enrichment_results = []
            
            # Process each bbox instance
            for instance in instances:
                cluster_id = instance.get('cluster_id', '')
                
                # Face/pose enrichment for person boxes
                if self.is_person_box(instance):
                    person_enrichment = self.enrich_person_bbox(image_path, instance)
                    person_enrichment['cluster_id'] = cluster_id
                    person_enrichment['bbox'] = instance['merged_bbox']
                    enrichment_results.append(person_enrichment)
                
                # Color enrichment for all boxes
                color_enrichment = self.enrich_bbox_colors(image_path, instance)
                color_enrichment['cluster_id'] = cluster_id
                color_enrichment['bbox'] = instance['merged_bbox']
                enrichment_results.append(color_enrichment)
            
            # Store enrichment results
            if enrichment_results:
                self.save_enrichment_results(image_id, merged_id, enrichment_results)
                
                person_count = sum(1 for r in enrichment_results if r.get('enrichment_type') == 'person_analysis')
                color_count = sum(1 for r in enrichment_results if r.get('enrichment_type') == 'color_analysis')
                
                self.logger.info(f"Enriched {image_filename}: {person_count} person boxes (face+pose), {color_count} color analyses")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error processing enrichment for merged box {merged_id}: {e}")
            return False
    
    def save_enrichment_results(self, image_id, merged_box_id, enrichment_results):
        """Save spatial enrichment results to postprocessing table"""
        try:
            cursor = self.db_conn.cursor()
            
            # Delete old enrichment for this merged box (atomic replacement)
            cursor.execute("""
                DELETE FROM postprocessing 
                WHERE merged_box_id = %s AND service = 'spatial_enrichment'
            """, (merged_box_id,))
            
            # Insert new enrichment results
            enrichment_data = {
                'enrichment_results': enrichment_results,
                'total_instances': len(enrichment_results),
                'processing_algorithm': 'spatial_enrichment_v1'
            }
            
            cursor.execute("""
                INSERT INTO postprocessing (image_id, merged_box_id, service, data, status)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                image_id,
                merged_box_id,
                'spatial_enrichment',
                json.dumps(enrichment_data),
                'success'
            ))
            
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Error saving enrichment results: {e}")
            raise
    
    def process_queue_message(self, ch, method, properties, body):
        """Process a message from the spatial enrichment queue"""
        try:
            # Parse the message
            message_data = json.loads(body.decode('utf-8'))
            image_id = message_data['image_id']
            image_filename = message_data.get('image_filename', f'image_{image_id}')
            
            self.logger.info(f"Processing spatial enrichment for: {image_filename}")
            
            # Process spatial enrichment for this specific image
            processed = self.enrich_image_boxes(image_id, image_filename)
            
            if processed > 0:
                # Publish to consensus queue
                self.publish_to_consensus_queue(image_id, image_filename)
                self.logger.info(f"Completed spatial enrichment for {image_filename} - {processed} boxes enriched")
            else:
                self.logger.debug(f"No boxes to enrich for {image_filename}")
            
            # Acknowledge the message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            self.logger.error(f"Error processing queue message: {e}")
            # Reject and requeue for retry
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    def enrich_image_boxes(self, image_id, image_filename):
        """Enrich boxes for a specific image"""
        try:
            # Get merged boxes for this specific image that need enrichment
            cursor = self.read_db_conn.cursor()
            query = """
                SELECT mb.image_id, mb.merged_id, i.image_filename, i.image_path, mb.merged_data
                FROM merged_boxes mb
                JOIN images i ON mb.image_id = i.image_id
                WHERE mb.image_id = %s
                AND (mb.enrichment_processed IS NULL OR mb.enrichment_processed = false)
                ORDER BY mb.created DESC
            """
            
            cursor.execute(query, (image_id,))
            boxes = cursor.fetchall()
            cursor.close()
            
            if not boxes:
                self.logger.debug(f"No boxes needing enrichment for image {image_id}")
                return 0
            
            processed_count = 0
            for image_id, merged_id, image_filename, image_path, merged_data in boxes:
                if self.process_merged_box_enrichment(image_id, merged_id, image_filename, image_path, merged_data):
                    processed_count += 1
            
            return processed_count
            
        except Exception as e:
            self.logger.error(f"Error enriching boxes for image {image_id}: {e}")
            return 0
    
    def run(self):
        """Main entry point - pure queue-based processing"""
        if not self.connect_to_database():
            return 1
        
        if not self.connect_to_rabbitmq():
            return 1
        
        self.logger.info(f"Starting spatial enrichment worker ({self.worker_id})")
        self.logger.info(f"Listening on queue: {self.queue_name}")
        self.logger.info(f"Publishing to: {self.downstream_queue}")
        
        # Setup message consumer
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.process_queue_message
        )
        
        self.logger.info("Waiting for spatial enrichment messages. Press CTRL+C to exit")
        
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.logger.info("Stopping spatial enrichment worker...")
            self.channel.stop_consuming()
        finally:
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            if self.db_conn:
                self.db_conn.close()
            if self.read_db_conn:
                self.read_db_conn.close()
            self.logger.info("Spatial enrichment worker stopped")
        
        return 0

def main():
    """Main entry point"""
    try:
        worker = SpatialEnrichmentWorker()
        return worker.run()
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Spatial enrichment worker error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())