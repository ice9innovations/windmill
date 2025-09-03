#!/usr/bin/env python3
"""
Bounding Box Merger Worker - Harmonizes bbox results from yolo, rtdetr, detectron2
Triggers immediately after each bbox service completes for continuous harmonization
"""
import os
import json
import time
import logging
import socket
import base64
import io
import psycopg2
import pika
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image

class BoundingBoxMergerWorker:
    """Continuous bounding box harmonization worker"""
    
    def __init__(self):
        # Load configuration
        if not load_dotenv():
            raise ValueError("Could not load .env file")
        
        # Database configuration
        self.db_host = self._get_required('DB_HOST')
        self.db_name = self._get_required('DB_NAME')
        self.db_user = self._get_required('DB_USER')
        self.db_password = self._get_required('DB_PASSWORD')
        
        # Worker configuration
        self.worker_id = os.getenv('WORKER_ID', f'bbox_merger_{int(time.time())}')
        
        # Queue configuration
        self.queue_name = os.getenv('BBOX_QUEUE_NAME', 'queue_bbox_merge')
        self.queue_host = self._get_required('QUEUE_HOST')
        self.queue_user = self._get_required('QUEUE_USER')
        self.queue_password = self._get_required('QUEUE_PASSWORD')
        # Removed downstream_queue - using new postprocessing dispatch instead
        
        # Bounding box services to harmonize
        self.bbox_services = ['yolov8', 'rtdetr', 'detectron2', 'xception', 'clip']
        
        
        # Logging
        self.setup_logging()
        self.db_conn = None
        self.read_db_conn = None
        
        # Queue connections
        self.queue_connection = None
        self.queue_channel = None
        
    
    def _get_required(self, key):
        """Get required environment variable or raise error"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value
        
    def setup_logging(self):
        """Configure logging"""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('bbox_merger')
    
    
    
    
    def connect_to_database(self):
        """Connect to PostgreSQL database"""
        try:
            # Close existing connections if any
            if hasattr(self, 'db_conn') and self.db_conn:
                try:
                    self.db_conn.close()
                except:
                    pass
            if hasattr(self, 'read_db_conn') and self.read_db_conn:
                try:
                    self.read_db_conn.close()
                except:
                    pass
            
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
    
    def ensure_database_connection(self):
        """Ensure database connections are healthy, reconnect if needed"""
        reconnect_needed = False
        
        # Check main connection
        try:
            if not self.db_conn or self.db_conn.closed:
                reconnect_needed = True
            else:
                # Test connection with simple query
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
                # Test connection with simple query
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
    
    def connect_to_queue(self):
        """Connect to RabbitMQ for queue-based processing"""
        try:
            credentials = pika.PlainCredentials(self.queue_user, self.queue_password)
            self.queue_connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=self.queue_host,
                    credentials=credentials
                )
            )
            self.queue_channel = self.queue_connection.channel()
            
            # Declare queues (create if they don't exist)
            self.queue_channel.queue_declare(queue=self.queue_name, durable=True)
            # Removed old downstream queue
            
            # Declare bbox postprocessing queues
            self.queue_channel.queue_declare(queue='queue_bbox_colors', durable=True)
            self.queue_channel.queue_declare(queue='queue_bbox_face', durable=True)
            self.queue_channel.queue_declare(queue='queue_bbox_pose', durable=True)
            
            # Set prefetch count for fair distribution
            self.queue_channel.basic_qos(prefetch_count=1)
            
            self.logger.info(f"Connected to RabbitMQ at {self.queue_host}")
            self.logger.info(f"Consuming from: {self.queue_name}")
            self.logger.info(f"Publishing to postprocessing queues: bbox_colors, bbox_face, bbox_pose")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False
    
    # Removed publish_to_downstream - using new dispatch_bbox_postprocessing instead
    
    def dispatch_bbox_postprocessing(self, image_id, image_filename, merged_data):
        """Dispatch bbox postprocessing jobs for each merged box"""
        try:
            # Get image path from database (use main connection to stay in transaction)
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT image_path FROM images WHERE image_id = %s", (image_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if not result:
                self.logger.warning(f"No image path found for image_id {image_id}")
                return
                
            image_path = result[0]
            
            # Process each merged box
            for group_key, group_data in merged_data['grouped_objects'].items():
                instances = group_data.get('instances', [])
                for instance in instances:
                    merged_bbox = instance.get('merged_bbox', {})
                    cluster_id = instance.get('cluster_id', '')
                    
                    # Get merged_box_id from database for this instance (use main connection to see uncommitted data)
                    cursor = self.db_conn.cursor()
                    cursor.execute("""
                        SELECT merged_id FROM merged_boxes 
                        WHERE image_id = %s AND merged_data->>'cluster_id' = %s
                        ORDER BY created DESC LIMIT 1
                    """, (image_id, cluster_id))
                    box_result = cursor.fetchone()
                    cursor.close()
                    
                    if not box_result:
                        continue
                        
                    merged_box_id = box_result[0]
                    
                    # Crop the bounding box
                    cropped_image_data = self.crop_bbox_from_image(image_path, merged_bbox)
                    if not cropped_image_data:
                        continue
                    
                    # Create postprocessing message
                    base_message = {
                        'merged_box_id': merged_box_id,
                        'image_id': image_id,
                        'cluster_id': cluster_id,
                        'bbox': merged_bbox,
                        'cropped_image_data': cropped_image_data.decode('latin-1')  # Encode bytes for JSON
                    }
                    
                    # Always dispatch colors
                    self.publish_bbox_message('queue_bbox_colors', base_message)
                    
                    # Dispatch face/pose for person boxes
                    if self.is_person_box(instance):
                        self.publish_bbox_message('queue_bbox_face', base_message)
                        self.publish_bbox_message('queue_bbox_pose', base_message)
                    
        except Exception as e:
            self.logger.error(f"Error in dispatch_bbox_postprocessing: {e}")
            raise
    
    def crop_bbox_from_image(self, image_path, bbox):
        """Crop bbox region from image and return as base64 encoded bytes"""
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
                
                # Convert to base64 encoded bytes for JSON transport
                img_buffer = io.BytesIO()
                cropped_img.save(img_buffer, format='JPEG', quality=90)
                img_buffer.seek(0)
                
                return base64.b64encode(img_buffer.getvalue())
                
        except Exception as e:
            self.logger.error(f"Failed to crop bbox from {image_path}: {e}")
            return None
    
    def is_person_box(self, instance):
        """Check if a bounding box should trigger face/pose processing"""
        emoji = instance.get('emoji', '')
        
        # Only trigger face/pose for specific person emojis
        face_pose_emojis = ['🧑', '🙂', '👩', '🧒']
        
        return emoji in face_pose_emojis
    
    def publish_bbox_message(self, queue_name, message):
        """Publish message to bbox postprocessing queue"""
        try:
            self.queue_channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2  # Make message persistent
                )
            )
            self.logger.debug(f"Published bbox message to {queue_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to publish to {queue_name}: {e}")
            raise
    
    def process_queue_message(self, ch, method, properties, body):
        """Process a single queue message"""
        try:
            # Parse message
            message = json.loads(body)
            self.logger.debug(f"Processing queue message: {message}")
            
            # Extract image info
            image_id = message['image_id']
            image_filename = message.get('image_filename', f'image_{image_id}')
            
            # Process the bbox merge for this image
            result = self.update_merged_boxes_for_image(image_id, image_filename)
            
            if result['success']:
                
                # Only send spatial enrichment message if merged boxes were actually created
                if result['merged_boxes_created']:
                    completion_message = {
                        'image_id': image_id,
                        'image_filename': image_filename,
                        'service': 'bbox_merger',
                        'worker_id': self.worker_id,
                        'processed_at': datetime.now().isoformat()
                    }
                    # Old downstream publishing removed - using new postprocessing dispatch
                    self.logger.info(f"Successfully processed bbox merge for {image_filename}, postprocessing dispatched")
                else:
                    self.logger.info(f"Successfully processed bbox merge for {image_filename}, no merged boxes to enrich")
                
                # Acknowledge the message
                ch.basic_ack(delivery_tag=method.delivery_tag)
                
            else:
                # Reject and requeue for retry
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                self.logger.error(f"Failed to process bbox merge for {image_filename}, requeuing")
                
        except Exception as e:
            self.logger.error(f"Error processing queue message: {e}")
            # Reject and requeue for retry
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    
    def get_bbox_results_for_image(self, image_id):
        """Get all bbox service results for an image"""
        try:
            # Ensure healthy database connection
            if not self.ensure_database_connection():
                self.logger.error("Could not establish database connection")
                return []
                
            cursor = self.read_db_conn.cursor()
            
            query = """
                SELECT service, data, result_id
                FROM results
                WHERE image_id = %s 
                AND service IN %s 
                AND status = 'success'
                ORDER BY service
            """
            
            cursor.execute(query, (image_id, tuple(self.bbox_services)))
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting bbox results for image {image_id}: {e}")
            # Try once more after reconnection
            try:
                if self.ensure_database_connection():
                    cursor = self.read_db_conn.cursor()
                    cursor.execute(query, (image_id, tuple(self.bbox_services)))
                    results = cursor.fetchall()
                    cursor.close()
                    self.logger.info(f"Successfully retried bbox results query for image {image_id}")
                    return results
            except Exception as retry_e:
                self.logger.error(f"Retry failed for image {image_id}: {retry_e}")
            
            return []
    
    def harmonize_bounding_boxes(self, bbox_results):
        """Harmonize bounding boxes using BoundingBoxService logic"""
        if not bbox_results:
            return None
        
        # Convert database results to service format
        all_detections = []
        source_result_ids = []
        
        for service, data, result_id in bbox_results:
            source_result_ids.append(result_id)
            
            if not isinstance(data, dict) or 'predictions' not in data:
                continue
            
            predictions = data['predictions']
            for prediction in predictions:
                if not prediction.get('bbox'):
                    continue
                
                # Extract detection with harmonized format
                detection = {
                    'service': service,
                    'label': prediction.get('label', ''),
                    'emoji': prediction.get('emoji', ''),
                    'bbox': prediction['bbox'],
                    'confidence': prediction.get('confidence', 0.0),
                    'type': prediction.get('type', 'object_detection')
                }
                all_detections.append(detection)
        
        if not all_detections:
            return None
        
        # Apply BoundingBoxService harmonization logic
        grouped_objects = self.group_by_label_with_cross_service_clustering(all_detections)
        
        # Package harmonized results
        harmonized_data = {
            'all_detections': all_detections,
            'grouped_objects': grouped_objects,
            'source_services': list(set(r[0] for r in bbox_results)),
            'total_detections': len(all_detections),
            'harmonization_algorithm': 'cross_service_clustering_v1',
            'source_result_ids': source_result_ids
        }
        
        return harmonized_data
    
    def group_by_label_with_cross_service_clustering(self, detections):
        """Simplified version of BoundingBoxService clustering logic"""
        groups = {}
        
        # Step 1: Group by label/emoji
        for detection in detections:
            key = detection['type'] if detection['type'] == 'face_detection' else detection['emoji']
            if key not in groups:
                groups[key] = {
                    'label': detection['label'],
                    'emoji': detection['emoji'],
                    'type': detection['type'],
                    'detections': [],
                    'instances': []
                }
            groups[key]['detections'].append(detection)
        
        # Step 2: Create cross-service instances for each group
        for group_key, group in groups.items():
            if group['detections']:
                group['instances'] = self.create_cross_service_instances(
                    group['detections'], 
                    group['emoji']
                )
        
        return groups
    
    def create_cross_service_instances(self, detections, emoji):
        """Create instances with cross-service clustering"""
        if not detections:
            return []
        
        # Find clusters of overlapping detections
        clusters = self.find_cross_service_clusters(detections)
        instances = []
        
        for i, cluster in enumerate(clusters):
            # Clean cluster (remove duplicates, filter weak detections)
            cleaned_cluster = self.clean_cluster(cluster)
            if not cleaned_cluster:
                continue
            
            # Calculate merged bounding box
            if len(cleaned_cluster) == 1:
                merged_bbox = cleaned_cluster[0]['bbox']
            else:
                boxes = [d['bbox'] for d in cleaned_cluster]
                x1 = min(b['x'] for b in boxes)
                y1 = min(b['y'] for b in boxes)
                x2 = max(b['x'] + b['width'] for b in boxes)
                y2 = max(b['y'] + b['height'] for b in boxes)
                
                merged_bbox = {
                    'x': x1,
                    'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1
                }
            
            # Create instance
            services = list(set(d['service'] for d in cleaned_cluster))
            avg_confidence = sum(d['confidence'] for d in cleaned_cluster) / len(cleaned_cluster)
            
            instance = {
                'cluster_id': f"{cleaned_cluster[0]['emoji']}_{i+1}",
                'emoji': emoji,
                'labels': list(set(d['label'] for d in cleaned_cluster)),
                'merged_bbox': merged_bbox,
                'detection_count': len(cleaned_cluster),
                'avg_confidence': round(avg_confidence, 3),
                'contributing_services': services,
                'detections': [{'service': d['service'], 'confidence': d['confidence']} 
                             for d in cleaned_cluster]
            }
            instances.append(instance)
        
        return instances
    
    def find_cross_service_clusters(self, detections):
        """Find clusters using IoU overlap"""
        clusters = []
        used = set()
        
        for i, detection in enumerate(detections):
            if i in used:
                continue
            
            cluster = [detection]
            used.add(i)
            
            # Find overlapping detections
            for j in range(i + 1, len(detections)):
                if j in used:
                    continue
                
                overlap = self.calculate_overlap_ratio(detection['bbox'], detections[j]['bbox'])
                if overlap > 0.3:  # 30% overlap threshold
                    cluster.append(detections[j])
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def clean_cluster(self, cluster):
        """Remove same-service duplicates and filter weak single detections"""
        if not cluster:
            return None
        
        # Group by service, keep highest confidence per service
        service_groups = {}
        for detection in cluster:
            service = detection['service']
            if service not in service_groups:
                service_groups[service] = []
            service_groups[service].append(detection)
        
        cleaned = []
        for service, detections in service_groups.items():
            if len(detections) > 1:
                # Keep highest confidence
                best = max(detections, key=lambda d: d['confidence'])
                cleaned.append(best)
            else:
                cleaned.append(detections[0])
        
        # Filter single weak detections
        if len(cleaned) == 1 and cleaned[0]['confidence'] < 0.85:
            return None
        
        return cleaned
    
    def calculate_overlap_ratio(self, box1, box2):
        """Calculate IoU overlap ratio"""
        x1 = max(box1['x'], box2['x'])
        y1 = max(box1['y'], box2['y'])
        x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
        y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])
        
        if x1 >= x2 or y1 >= y2:
            return 0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def update_merged_boxes_for_image(self, image_id, image_filename):
        """Update merged boxes for a single image using safe DELETE+INSERT pattern"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # Ensure healthy database connection
                if not self.ensure_database_connection():
                    self.logger.error("Could not establish database connection")
                    if attempt < max_retries - 1:
                        self.logger.info(f"Retrying database operation (attempt {attempt + 1}/{max_retries})")
                        time.sleep(2)  # Wait before retry
                        continue
                    return {'success': False, 'merged_boxes_created': False}
                
                # Get bbox results
                bbox_results = self.get_bbox_results_for_image(image_id)
                if not bbox_results:
                    self.logger.debug(f"No bbox results for image {image_id}, skipping")
                    return {'success': True, 'merged_boxes_created': False}  # Not an error - just no data to process
                
                # Harmonize bounding boxes
                start_time = time.time()
                merged_data = self.harmonize_bounding_boxes(bbox_results)
                processing_time = time.time() - start_time
                
                if not merged_data:
                    self.logger.warning(f"Could not harmonize boxes for image {image_id}")
                    return {'success': False, 'merged_boxes_created': False}
                
                # Safe atomic DELETE + INSERT with foreign key handling
                cursor = self.db_conn.cursor()
                
                # Step 1: Clear postprocessing references to avoid FK constraint violation
                cursor.execute("""
                    UPDATE postprocessing 
                    SET merged_box_id = NULL 
                    WHERE merged_box_id IN (
                        SELECT merged_id FROM merged_boxes WHERE image_id = %s
                    )
                """, (image_id,))
                
                # Step 2: Delete old merged boxes (now safe)
                cursor.execute("DELETE FROM merged_boxes WHERE image_id = %s", (image_id,))
                deleted_count = cursor.rowcount
                
                # Step 3: Insert new merged boxes - ONE ROW PER MERGED BOX
                merged_box_count = 0
                for group_key, group_data in merged_data['grouped_objects'].items():
                    instances = group_data.get('instances', [])
                    for instance in instances:
                        # Store merged box data directly
                        merged_box_data = {
                            'cluster_id': instance.get('cluster_id', ''),
                            'emoji': instance.get('emoji', ''),
                            'labels': instance.get('labels', []),
                            'merged_bbox': instance.get('merged_bbox', {}),
                            'detection_count': instance.get('detection_count', 0),
                            'avg_confidence': instance.get('avg_confidence', 0.0),
                            'contributing_services': instance.get('contributing_services', []),
                            'detections': instance.get('detections', []),
                            'harmonization_algorithm': merged_data['harmonization_algorithm'],
                            'source_result_ids': merged_data['source_result_ids']
                        }
                        
                        cursor.execute("""
                            INSERT INTO merged_boxes (image_id, source_result_ids, merged_data, worker_id, status)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (
                            image_id,
                            merged_data['source_result_ids'],
                            json.dumps(merged_box_data),
                            self.worker_id,
                            'success'
                        ))
                        merged_box_count += 1
                
                # Commit the transaction
                self.db_conn.commit()
                cursor.close()
                
                services = merged_data['source_services']
                detection_count = merged_data['total_detections']
                
                if deleted_count > 0:
                    self.logger.info(f"Reharmonized boxes for {image_filename} ({detection_count} detections → {merged_box_count} merged boxes from {services})")
                else:
                    self.logger.info(f"Harmonized boxes for {image_filename} ({detection_count} detections → {merged_box_count} merged boxes from {services})")
                
                # Dispatch bbox postprocessing jobs for each merged box
                if merged_box_count > 0:
                    try:
                        self.dispatch_bbox_postprocessing(image_id, image_filename, merged_data)
                    except Exception as e:
                        self.logger.warning(f"Failed to dispatch postprocessing for {image_filename}: {e}")
                        # Don't fail the whole operation if postprocessing dispatch fails
                
                if attempt > 0:
                    self.logger.info(f"Successfully processed {image_filename} after {attempt + 1} attempts")
                
                return {'success': True, 'merged_boxes_created': merged_box_count > 0}
                
            except Exception as e:
                self.logger.error(f"Error updating merged boxes for image {image_id} (attempt {attempt + 1}/{max_retries}): {e}")
                
                # Rollback transaction on error
                try:
                    if self.db_conn:
                        self.db_conn.rollback()
                except:
                    pass
                
                # If not the last attempt, wait and retry
                if attempt < max_retries - 1:
                    self.logger.info(f"Will retry after 2 seconds...")
                    time.sleep(2)
                    continue
                    
                return {'success': False, 'merged_boxes_created': False}
        
        return {'success': False, 'merged_boxes_created': False}
    
    
    
    def run(self):
        """Main entry point - pure queue-based processing"""
        self.logger.info(f"Starting bbox merger worker in QUEUE MODE ({self.worker_id})")
        self.logger.info(f"Queue: {self.queue_name} → postprocessing queues")
        
        if not self.connect_to_database():
            return 1
            
        if not self.connect_to_queue():
            return 1
        
        # Setup queue consumer
        self.queue_channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.process_queue_message
        )
        
        self.logger.info("Waiting for bbox merge messages. Press CTRL+C to exit")
        
        try:
            self.queue_channel.start_consuming()
        except KeyboardInterrupt:
            self.logger.info("Stopping bbox merger worker...")
            self.queue_channel.stop_consuming()
        finally:
            self.cleanup_connections()
            
        return 0
    
    def cleanup_connections(self):
        """Clean up all connections"""
        pass  # No connections to clean up
        
        if self.db_conn:
            self.db_conn.close()
        if self.read_db_conn:
            self.read_db_conn.close()
        if self.queue_connection and not self.queue_connection.is_closed:
            self.queue_connection.close()
        
        self.logger.info("Bbox merger worker stopped")
    

def main():
    """Main entry point"""
    try:
        worker = BoundingBoxMergerWorker()
        return worker.run()
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Bbox merger worker error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
