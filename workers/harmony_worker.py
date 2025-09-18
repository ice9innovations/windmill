#!/usr/bin/env python3
"""
Harmony Worker - Harmonizes bbox results from yolo, rtdetr, detectron2
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
import re
import requests
from datetime import datetime
from base_worker import BaseWorker
from PIL import Image

THRESHOLD = .1  # Balanced for 640px images
AREA_THRESHOLD = 0.8  # Balanced area matching for 640px
SMALL_OBJECT_AREA = 1024  # COCO small object definition (32x32 pixels)
SMALL_OBJECT_DISTANCE = 20  # ~3% of 640px width for small objects
# MINIMUM_CONFIDENCE = 0.4  # Confidence filtering moved to consensus worker

# Extensible merge rules: containment-based merging where smaller objects get absorbed into larger ones
CONTAINMENT_MERGE_RULES = [
    {
        'contained_emoji': '⌨️',  # keyboard
        'container_emoji': '💻',  # laptop
        'containment_threshold': 0.9,  # 90% of keyboard must be within laptop (almost entirely contained)
        'description': 'Merge keyboard into laptop when keyboard is entirely contained within laptop'
    }
    # Add more rules here as needed, e.g.:
    # {
    #     'contained_emoji': '🖱️',  # mouse
    #     'container_emoji': '💻',  # laptop
    #     'containment_threshold': 0.6,
    #     'description': 'Merge mouse into laptop when mouse is near laptop'
    # }
]

def normalize_emoji(emoji):
    """Remove variation selectors and other invisible modifiers from emoji"""
    if not emoji:
        return emoji
    # Remove variation selectors (U+FE00-U+FE0F), Mongolian selectors (U+180B-U+180D),
    # and zero-width joiner (U+200D) for consistent grouping
    return re.sub(r'[\uFE00-\uFE0F\u180B-\u180D\u200D]', '', emoji)

def normalize_person_emoji(emoji):
    """Group person emojis together using neutral person emoji as the canonical grouping key"""
    normalized = normalize_emoji(emoji)
    if normalized in ['🧑', '👩', '🧒']:
        return '🧑'  # Group all person types under neutral person emoji
    return normalized

class HarmonyWorker(BaseWorker):
    """Continuous bounding box harmonization worker"""
    
    def __init__(self):
        # Initialize with harmony service type
        super().__init__('system.harmony')
        
        # Bounding box services to harmonize - use new array notation
        self.bbox_services = self.config.get_service_group('primary.spatial[]')
        # Get clean service names for database queries (remove category prefix)
        self.clean_bbox_services = [service.split('.', 1)[1] if '.' in service else service for service in self.bbox_services]
        
        # Harmony configuration - using highest confidence for better mAP
        self.use_highest_confidence_bbox = os.getenv('HARMONY_USE_HIGHEST_CONFIDENCE', 'true').lower() == 'true'
        bbox_method = "highest-confidence selection" if self.use_highest_confidence_bbox else "union encompassing"
        self.logger.info(f"Using bbox harmonization method: {bbox_method}")
        
        # Harmony worker needs separate read connection for queries
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
                    credentials=credentials,
                    heartbeat=60,
                    blocked_connection_timeout=300,
                    connection_attempts=10,
                    retry_delay=5,
                    socket_timeout=10
                )
            )
            self.queue_channel = self.queue_connection.channel()
            
            # Also set BaseWorker's expected connection variables for trigger_consensus()
            self.connection = self.queue_connection
            self.channel = self.queue_channel
            
            # Declare queues (create if they don't exist) with DLQ (TTL optional)
            def declare_with_dlq(channel, queue_name):
                dlq_name = f"{queue_name}.dlq"
                channel.queue_declare(queue=dlq_name, durable=True)
                args = {
                    'x-dead-letter-exchange': '',
                    'x-dead-letter-routing-key': dlq_name,
                    'x-max-length': int(os.getenv('QUEUE_MAX_LENGTH', '100000'))
                }
                ttl_env = os.getenv('QUEUE_MESSAGE_TTL_MS')
                if ttl_env and ttl_env.isdigit() and int(ttl_env) > 0:
                    args['x-message-ttl'] = int(ttl_env)
                channel.queue_declare(queue=queue_name, durable=True, arguments=args)

            declare_with_dlq(self.queue_channel, self.queue_name)
            # Removed old downstream queue
            
            # Declare bbox postprocessing queues - read from config
            colors_queue = self._get_queue_name('postprocessing.colors')
            face_queue = self._get_queue_name('postprocessing.face')
            pose_queue = self._get_queue_name('postprocessing.pose')
            
            declare_with_dlq(self.queue_channel, colors_queue)
            declare_with_dlq(self.queue_channel, face_queue)
            declare_with_dlq(self.queue_channel, pose_queue)
            
            # Store queue names for later use
            self.postprocessing_queues = {
                'colors': colors_queue,
                'face': face_queue, 
                'pose': pose_queue
            }
            
            # Set prefetch count for fair distribution
            self.queue_channel.basic_qos(prefetch_count=1)
            
            self.logger.info(f"Connected to RabbitMQ at {self.queue_host}")
            self.logger.info(f"Consuming from: {self.queue_name}")
            queue_names = ', '.join([self.postprocessing_queues['colors'], self.postprocessing_queues['face'], self.postprocessing_queues['pose']])
            self.logger.info(f"Publishing to postprocessing queues: {queue_names}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False
    
    # Removed publish_to_downstream - using new dispatch_bbox_postprocessing instead
    
    def dispatch_bbox_postprocessing(self, image_id, image_filename, merged_data, image_data):
        """Dispatch bbox postprocessing jobs for each merged box"""
        try:
            
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
                    cropped_image_data = self.crop_bbox_from_data(image_data, merged_bbox)
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
                    # Propagate trace_id if present on harmony input
                    if hasattr(self, 'current_trace_id') and self.current_trace_id:
                        base_message['trace_id'] = self.current_trace_id
                    
                    # Dispatch colors only for boxes >= 24x24 pixels (filter out tiny icons)
                    width = merged_bbox.get('width', 0)
                    height = merged_bbox.get('height', 0)
                    if width >= 24 and height >= 24:
                        self.publish_bbox_message(self.postprocessing_queues['colors'], base_message)
                        self.logger.debug(f"Dispatched colors for {width}x{height} box")
                    else:
                        self.logger.debug(f"Skipped colors for {width}x{height} box (too small)")
                    
                    # Dispatch face/pose for person boxes
                    if self.is_person_box(instance):
                        self.publish_bbox_message(self.postprocessing_queues['face'], base_message)
                        self.publish_bbox_message(self.postprocessing_queues['pose'], base_message)
                    
        except Exception as e:
            self.logger.error(f"Error in dispatch_bbox_postprocessing: {e}")
            raise
    
    def crop_bbox_from_data(self, image_data, bbox):
        """Crop bbox region from image data and return as base64 encoded bytes"""
        try:
            # Decode base64 image data
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes))
            
            with img:
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
            self.logger.error(f"Failed to crop bbox from image data: {e}")
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
            # Check if connection is healthy
            if not self.queue_channel or self.queue_connection.is_closed:
                self.logger.warning("RabbitMQ queue connection lost, reconnecting...")
                self.connect_to_queue()
                # Update BaseWorker variables after reconnection
                self.connection = self.queue_connection
                self.channel = self.queue_channel
            
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
            # Capture trace_id for propagation to postprocessing
            self.current_trace_id = message.get('trace_id')
            self.logger.debug(f"Processing queue message: {message}")
            
            # Extract image info
            image_id = message['image_id']
            image_filename = message.get('image_filename', f'image_{image_id}')
            image_data = message.get('image_data')  # Base64 encoded image data
            
            # Process the harmony merge for this image
            result = self.update_merged_boxes_for_image(image_id, image_filename, image_data)
            
            if result['success']:
                
                # Only send spatial enrichment message if merged boxes were actually created
                if result['merged_boxes_created']:
                    completion_message = {
                        'image_id': image_id,
                        'image_filename': image_filename,
                        'service': 'harmony',
                        'worker_id': self.worker_id,
                        'processed_at': datetime.now().isoformat()
                    }
                    # Old downstream publishing removed - using new postprocessing dispatch
                    self.logger.info(f"Successfully processed harmony for {image_filename}, postprocessing dispatched")
                else:
                    self.logger.info(f"Successfully processed harmony for {image_filename}, no merged boxes to enrich")
                
                # Trigger consensus after harmony completes
                self.trigger_consensus(image_id, message)
                
                # Acknowledge the message
                ch.basic_ack(delivery_tag=method.delivery_tag)
                
            else:
                # Reject and requeue for retry
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                self.logger.error(f"Failed to process harmony for {image_filename}, requeuing")
                
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
            
            cursor.execute(query, (image_id, tuple(self.clean_bbox_services)))
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
            self.logger.debug("No bbox_results provided")
            return None
        
        # Convert database results to service format
        all_detections = []
        source_result_ids = []
        
        for service, data, result_id in bbox_results:
            source_result_ids.append(result_id)
            
            if not isinstance(data, dict) or 'predictions' not in data:
                self.logger.debug(f"Skipping {service} - invalid data format")
                continue
            
            predictions = data['predictions']
            if not predictions:
                self.logger.debug(f"Skipping {service} - empty predictions array")
                continue
            for prediction in predictions:
                if not prediction.get('bbox'):
                    self.logger.debug(f"Skipping {service} prediction - no bbox")
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
                self.logger.debug(f"Added detection: {service} {detection['emoji']} {detection['label']}")
        
        if not all_detections:
            self.logger.info(f"No valid detections found after processing {len(bbox_results)} bbox results")
            return None
        
        self.logger.debug(f"Processing {len(all_detections)} detections from {len(bbox_results)} services")
        
        # Apply BoundingBoxService harmonization logic
        grouped_objects = self.group_by_label_with_cross_service_clustering(all_detections)
        
        # Count total instances for logging
        total_instances = sum(len(group_data.get('instances', [])) for group_data in grouped_objects.values())
        self.logger.info(f"Grouped {len(all_detections)} detections into {len(grouped_objects)} object groups ({total_instances} instances, confidence filtering moved to consensus)")
        
        # Package harmonized results  
        if not grouped_objects:
            self.logger.info("No valid object groups created - all detections filtered out")
            return None
            
        harmonized_data = {
            'all_detections': all_detections,
            'grouped_objects': grouped_objects,
            'source_services': list(set(r[0] for r in bbox_results)),
            'total_detections': len(all_detections),
            'harmonization_algorithm': 'greedy_many_to_one_v2',
            'source_result_ids': source_result_ids
        }
        
        return harmonized_data
    
    def group_by_label_with_cross_service_clustering(self, detections):
        """Greedy many-to-one detection assignment for multi-service consensus"""
        if not detections:
            self.logger.info("Hungarian: No detections provided")
            return {}
        
        try:
            # Discover potential objects through initial spatial clustering
            potential_objects = self.discover_potential_objects(detections)

            if not potential_objects:
                self.logger.warning(f"Greedy: No potential objects discovered from {len(detections)} detections")
                return {}

            # Build assignments using greedy many-to-one approach
            assignments = self.solve_greedy_assignment(detections, potential_objects)
            
            if not assignments:
                self.logger.warning(f"Greedy: No valid assignments from {len(potential_objects)} objects and {len(detections)} detections")
                return {}
            
            # Build final groups from assignments
            result = self.build_groups_from_assignments(detections, potential_objects, assignments)
            self.logger.info(f"Greedy: Successfully processed {len(detections)} detections → {len(result)} groups")
            return result
            
        except Exception as e:
            self.logger.error(f"Greedy assignment algorithm failed: {e}")
            # Fall back to empty result rather than crashing
            return {}
    
    def discover_potential_objects(self, detections):
        """Discover potential objects through coarse spatial clustering"""
        # Group detections into spatial clusters using generous overlap threshold
        clusters = []
        used = set()
        
        self.logger.info(f"DEBUG: Starting clustering with {len(detections)} detections")
        for i, det in enumerate(detections):
            self.logger.info(f"DEBUG: Detection {i}: {det['service']} -> {det['emoji']} at {det['bbox']}")
        
        for i, detection in enumerate(detections):
            if i in used:
                continue
                
            cluster = [detection]
            used.add(i)
            
            # Find all detections that overlap with this one
            for j, other_detection in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue

                # Check for containment merge rules first (cross-emoji merging)
                merge_info = self.check_containment_merge_rules(detection, other_detection)
                if merge_info and merge_info['should_merge']:
                    # Containment-based merge: always use container emoji and bbox
                    container_det = merge_info['container_detection']
                    contained_det = merge_info['contained_detection']
                    rule = merge_info['rule']
                    ratio = merge_info['containment_ratio']

                    self.logger.info(f"DEBUG: Containment merge triggered - {contained_det['emoji']} (containment={ratio:.2f}) → {container_det['emoji']} with confidence boost per rule: {rule['description']}")

                    # For containment merges, we want to preserve the container and absorb the contained
                    # Create a new merged detection with container properties but combined metadata
                    # Strengthen the container's vote by boosting confidence (simulates additional vote)
                    base_confidence = max(container_det['confidence'], contained_det['confidence'])
                    boosted_confidence = min(1.0, base_confidence + 0.1)  # Add 0.1 boost, cap at 1.0

                    merged_detection = {
                        'service': container_det['service'],  # Use container's service
                        'label': container_det['label'],      # Use container's label
                        'emoji': container_det['emoji'],      # Use container's emoji
                        'bbox': container_det['bbox'],        # Use container's bbox
                        'confidence': boosted_confidence,     # Boosted confidence to strengthen vote
                        'type': container_det['type'],
                        'merged_from': [detection, other_detection],  # Track what was merged
                        'merge_reason': f"containment_rule_{rule['contained_emoji']}_into_{rule['container_emoji']}",
                        'confidence_boost': 0.1,  # Track the boost applied
                        'original_confidence': base_confidence
                    }

                    # Replace current cluster with merged detection
                    cluster = [merged_detection]
                    used.add(j)
                    self.logger.info(f"DEBUG: Added to containment-merge cluster")
                    continue

                # Only cluster detections with the same emoji (same object type)
                emoji1 = normalize_person_emoji(detection['emoji'])
                emoji2 = normalize_person_emoji(other_detection['emoji'])

                if emoji1 == emoji2:
                    # Same object type - check if detections are near-identical
                    overlap = self.calculate_overlap_ratio(detection['bbox'], other_detection['bbox'])

                    # Calculate area similarity ratio (for logging)
                    area1 = detection['bbox']['width'] * detection['bbox']['height']
                    area2 = other_detection['bbox']['width'] * other_detection['bbox']['height']
                    area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0

                    # Check if one box contains the other (for different service scales)
                    containment_ratio = self.calculate_containment_ratio(detection['bbox'], other_detection['bbox'])

                    # Primary clustering: use multi-criteria near-identical detection
                    should_cluster = False

                    if self.are_detections_near_identical(detection['bbox'], other_detection['bbox']):
                        # Multi-criteria clustering (IoU + center distance + size similarity)
                        should_cluster = True
                        self.logger.info(f"DEBUG: Clustering via multi-criteria (IoU={overlap:.3f}, area_ratio={area_ratio:.3f})")
                    elif containment_ratio > 0.7:  # One box is 70% contained in the other
                        # Fallback: containment-based clustering (different scales of same object)
                        should_cluster = True
                        self.logger.info(f"DEBUG: Clustering via containment ({containment_ratio:.3f})")
                    else:
                        # For small objects, use distance-based clustering as last resort
                        max_area = max(area1, area2)
                        is_small_object = max_area < SMALL_OBJECT_AREA  # COCO small object definition

                        if is_small_object:
                            cx1 = detection['bbox']['x'] + detection['bbox']['width'] / 2
                            cy1 = detection['bbox']['y'] + detection['bbox']['height'] / 2
                            cx2 = other_detection['bbox']['x'] + other_detection['bbox']['width'] / 2
                            cy2 = other_detection['bbox']['y'] + other_detection['bbox']['height'] / 2
                            distance = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5

                            if distance < SMALL_OBJECT_DISTANCE:
                                should_cluster = True
                                self.logger.info(f"DEBUG: Clustering small objects via distance ({distance:.1f}px)")

                    self.logger.info(f"DEBUG: Same emoji {emoji1}, overlap = {overlap:.3f}, area_ratio = {area_ratio:.3f}, should_cluster = {should_cluster}")
                    
                    if should_cluster:
                        cluster.append(other_detection)
                        used.add(j)
                        self.logger.info(f"DEBUG: Added to same-emoji cluster")
                    else:
                        self.logger.info(f"DEBUG: Not clustering - insufficient criteria")
                else:
                    self.logger.info(f"DEBUG: Different emojis {emoji1} vs {emoji2} - not clustering")
            
            if cluster:
                clusters.append(cluster)
                self.logger.info(f"DEBUG: Created cluster with {len(cluster)} detections: {[d['emoji'] for d in cluster]}")
        
        # Convert clusters to potential objects
        potential_objects = []
        for i, cluster in enumerate(clusters):
            # Calculate cluster centroid and representative properties
            potential_object = {
                'id': i,
                'representative_bbox': self.calculate_cluster_bbox(cluster),
                'candidate_emojis': list(set(normalize_person_emoji(d['emoji']) for d in cluster)),
                'cluster_detections': cluster
            }
            potential_objects.append(potential_object)
            self.logger.info(f"DEBUG: Potential object {i}: emojis={potential_object['candidate_emojis']}, services={[d['service'] for d in cluster]}")
        
        self.logger.debug(f"Discovered {len(potential_objects)} potential objects from {len(detections)} detections")
        return potential_objects
    
    def calculate_cluster_bbox(self, cluster):
        """Calculate bounding box that encompasses all detections in cluster"""
        if not cluster:
            return {}
        
        if len(cluster) == 1:
            return cluster[0]['bbox']
            
        boxes = [d['bbox'] for d in cluster]
        x1 = min(b['x'] for b in boxes)
        y1 = min(b['y'] for b in boxes)
        x2 = max(b['x'] + b['width'] for b in boxes)
        y2 = max(b['y'] + b['height'] for b in boxes)
        
        return {
            'x': x1,
            'y': y1,
            'width': x2 - x1,
            'height': y2 - y1
        }
    
    def select_highest_confidence_bbox(self, cluster):
        """Select the bounding box with highest confidence from cluster"""
        if not cluster:
            return {}
        
        if len(cluster) == 1:
            return cluster[0]['bbox']
            
        # Find detection with highest confidence
        best_detection = max(cluster, key=lambda d: d['confidence'])
        return best_detection['bbox']
    
    def solve_greedy_assignment(self, detections, potential_objects):
        """Solve many-to-one assignment using greedy approach (replaces Hungarian)"""
        assignments = []
        
        # For each detection, assign it to the best matching object
        for det_idx, detection in enumerate(detections):
            best_cost = float('inf')
            best_obj_idx = None
            
            # Find the object with lowest cost for this detection
            for obj_idx, obj in enumerate(potential_objects):
                cost = self.calculate_assignment_cost(detection, obj)
                
                # Accept assignment if cost is reasonable (good spatial overlap + emoji match)
                if cost < 1.0:  # Spatial cost < 1.0 means some overlap, semantic cost adds 0.5 max
                    if cost < best_cost:
                        best_cost = cost
                        best_obj_idx = obj_idx
            
            # Assign detection to best object if found
            if best_obj_idx is not None:
                assignments.append((det_idx, best_obj_idx, best_cost))
                self.logger.debug(f"Assigned detection {det_idx} ({detection['service']}:{detection['emoji']}) to object {best_obj_idx} (cost: {best_cost:.3f})")
        
        self.logger.debug(f"Greedy assignment: {len(assignments)} assignments from {len(detections)} detections")
        return assignments
    
    def calculate_assignment_cost(self, detection, potential_object):
        """Calculate cost of assigning detection to potential object"""
        # Strict emoji matching - reject cross-emoji assignments completely
        normalized_detection_emoji = normalize_person_emoji(detection['emoji'])
        if normalized_detection_emoji not in potential_object['candidate_emojis']:
            return float('inf')  # Reject cross-emoji assignments completely

        # Spatial cost (only for same-emoji assignments)
        spatial_cost = 1.0 - self.calculate_overlap_ratio(detection['bbox'], potential_object['representative_bbox'])

        return spatial_cost
    
    def build_groups_from_assignments(self, detections, potential_objects, assignments):
        """Build final groups from Hungarian algorithm assignments"""
        groups = {}
        
        # Group assignments by object
        object_assignments = {}
        for det_idx, obj_idx, cost in assignments:
            if obj_idx not in object_assignments:
                object_assignments[obj_idx] = []
            object_assignments[obj_idx].append((det_idx, cost))
        
        # Build groups
        for obj_idx, det_assignments in object_assignments.items():
            potential_object = potential_objects[obj_idx]
            assigned_detections = [detections[det_idx] for det_idx, _ in det_assignments]
            
            # Choose primary emoji (most frequent in assignments)
            emoji_counts = {}
            for detection in assigned_detections:
                emoji = normalize_person_emoji(detection['emoji'])
                emoji_counts[emoji] = emoji_counts.get(emoji, 0) + 1
            
            primary_emoji = max(emoji_counts.keys(), key=emoji_counts.get)
            
            # Create instance using configured bbox method
            instance = self.create_instance_from_assignments(
                assigned_detections, primary_emoji, obj_idx, 
                use_highest_confidence=self.use_highest_confidence_bbox
            )
            
            # Confidence filtering removed - let consensus handle quality filtering based on votes
            # if instance['max_confidence'] < MINIMUM_CONFIDENCE:
            #     self.logger.debug(f"Filtered out instance {instance['cluster_id']} with max_confidence {instance['max_confidence']} < {MINIMUM_CONFIDENCE}")
            #     continue
            
            # Create group
            if primary_emoji not in groups:
                groups[primary_emoji] = {
                    'label': assigned_detections[0]['label'],
                    'emoji': primary_emoji,
                    'type': assigned_detections[0]['type'],
                    'detections': [],
                    'instances': []
                }
            
            # Add detections to group
            groups[primary_emoji]['detections'].extend(assigned_detections)
            
            # Add instance to group
            groups[primary_emoji]['instances'].append(instance)
        
        return groups
    
    def calculate_confidence_weighted_bbox(self, detections):
        """Average bbox coordinates weighted by confidence scores"""
        if not detections:
            return {}
        if len(detections) == 1:
            return detections[0]['bbox'].copy()

        total_weight = sum(d['confidence'] for d in detections)
        if total_weight == 0:
            return detections[0]['bbox'].copy()

        # Weighted average of centers and sizes
        weighted_cx = sum(d['confidence'] * (d['bbox']['x'] + d['bbox']['width']/2) for d in detections) / total_weight
        weighted_cy = sum(d['confidence'] * (d['bbox']['y'] + d['bbox']['height']/2) for d in detections) / total_weight
        weighted_w = sum(d['confidence'] * d['bbox']['width'] for d in detections) / total_weight
        weighted_h = sum(d['confidence'] * d['bbox']['height'] for d in detections) / total_weight

        return {
            'x': int(weighted_cx - weighted_w/2),
            'y': int(weighted_cy - weighted_h/2),
            'width': int(weighted_w),
            'height': int(weighted_h)
        }

    def create_instance_from_assignments(self, assigned_detections, primary_emoji, object_id, use_highest_confidence=False):
        """Create instance from assigned detections

        Args:
            assigned_detections: List of detections assigned to this instance
            primary_emoji: Primary emoji for this instance
            object_id: Object ID for cluster naming
            use_highest_confidence: If True, use highest confidence bbox instead of averaging
        """
        # Expand merged detections first
        expanded_detections = []
        for detection in assigned_detections:
            if 'merged_from' in detection:
                # This is a merged detection from containment rules
                expanded_detections.extend(detection['merged_from'])
            else:
                expanded_detections.append(detection)

        # Clean assignments (remove same-service duplicates)
        cleaned_detections = self.clean_cluster(expanded_detections)

        # Calculate merged bounding box using selected method
        if use_highest_confidence:
            merged_bbox = self.select_highest_confidence_bbox(cleaned_detections)
            bbox_method = 'highest_confidence'
        else:
            # Use confidence-weighted averaging instead of union encompassing for better precision
            merged_bbox = self.calculate_confidence_weighted_bbox(cleaned_detections)
            bbox_method = 'confidence_weighted_average'
        
        # Calculate metrics
        services = list(set(d['service'] for d in cleaned_detections))
        confidences = [d['confidence'] for d in cleaned_detections]
        avg_confidence = sum(confidences) / len(confidences)
        max_confidence = max(confidences)  # Maximum confidence for filtering

        instance = {
            'cluster_id': f"{primary_emoji}_{object_id}",
            'emoji': primary_emoji,
            'labels': list(set(d['label'] for d in cleaned_detections)),
            'merged_bbox': merged_bbox,
            'detection_count': len(cleaned_detections),
            'avg_confidence': round(avg_confidence, 3),
            'max_confidence': round(max_confidence, 3),
            'contributing_services': services,
            'detections': [{'service': d['service'], 'confidence': d['confidence']}
                         for d in cleaned_detections],
            'assignment_method': 'greedy_assignment',
            'bbox_merge_method': bbox_method
        }

        # Add containment merge information if applicable
        for detection in assigned_detections:
            if 'merge_reason' in detection and detection['merge_reason'].startswith('containment_rule'):
                instance['containment_merge_applied'] = True
                instance['merge_reason'] = detection['merge_reason']
                break
        
        return instance
    
    def clean_cluster(self, cluster):
        """Remove same-service duplicates"""
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
        
        return cleaned
    
    def should_merge_boxes(self, box1, box2):
        """Hybrid approach: IoU for similar sizes, containment for different scales"""
        # Calculate basic metrics
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height']
        
        # Calculate intersection
        x1 = max(box1['x'], box2['x'])
        y1 = max(box1['y'], box2['y'])
        x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
        y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])
        
        if x1 >= x2 or y1 >= y2:
            return False  # No intersection
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Containment check: if one box is mostly contained within the other
        containment_threshold = 0.8  # 80% containment
        smaller_area = min(area1, area2)
        containment_ratio = intersection / smaller_area
        
        if containment_ratio >= containment_threshold:
            return True  # One box is mostly contained in the other
        
        # Fall back to standard IoU for similar-sized boxes
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0
        return iou > 0.3  # 30% IoU threshold
    
    def calculate_overlap_ratio(self, box1, box2):
        """Calculate IoU overlap ratio (kept for backward compatibility)"""
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
    
    def calculate_containment_ratio(self, box1, box2):
        """Calculate how much one box is contained within the other"""
        x1 = max(box1['x'], box2['x'])
        y1 = max(box1['y'], box2['y'])
        x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
        y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])

        if x1 >= x2 or y1 >= y2:
            return 0  # No intersection

        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height']

        # Return the ratio of intersection to the smaller box (max containment)
        smaller_area = min(area1, area2)
        return intersection / smaller_area if smaller_area > 0 else 0

    def check_containment_merge_rules(self, detection1, detection2):
        """Check if two detections should be merged based on containment rules

        Returns:
            dict: Merge info if should merge, None otherwise
            Format: {
                'should_merge': True,
                'container_detection': detection_obj,
                'contained_detection': detection_obj,
                'rule': rule_obj,
                'containment_ratio': float
            }
        """
        emoji1 = normalize_emoji(detection1['emoji'])
        emoji2 = normalize_emoji(detection2['emoji'])

        for rule in CONTAINMENT_MERGE_RULES:
            contained_emoji = normalize_emoji(rule['contained_emoji'])
            container_emoji = normalize_emoji(rule['container_emoji'])
            threshold = rule['containment_threshold']

            # Check if detection1 is contained in detection2
            if emoji1 == contained_emoji and emoji2 == container_emoji:
                containment_ratio = self.calculate_specific_containment_ratio(
                    detection1['bbox'], detection2['bbox']
                )
                if containment_ratio >= threshold:
                    return {
                        'should_merge': True,
                        'container_detection': detection2,
                        'contained_detection': detection1,
                        'rule': rule,
                        'containment_ratio': containment_ratio
                    }

            # Check if detection2 is contained in detection1
            elif emoji2 == contained_emoji and emoji1 == container_emoji:
                containment_ratio = self.calculate_specific_containment_ratio(
                    detection2['bbox'], detection1['bbox']
                )
                if containment_ratio >= threshold:
                    return {
                        'should_merge': True,
                        'container_detection': detection1,
                        'contained_detection': detection2,
                        'rule': rule,
                        'containment_ratio': containment_ratio
                    }

        return None

    def calculate_specific_containment_ratio(self, contained_box, container_box):
        """Calculate how much the contained_box is within the container_box"""
        x1 = max(contained_box['x'], container_box['x'])
        y1 = max(contained_box['y'], container_box['y'])
        x2 = min(contained_box['x'] + contained_box['width'], container_box['x'] + container_box['width'])
        y2 = min(contained_box['y'] + contained_box['height'], container_box['y'] + container_box['height'])

        if x1 >= x2 or y1 >= y2:
            return 0  # No intersection

        intersection = (x2 - x1) * (y2 - y1)
        contained_area = contained_box['width'] * contained_box['height']

        return intersection / contained_area if contained_area > 0 else 0

    def are_detections_near_identical(self, bbox1, bbox2, iou_threshold=0.8, center_distance_threshold=15, size_similarity_threshold=0.9):
        """Check if two detections are near-identical using multiple criteria (same as consensus logic)"""
        # 1. IoU check
        iou = self.calculate_overlap_ratio(bbox1, bbox2)
        if iou < iou_threshold:
            return False

        # 2. Center distance check
        cx1 = bbox1['x'] + bbox1['width'] / 2
        cy1 = bbox1['y'] + bbox1['height'] / 2
        cx2 = bbox2['x'] + bbox2['width'] / 2
        cy2 = bbox2['y'] + bbox2['height'] / 2
        center_distance = ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5

        if center_distance > center_distance_threshold:
            return False

        # 3. Size similarity check
        w1, h1 = bbox1['width'], bbox1['height']
        w2, h2 = bbox2['width'], bbox2['height']
        width_similarity = min(w1, w2) / max(w1, w2) if max(w1, w2) > 0 else 0
        height_similarity = min(h1, h2) / max(h1, h2) if max(h1, h2) > 0 else 0
        size_similarity = min(width_similarity, height_similarity)

        if size_similarity < size_similarity_threshold:
            return False

        return True

    def add_clip_fallback_boxes(self, image_id, image_filename, existing_merged_data):
        """Add CLIP bounding boxes as fallback for emojis with no spatial evidence"""
        try:
            # Get CLIP results for this image
            clip_results = self.get_clip_results_for_image(image_id)
            if not clip_results:
                return 0

            # Get existing emojis that already have spatial evidence
            existing_emojis = set()
            if existing_merged_data:
                self.logger.debug(f"CLIP fallback: existing_merged_data keys: {list(existing_merged_data.keys())}")
                grouped_objects = existing_merged_data.get('grouped_objects', {})
                self.logger.debug(f"CLIP fallback: grouped_objects keys: {list(grouped_objects.keys())}")
                for group_key, group_data in grouped_objects.items():
                    normalized_key = normalize_person_emoji(group_key)
                    existing_emojis.add(normalized_key)
                    self.logger.debug(f"CLIP fallback: added existing emoji {group_key} -> {normalized_key}")

            self.logger.debug(f"CLIP fallback: existing_emojis = {existing_emojis}")

            fallback_count = 0
            cursor = self.db_conn.cursor()

            # Process CLIP predictions for fallback opportunities
            for service, data, result_id in clip_results:
                if not isinstance(data, dict) or 'predictions' not in data:
                    continue

                predictions = data.get('predictions', [])
                for prediction in predictions:
                    if not prediction.get('bbox') or not prediction.get('emoji'):
                        continue

                    original_emoji = prediction['emoji']
                    emoji = normalize_person_emoji(original_emoji)

                    self.logger.debug(f"CLIP fallback: checking {original_emoji} -> {emoji}, existing: {emoji in existing_emojis}")

                    # Only use as fallback if this emoji has no existing spatial evidence
                    if emoji not in existing_emojis:
                        # Create a merged box entry for this CLIP detection
                        merged_box_data = {
                            'cluster_id': f"{emoji}_clip_fallback",
                            'emoji': emoji,
                            'labels': [prediction.get('label', '')],
                            'merged_bbox': prediction['bbox'],
                            'detection_count': 1,
                            'avg_confidence': prediction.get('confidence', 0.5),
                            'max_confidence': prediction.get('confidence', 0.5),
                            'contributing_services': ['clip'],
                            'detections': [{'service': 'clip', 'confidence': prediction.get('confidence', 0.5)}],
                            'harmonization_algorithm': 'clip_fallback',
                            'source_result_ids': [result_id],
                            'fallback_source': 'clip_spatial_fallback'
                        }

                        cursor.execute("""
                            INSERT INTO merged_boxes (image_id, source_result_ids, merged_data, worker_id, status)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (
                            image_id,
                            [result_id],
                            json.dumps({k: v for k, v in merged_box_data.items() if k != 'source_result_ids'}),
                            self.worker_id,
                            'success'
                        ))

                        fallback_count += 1
                        existing_emojis.add(emoji)  # Prevent duplicates
                        self.logger.debug(f"Added CLIP fallback box for {emoji} with confidence {prediction.get('confidence', 0.5):.3f}")

            cursor.close()
            return fallback_count

        except Exception as e:
            self.logger.error(f"Error adding CLIP fallback boxes for image {image_id}: {e}")
            return 0

    def get_clip_results_for_image(self, image_id):
        """Get CLIP results for an image"""
        try:
            cursor = self.read_db_conn.cursor()

            query = """
                SELECT service, data, result_id
                FROM results
                WHERE image_id = %s
                AND service = 'clip'
                AND status = 'success'
                ORDER BY service
            """

            cursor.execute(query, (image_id,))
            results = cursor.fetchall()
            cursor.close()

            return results

        except Exception as e:
            self.logger.error(f"Error getting CLIP results for image {image_id}: {e}")
            return []

    def update_merged_boxes_for_image(self, image_id, image_filename, image_data=None):
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
                    self.logger.info(f"No bbox results found for image {image_id}, skipping")
                    return {'success': True, 'merged_boxes_created': False}  # Not an error - just no data to process
                else:
                    self.logger.info(f"Found {len(bbox_results)} bbox results for image {image_id}")
                
                # Harmonize bounding boxes
                start_time = time.time()
                merged_data = self.harmonize_bounding_boxes(bbox_results)
                processing_time = time.time() - start_time
                
                if not merged_data:
                    self.logger.info(f"No bounding boxes to harmonize for image {image_id} - all detection services returned empty predictions")
                    return {'success': True, 'merged_boxes_created': False}
                
                # Safe atomic DELETE + INSERT with foreign key handling
                cursor = self.db_conn.cursor()
                
                # Progressive harmonization: always reharmonize with latest bbox results
                
                # Step 1: Clear postprocessing references to avoid FK constraint violation
                cursor.execute("""
                    DELETE FROM postprocessing 
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
                            'max_confidence': instance.get('max_confidence', 0.0),
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
                            merged_box_data['source_result_ids'],
                            json.dumps({k: v for k, v in merged_box_data.items() if k != 'source_result_ids'}),
                            self.worker_id,
                            'success'
                        ))
                        merged_box_count += 1
                
                cursor.close()
                
                services = merged_data['source_services']
                detection_count = merged_data['total_detections']
                
                if deleted_count > 0:
                    self.logger.info(f"Reharmonized boxes for {image_filename} ({detection_count} detections → {merged_box_count} merged boxes from {services})")
                else:
                    self.logger.info(f"Harmonized boxes for {image_filename} ({detection_count} detections → {merged_box_count} merged boxes from {services})")
                
                # Check for CLIP fallback spatial evidence before finalizing
                clip_fallback_count = self.add_clip_fallback_boxes(image_id, image_filename, merged_data)
                if clip_fallback_count > 0:
                    merged_box_count += clip_fallback_count
                    self.logger.info(f"Added {clip_fallback_count} CLIP fallback bounding boxes for {image_filename}")

                # Dispatch bbox postprocessing jobs for each merged box
                if merged_box_count > 0 and image_data:
                    try:
                        self.dispatch_bbox_postprocessing(image_id, image_filename, merged_data, image_data)
                    except Exception as e:
                        self.logger.error(f"Failed to dispatch postprocessing for {image_filename}: {e}")
                        # Rollback transaction and fail - postprocessing dispatch is critical
                        try:
                            if self.db_conn:
                                self.db_conn.rollback()
                        except:
                            pass
                        return {'success': False, 'merged_boxes_created': False}
                
                # Commit the transaction only after successful postprocessing dispatch
                self.db_conn.commit()
                
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
        self.logger.info(f"Starting harmony worker in QUEUE MODE ({self.worker_id})")
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
        
        self.logger.info("Waiting for harmony messages. Press CTRL+C to exit")
        
        try:
            self.queue_channel.start_consuming()
        except KeyboardInterrupt:
            self.logger.info("Stopping harmony worker...")
            self.queue_channel.stop_consuming()
        finally:
            self.cleanup_connections()
            
        return 0
    
    def cleanup_connections(self):
        """Clean up all connections"""
        
        if self.db_conn:
            self.db_conn.close()
        if self.read_db_conn:
            self.read_db_conn.close()
        if self.queue_connection and not self.queue_connection.is_closed:
            self.queue_connection.close()
        
        self.logger.info("Harmony worker stopped")
    

def main():
    """Main entry point"""
    try:
        worker = HarmonyWorker()
        return worker.run()
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Harmony worker error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
