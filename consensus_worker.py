#!/usr/bin/env python3
"""
Continuous Consensus Worker - Scans for new ML service results and updates consensus
Implements the DELETE+INSERT pattern for atomic consensus updates
"""
import os
import json
import time
import logging
import socket
import psycopg2
import mysql.connector
import pika
from datetime import datetime
from dotenv import load_dotenv

class ConsensusWorker:
    """Continuous consensus/voting worker"""
    
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
        self.worker_id = os.getenv('WORKER_ID', f'consensus_worker_{int(time.time())}')
        
        # Queue configuration
        self.queue_name = os.getenv('CONSENSUS_QUEUE_NAME', 'queue_consensus')
        self.queue_host = self._get_required('QUEUE_HOST')
        self.queue_user = self._get_required('QUEUE_USER')
        self.queue_password = self._get_required('QUEUE_PASSWORD')
        
        # Monitoring configuration  
        self.enable_monitoring = os.getenv('ENABLE_MONITORING', 'false').lower() == 'true'
        if self.enable_monitoring:
            self.monitoring_db_host = self._get_required('MONITORING_DB_HOST')
            self.monitoring_db_user = self._get_required('MONITORING_DB_USER') 
            self.monitoring_db_password = self._get_required('MONITORING_DB_PASSWORD')
            self.monitoring_db_name = self._get_required('MONITORING_DB_NAME')
        else:
            self.monitoring_db_host = None
            self.monitoring_db_user = None
            self.monitoring_db_password = None
            self.monitoring_db_name = None
        
        # V3 Voting configuration
        self.service_names = {
            'blip': 'blip',
            'clip': 'clip', 
            'yolov8': 'yolo',
            'colors': 'colors',
            'detectron2': 'detectron2',
            'face': 'face',
            'nsfw2': 'nsfw',
            'ocr': 'ocr',
            'inception_v3': 'inception',
            'rtdetr': 'rtdetr',
            'metadata': 'metadata',
            'ollama': 'llama',
            'pose': 'pose'
        }
        
        # Special emojis that auto-promote
        self.special_emojis = ['🔞', '💬']
        
        # Democratic voting configuration
        self.default_confidence = float(os.getenv('DEFAULT_CONFIDENCE', '0.75'))
        self.low_confidence_threshold = float(os.getenv('LOW_CONFIDENCE_THRESHOLD', '0.4'))
        
        # Logging
        self.setup_logging()
        self.db_conn = None
        self.read_db_conn = None
        
        # Initialize monitoring
        self.setup_monitoring()
        
    def setup_logging(self):
        """Configure logging"""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('consensus_worker')
    
    def setup_monitoring(self):
        """Initialize monitoring connection"""
        self.mysql_conn = None
        self.last_heartbeat = 0
        self.jobs_completed = 0
        self.start_time = time.time()
        self.hostname = socket.gethostname()
        
        if self.enable_monitoring and self.monitoring_db_password:
            try:
                self.mysql_conn = mysql.connector.connect(
                    host=self.monitoring_db_host,
                    user=self.monitoring_db_user,
                    password=self.monitoring_db_password,
                    database=self.monitoring_db_name,
                    autocommit=True
                )
                self.send_heartbeat('starting')
            except Exception as e:
                self.logger.warning(f"Could not connect to monitoring database: {e}")
    
    def send_heartbeat(self, status, error_msg=None):
        """Send heartbeat to monitoring database"""
        if not self.enable_monitoring or not self.mysql_conn:
            return
        
        try:
            cursor = self.mysql_conn.cursor()
            cursor.execute("""
                INSERT INTO worker_heartbeats 
                (worker_id, service_name, node_hostname, status, jobs_completed, error_message, last_job_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                status = VALUES(status), 
                jobs_completed = VALUES(jobs_completed),
                error_message = VALUES(error_message),
                last_job_time = VALUES(last_job_time)
            """, (
                self.worker_id, 
                'consensus_worker',
                self.hostname,
                status,
                self.jobs_completed,
                error_msg,
                datetime.now() if self.jobs_completed > 0 else None
            ))
            cursor.close()
            self.last_heartbeat = time.time()
        except Exception as e:
            self.logger.warning(f"Failed to send heartbeat: {e}")
    
    def maybe_send_heartbeat(self):
        """Send heartbeat if enough time has passed"""
        if time.time() - self.last_heartbeat > 120:  # 2 minutes
            self.send_heartbeat('alive')
    
    def _get_required(self, key):
        """Get required environment variable with no fallback"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value
    
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
            self.db_conn.autocommit = False  # Use transactions
            
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
            
            # Declare the consensus queue
            self.channel.queue_declare(queue=self.queue_name, durable=True)
            
            # Set prefetch count for fair distribution
            self.channel.basic_qos(prefetch_count=1)
            
            self.logger.info(f"Connected to RabbitMQ at {self.queue_host}, queue: {self.queue_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False
    
    
    def get_service_results_for_image(self, image_id):
        """Get all successful service results for an image"""
        try:
            if self.read_db_conn.closed:
                self.logger.error("Read database connection is closed, reconnecting...")
                self.connect_to_database()
            cursor = self.read_db_conn.cursor()
            
            query = """
                SELECT service, data, processing_time, result_created
                FROM results
                WHERE image_id = %s AND status = 'success'
                ORDER BY service
            """
            
            cursor.execute(query, (image_id,))
            raw_results = cursor.fetchall()
            cursor.close()
            
            # Parse JSON data at the source to ensure all downstream code gets proper dicts
            parsed_results = []
            for service, data, processing_time, result_created in raw_results:
                # Ensure data is properly parsed as JSON
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse JSON for service {service} on image {image_id}, skipping")
                        continue
                
                parsed_results.append((service, data, processing_time, result_created))
            
            return parsed_results
            
        except Exception as e:
            self.logger.error(f"Error getting results for image {image_id}: {e}")
            return []
    
    def calculate_consensus(self, service_results, image_id):
        """Calculate consensus using full V3 voting algorithm (ported from V3VotingService.js)"""
        if not service_results:
            return None
        
        # Convert database results to V3VotingService format
        service_results_dict = {}
        for service, data, proc_time, created in service_results:
            service_results_dict[service] = {
                'success': True,
                'predictions': data.get('predictions', []) if isinstance(data, dict) else []
            }
        
        # Get harmonized bounding box data from merged_boxes table
        bounding_box_data = self.get_bounding_box_data_for_image(image_id)
        
        # Implement V3 voting algorithm
        votes_result = self.process_votes(service_results_dict, bounding_box_data)
        
        # Package results for storage
        consensus = {
            'services_count': len(service_results),
            'services_list': [row[0] for row in service_results],
            'total_processing_time': sum(row[2] or 0 for row in service_results),
            'latest_result_time': max(row[3] for row in service_results).isoformat(),
            'consensus_algorithm': 'v3_voting_full_port',
            'votes': votes_result['votes'],
            'special': votes_result['special'],
            'debug': votes_result['debug']
        }
        
        return consensus
    
    def get_bounding_box_data_for_image(self, image_id):
        """Get harmonized bounding box data from merged_boxes table"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT merged_id, merged_data, status
                FROM merged_boxes 
                WHERE image_id = %s AND status = 'success'
                ORDER BY created DESC
            """, (image_id,))
            
            merged_boxes = []
            for row in cursor.fetchall():
                merged_data = row[1]  # JSONB data
                
                # Only include multi-service merged boxes (detection_count >= 2)
                if merged_data.get('detection_count', 0) >= 2:
                    merged_boxes.append({
                        'merged_id': row[0],
                        'emoji': merged_data.get('emoji'),
                        'detection_count': merged_data.get('detection_count', 0),
                        'avg_confidence': merged_data.get('avg_confidence', 0),
                        'contributing_services': merged_data.get('contributing_services', []),
                        'merged_bbox': merged_data.get('merged_bbox', {}),
                        'labels': merged_data.get('labels', [])
                    })
            
            cursor.close()
            return merged_boxes if merged_boxes else None
            
        except Exception as e:
            self.logger.error(f"Error fetching merged boxes: {e}")
            return None
    
    def process_votes(self, service_results, bounding_box_data=None):
        """Main V3 voting algorithm entry point (ported from V3VotingService.js)"""
        # Step 1: Extract all detections from all services
        all_detections = self.extract_all_detections(service_results, bounding_box_data)
        
        # Step 2: Group detections by emoji (democratic voting)
        emoji_groups = self.group_detections_by_emoji(all_detections)
        
        # Debug: Log emoji groups
        for emoji, detections in emoji_groups.items():
            services = [d['service'] for d in detections]
            self.logger.debug(f"Emoji group {emoji}: {len(detections)} votes from {services}")
        
        # Step 3: Analyze evidence for each emoji
        emoji_analysis = self.analyze_emoji_evidence(emoji_groups, service_results, bounding_box_data)
        
        # Step 4: Calculate evidence weights and final ranking
        ranked_consensus = self.calculate_final_ranking(emoji_analysis)
        
        # Step 5: Apply post-processing curation (quality adjustments)
        self.apply_post_processing_curation(ranked_consensus)
        
        return {
            'votes': {
                'consensus': ranked_consensus
            },
            'special': self.extract_special_detections(service_results),
            'debug': {
                'detection_count': len(all_detections),
                'emoji_groups': len(emoji_groups)
            }
        }
    
    def extract_all_detections(self, service_results, bounding_box_data=None):
        """Extract all detections from all services with metadata (ported from V3VotingService.js)"""
        all_detections = []

        for service_name, result in service_results.items():
            if not result.get('success') or not result.get('predictions'):
                continue

            service_display_name = self.service_names.get(service_name, service_name)
            seen_emojis = set()  # Deduplicate within service

            for prediction in result['predictions']:
                # Debug: Check if prediction needs JSON parsing
                if isinstance(prediction, str):
                    try:
                        prediction = json.loads(prediction)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse prediction JSON for service {service_name}, skipping: {prediction[:100]}")
                        continue
                
                # Handle emoji_mappings format (standardized format for all services)
                if prediction.get('emoji_mappings') and isinstance(prediction['emoji_mappings'], list):
                    for mapping in prediction['emoji_mappings']:
                        if mapping.get('emoji') and mapping['emoji'] not in seen_emojis:
                            seen_emojis.add(mapping['emoji'])
                            detection = {
                                'emoji': mapping['emoji'],
                                'service': service_display_name,
                                'evidence_type': self.get_evidence_type(service_name),
                                'confidence': self.default_confidence,
                                'context': {
                                    'word': mapping.get('word', ''),
                                    'source': 'caption_mapping'
                                },
                                'shiny': mapping.get('shiny', False)
                            }
                            all_detections.append(detection)
                            self.logger.debug(f"Added detection: {detection['emoji']} from {service_name} ({detection['evidence_type']}) for word '{mapping.get('word')}'")
                
                # Handle direct emoji format (CLIP, object detection, etc.)
                elif prediction.get('emoji') and prediction.get('type') != 'color_analysis':
                    emoji = prediction['emoji']
                    
                    if emoji and emoji not in seen_emojis:
                        seen_emojis.add(emoji)
                        all_detections.append({
                            'emoji': emoji,
                            'service': service_display_name,
                            'evidence_type': self.get_evidence_type(service_name),
                            'confidence': prediction.get('confidence', self.default_confidence),
                            'context': self.extract_context(prediction, service_name),
                            'shiny': prediction.get('shiny', False)
                        })

        # Extract spatial detections from merged_boxes data (harmonized bbox data)
        if bounding_box_data and isinstance(bounding_box_data, list):
            for merged_box in bounding_box_data:
                if merged_box.get('emoji') and merged_box.get('detection_count', 0) >= 2:
                    all_detections.append({
                        'emoji': merged_box['emoji'],
                        'service': 'merged_boxes',
                        'evidence_type': 'spatial',
                        'confidence': merged_box.get('avg_confidence', 0.75),
                        'context': {
                            'source': 'merged_boxes',
                            'contributing_services': merged_box.get('contributing_services', []),
                            'detection_count': merged_box.get('detection_count', 0)
                        },
                        'shiny': False,
                        'spatial_data': {
                            'detection_count': merged_box.get('detection_count', 0),
                            'avg_confidence': merged_box.get('avg_confidence', 0),
                            'contributing_services': merged_box.get('contributing_services', []),
                            'bbox': merged_box.get('merged_bbox', {}),
                            'labels': merged_box.get('labels', [])
                        }
                    })

        return all_detections
    
    def get_evidence_type(self, service_name):
        """Determine evidence type based on service name (ported from V3VotingService.js)"""
        spatial_services = ['yolov8', 'detectron2', 'rtdetr', 'clip', 'xception']
        semantic_services = ['blip', 'ollama']  # Smart captioning services
        classification_services = ['inception_v3']  # Image classification services
        specialized_services = ['face', 'nsfw2', 'ocr', 'pose']

        if service_name in spatial_services:
            return 'spatial'
        if service_name in semantic_services:
            return 'semantic'
        if service_name in classification_services:
            return 'classification'
        if service_name in specialized_services:
            return 'specialized'
        return 'other'

    def extract_context(self, prediction, service_name):
        """Extract context information from prediction (ported from V3VotingService.js)"""
        context = {}
        
        if service_name == 'face':
            context['pose'] = prediction.get('pose')
        if service_name == 'nsfw2':
            context['nsfw_confidence'] = prediction.get('confidence')
        if service_name == 'ocr':
            context['text_detected'] = prediction.get('has_text', False)
            context['text_content'] = prediction.get('text')
        
        return context

    def group_detections_by_emoji(self, all_detections):
        """Group detections by emoji for democratic voting (ported from V3VotingService.js)"""
        groups = {}
        
        for detection in all_detections:
            emoji = detection['emoji']
            if emoji not in groups:
                groups[emoji] = []
            groups[emoji].append(detection)
        
        return groups

    def analyze_emoji_evidence(self, emoji_groups, service_results, merged_boxes_data=None):
        """Analyze evidence for each emoji group (ported from V3VotingService.js)"""
        analysis = []
        
        for emoji, detections in emoji_groups.items():
            voting_services = list(set(d['service'] for d in detections if d['service'] != 'spatial_clustering'))
            
            evidence_analysis = {
                'emoji': emoji,
                'total_votes': len(voting_services),
                'voting_services': voting_services,
                'detections': detections,
                'evidence': {
                    'spatial': self.analyze_spatial_evidence(detections, merged_boxes_data),
                    'semantic': self.analyze_semantic_evidence(detections),
                    'specialized': self.analyze_specialized_evidence(detections)
                },
                'instances': self.extract_instance_information(detections),
                'shiny': any(d.get('shiny', False) for d in detections)
            }
            
            analysis.append(evidence_analysis)
        
        return analysis

    def analyze_spatial_evidence(self, detections, merged_boxes_data=None):
        """Analyze spatial evidence from merged_boxes table (harmonized bbox data)"""
        if not merged_boxes_data:
            return None
            
        # Find merged boxes that match this emoji
        emoji = detections[0].get('emoji') if detections else None
        if not emoji:
            return None
            
        matching_boxes = [box for box in merged_boxes_data if box.get('emoji') == emoji]
        if not matching_boxes:
            return None
        
        # Use the harmonized spatial data from merged_boxes
        return {
            'service_count': sum(box.get('detection_count', 0) for box in matching_boxes),
            'clusters': [{
                'detection_count': box.get('detection_count', 0),
                'avg_confidence': box.get('avg_confidence', 0),
                'contributing_services': box.get('contributing_services', []),
                'bbox': box.get('merged_bbox', {}),
                'labels': box.get('labels', [])
            } for box in matching_boxes],
            'max_detection_count': max(box.get('detection_count', 0) for box in matching_boxes),
            'avg_confidence': sum(box.get('avg_confidence', 0) for box in matching_boxes) / len(matching_boxes),
            'total_instances': len(matching_boxes)
        }

    def analyze_semantic_evidence(self, detections):
        """Analyze semantic evidence from captioning services (ported from V3VotingService.js)"""
        semantic_detections = [d for d in detections if d.get('evidence_type') == 'semantic']
        if not semantic_detections:
            return None
        
        return {
            'service_count': len(semantic_detections),
            'words': [d['context'].get('word') for d in semantic_detections if d.get('context', {}).get('word')],
            'sources': [d['service'] for d in semantic_detections]
        }

    def analyze_classification_evidence(self, detections):
        """Analyze classification evidence from image classification services (ported from V3VotingService.js)"""
        classification_detections = [d for d in detections if d.get('evidence_type') == 'classification']
        if not classification_detections:
            return None
        
        return {
            'service_count': len(classification_detections),
            'sources': [d['service'] for d in classification_detections]
        }

    def analyze_specialized_evidence(self, detections):
        """Analyze specialized evidence (Face, NSFW, OCR) (ported from V3VotingService.js)"""
        specialized_detections = [d for d in detections if d.get('evidence_type') == 'specialized']
        if not specialized_detections:
            return None
        
        by_type = {}
        for d in specialized_detections:
            service_type = d['service'].lower()
            if service_type not in by_type:
                by_type[service_type] = []
            by_type[service_type].append(d)
        
        return by_type

    def extract_instance_information(self, detections):
        """Extract instance information (ported from V3VotingService.js)"""
        spatial_detections = [d for d in detections if d.get('spatial_data')]
        
        if not spatial_detections:
            return {'count': 1, 'type': 'non_spatial'}
        
        return {
            'count': len(spatial_detections),
            'type': 'spatial'
        }

    def calculate_evidence_weight(self, analysis):
        """Calculate evidence weight using consensus bonus system (ported from V3VotingService.js)"""
        weight = 0
        
        # Base democratic weight: 1 vote per service (pure democracy)
        base_votes = analysis['total_votes']
        
        # Spatial consensus bonus: Agreement on location
        spatial_consensus_bonus = 0
        if analysis['evidence']['spatial']:
            # Consensus = detection_count - 1 (one vote doesn't count as consensus)
            spatial_consensus_bonus = max(0, analysis['evidence']['spatial']['max_detection_count'] - 1)
        
        # Content consensus bonus: Agreement across semantic services
        content_consensus_bonus = 0
        semantic_count = analysis['evidence']['semantic']['service_count'] if analysis['evidence']['semantic'] else 0
        total_content_services = semantic_count
        
        if total_content_services >= 2:
            # Consensus = total_content_services - 1 (one vote doesn't count as consensus)
            content_consensus_bonus = total_content_services - 1
        
        # Total weight = democratic votes + consensus bonuses
        weight = base_votes + spatial_consensus_bonus + content_consensus_bonus
        
        return max(0, weight)  # Don't go negative

    def calculate_final_ranking(self, emoji_analysis):
        """Calculate final ranking with democratic voting + evidence weighting (ported from V3VotingService.js)"""
        # Calculate evidence weights
        for analysis in emoji_analysis:
            analysis['evidence_weight'] = self.calculate_evidence_weight(analysis)
            analysis['final_score'] = analysis['total_votes'] + analysis['evidence_weight']
            analysis['should_include'] = self.should_include_in_results(analysis)
        
        # Filter and sort
        filtered_analysis = [a for a in emoji_analysis if a['should_include']]
        
        # Sort by total votes (primary), then evidence weight (secondary)
        filtered_analysis.sort(key=lambda a: (a['total_votes'], a['evidence_weight']), reverse=True)
        
        # Convert to final result format
        results = []
        for analysis in filtered_analysis:
            result = {
                'emoji': analysis['emoji'],
                'votes': analysis['total_votes'],
                'evidence_weight': round(analysis['evidence_weight'], 2),
                'final_score': round(analysis['final_score'], 2),
                'instances': analysis['instances'],
                'evidence': {
                    'spatial': self.format_spatial_evidence(analysis['evidence']['spatial']) if analysis['evidence']['spatial'] else None,
                    'semantic': analysis['evidence']['semantic'],
                    'specialized': list(analysis['evidence']['specialized'].keys()) if analysis['evidence']['specialized'] else None
                },
                'services': analysis['voting_services']
            }
            
            # Add bounding boxes if available
            if analysis['evidence']['spatial'] and analysis['evidence']['spatial']['clusters']:
                result['bounding_boxes'] = self.format_bounding_boxes(analysis['evidence']['spatial']['clusters'], analysis['emoji'])
            
            # Add validation/correlation if they exist
            if analysis.get('validation'):
                result['validation'] = analysis['validation']
            if analysis.get('correlation'):
                result['correlation'] = analysis['correlation']
            if analysis.get('shiny'):
                result['shiny'] = True
            
            results.append(result)
        
        return results

    def format_spatial_evidence(self, spatial_evidence):
        """Format spatial evidence for output"""
        if not spatial_evidence:
            return None
        
        return {
            'detection_count': spatial_evidence['max_detection_count'],
            'avg_confidence': round(spatial_evidence['avg_confidence'], 3),
            'instance_count': spatial_evidence['total_instances']
        }

    def format_bounding_boxes(self, clusters, emoji):
        """Format bounding box data for output"""
        bounding_boxes = []
        for cluster in clusters:
            bounding_boxes.append({
                'cluster_id': cluster.get('cluster_id'),
                'merged_bbox': cluster.get('bbox', {}),
                'emoji': emoji,
                'label': cluster.get('cluster_id', '').split('_')[0] if cluster.get('cluster_id') else emoji,
                'detection_count': cluster.get('detection_count', 1),
                'avg_confidence': cluster.get('avg_confidence', 0.75),
                'detections': cluster.get('individual_detections', [])
            })
        return bounding_boxes

    def should_include_in_results(self, analysis):
        """Determine if emoji should be included in results (ported from V3VotingService.js)"""
        # Only include if has multiple votes (filter out single-vote emojis)
        return analysis['total_votes'] > 1

    def apply_post_processing_curation(self, ranked_consensus):
        """Apply post-processing curation (ported from V3VotingService.js)"""
        # Build lookup for cross-emoji validation
        emoji_map = {item['emoji']: item for item in ranked_consensus}
        
        for item in ranked_consensus:
            curation_adjustment = 0
            
            # Face validates Person (+1 confidence boost)
            if item['emoji'] == '🧑' and '🙂' in emoji_map:
                curation_adjustment += 1
                if 'validation' not in item:
                    item['validation'] = []
                item['validation'].append('face_confirmed')
            
            # Pose validates Person (+1 confidence boost)  
            has_pose_detection = any(
                other.get('evidence', {}).get('specialized') and 'pose' in other['evidence']['specialized']
                for other in ranked_consensus
            )
            if item['emoji'] == '🧑' and has_pose_detection:
                curation_adjustment += 1
                if 'validation' not in item:
                    item['validation'] = []
                item['validation'].append('pose_confirmed')
            
            # NSFW requires human context (quality filter)
            if item['emoji'] == '🔞':
                if '🧑' in emoji_map:
                    curation_adjustment += 1
                    if 'validation' not in item:
                        item['validation'] = []
                    item['validation'].append('human_context_confirmed')
                else:
                    curation_adjustment -= 1
                    if 'validation' not in item:
                        item['validation'] = []
                    item['validation'].append('suspicious_no_humans')
            
            # Apply curation adjustment
            if curation_adjustment != 0:
                item['evidence_weight'] += curation_adjustment
                item['final_score'] += curation_adjustment
                # Ensure we don't go negative
                item['evidence_weight'] = max(0, item['evidence_weight'])
                item['final_score'] = max(0, item['final_score'])

    def extract_special_detections(self, service_results):
        """Extract special detections (non-competing) (ported from V3VotingService.js)"""
        special = {}
        
        # Text detection from OCR
        if service_results.get('ocr', {}).get('predictions'):
            has_text = any(pred.get('has_text') for pred in service_results['ocr']['predictions'])
            if has_text:
                text_pred = next(pred for pred in service_results['ocr']['predictions'] if pred.get('has_text'))
                special['text'] = {
                    'emoji': '💬',
                    'detected': True,
                    'confidence': text_pred.get('confidence', 1.0),
                    'content': text_pred.get('text')
                }
            else:
                special['text'] = {'detected': False}
        else:
            special['text'] = {'detected': False}
        
        # Face detection from Face service
        if service_results.get('face', {}).get('predictions'):
            face_pred = next((pred for pred in service_results['face']['predictions'] if pred.get('emoji') == '🙂'), None)
            if face_pred:
                special['face'] = {
                    'emoji': '🙂',
                    'detected': True,
                    'confidence': face_pred.get('confidence', 1.0),
                    'pose': face_pred.get('pose')
                }
            else:
                special['face'] = {'detected': False}
        else:
            special['face'] = {'detected': False}
        
        # NSFW detection from NSFW service
        if service_results.get('nsfw2', {}).get('predictions'):
            nsfw_pred = next((pred for pred in service_results['nsfw2']['predictions'] if pred.get('emoji') == '🔞'), None)
            if nsfw_pred:
                special['nsfw'] = {
                    'emoji': '🔞',
                    'detected': True,
                    'confidence': nsfw_pred.get('confidence', 1.0)
                }
            else:
                special['nsfw'] = {'detected': False}
        else:
            special['nsfw'] = {'detected': False}
        
        return special
    
    def update_consensus_for_image(self, image_id, image_filename):
        """Update consensus for a single image using DELETE+INSERT pattern"""
        try:
            # Get all service results
            service_results = self.get_service_results_for_image(image_id)
            if not service_results:
                self.logger.debug(f"No results for image {image_id}, skipping")
                return False
            
            # Calculate consensus
            start_time = time.time()
            consensus_data = self.calculate_consensus(service_results, image_id)
            processing_time = time.time() - start_time
            
            if not consensus_data:
                self.logger.warning(f"Could not calculate consensus for image {image_id}")
                return False
            
            # Ensure healthy database connection before operations
            if not self.ensure_database_connection():
                self.logger.error("Could not establish database connection")
                return False
            
            # Atomic DELETE + INSERT
            cursor = self.db_conn.cursor()
            
            # Delete old consensus
            cursor.execute("DELETE FROM consensus WHERE image_id = %s", (image_id,))
            deleted_count = cursor.rowcount
            
            # Insert new consensus
            cursor.execute("""
                INSERT INTO consensus (image_id, consensus_data, processing_time)
                VALUES (%s, %s, %s)
            """, (image_id, json.dumps(consensus_data), processing_time))
            
            # CRITICAL: Commit the transaction!
            self.db_conn.commit()
            
            cursor.close()
            
            if deleted_count > 0:
                self.logger.debug(f"Updated consensus for {image_filename} ({len(service_results)} services)")
            else:
                self.logger.debug(f"Created consensus for {image_filename} ({len(service_results)} services)")
            
            return True
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error updating consensus for image {image_id}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Rollback transaction on error
            try:
                if self.db_conn:
                    self.db_conn.rollback()
            except:
                pass
            return False
    
    
    def process_queue_message(self, ch, method, properties, body):
        """Process a message from the consensus queue"""
        try:
            # Parse the message
            message_data = json.loads(body.decode('utf-8'))
            image_id = message_data['image_id']
            image_filename = message_data.get('image_filename', f'image_{image_id}')
            
            self.logger.info(f"Processing consensus for: {image_filename}")
            
            # Update consensus for this specific image
            success = self.update_consensus_for_image(image_id, image_filename)
            
            if success:
                self.logger.info(f"Completed consensus for {image_filename}")
            else:
                self.logger.warning(f"Failed to update consensus for {image_filename}")
            
            # Acknowledge the message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            self.logger.error(f"Error processing consensus queue message: {e}")
            # Reject and requeue for retry
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    
    def run(self):
        """Main entry point - pure queue-based processing"""
        if not self.connect_to_database():
            return 1
        
        if not self.connect_to_rabbitmq():
            return 1
        
        self.logger.info(f"Starting consensus worker ({self.worker_id})")
        self.logger.info(f"Listening on queue: {self.queue_name}")
        
        # Setup message consumer
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.process_queue_message
        )
        
        self.logger.info("Waiting for consensus messages. Press CTRL+C to exit")
        
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.logger.info("Stopping consensus worker...")
            self.channel.stop_consuming()
        finally:
            self.send_heartbeat('stopping')
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            if self.db_conn:
                self.db_conn.close()
            if self.read_db_conn:
                self.read_db_conn.close()
            if self.mysql_conn:
                self.mysql_conn.close()
            self.logger.info("Consensus worker stopped")
        
        return 0

def main():
    """Main entry point"""
    try:
        worker = ConsensusWorker()
        return worker.run()
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Consensus worker error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())