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
import pika
import re
from datetime import datetime
from base_worker import BaseWorker

def normalize_emoji(emoji):
    """Remove variation selectors and other invisible modifiers from emoji"""
    if not emoji:
        return emoji
    # Remove variation selectors (U+FE00-U+FE0F), Mongolian selectors (U+180B-U+180D),
    # and zero-width joiner (U+200D) for consistent grouping
    return re.sub(r'[\uFE00-\uFE0F\u180B-\u180D\u200D]', '', emoji)

class ConsensusWorker(BaseWorker):
    """Continuous consensus/voting worker"""
    
    def __init__(self):
        # Initialize with consensus service type
        super().__init__('system.consensus')
        
        
        
        # V3 Voting configuration
        self.special_emojis = ['🔞', '💬']
        self.default_confidence = float(os.getenv('DEFAULT_CONFIDENCE', '0.75'))
        self.low_confidence_threshold = float(os.getenv('LOW_CONFIDENCE_THRESHOLD', '0.4'))
        
        # Consensus worker needs separate read connection for queries
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
                'predictions': data.get('predictions', []) if isinstance(data, dict) else [],
                'full_image': data.get('full_image', []) if isinstance(data, dict) else []
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
                SELECT merged_id, source_result_ids, merged_data
                FROM merged_boxes 
                WHERE image_id = %s
                ORDER BY created DESC
            """, (image_id,))
            
            merged_boxes = []
            for row in cursor.fetchall():
                merged_data = row[2]  # JSONB data
                
                # Include all merged boxes (both single and multi-detection)
                # Single detections will be validated against classification/semantic services
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

            service_display_name = service_name
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

            # Handle full_image predictions as classification evidence (CLIP/Xception services)
            if 'full_image' in result and isinstance(result['full_image'], list):
                for prediction in result['full_image']:
                    if prediction.get('emoji') and prediction['emoji'] not in seen_emojis:
                        seen_emojis.add(prediction['emoji'])
                        all_detections.append({
                            'emoji': prediction['emoji'],
                            'service': service_display_name,
                            'evidence_type': 'classification',  # Force classification type for full image predictions
                            'confidence': prediction.get('confidence', self.default_confidence),
                            'context': {
                                'source': 'full_image_classification',
                                'label': prediction.get('label', '')
                            },
                            'shiny': prediction.get('shiny', False)
                        })

        # Extract spatial detections from merged_boxes data (harmonized bbox data)
        if bounding_box_data and isinstance(bounding_box_data, list):
            for merged_box in bounding_box_data:
                if merged_box.get('emoji'):
                    detection_count = merged_box.get('detection_count', 0)
                    
                    # Multi-detection boxes are always included
                    if detection_count >= 2:
                        include_detection = True
                    # Single-detection boxes need validation against classification/semantic services
                    elif detection_count == 1:
                        include_detection = self.validate_single_detection_with_other_services(
                            merged_box['emoji'], service_results
                        )
                    else:
                        include_detection = False
                    
                    if include_detection:
                        all_detections.append({
                            'emoji': merged_box['emoji'],
                            'service': 'merged_boxes',
                            'evidence_type': 'spatial',
                            'confidence': merged_box.get('avg_confidence', 0.75),
                            'context': {
                                'source': 'merged_boxes',
                                'contributing_services': merged_box.get('contributing_services', []),
                                'detection_count': detection_count,
                                'single_detection_validated': detection_count == 1
                            },
                            'shiny': False,
                            'spatial_data': {
                                'detection_count': detection_count,
                                'avg_confidence': merged_box.get('avg_confidence', 0),
                                'contributing_services': merged_box.get('contributing_services', []),
                                'bbox': merged_box.get('merged_bbox', {}),
                                'labels': merged_box.get('labels', [])
                            }
                        })

        return all_detections
    
    def validate_single_detection_with_other_services(self, emoji, service_results):
        """Check if a single spatial detection has support from classification/semantic services"""
        for service_name, result_data in service_results.items():
            # Check if service is classification type or semantic type using config
            is_classification = service_name in self.config.get_services_by_type('classification')
            is_semantic = self.config.is_semantic_service(service_name)
            
            if is_classification or is_semantic:
                if self.service_supports_emoji(emoji, result_data):
                    self.logger.info(f"Single detection {emoji} validated by {service_name}")
                    return True
        
        return False
    
    def service_supports_emoji(self, emoji, result_data):
        """Check if a service result includes the given emoji"""
        try:
            if isinstance(result_data, dict):
                predictions = result_data.get('predictions', [])
                for prediction in predictions:
                    if normalize_emoji(prediction.get('emoji', '')) == normalize_emoji(emoji):
                        return True
        except Exception as e:
            self.logger.debug(f"Error checking service support for emoji {emoji}: {e}")
        return False
    
    def get_evidence_type(self, service_name):
        """Determine evidence type based on service configuration"""
        # Use the same methods as BaseWorker
        if self.config.is_spatial_service(service_name):
            return 'spatial'
        elif self.config.is_semantic_service(service_name):
            return 'semantic'
        else:
            # For classification services like CLIP, xception
            return 'classification'

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
            emoji = normalize_emoji(detection['emoji'])
            if emoji not in groups:
                groups[emoji] = []
            groups[emoji].append(detection)
        
        return groups

    def analyze_emoji_evidence(self, emoji_groups, service_results, merged_boxes_data=None):
        """Analyze evidence for each emoji group (ported from V3VotingService.js)"""
        analysis = []
        
        for emoji, detections in emoji_groups.items():
            voting_services = list(set(d['service'] for d in detections if d['service'] != 'spatial_clustering'))
            
            # Analyze evidence first
            spatial_evidence = self.analyze_spatial_evidence(detections, merged_boxes_data)
            semantic_evidence = self.analyze_semantic_evidence(detections)
            classification_evidence = self.analyze_classification_evidence(detections)
            specialized_evidence = self.analyze_specialized_evidence(detections)
            
            evidence_analysis = {
                'emoji': emoji,
                'total_votes': len(voting_services),
                'voting_services': voting_services,
                'detections': detections,
                'evidence': {
                    'spatial': spatial_evidence,
                    'semantic': semantic_evidence,
                    'classification': classification_evidence,
                    'specialized': specialized_evidence
                },
                'instances': self.extract_instance_information(detections, spatial_evidence),
                'shiny': any(d.get('shiny', False) for d in detections)
            }
            
            analysis.append(evidence_analysis)
        
        return analysis

    def analyze_spatial_evidence(self, detections, merged_boxes_data=None):
        """Analyze spatial evidence from merged_boxes data (which only exists if spatial services were involved)"""
        # No merged_boxes data = no spatial evidence  
        if not merged_boxes_data:
            return None
            
        # Find merged boxes that match this emoji
        emoji = detections[0].get('emoji') if detections else None
        if not emoji:
            return None
            
        # Normalize emoji for consistent matching
        normalized_emoji = normalize_emoji(emoji)
        matching_boxes = [box for box in merged_boxes_data if normalize_emoji(box.get('emoji', '')) == normalized_emoji]
        if not matching_boxes:
            return None
        
        # Verify that contributing services are actually spatial services according to config
        valid_boxes = []
        for box in matching_boxes:
            contributing_services = box.get('contributing_services', [])
            has_spatial_services = False
            for service in contributing_services:
                # Check if service is spatial (service names in merged_boxes are simple names like "detectron2")
                self.logger.debug(f"Checking service: {service} -> primary.{service}")
                if self.config.is_spatial_service(f"primary.{service}"):
                    self.logger.debug(f"Service {service} is spatial")
                    has_spatial_services = True
                    break
                else:
                    self.logger.debug(f"Service {service} is NOT spatial")
            if has_spatial_services:
                valid_boxes.append(box)
        
        if not valid_boxes:
            return None
        
        # Filter out single-service detections if multi-service instances exist for this emoji
        # Pass detections for classification/semantic validation of single-service detections
        filtered_boxes = self.filter_single_service_detections(valid_boxes, emoji, detections)
        
        if not filtered_boxes:
            return None
        
        # Use the harmonized spatial data from filtered valid merged_boxes
        confidences = [box.get('avg_confidence', 0) for box in filtered_boxes]
        return {
            'service_count': sum(box.get('detection_count', 0) for box in filtered_boxes),
            'clusters': [{
                'detection_count': box.get('detection_count', 0),
                'avg_confidence': box.get('avg_confidence', 0),
                'contributing_services': box.get('contributing_services', []),
                'bbox': box.get('merged_bbox', {}),
                'labels': box.get('labels', [])
            } for box in filtered_boxes],
            'max_detection_count': max(box.get('detection_count', 0) for box in filtered_boxes),
            'avg_confidence': sum(confidences) / len(confidences),
            'peak_confidence': max(confidences),
            'total_instances': len(filtered_boxes)
        }

    def filter_single_service_detections(self, valid_boxes, emoji, detections):
        """Filter out single-service detections when multi-service instances exist for the same emoji"""
        # Separate single-service from multi-service detections
        multi_service_boxes = [box for box in valid_boxes if box.get('detection_count', 0) > 1]
        single_service_boxes = [box for box in valid_boxes if box.get('detection_count', 0) == 1]
        
        # If there are multi-service detections, discard single-service ones for this emoji
        if multi_service_boxes:
            self.logger.info(f"Filtering out {len(single_service_boxes)} single-service detections for {emoji} - multi-service instances exist")
            return multi_service_boxes
        
        # Only single-service detections exist - validate them against semantic/classification evidence
        if single_service_boxes:
            if self.validate_single_service_detections(emoji, detections):
                self.logger.info(f"Keeping {len(single_service_boxes)} single-service detections for {emoji} - validated by semantic/classification services")
                return valid_boxes
            else:
                self.logger.info(f"Filtering out {len(single_service_boxes)} single-service detections for {emoji} - no semantic/classification validation")
                return []
        
        return valid_boxes

    def validate_single_service_detections(self, emoji, detections):
        """Validate single-service detections against semantic and classification evidence"""
        # Check for semantic evidence (captioning services mentioned this emoji)
        semantic_detections = [d for d in detections if d.get('evidence_type') == 'semantic' and d.get('emoji') == emoji]
        has_semantic_support = len(semantic_detections) > 0
        
        # Check for classification evidence (classification services detected this emoji)
        classification_detections = [d for d in detections if d.get('evidence_type') == 'classification' and d.get('emoji') == emoji]
        has_classification_support = len(classification_detections) > 0
        
        if has_semantic_support:
            self.logger.debug(f"Single-service {emoji} validated by semantic evidence from {len(semantic_detections)} services")
            return True
        
        if has_classification_support:
            self.logger.debug(f"Single-service {emoji} validated by classification evidence from {len(classification_detections)} services")
            return True
        
        self.logger.debug(f"Single-service {emoji} has no semantic or classification validation")
        return False

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

    def extract_instance_information(self, detections, spatial_evidence=None):
        """Extract instance information - if there's spatial evidence, it's spatial"""
        # If there's any spatial evidence (bounding boxes), it's spatial
        if spatial_evidence:
            return {
                'count': spatial_evidence.get('total_instances', 1),
                'type': 'spatial'
            }
        
        # No spatial evidence = non-spatial
        return {'count': 1, 'type': 'non_spatial'}

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
        
        # Classification consensus bonus: Agreement across classification services (CLIP/Xception full image)
        classification_consensus_bonus = 0
        classification_count = analysis['evidence']['classification']['service_count'] if analysis['evidence']['classification'] else 0
        
        if classification_count >= 2:
            # Consensus = classification_count - 1 (one vote doesn't count as consensus)
            classification_consensus_bonus = classification_count - 1
        
        # Total weight = democratic votes + consensus bonuses
        weight = base_votes + spatial_consensus_bonus + content_consensus_bonus + classification_consensus_bonus
        
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
                    'classification': analysis['evidence']['classification'],
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
            'peak_confidence': round(spatial_evidence['peak_confidence'], 3),
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
        
        # Spatial relationships handled by harmony worker - consensus focuses on voting/scoring
        
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
                INSERT INTO consensus (image_id, consensus_data)
                VALUES (%s, %s)
            """, (image_id, json.dumps(consensus_data)))
            
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
        
        if not self.connect_to_queue():
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
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            if self.db_conn:
                self.db_conn.close()
            if self.read_db_conn:
                self.read_db_conn.close()
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