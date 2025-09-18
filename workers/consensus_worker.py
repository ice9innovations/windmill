#!/usr/bin/env python3
"""
Consensus Worker - Merge Focused
Completely redesigned to MERGE detections instead of filtering them out
Designed to fix the catastrophic 90%+ filter rates
"""
import os
import json
import time
import logging
import psycopg2
import pika
import re
from datetime import datetime
from base_worker import BaseWorker

# MERGE-FOCUSED configuration with democratic confidence
MINIMUM_VOTES_REQUIRED = 3  # Require at least 2-service consensus
TOTAL_AVAILABLE_SERVICES = 8  # Approximate total services that could vote

def normalize_emoji(emoji):
    """Remove variation selectors and other invisible modifiers from emoji"""
    if not emoji:
        return emoji
    return re.sub(r'[\uFE00-\uFE0F\u180B-\u180D\u200D]', '', emoji)

class ConsensusWorkerMergeFocused(BaseWorker):
    """Merge-focused consensus worker that prioritizes inclusion over filtering"""

    def __init__(self):
        super().__init__('system.consensus')

        # Merge-focused configuration
        self.special_emojis = ['🔞', '💬']
        self.default_confidence = float(os.getenv('DEFAULT_CONFIDENCE', '0.75'))

        # Dual database connections for read/write separation
        self.read_db_conn = None

    def connect_to_database(self):
        """Connect to PostgreSQL database with dual connections"""
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
            self.read_db_conn.autocommit = True

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

            # Parse JSON data
            parsed_results = []
            for service, data, processing_time, result_created in raw_results:
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

    def get_spatial_evidence_for_image(self, image_id):
        """Get harmonized spatial evidence from merged_boxes table"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT merged_id, source_result_ids, merged_data
                FROM merged_boxes
                WHERE image_id = %s
                ORDER BY created DESC
            """, (image_id,))

            spatial_evidence = {}
            for row in cursor.fetchall():
                merged_data = row[2]  # JSONB data
                emoji = merged_data.get('emoji')

                if emoji:
                    normalized_emoji = normalize_emoji(emoji)
                    if normalized_emoji not in spatial_evidence:
                        spatial_evidence[normalized_emoji] = []

                    spatial_evidence[normalized_emoji].append({
                        'merged_id': row[0],
                        'detection_count': merged_data.get('detection_count', 0),
                        'avg_confidence': merged_data.get('avg_confidence', 0),
                        'contributing_services': merged_data.get('contributing_services', []),
                        'merged_bbox': merged_data.get('merged_bbox', {}),
                        'labels': merged_data.get('labels', [])
                    })

            cursor.close()
            return spatial_evidence

        except Exception as e:
            self.logger.error(f"Error fetching spatial evidence: {e}")
            return {}

    def extract_detections_from_services(self, service_results):
        """Extract all detections from service results"""
        all_detections = []

        for service_name, result in service_results.items():
            if not result.get('success') or not result.get('predictions'):
                continue

            seen_emojis = set()  # Deduplicate within service

            for prediction in result['predictions']:
                # Handle emoji_mappings format
                if prediction.get('emoji_mappings') and isinstance(prediction['emoji_mappings'], list):
                    for mapping in prediction['emoji_mappings']:
                        emoji = mapping.get('emoji')
                        if emoji and emoji not in seen_emojis:
                            seen_emojis.add(emoji)
                            all_detections.append({
                                'emoji': emoji,
                                'service': service_name,
                                'evidence_type': self.get_evidence_type(service_name),
                                'confidence': self.default_confidence,
                                'context': {
                                    'word': mapping.get('word', ''),
                                    'source': 'caption_mapping'
                                }
                            })

                # Handle direct emoji format
                elif prediction.get('emoji') and prediction.get('type') != 'color_analysis':
                    emoji = prediction['emoji']
                    if emoji and emoji not in seen_emojis:
                        seen_emojis.add(emoji)
                        all_detections.append({
                            'emoji': emoji,
                            'service': service_name,
                            'evidence_type': self.get_evidence_type(service_name),
                            'confidence': prediction.get('confidence', self.default_confidence),
                            'context': self.extract_context(prediction, service_name)
                        })

            # Handle full_image predictions
            if 'full_image' in result and isinstance(result['full_image'], list):
                for prediction in result['full_image']:
                    emoji = prediction.get('emoji')
                    if emoji and emoji not in seen_emojis:
                        seen_emojis.add(emoji)
                        all_detections.append({
                            'emoji': emoji,
                            'service': service_name,
                            'evidence_type': 'classification',
                            'confidence': prediction.get('confidence', self.default_confidence),
                            'context': {
                                'source': 'full_image_classification',
                                'label': prediction.get('label', '')
                            }
                        })

        return all_detections

    def get_evidence_type(self, service_name):
        """Determine evidence type based on service configuration"""
        full_service_name = f"primary.{service_name}"

        if self.config.is_spatial_service(full_service_name) or self.config.is_spatial_service(service_name):
            return 'spatial'
        elif self.config.is_semantic_service(full_service_name) or self.config.is_semantic_service(service_name):
            return 'semantic'
        else:
            return 'classification'

    def extract_context(self, prediction, service_name):
        """Extract context information from prediction"""
        context = {}

        if service_name == 'face':
            context['pose'] = prediction.get('pose')
        if service_name == 'nsfw2':
            context['nsfw_confidence'] = prediction.get('confidence')
        if service_name == 'ocr':
            context['text_detected'] = prediction.get('has_text', False)
            context['text_content'] = prediction.get('text')

        return context

    def group_detections_by_emoji(self, detections):
        """Group detections by normalized emoji"""
        groups = {}

        for detection in detections:
            emoji = normalize_emoji(detection['emoji'])
            if emoji not in groups:
                groups[emoji] = []
            groups[emoji].append(detection)

        return groups

    def analyze_per_bbox_consensus(self, all_detections, spatial_evidence):
        """Analyze consensus per bounding box instance instead of per emoji"""
        bbox_results = []

        # Group detections by evidence type for easier lookup
        spatial_detections = [d for d in all_detections if d['evidence_type'] == 'spatial']
        semantic_detections = [d for d in all_detections if d['evidence_type'] == 'semantic']
        classification_detections = [d for d in all_detections if d['evidence_type'] == 'classification']

        # Process each bounding box as a separate voting unit
        for emoji, bbox_list in spatial_evidence.items():
            normalized_emoji = normalize_emoji(emoji)

            for bbox_data in bbox_list:
                merged_id = bbox_data['merged_id']
                contributing_services = bbox_data.get('contributing_services', [])

                # Count spatial votes (services that contributed to this specific bbox)
                spatial_votes = len(contributing_services)

                # Count semantic votes (all semantic services that detected this emoji support this bbox)
                semantic_votes = len([d for d in semantic_detections if normalize_emoji(d['emoji']) == normalized_emoji])

                # Count classification votes (all classification services that detected this emoji support this bbox)
                classification_votes = len([d for d in classification_detections if normalize_emoji(d['emoji']) == normalized_emoji])

                total_votes = spatial_votes + semantic_votes + classification_votes

                # Get all supporting detections for this bbox
                supporting_detections = []

                # Add spatial detections from contributing services
                for detection in spatial_detections:
                    if (normalize_emoji(detection['emoji']) == normalized_emoji and
                        detection['service'] in contributing_services):
                        supporting_detections.append(detection)

                # Add semantic detections for this emoji
                for detection in semantic_detections:
                    if normalize_emoji(detection['emoji']) == normalized_emoji:
                        supporting_detections.append(detection)

                # Add classification detections for this emoji
                for detection in classification_detections:
                    if normalize_emoji(detection['emoji']) == normalized_emoji:
                        supporting_detections.append(detection)

                bbox_analysis = {
                    'merged_id': merged_id,
                    'emoji': emoji,
                    'normalized_emoji': normalized_emoji,
                    'total_votes': total_votes,
                    'spatial_votes': spatial_votes,
                    'semantic_votes': semantic_votes,
                    'classification_votes': classification_votes,
                    'bbox_data': bbox_data,
                    'supporting_detections': supporting_detections,
                    'contributing_services': contributing_services
                }

                bbox_results.append(bbox_analysis)

        # Also handle non-spatial detections (emojis with no bounding boxes)
        # These get processed as single instances per emoji type
        all_spatial_emojis = set(normalize_emoji(emoji) for emoji in spatial_evidence.keys())
        processed_non_spatial = set()

        for detection in semantic_detections + classification_detections:
            normalized_emoji = normalize_emoji(detection['emoji'])
            if normalized_emoji not in all_spatial_emojis and normalized_emoji not in processed_non_spatial:
                # This emoji has no spatial evidence, create a non-spatial instance
                # Collect all detections for this emoji
                emoji_detections = [d for d in semantic_detections + classification_detections
                                  if normalize_emoji(d['emoji']) == normalized_emoji]

                semantic_count = len([d for d in emoji_detections if d['evidence_type'] == 'semantic'])
                classification_count = len([d for d in emoji_detections if d['evidence_type'] == 'classification'])

                bbox_analysis = {
                    'merged_id': None,
                    'emoji': detection['emoji'],
                    'normalized_emoji': normalized_emoji,
                    'total_votes': semantic_count + classification_count,
                    'spatial_votes': 0,
                    'semantic_votes': semantic_count,
                    'classification_votes': classification_count,
                    'bbox_data': None,
                    'supporting_detections': emoji_detections,
                    'contributing_services': [d['service'] for d in emoji_detections]
                }
                bbox_results.append(bbox_analysis)
                processed_non_spatial.add(normalized_emoji)

        return bbox_results

    def analyze_emoji_consensus_merge_focused(self, emoji, detections, spatial_evidence):
        """Analyze consensus for a specific emoji with merge-focused logic"""
        # Count votes by evidence type
        spatial_votes = [d for d in detections if d['evidence_type'] == 'spatial']
        semantic_votes = [d for d in detections if d['evidence_type'] == 'semantic']
        classification_votes = [d for d in detections if d['evidence_type'] == 'classification']

        total_votes = len(set(d['service'] for d in detections))

        # Analyze spatial evidence
        spatial_analysis = None
        if emoji in spatial_evidence:
            boxes = spatial_evidence[emoji]
            max_detection_count = max(box['detection_count'] for box in boxes)
            confidences = [box['avg_confidence'] for box in boxes]

            spatial_analysis = {
                'max_services_per_box': max_detection_count,
                'total_boxes': len(boxes),
                'avg_confidence': sum(confidences) / len(confidences),
                'peak_confidence': max(confidences),
                'contributing_services': list(set().union(*[box['contributing_services'] for box in boxes]))
            }

        return {
            'emoji': emoji,
            'total_votes': total_votes,
            'spatial_votes': len(spatial_votes),
            'semantic_votes': len(semantic_votes),
            'classification_votes': len(classification_votes),
            'spatial_analysis': spatial_analysis,
            'all_detections': detections,
            'spatial_boxes': spatial_evidence.get(emoji, [])
        }

    def apply_vote_filtering(self, emoji_analysis):
        """Apply simple vote-based filtering"""
        accepted_results = []

        for analysis in emoji_analysis:
            should_keep = False
            reason = ""

            # Vote threshold check with spatial preference
            spatial_votes_count = analysis['spatial_votes']
            total_votes_count = analysis['total_votes']

            # Accept if 3+ total votes OR 2+ spatial votes
            if total_votes_count >= MINIMUM_VOTES_REQUIRED:
                should_keep = True
                reason = f"sufficient total votes ({total_votes_count} >= {MINIMUM_VOTES_REQUIRED})"
            elif spatial_votes_count >= (MINIMUM_VOTES_REQUIRED - 1):
                should_keep = True
                reason = f"sufficient spatial votes ({spatial_votes_count} >= {MINIMUM_VOTES_REQUIRED - 1})"
            else:
                should_keep = False
                reason = f"insufficient votes (total: {total_votes_count}, spatial: {spatial_votes_count})"

            if should_keep and analysis['spatial_analysis']:
                spatial = analysis['spatial_analysis']
                max_services = spatial['max_services_per_box']
                peak_confidence = spatial['peak_confidence']
                reason += f" - spatial consensus ({max_services} services, peak conf={peak_confidence:.3f})"

            # Log the decision for debugging
            action = "ACCEPT" if should_keep else "FILTER"
            self.logger.debug(f"{action} {analysis['emoji']}: {reason}")

            if should_keep:
                analysis['filter_reason'] = reason
                accepted_results.append(analysis)

        kept_count = len(accepted_results)
        total_count = len(emoji_analysis)
        survival_rate = (kept_count / total_count * 100) if total_count > 0 else 0

        self.logger.info(f"Vote filtering: kept {kept_count}/{total_count} emojis ({survival_rate:.1f}% survival rate)")
        return accepted_results

    def apply_bbox_vote_filtering(self, bbox_analysis):
        """Apply vote-based filtering per bounding box"""
        accepted_results = []

        for analysis in bbox_analysis:
            should_keep = False
            reason = ""

            # Vote threshold check with spatial preference
            spatial_votes_count = analysis['spatial_votes']
            total_votes_count = analysis['total_votes']

            # Accept if 3+ total votes OR 2+ spatial votes
            if total_votes_count >= MINIMUM_VOTES_REQUIRED:
                should_keep = True
                reason = f"sufficient total votes ({total_votes_count} >= {MINIMUM_VOTES_REQUIRED})"
            elif spatial_votes_count >= (MINIMUM_VOTES_REQUIRED - 1):
                should_keep = True
                reason = f"sufficient spatial votes ({spatial_votes_count} >= {MINIMUM_VOTES_REQUIRED - 1})"
            else:
                should_keep = False
                reason = f"insufficient votes (total: {total_votes_count}, spatial: {spatial_votes_count})"

            # Add spatial details for debugging
            if should_keep and analysis['bbox_data']:
                bbox_data = analysis['bbox_data']
                avg_confidence = bbox_data.get('avg_confidence', 0)
                detection_count = bbox_data.get('detection_count', 0)
                reason += f" - bbox consensus ({detection_count} detections, avg conf={avg_confidence:.3f})"

            # Log the decision for debugging
            action = "ACCEPT" if should_keep else "FILTER"
            bbox_id = analysis.get('merged_id', 'non-spatial')
            self.logger.debug(f"{action} {analysis['emoji']} (bbox:{bbox_id}): {reason}")

            if should_keep:
                analysis['filter_reason'] = reason
                accepted_results.append(analysis)

        kept_count = len(accepted_results)
        total_count = len(bbox_analysis)
        survival_rate = (kept_count / total_count * 100) if total_count > 0 else 0

        self.logger.info(f"Bbox vote filtering: kept {kept_count}/{total_count} instances ({survival_rate:.1f}% survival rate)")

        # Apply near-identical box conflict resolution
        resolved_results = self.resolve_near_identical_conflicts(accepted_results)

        return resolved_results

    def resolve_near_identical_conflicts(self, bbox_results):
        """Resolve conflicts between nearly identical bounding boxes with different emojis"""
        if len(bbox_results) < 2:
            return bbox_results

        # Multi-criteria thresholds for near-identical boxes
        IOU_THRESHOLD = 0.8   # 80% overlap required
        CENTER_DISTANCE_THRESHOLD = 15  # pixels
        SIZE_SIMILARITY_THRESHOLD = 0.9  # 90% size similarity

        resolved_results = []
        eliminated_indices = set()

        for i, result1 in enumerate(bbox_results):
            if i in eliminated_indices:
                continue

            # Only check boxes that have spatial evidence
            if not result1.get('bbox_data') or not result1['bbox_data'].get('merged_bbox'):
                resolved_results.append(result1)
                continue

            bbox1 = result1['bbox_data']['merged_bbox']

            for j, result2 in enumerate(bbox_results[i+1:], i+1):
                if j in eliminated_indices:
                    continue

                # Only check boxes that have spatial evidence and different emojis
                if (not result2.get('bbox_data') or
                    not result2['bbox_data'].get('merged_bbox') or
                    result1['normalized_emoji'] == result2['normalized_emoji']):
                    continue

                bbox2 = result2['bbox_data']['merged_bbox']

                # Check if boxes are near-identical using multiple criteria
                if self.are_boxes_near_identical(bbox1, bbox2, IOU_THRESHOLD, CENTER_DISTANCE_THRESHOLD, SIZE_SIMILARITY_THRESHOLD):
                    # Near-identical boxes with different emojis - resolve conflict
                    votes1 = result1['total_votes']
                    votes2 = result2['total_votes']

                    # Get detailed metrics for logging
                    iou = self.calculate_bbox_iou(bbox1, bbox2)
                    center_dist = self.calculate_center_distance(bbox1, bbox2)
                    size_sim = self.calculate_size_similarity(bbox1, bbox2)

                    if votes1 > votes2:
                        # result1 wins, eliminate result2
                        eliminated_indices.add(j)
                        self.logger.info(f"Conflict resolved: {result1['emoji']} ({votes1} votes) beats {result2['emoji']} ({votes2} votes) - IoU: {iou:.3f}, center_dist: {center_dist:.1f}px, size_sim: {size_sim:.3f}")
                    elif votes2 > votes1:
                        # result2 wins, eliminate result1
                        eliminated_indices.add(i)
                        self.logger.info(f"Conflict resolved: {result2['emoji']} ({votes2} votes) beats {result1['emoji']} ({votes1} votes) - IoU: {iou:.3f}, center_dist: {center_dist:.1f}px, size_sim: {size_sim:.3f}")
                        break  # result1 is eliminated, stop checking
                    else:
                        # Tie - use confidence as tiebreaker
                        conf1 = result1['bbox_data'].get('avg_confidence', 0)
                        conf2 = result2['bbox_data'].get('avg_confidence', 0)

                        if conf1 > conf2:
                            eliminated_indices.add(j)
                            self.logger.info(f"Conflict resolved by confidence: {result1['emoji']} (conf: {conf1:.3f}) beats {result2['emoji']} (conf: {conf2:.3f}) - IoU: {iou:.3f}, center_dist: {center_dist:.1f}px, size_sim: {size_sim:.3f}")
                        elif conf2 > conf1:
                            eliminated_indices.add(i)
                            self.logger.info(f"Conflict resolved by confidence: {result2['emoji']} (conf: {conf2:.3f}) beats {result1['emoji']} (conf: {conf1:.3f}) - IoU: {iou:.3f}, center_dist: {center_dist:.1f}px, size_sim: {size_sim:.3f}")
                            break
                        # If still tied, keep both (very rare)

            # Add result1 if it wasn't eliminated
            if i not in eliminated_indices:
                resolved_results.append(result1)

        eliminated_count = len(eliminated_indices)
        if eliminated_count > 0:
            self.logger.info(f"Near-identical conflict resolution: eliminated {eliminated_count} conflicting boxes")

        return resolved_results

    def calculate_bbox_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1 = bbox1['x'], bbox1['y']
        x2_1, y2_1 = x1_1 + bbox1['width'], y1_1 + bbox1['height']

        x1_2, y1_2 = bbox2['x'], bbox2['y']
        x2_2, y2_2 = x1_2 + bbox2['width'], y1_2 + bbox2['height']

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0  # No intersection

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate areas
        area1 = bbox1['width'] * bbox1['height']
        area2 = bbox2['width'] * bbox2['height']
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def are_boxes_near_identical(self, bbox1, bbox2, iou_threshold, center_distance_threshold, size_similarity_threshold):
        """Check if two bounding boxes are near-identical using multiple criteria"""
        # 1. IoU check
        iou = self.calculate_bbox_iou(bbox1, bbox2)
        if iou < iou_threshold:
            return False

        # 2. Center distance check
        center_distance = self.calculate_center_distance(bbox1, bbox2)
        if center_distance > center_distance_threshold:
            return False

        # 3. Size similarity check
        size_similarity = self.calculate_size_similarity(bbox1, bbox2)
        if size_similarity < size_similarity_threshold:
            return False

        return True

    def calculate_center_distance(self, bbox1, bbox2):
        """Calculate distance between bounding box centers"""
        cx1 = bbox1['x'] + bbox1['width'] / 2
        cy1 = bbox1['y'] + bbox1['height'] / 2
        cx2 = bbox2['x'] + bbox2['width'] / 2
        cy2 = bbox2['y'] + bbox2['height'] / 2

        return ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5

    def calculate_size_similarity(self, bbox1, bbox2):
        """Calculate size similarity between two bounding boxes"""
        w1, h1 = bbox1['width'], bbox1['height']
        w2, h2 = bbox2['width'], bbox2['height']

        # Calculate similarity for both dimensions
        width_similarity = min(w1, w2) / max(w1, w2) if max(w1, w2) > 0 else 0
        height_similarity = min(h1, h2) / max(h1, h2) if max(h1, h2) > 0 else 0

        # Return the minimum similarity (both dimensions must be similar)
        return min(width_similarity, height_similarity)

    def calculate_consensus(self, service_results, image_id):
        """Main consensus calculation with per-bounding-box voting"""
        if not service_results:
            return None

        # Convert to dict format
        service_results_dict = {}
        for service, data, proc_time, created in service_results:
            service_results_dict[service] = {
                'success': True,
                'predictions': data.get('predictions', []),
                'full_image': data.get('full_image', [])
            }

        # Extract all detections
        all_detections = self.extract_detections_from_services(service_results_dict)

        # Get spatial evidence (bounding boxes) as primary voting units
        spatial_evidence = self.get_spatial_evidence_for_image(image_id)

        # NEW: Calculate votes per bounding box instead of per emoji
        bbox_analysis = self.analyze_per_bbox_consensus(all_detections, spatial_evidence)

        # Apply vote-based filtering per bounding box
        accepted_results = self.apply_bbox_vote_filtering(bbox_analysis)

        # Sort by total votes (descending)
        accepted_results.sort(key=lambda x: x['total_votes'], reverse=True)

        # Format final results per bounding box
        consensus_results = []
        for analysis in accepted_results:
            # Calculate evidence breakdown
            non_spatial_votes = analysis['semantic_votes'] + analysis['classification_votes']

            result = {
                'emoji': analysis['emoji'],
                'votes': analysis['total_votes'],
                'merged_id': analysis.get('merged_id'),  # Include bounding box ID
                'democratic_confidence': analysis.get('democratic_confidence', 0),
                'evidence': {
                    'total_votes': analysis['total_votes'],
                    'spatial_votes': analysis['spatial_votes'],
                    'semantic_votes': analysis['semantic_votes'],
                    'classification_votes': analysis['classification_votes'],
                    'non_spatial_votes': non_spatial_votes,
                    'democratic_confidence': analysis.get('democratic_confidence', 0),
                    'summary': f"{non_spatial_votes} non-spatial" if non_spatial_votes > 0 else "spatial only",
                    'evidence_summary': f"{non_spatial_votes} non-spatial" if non_spatial_votes > 0 else "spatial only",
                    'final_decision': "consensus",
                    'total_evidence': non_spatial_votes
                },
                'services': list(set(d['service'] for d in analysis['supporting_detections'])),
                'filter_reason': analysis['filter_reason'],
                # Add top-level evidence fields in case frontend looks there
                'total_votes': analysis['total_votes'],
                'spatial_votes': analysis['spatial_votes'],
                'semantic_votes': analysis['semantic_votes'],
                'classification_votes': analysis['classification_votes'],
                'non_spatial_evidence': f"{non_spatial_votes} non-spatial" if non_spatial_votes > 0 else "spatial only"
            }

            # Add spatial evidence for this specific bounding box
            if analysis['bbox_data']:
                bbox_data = analysis['bbox_data']
                result['evidence']['spatial'] = {
                    'max_services_per_box': bbox_data.get('detection_count', 0),
                    'peak_confidence': round(bbox_data.get('avg_confidence', 0), 3),
                    'avg_confidence': round(bbox_data.get('avg_confidence', 0), 3),
                    'total_boxes': 1,  # This is for a single bounding box
                    'detection_count': bbox_data.get('detection_count', 0)
                }
                result['instances'] = {'type': 'spatial'}

                # Add bounding box coordinates for this specific box
                bbox = bbox_data.get('merged_bbox', {})
                if bbox and all(k in bbox for k in ['x', 'y', 'width', 'height']):
                    result['bounding_boxes'] = [{
                        'merged_bbox': bbox,
                        'avg_confidence': bbox_data.get('avg_confidence', 0),
                        'peak_confidence': bbox_data.get('avg_confidence', 0),
                        'detection_count': bbox_data.get('detection_count', 0)
                    }]
                else:
                    result['bounding_boxes'] = []
            else:
                result['instances'] = {'type': 'non_spatial'}
                # Add empty spatial evidence for non-spatial items
                result['evidence']['spatial'] = {
                    'detection_count': 0,
                    'total_boxes': 0,
                    'avg_confidence': 0,
                    'peak_confidence': 0
                }
                result['bounding_boxes'] = []

            consensus_results.append(result)

        # Package final consensus in original format for benchmark compatibility
        consensus = {
            'services_count': len(service_results),
            'services_list': [row[0] for row in service_results],
            'total_processing_time': sum(row[2] or 0 for row in service_results),
            'latest_result_time': max(row[3] for row in service_results).isoformat(),
            'consensus_algorithm': 'merge_focused_permissive',
            'votes': {
                'consensus': consensus_results
            },
            'special': {},  # Placeholder for special detections
            'debug': {
                'total_detections': len(all_detections),
                'bbox_instances_analyzed': len(bbox_analysis),
                'optimization': 'merge_over_filter'
            }
        }

        return consensus

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

            # Ensure healthy database connection
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

            # Commit the transaction
            self.db_conn.commit()
            cursor.close()

            action = "Updated" if deleted_count > 0 else "Created"
            result_count = len(consensus_data.get('votes', {}).get('consensus', []))
            self.logger.info(f"{action} consensus for {image_filename}: {result_count} emojis from {len(service_results)} services")

            return True

        except Exception as e:
            self.logger.error(f"Error updating consensus for image {image_id}: {e}")
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

            self.logger.info(f"Processing merge-focused consensus for: {image_filename}")

            # Update consensus for this specific image
            success = self.update_consensus_for_image(image_id, image_filename)

            if success:
                self.logger.info(f"Completed merge-focused consensus for {image_filename}")
            else:
                self.logger.error(f"Failed to update merge-focused consensus for {image_filename}")

            # Acknowledge the message
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            self.logger.error(f"Error processing merge-focused consensus queue message: {e}")
            # Reject and requeue for retry
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def run(self):
        """Main entry point - pure queue-based processing"""
        if not self.connect_to_database():
            return 1

        if not self.connect_to_queue():
            return 1

        self.logger.info(f"Starting merge-focused consensus worker ({self.worker_id})")
        self.logger.info(f"Listening on queue: {self.queue_name}")
        self.logger.info(f"SIMPLE VOTE MODE: Vote count based filtering")
        self.logger.info(f"Minimum votes required: {MINIMUM_VOTES_REQUIRED}")

        # Setup message consumer
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.process_queue_message
        )

        self.logger.info("Waiting for merge-focused consensus messages. Press CTRL+C to exit")

        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.logger.info("Stopping merge-focused consensus worker...")
            self.channel.stop_consuming()
        finally:
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            if self.db_conn:
                self.db_conn.close()
            if self.read_db_conn:
                self.read_db_conn.close()
            self.logger.info("Merge-focused consensus worker stopped")

        return 0

def main():
    """Main entry point"""
    try:
        worker = ConsensusWorkerMergeFocused()
        return worker.run()

    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Merge-focused consensus worker error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
