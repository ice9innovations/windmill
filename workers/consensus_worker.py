#!/usr/bin/env python3
"""
Consensus Worker V2 - Clean rewrite with explicit logic
Implements V3 voting algorithm with clear, debuggable confidence filtering
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

# Configuration constants
MINIMUM_CONFIDENCE = 0.8  # High confidence threshold
MULTI_SERVICE_MINIMUM_CONFIDENCE = 0.51  # Moderate confidence threshold
STRONG_CONSENSUS_THRESHOLD = 3  # Services needed for strong consensus
VERY_STRONG_CONSENSUS_THRESHOLD = 4  # Services needed to bypass confidence
MINIMUM_VOTES_REQUIRED = 2  # Minimum votes to include in results

def normalize_emoji(emoji):
    """Remove variation selectors and other invisible modifiers from emoji"""
    if not emoji:
        return emoji
    return re.sub(r'[\uFE00-\uFE0F\u180B-\u180D\u200D]', '', emoji)

class ConsensusWorkerV2(BaseWorker):
    """Clean consensus worker with explicit filtering logic"""

    def __init__(self):
        super().__init__('system.consensus')

        # V3 Voting configuration
        self.special_emojis = ['🔞', '💬']
        self.default_confidence = float(os.getenv('DEFAULT_CONFIDENCE', '0.75'))
        self.low_confidence_threshold = float(os.getenv('LOW_CONFIDENCE_THRESHOLD', '0.4'))

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

    def analyze_emoji_consensus(self, emoji, detections, spatial_evidence):
        """Analyze consensus for a specific emoji with clear logic"""
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
            'spatial_boxes': spatial_evidence.get(emoji, [])  # Include raw spatial boxes for bbox extraction
        }

    def apply_consensus_filtering(self, emoji_analysis):
        """Apply clear, debuggable consensus filtering"""
        filtered_results = []

        for analysis in emoji_analysis:
            should_keep = False
            reason = ""

            # Rule 1: Minimum votes required
            if analysis['total_votes'] < MINIMUM_VOTES_REQUIRED:
                reason = f"insufficient votes ({analysis['total_votes']} < {MINIMUM_VOTES_REQUIRED})"

            # Rule 2: Non-spatial detections (semantic/classification only)
            elif not analysis['spatial_analysis']:
                should_keep = True
                reason = "non-spatial detection (semantic/classification only)"

            # Rule 3: Spatial detections with confidence filtering
            else:
                spatial = analysis['spatial_analysis']
                max_services = spatial['max_services_per_box']
                peak_confidence = spatial['peak_confidence']

                # Very strong consensus (4+ services) - always keep
                if max_services >= VERY_STRONG_CONSENSUS_THRESHOLD:
                    should_keep = True
                    reason = f"very strong consensus ({max_services} services)"

                # Strong consensus (3 services) - lower confidence acceptable
                elif max_services >= STRONG_CONSENSUS_THRESHOLD:
                    should_keep = True
                    reason = f"strong consensus ({max_services} services, conf={peak_confidence:.3f})"

                # High confidence - keep regardless of consensus
                elif peak_confidence >= MINIMUM_CONFIDENCE:
                    should_keep = True
                    reason = f"high confidence ({peak_confidence:.3f})"

                # Moderate consensus (2 services) - need decent confidence AND semantic support
                elif max_services >= 2 and peak_confidence >= MULTI_SERVICE_MINIMUM_CONFIDENCE:
                    has_semantic_support = analysis['semantic_votes'] > 0 or analysis['classification_votes'] > 0
                    if has_semantic_support:
                        should_keep = True
                        reason = f"moderate consensus with semantic support ({max_services} services, conf={peak_confidence:.3f})"
                    else:
                        reason = f"moderate consensus without semantic support ({max_services} services, conf={peak_confidence:.3f})"

                # Everything else gets filtered
                else:
                    reason = f"low confidence and insufficient consensus (services={max_services}, conf={peak_confidence:.3f})"

            # Log the decision for debugging
            action = "KEEP" if should_keep else "FILTER"
            self.logger.debug(f"{action} {analysis['emoji']}: {reason}")

            if should_keep:
                analysis['filter_reason'] = reason
                filtered_results.append(analysis)

        self.logger.info(f"Consensus filtering: kept {len(filtered_results)}/{len(emoji_analysis)} emojis")
        return filtered_results

    def calculate_consensus(self, service_results, image_id):
        """Main consensus calculation with clear logic"""
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

        # Get spatial evidence
        spatial_evidence = self.get_spatial_evidence_for_image(image_id)

        # Group by emoji and analyze
        emoji_groups = self.group_detections_by_emoji(all_detections)
        emoji_analysis = []

        for emoji, detections in emoji_groups.items():
            analysis = self.analyze_emoji_consensus(emoji, detections, spatial_evidence)
            emoji_analysis.append(analysis)

        # Apply filtering
        filtered_results = self.apply_consensus_filtering(emoji_analysis)

        # Sort by total votes (descending)
        filtered_results.sort(key=lambda x: x['total_votes'], reverse=True)

        # Format final results
        consensus_results = []
        for analysis in filtered_results:
            result = {
                'emoji': analysis['emoji'],
                'votes': analysis['total_votes'],
                'evidence': {},
                'services': list(set(d['service'] for d in analysis['all_detections'])),
                'filter_reason': analysis['filter_reason']
            }

            # Add spatial evidence in the format the benchmark expects
            if analysis['spatial_analysis']:
                result['evidence']['spatial'] = {
                    'max_services_per_box': analysis['spatial_analysis']['max_services_per_box'],
                    'peak_confidence': round(analysis['spatial_analysis']['peak_confidence'], 3),
                    'avg_confidence': round(analysis['spatial_analysis']['avg_confidence'], 3),
                    'total_boxes': analysis['spatial_analysis']['total_boxes']
                }
                result['instances'] = {'type': 'spatial'}

                # Add bounding box coordinates for mAP calculation
                result['bounding_boxes'] = []
                for box in analysis['spatial_boxes']:
                    bbox = box.get('merged_bbox', {})
                    if bbox and all(k in bbox for k in ['x', 'y', 'width', 'height']):
                        result['bounding_boxes'].append({
                            'merged_bbox': bbox,  # Keep the nested structure the benchmark expects
                            'avg_confidence': box.get('avg_confidence', 0),
                            'peak_confidence': box.get('avg_confidence', 0),  # Use same value for peak
                            'detection_count': box.get('detection_count', 0)
                        })
            else:
                result['instances'] = {'type': 'non_spatial'}

            consensus_results.append(result)

        # Package final consensus in original format for benchmark compatibility
        consensus = {
            'services_count': len(service_results),
            'services_list': [row[0] for row in service_results],
            'total_processing_time': sum(row[2] or 0 for row in service_results),
            'latest_result_time': max(row[3] for row in service_results).isoformat(),
            'consensus_algorithm': 'v2_clear_logic',
            'votes': {
                'consensus': consensus_results
            },
            'special': {},  # Placeholder for special detections
            'debug': {
                'total_detections': len(all_detections),
                'emoji_groups_analyzed': len(emoji_analysis)
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
            result_count = len(consensus_data.get('consensus_results', []))
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

        self.logger.info(f"Starting consensus worker V2 ({self.worker_id})")
        self.logger.info(f"Listening on queue: {self.queue_name}")
        self.logger.info(f"Consensus thresholds: strong={STRONG_CONSENSUS_THRESHOLD}, very_strong={VERY_STRONG_CONSENSUS_THRESHOLD}")
        self.logger.info(f"Confidence thresholds: high={MINIMUM_CONFIDENCE}, multi_service={MULTI_SERVICE_MINIMUM_CONFIDENCE}")

        # Setup message consumer
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.process_queue_message
        )

        self.logger.info("Waiting for consensus messages. Press CTRL+C to exit")

        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.logger.info("Stopping consensus worker V2...")
            self.channel.stop_consuming()
        finally:
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            if self.db_conn:
                self.db_conn.close()
            if self.read_db_conn:
                self.read_db_conn.close()
            self.logger.info("Consensus worker V2 stopped")

        return 0

def main():
    """Main entry point"""
    try:
        worker = ConsensusWorkerV2()
        return worker.run()

    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Consensus worker V2 error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())