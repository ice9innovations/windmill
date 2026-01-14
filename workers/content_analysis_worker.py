#!/usr/bin/env python3
"""
Content Analysis Worker
Semantic-spatial content understanding and scene classification
Analyzes NudeNet detections + VLM captions + bbox relationships
"""
import os
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import psycopg2
from datetime import datetime
from base_worker import BaseWorker

# Import our analysis utilities
from utils.semantic_validation import (
    extract_keywords_from_captions,
    validate_category_with_captions,
    infer_gender_from_anatomy,
    detect_gender_hallucination,
    vote_on_gender,
    classify_female_solo_nudity
)
from utils.spatial_analysis import (
    detect_person_containment,
    detect_sexual_activities
)
from utils.framing_analysis import classify_framing
from utils.face_correlation import correlate_faces, get_face_gender_attribution

ANALYSIS_VERSION = '1.2.0'


class ContentAnalysisWorker(BaseWorker):
    """Worker for semantic-spatial content analysis"""

    def __init__(self):
        super().__init__('system.content_analysis', env_file='.env')

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
        """
        Ensure database connections are healthy, reconnect if needed.
        Overrides base class to also check read connection.
        """
        # Check main connection using base class (includes backoff logic)
        if not super().ensure_database_connection():
            return False

        # Also check read connection
        try:
            if not self.read_db_conn or self.read_db_conn.closed:
                self.logger.warning("Read database connection is closed")
                return self.connect_to_database()

            # Validate read connection with test query
            cursor = self.read_db_conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()

            return True

        except Exception as e:
            self.logger.warning(f"Read database connection unhealthy: {e}")
            return self.connect_to_database()

    def get_image_data(self, image_id):
        """
        Fetch all data needed for content analysis.

        Returns dict with:
        - results: all service results
        - merged_boxes: harmonized bboxes
        - consensus: consensus detections
        - captions: VLM captions
        - nudenet_detections: NudeNet anatomy detections
        """
        try:
            cursor = self.read_db_conn.cursor()

            # Get all service results
            cursor.execute("""
                SELECT service, data, status
                FROM results
                WHERE image_id = %s AND status = 'success'
            """, (image_id,))

            results = []
            captions = []
            nudenet_detections = []
            image_width = 0
            image_height = 0

            for row in cursor.fetchall():
                service, data, status = row
                results.append({'service': service, 'data': data, 'status': status})

                # Extract captions from VLM services
                if service in ['blip', 'ollama', 'cogvlm']:
                    predictions = data.get('predictions', [])
                    for pred in predictions:
                        if 'text' in pred:
                            captions.append({'service': service, 'text': pred['text']})

                # Extract NudeNet anatomy detections
                if service == 'nudenet':
                    predictions = data.get('predictions', [])
                    for pred in predictions:
                        if pred.get('label') and pred.get('bbox'):
                            nudenet_detections.append({
                                'label': pred['label'],
                                'bbox': pred['bbox'],
                                'confidence': pred.get('confidence', 0.0)
                            })

                # Extract image dimensions from metadata service
                if service == 'metadata':
                    predictions = data.get('predictions', [])
                    if predictions and 'dimensions' in predictions[0]:
                        image_width = predictions[0]['dimensions'].get('width', 0)
                        image_height = predictions[0]['dimensions'].get('height', 0)

            # Get merged boxes (harmonized bboxes)
            cursor.execute("""
                SELECT merged_id, merged_data
                FROM merged_boxes
                WHERE image_id = %s
            """, (image_id,))

            merged_boxes = []
            for row in cursor.fetchall():
                merged_id, merged_data = row
                merged_boxes.append({
                    'merged_id': merged_id,
                    'data': merged_data
                })

            # Get consensus
            cursor.execute("""
                SELECT consensus_data
                FROM consensus
                WHERE image_id = %s
            """, (image_id,))

            consensus_data = None
            row = cursor.fetchone()
            if row:
                consensus_data = row[0]

            # Get face service results from postprocessing
            cursor.execute("""
                SELECT data
                FROM postprocessing
                WHERE image_id = %s AND service = 'face' AND status = 'success'
            """, (image_id,))

            face_service_detections = []
            for row in cursor.fetchall():
                face_data = row[0]
                if face_data and face_data.get('predictions'):
                    for pred in face_data['predictions']:
                        if pred.get('bbox'):
                            face_service_detections.append({
                                'bbox': pred['bbox'],
                                'keypoints': pred.get('keypoints'),
                                'confidence': pred.get('confidence', 0.0),
                                'label': pred.get('label', 'face')
                            })

            cursor.close()

            return {
                'results': results,
                'merged_boxes': merged_boxes,
                'consensus': consensus_data,
                'captions': captions,
                'nudenet_detections': nudenet_detections,
                'face_service_detections': face_service_detections,
                'image_width': image_width,
                'image_height': image_height
            }

        except Exception as e:
            self.logger.error(f"Error fetching image data: {e}")
            return None

    def analyze_content(self, image_id, image_data):
        """
        Run full semantic-spatial content analysis.

        Returns dict with complete analysis results.
        """
        try:
            captions = image_data['captions']
            nudenet_detections = image_data['nudenet_detections']
            merged_boxes = image_data['merged_boxes']
            consensus_data = image_data.get('consensus')

            # Extract consensus emojis if available
            consensus_emojis = []
            if consensus_data and 'votes' in consensus_data and 'consensus' in consensus_data['votes']:
                consensus_emojis = [v['emoji'] for v in consensus_data['votes']['consensus']]

            # Extract keywords from captions
            extracted_keywords = extract_keywords_from_captions(captions)

            # Get person bboxes from merged_boxes
            person_bboxes = []
            for mb in merged_boxes:
                merged_data = mb['data']
                emoji = merged_data.get('emoji', '')
                if emoji == '🧑' and merged_data.get('merged_bbox'):
                    person_bboxes.append({
                        'id': mb['merged_id'],
                        'bbox': merged_data['merged_bbox']
                    })

            # Detect person bbox containment (same person detected multiple times)
            containment_relationships = detect_person_containment(person_bboxes)

            # Deduplicate person count
            person_bboxes_raw = len(person_bboxes)
            contained_ids = set(c['contained_bbox_id'] for c in containment_relationships)
            deduplicated_person_bboxes = [p for p in person_bboxes if p['id'] not in contained_ids]
            person_bboxes_deduplicated = len(deduplicated_person_bboxes)

            # Infer gender from anatomy (spatial evidence)
            spatial_gender = infer_gender_from_anatomy(nudenet_detections)

            # Vote on gender using both NudeNet and VLM evidence
            gender_vote = vote_on_gender(spatial_gender, captions)

            # Detect VLM gender hallucinations (for flagging, not confidence)
            vlm_hallucinations = detect_gender_hallucination(captions, spatial_gender)

            # Correlate face detections from NudeNet and face service
            nudenet_face_detections = [
                d for d in nudenet_detections
                if 'FACE' in d['label']
            ]
            face_service_detections = image_data.get('face_service_detections', [])

            face_correlations = correlate_faces(
                nudenet_face_detections,
                face_service_detections,
                iou_threshold=0.3
            )

            # Validate NudeNet categories with captions
            semantic_validations = []
            for detection in nudenet_detections:
                validation = validate_category_with_captions(detection['label'], captions)
                semantic_validations.append(validation)

            # Detect sexual activities from spatial relationships
            activity_analysis = detect_sexual_activities(
                nudenet_detections,
                deduplicated_person_bboxes,
                extracted_keywords['combined_text']
            )

            # Classify scene type
            scene_type = activity_analysis['scene_type']
            intimacy_level = activity_analysis['intimacy_level']

            # Analyze framing based on bbox sizes
            framing_analysis = classify_framing(
                nudenet_detections,
                deduplicated_person_bboxes,
                image_data['image_width'],
                image_data['image_height']
            )

            # Determine gender breakdown
            anatomy_labels = [d['label'] for d in nudenet_detections]
            female_nudity = any('FEMALE' in label for label in anatomy_labels)
            male_nudity = any('MALE_GENITALIA' in label for label in anatomy_labels)
            mixed_gender = female_nudity and male_nudity

            # Extract anatomy exposed
            anatomy_exposed = list(set([d['label'] for d in nudenet_detections]))

            # Gender breakdown with voted confidence (NudeNet + VLM corroboration)
            gender_breakdown = {
                'female_nudity': female_nudity,
                'male_nudity': male_nudity,
                'mixed_gender': mixed_gender,
                'confidence': {
                    'female': gender_vote['confidence'] if gender_vote['gender'] == 'female' else 0.0,
                    'male': gender_vote['confidence'] if gender_vote['gender'] == 'male' else 0.0
                },
                'vote_details': gender_vote['votes'],
                'reasoning': gender_vote['reasoning']
            }

            # Person attributions using voted gender
            person_attributions = []
            if person_bboxes_deduplicated > 0:
                # Use voted gender (combines NudeNet + VLM evidence)
                person_attributions.append({
                    'bbox_ids': [p['id'] for p in deduplicated_person_bboxes],
                    'gender': gender_vote['gender'],
                    'confidence': gender_vote['confidence'],
                    'spatial_markers': anatomy_exposed,
                    'vlm_agreement': gender_vote['votes']['agreement_count'],
                    'vlm_disagreement': gender_vote['votes']['disagreement_count'],
                    'reasoning': gender_vote['reasoning']
                })

            # Semantic validation summary
            semantic_validation = {
                'corroborated': any(v['corroborated'] for v in semantic_validations),
                'conflicts': [v for v in semantic_validations if v['conflicted']],
                'confidence': sum(v['confidence'] for v in semantic_validations) / len(semantic_validations) if semantic_validations else 0.0
            }

            # Full analysis output
            full_analysis = {
                'image_id': image_id,
                'version': ANALYSIS_VERSION,
                'timestamp': datetime.now().isoformat(),
                'keyword_extraction': extracted_keywords,
                'semantic_validations': semantic_validations,
                'spatial_gender_inference': spatial_gender,
                'gender_vote': gender_vote,
                'vlm_hallucinations': vlm_hallucinations,
                'activity_analysis': activity_analysis,
                'framing_analysis': framing_analysis,
                'face_correlations': face_correlations,
                'person_deduplication': {
                    'raw_count': person_bboxes_raw,
                    'deduplicated_count': person_bboxes_deduplicated,
                    'containments': containment_relationships
                }
            }

            return {
                'image_id': image_id,
                'gender_breakdown': gender_breakdown,
                'anatomy_exposed': anatomy_exposed,
                'scene_type': scene_type,
                'intimacy_level': intimacy_level,
                'activities_detected': activity_analysis['activities'],
                'spatial_relationships': activity_analysis['spatial_relationships'],
                'person_bboxes_raw': person_bboxes_raw,
                'person_bboxes_deduplicated': person_bboxes_deduplicated,
                'containment_relationships': containment_relationships,
                'semantic_validation': semantic_validation,
                'vlm_hallucinations': vlm_hallucinations,
                'people_count': person_bboxes_deduplicated,
                'person_attributions': person_attributions,
                'framing_analysis': framing_analysis,
                'face_correlations': face_correlations,
                'full_analysis': full_analysis,
                'analysis_version': ANALYSIS_VERSION
            }

        except Exception as e:
            self.logger.error(f"Error analyzing content for image {image_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def store_analysis(self, analysis):
        """Store content analysis in database"""
        try:
            cursor = self.db_conn.cursor()

            # Use INSERT ... ON CONFLICT UPDATE to handle re-analysis
            cursor.execute("""
                INSERT INTO content_analysis (
                    image_id, gender_breakdown, anatomy_exposed, scene_type, intimacy_level,
                    activities_detected, spatial_relationships, person_bboxes_raw,
                    person_bboxes_deduplicated, containment_relationships, semantic_validation,
                    vlm_hallucinations, people_count, person_attributions, framing_analysis,
                    face_correlations, full_analysis, analysis_version, created
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (image_id) DO UPDATE SET
                    gender_breakdown = EXCLUDED.gender_breakdown,
                    anatomy_exposed = EXCLUDED.anatomy_exposed,
                    scene_type = EXCLUDED.scene_type,
                    intimacy_level = EXCLUDED.intimacy_level,
                    activities_detected = EXCLUDED.activities_detected,
                    spatial_relationships = EXCLUDED.spatial_relationships,
                    person_bboxes_raw = EXCLUDED.person_bboxes_raw,
                    person_bboxes_deduplicated = EXCLUDED.person_bboxes_deduplicated,
                    containment_relationships = EXCLUDED.containment_relationships,
                    semantic_validation = EXCLUDED.semantic_validation,
                    vlm_hallucinations = EXCLUDED.vlm_hallucinations,
                    people_count = EXCLUDED.people_count,
                    person_attributions = EXCLUDED.person_attributions,
                    framing_analysis = EXCLUDED.framing_analysis,
                    face_correlations = EXCLUDED.face_correlations,
                    full_analysis = EXCLUDED.full_analysis,
                    analysis_version = EXCLUDED.analysis_version,
                    created = EXCLUDED.created
            """, (
                analysis['image_id'],
                json.dumps(analysis['gender_breakdown']),
                analysis['anatomy_exposed'],
                analysis['scene_type'],
                analysis['intimacy_level'],
                analysis['activities_detected'],
                json.dumps(analysis['spatial_relationships']),
                analysis['person_bboxes_raw'],
                analysis['person_bboxes_deduplicated'],
                json.dumps(analysis['containment_relationships']),
                json.dumps(analysis['semantic_validation']),
                json.dumps(analysis['vlm_hallucinations']),
                analysis['people_count'],
                json.dumps(analysis['person_attributions']),
                json.dumps(analysis['framing_analysis']),
                json.dumps(analysis['face_correlations']),
                json.dumps(analysis['full_analysis']),
                analysis['analysis_version'],
                datetime.now()
            ))

            self.db_conn.commit()
            cursor.close()

            self.logger.info(f"Stored content analysis for image {analysis['image_id']}: "
                           f"scene={analysis['scene_type']}, "
                           f"activities={len(analysis['activities_detected'])}, "
                           f"people={analysis['people_count']}")

            return True

        except Exception as e:
            self.logger.error(f"Error storing analysis: {e}")
            self.db_conn.rollback()
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def process_message(self, ch, method, properties, body):
        """Process content analysis message"""
        try:
            # Ensure database connection is healthy before processing
            if not self.ensure_database_connection():
                self.logger.error(
                    "Database connection unavailable, rejecting message without requeue. "
                    "Worker will retry after backoff delay."
                )
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                self.job_failed("Database unavailable")
                return

            # Parse message
            message = json.loads(body)
            image_id = message['image_id']
            trace_id = message.get('trace_id')

            if trace_id:
                self.logger.debug(f"[{trace_id}] Processing content analysis for image {image_id}")
            else:
                self.logger.debug(f"Processing content analysis for image {image_id}")

            # Fetch all image data
            image_data = self.get_image_data(image_id)
            if not image_data:
                self.logger.error(f"Failed to fetch image data for {image_id}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                self.job_failed("Failed to fetch image data")
                return

            # Run content analysis
            analysis = self.analyze_content(image_id, image_data)
            if not analysis:
                self.logger.error(f"Failed to analyze content for {image_id}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                self.job_failed("Analysis failed")
                return

            # Store analysis
            if not self.store_analysis(analysis):
                self.logger.error(f"Failed to store analysis for {image_id}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                self.job_failed("Storage failed")
                return

            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            self.job_completed_successfully()

            self.logger.info(f"Successfully completed content analysis for image {image_id}")

        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            # Database connection errors - don't requeue to prevent CPU spin
            self.logger.error(f"Database error processing content analysis message: {e}")
            self.logger.warning("Rejecting message without requeue due to database error")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            self.job_failed(str(e))
        except Exception as e:
            # Other errors - requeue for retry
            self.logger.error(f"Error processing content analysis message: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            self.job_failed(str(e))


if __name__ == "__main__":
    worker = ContentAnalysisWorker()
    worker.start()
