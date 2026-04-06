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
import hashlib
import time
import psycopg2
import yaml
from datetime import datetime
from base_worker import BaseWorker
from core.postgres_connection import close_quietly, commit_if_needed, rollback_quietly

# Import our analysis utilities
from utils.semantic_validation import (
    extract_keywords_from_captions,
    validate_category_with_captions,
    infer_gender_from_anatomy,
    detect_gender_hallucination,
    vote_on_gender
)
from utils.spatial_analysis import (
    detect_person_containment,
    detect_sexual_activities
)
from utils.framing_analysis import classify_framing
from utils.face_correlation import correlate_faces, get_face_gender_attribution

ANALYSIS_VERSION = '1.4.8'  # Added canonical category + scene summary payload

_PERSON_EMOJIS = frozenset(['🧑', '👩', '🧒'])


def load_content_flags():
    """Load content flagging configuration from YAML file.

    Returns dict mapping category -> list of trigger words, or None on error.
    """
    # Config is in project root, not workers directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'moderation.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract just the triggers from each category
        flags = {}
        if config and 'flag_categories' in config:
            for category, data in config['flag_categories'].items():
                if 'triggers' in data:
                    flags[category] = data['triggers']

        return flags
    except Exception as e:
        # Log error but don't crash - flagging is optional
        print(f"Warning: Could not load content_flags.yaml: {e}")
        return None


def check_content_flags(noun_consensus_data, flag_config):
    """Check extracted nouns against content flag triggers.

    Args:
        noun_consensus_data: List of noun entries from noun_consensus.nouns
        flag_config: Dict from load_content_flags()

    Returns:
        List of flag dicts with category, value, vote_count, service_count
    """
    if not flag_config or not noun_consensus_data:
        return []

    flags = []

    for noun_entry in noun_consensus_data:
        canonical = noun_entry.get('canonical', '').lower()
        vote_count = noun_entry.get('vote_count', 1)
        services = noun_entry.get('services', [])
        service_count = len(services)

        if not canonical:
            continue

        # Check against all flag triggers
        for category, triggers in flag_config.items():
            for trigger in triggers:
                trigger_lower = trigger.lower()

                # Match if trigger appears in noun or noun contains trigger
                # Examples: "gun" matches "gun", "guns", "handgun"
                if trigger_lower in canonical or canonical in trigger_lower:
                    flags.append({
                        'category': category,
                        'value': canonical,
                        'vote_count': vote_count,
                        'service_count': service_count
                    })
                    break  # Don't match same noun multiple times in same category

    return flags


class ContentAnalysisWorker(BaseWorker):
    """Worker for semantic-spatial content analysis"""

    def __init__(self):
        super().__init__('system.content_analysis', env_file='.env')

        # Dual database connections for read/write separation
        self.read_db = self._new_managed_db_connection(
            autocommit=True,
            label='read database',
        )
        self.read_db_conn = None

        # Load content flagging config into memory once at startup
        self.content_flag_config = load_content_flags()
        if self.content_flag_config:
            category_count = len(self.content_flag_config)
            trigger_count = sum(len(triggers) for triggers in self.content_flag_config.values())
            self.logger.info(f"Loaded content flags: {category_count} categories, {trigger_count} triggers")
        else:
            self.logger.warning("Content flagging disabled (config not loaded)")

    def connect_to_database(self):
        """Connect to PostgreSQL database with dual connections"""
        if not super().connect_to_database():
            return False

        try:
            # Set up transaction mode for main connection
            self.db_conn.autocommit = False

            # Create separate read connection for queries
            self.read_db_conn = self.read_db.connect()

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

            return True

        except Exception as e:
            self.logger.warning(f"Read database connection unhealthy: {e}")
            return self.connect_to_database()

    def get_image_data(self, image_id):
        """
        Fetch all data needed for content analysis.

        Returns dict with:
        - results: all service results
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
            nsfw2_result = None
            image_width = 0
            image_height = 0
            yolo_person_bboxes = []

            for row in cursor.fetchall():
                service, data, status = row
                results.append({'service': service, 'data': data, 'status': status})

                # Extract captions from all VLM services
                if service in ['blip', 'ollama', 'cogvlm', 'haiku', 'moondream', 'qwen']:
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

                # Extract NSFW2 result
                if service == 'nsfw2':
                    predictions = data.get('predictions', [])
                    if predictions:
                        nsfw2_result = {
                            'nsfw': predictions[0].get('nsfw', False),
                            'confidence': predictions[0].get('confidence', 0.0),
                            'emoji': predictions[0].get('emoji', '')
                        }

                # Extract image dimensions from metadata service
                if service == 'metadata':
                    predictions = data.get('predictions', [])
                    if predictions and 'dimensions' in predictions[0]:
                        image_width = predictions[0]['dimensions'].get('width', 0)
                        image_height = predictions[0]['dimensions'].get('height', 0)

                # Extract person bboxes from YOLO (primary source for people count)
                if service == 'yolo_v8':
                    for i, pred in enumerate(data.get('predictions', [])):
                        raw_emoji = pred.get('emoji', '')
                        clean_emoji = ''.join(c for c in raw_emoji if ord(c) < 0xFE00 or ord(c) > 0xFE0F)
                        if clean_emoji not in _PERSON_EMOJIS:
                            continue
                        raw_bbox = pred.get('bbox')
                        if not raw_bbox:
                            continue
                        if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
                            bbox = {'x': raw_bbox[0], 'y': raw_bbox[1], 'width': raw_bbox[2], 'height': raw_bbox[3]}
                        elif isinstance(raw_bbox, dict) and all(k in raw_bbox for k in ('x', 'y', 'width', 'height')):
                            bbox = raw_bbox
                        else:
                            continue
                        yolo_person_bboxes.append({'id': i, 'bbox': bbox})

            face_service_detections = []
            cursor.execute("""
                SELECT data
                FROM results
                WHERE image_id = %s AND service = 'face' AND status = 'success'
                ORDER BY result_created DESC
                LIMIT 1
            """, (image_id,))
            row = cursor.fetchone()
            if row:
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
            else:
                cursor.execute("""
                    SELECT data
                    FROM postprocessing
                    WHERE image_id = %s AND service = 'face' AND status = 'success'
                """, (image_id,))
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

            # Fetch noun consensus data
            GENDERED_NOUNS = {
                'male':   {'man', 'men', 'boy', 'boys', 'gentleman', 'father',
                           'son', 'brother', 'husband', 'male'},
                'female': {'woman', 'women', 'girl', 'girls', 'lady', 'mother',
                           'daughter', 'sister', 'wife', 'female'},
            }
            validated_gendered_nouns = []
            noun_consensus_nouns = []

            cursor.execute("""
                SELECT nouns, service_count FROM noun_consensus WHERE image_id = %s
            """, (image_id,))
            nc_row = cursor.fetchone()
            if nc_row and nc_row[0]:
                noun_consensus_nouns = nc_row[0]  # Full noun list for flag checking

                # Extract SAM3-validated gendered nouns for gender analysis
                for entry in nc_row[0]:
                    if not entry.get('sam3_validated'):
                        continue
                    canonical = entry.get('canonical', '')
                    for gender, terms in GENDERED_NOUNS.items():
                        if canonical in terms:
                            validated_gendered_nouns.append({
                                'canonical': canonical,
                                'gender': gender,
                                'vote_count': entry.get('vote_count', 1),
                            })
                            break

            cursor.close()

            return {
                'results': results,
                'yolo_person_bboxes': yolo_person_bboxes,
                'captions': captions,
                'nudenet_detections': nudenet_detections,
                'nsfw2_result': nsfw2_result,
                'face_service_detections': face_service_detections,
                'image_width': image_width,
                'image_height': image_height,
                'validated_gendered_nouns': validated_gendered_nouns,
                'noun_consensus_nouns': noun_consensus_nouns,
            }

        except Exception as e:
            self.logger.error(f"Error fetching image data: {e}")
            return None

    def _build_input_fingerprint(self, image_data):
        """Build a stable fingerprint for the inputs that drive content analysis."""
        try:
            captions = sorted(
                (
                    item.get('service', ''),
                    item.get('text', ''),
                )
                for item in (image_data.get('captions') or [])
            )

            nudenet_detections = sorted(
                json.dumps(item, sort_keys=True, separators=(',', ':'))
                for item in (image_data.get('nudenet_detections') or [])
            )

            face_service_detections = sorted(
                json.dumps(item, sort_keys=True, separators=(',', ':'))
                for item in (image_data.get('face_service_detections') or [])
            )

            yolo_person_bboxes = sorted(
                json.dumps(item, sort_keys=True, separators=(',', ':'))
                for item in (image_data.get('yolo_person_bboxes') or [])
            )

            noun_consensus_nouns = sorted(
                json.dumps(item, sort_keys=True, separators=(',', ':'))
                for item in (image_data.get('noun_consensus_nouns') or [])
            )

            fingerprint_payload = {
                'captions': captions,
                'nudenet_detections': nudenet_detections,
                'nsfw2_result': image_data.get('nsfw2_result'),
                'image_width': image_data.get('image_width'),
                'image_height': image_data.get('image_height'),
                'yolo_person_bboxes': yolo_person_bboxes,
                'face_service_detections': face_service_detections,
                'noun_consensus_nouns': noun_consensus_nouns,
                'noun_consensus_service_count': image_data.get('noun_consensus_service_count'),
            }
            serialized = json.dumps(fingerprint_payload, sort_keys=True, separators=(',', ':'))
            return hashlib.sha256(serialized.encode('utf-8')).hexdigest()
        except Exception as e:
            self.logger.warning(f"Failed to build content analysis fingerprint: {e}")
            return None

    def _latest_input_fingerprint(self, image_id):
        """Return the last stored content-analysis input fingerprint, if any."""
        try:
            cursor = self.read_db_conn.cursor()
            cursor.execute(
                """
                SELECT data->'metadata'->>'input_fingerprint'
                FROM results
                WHERE image_id = %s
                  AND service = 'content_analysis'
                  AND status = 'success'
                ORDER BY result_created DESC
                LIMIT 1
                """,
                (image_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            return row[0] if row and row[0] else None
        except Exception as e:
            self.logger.warning(
                f"Failed to fetch latest content-analysis fingerprint for image {image_id}: {e}"
            )
            return None

    def correlate_with_nsfw2(self, scene_type, nsfw2_result, nudenet_detections=None):
        """
        Correlate our scene classification with NSFW2's verdict.

        When we classify as 'sfw' but NSFW2 says NSFW, we defer to NSFW2's judgment
        since it may have detected something we missed (e.g., suggestive poses,
        partial nudity, or context we didn't capture).

        Confidence thresholds:
        - High (>0.9): Override to 'suggestive' - NSFW2 is very confident
        - Medium (0.7-0.9): Override to 'suggestive' - likely suggestive content
        - Lower (0.5-0.7): Override to 'suggestive' if NudeNet found covered anatomy
                           (two weak signals together are meaningful); otherwise flag only

        Args:
            scene_type: Our current scene classification
            nsfw2_result: NSFW2 service result dict with 'nsfw', 'confidence', 'emoji'
            nudenet_detections: List of NudeNet detection dicts (optional, used for combined signal)

        Returns:
            dict: {
                'agreement': bool - whether we agree with NSFW2,
                'nsfw2_verdict': str - 'nsfw', 'sfw', or 'unknown',
                'nsfw2_confidence': float,
                'override_scene_type': bool,
                'new_scene_type': str or None,
                'new_intimacy_level': str or None,
                'reasoning': str
            }
        """
        # Default result - no NSFW2 data available
        if not nsfw2_result:
            return {
                'agreement': None,
                'nsfw2_verdict': 'unknown',
                'nsfw2_confidence': 0.0,
                'override_scene_type': False,
                'new_scene_type': None,
                'new_intimacy_level': None,
                'reasoning': 'no_nsfw2_data'
            }

        nsfw2_says_nsfw = nsfw2_result.get('nsfw', False)
        nsfw2_confidence = nsfw2_result.get('confidence', 0.0)
        nsfw2_verdict = 'nsfw' if nsfw2_says_nsfw else 'sfw'

        # Determine agreement
        # Canonical content categories above sfw are treated as NSFW for correlation.
        # Our SFW category: sfw
        our_nsfw = scene_type in ['suggestive', 'nudity', 'softcore_pornography', 'sexually_explicit']
        agreement = our_nsfw == nsfw2_says_nsfw

        # Only consider overrides when we say SFW but NSFW2 says NSFW
        if scene_type != 'sfw' or not nsfw2_says_nsfw:
            return {
                'agreement': agreement,
                'nsfw2_verdict': nsfw2_verdict,
                'nsfw2_confidence': nsfw2_confidence,
                'override_scene_type': False,
                'new_scene_type': None,
                'new_intimacy_level': None,
                'reasoning': 'agreement' if agreement else 'no_override_needed'
            }

        # We classified as SFW but NSFW2 says NSFW - apply confidence thresholds
        if nsfw2_confidence > 0.9:
            # High confidence: NSFW2 is very sure this is at least suggestive.
            return {
                'agreement': False,
                'nsfw2_verdict': nsfw2_verdict,
                'nsfw2_confidence': nsfw2_confidence,
                'override_scene_type': True,
                'new_scene_type': 'suggestive',
                'new_intimacy_level': 'suggestive',
                'reasoning': 'nsfw2_high_confidence_override'
            }
        elif nsfw2_confidence > 0.7:
            # Medium confidence: likely suggestive content
            return {
                'agreement': False,
                'nsfw2_verdict': nsfw2_verdict,
                'nsfw2_confidence': nsfw2_confidence,
                'override_scene_type': True,
                'new_scene_type': 'suggestive',
                'new_intimacy_level': 'suggestive',
                'reasoning': 'nsfw2_medium_confidence_override'
            }
        else:
            # Lower confidence: check for corroborating covered anatomy from NudeNet.
            # A weak nsfw2 signal combined with NudeNet confirming something is being
            # covered is enough to call it suggestive (lingerie, swimwear, etc.).
            covered_labels = [
                d['label'] for d in (nudenet_detections or [])
                if 'COVERED' in d.get('label', '')
            ]
            if covered_labels:
                return {
                    'agreement': False,
                    'nsfw2_verdict': nsfw2_verdict,
                    'nsfw2_confidence': nsfw2_confidence,
                    'override_scene_type': True,
                    'new_scene_type': 'suggestive',
                    'new_intimacy_level': 'suggestive',
                    'reasoning': 'nsfw2_covered_anatomy_combined_signal'
                }
            # No corroborating evidence: flag the disagreement but don't override
            return {
                'agreement': False,
                'nsfw2_verdict': nsfw2_verdict,
                'nsfw2_confidence': nsfw2_confidence,
                'override_scene_type': False,
                'new_scene_type': None,
                'new_intimacy_level': None,
                'reasoning': 'nsfw2_low_confidence_disagreement'
            }

    def analyze_content(self, image_id, image_data):
        """
        Run full semantic-spatial content analysis.

        Returns dict with complete analysis results.
        """
        try:
            captions = image_data['captions']
            nudenet_detections = image_data['nudenet_detections']
            validated_gendered_nouns = image_data.get('validated_gendered_nouns', [])

            # Extract keywords from captions
            extracted_keywords = extract_keywords_from_captions(captions)

            # Person bboxes come from YOLO results (available before harmony runs)
            person_bboxes = image_data.get('yolo_person_bboxes', [])

            # Detect person bbox containment (same person detected multiple times)
            containment_relationships = detect_person_containment(person_bboxes)

            # Deduplicate person count
            person_bboxes_raw = len(person_bboxes)
            contained_ids = set(c['contained_bbox_id'] for c in containment_relationships)
            deduplicated_person_bboxes = [p for p in person_bboxes if p['id'] not in contained_ids]
            person_bboxes_deduplicated = len(deduplicated_person_bboxes)

            # Infer gender from anatomy (spatial evidence)
            spatial_gender = infer_gender_from_anatomy(nudenet_detections)

            # Vote on gender using NudeNet, VLM captions, and SAM3-validated nouns
            gender_vote = vote_on_gender(spatial_gender, captions, validated_gendered_nouns)

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

            # NSFW2 correlation: when we classify as sfw, check NSFW2's verdict
            nsfw2_result = image_data.get('nsfw2_result')
            nsfw2_correlation = self.correlate_with_nsfw2(scene_type, nsfw2_result, nudenet_detections)

            # Override scene_type if NSFW2 disagrees — write back into activity_analysis
            # so the stored data is consistent and downstream consumers read one truth.
            if nsfw2_correlation['override_scene_type']:
                original_scene_type = scene_type
                scene_type = nsfw2_correlation['new_scene_type']
                intimacy_level = nsfw2_correlation['new_intimacy_level']
                activity_analysis['scene_type'] = scene_type
                activity_analysis['intimacy_level'] = intimacy_level
                self.logger.debug(
                    f"NSFW2 override: {original_scene_type} -> {scene_type} "
                    f"(nsfw2_confidence={nsfw2_correlation['nsfw2_confidence']:.3f})"
                )

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

            if mixed_gender:
                scene_gender_presentation = 'mixed'
                scene_gender_confidence = max(
                    gender_breakdown['confidence']['female'],
                    gender_breakdown['confidence']['male']
                )
            elif gender_vote['gender'] in ['female', 'male']:
                scene_gender_presentation = gender_vote['gender']
                scene_gender_confidence = gender_vote['confidence']
            else:
                scene_gender_presentation = 'unknown'
                scene_gender_confidence = 0.0

            scene_summary = {
                'people': person_bboxes_deduplicated,
                'gender': {
                    'presentation': scene_gender_presentation,
                    'mixed': mixed_gender,
                    'confidence': scene_gender_confidence
                }
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

            # Check for content flags based on extracted nouns
            noun_consensus_nouns = image_data.get('noun_consensus_nouns', [])
            content_flags = check_content_flags(noun_consensus_nouns, self.content_flag_config)

            # Full analysis output
            full_analysis = {
                'image_id': image_id,
                'version': ANALYSIS_VERSION,
                'timestamp': datetime.now().isoformat(),
                'category': scene_type,
                'scene': scene_summary,
                'anatomy_exposed': anatomy_exposed,
                'gender_breakdown': gender_breakdown,
                'person_attributions': person_attributions,
                'content_flags': content_flags,
                'keyword_extraction': extracted_keywords,
                'semantic_validations': semantic_validations,
                'spatial_gender_inference': spatial_gender,
                'gender_vote': gender_vote,
                'vlm_hallucinations': vlm_hallucinations,
                'activity_analysis': activity_analysis,
                'framing_analysis': framing_analysis,
                'face_correlations': face_correlations,
                'nsfw2_correlation': nsfw2_correlation,
                'person_deduplication': {
                    'raw_count': person_bboxes_raw,
                    'deduplicated_count': person_bboxes_deduplicated,
                    'containments': containment_relationships
                }
            }

            return {
                'image_id': image_id,
                'full_analysis': full_analysis,
                'analysis_version': ANALYSIS_VERSION
            }

        except Exception as e:
            self.logger.error(f"Error analyzing content for image {image_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def store_analysis(self, analysis, processing_time=None, commit=True):
        """Store content analysis in database."""
        try:
            cursor = self.db_conn.cursor()

            # Write only to canonical full_analysis schema
            cursor.execute("""
                INSERT INTO content_analysis (
                    image_id, full_analysis, analysis_version, processing_time
                ) VALUES (
                    %s, %s, %s, %s
                )
                ON CONFLICT (image_id) DO UPDATE SET
                    full_analysis = EXCLUDED.full_analysis,
                    analysis_version = EXCLUDED.analysis_version,
                    processing_time = EXCLUDED.processing_time
            """, (
                analysis['image_id'],
                json.dumps(analysis['full_analysis']),
                analysis['analysis_version'],
                processing_time
            ))

            commit_if_needed(self.db_conn, force=commit)
            close_quietly(cursor)

            # Build log message from nested full_analysis structure
            full = analysis['full_analysis']
            scene_type = full['activity_analysis']['scene_type']
            activities_count = len(full['activity_analysis']['activities'])
            people_count = full['person_deduplication']['deduplicated_count']

            nsfw2_info = ""
            nsfw2_corr = full.get('nsfw2_correlation', {})
            if nsfw2_corr.get('override_scene_type'):
                nsfw2_info = f", nsfw2_override={nsfw2_corr.get('reasoning', 'unknown')}"
            elif nsfw2_corr.get('reasoning') == 'nsfw2_low_confidence_disagreement':
                nsfw2_info = f", nsfw2_disagrees_low_conf"

            self.logger.info(f"Stored content analysis for image {analysis['image_id']}: "
                           f"scene={scene_type}, "
                           f"activities={activities_count}, "
                           f"people={people_count}{nsfw2_info}")

            return True

        except Exception as e:
            self.logger.error(f"Error storing analysis: {e}")
            rollback_quietly(self.db_conn)
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _persist_terminal_result(self, image_id, payload, processing_time, event_type, source_service, event_data, status):
        """Persist terminal content_analysis result and service event in one DB round trip."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                WITH inserted_result AS (
                    INSERT INTO results (
                        image_id, service, data, status, http_status, worker_id, processing_time
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING 1
                )
                INSERT INTO service_events (
                    image_id, service, event_type, source_service, source_stage, data
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    image_id,
                    'content_analysis',
                    json.dumps(payload),
                    status,
                    self._extract_http_status(payload),
                    self.worker_id,
                    processing_time,
                    image_id,
                    'content_analysis',
                    event_type,
                    source_service,
                    'content_analysis_run',
                    json.dumps(event_data),
                ),
            )
            commit_if_needed(self.db_conn, force=True)
            close_quietly(cursor)
        except Exception as e:
            self.logger.error(
                f"content_analysis: failed to persist terminal result for image {image_id}: {e}"
            )
            raise

    def _persist_successful_analysis(self, analysis, processing_time, input_fingerprint, source_service, services_present, tier):
        """Persist canonical analysis row, terminal result, and completed event in one DB round trip."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                WITH upserted_analysis AS (
                    INSERT INTO content_analysis (
                        image_id, full_analysis, analysis_version, processing_time
                    ) VALUES (
                        %s, %s, %s, %s
                    )
                    ON CONFLICT (image_id) DO UPDATE SET
                        full_analysis = EXCLUDED.full_analysis,
                        analysis_version = EXCLUDED.analysis_version,
                        processing_time = EXCLUDED.processing_time
                    RETURNING 1
                ),
                inserted_result AS (
                    INSERT INTO results (
                        image_id, service, data, status, http_status, worker_id, processing_time
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING 1
                )
                INSERT INTO service_events (
                    image_id, service, event_type, source_service, source_stage, data
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    analysis['image_id'],
                    json.dumps(analysis['full_analysis']),
                    analysis['analysis_version'],
                    processing_time,
                    analysis['image_id'],
                    'content_analysis',
                    json.dumps({
                        'service': 'content_analysis',
                        'status': 'success',
                        'full_analysis': analysis['full_analysis'],
                        'analysis_version': analysis['analysis_version'],
                        'metadata': {
                            'input_fingerprint': input_fingerprint,
                        },
                    }),
                    'success',
                    None,
                    self.worker_id,
                    processing_time,
                    analysis['image_id'],
                    'content_analysis',
                    'completed',
                    source_service,
                    'content_analysis_run',
                    json.dumps({
                        'input_fingerprint': input_fingerprint,
                        'services_present': services_present,
                        'tier': tier,
                    }),
                ),
            )
            commit_if_needed(self.db_conn, force=True)
            close_quietly(cursor)

            full = analysis['full_analysis']
            scene_type = full['activity_analysis']['scene_type']
            activities_count = len(full['activity_analysis']['activities'])
            people_count = full['person_deduplication']['deduplicated_count']

            nsfw2_info = ""
            nsfw2_corr = full.get('nsfw2_correlation', {})
            if nsfw2_corr.get('override_scene_type'):
                nsfw2_info = f", nsfw2_override={nsfw2_corr.get('reasoning', 'unknown')}"
            elif nsfw2_corr.get('reasoning') == 'nsfw2_low_confidence_disagreement':
                nsfw2_info = ", nsfw2_disagrees_low_conf"

            self.logger.info(
                f"Stored content analysis for image {analysis['image_id']}: "
                f"scene={scene_type}, activities={activities_count}, people={people_count}{nsfw2_info}"
            )
        except Exception as e:
            self.logger.error(
                f"content_analysis: failed to persist successful analysis for image {analysis['image_id']}: {e}"
            )
            raise

    def process_message(self, ch, method, properties, body):
        """Process content analysis message"""
        try:
            timing = {}
            # Ensure database connection is healthy before processing
            if not self.ensure_database_connection():
                self.logger.error(
                    "Database connection unavailable, requeueing message. "
                    "Worker will retry after backoff delay."
                )
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                self.job_failed("Database unavailable")
                return

            # Parse message
            message = json.loads(body)
            image_id = message['image_id']
            trace_id = message.get('trace_id')
            services_present = sorted(message.get('services_present') or [])
            tier = message.get('tier')

            if trace_id:
                self.logger.debug(f"[{trace_id}] Processing content analysis for image {image_id}")
            else:
                self.logger.debug(f"Processing content analysis for image {image_id}")

            start_time = time.time()

            # Fetch all image data
            t0 = time.time()
            image_data = self.get_image_data(image_id)
            timing['get_image_data'] = time.time() - t0
            if not image_data:
                self.logger.error(f"Failed to fetch image data for {image_id}")
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                self.job_failed("Failed to fetch image data")
                return

            t0 = time.time()
            input_fingerprint = self._build_input_fingerprint(image_data)
            timing['build_input_fingerprint'] = time.time() - t0
            t0 = time.time()
            previous_fingerprint = self._latest_input_fingerprint(image_id)
            timing['fetch_previous_fingerprint'] = time.time() - t0
            if input_fingerprint and previous_fingerprint == input_fingerprint:
                self.logger.info(
                    f"Skipping unchanged content analysis for image {image_id} "
                    f"(fingerprint={input_fingerprint[:12]})"
                )
                self._record_service_event(
                    image_id=image_id,
                    service='content_analysis',
                    event_type='completed',
                    source_service=message.get('triggered_by'),
                    source_stage='content_analysis_run',
                    data={
                        'reason': 'unchanged_input',
                        'input_fingerprint': input_fingerprint,
                        'services_present': services_present,
                        'tier': tier,
                    },
                    commit=True,
                )
                self._safe_ack(ch, method.delivery_tag)
                self.job_completed_successfully()
                return

            # Run content analysis
            t0 = time.time()
            analysis = self.analyze_content(image_id, image_data)
            timing['analyze_content'] = time.time() - t0
            if not analysis:
                self._persist_terminal_result(
                    image_id=image_id,
                    payload={
                        'service': 'content_analysis',
                        'status': 'failed',
                        'full_analysis': None,
                        'error_message': 'Content analysis returned no terminal analysis payload',
                    },
                    processing_time=round(time.time() - start_time, 3),
                    event_type='failed',
                    source_service=message.get('triggered_by'),
                    event_data={'error_message': 'Content analysis returned no terminal analysis payload'},
                    status='failed',
                )
                self.logger.error(f"Failed to analyze content for {image_id}")
                self._safe_ack(ch, method.delivery_tag)
                self.job_completed_successfully()
                return

            # Store analysis
            t0 = time.time()
            self._persist_successful_analysis(
                analysis=analysis,
                processing_time=round(time.time() - start_time, 3),
                input_fingerprint=input_fingerprint,
                source_service=message.get('triggered_by'),
                services_present=services_present,
                tier=tier,
            )
            timing['persist_success'] = time.time() - t0

            # Acknowledge message
            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()

            total_duration = time.time() - start_time
            slow_bits = " ".join(
                f"{name}={duration:.3f}s"
                for name, duration in timing.items()
                if duration >= 0.05
            )
            self.logger.info(
                f"content_analysis timing image={image_id} total={total_duration:.3f}s "
                f"{slow_bits}".rstrip()
            )
            self.logger.info(f"Successfully completed content analysis for image {image_id}")

        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            # Database connection errors should be retried after reconnect.
            self.logger.error(f"Database error processing content analysis message: {e}")
            self.logger.warning("Requeueing message after database error")
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self.job_failed(str(e))
        except Exception as e:
            # Other errors - requeue for retry
            self.logger.error(f"Error processing content analysis message: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self.job_failed(str(e))


if __name__ == "__main__":
    worker = ContentAnalysisWorker()
    worker.start()
