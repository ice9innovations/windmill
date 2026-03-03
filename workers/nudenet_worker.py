#!/usr/bin/env python3
"""
NudenetWorker — NudeNet ML service worker with inline spatial analysis.

After storing the NudeNet result, runs the NudeNet-only subset of content
analysis immediately (nudenet++), without a separate queue hop. This is
valid because all the required input data is either in the service result
or derivable from the image that's already in memory.

Pass 2 enrichment (VLM corroboration, face correlation, nsfw2) runs in
content_analysis_worker after consensus, overwriting this row via
ON CONFLICT DO UPDATE.

Architectural principle: if the data lives in service X, the extrapolation
from that data lives in service X. A separate worker is only warranted when
correlating across multiple services.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import base64
import io
from PIL import Image

from base_worker import BaseWorker
from utils.semantic_validation import infer_gender_from_anatomy
from utils.spatial_analysis import detect_person_containment, detect_sexual_activities
from utils.framing_analysis import classify_framing

SPATIAL_ANALYSIS_VERSION = 'spatial_1.0'

# Subsecond timeout — YOLO data is opportunistic. If it's not in the DB
# within this window we proceed without person boxes. No performance penalty
# compared to the status quo (person_bboxes was always [] before).
YOLO_PERSON_TIMEOUT = 0.5   # seconds
YOLO_PERSON_POLL    = 0.05  # seconds between polls

# Person emojis that harmony groups under 🧑
_PERSON_EMOJIS = frozenset(['🧑', '👩', '🧒'])


class NudenetWorker(BaseWorker):
    """NudeNet ML service worker with inline spatial analysis (nudenet++)."""

    def __init__(self):
        super().__init__('primary.nudenet')

    def after_result_stored(self, image_id, result, message):
        """Run spatial content analysis immediately after the NudeNet result is stored."""
        try:
            nudenet_detections = self._extract_detections(result)
            if not nudenet_detections:
                # No detections — write a clean SFW record so Pass 2 has a row to update
                self._store_spatial_analysis_dict({
                    'image_id': image_id,
                    'gender_breakdown': {
                        'female_nudity': False, 'male_nudity': False, 'mixed_gender': False,
                        'confidence': {'female': 0.0, 'male': 0.0},
                        'vote_details': None, 'reasoning': 'no_detections',
                    },
                    'anatomy_exposed': [],
                    'scene_type': 'sfw',
                    'intimacy_level': 'none',
                    'activities_detected': [],
                    'spatial_relationships': [],
                    'person_bboxes_raw': 0,
                    'person_bboxes_deduplicated': 0,
                    'containment_relationships': [],
                    'people_count': 0,
                    'framing_analysis': {},
                    'full_analysis': {
                        'pass': SPATIAL_ANALYSIS_VERSION,
                        'spatial_gender_inference': {'gender': 'unknown', 'confidence': 0.0, 'reasoning': 'no_detections'},
                        'activity_analysis': {'scene_type': 'sfw', 'intimacy_level': 'none', 'activities': [], 'spatial_relationships': []},
                        'framing_analysis': {},
                        'person_deduplication': {'raw_count': 0, 'deduplicated_count': 0, 'containments': []},
                    },
                })
                return

            image_width, image_height = self._image_dimensions(message['image_data'])

            # Best-effort: wait up to YOLO_PERSON_TIMEOUT for YOLO person boxes.
            # If YOLO isn't done yet, person_bboxes stays [] and people_count stays 0.
            # No functional regression — these fields were always empty before.
            person_bboxes, containment_relationships, person_bboxes_deduplicated = \
                self._fetch_yolo_persons_with_dedup(image_id)

            spatial_gender = infer_gender_from_anatomy(nudenet_detections)
            anatomy_labels = [d['label'] for d in nudenet_detections]
            anatomy_exposed = list(set(anatomy_labels))
            female_nudity = any('FEMALE' in label for label in anatomy_labels)
            male_nudity = any('MALE_GENITALIA' in label for label in anatomy_labels)

            gender_breakdown = {
                'female_nudity': female_nudity,
                'male_nudity': male_nudity,
                'mixed_gender': female_nudity and male_nudity,
                'confidence': {
                    'female': spatial_gender['confidence'] if spatial_gender['gender'] == 'female' else 0.0,
                    'male': spatial_gender['confidence'] if spatial_gender['gender'] == 'male' else 0.0,
                },
                'vote_details': None,
                'reasoning': spatial_gender['reasoning']
            }

            # No VLM captions yet — pass empty string; activity detection is
            # spatial only for this pass.
            activity_analysis = detect_sexual_activities(
                nudenet_detections,
                person_bboxes_deduplicated,
                ''
            )

            framing_analysis = classify_framing(
                nudenet_detections,
                person_bboxes_deduplicated,
                image_width,
                image_height
            )

            analysis = {
                'image_id': image_id,
                'gender_breakdown': gender_breakdown,
                'anatomy_exposed': anatomy_exposed,
                'scene_type': activity_analysis['scene_type'],
                'intimacy_level': activity_analysis['intimacy_level'],
                'activities_detected': activity_analysis['activities'],
                'spatial_relationships': activity_analysis['spatial_relationships'],
                'person_bboxes_raw': len(person_bboxes),
                'person_bboxes_deduplicated': len(person_bboxes_deduplicated),
                'containment_relationships': containment_relationships,
                'people_count': len(person_bboxes_deduplicated),
                'framing_analysis': framing_analysis,
                'full_analysis': {
                    'pass': SPATIAL_ANALYSIS_VERSION,
                    'spatial_gender_inference': spatial_gender,
                    'activity_analysis': activity_analysis,
                    'framing_analysis': framing_analysis,
                    'person_deduplication': {
                        'raw_count': len(person_bboxes),
                        'deduplicated_count': len(person_bboxes_deduplicated),
                        'containments': containment_relationships,
                    },
                },
            }

            self._store_spatial_analysis_dict(analysis)

        except Exception as e:
            # Never fail the NudeNet job over analysis errors — log and move on
            self.logger.error(f"Inline spatial analysis failed for image {image_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_detections(self, result):
        detections = []
        for pred in result.get('predictions', []):
            if pred.get('label') and pred.get('bbox'):
                detections.append({
                    'label': pred['label'],
                    'bbox': pred['bbox'],
                    'confidence': pred.get('confidence', 0.0)
                })
        return detections

    def _image_dimensions(self, image_data_b64):
        """Derive width/height from the base64 image already in memory."""
        img_bytes = base64.b64decode(image_data_b64)
        with Image.open(io.BytesIO(img_bytes)) as img:
            return img.size  # (width, height)

    def _fetch_yolo_persons_with_dedup(self, image_id):
        """Poll for YOLO person bboxes, deduplicate, and return (raw, containments, deduped).

        Returns empty lists immediately if YOLO doesn't respond within YOLO_PERSON_TIMEOUT.
        Person bboxes are opportunistic — no functional regression if unavailable.
        """
        deadline = time.time() + YOLO_PERSON_TIMEOUT
        while time.time() < deadline:
            person_bboxes = self._fetch_yolo_person_bboxes(image_id)
            if person_bboxes is not None:
                containment = detect_person_containment(person_bboxes)
                contained_ids = {c['contained_bbox_id'] for c in containment}
                deduped = [p for p in person_bboxes if p['id'] not in contained_ids]
                self.logger.debug(
                    f"YOLO persons for image {image_id}: "
                    f"raw={len(person_bboxes)}, deduped={len(deduped)}"
                )
                return person_bboxes, containment, deduped
            time.sleep(YOLO_PERSON_POLL)

        self.logger.debug(f"YOLO result not available within {YOLO_PERSON_TIMEOUT}s for image {image_id} — proceeding without person boxes")
        return [], [], []

    def _fetch_yolo_person_bboxes(self, image_id):
        """Return person bboxes from YOLO result, or None if not yet in DB.

        Each bbox is {id, bbox: {x, y, width, height}} for compatibility with
        detect_person_containment() and detect_sexual_activities().
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT data FROM results
                WHERE image_id = %s AND service = 'yolo_v8' AND status = 'success'
                LIMIT 1
            """, (image_id,))
            row = cursor.fetchone()
            cursor.close()
        except Exception as e:
            self.logger.warning(f"Error fetching YOLO result for image {image_id}: {e}")
            return None

        if row is None:
            return None

        person_bboxes = []
        for i, pred in enumerate(row[0].get('predictions', [])):
            # Strip variation selectors; group 🧑/👩/🧒 under 🧑
            raw_emoji = pred.get('emoji', '')
            clean_emoji = ''.join(c for c in raw_emoji if ord(c) < 0xFE00 or ord(c) > 0xFE0F)
            if clean_emoji not in _PERSON_EMOJIS:
                continue
            raw_bbox = pred.get('bbox')
            if not raw_bbox:
                continue
            # Normalise array [x, y, w, h] → dict
            if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
                bbox = {'x': raw_bbox[0], 'y': raw_bbox[1], 'width': raw_bbox[2], 'height': raw_bbox[3]}
            elif isinstance(raw_bbox, dict) and all(k in raw_bbox for k in ('x', 'y', 'width', 'height')):
                bbox = raw_bbox
            else:
                continue
            person_bboxes.append({'id': i, 'bbox': bbox})

        return person_bboxes

    def _store_spatial_analysis_dict(self, analysis):
        """Write Pass 1 result to content_analysis.

        The WHERE guard on ON CONFLICT prevents overwriting a completed Pass 2
        result if a retry races with it.
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO content_analysis (
                    image_id, gender_breakdown, anatomy_exposed, scene_type, intimacy_level,
                    activities_detected, spatial_relationships, person_bboxes_raw,
                    person_bboxes_deduplicated, containment_relationships, semantic_validation,
                    vlm_hallucinations, people_count, person_attributions, framing_analysis,
                    face_correlations, nsfw2_correlation, full_analysis, analysis_version
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (image_id) DO UPDATE SET
                    gender_breakdown           = EXCLUDED.gender_breakdown,
                    anatomy_exposed            = EXCLUDED.anatomy_exposed,
                    scene_type                 = EXCLUDED.scene_type,
                    intimacy_level             = EXCLUDED.intimacy_level,
                    activities_detected        = EXCLUDED.activities_detected,
                    spatial_relationships      = EXCLUDED.spatial_relationships,
                    person_bboxes_raw          = EXCLUDED.person_bboxes_raw,
                    person_bboxes_deduplicated = EXCLUDED.person_bboxes_deduplicated,
                    containment_relationships  = EXCLUDED.containment_relationships,
                    semantic_validation        = EXCLUDED.semantic_validation,
                    vlm_hallucinations         = EXCLUDED.vlm_hallucinations,
                    people_count               = EXCLUDED.people_count,
                    person_attributions        = EXCLUDED.person_attributions,
                    framing_analysis           = EXCLUDED.framing_analysis,
                    face_correlations          = EXCLUDED.face_correlations,
                    nsfw2_correlation          = EXCLUDED.nsfw2_correlation,
                    full_analysis              = EXCLUDED.full_analysis,
                    analysis_version           = EXCLUDED.analysis_version
                WHERE content_analysis.analysis_version = 'spatial_1.0'
                   OR content_analysis.analysis_version IS NULL
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
                json.dumps({'corroborated': False, 'conflicts': [], 'confidence': 0.0}),
                json.dumps({}),
                analysis['people_count'],
                json.dumps([]),
                json.dumps(analysis['framing_analysis']),
                json.dumps({}),
                json.dumps({'agreement': None, 'nsfw2_verdict': 'unknown',
                            'nsfw2_confidence': 0.0, 'override_scene_type': False,
                            'new_scene_type': None, 'new_intimacy_level': None,
                            'reasoning': 'no_nsfw2_data'}),
                json.dumps(analysis['full_analysis']),
                SPATIAL_ANALYSIS_VERSION
            ))
            self.db_conn.commit()
            cursor.close()

            self.logger.info(
                f"Spatial pass stored for image {analysis['image_id']}: "
                f"scene={analysis['scene_type']}, "
                f"activities={len(analysis['activities_detected'])}, "
                f"people={analysis['people_count']}"
            )
        except Exception as e:
            self.logger.error(f"Failed to store spatial analysis for image {analysis['image_id']}: {e}")
            self.db_conn.rollback()
            raise


if __name__ == "__main__":
    worker = NudenetWorker()
    worker.start()
