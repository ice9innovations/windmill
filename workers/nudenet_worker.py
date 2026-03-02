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
import base64
import io
from PIL import Image

from base_worker import BaseWorker
from utils.semantic_validation import infer_gender_from_anatomy
from utils.spatial_analysis import detect_person_containment, detect_sexual_activities
from utils.framing_analysis import classify_framing

SPATIAL_ANALYSIS_VERSION = 'spatial_1.0'


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
                self._store_spatial_analysis(image_id, nudenet_detections, [], 0, 0)
                return

            image_width, image_height = self._image_dimensions(message['image_data'])
            merged_boxes = self._fetch_merged_boxes(image_id)

            person_bboxes, containment_relationships, person_bboxes_deduplicated = \
                self._deduplicate_persons(merged_boxes)

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
            # spatial only for this pass
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
                'person_bboxes_raw': len(merged_boxes),
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
                        'raw_count': len(merged_boxes),
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

    def _fetch_merged_boxes(self, image_id):
        """Query merged_boxes for person bboxes. Likely empty at this point;
        Pass 2 refines with the complete merged set."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT merged_id, merged_data
                FROM merged_boxes
                WHERE image_id = %s
            """, (image_id,))
            rows = cursor.fetchall()
            cursor.close()
            return [{'merged_id': r[0], 'data': r[1]} for r in rows]
        except Exception as e:
            self.logger.warning(f"Could not fetch merged_boxes for image {image_id}: {e}")
            return []

    def _deduplicate_persons(self, merged_boxes):
        person_bboxes = [
            {'id': mb['merged_id'], 'bbox': mb['data']['merged_bbox']}
            for mb in merged_boxes
            if mb['data'].get('emoji') == '🧑' and mb['data'].get('merged_bbox')
        ]
        containment = detect_person_containment(person_bboxes)
        contained_ids = {c['contained_bbox_id'] for c in containment}
        deduped = [p for p in person_bboxes if p['id'] not in contained_ids]
        return person_bboxes, containment, deduped

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
