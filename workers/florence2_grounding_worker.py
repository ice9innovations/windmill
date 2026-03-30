#!/usr/bin/env python3
"""
Florence2GroundingWorker - Phrase grounding triggered by noun consensus.

Receives image data + consensus nouns from noun_consensus_worker, constructs
a comma-separated noun prompt, calls Florence-2 CAPTION_TO_PHRASE_GROUNDING,
stores bbox results in the results table, and writes grounding_validated=true
on matched nouns in noun_consensus.

Dedup: skips re-running if the noun set is identical to the last grounding
call for this image (noun_consensus triggers progressively as VLMs complete).

Pattern mirrors sam3_worker: triggered internally, not by the producer.
"""

import io
import json
import os
import sys
import time
import requests
from datetime import datetime
from PIL import Image

sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker
from core.image_store import is_valkey_image_store_enabled

MIN_COLORS_SIZE = (8, 8)


class Florence2GroundingWorker(BaseWorker):
    """Phrase grounding via Florence-2 CAPTION_TO_PHRASE_GROUNDING."""

    def __init__(self):
        super().__init__('system.florence2_grounding')


    def _declare_additional_queues(self, declare_with_dlq):
        """No additional queues."""
        return

    def _crop_bbox_from_image_data(self, image_bytes: bytes, bbox: dict):
        try:
            img = Image.open(io.BytesIO(image_bytes))
            with img:
                crop_box = (
                    bbox['x'],
                    bbox['y'],
                    bbox['x'] + bbox['width'],
                    bbox['y'] + bbox['height'],
                )
                cropped_img = img.crop(crop_box)
                if cropped_img.mode not in ('RGB', 'L'):
                    cropped_img = cropped_img.convert('RGB')
                img_buffer = io.BytesIO()
                cropped_img.save(img_buffer, format='JPEG', quality=90)
                img_buffer.seek(0)
                return img_buffer.getvalue()
        except Exception as e:
            self.logger.error(f"florence2_grounding: crop failed: {e}")
            return None

    def process_message(self, ch, method, properties, body):
        start_time = time.time()
        previous_autocommit = None
        stage_timings = {}

        try:
            if not self.ensure_database_connection():
                self._safe_nack(ch, method.delivery_tag, requeue=False)
                self.job_failed("Database unavailable")
                return

            message = json.loads(body)
            image_id = message['image_id']
            tier = message.get('tier', 'free')
            has_image_ref = bool(message.get('image_ref'))
            has_image_data = bool(message.get('image_data'))

            if is_valkey_image_store_enabled():
                if has_image_data and not has_image_ref:
                    self.logger.warning(
                        f"florence2_grounding: image {image_id} arrived with inline image_data "
                        f"while IMAGE_STORE_MODE=valkey"
                    )
                else:
                    self.logger.info(
                        f"florence2_grounding: image {image_id} transport "
                        f"image_ref={has_image_ref} image_data={has_image_data}"
                    )

            t0 = time.time()
            image_bytes = self.resolve_image_bytes(message, required=True)
            stage_timings['resolve_image'] = time.time() - t0

            if not image_bytes:
                self.logger.error(
                    f"florence2_grounding: no image bytes in message for image {image_id}"
                )
                self._safe_nack(ch, method.delivery_tag, requeue=False)
                self.job_failed("No image bytes")
                return

            # Read nouns from noun_consensus at processing time, not from the
            # message. By the time this job is picked up, more VLMs will have
            # reported than when the dispatch was created, so the DB gives a
            # fresher (and usually complete) noun set. This also means we never
            # run with more nouns than we need to — the threshold is re-applied
            # against the tier's expected VLM count at this moment.
            t0 = time.time()
            nouns = self._fetch_consensus_nouns(image_id, tier)
            stage_timings['fetch_nouns'] = time.time() - t0

            if not nouns:
                self.logger.info(
                    f"florence2_grounding: no consensus nouns for image {image_id}, skipping"
                )
                self._record_service_event(
                    image_id=image_id,
                    service='florence2_grounding',
                    event_type='completed',
                    source_service='noun_consensus',
                    source_stage='grounding_run',
                    data={'reason': 'No consensus nouns available at processing time'},
                    commit=True,
                )
                self._safe_ack(ch, method.delivery_tag)
                return

            # Dedup: skip if already grounded with this exact noun set
            previous_result = self._fetch_previous_grounding_result(image_id)
            previous_nouns = previous_result['nouns']
            if set(nouns) == previous_nouns:
                self.logger.info(
                    f"florence2_grounding: image {image_id} already grounded "
                    f"with same {len(nouns)} nouns, skipping model run"
                )
                previous_predictions = previous_result['predictions']
                self._record_service_event(
                    image_id=image_id,
                    service='florence2_grounding',
                    event_type='completed',
                    source_service='noun_consensus',
                    source_stage='grounding_reuse',
                    data={
                        'reused_previous_result': True,
                        'prediction_count': len(previous_predictions),
                    },
                    commit=True,
                )
                self._safe_ack(ch, method.delivery_tag)
                return

            prompt = self._build_grounding_prompt(nouns)
            self.logger.info(
                f"florence2_grounding: image {image_id} — "
                f"{len(nouns)} nouns, prompt: \"{prompt}\""
            )

            t0 = time.time()
            result = self._call_florence2_grounding(image_bytes, prompt)
            stage_timings['call_florence'] = time.time() - t0
            if result is None:
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                self.job_failed("Florence-2 grounding call failed")
                return

            result_status = result.get('status', 'success') if isinstance(result, dict) else 'success'

            processing_time = round(time.time() - start_time, 3)

            if result_status != 'success':
                previous_autocommit = self.db_conn.autocommit
                self.db_conn.autocommit = False
                t0 = time.time()
                self._store_result(image_id, result, processing_time, commit=False)
                stage_timings['store_result'] = time.time() - t0
                t0 = time.time()
                self._record_service_event(
                    image_id=image_id,
                    service='florence2_grounding',
                    event_type='failed',
                    source_service='noun_consensus',
                    source_stage='grounding_run',
                    data={
                        'error_message': result.get('error') or result.get('message'),
                    },
                    commit=False,
                )
                stage_timings['record_failure_event'] = time.time() - t0
                t0 = time.time()
                self.db_conn.commit()
                stage_timings['db_commit'] = time.time() - t0
                self._safe_ack(ch, method.delivery_tag)
                self.job_completed_successfully()
                self.logger.warning(
                    f"florence2_grounding: image {image_id} returned terminal non-success: "
                    f"{result.get('error') or result.get('message') or 'no reason provided'}"
                )
                return

            # Embed the noun set in metadata so dedup can read it back next trigger.
            predictions = result.get('predictions', [])

            # Florence-2 occasionally returns inverted bbox coordinates (y2 < y1 in
            # raw [x1,y1,x2,y2] space). REST.py converts to [x, y, w, h], so an
            # inverted axis surfaces as a negative w or h. Correct by adjusting the
            # origin and taking the absolute value of the dimension.
            for p in predictions:
                bbox = p.get('bbox')
                if bbox and len(bbox) >= 4:
                    x, y, w, h = bbox[:4]
                    if w < 0:
                        x += w
                        w = -w
                    if h < 0:
                        y += h
                        h = -h
                    p['bbox'] = [x, y, w, h]

            result.setdefault('metadata', {})['nouns_queried'] = json.dumps(nouns)
            result['metadata']['colors_post_dispatched'] = 0

            previous_autocommit = self.db_conn.autocommit
            self.db_conn.autocommit = False
            t0 = time.time()
            self._store_result(image_id, result, processing_time, commit=False)
            stage_timings['store_result'] = time.time() - t0

            # Mark nouns that were successfully grounded back in noun_consensus.
            # Must happen before ACK so the grounding_validated backfill is
            # committed before the NOTIFY fires and the SSE reads results.
            grounded_labels = [
                p['label'].lower().strip()
                for p in result.get('predictions', [])
                if p.get('label')
            ]
            if grounded_labels:
                t0 = time.time()
                self._validate_noun_consensus(image_id, nouns, grounded_labels, commit=False)
                stage_timings['validate_nouns'] = time.time() - t0

            t0 = time.time()
            self._record_service_event(
                image_id=image_id,
                service='florence2_grounding',
                event_type='completed',
                source_service='noun_consensus',
                source_stage='grounding_run',
                data={
                    'prediction_count': len(result.get('predictions', [])),
                    'nouns_queried': nouns,
                },
                commit=False,
            )
            stage_timings['record_completed_event'] = time.time() - t0
            t0 = time.time()
            self.db_conn.commit()
            stage_timings['db_commit'] = time.time() - t0
            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()

            timing_bits = " ".join(
                f"{name}={duration:.3f}s" for name, duration in stage_timings.items()
            )
            self.logger.info(
                f"florence2_grounding: image {image_id} stage timings {timing_bits}"
            )
            self.logger.info(
                f"florence2_grounding: image {image_id} — "
                f"{len(result.get('predictions', []))} regions grounded "
                f"from {len(nouns)} nouns in {processing_time}s"
            )

        except Exception as e:
            if self.db_conn and previous_autocommit is False:
                try:
                    self.db_conn.rollback()
                except Exception:
                    pass
            self.logger.error(f"florence2_grounding: error processing message: {e}")
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self.job_failed(str(e))
        finally:
            if self.db_conn and previous_autocommit is not None:
                try:
                    self.db_conn.autocommit = previous_autocommit
                except Exception:
                    pass

    def _build_grounding_prompt(self, nouns: list) -> str:
        """Build a comma-separated noun prompt for CAPTION_TO_PHRASE_GROUNDING.

        Using a bare noun list rather than "an image containing ..." avoids
        Florence-2 grounding the intro phrase itself as a spurious full-image bbox.
        """
        if not nouns:
            return ""
        if len(nouns) == 1:
            return nouns[0]
        if len(nouns) == 2:
            return f"{nouns[0]} and {nouns[1]}"
        return f"{', '.join(nouns[:-1])}, and {nouns[-1]}"

    def _fetch_consensus_nouns(self, image_id: int, tier: str) -> list:
        """Return consensus nouns from noun_consensus at processing time.

        Reads the current DB state rather than using nouns baked into the queue
        message. Applies the same majority-vote threshold as noun_consensus_worker:
        min_votes = max(2, ceil(tier_vlm_count / 2)), where tier_vlm_count is the
        total VLMs expected for the tier — not the number that have reported so far.
        This keeps the bar stable across progressive triggers.
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                "SELECT nouns FROM noun_consensus WHERE image_id = %s",
                (image_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            if not row or not row[0]:
                return []
            nouns_data = row[0]  # jsonb — already a list
            vlm_names = set(self.config.get_vlm_service_names())
            tier_vlm_count = len([
                name for name in self.config.get_services_by_tier(tier)
                if name.startswith('primary.')
                and name.split('.', 1)[1] in vlm_names
            ]) or 1
            min_votes = max(2, (tier_vlm_count + 1) // 2)
            return [
                n['canonical'] for n in nouns_data
                if n.get('vote_count', 0) >= min_votes
            ]
        except Exception as e:
            self.logger.warning(
                f"florence2_grounding: could not fetch consensus nouns for image {image_id}: {e}"
            )
            return []

    def _fetch_previous_grounding_result(self, image_id: int) -> dict:
        """Return noun set and predictions from the most recent grounding result.

        Returns {'nouns': set(), 'predictions': []} if no prior result exists.
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                SELECT data->'metadata'->>'nouns_queried', data->'predictions'
                FROM results
                WHERE image_id = %s AND service = 'florence2_grounding'
                ORDER BY result_created DESC
                LIMIT 1
                """,
                (image_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            if not row or not row[0]:
                return {'nouns': set(), 'predictions': []}
            predictions = row[1] if len(row) > 1 and row[1] else []
            return {
                'nouns': set(json.loads(row[0])),
                'predictions': predictions,
            }
        except Exception as e:
            self.logger.warning(
                f"florence2_grounding: could not fetch previous grounding result for image {image_id}: {e}"
            )
            return {'nouns': set(), 'predictions': []}

    def _call_florence2_grounding(self, image_bytes: bytes, prompt: str):
        """POST image + prompt to Florence-2 CAPTION_TO_PHRASE_GROUNDING."""
        try:
            files = {'file': ('image.jpg', io.BytesIO(image_bytes), 'image/jpeg')}
            url = f"http://{self.service_host}:{self.service_port}/v3/analyze"
            resp = requests.post(
                url,
                params={'task': 'CAPTION_TO_PHRASE_GROUNDING', 'text': prompt},
                files=files,
                timeout=self.request_timeout,
            )
            return self._coerce_terminal_http_response(resp)
        except requests.RequestException as e:
            if getattr(e, 'response', None) is not None:
                return self._coerce_terminal_http_response(e.response)
            self.logger.error(f"florence2_grounding: REST call failed before terminal response: {e}")
            return None

    def _store_result(self, image_id: int, data: dict, processing_time: float, commit: bool = True):
        """Insert Florence-2 grounding result into the results table."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO results
                    (image_id, service, data, status, worker_id, processing_time)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    image_id,
                    self._get_clean_service_name(),
                    json.dumps(data),
                    data.get('status', 'success') or 'success',
                    self.worker_id,
                    processing_time,
                ),
            )
            if commit:
                self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(
                f"florence2_grounding: failed to store result for image {image_id}: {e}"
            )
            raise

    def _validate_noun_consensus(
        self, image_id: int, queried_nouns: list, grounded_labels: list, commit: bool = True
    ):
        """Set grounding_validated=true on noun_consensus nouns that Florence-2 grounded.

        A noun is validated if its canonical form appears (case-insensitive) in
        the grounded label set returned by Florence-2. Nouns not found are left
        untagged — no invalidation.
        """
        grounded_set = set(grounded_labels)

        # Match each queried noun against grounded labels
        validated = [n for n in queried_nouns if n.lower() in grounded_set]
        if not validated:
            self.logger.info(
                f"florence2_grounding: no noun matches found for image {image_id} "
                f"(queried: {queried_nouns}, grounded: {list(grounded_set)})"
            )
            return

        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                UPDATE noun_consensus
                SET nouns = (
                    SELECT jsonb_agg(
                        CASE
                            WHEN n->>'canonical' = ANY(%s)
                            THEN n || '{"grounding_validated": true}'::jsonb
                            ELSE n
                        END
                    )
                    FROM jsonb_array_elements(nouns) n
                ),
                updated_at = NOW()
                WHERE image_id = %s
                """,
                (validated, image_id),
            )
            if commit:
                self.db_conn.commit()
            cursor.close()
            self.logger.info(
                f"florence2_grounding: validated {len(validated)} nouns "
                f"for image {image_id}: {validated}"
            )
        except Exception as e:
            self.logger.error(
                f"florence2_grounding: failed to write validation "
                f"for image {image_id}: {e}"
            )


if __name__ == '__main__':
    worker = Florence2GroundingWorker()
    worker.start()
