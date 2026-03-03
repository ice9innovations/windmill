#!/usr/bin/env python3
"""
CaptionSummaryWorker - Synthesize VLM captions into a single description.

Triggered by SAM3 after it has finalized noun consensus for an image.
By that point all VLMs that will return for this image have returned,
and SAM3-validated noun data is available.

Collects:
  - All available VLM captions (from results table)
  - Noun consensus with SAM3-validated nouns and vote counts
  - Verb consensus with vote counts

Sends captions and consensus data to the caption-synthesis Animal Farm
service, which synthesizes them into a single sentence and returns it.
Writes the result to the caption_summary table.

Skipped entirely if fewer than 2 VLM captions are present — no value
in synthesizing a single caption.
"""

import base64
import io
import os
import sys
import json
import time
import logging
import requests
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

logger = logging.getLogger(__name__)

VLM_SERVICES = ['blip', 'gemini', 'gpt_nano', 'haiku', 'moondream', 'ollama', 'qwen']

MIN_CAPTIONS = 2  # Skip synthesis if fewer than this many VLMs returned


def _extract_caption(data: dict) -> str:
    """
    Extract caption text from a VLM result data dict.

    BLIP and moondream store it at data['caption'].
    Ollama, haiku, and qwen store it at data['predictions'][0]['text'].
    """
    if 'caption' in data:
        return str(data['caption']).strip()
    predictions = data.get('predictions', [])
    if predictions and isinstance(predictions[0], dict):
        return str(predictions[0].get('text', '')).strip()
    return ''


class CaptionSummaryWorker(BaseWorker):
    """
    Synthesizes VLM captions into a single sentence via the caption-synthesis
    Animal Farm service. Writes the result to the caption_summary table.
    """

    def __init__(self):
        super().__init__('system.caption_summary')

        # Caption summary AF service endpoint
        synth_cfg = self.config.get_service_config('system.caption_summary')
        self.synthesis_url = (
            f"http://{synth_cfg['host']}:{synth_cfg['port']}{synth_cfg['endpoint']}"
        )

        # CLIP score endpoint — same resolution pattern as caption_score_worker
        clip_services = self.config.get_service_group('postprocessing.clip_score[]')
        clip_cfg = self.config.get_service_config(clip_services[0])
        self.clip_score_url = (
            f"http://{clip_cfg['host']}:{clip_cfg['port']}{clip_cfg['endpoint']}"
        )

    def process_message(self, ch, method, properties, body):
        """Override base process_message — DB fetch + caption-synthesis service call."""
        start_time = time.time()

        try:
            if not self.ensure_database_connection():
                self._safe_nack(ch, method.delivery_tag, requeue=False)
                self.job_failed("Database unavailable")
                return

            message = json.loads(body)
            image_id = message['image_id']
            image_data = message.get('image_data')  # base64; may be None for legacy messages

            self.logger.debug(f"caption_summary: processing image {image_id}")

            # Dedup guard — only one synthesis per image
            cursor = self.db_conn.cursor()
            cursor.execute(
                "SELECT 1 FROM caption_summary WHERE image_id = %s LIMIT 1",
                (image_id,)
            )
            if cursor.fetchone():
                self.logger.info(
                    f"caption_summary: image {image_id} already synthesized, skipping"
                )
                cursor.close()
                self._safe_ack(ch, method.delivery_tag)
                return
            cursor.close()

            # Fetch VLM captions
            captions = self._fetch_captions(image_id)
            if len(captions) < MIN_CAPTIONS:
                self.logger.info(
                    f"caption_summary: image {image_id} has {len(captions)} caption(s), "
                    f"need {MIN_CAPTIONS} — skipping"
                )
                self._safe_ack(ch, method.delivery_tag)
                return

            # Record that caption_summary was triggered — after both skip guards
            self._record_service_dispatch(image_id, 'caption_summary')

            # Fetch supporting consensus data (both optional — prompt degrades gracefully)
            nouns = self._fetch_noun_consensus(image_id)
            verbs = self._fetch_verb_consensus(image_id)

            # Call caption-synthesis service
            summary, synthesis_model = self._call_synthesis_service(captions, nouns, verbs, image_id)

            if not summary:
                self.logger.error(
                    f"caption_summary: empty response from synthesis service for image {image_id}, skipping"
                )
                self._safe_nack(ch, method.delivery_tag, requeue=False)
                self.job_failed("Empty synthesis response")
                return

            services_present = sorted(captions.keys())
            self._upsert(image_id, summary, synthesis_model, services_present,
                         processing_time=round(time.time() - start_time, 3))
            self._update_service_dispatch(image_id, service='caption_summary')

            # CLIP score the synthesized caption — non-fatal if unavailable
            if image_data:
                self._score_and_save(image_id, summary, image_data)
            else:
                self.logger.debug(
                    f"caption_summary: no image_data for image {image_id}, skipping CLIP score"
                )

            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()

            self.logger.info(
                f"caption_summary: image {image_id} — "
                f"{len(services_present)} VLMs, {round(time.time() - start_time, 3)}s — "
                f'"{summary[:80]}{"..." if len(summary) > 80 else ""}"'
            )

        except Exception as e:
            self.logger.error(f"caption_summary: error processing message: {e}")
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self.job_failed(str(e))

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_captions(self, image_id: int) -> dict:
        """
        Return {service: caption_text} for all VLM services that produced
        a non-empty caption for this image.
        """
        captions = {}
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                SELECT service, data
                FROM results
                WHERE image_id = %s
                  AND service = ANY(%s::text[])
                  AND status = 'success'
                ORDER BY result_created ASC
                """,
                (image_id, VLM_SERVICES)
            )
            for service, data in cursor.fetchall():
                if not isinstance(data, dict):
                    continue
                text = _extract_caption(data)
                if text:
                    captions[service] = text
            cursor.close()
        except Exception as e:
            self.logger.error(f"caption_summary: caption fetch error for image {image_id}: {e}")
        return captions

    def _fetch_noun_consensus(self, image_id: int) -> list:
        """
        Return the nouns list from noun_consensus, or [] if not present.
        Each element is a dict with 'canonical', 'category', 'vote_count'.
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                "SELECT nouns FROM noun_consensus WHERE image_id = %s",
                (image_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            if row and isinstance(row[0], list):
                return row[0]
        except Exception as e:
            self.logger.error(f"caption_summary: noun_consensus fetch error for image {image_id}: {e}")
        return []

    def _fetch_verb_consensus(self, image_id: int) -> list:
        """
        Return the verbs list from verb_consensus, or [] if not present.
        Each element is a dict with 'canonical', 'vote_count'.
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                "SELECT verbs FROM verb_consensus WHERE image_id = %s",
                (image_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            if row and isinstance(row[0], list):
                return row[0]
        except Exception as e:
            self.logger.error(f"caption_summary: verb_consensus fetch error for image {image_id}: {e}")
        return []

    # ------------------------------------------------------------------
    # Synthesis service call
    # ------------------------------------------------------------------

    def _call_synthesis_service(
        self, captions: dict, nouns: list, verbs: list, image_id: int
    ) -> tuple:
        """POST captions and consensus data to the caption-synthesis AF service.

        Returns (summary: str, model: str). Both are empty strings on failure.
        """
        try:
            resp = requests.post(
                self.synthesis_url,
                json={"captions": captions, "nouns": nouns, "verbs": verbs},
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get('status') != 'success':
                self.logger.error(
                    f"caption_summary: synthesis service error for image {image_id}: "
                    f"{data.get('error', 'unknown')}"
                )
                return '', ''
            return data.get('summary', '').strip(), data.get('model', 'unknown')
        except Exception as e:
            self.logger.error(
                f"caption_summary: synthesis service call failed for image {image_id}: {e}"
            )
            return '', ''

    # ------------------------------------------------------------------
    # CLIP scoring
    # ------------------------------------------------------------------

    def _score_and_save(self, image_id: int, caption: str, image_data: str):
        """Score the synthesized caption against the image via CLIP, save to postprocessing."""
        try:
            image_bytes = base64.b64decode(image_data)
            resp = requests.post(
                self.clip_score_url,
                files={'file': ('image.jpg', io.BytesIO(image_bytes), 'image/jpeg')},
                data={'caption': caption},
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()

            if result.get('status') != 'success':
                self.logger.warning(
                    f"caption_summary: CLIP score returned non-success for image {image_id}"
                )
                return

            score = result.get('similarity_score', 0.0)

            score_data = {
                'caption_score': {
                    'service': 'synthesis',
                    'caption': caption,
                    'similarity_score': score,
                    'scored_at': datetime.now().isoformat(),
                },
                'processing_algorithm': 'clip_similarity_v1',
                'processed_at': datetime.now().isoformat(),
            }

            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO postprocessing (image_id, service, data, status)
                VALUES (%s, %s, %s, 'success')
                """,
                (image_id, 'caption_score_synthesis', json.dumps(score_data))
            )
            self.db_conn.commit()
            cursor.close()

            self.logger.info(
                f"caption_summary: synthesis CLIP score {score:.3f} for image {image_id}"
            )

        except Exception as e:
            self.logger.error(
                f"caption_summary: CLIP scoring failed for image {image_id}: {e}"
            )

    # ------------------------------------------------------------------
    # DB write
    # ------------------------------------------------------------------

    def _upsert(self, image_id: int, summary: str, model: str, services_present: list,
                processing_time: float = None):
        """Insert or update the caption_summary row for this image."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO caption_summary
                    (image_id, summary_caption, model, services_present, service_count,
                     processing_time, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (image_id) DO UPDATE SET
                    summary_caption  = EXCLUDED.summary_caption,
                    model            = EXCLUDED.model,
                    services_present = EXCLUDED.services_present,
                    service_count    = EXCLUDED.service_count,
                    processing_time  = EXCLUDED.processing_time,
                    updated_at       = NOW()
                """,
                (image_id, summary, model, services_present, len(services_present), processing_time)
            )
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"caption_summary: upsert failed for image {image_id}: {e}")
            raise


if __name__ == "__main__":
    worker = CaptionSummaryWorker()
    worker.start()
