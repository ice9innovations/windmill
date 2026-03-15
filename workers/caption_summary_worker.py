#!/usr/bin/env python3
"""
CaptionSummaryWorker - Synthesize VLM captions into a single description.

Triggered by noun_consensus_worker as VLM captions arrive. Fires as soon
as MIN_CAPTIONS are available and re-fires progressively as stragglers
arrive. Does not depend on SAM3 completing first.

Collects:
  - All available VLM captions (from results table)
  - Noun consensus with vote counts (from noun_consensus table)
  - Verb consensus with vote counts (from verb_consensus table)

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
from service_config import get_service_config

logger = logging.getLogger(__name__)

# Derived from service_type: vlm entries in service_config.yaml — do not hardcode here.
VLM_SERVICES = get_service_config().get_vlm_service_names()

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
            tier = message.get('tier', 'free')

            self.logger.debug(f"caption_summary: processing image {image_id} (tier={tier})")

            # Derive tier-appropriate VLM list — only synthesize from services this tier ran
            tier_vlm_services = self._tier_vlm_services(tier)

            # Dedup guard — skip if we have no more captions than the last synthesis used.
            # Allows re-synthesis when stragglers (e.g. gpt_nano) arrive after first pass.
            cursor = self.db_conn.cursor()
            cursor.execute(
                "SELECT service_count FROM caption_summary WHERE image_id = %s LIMIT 1",
                (image_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            existing_count = row[0] if row else 0

            # Fetch VLM captions — tier-scoped
            captions = self._fetch_captions(image_id, tier_vlm_services)

            if len(captions) <= existing_count:
                self.logger.info(
                    f"caption_summary: image {image_id} already synthesized with "
                    f"{existing_count} caption(s), still {len(captions)} available — skipping"
                )
                self._safe_ack(ch, method.delivery_tag)
                return

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

    def _tier_vlm_services(self, tier: str) -> list:
        """Return the VLM services active for a given tier, ordered as VLM_SERVICES."""
        tier_primary = {
            name.split('.', 1)[1]
            for name in self.config.get_services_by_tier(tier)
            if name.startswith('primary.')
        }
        return [s for s in VLM_SERVICES if s in tier_primary]

    def _fetch_captions(self, image_id: int, vlm_services: list = None) -> dict:
        """
        Return {service: caption_text} for the given VLM services that produced
        a non-empty caption for this image. Defaults to all VLM_SERVICES.
        """
        services = vlm_services if vlm_services is not None else VLM_SERVICES
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
                (image_id, services)
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
