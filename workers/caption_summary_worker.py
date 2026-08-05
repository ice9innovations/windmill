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
from core.postgres_connection import close_quietly, commit_if_needed, rollback_quietly
from service_config import get_service_config

logger = logging.getLogger(__name__)

# Derived from service_type: vlm entries in service_config.yaml — do not hardcode here.
VLM_SERVICES = get_service_config().get_vlm_service_names()

MIN_CAPTIONS = 2  # Skip synthesis if fewer than this many VLMs returned
NSFW_PREFERRED_SERVICE = 'joycaption'
NSFW_LABELS = frozenset({
    'FEMALE_BREAST_EXPOSED',
    'FEMALE_GENITALIA_EXPOSED',
    'MALE_GENITALIA_EXPOSED',
    'BUTTOCKS_EXPOSED',
    'ANUS_EXPOSED',
})


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


def _nsfw2_flags_nsfw(data: dict) -> bool:
    predictions = data.get('predictions', []) if isinstance(data, dict) else []
    if not predictions or not isinstance(predictions[0], dict):
        return False
    return predictions[0].get('nsfw') is True


def _nudenet_flags_nsfw(data: dict) -> bool:
    predictions = data.get('predictions', []) if isinstance(data, dict) else []
    return any(
        isinstance(prediction, dict) and prediction.get('label') in NSFW_LABELS
        for prediction in predictions
    )


def _select_preferred_caption_service(captions: dict, nsfw: bool) -> str:
    """Return the preferred caption service for synthesis, when one applies."""
    if nsfw and captions.get(NSFW_PREFERRED_SERVICE):
        return NSFW_PREFERRED_SERVICE
    return None


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
        previous_autocommit = None
        timing = {}

        try:
            if not self.ensure_database_connection():
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                self.job_failed("Database unavailable")
                return

            message = json.loads(body)
            image_id = message['image_id']
            tier = message.get('tier', 'free')

            self.logger.debug(f"caption_summary: processing image {image_id} (tier={tier})")

            # Derive tier-appropriate VLM list — only synthesize from services this tier ran
            tier_vlm_services = self._tier_vlm_services(tier)

            # Dedup guard — skip if we have no more captions than the last synthesis used.
            # Allows re-synthesis when stragglers (e.g. gpt_nano) arrive after first pass.
            cursor = self.db_conn.cursor()
            cursor.execute(
                "SELECT service_count, services_present FROM caption_summary WHERE image_id = %s LIMIT 1",
                (image_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            existing_count = row[0] if row else 0
            existing_services = set(row[1] or []) if row else set()

            # Fetch VLM captions — tier-scoped
            t0 = time.time()
            captions = self._fetch_captions(image_id, tier_vlm_services)
            timing['fetch_captions'] = time.time() - t0

            t0 = time.time()
            nsfw = self._image_flags_nsfw(image_id)
            timing['fetch_nsfw_flag'] = time.time() - t0
            preferred_service = _select_preferred_caption_service(captions, nsfw)

            if len(captions) <= existing_count and (
                not preferred_service or preferred_service in existing_services
            ):
                previous_autocommit = self.db_conn.autocommit
                self.db_conn.autocommit = False
                self._persist_terminal_result(
                    image_id=image_id,
                    payload={
                        'service': 'caption_summary',
                        'status': 'success',
                        'summary_caption': None,
                        'metadata': {
                            'no_usable_data': True,
                            'reason': 'Already synthesized with current caption count',
                            'existing_count': existing_count,
                            'available_count': len(captions),
                            'nsfw': nsfw,
                            'preferred_service': preferred_service,
                            'processed_at': datetime.now().isoformat(),
                        },
                    },
                    processing_time=round(time.time() - start_time, 3),
                    event_type='completed',
                    event_data={'reason': 'Already synthesized with current caption count'},
                )
                self.logger.info(
                    f"caption_summary: image {image_id} already synthesized with "
                    f"{existing_count} caption(s), still {len(captions)} available — skipping"
                )
                self._safe_ack(ch, method.delivery_tag)
                return

            min_captions = 1 if preferred_service else MIN_CAPTIONS
            if len(captions) < min_captions:
                previous_autocommit = self.db_conn.autocommit
                self.db_conn.autocommit = False
                self._persist_terminal_result(
                    image_id=image_id,
                    payload={
                        'service': 'caption_summary',
                        'status': 'success',
                        'summary_caption': None,
                        'metadata': {
                            'no_usable_data': True,
                            'reason': f'Need at least {min_captions} captions',
                            'available_count': len(captions),
                            'nsfw': nsfw,
                            'preferred_service': preferred_service,
                            'processed_at': datetime.now().isoformat(),
                        },
                    },
                    processing_time=round(time.time() - start_time, 3),
                    event_type='completed',
                    event_data={'reason': f'Need at least {min_captions} captions'},
                )
                self.logger.info(
                    f"caption_summary: image {image_id} has {len(captions)} caption(s), "
                    f"need {min_captions} — skipping"
                )
                self._safe_ack(ch, method.delivery_tag)
                return

            # Fetch supporting consensus data (both optional — prompt degrades gracefully)
            t0 = time.time()
            nouns = self._fetch_noun_consensus(image_id)
            timing['fetch_noun_consensus'] = time.time() - t0
            t0 = time.time()
            verbs = self._fetch_verb_consensus(image_id)
            timing['fetch_verb_consensus'] = time.time() - t0

            # Call caption-synthesis service
            t0 = time.time()
            synthesis_result = self._call_synthesis_service(
                captions,
                nouns,
                verbs,
                image_id,
                preferred_caption_service=preferred_service,
                nsfw=nsfw,
            )
            timing['call_synthesis_service'] = time.time() - t0
            if synthesis_result is None:
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                self.job_failed("Caption synthesis transport failure")
                return

            if synthesis_result.get('status', 'success') != 'success':
                previous_autocommit = self.db_conn.autocommit
                self.db_conn.autocommit = False
                self._persist_terminal_result(
                    image_id=image_id,
                    payload=synthesis_result,
                    processing_time=round(time.time() - start_time, 3),
                    status=synthesis_result.get('status', 'failed') or 'failed',
                    event_type='failed',
                    event_data={
                        'error_message': synthesis_result.get('error_message')
                        or synthesis_result.get('error')
                        or synthesis_result.get('message')
                    },
                )
                self._safe_ack(ch, method.delivery_tag)
                self.job_completed_successfully()
                self.logger.warning(
                    f"caption_summary: image {image_id} returned terminal non-success: "
                    f"{synthesis_result.get('error_message') or synthesis_result.get('error') or synthesis_result.get('message') or 'no reason provided'}"
                )
                return

            summary = synthesis_result.get('summary', '').strip()
            synthesis_model = synthesis_result.get('model', 'unknown')
            if not summary:
                previous_autocommit = self.db_conn.autocommit
                self.db_conn.autocommit = False
                synthesis_result = {
                    'service': 'caption_summary',
                    'status': 'failed',
                    'summary_caption': None,
                    'error_message': 'Caption synthesis returned empty summary',
                    'metadata': {
                        'terminal_http_error': False,
                        'processed_at': datetime.now().isoformat(),
                    },
                }
                self._persist_terminal_result(
                    image_id=image_id,
                    payload=synthesis_result,
                    processing_time=round(time.time() - start_time, 3),
                    status='failed',
                    event_type='failed',
                    event_data={'error_message': 'Caption synthesis returned empty summary'},
                )
                self._safe_ack(ch, method.delivery_tag)
                self.job_completed_successfully()
                self.logger.warning(
                    f"caption_summary: empty response from synthesis service for image {image_id}"
                )
                return

            services_present = sorted(captions.keys())
            previous_autocommit = self.db_conn.autocommit
            self.db_conn.autocommit = False
            t0 = time.time()
            self._persist_successful_summary_result(
                image_id=image_id,
                summary=summary,
                model=synthesis_model,
                services_present=services_present,
                processing_time=round(time.time() - start_time, 3),
                metadata={
                    'nsfw': nsfw,
                    'preferred_service': preferred_service,
                    'available_services': sorted(captions.keys()),
                },
            )
            timing['persist_success'] = time.time() - t0
            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()

            total_duration = time.time() - start_time
            slow_bits = " ".join(
                f"{name}={duration:.3f}s"
                for name, duration in timing.items()
                if duration >= 0.05
            )
            self.logger.info(
                f"caption_summary timing image={image_id} total={total_duration:.3f}s "
                f"{slow_bits}".rstrip()
            )
            self.logger.info(
                f"caption_summary: image {image_id} — "
                f"{len(services_present)} VLMs, {round(total_duration, 3)}s — "
                f'"{summary[:80]}{"..." if len(summary) > 80 else ""}"'
            )

        except Exception as e:
            if self.db_conn and previous_autocommit is False:
                rollback_quietly(self.db_conn)
            self.logger.error(f"caption_summary: error processing message: {e}")
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self.job_failed(str(e))
        finally:
            if self.db_conn and previous_autocommit is not None:
                try:
                    self.db_conn.autocommit = previous_autocommit
                except Exception:
                    pass

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

    def _image_flags_nsfw(self, image_id: int) -> bool:
        """Return True when stored NSFW classifiers mark the image as NSFW."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                SELECT service, data
                FROM results
                WHERE image_id = %s
                  AND service = ANY(%s::text[])
                  AND status = 'success'
                """,
                (image_id, ['nsfw2', 'nudenet'])
            )
            rows = cursor.fetchall()
            cursor.close()
        except Exception as e:
            self.logger.error(f"caption_summary: NSFW flag fetch error for image {image_id}: {e}")
            return False

        for service, data in rows:
            if not isinstance(data, dict):
                continue
            if service == 'nsfw2' and _nsfw2_flags_nsfw(data):
                return True
            if service == 'nudenet' and _nudenet_flags_nsfw(data):
                return True
        return False

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
        self,
        captions: dict,
        nouns: list,
        verbs: list,
        image_id: int,
        preferred_caption_service: str = None,
        nsfw: bool = False,
    ) -> dict:
        """POST captions and consensus data to the caption-synthesis AF service."""
        payload = {"captions": captions, "nouns": nouns, "verbs": verbs}
        if preferred_caption_service:
            payload["preferred_caption_service"] = preferred_caption_service
            payload["preference_reason"] = "nsfw" if nsfw else "configured"
        try:
            resp = requests.post(
                self.synthesis_url,
                json=payload,
                timeout=90,
            )
            return self._coerce_terminal_http_response(resp, service='caption_summary')
        except requests.RequestException as e:
            if getattr(e, 'response', None) is not None:
                return self._coerce_terminal_http_response(e.response, service='caption_summary')
            self.logger.error(
                f"caption_summary: synthesis service call failed for image {image_id} before terminal response: {e}"
            )
            return None

    # ------------------------------------------------------------------
    # DB write
    # ------------------------------------------------------------------

    def _persist_terminal_result(
        self,
        image_id: int,
        payload: dict,
        processing_time: float,
        event_type: str,
        event_data: dict,
        status: str = 'success',
    ):
        """Persist terminal result and event in one DB round trip."""
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
                    'caption_summary',
                    json.dumps(payload),
                    status,
                    self._extract_http_status(payload),
                    self.worker_id,
                    processing_time,
                    image_id,
                    'caption_summary',
                    event_type,
                    'noun_consensus',
                    'caption_summary_run',
                    json.dumps(event_data),
                ),
            )
            commit_if_needed(self.db_conn, force=True)
            close_quietly(cursor)
        except Exception as e:
            self.logger.error(
                f"caption_summary: failed to persist terminal result for image {image_id}: {e}"
            )
            raise

    def _persist_successful_summary_result(
        self,
        image_id: int,
        summary: str,
        model: str,
        services_present: list,
        processing_time: float,
        metadata: dict = None,
    ):
        """Persist caption_summary row, terminal result, and event in one DB round trip."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                WITH upserted_summary AS (
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
                    image_id,
                    summary,
                    model,
                    services_present,
                    len(services_present),
                    processing_time,
                    image_id,
                    'caption_summary',
                    json.dumps({
                        'service': 'caption_summary',
                        'status': 'success',
                        'summary_caption': summary,
                        'model': model,
                        'services_present': services_present,
                        'metadata': {
                            **(metadata or {}),
                            'processed_at': datetime.now().isoformat(),
                        },
                    }),
                    'success',
                    None,
                    self.worker_id,
                    processing_time,
                    image_id,
                    'caption_summary',
                    'completed',
                    'noun_consensus',
                    'caption_summary_run',
                    json.dumps({'services_present': services_present}),
                ),
            )
            commit_if_needed(self.db_conn, force=True)
            close_quietly(cursor)
        except Exception as e:
            self.logger.error(
                f"caption_summary: failed to persist successful summary for image {image_id}: {e}"
            )
            raise


if __name__ == "__main__":
    worker = CaptionSummaryWorker()
    worker.start()
