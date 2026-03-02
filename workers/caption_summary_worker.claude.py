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

Sends a single prompt to Qwen via Ollama and writes the synthesized
caption to the caption_summary table.

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
import psycopg2
from datetime import datetime

import anthropic

sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

logger = logging.getLogger(__name__)

VLM_SERVICES = ['blip', 'haiku', 'moondream', 'ollama', 'qwen']

MIN_CAPTIONS = 2  # Skip synthesis if fewer than this many VLMs returned

SYNTHESIS_PROMPT_HEADER = (
    "Here are several captions returned by our VLMs along with voting results of "
    "what the image contains. Please rewrite this into a single sentence to "
    "describe the image. Return only the sentence, please."
)


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
    Synthesizes VLM captions into a single sentence using Qwen via Ollama.
    Writes the result to the caption_summary table.
    """

    def __init__(self):
        super().__init__('system.caption_summary')
        self.ollama_host = self._get_required('OLLAMA_HOST')
        self.synthesis_model = os.getenv('CAPTION_SYNTHESIS_MODEL', 'qwen3-vl:4b-instruct')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

        # CLIP score endpoint — same resolution pattern as caption_score_worker
        clip_services = self.config.get_service_group('postprocessing.clip_score[]')
        clip_cfg = self.config.get_service_config(clip_services[0])
        self.clip_score_url = (
            f"http://{clip_cfg['host']}:{clip_cfg['port']}{clip_cfg['endpoint']}"
        )

    def process_message(self, ch, method, properties, body):
        """Override base process_message — DB-only fetch + Ollama text call."""
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

            # Fetch supporting consensus data (both optional — prompt degrades gracefully)
            nouns = self._fetch_noun_consensus(image_id)
            verbs = self._fetch_verb_consensus(image_id)

            # Build prompt and synthesize
            prompt = self._build_prompt(captions, nouns, verbs)
            summary = self._synthesize(prompt)

            if not summary:
                self.logger.error(
                    f"caption_summary: empty response from Ollama for image {image_id}, skipping"
                )
                self._safe_nack(ch, method.delivery_tag, requeue=False)
                self.job_failed("Empty Ollama response")
                return

            services_present = sorted(captions.keys())
            self._upsert(image_id, summary, services_present)

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
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, captions: dict, nouns: list, verbs: list) -> str:
        parts = [SYNTHESIS_PROMPT_HEADER]

        if nouns:
            parts.append(self._format_noun_consensus(nouns))

        if verbs:
            parts.append(self._format_verb_consensus(verbs))

        parts.append(self._format_captions(captions))

        return "\n\n".join(parts)

    def _format_noun_consensus(self, nouns: list) -> str:
        # Category totals line
        category_totals: dict = {}
        for noun in nouns:
            cat = noun.get('category', 'object')
            category_totals[cat] = category_totals.get(cat, 0) + noun.get('vote_count', 1)

        sorted_cats = sorted(category_totals.items(), key=lambda x: -x[1])
        category_line = ' '.join(f"{cat} {count}" for cat, count in sorted_cats)

        # Individual nouns line, sorted by vote_count desc
        sorted_nouns = sorted(nouns, key=lambda n: -n.get('vote_count', 0))
        nouns_line = ' '.join(
            f"{n['canonical']} {n.get('category', 'object')} {n.get('vote_count', 1)}"
            for n in sorted_nouns
        )

        return f"Noun Consensus\n{category_line}\n{nouns_line}"

    def _format_verb_consensus(self, verbs: list) -> str:
        sorted_verbs = sorted(verbs, key=lambda v: -v.get('vote_count', 0))
        verbs_line = ' '.join(
            f"{v['canonical']} {v.get('vote_count', 1)}"
            for v in sorted_verbs
        )
        return f"Verb Consensus\n{verbs_line}"

    def _format_captions(self, captions: dict) -> str:
        lines = ["VLM Captions"]
        for service, text in captions.items():
            lines.append(service)
            lines.append(text)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM dispatch
    # ------------------------------------------------------------------

    def _synthesize(self, prompt: str) -> str:
        """Route to Claude or Ollama based on model name prefix."""
        if self.synthesis_model.startswith('claude-'):
            return self._call_claude(prompt)
        return self._call_ollama(prompt)

    def _call_claude(self, prompt: str) -> str:
        """Text-only synthesis via Anthropic API."""
        if not self.anthropic_api_key:
            self.logger.error("caption_summary: ANTHROPIC_API_KEY not set")
            return ''
        try:
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            message = client.messages.create(
                model=self.synthesis_model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            self.logger.error(f"caption_summary: Claude call failed: {e}")
            return ''

    def _call_ollama(self, prompt: str) -> str:
        """Text-only synthesis via Ollama HTTP API."""
        try:
            resp = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.synthesis_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json().get('message', {}).get('content', '').strip()
        except Exception as e:
            self.logger.error(f"caption_summary: Ollama call failed: {e}")
            return ''

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

    def _upsert(self, image_id: int, summary: str, services_present: list):
        """Insert or update the caption_summary row for this image."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO caption_summary
                    (image_id, summary_caption, model, services_present, service_count,
                     created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (image_id) DO UPDATE SET
                    summary_caption  = EXCLUDED.summary_caption,
                    model            = EXCLUDED.model,
                    services_present = EXCLUDED.services_present,
                    service_count    = EXCLUDED.service_count,
                    updated_at       = NOW()
                """,
                (image_id, summary, self.synthesis_model, services_present, len(services_present))
            )
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"caption_summary: upsert failed for image {image_id}: {e}")
            raise


if __name__ == "__main__":
    worker = CaptionSummaryWorker()
    worker.start()
