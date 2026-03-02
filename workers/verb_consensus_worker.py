#!/usr/bin/env python3
"""
VerbConsensusWorker - Cross-VLM verb synonym collapsing and vote counting.

Triggered by VLM services (blip, moondream, ollama, qwen, haiku) after each
processes an image. Fetches verb lists from all available VLM results, collapses
synonyms via WordNet, counts agreements, and writes to verb_consensus table.

Runs progressively: first trigger produces a partial result (1 VLM),
subsequent triggers overwrite with richer results as more VLMs complete.
Final result when all configured VLM services have reported.

No SAM3 integration — verbs have no spatial grounding and cannot be
validated against segmentation masks.
"""

import os
import sys
import json
import time
import logging
import psycopg2
import pika
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker
from verb_utils import collapse_synonyms, warmup_wordnet

logger = logging.getLogger(__name__)

# Must match VLM_SERVICES in noun_consensus_worker.py
VLM_SERVICES = ['blip', 'haiku', 'moondream', 'ollama', 'qwen']


class VerbConsensusWorker(BaseWorker):
    """
    Aggregates verb lists from VLM services and produces a synonym-collapsed
    consensus written to the verb_consensus table.
    """

    def __init__(self):
        super().__init__('system.verb_consensus')
        warmup_wordnet()  # Load corpus now; first message pays no disk-load penalty

    def process_message(self, ch, method, properties, body):
        """Override base process_message - no ML service call, DB-only logic."""
        start_time = time.time()

        try:
            if not self.ensure_database_connection():
                self._safe_nack(ch, method.delivery_tag, requeue=False)
                self.job_failed("Database unavailable")
                return

            message = json.loads(body)
            image_id = message['image_id']
            triggering_service = message.get('service', 'unknown')

            self.logger.debug(
                f"verb_consensus: processing image {image_id} "
                f"(triggered by {triggering_service})"
            )

            service_verb_map, service_svo_map = self._fetch_vlm_verbs(image_id)

            if not service_verb_map:
                self.logger.debug(
                    f"verb_consensus: no VLM verb data yet for image {image_id}, skipping"
                )
                self._safe_ack(ch, method.delivery_tag)
                return

            collapsed = collapse_synonyms(service_verb_map)

            services_present = sorted(service_verb_map.keys())
            self._upsert_verb_consensus(
                image_id, collapsed, service_svo_map, services_present,
                processing_time=round(time.time() - start_time, 3),
            )

            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()

            self.logger.info(
                f"verb_consensus: image {image_id} - "
                f"{len(collapsed)} canonical verbs from {len(services_present)} VLMs "
                f"({', '.join(services_present)})"
            )

        except Exception as e:
            self.logger.error(f"verb_consensus: error processing message: {e}")
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self.job_failed(str(e))

    def _fetch_vlm_verbs(self, image_id: int) -> tuple:
        """
        Fetch verb lists and SVO triples from results table for all VLM services.
        Returns (service_verb_map, service_svo_map):
          service_verb_map: {service: [verbs]}          — services with non-empty verb lists
          service_svo_map:  {service: [[s, v, o], ...]} — SVO triples per service
        """
        service_verb_map = {}
        service_svo_map = {}
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
            rows = cursor.fetchall()
            cursor.close()

            for service, data in rows:
                if not isinstance(data, dict):
                    continue
                verbs = data.get('verbs', [])
                svo_triples = data.get('svo_triples', [])
                if verbs:
                    service_verb_map[service] = verbs
                    service_svo_map[service] = svo_triples if isinstance(svo_triples, list) else []

        except Exception as e:
            self.logger.error(f"verb_consensus: DB fetch error for image {image_id}: {e}")

        return service_verb_map, service_svo_map

    def _upsert_verb_consensus(
        self, image_id: int, verbs: list,
        service_svo_map: dict, services_present: list,
        processing_time: float,
    ):
        """Insert or update the verb_consensus row for this image."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO verb_consensus
                    (image_id, verbs, svo_triples, services_present, service_count, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (image_id) DO UPDATE SET
                    verbs            = EXCLUDED.verbs,
                    svo_triples      = EXCLUDED.svo_triples,
                    services_present = EXCLUDED.services_present,
                    service_count    = EXCLUDED.service_count,
                    updated_at       = NOW()
                """,
                (
                    image_id,
                    json.dumps(verbs),
                    json.dumps(service_svo_map),
                    services_present,
                    len(services_present),
                )
            )
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(
                f"verb_consensus: upsert failed for image {image_id}: {e}"
            )
            raise


if __name__ == "__main__":
    worker = VerbConsensusWorker()
    worker.start()
