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
from service_config import get_service_config
from verb_utils import collapse_synonyms, warmup_wordnet
from verb_extractor import extract_verbs_and_svo, warmup_verb_extractor

logger = logging.getLogger(__name__)

# Derived from service_type: vlm entries in service_config.yaml — do not hardcode here.
VLM_SERVICES = get_service_config().get_vlm_service_names()


class VerbConsensusWorker(BaseWorker):
    """
    Aggregates verb lists from VLM services and produces a synonym-collapsed
    consensus written to the verb_consensus table.
    """

    def __init__(self):
        super().__init__('system.verb_consensus')
        warmup_wordnet()         # Load WordNet corpus
        warmup_verb_extractor()  # Load spaCy en_core_web_lg

    def process_message(self, ch, method, properties, body):
        """Override base process_message - no ML service call, DB-only logic."""
        start_time = time.time()
        timing = {}

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

            t0 = time.time()
            service_verb_map, service_svo_map = self._fetch_vlm_verbs(image_id)
            timing['fetch_vlm_verbs'] = time.time() - t0

            if not service_verb_map:
                processing_time = round(time.time() - start_time, 3)
                self._store_terminal_service_result(
                    image_id,
                    {
                        'service': 'verb_consensus',
                        'status': 'success',
                        'verbs': [],
                        'svo_triples': {},
                        'metadata': {
                            'no_usable_data': True,
                            'reason': 'No VLM verb data yet',
                            'triggered_by': triggering_service,
                            'processed_at': datetime.now().isoformat(),
                        },
                    },
                    processing_time=processing_time,
                )
                self.logger.debug(
                    f"verb_consensus: no VLM verb data yet for image {image_id}, skipping"
                )
                self._record_service_event(
                    image_id=image_id,
                    service='verb_consensus',
                    event_type='completed',
                    source_service=triggering_service,
                    source_stage='verb_consensus_run',
                    data={'reason': 'No VLM verb data yet'},
                    commit=True,
                )
                self._safe_ack(ch, method.delivery_tag)
                self.job_completed_successfully()
                return

            t0 = time.time()
            collapsed = collapse_synonyms(service_verb_map)
            timing['collapse_synonyms'] = time.time() - t0

            services_present = sorted(service_verb_map.keys())
            t0 = time.time()
            self._upsert_verb_consensus(
                image_id, collapsed, service_svo_map, services_present,
                processing_time=round(time.time() - start_time, 3),
            )
            timing['upsert_verb_consensus'] = time.time() - t0

            t0 = time.time()
            self._store_terminal_service_result(
                image_id,
                {
                    'service': 'verb_consensus',
                    'status': 'success',
                    'verbs': collapsed,
                    'svo_triples': service_svo_map,
                    'services_present': services_present,
                    'metadata': {
                        'processed_at': datetime.now().isoformat(),
                        'services_present': services_present,
                    },
                },
                processing_time=round(time.time() - start_time, 3),
            )
            timing['store_terminal_result'] = time.time() - t0
            t0 = time.time()
            self._record_service_event(
                image_id=image_id,
                service='verb_consensus',
                event_type='completed',
                source_service=triggering_service,
                source_stage='verb_consensus_run',
                data={'services_present': services_present},
                commit=True,
            )
            timing['record_completed_event'] = time.time() - t0

            total_duration = time.time() - start_time
            slow_bits = " ".join(
                f"{name}={duration:.3f}s"
                for name, duration in timing.items()
                if duration >= 0.05
            )
            self.logger.info(
                f"verb_consensus timing image={image_id} total={total_duration:.3f}s "
                f"{slow_bits}".rstrip()
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
        Fetch captions from results table for all VLM services and extract
        verbs and SVO triples locally — parallel to noun_consensus_worker's
        _fetch_vlm_nouns() which extracts nouns from captions rather than
        reading pre-stored noun lists.

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
                # Skip blocked results — their sentinel text is not a caption
                # and would produce garbage verbs.
                if data.get('metadata', {}).get('blocked'):
                    continue
                predictions = data.get('predictions', [])
                caption = predictions[0].get('text', '').strip() if predictions else ''
                if not caption:
                    continue
                verbs, svo_triples = extract_verbs_and_svo(caption)
                if verbs:
                    service_verb_map[service] = verbs
                    service_svo_map[service] = svo_triples

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
                    (image_id, verbs, svo_triples, services_present, service_count,
                     processing_time, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (image_id) DO UPDATE SET
                    verbs            = EXCLUDED.verbs,
                    svo_triples      = EXCLUDED.svo_triples,
                    services_present = EXCLUDED.services_present,
                    service_count    = EXCLUDED.service_count,
                    processing_time  = EXCLUDED.processing_time,
                    updated_at       = NOW()
                """,
                (
                    image_id,
                    json.dumps(verbs),
                    json.dumps(service_svo_map),
                    services_present,
                    len(services_present),
                    processing_time,
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
