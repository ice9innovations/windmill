#!/usr/bin/env python3
"""
NounConsensusWorker - Cross-VLM noun synonym collapsing and vote counting.

Triggered by VLM services (blip, moondream, ollama) after each processes
an image. Fetches noun lists from all available VLM results, collapses
synonyms via WordNet, counts agreements, and writes to noun_consensus table.

Runs progressively: first trigger produces a partial result (1 VLM),
subsequent triggers overwrite with richer results as more VLMs complete.
Final result when all configured VLM services have reported.
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
from noun_utils import collapse_synonyms, warmup_wordnet, load_conceptnet, load_mwe, apply_mwe_normalization
from noun_extractor import extract_nouns, categorize_nouns, extract_subject, warmup_noun_extractor

logger = logging.getLogger(__name__)

# Services whose noun lists participate in consensus.
# Matches the 'vlm' service_type entries in service_config.yaml.
VLM_SERVICES = ['blip', 'haiku', 'moondream', 'ollama', 'qwen']


class NounConsensusWorker(BaseWorker):
    """
    Aggregates noun lists from VLM services and produces a synonym-collapsed
    consensus written to the noun_consensus table.
    """

    def __init__(self):
        super().__init__('system.noun_consensus')
        warmup_wordnet()         # Load WordNet corpus
        warmup_noun_extractor()  # Load spaCy en_core_web_sm
        load_mwe()
        if self.connect_to_database():
            load_conceptnet(self.db_conn)

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
            image_data = message.get('image_data')
            triggering_service = message.get('service', 'unknown')

            self.logger.debug(
                f"noun_consensus: processing image {image_id} "
                f"(triggered by {triggering_service})"
            )

            # Fetch noun lists, categories, and subject guesses from all VLM results
            service_noun_map, service_category_map, service_subject_map = self._fetch_vlm_nouns(image_id)

            # Strip adjective modifiers from non-MWE compound nouns
            # e.g. 'dirt path' -> 'path', 'dirt road' -> 'road'
            # but 'rain forest' -> 'rain forest' (recognized MWE)
            service_noun_map, service_category_map = apply_mwe_normalization(
                service_noun_map, service_category_map
            )

            if not service_noun_map:
                self.logger.debug(
                    f"noun_consensus: no VLM noun data yet for image {image_id}, skipping"
                )
                self._safe_ack(ch, method.delivery_tag)
                return

            # Collapse synonyms and count votes (ConceptNet via in-memory edges)
            collapsed = collapse_synonyms(service_noun_map)

            # Annotate each canonical noun with its agreed-upon category
            self._annotate_categories(collapsed, service_category_map)

            # Build category-level vote tally
            category_tally = self._compute_category_tally(collapsed)

            # Upsert into noun_consensus table
            services_present = sorted(service_noun_map.keys())
            self._upsert_noun_consensus(
                image_id, collapsed, services_present,
                processing_time=round(time.time() - start_time, 3),
                category_tally=category_tally,
            )

            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()

            self.logger.info(
                f"noun_consensus: image {image_id} - "
                f"{len(collapsed)} canonical nouns from {len(services_present)} VLMs "
                f"({', '.join(services_present)})"
            )

            # Vote on subject noun across VLM captions.
            from collections import Counter
            subject_votes = Counter(
                s for s in service_subject_map.values() if s
            )
            subject_noun = subject_votes.most_common(1)[0][0] if subject_votes else None
            if subject_noun:
                self.logger.info(
                    f"noun_consensus: subject vote for image {image_id}: "
                    f"{dict(subject_votes)} → '{subject_noun}'"
                )

            # Trigger SAM3 when all VLM services have reported.
            # Compare the current consensus nouns against what SAM3 was last
            # sent — only re-trigger if the noun list has changed (new nouns
            # crossed the confidence threshold as more VLMs reported in).
            if self._count_vlm_results(image_id) == len(VLM_SERVICES) and image_data:
                consensus_nouns = [
                    n['canonical'] for n in collapsed
                    if n.get('confidence', 0) > 0.5
                ]
                previous_nouns = self._sam3_previous_nouns(image_id)
                if previous_nouns is not None and set(consensus_nouns) == previous_nouns:
                    self.logger.debug(
                        f"noun_consensus: SAM3 already ran for image {image_id} "
                        f"with same nouns, skipping"
                    )
                elif consensus_nouns:
                    self._trigger_sam3(image_id, image_data, consensus_nouns, subject_noun)
                else:
                    # No consensus reached — promote the top noun if it's a clear
                    # leader (strictly ahead of second place). Fires only when the
                    # normal path produced nothing, so SAM3 isn't flooded with
                    # low-confidence hits.
                    sorted_nouns = sorted(
                        collapsed, key=lambda n: n.get('confidence', 0), reverse=True
                    )
                    if (len(sorted_nouns) >= 2
                            and sorted_nouns[0].get('confidence', 0)
                            > sorted_nouns[1].get('confidence', 0)):
                        top = sorted_nouns[0]['canonical']
                        if previous_nouns is not None and set([top]) == previous_nouns:
                            self.logger.debug(
                                f"noun_consensus: SAM3 already ran for image {image_id} "
                                f"with promoted noun '{top}', skipping"
                            )
                        else:
                            self.logger.info(
                                f"noun_consensus: no consensus for image {image_id} — "
                                f"promoting clear leader '{top}' to SAM3"
                            )
                            # Mark the promoted noun in the stored data so the read
                            # layer can surface it in nouns (not just nouns_all).
                            sorted_nouns[0]['promoted'] = True
                            self._upsert_noun_consensus(
                                image_id, collapsed, services_present,
                                processing_time=round(time.time() - start_time, 3),
                                category_tally=category_tally,
                            )
                            self._trigger_sam3(image_id, image_data, [top], subject_noun)

        except Exception as e:
            self.logger.error(f"noun_consensus: error processing message: {e}")
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self.job_failed(str(e))

    def _count_vlm_results(self, image_id: int) -> int:
        """Return the number of VLM_SERVICES that have any status='success' result.

        Counts all VLMs regardless of whether they produced nouns — a blocked
        haiku result (empty nouns, synthetic blocked text) still counts.
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """SELECT COUNT(DISTINCT service) FROM results
                   WHERE image_id = %s
                     AND service = ANY(%s::text[])
                     AND status = 'success'""",
                (image_id, VLM_SERVICES)
            )
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        except Exception as e:
            self.logger.error(f"noun_consensus: result count error for image {image_id}: {e}")
            return 0

    def _sam3_previous_nouns(self, image_id: int):
        """Return the set of nouns from the last SAM3 run, or None if SAM3
        has never run for this image."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                "SELECT nouns_queried FROM sam3_results WHERE image_id = %s LIMIT 1",
                (image_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            if row is None:
                return None
            # nouns_queried is stored as a list (JSON array or TEXT[])
            prev = row[0]
            if isinstance(prev, str):
                prev = json.loads(prev)
            return set(prev) if prev else set()
        except Exception as e:
            self.logger.error(f"noun_consensus: sam3 check failed for image {image_id}: {e}")
            return None

    def _trigger_sam3(self, image_id: int, image_data: str, nouns: list, subject_noun: str = None):
        """Publish a SAM3 segmentation request for the final noun consensus."""
        try:
            queue_name = self._get_queue_by_service_type('sam3')
            self._enqueue_publish(
                queue_name,
                json.dumps({
                    'image_id': image_id,
                    'image_data': image_data,
                    'nouns': nouns,
                    'subject_noun': subject_noun,
                    'triggered_at': datetime.now().isoformat(),
                }),
            )
            self.logger.info(
                f"noun_consensus: triggered SAM3 for image {image_id} "
                f"with {len(nouns)} nouns: {nouns}"
            )
        except Exception as e:
            self.logger.error(f"noun_consensus: failed to trigger SAM3 for image {image_id}: {e}")

    def _fetch_vlm_nouns(self, image_id: int) -> tuple:
        """
        Fetch noun lists, categories, and subject guesses from results table
        for all VLM services.

        Returns (service_noun_map, service_category_map, service_subject_map):
          service_noun_map:     {service: [nouns]}     — services with non-empty noun lists
          service_category_map: {service: {noun: cat}} — noun_categories per service
          service_subject_map:  {service: subject_noun | None} — grammatical subject per caption
        """
        service_noun_map = {}
        service_category_map = {}
        service_subject_map = {}
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
                predictions = data.get('predictions', [])
                caption = predictions[0].get('text', '').strip() if predictions else ''
                if not caption:
                    continue
                nouns = extract_nouns(caption)
                categories = categorize_nouns(nouns)
                subject = extract_subject(caption)
                if nouns:
                    service_noun_map[service] = nouns
                    service_category_map[service] = categories
                service_subject_map[service] = subject

        except Exception as e:
            self.logger.error(f"noun_consensus: DB fetch error for image {image_id}: {e}")

        return service_noun_map, service_category_map, service_subject_map

    def _compute_category_tally(self, collapsed: list) -> list:
        """Build a hierarchical category → nouns map from collapsed noun entries.

        Groups nouns by their category, nests them under each category entry,
        and computes category-level vote counts from the union of all noun services.

        Returns a list sorted by vote_count desc, with 'object' last
        (it is the catch-all fallback, not a meaningful category).

        Each entry:
        {
            "category": "food",
            "vote_count": 5,
            "services": ["blip", "haiku", ...],
            "nouns": [
                {"canonical": "fries", "vote_count": 5, "services": [...]},
                ...
            ]
        }
        """
        from collections import defaultdict

        cat_services = defaultdict(set)
        cat_nouns = defaultdict(list)

        for entry in collapsed:
            cat = entry.get('category')
            if not cat:
                continue
            for service in entry.get('services', []):
                cat_services[cat].add(service)
            cat_nouns[cat].append({
                'canonical': entry['canonical'],
                'vote_count': entry.get('vote_count', 0),
                'services': sorted(entry.get('services', [])),
            })

        # Sort nouns within each category by vote_count desc
        for cat in cat_nouns:
            cat_nouns[cat].sort(key=lambda x: (-x['vote_count'], x['canonical']))

        tally = [
            {
                'category': cat,
                'vote_count': len(svcs),
                'services': sorted(svcs),
                'nouns': cat_nouns[cat],
            }
            for cat, svcs in cat_services.items()
        ]
        tally.sort(key=lambda x: (x['category'] == 'object', -x['vote_count']))
        return tally

    def _annotate_categories(self, collapsed: list, service_category_map: dict):
        """Add a 'category' field to each canonical noun entry in collapsed.

        For each canonical noun, collect the category assigned by each contributing
        service for any of that noun's surface_forms, then take the mode (most
        common) category. Mutates collapsed in place.
        """
        from collections import Counter
        for entry in collapsed:
            votes = []
            for service in entry.get('services', []):
                cats = service_category_map.get(service, {})
                for surface in entry.get('surface_forms', [entry['canonical']]):
                    cat = cats.get(surface)
                    if cat:
                        votes.append(cat)
                        break  # one vote per service per canonical noun
            if votes:
                entry['category'] = Counter(votes).most_common(1)[0][0]

    def _upsert_noun_consensus(
        self, image_id: int, nouns: list,
        services_present: list, processing_time: float,
        category_tally: list = None
    ):
        """Insert or update the noun_consensus row for this image."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO noun_consensus
                    (image_id, nouns, category_tally, services_present, service_count, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (image_id) DO UPDATE SET
                    nouns = (
                        -- Preserve sam3_validated=true from the existing row.
                        -- noun_consensus runs once per VLM completion; SAM3 can
                        -- validate between the first and last VLM triggers, so
                        -- later upserts must not clear flags set by SAM3.
                        SELECT jsonb_agg(
                            CASE
                                WHEN EXISTS (
                                    SELECT 1
                                    FROM jsonb_array_elements(noun_consensus.nouns) old_n
                                    WHERE (old_n->>'canonical') = (new_n->>'canonical')
                                      AND (old_n->>'sam3_validated')::boolean = true
                                )
                                THEN new_n || '{"sam3_validated": true}'::jsonb
                                ELSE new_n
                            END
                        )
                        FROM jsonb_array_elements(EXCLUDED.nouns) new_n
                    ),
                    category_tally   = EXCLUDED.category_tally,
                    services_present = EXCLUDED.services_present,
                    service_count    = EXCLUDED.service_count,
                    updated_at       = NOW()
                """,
                (
                    image_id,
                    json.dumps(nouns),
                    json.dumps(category_tally or []),
                    services_present,
                    len(services_present),
                )
            )
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(
                f"noun_consensus: upsert failed for image {image_id}: {e}"
            )
            raise


if __name__ == "__main__":
    worker = NounConsensusWorker()
    worker.start()
