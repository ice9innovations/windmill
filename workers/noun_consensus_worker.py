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
from service_config import get_service_config
from noun_utils import collapse_synonyms, warmup_wordnet, load_conceptnet, load_mwe, apply_mwe_normalization
from noun_extractor import extract_nouns, categorize_nouns, extract_subject, extract_nouns_and_subject, warmup_noun_extractor

logger = logging.getLogger(__name__)

# Services whose noun lists participate in consensus.
# Derived from service_type: vlm entries in service_config.yaml — do not hardcode here.
VLM_SERVICES = get_service_config().get_vlm_service_names()


class NounConsensusWorker(BaseWorker):
    """
    Aggregates noun lists from VLM services and produces a synonym-collapsed
    consensus written to the noun_consensus table.
    """

    def __init__(self):
        super().__init__('system.noun_consensus')
        warmup_wordnet()         # Load WordNet corpus
        warmup_noun_extractor()  # Load spaCy en_core_web_lg
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
            tier = message.get('tier', 'free')

            self._record_service_dispatch(image_id, 'noun_consensus')

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

            self._update_service_dispatch(image_id, service='noun_consensus')
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

            # Trigger grounding as soon as any VLM has reported.
            # The pending-dispatch guard ensures at most one grounding job is
            # in-flight per image at a time. If one is already pending, the
            # trigger is skipped — the in-flight job processes the current noun
            # set when it runs, and later noun_consensus triggers will dispatch
            # again once that job completes if the noun set has grown.
            if self._count_vlm_results(image_id) > 0 and image_data:
                # Threshold based on tier's total expected VLM count, not how
                # many have reported so far. This keeps the bar stable across
                # all progressive triggers — a noun that passes on trigger 1
                # should still pass on trigger 5, and vice versa.
                tier_vlm_count = len([
                    name for name in self.config.get_services_by_tier(tier)
                    if name.startswith('primary.')
                    and name.split('.', 1)[1] in VLM_SERVICES
                ]) or len(service_noun_map)
                min_votes = max(2, (tier_vlm_count + 1) // 2)
                self.logger.debug(
                    f"noun_consensus: grounding threshold for image {image_id}: "
                    f">= {min_votes} votes (tier has {tier_vlm_count} VLMs)"
                )
                consensus_nouns = [
                    n['canonical'] for n in collapsed
                    if n.get('vote_count', 0) >= min_votes
                ]
                if consensus_nouns:
                    if self._has_pending_grounding_dispatch(image_id):
                        self.logger.debug(
                            f"noun_consensus: grounding already pending "
                            f"for image {image_id}, skipping"
                        )
                    else:
                        self._trigger_florence2_grounding(image_id, image_data, consensus_nouns, subject_noun, tier)
                else:
                    # No consensus reached — promote the top noun if it's a clear
                    # leader (strictly ahead of second place). Fires only when the
                    # normal path produced nothing, so grounding isn't flooded with
                    # low-confidence hits.
                    sorted_nouns = sorted(
                        collapsed, key=lambda n: n.get('confidence', 0), reverse=True
                    )
                    if (len(sorted_nouns) >= 2
                            and sorted_nouns[0].get('confidence', 0)
                            > sorted_nouns[1].get('confidence', 0)):
                        top = sorted_nouns[0]['canonical']
                        self.logger.info(
                            f"noun_consensus: no consensus for image {image_id} — "
                            f"promoting clear leader '{top}' to grounding"
                        )
                        # Mark the promoted noun in the stored data so the read
                        # layer can surface it in nouns (not just nouns_all).
                        sorted_nouns[0]['promoted'] = True
                        self._upsert_noun_consensus(
                            image_id, collapsed, services_present,
                            processing_time=round(time.time() - start_time, 3),
                            category_tally=category_tally,
                        )
                        if self._has_pending_grounding_dispatch(image_id):
                            self.logger.debug(
                                f"noun_consensus: grounding already pending "
                                f"for image {image_id} (promoted '{top}'), skipping"
                            )
                        else:
                            self._trigger_florence2_grounding(image_id, image_data, [top], subject_noun, tier)

            # Trigger caption_summary directly — fires as soon as enough captions are
            # available, and re-fires progressively as stragglers arrive. No longer
            # depends on SAM3 completing first.
            if self.config.is_available_for_tier('system.caption_summary', tier) and image_data:
                self._maybe_update_caption_summary(image_id, image_data, tier)

        except Exception as e:
            self.logger.error(f"noun_consensus: error processing message: {e}")
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self.job_failed(str(e))

    def _count_vlm_results(self, image_id: int) -> int:
        """Return the number of VLM_SERVICES that have a status='success' result.

        Counts all VLMs regardless of whether they produced nouns — a blocked
        result (empty nouns, synthetic blocked text) still counts.
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

    def _has_pending_grounding_dispatch(self, image_id: int) -> bool:
        """Return True if a grounding dispatch is already pending for this image.

        Used to prevent multiple simultaneous grounding jobs — noun_consensus fires
        on every VLM completion, but only one grounding job should be in-flight at
        a time. Safe to use as a non-atomic guard here because noun_consensus_worker
        is single-consumer and processes one message at a time.
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """SELECT 1 FROM service_dispatch
                   WHERE image_id = %s
                     AND service = 'florence2_grounding'
                     AND cluster_id IS NULL
                     AND status = 'pending'
                   LIMIT 1""",
                (image_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            return row is not None
        except Exception as e:
            self.logger.warning(
                f"noun_consensus: pending grounding check failed for image {image_id}: {e}"
            )
            return False

    def _maybe_update_caption_summary(self, image_id: int, image_data: str, tier: str):
        """Trigger caption_summary synthesis when new VLM captions are available.

        Fires on every noun_consensus message. On first run (no existing synthesis)
        it triggers as soon as MIN_CAPTIONS are available. On subsequent runs it
        re-triggers only when additional captions have arrived since the last synthesis,
        covering stragglers (e.g. gpt_nano) progressively."""
        try:
            # Count captions from tier VLMs available right now
            tier_vlms = [
                name.split('.', 1)[1]
                for name in self.config.get_services_by_tier(tier)
                if name.startswith('primary.')
                and name.split('.', 1)[1] in VLM_SERVICES
            ]
            if not tier_vlms:
                return

            cursor = self.db_conn.cursor()
            cursor.execute(
                """SELECT COUNT(DISTINCT service) FROM results
                   WHERE image_id = %s AND service = ANY(%s) AND status = 'success'""",
                (image_id, tier_vlms),
            )
            available_count = cursor.fetchone()[0]

            # Check what the last synthesis used
            cursor.execute(
                "SELECT service_count FROM caption_summary WHERE image_id = %s LIMIT 1",
                (image_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            last_count = row[0] if row else 0

            if available_count <= last_count:
                return  # Nothing new since last synthesis

            # Declare caption_summary queue on consume channel (idempotent — queue is broker-side)
            queue_name = self._get_queue_name('system.caption_summary')
            dlq = f"{queue_name}.dlq"
            self.channel.queue_declare(queue=dlq, durable=True)
            self.channel.queue_declare(
                queue=queue_name, durable=True,
                arguments={'x-dead-letter-exchange': '', 'x-dead-letter-routing-key': dlq},
            )
            self._enqueue_publish(
                queue_name,
                json.dumps({'image_id': image_id, 'image_data': image_data, 'tier': tier}),
            )
            self.logger.info(
                f"noun_consensus: triggered caption_summary update for image {image_id} "
                f"({available_count} captions available, last synthesis had {last_count})"
            )
        except Exception as e:
            self.logger.error(
                f"noun_consensus: failed to check/trigger caption_summary for image {image_id}: {e}"
            )

    def _trigger_florence2_grounding(self, image_id: int, image_data: str, nouns: list, subject_noun: str = None, tier: str = 'free'):
        """Publish a Florence-2 grounding request for the final noun consensus.

        The dispatch_id (service_dispatch PK) is embedded in the queue message so the
        worker can use targeted completion tracking rather than the bulk-clear path.
        This is required because noun_consensus fires florence2_grounding progressively
        (once per VLM completion), creating multiple pending rows per image. Without
        dispatch_id, the first completion would bulk-clear all pending rows and the SSE
        stream would fire complete with only the partial first result.
        """
        if not self.config.is_available_for_tier('primary.florence2', tier):
            self.logger.debug(f"noun_consensus: skipping Florence-2 grounding for tier '{tier}' image {image_id}")
            return
        try:
            queue_name = self._get_queue_by_service_type('grounding')
            dispatch_id = self._record_service_dispatch(image_id, 'florence2_grounding', None)
            self._enqueue_publish(
                queue_name,
                json.dumps({
                    'image_id': image_id,
                    'image_data': image_data,
                    'nouns': nouns,
                    'subject_noun': subject_noun,
                    'triggered_at': datetime.now().isoformat(),
                    'tier': tier,
                    'dispatch_id': dispatch_id,
                }),
            )
            self.logger.info(
                f"noun_consensus: triggered Florence-2 grounding for image {image_id} "
                f"with {len(nouns)} nouns: {nouns} (dispatch_id={dispatch_id})"
            )
        except Exception as e:
            self.logger.error(f"noun_consensus: failed to trigger Florence-2 grounding for image {image_id}: {e}")


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
                # Skip blocked results — their sentinel text (e.g. "[NSFW — not sent to
                # Gemini API]") is not a caption and would produce garbage nouns.
                if data.get('metadata', {}).get('blocked'):
                    continue
                predictions = data.get('predictions', [])
                caption = predictions[0].get('text', '').strip() if predictions else ''
                if not caption:
                    continue
                nouns, subject = extract_nouns_and_subject(caption)
                categories = categorize_nouns(nouns)
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
                    (image_id, nouns, category_tally, services_present, service_count,
                     processing_time, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (image_id) DO UPDATE SET
                    nouns = (
                        -- Preserve sam3_validated and grounding_validated from
                        -- the existing row. noun_consensus runs once per VLM
                        -- completion; SAM3 and Florence-2 grounding can validate
                        -- between the first and last VLM triggers, so later
                        -- upserts must not clear flags set by those workers.
                        SELECT jsonb_agg(
                            new_n
                            || CASE WHEN EXISTS (
                                    SELECT 1
                                    FROM jsonb_array_elements(noun_consensus.nouns) old_n
                                    WHERE (old_n->>'canonical') = (new_n->>'canonical')
                                      AND (old_n->>'sam3_validated')::boolean = true
                                )
                                THEN '{"sam3_validated": true}'::jsonb
                                ELSE '{}'::jsonb
                               END
                            || CASE WHEN EXISTS (
                                    SELECT 1
                                    FROM jsonb_array_elements(noun_consensus.nouns) old_n
                                    WHERE (old_n->>'canonical') = (new_n->>'canonical')
                                      AND (old_n->>'grounding_validated')::boolean = true
                                )
                                THEN '{"grounding_validated": true}'::jsonb
                                ELSE '{}'::jsonb
                               END
                        )
                        FROM jsonb_array_elements(EXCLUDED.nouns) new_n
                    ),
                    category_tally   = EXCLUDED.category_tally,
                    services_present = EXCLUDED.services_present,
                    service_count    = EXCLUDED.service_count,
                    processing_time  = EXCLUDED.processing_time,
                    updated_at       = NOW()
                """,
                (
                    image_id,
                    json.dumps(nouns),
                    json.dumps(category_tally or []),
                    services_present,
                    len(services_present),
                    processing_time,
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
