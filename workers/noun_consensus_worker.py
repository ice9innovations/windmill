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
from contextlib import contextmanager
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

    def connect_to_database(self):
        """Connect with autocommit disabled so this worker can batch writes."""
        try:
            if self.db_conn:
                try:
                    self.db_conn.close()
                except Exception:
                    pass
            self.db_conn = self._new_db_connection(autocommit=False)
            self.logger.info(f"Connected to PostgreSQL at {self.db_host}")
            self.consecutive_db_failures = 0
            self.db_backoff_delay = 1
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False

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
            image_transport = self._image_transport_fields(message)
            triggering_service = message.get('service', 'unknown')
            tier = message.get('tier', 'free')
            submitted_at_epoch = message.get('submitted_at_epoch')
            consensus_enqueued_at_epoch = message.get('consensus_enqueued_at_epoch')
            queue_wait = None
            if consensus_enqueued_at_epoch is not None:
                try:
                    queue_wait = max(0.0, time.time() - float(consensus_enqueued_at_epoch))
                except (TypeError, ValueError):
                    queue_wait = None
            submit_age = None
            if submitted_at_epoch is not None:
                try:
                    submit_age = max(0.0, time.time() - float(submitted_at_epoch))
                except (TypeError, ValueError):
                    submit_age = None

            self.logger.debug(
                f"noun_consensus: processing image {image_id} "
                f"(triggered by {triggering_service})"
            )

            # Fetch noun lists, categories, and subject guesses from all VLM results.
            t0 = time.time()
            service_noun_map, service_category_map, service_subject_map = self._fetch_vlm_nouns(image_id)
            timing['fetch_vlm_nouns'] = time.time() - t0
            raw_noun_count = sum(len(nouns) for nouns in service_noun_map.values())

            # Strip adjective modifiers from non-MWE compound nouns
            # e.g. 'dirt path' -> 'path', 'dirt road' -> 'road'
            # but 'rain forest' -> 'rain forest' (recognized MWE)
            t0 = time.time()
            service_noun_map, service_category_map = apply_mwe_normalization(
                service_noun_map, service_category_map
            )
            timing['mwe_normalization'] = time.time() - t0

            if not service_noun_map:
                processing_time = round(time.time() - start_time, 3)
                self._persist_terminal_noun_consensus_result(
                    image_id=image_id,
                    payload={
                        'service': 'noun_consensus',
                        'status': 'success',
                        'nouns': [],
                        'metadata': {
                            'no_usable_data': True,
                            'reason': 'No VLM noun data yet',
                            'triggered_by': triggering_service,
                            'processed_at': datetime.now().isoformat(),
                        },
                    },
                    processing_time=processing_time,
                    source_service=triggering_service,
                    event_data={'reason': 'No VLM noun data yet'},
                )
                self.logger.debug(
                    f"noun_consensus: no VLM noun data yet for image {image_id}, skipping"
                )
                self._safe_ack(ch, method.delivery_tag)
                self.job_completed_successfully()
                return

            # Collapse synonyms and count votes (ConceptNet via in-memory edges)
            t0 = time.time()
            collapsed = collapse_synonyms(service_noun_map)
            timing['collapse_synonyms'] = time.time() - t0

            # Annotate each canonical noun with its agreed-upon category
            t0 = time.time()
            self._annotate_categories(collapsed, service_category_map)
            timing['annotate_categories'] = time.time() - t0

            # Build category-level vote tally
            t0 = time.time()
            category_tally = self._compute_category_tally(collapsed)
            timing['compute_category_tally'] = time.time() - t0

            services_present = sorted(service_noun_map.keys())
            t0 = time.time()
            self._upsert_noun_consensus(
                image_id, collapsed, services_present,
                processing_time=round(time.time() - start_time, 3),
                category_tally=category_tally,
                commit=False,
            )
            timing['upsert_noun_consensus'] = time.time() - t0

            t0 = time.time()
            self.db_conn.commit()
            timing['commit_consensus_artifact'] = time.time() - t0

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

            downstream_actions = []
            grounding_noun_count = 0

            # Trigger grounding only once the tier's full VLM set has reported.
            # Progressive partial noun sets are too expensive to ground and were
            # causing obvious wasted tail latency.
            if image_transport:
                tier_vlm_count = self._expected_tier_vlm_count(tier) or len(service_noun_map)
                current_vlm_count = len(services_present)
                if current_vlm_count < tier_vlm_count:
                    self.logger.debug(
                        f"noun_consensus: delaying Florence-2 grounding for image {image_id} "
                        f"until full VLM set arrives ({current_vlm_count}/{tier_vlm_count})"
                    )
                else:
                    min_votes = max(2, (tier_vlm_count + 1) // 2)
                    self.logger.debug(
                        f"noun_consensus: grounding threshold for image {image_id}: "
                        f">= {min_votes} votes (tier has {tier_vlm_count} VLMs)"
                    )
                    consensus_nouns = [
                        n['canonical'] for n in collapsed
                        if n.get('vote_count', 0) >= min_votes
                    ]
                    grounding_noun_count = len(consensus_nouns)
                    if consensus_nouns:
                        t0 = time.time()
                        should_trigger_grounding = self._should_trigger_florence2_grounding(image_id, consensus_nouns)
                        timing['check_florence2_grounding'] = timing.get('check_florence2_grounding', 0.0) + (time.time() - t0)
                        t0 = time.time()
                        if should_trigger_grounding:
                            downstream_actions.append(
                                self._build_florence2_grounding_action(
                                    image_id=image_id,
                                    image_transport=image_transport,
                                    nouns=consensus_nouns,
                                    subject_noun=subject_noun,
                                    tier=tier,
                                )
                            )
                        else:
                            self.logger.debug(
                                f"noun_consensus: skipping duplicate Florence-2 grounding trigger "
                                f"for image {image_id} with nouns={sorted(consensus_nouns)}"
                            )
                        timing['build_florence2_grounding'] = timing.get('build_florence2_grounding', 0.0) + (time.time() - t0)
                    else:
                        # No consensus reached — promote the top noun if it's a clear
                        # leader (strictly ahead of second place). Fires only when the
                        # final full-tier path produced nothing, so grounding isn't
                        # flooded with low-confidence partial hits.
                        sorted_nouns = sorted(
                            collapsed, key=lambda n: n.get('confidence', 0), reverse=True
                        )
                        if (len(sorted_nouns) >= 2
                                and sorted_nouns[0].get('confidence', 0)
                                > sorted_nouns[1].get('confidence', 0)):
                            top = sorted_nouns[0]['canonical']
                            grounding_noun_count = 1
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
                                commit=False,
                            )
                            self.db_conn.commit()
                            t0 = time.time()
                            should_trigger_grounding = self._should_trigger_florence2_grounding(image_id, [top])
                            timing['check_florence2_grounding'] = timing.get('check_florence2_grounding', 0.0) + (time.time() - t0)
                            t0 = time.time()
                            if should_trigger_grounding:
                                downstream_actions.append(
                                    self._build_florence2_grounding_action(
                                        image_id=image_id,
                                        image_transport=image_transport,
                                        nouns=[top],
                                        subject_noun=subject_noun,
                                        tier=tier,
                                    )
                                )
                            else:
                                self.logger.debug(
                                    f"noun_consensus: skipping duplicate promoted Florence-2 grounding trigger "
                                    f"for image {image_id} with noun={top}"
                                )
                            timing['build_florence2_grounding'] = timing.get('build_florence2_grounding', 0.0) + (time.time() - t0)

            # Trigger caption_summary only once the tier's full VLM set has
            # reported. Progressive partial caption synthesis is currently
            # costing too much tail latency and creating inconsistent
            # downstream visibility when stragglers arrive.
            if self.config.is_available_for_tier('system.caption_summary', tier):
                expected_vlm_count = self._expected_tier_vlm_count(tier) or len(services_present)
                if len(services_present) < expected_vlm_count:
                    self.logger.debug(
                        f"noun_consensus: delaying caption_summary for image {image_id} "
                        f"until full VLM set arrives ({len(services_present)}/{expected_vlm_count})"
                    )
                else:
                    t0 = time.time()
                    should_trigger_caption = self._should_trigger_caption_summary(image_id, tier)
                    timing['check_caption_summary'] = time.time() - t0
                    t0 = time.time()
                    if should_trigger_caption:
                        downstream_actions.append(
                            self._build_caption_summary_action(
                                image_id=image_id,
                                tier=tier,
                            )
                        )
                    else:
                        self.logger.debug(
                            f"noun_consensus: skipping duplicate caption_summary trigger for image {image_id}"
                        )
                    timing['build_caption_summary'] = time.time() - t0

            # Trigger content_analysis only once the tier's full VLM set has
            # reported. The partial progressive pass was still costing seconds
            # while rarely producing the final answer.
            if self.config.is_available_for_tier('system.content_analysis', tier):
                expected_vlm_count = self._expected_tier_vlm_count(tier) or len(services_present)
                if len(services_present) < expected_vlm_count:
                    self.logger.debug(
                        f"noun_consensus: delaying content_analysis for image {image_id} "
                        f"until full VLM set arrives ({len(services_present)}/{expected_vlm_count})"
                    )
                else:
                    t0 = time.time()
                    should_trigger_content = self._should_trigger_content_analysis(image_id, services_present, tier)
                    timing['check_content_analysis'] = time.time() - t0
                    t0 = time.time()
                    if should_trigger_content:
                        downstream_actions.append(
                            self._build_content_analysis_action(
                                image_id=image_id,
                                services_present=services_present,
                                tier=tier,
                            )
                        )
                    else:
                        self.logger.debug(
                            f"noun_consensus: skipping duplicate content_analysis trigger for image {image_id} "
                            f"with services_present={services_present}"
                        )
                    timing['build_content_analysis'] = time.time() - t0

            publish_timings = self._publish_downstream_actions(downstream_actions)
            for service_name, duration in publish_timings.items():
                timing[f'publish_{service_name}'] = duration

            # The dedupe SELECTs above start a transaction on this non-autocommit
            # connection. Close it before writing the terminal result so
            # result_created / service_events.created_at reflect the actual
            # post-fanout completion time rather than the earlier read phase.
            t0 = time.time()
            self.db_conn.commit()
            timing['commit_post_fanout_reads'] = time.time() - t0

            t0 = time.time()
            self._persist_terminal_noun_consensus_result(
                image_id=image_id,
                payload={
                    'service': 'noun_consensus',
                    'status': 'success',
                    'nouns': collapsed,
                    'category_tally': category_tally,
                    'services_present': services_present,
                    'metadata': {
                        'processed_at': datetime.now().isoformat(),
                        'services_present': services_present,
                    },
                },
                processing_time=round(time.time() - start_time, 3),
                source_service=triggering_service,
                event_data={
                    'services_present': services_present,
                    'downstream_services': [a['service'] for a in downstream_actions if a],
                },
            )
            timing['persist_completion'] = time.time() - t0

            total_duration = time.time() - start_time
            slow_bits = " ".join(
                f"{name}={duration:.3f}s"
                for name, duration in timing.items()
                if duration >= 0.05
            )
            self.logger.info(
                (
                    f"noun_consensus input image={image_id} "
                    f"vlms={len(services_present)} "
                    f"raw_nouns={raw_noun_count} "
                    f"canonical_nouns={len(collapsed)} "
                    f"grounding_nouns={grounding_noun_count}"
                )
            )
            self.logger.info(
                (
                    f"noun_consensus timing image={image_id} "
                    f"{f'queue_wait={queue_wait:.3f}s ' if queue_wait is not None else ''}"
                    f"{f'submit_age={submit_age:.3f}s ' if submit_age is not None else ''}"
                    f"total={total_duration:.3f}s {slow_bits}"
                ).rstrip()
            )

            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()

        except Exception as e:
            try:
                self.db_conn.rollback()
            except Exception:
                pass
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

    def _expected_tier_vlm_count(self, tier: str) -> int:
        """Return the number of VLM services configured for the given tier."""
        return len([
            name for name in self.config.get_services_by_tier(tier)
            if name.startswith('primary.')
            and name.split('.', 1)[1] in VLM_SERVICES
        ])

    def _build_caption_summary_action(self, image_id: int, tier: str) -> dict:
        """Build the downstream caption_summary trigger action."""
        return {
            'image_id': image_id,
            'service': 'caption_summary',
            'source_stage': 'caption_summary_trigger',
            'event_data': {'tier': tier},
            'queue_name': self._get_queue_name('system.caption_summary'),
            'message_body': json.dumps({'image_id': image_id, 'tier': tier}),
            'log_message': f"noun_consensus: triggered caption_summary update for image {image_id}",
        }

    def _build_content_analysis_action(self, image_id: int, services_present: list, tier: str) -> dict:
        """Build the downstream content_analysis trigger action."""
        normalized_services = sorted(services_present or [])
        return {
            'image_id': image_id,
            'service': 'content_analysis',
            'source_stage': 'content_analysis_trigger',
            'event_data': {
                'services_present': normalized_services,
                'tier': tier,
            },
            'queue_name': self._get_queue_name('system.content_analysis'),
            'message_body': json.dumps({
                'image_id': image_id,
                'tier': tier,
                'services_present': normalized_services,
            }),
            'log_message': (
                f"noun_consensus: triggered content_analysis for image {image_id} "
                f"with {len(normalized_services)} VLMs: {normalized_services}"
            ),
        }

    def _should_trigger_content_analysis(self, image_id: int, services_present: list, tier: str) -> bool:
        """Return True unless content_analysis already completed successfully.

        This worker now emits a single downstream fanout from the final noun-consensus
        pass, so tracking in-flight `enqueued` state per service is no longer part
        of the hot path.
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                SELECT 1
                FROM results
                WHERE image_id = %s
                  AND service = 'content_analysis'
                  AND status = 'success'
                LIMIT 1
                """,
                (image_id,),
            )
            exists = cursor.fetchone() is not None
            cursor.close()
            return not exists
        except Exception as e:
            self.logger.warning(
                f"noun_consensus: failed content_analysis dedupe check for image {image_id}: {e}"
            )
            return True

    def _should_trigger_caption_summary(self, image_id: int, tier: str) -> bool:
        """Return True unless caption_summary already completed successfully."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                SELECT 1
                FROM results
                WHERE image_id = %s
                  AND service = 'caption_summary'
                  AND status = 'success'
                ORDER BY result_created DESC
                LIMIT 1
                """,
                (image_id,),
            )
            exists = cursor.fetchone() is not None
            cursor.close()
            return not exists
        except Exception as e:
            self.logger.warning(
                f"noun_consensus: failed caption_summary dedupe check for image {image_id}: {e}"
            )
            return True

    def _build_florence2_grounding_action(self, image_id: int, image_transport: dict, nouns: list, subject_noun: str = None, tier: str = 'free') -> dict:
        """Build the downstream Florence-2 grounding trigger action."""
        if not self.config.is_available_for_tier('primary.florence2', tier):
            self.logger.debug(f"noun_consensus: skipping Florence-2 grounding for tier '{tier}' image {image_id}")
            return None
        return {
            'image_id': image_id,
            'service': 'florence2_grounding',
            'source_stage': 'grounding_trigger',
            'event_data': {
                'nouns': nouns,
                'subject_noun': subject_noun,
                'tier': tier,
            },
            'queue_name': self._get_queue_by_service_type('grounding'),
            'message_body': json.dumps({
                'image_id': image_id,
                'nouns': nouns,
                'subject_noun': subject_noun,
                'triggered_at': datetime.now().isoformat(),
                'tier': tier,
                **(image_transport or {}),
            }),
            'log_message': (
                f"noun_consensus: triggered Florence-2 grounding for image {image_id} "
                f"with {len(nouns)} nouns: {nouns}"
            ),
        }

    def _publish_downstream_actions(self, actions: list) -> dict:
        """Publish all downstream queue messages after their enqueued events commit."""
        timings = {}
        for action in actions:
            if not action:
                continue
            t0 = time.time()
            self._enqueue_publish(action['queue_name'], action['message_body'])
            timings[action['service']] = time.time() - t0
            self.logger.info(action['log_message'])
        return timings


    @contextmanager
    def _image_service_lock(self, image_id: int, lock_key: int):
        cursor = self.db_conn.cursor()
        try:
            cursor.execute("SELECT pg_advisory_lock(%s, %s)", (int(lock_key), int(image_id)))
            yield
        finally:
            try:
                cursor.execute("SELECT pg_advisory_unlock(%s, %s)", (int(lock_key), int(image_id)))
            finally:
                cursor.close()

    def _should_trigger_florence2_grounding(self, image_id: int, nouns: list) -> bool:
        """Return True only when Florence-2 needs a new run for this noun set.

        The current hot path emits a single final noun-consensus fanout and no
        longer records downstream `enqueued` service-events, so dedupe is based
        only on the latest stored Florence-2 result.
        """
        normalized = sorted({n.lower().strip() for n in nouns if n})
        if not normalized:
            return False

        try:
            cursor = self.db_conn.cursor()

            # If the last stored Florence-2 result already used this exact noun set,
            # noun_consensus does not need to enqueue the same work again.
            cursor.execute(
                """
                SELECT data->'metadata'->>'nouns_queried'
                FROM results
                WHERE image_id = %s
                  AND service = 'florence2_grounding'
                ORDER BY result_created DESC
                LIMIT 1
                """,
                (image_id,),
            )
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    previous = sorted({n.lower().strip() for n in json.loads(row[0]) if n})
                    if previous == normalized:
                        cursor.close()
                        return False
                except Exception:
                    pass
            cursor.close()
            return True
        except Exception as e:
            self.logger.warning(
                f"noun_consensus: failed Florence-2 dedupe check for image {image_id}: {e}"
            )
            return True


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
        category_tally: list = None,
        commit: bool = True,
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
                        -- the existing row. Build the old flag map once, then
                        -- join it against each new noun instead of rescanning
                        -- the full old JSON array for every flag on every noun.
                        WITH old_flags AS (
                            SELECT
                                old_n->>'canonical' AS canonical,
                                bool_or(COALESCE((old_n->>'sam3_validated')::boolean, false)) AS sam3_validated,
                                bool_or(COALESCE((old_n->>'grounding_validated')::boolean, false)) AS grounding_validated
                            FROM jsonb_array_elements(noun_consensus.nouns) old_n
                            GROUP BY 1
                        )
                        SELECT COALESCE(
                            jsonb_agg(
                                new_n
                                || CASE
                                    WHEN COALESCE(old_flags.sam3_validated, false)
                                    THEN '{"sam3_validated": true}'::jsonb
                                    ELSE '{}'::jsonb
                                   END
                                || CASE
                                    WHEN COALESCE(old_flags.grounding_validated, false)
                                    THEN '{"grounding_validated": true}'::jsonb
                                    ELSE '{}'::jsonb
                                   END
                            ),
                            '[]'::jsonb
                        )
                        FROM jsonb_array_elements(EXCLUDED.nouns) new_n
                        LEFT JOIN old_flags
                          ON old_flags.canonical = (new_n->>'canonical')
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
            if commit:
                self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(
                f"noun_consensus: upsert failed for image {image_id}: {e}"
            )
            raise

    def _persist_terminal_noun_consensus_result(
        self,
        image_id: int,
        payload: dict,
        processing_time: float,
        source_service: str,
        event_data: dict,
    ):
        """Persist terminal noun_consensus result and completed event in one DB round trip."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                WITH inserted_result AS (
                    INSERT INTO results (
                        image_id, service, data, status, worker_id, processing_time
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING 1
                )
                INSERT INTO service_events (
                    image_id, service, event_type, source_service, source_stage, data
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    image_id,
                    'noun_consensus',
                    json.dumps(payload),
                    payload.get('status', 'success') or 'success',
                    self.worker_id,
                    processing_time,
                    image_id,
                    'noun_consensus',
                    'completed',
                    source_service,
                    'noun_consensus_run',
                    json.dumps(event_data),
                ),
            )
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(
                f"noun_consensus: failed to persist terminal result for image {image_id}: {e}"
            )
            raise


if __name__ == "__main__":
    worker = NounConsensusWorker()
    worker.start()
