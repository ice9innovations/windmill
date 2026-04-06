#!/usr/bin/env python3
"""
PostprocessingOrchestratorWorker - lightweight task-driven post fanout.

Consumes tiny task messages and enqueues the relevant downstream services:
- primary_complete -> content_analysis
- nouns_ready -> florence2_grounding, caption_summary
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker
from core.postgres_connection import close_quietly, commit_if_needed, rollback_quietly

logger = logging.getLogger(__name__)


class PostprocessingOrchestratorWorker(BaseWorker):
    def __init__(self):
        super().__init__('system.postprocessing_orchestrator')
        # results.worker_id is varchar(50); the default generated id for this
        # long service name can exceed that limit and poison-loop the queue.
        self.worker_id = f"worker_postproc_orch_{int(time.time())}"

    def connect_to_database(self):
        try:
            return self._connect_main_database(autocommit=False)
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False

    def _declare_additional_queues(self, declare_queue):
        declare_queue(self._get_queue_name('system.content_analysis'))
        declare_queue(self._get_queue_by_service_type('grounding'))
        declare_queue(self._get_queue_name('system.caption_summary'))

    def process_message(self, ch, method, properties, body):
        start_time = time.time()
        timing = {}

        try:
            if not self.ensure_database_connection():
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                self.job_failed("Database unavailable")
                return

            message = json.loads(body)
            image_id = message['image_id']
            tier = message.get('tier', 'free')
            task_type = message.get('task_type')
            submitted_at_epoch = message.get('submitted_at_epoch')
            submit_age = None
            if submitted_at_epoch is not None:
                try:
                    submit_age = max(0.0, time.time() - float(submitted_at_epoch))
                except (TypeError, ValueError):
                    submit_age = None

            triggered = []

            if task_type == 'primary_complete':
                t0 = time.time()
                if (
                    self.config.is_available_for_tier('system.content_analysis', tier)
                    and not self._has_success_result(image_id, 'content_analysis')
                ):
                    services_present = sorted(message.get('services_present') or [])
                    self._enqueue_publish(
                        self._get_queue_name('system.content_analysis'),
                        json.dumps({
                            'image_id': image_id,
                            'tier': tier,
                            'services_present': services_present,
                        }),
                    )
                    triggered.append('content_analysis')
                timing['primary_complete'] = time.time() - t0

            elif task_type == 'nouns_ready':
                image_transport = self._image_transport_fields(message)
                consensus_nouns = list(message.get('consensus_nouns') or [])
                subject_noun = message.get('subject_noun')

                t0 = time.time()
                if (
                    consensus_nouns
                    and self.config.is_available_for_tier('primary.florence2', tier)
                    and self._should_trigger_grounding(image_id, consensus_nouns)
                ):
                    self._enqueue_publish(
                        self._get_queue_by_service_type('grounding'),
                        json.dumps({
                            'image_id': image_id,
                            'nouns': consensus_nouns,
                            'subject_noun': subject_noun,
                            'triggered_at': datetime.now().isoformat(),
                            'tier': tier,
                            **(image_transport or {}),
                        }),
                    )
                    triggered.append('florence2_grounding')

                if (
                    self.config.is_available_for_tier('system.caption_summary', tier)
                    and not self._has_success_result(image_id, 'caption_summary')
                ):
                    self._enqueue_publish(
                        self._get_queue_name('system.caption_summary'),
                        json.dumps({'image_id': image_id, 'tier': tier}),
                    )
                    triggered.append('caption_summary')
                timing['nouns_ready'] = time.time() - t0

            t0 = time.time()
            self._persist_terminal_result(
                image_id=image_id,
                payload={
                    'service': 'postprocessing_orchestrator',
                    'status': 'success',
                    'task_type': task_type,
                    'triggered_services': triggered,
                    'metadata': {'processed_at': datetime.now().isoformat()},
                },
                processing_time=round(time.time() - start_time, 3),
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
                    f"postprocessing_orchestrator timing image={image_id} "
                    f"task={task_type} "
                    f"{f'submit_age={submit_age:.3f}s ' if submit_age is not None else ''}"
                    f"total={total_duration:.3f}s {slow_bits}"
                ).rstrip()
            )
            if triggered:
                self.logger.info(
                    f"postprocessing_orchestrator: image {image_id} task={task_type} triggered {triggered}"
                )

            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()
        except Exception as e:
            rollback_quietly(self.db_conn)
            self.logger.error(f"postprocessing_orchestrator: error processing message: {e}")
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self.job_failed(str(e))

    def _has_success_result(self, image_id: int, service: str) -> bool:
        cursor = self.db_conn.cursor()
        try:
            cursor.execute(
                """
                SELECT 1
                FROM results
                WHERE image_id = %s
                  AND service = %s
                  AND status = 'success'
                LIMIT 1
                """,
                (image_id, service),
            )
            return cursor.fetchone() is not None
        finally:
            cursor.close()

    def _should_trigger_grounding(self, image_id: int, nouns: list) -> bool:
        normalized = sorted({n.lower().strip() for n in nouns if n})
        if not normalized:
            return False
        cursor = self.db_conn.cursor()
        try:
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
                    return previous != normalized
                except Exception:
                    return True
            return True
        finally:
            cursor.close()

    def _persist_terminal_result(self, image_id: int, payload: dict, processing_time: float):
        cursor = self.db_conn.cursor()
        try:
            cursor.execute(
                """
                WITH inserted_result AS (
                    INSERT INTO results (image_id, service, data, status, worker_id, processing_time)
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
                    'postprocessing_orchestrator',
                    json.dumps(payload),
                    payload.get('status', 'success') or 'success',
                    self.worker_id,
                    processing_time,
                    image_id,
                    'postprocessing_orchestrator',
                    'completed',
                    'postprocessing_orchestrator',
                    'postprocessing_orchestrator_run',
                    json.dumps({
                        'task_type': payload.get('task_type'),
                        'triggered_services': payload.get('triggered_services') or [],
                    }),
                ),
            )
            commit_if_needed(self.db_conn, force=True)
        finally:
            close_quietly(cursor)


if __name__ == "__main__":
    worker = PostprocessingOrchestratorWorker()
    worker.start()
