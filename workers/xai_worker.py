#!/usr/bin/env python3
"""
XaiWorker - xAI Grok VLM service worker

Intercepts NSFW images before calling the xAI API by checking NudeNet
results first. If explicit content is detected, a synthetic blocked result
is stored and the xAI API is never called.
"""
import sys
import os
import json
import time
import logging
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker
from datetime import datetime

logger = logging.getLogger(__name__)

# NudeNet labels that indicate content xAI should not describe.
# MALE_BREAST_EXPOSED (shirtless) and face labels are not included.
NSFW_LABELS = frozenset({
    'FEMALE_BREAST_EXPOSED',
    'FEMALE_GENITALIA_EXPOSED',
    'MALE_GENITALIA_EXPOSED',
    'BUTTOCKS_EXPOSED',
    'ANUS_EXPOSED',
})

BLOCKED_TEXT = "[NSFW — not sent to xAI API]"

NUDENET_TIMEOUT  = 5    # seconds to wait for NudeNet result (normally ~23ms)
NUDENET_POLL     = 0.1  # seconds between polls


class XaiWorker(BaseWorker):
    """Worker for xAI Grok VLM service"""

    def __init__(self):
        super().__init__('primary.xai')

    def process_message(self, ch, method, properties, body):
        if not self.ensure_database_connection():
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self.job_failed("Database unavailable")
            return

        message  = json.loads(body)
        image_id = message['image_id']

        # Wait for NudeNet result before calling the xAI API.
        # NudeNet runs in ~23ms so it is almost always done by the time
        # this worker picks up the message.
        predictions = self._wait_for_nudenet(image_id)

        if predictions is None:
            self.logger.warning(
                f"xai: NudeNet result not available for image {image_id} "
                f"after {NUDENET_TIMEOUT}s — skipping xAI API call"
            )
            self._store_gate_result(image_id, "NudeNet unavailable", status='failed', blocked=False)
            self._update_service_dispatch(
                image_id,
                service='xai',
                status='failed',
                reason='NudeNet unavailable',
            )
            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()
            return

        detected = {p.get('label') for p in predictions} & NSFW_LABELS
        if detected:
            self.logger.info(
                f"xai: NSFW detected for image {image_id} "
                f"({', '.join(sorted(detected))}) — skipping xAI API call"
            )
            self._store_gate_result(image_id, BLOCKED_TEXT, status='success', blocked=True)
            self._update_service_dispatch(image_id, service='xai')
            self._fire_downstream(image_id, message)
            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()
            return

        # SFW — proceed with normal base worker pipeline
        super().process_message(ch, method, properties, body)

    def _wait_for_nudenet(self, image_id):
        """Poll for NudeNet result up to NUDENET_TIMEOUT seconds.

        Returns the predictions list on success, None on timeout.
        """
        deadline = time.time() + NUDENET_TIMEOUT
        while time.time() < deadline:
            result = self._fetch_nudenet(image_id)
            if result is not None:
                return result
            time.sleep(NUDENET_POLL)
        return None

    def _fetch_nudenet(self, image_id):
        """Return NudeNet predictions list if result is in DB, else None."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """SELECT data FROM results
                   WHERE image_id = %s AND service = 'nudenet' AND status = 'success'
                   LIMIT 1""",
                (image_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            if row is None:
                return None
            data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            return data.get('predictions', [])
        except Exception as e:
            self.logger.error(f"xai: nudenet fetch error for image {image_id}: {e}")
            return None

    def _store_gate_result(self, image_id, reason, status='success', blocked=True):
        """Store a synthetic gating result so the pipeline continues cleanly."""
        result = {
            'status':  status,
            'service': 'xai',
            'predictions': [{'text': BLOCKED_TEXT}],
            'nouns': [],
            'metadata': {
                'blocked': blocked,
                'reason':  reason,
                'processed_at': datetime.now().isoformat(),
            },
        }
        self._store_terminal_service_result(
            image_id,
            result,
            status=status,
            service='xai',
        )

    def _fire_downstream(self, image_id, message):
        """Trigger noun_consensus and verb_consensus so the pipeline doesn't stall."""
        self.trigger_noun_consensus(image_id, message)
        self.trigger_verb_consensus(image_id, message)


if __name__ == "__main__":
    worker = XaiWorker()
    worker.start()
