#!/usr/bin/env python3
"""
Florence2GroundingWorker - Phrase grounding triggered by noun consensus.

Receives image data + consensus nouns from noun_consensus_worker, constructs
a comma-separated noun prompt, calls Florence-2 CAPTION_TO_PHRASE_GROUNDING,
stores bbox results in the results table, and writes grounding_validated=true
on matched nouns in noun_consensus.

Dedup: skips re-running if the noun set is identical to the last grounding
call for this image (noun_consensus triggers progressively as VLMs complete).

Pattern mirrors sam3_worker: triggered internally, not by the producer.
"""

import base64
import io
import json
import os
import sys
import time
import requests

sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker


class Florence2GroundingWorker(BaseWorker):
    """Phrase grounding via Florence-2 CAPTION_TO_PHRASE_GROUNDING."""

    def __init__(self):
        super().__init__('system.florence2_grounding')

    def process_message(self, ch, method, properties, body):
        start_time = time.time()

        try:
            if not self.ensure_database_connection():
                self._safe_nack(ch, method.delivery_tag, requeue=False)
                self.job_failed("Database unavailable")
                return

            message = json.loads(body)
            image_id = message['image_id']
            image_data = message.get('image_data')
            nouns = message.get('nouns', [])
            tier = message.get('tier', 'free')

            if not image_data:
                self.logger.error(
                    f"florence2_grounding: no image_data in message for image {image_id}"
                )
                self._safe_nack(ch, method.delivery_tag, requeue=False)
                self.job_failed("No image data")
                return

            if not nouns:
                self.logger.info(
                    f"florence2_grounding: no nouns to ground for image {image_id}, skipping"
                )
                self._safe_ack(ch, method.delivery_tag)
                return

            # Dedup: skip if already grounded with this exact noun set
            previous_nouns = self._fetch_previous_nouns(image_id)
            if set(nouns) == previous_nouns:
                self.logger.info(
                    f"florence2_grounding: image {image_id} already grounded "
                    f"with same {len(nouns)} nouns, skipping"
                )
                self._safe_ack(ch, method.delivery_tag)
                return

            prompt = self._build_grounding_prompt(nouns)
            self.logger.info(
                f"florence2_grounding: image {image_id} — "
                f"{len(nouns)} nouns, prompt: \"{prompt}\""
            )

            result = self._call_florence2_grounding(image_data, prompt)
            if result is None:
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                self.job_failed("Florence-2 grounding call failed")
                return

            processing_time = round(time.time() - start_time, 3)

            # Embed the noun set in metadata so dedup can read it back next trigger
            result.setdefault('metadata', {})['nouns_queried'] = json.dumps(nouns)

            self._store_result(image_id, result, processing_time)
            self._update_service_dispatch(image_id)

            # Mark nouns that were successfully grounded back in noun_consensus
            grounded_labels = [
                p['label'].lower().strip()
                for p in result.get('predictions', [])
                if p.get('label')
            ]
            if grounded_labels:
                self._validate_noun_consensus(image_id, nouns, grounded_labels)

            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()

            self.logger.info(
                f"florence2_grounding: image {image_id} — "
                f"{len(result.get('predictions', []))} regions grounded "
                f"from {len(nouns)} nouns in {processing_time}s"
            )

        except Exception as e:
            self.logger.error(f"florence2_grounding: error processing message: {e}")
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self.job_failed(str(e))

    def _build_grounding_prompt(self, nouns: list) -> str:
        """Build a comma-separated noun prompt for CAPTION_TO_PHRASE_GROUNDING.

        Using a bare noun list rather than "an image containing ..." avoids
        Florence-2 grounding the intro phrase itself as a spurious full-image bbox.
        """
        if not nouns:
            return ""
        if len(nouns) == 1:
            return nouns[0]
        if len(nouns) == 2:
            return f"{nouns[0]} and {nouns[1]}"
        return f"{', '.join(nouns[:-1])}, and {nouns[-1]}"

    def _fetch_previous_nouns(self, image_id: int) -> set:
        """Return the noun set from the most recent grounding result for this image.

        Returns an empty set if no prior result exists.
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                SELECT data->'metadata'->>'nouns_queried'
                FROM results
                WHERE image_id = %s AND service = 'florence2_grounding'
                ORDER BY result_created DESC
                LIMIT 1
                """,
                (image_id,),
            )
            row = cursor.fetchone()
            cursor.close()
            if not row or not row[0]:
                return set()
            return set(json.loads(row[0]))
        except Exception as e:
            self.logger.warning(
                f"florence2_grounding: could not fetch previous nouns for image {image_id}: {e}"
            )
            return set()

    def _call_florence2_grounding(self, image_data_b64: str, prompt: str):
        """POST image + prompt to Florence-2 CAPTION_TO_PHRASE_GROUNDING."""
        try:
            image_bytes = base64.b64decode(image_data_b64)
            files = {'file': ('image.jpg', io.BytesIO(image_bytes), 'image/jpeg')}
            url = f"http://{self.service_host}:{self.service_port}/v3/analyze"
            resp = requests.post(
                url,
                params={'task': 'CAPTION_TO_PHRASE_GROUNDING', 'text': prompt},
                files=files,
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            if payload.get('status') != 'success':
                self.logger.error(
                    f"florence2_grounding: service returned non-success: "
                    f"{payload.get('error')}"
                )
                return None
            return payload
        except Exception as e:
            self.logger.error(f"florence2_grounding: REST call failed: {e}")
            return None

    def _store_result(self, image_id: int, data: dict, processing_time: float):
        """Insert Florence-2 grounding result into the results table."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO results
                    (image_id, service, data, status, worker_id, processing_time)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    image_id,
                    self._get_clean_service_name(),
                    json.dumps(data),
                    'success',
                    self.worker_id,
                    processing_time,
                ),
            )
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(
                f"florence2_grounding: failed to store result for image {image_id}: {e}"
            )
            raise

    def _validate_noun_consensus(
        self, image_id: int, queried_nouns: list, grounded_labels: list
    ):
        """Set grounding_validated=true on noun_consensus nouns that Florence-2 grounded.

        A noun is validated if its canonical form appears (case-insensitive) in
        the grounded label set returned by Florence-2. Nouns not found are left
        untagged — no invalidation.
        """
        grounded_set = set(grounded_labels)

        # Match each queried noun against grounded labels
        validated = [n for n in queried_nouns if n.lower() in grounded_set]
        if not validated:
            self.logger.info(
                f"florence2_grounding: no noun matches found for image {image_id} "
                f"(queried: {queried_nouns}, grounded: {list(grounded_set)})"
            )
            return

        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                UPDATE noun_consensus
                SET nouns = (
                    SELECT jsonb_agg(
                        CASE
                            WHEN n->>'canonical' = ANY(%s)
                            THEN n || '{"grounding_validated": true}'::jsonb
                            ELSE n
                        END
                    )
                    FROM jsonb_array_elements(nouns) n
                ),
                updated_at = NOW()
                WHERE image_id = %s
                """,
                (validated, image_id),
            )
            self.db_conn.commit()
            cursor.close()
            self.logger.info(
                f"florence2_grounding: validated {len(validated)} nouns "
                f"for image {image_id}: {validated}"
            )
        except Exception as e:
            self.logger.error(
                f"florence2_grounding: failed to write validation "
                f"for image {image_id}: {e}"
            )


if __name__ == '__main__':
    worker = Florence2GroundingWorker()
    worker.start()
