#!/usr/bin/env python3
"""
RembgWorker - Background removal for analyzed images.

Consumes requests from the 'rembg' queue. For each request:
  1. POSTs the raw image to the rembg Animal Farm service (BEN2 model).
  2. Writes the resulting alpha matte to rembg_results.

Triggered on-demand by ice9-api when a user requests a matte —
not automatically after every pipeline run.
"""

import base64
import io
import json
import os
import sys
import time
import logging
import requests

sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

logger = logging.getLogger(__name__)

REMBG_TIMEOUT = int(os.getenv('REMBG_TIMEOUT', 60))


class RembgWorker(BaseWorker):
    """
    Calls the rembg Animal Farm service (BEN2) and stores the resulting
    alpha matte in rembg_results.
    """

    def __init__(self):
        super().__init__('system.rembg')
        cfg = self.config.get_service_config('system.rembg')
        self.rembg_url = f"http://{cfg['host']}:{cfg['port']}{cfg['endpoint']}"

    def process_message(self, ch, method, properties, body):
        start_time = time.time()

        try:
            if not self.ensure_database_connection():
                self._safe_nack(ch, method.delivery_tag, requeue=False)
                self.job_failed("Database unavailable")
                return

            message = json.loads(body)
            image_id = message['image_id']
            image_bytes = self.resolve_image_bytes(message, required=True)

            if not image_bytes:
                self.logger.error(f"rembg: no usable image payload in message for image {image_id}")
                self._safe_nack(ch, method.delivery_tag, requeue=False)
                self.job_failed("No image data")
                return

            # Dedup guard — skip if already processed
            cursor = self.db_conn.cursor()
            cursor.execute(
                "SELECT 1 FROM rembg_results WHERE image_id = %s LIMIT 1",
                (image_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            if row:
                self.logger.info(f"rembg: image {image_id} already processed, skipping")
                self._safe_ack(ch, method.delivery_tag)
                return

            # Call rembg service — BEN2 handles subject isolation internally
            result = self._call_rembg(image_bytes, image_id)
            if result is None:
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                self.job_failed("rembg service call failed")
                return

            if result.get('status', 'success') != 'success':
                self._store_terminal_service_result(
                    image_id,
                    result,
                    status=result.get('status', 'failed') or 'failed',
                    processing_time=round(time.time() - start_time, 3),
                )
                self._safe_ack(ch, method.delivery_tag)
                self.job_completed_successfully()
                self.logger.warning(
                    f"rembg: image {image_id} returned terminal non-success: "
                    f"{result.get('error_message') or result.get('error') or result.get('message') or 'no reason provided'}"
                )
                return

            png_b64 = result['mask']
            width = result['metadata']['width']
            height = result['metadata']['height']
            model = result['metadata'].get('model_info', {}).get('model', 'unknown')
            processing_time = round(time.time() - start_time, 3)

            self._upsert(image_id, png_b64, height, width, model, False, processing_time)

            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()
            self.logger.info(
                f"rembg: image {image_id} {width}x{height} "
                f"model={model} {processing_time}s"
            )

        except Exception as e:
            self.logger.error(f"rembg: error processing message: {e}")
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self.job_failed(str(e))

    def _call_rembg(self, image_bytes: bytes, image_id: int):
        """POST image to rembg service, return parsed response or None on failure."""
        try:
            resp = requests.post(
                self.rembg_url,
                files={'file': ('image.png', io.BytesIO(image_bytes), 'image/png')},
                timeout=REMBG_TIMEOUT,
            )
            return self._coerce_terminal_http_response(resp)
        except requests.RequestException as e:
            if getattr(e, 'response', None) is not None:
                return self._coerce_terminal_http_response(e.response)
            self.logger.error(
                f"rembg: service call failed for image {image_id} before terminal response: {e}"
            )
            return None

    def _upsert(
        self,
        image_id: int,
        png_b64: str,
        height: int,
        width: int,
        model: str,
        premasked: bool,
        processing_time: float,
    ):
        """Insert or update the rembg_results row for this image."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO rembg_results
                    (image_id, png_b64, shape, model, premasked,
                     processing_time, created_at, updated_at)
                VALUES (%s, %s, %s::jsonb, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (image_id) DO UPDATE SET
                    png_b64         = EXCLUDED.png_b64,
                    shape           = EXCLUDED.shape,
                    model           = EXCLUDED.model,
                    premasked       = EXCLUDED.premasked,
                    processing_time = EXCLUDED.processing_time,
                    updated_at      = NOW()
                """,
                (
                    image_id,
                    png_b64,
                    json.dumps([height, width]),
                    model,
                    premasked,
                    processing_time,
                )
            )
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"rembg: upsert failed for image {image_id}: {e}")
            raise


if __name__ == '__main__':
    worker = RembgWorker()
    worker.start()
