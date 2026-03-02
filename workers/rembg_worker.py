#!/usr/bin/env python3
"""
RembgWorker - Background removal for analyzed images.

Consumes requests from the 'rembg' queue. For each request:
  1. Fetches the SAM3 _subject mask (if available) and pre-masks the image
     so the rembg service focuses on edge cleanup rather than subject/scene
     disambiguation. Falls back to raw image if _subject is absent.
  2. POSTs the (pre-masked) image to the rembg Animal Farm service.
  3. Writes the resulting alpha matte to rembg_results.

Triggered on-demand by ice9-api when a user requests a matte —
not automatically after every SAM3 run.
"""

import base64
import io
import json
import os
import sys
import time
import logging
import requests
import numpy as np
import pika
from PIL import Image

sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

logger = logging.getLogger(__name__)

REMBG_TIMEOUT = int(os.getenv('REMBG_TIMEOUT', 60))


class RembgWorker(BaseWorker):
    """
    Calls the rembg Animal Farm service and stores the resulting alpha matte
    in rembg_results. Pre-masks the image with the SAM3 subject mask when
    available so the model focuses on edge refinement rather than subject
    identification.
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
            image_data = message.get('image_data')

            if not image_data:
                self.logger.error(f"rembg: no image_data in message for image {image_id}")
                self._safe_nack(ch, method.delivery_tag, requeue=False)
                self.job_failed("No image data")
                return

            # Dedup guard — skip if already processed with pre-masking.
            # A blind first-pass result (premasked=false) is always worth
            # overwriting once SAM3 has a subject mask available.
            cursor = self.db_conn.cursor()
            cursor.execute(
                "SELECT premasked FROM rembg_results WHERE image_id = %s LIMIT 1",
                (image_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            if row and row[0]:
                # Already have a pre-masked result — nothing to improve
                self.logger.info(
                    f"rembg: image {image_id} already has pre-masked result, skipping"
                )
                self._safe_ack(ch, method.delivery_tag)
                return
            # row is None (no result yet) or row[0] is False (blind pass) — proceed

            image_bytes = base64.b64decode(image_data)

            # Pre-mask with _subject if available — eliminates scene/background before
            # rembg sees the image so it focuses entirely on edge cleanup
            premasked = False
            subject = self._fetch_subject(image_id)
            if subject:
                try:
                    image_bytes = self._apply_subject_mask(image_bytes, subject)
                    premasked = True
                    self.logger.info(
                        f"rembg: pre-masked image {image_id} "
                        f"with _subject ('{subject.get('noun')}')"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"rembg: pre-mask failed for image {image_id}, "
                        f"using raw image: {e}"
                    )
            else:
                self.logger.info(
                    f"rembg: no _subject for image {image_id}, sending raw image"
                )

            # Call rembg service
            result = self._call_rembg(image_bytes, image_id)
            if not result:
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                self.job_failed("rembg service call failed")
                return

            png_b64 = result['mask']
            width = result['metadata']['width']
            height = result['metadata']['height']
            model = result['metadata'].get('model_info', {}).get('model', 'unknown')
            processing_time = round(time.time() - start_time, 3)

            self._upsert(image_id, png_b64, height, width, model, premasked, processing_time)

            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()
            self.logger.info(
                f"rembg: image {image_id} {width}x{height} "
                f"premasked={premasked} model={model} {processing_time}s"
            )

        except Exception as e:
            self.logger.error(f"rembg: error processing message: {e}")
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self.job_failed(str(e))

    def _fetch_subject(self, image_id: int):
        """Return the _subject dict from sam3_results, or None if absent."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                "SELECT data->'_subject' FROM sam3_results WHERE image_id = %s",
                (image_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            if row and row[0] and row[0].get('instances'):
                return row[0]
        except Exception as e:
            self.logger.warning(
                f"rembg: could not fetch _subject for image {image_id}: {e}"
            )
        return None

    def _apply_subject_mask(self, image_bytes: bytes, subject: dict) -> bytes:
        """
        Zero out background pixels using the _subject RLE mask.

        Returns the pre-masked image as PNG bytes. The image is resized to
        match the mask dimensions if they differ (SAM3 rescales to 1008px).
        """
        instances = subject.get('instances', [])
        first = instances[0]
        h, w = first['mask_shape']

        # Union all subject instance masks
        union = np.zeros(h * w, dtype=bool)
        for inst in instances:
            rle = inst.get('mask_rle', [])
            if not rle:
                continue
            flat = np.zeros(h * w, dtype=bool)
            pos, val = 0, False
            for count in rle:
                flat[pos:pos + count] = val
                pos += count
                val = not val
            union |= flat

        img = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
        img = img.resize((w, h), Image.LANCZOS)
        pixels = np.array(img)
        pixels[~union.reshape(h, w), 3] = 0  # background → fully transparent

        buf = io.BytesIO()
        Image.fromarray(pixels, 'RGBA').save(buf, format='PNG')
        return buf.getvalue()

    def _call_rembg(self, image_bytes: bytes, image_id: int):
        """POST image to rembg service, return parsed response or None on failure."""
        try:
            resp = requests.post(
                self.rembg_url,
                files={'file': ('image.png', io.BytesIO(image_bytes), 'image/png')},
                timeout=REMBG_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get('status') != 'success':
                self.logger.error(
                    f"rembg: service returned error for image {image_id}: "
                    f"{data.get('error')}"
                )
                return None
            return data
        except Exception as e:
            self.logger.error(
                f"rembg: service call failed for image {image_id}: {e}"
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
