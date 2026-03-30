#!/usr/bin/env python3
"""
SAM3Worker - Text-prompted segmentation using noun consensus results.

Triggered by noun_consensus_worker after all VLM services have reported.
Receives image data + consensus nouns, calls the SAM3 REST service, and
stores bounding boxes and masks in the sam3_results table.
"""

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
from noun_extractor import categorize_nouns

logger = logging.getLogger(__name__)

SAM3_INPUT_SIZE = 1008  # SAM3's native resolution (divisible by patch size 14)

SAM3_HOST = os.getenv('SAM3_HOST', 'localhost')
SAM3_PORT = int(os.getenv('SAM3_PORT', 9779))
SAM3_ENDPOINT = f'http://{SAM3_HOST}:{SAM3_PORT}/analyze'
SAM3_TIMEOUT = int(os.getenv('SAM3_TIMEOUT', 120))


class Sam3Worker(BaseWorker):
    """
    Calls the SAM3 REST service with consensus nouns and stores segmentation
    results in sam3_results.
    """

    def __init__(self):
        super().__init__('system.sam3')

    def process_message(self, ch, method, properties, body):
        """Override base process_message — calls SAM3 REST directly."""
        start_time = time.time()

        try:
            if not self.ensure_database_connection():
                self._safe_nack(ch, method.delivery_tag, requeue=False)
                self.job_failed("Database unavailable")
                return

            message = json.loads(body)
            image_id = message['image_id']
            image_bytes = self.resolve_image_bytes(message, required=True)
            nouns = message.get('nouns', [])
            subject_noun = message.get('subject_noun')
            tier = message.get('tier', 'free')

            if not image_bytes:
                self.logger.error(f"sam3: no image bytes in message for image {image_id}")
                self._safe_nack(ch, method.delivery_tag, requeue=False)
                self.job_failed("No image bytes")
                return

            if not nouns:
                self._store_terminal_service_result(
                    image_id,
                    {
                        'service': 'sam3',
                        'status': 'success',
                        'results': {},
                        'metadata': {
                            'no_usable_data': True,
                            'reason': 'No nouns to query',
                        },
                    },
                    processing_time=round(time.time() - start_time, 3),
                )
                self.logger.info(f"sam3: no nouns to query for image {image_id}, skipping")
                self._safe_ack(ch, method.delivery_tag)
                return

            # Additive dedup: fetch any existing results so we only segment
            # nouns SAM3 hasn't already processed for this image.
            existing_nouns, existing_data = self._fetch_existing_results(image_id)
            new_nouns = [n for n in nouns if n not in set(existing_nouns)]

            if not new_nouns:
                self._store_terminal_service_result(
                    image_id,
                    {
                        'service': 'sam3',
                        'status': 'success',
                        'results': existing_data,
                        'metadata': {
                            'no_usable_data': True,
                            'reason': 'No new nouns to process',
                            'nouns_queried': nouns,
                            'existing_nouns': existing_nouns,
                        },
                    },
                    processing_time=round(time.time() - start_time, 3),
                )
                self.logger.info(
                    f"sam3: image {image_id} already processed with same nouns, skipping"
                )
                self._safe_ack(ch, method.delivery_tag)
                return

            self.logger.info(
                f"sam3: processing image {image_id} — "
                f"new nouns: {new_nouns}"
                + (f" (adding to existing: {existing_nouns})" if existing_nouns else "")
            )

            # Call SAM3 only for the new nouns
            sam3_response = self._call_sam3(image_bytes, new_nouns)
            if sam3_response is None:
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                self.job_failed("SAM3 REST call failed")
                return
            if sam3_response.get('status', 'success') != 'success':
                self._store_terminal_service_result(
                    image_id,
                    sam3_response,
                    status=sam3_response.get('status', 'failed') or 'failed',
                    processing_time=round(time.time() - start_time, 3),
                )
                self._safe_ack(ch, method.delivery_tag)
                self.job_completed_successfully()
                self.logger.warning(
                    f"sam3: image {image_id} returned terminal non-success: "
                    f"{sam3_response.get('error_message') or sam3_response.get('error') or sam3_response.get('message') or 'no reason provided'}"
                )
                return
            new_results = sam3_response.get('results', {})

            # Merge with existing noun results. Strip underscore keys from the
            # existing data — _subject and _background will be recomputed below
            # from the full merged set so they always reflect all known nouns.
            results = {k: v for k, v in existing_data.items() if not k.startswith('_')}
            results.update(new_results)
            all_nouns = existing_nouns + new_nouns

            # Subject mask: store the best segmentation of the voted subject noun
            # under "_subject" so downstream consumers can invert it for background
            # removal without reconstructing it from the full noun results.
            #
            # Fallback chain when the winning subject noun yields no instances:
            #   1. Reuse from main call if already queried and has instances
            #   2. Fresh focused SAM3 call if noun wasn't in the main call
            #   3. Category-matched fallback: best validated noun in the same category
            #   4. Largest single bbox across all validated nouns (last resort)
            if subject_noun:
                if subject_noun in results and results[subject_noun].get('instances'):
                    # Subject noun was in the main call and has instances — reuse.
                    results['_subject'] = {'instances': results[subject_noun]['instances'], 'noun': subject_noun}
                    self.logger.info(
                        f"sam3: subject '{subject_noun}' reused from main call "
                        f"({len(results[subject_noun]['instances'])} instance(s)) "
                        f"for image {image_id}"
                    )
                elif subject_noun not in results:
                    # Subject noun wasn't queried in the main call — try a focused call.
                    # (e.g. "person" voted when only "woman" was in noun consensus)
                    subject_response = self._call_sam3(image_bytes, [subject_noun])
                    subject_result = (
                        subject_response.get('results', {})
                        if isinstance(subject_response, dict) and subject_response.get('status', 'success') == 'success'
                        else {}
                    )
                    if subject_result and subject_result.get(subject_noun, {}).get('instances'):
                        instances = subject_result[subject_noun]['instances']
                        results['_subject'] = {'instances': instances, 'noun': subject_noun}
                        self.logger.info(
                            f"sam3: subject probe '{subject_noun}' returned "
                            f"{len(instances)} instance(s) for image {image_id}"
                        )
                    else:
                        self.logger.info(
                            f"sam3: subject probe '{subject_noun}' returned no instances "
                            f"for image {image_id}, trying fallback"
                        )
                        self._apply_subject_fallback(results, subject_noun, image_id)
                else:
                    # Subject noun was queried but returned 0 instances — SAM3 already
                    # tried and failed for this noun.  Skip the fresh call (it would
                    # time out identically) and go straight to the fallback chain.
                    self.logger.info(
                        f"sam3: subject '{subject_noun}' queried but returned 0 instances "
                        f"for image {image_id}, trying fallback"
                    )
                    self._apply_subject_fallback(results, subject_noun, image_id)

            # Background mask: union all subject instance masks and invert.
            # Stored under "_background" so downstream consumers get a ready-made
            # mask without needing to understand the full noun result structure.
            background = self._compute_background_mask(results)
            if background:
                results['_background'] = background
                self.logger.info(
                    f"sam3: background mask computed for image {image_id} "
                    f"({background['rle_runs']} RLE runs)"
                )

            processing_time = round(time.time() - start_time, 3)

            # Store full merged results
            self._upsert_sam3_results(image_id, all_nouns, results, processing_time)
            self._store_terminal_service_result(
                image_id,
                {
                    'service': 'sam3',
                    'status': 'success',
                    'results': results,
                    'nouns_queried': all_nouns,
                    'metadata': {
                        'processed_at': time.time(),
                    },
                },
                processing_time=processing_time,
            )

            # Mark all pending SAM3 dispatch records complete for this image
            self._update_service_dispatch(image_id)

            # Mark validated nouns back in noun_consensus
            self._validate_noun_consensus(image_id, results)

            # Trigger rembg refinement if _subject was set — re-runs background
            # removal with the subject mask applied, improving on any blind first pass.
            # Disabled: rembg is now triggered at upload time in api.py instead.
            # if '_subject' in results:
            #     self._trigger_rembg(image_id, image_bytes)

            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()

            total_instances = sum(
                len(v.get('instances', [])) for k, v in results.items()
                if not k.startswith('_')
            )
            self.logger.info(
                f"sam3: image {image_id} - {len(nouns)} nouns, "
                f"{total_instances} instances, {processing_time}s"
            )

        except Exception as e:
            self.logger.error(f"sam3: error processing message: {e}")
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self.job_failed(str(e))

    def _call_sam3(self, image_bytes: bytes, nouns: list):
        """POST image + nouns to SAM3 REST service, return a terminal payload."""
        try:
            prepared_bytes, scale, orig_w, orig_h = self._prepare_image(image_bytes)

            files = {'file': ('image.jpg', io.BytesIO(prepared_bytes), 'image/jpeg')}
            data = {'nouns': json.dumps(nouns)}

            resp = requests.post(
                SAM3_ENDPOINT, files=files, data=data, timeout=SAM3_TIMEOUT
            )
            payload = self._coerce_terminal_http_response(resp, service='sam3')
            if payload.get('status', 'success') != 'success':
                return payload
            results = payload.get('results', {})
            payload['results'] = self._scale_results(results, scale, orig_w, orig_h)
            return payload

        except requests.RequestException as e:
            if getattr(e, 'response', None) is not None:
                return self._coerce_terminal_http_response(e.response, service='sam3')
            self.logger.error(f"sam3: REST call failed before terminal response: {e}")
            return None

    def _prepare_image(self, image_bytes: bytes):
        """
        Resize image to SAM3_INPUT_SIZE on the longest side and pad to square
        if needed. Image is placed top-left; black padding fills right/bottom.

        Returns (prepared_bytes, scale, orig_w, orig_h).
        scale is 1.0 if the image was already within SAM3_INPUT_SIZE.
        """
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        orig_w, orig_h = img.size

        if max(orig_w, orig_h) <= SAM3_INPUT_SIZE:
            return image_bytes, 1.0, orig_w, orig_h

        scale = SAM3_INPUT_SIZE / max(orig_w, orig_h)
        new_w = round(orig_w * scale)
        new_h = round(orig_h * scale)

        resized = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new('RGB', (SAM3_INPUT_SIZE, SAM3_INPUT_SIZE), (0, 0, 0))
        canvas.paste(resized, (0, 0))

        buf = io.BytesIO()
        canvas.save(buf, 'JPEG', quality=95)

        self.logger.info(
            f"sam3: resized {orig_w}x{orig_h} → {new_w}x{new_h} "
            f"padded to {SAM3_INPUT_SIZE}x{SAM3_INPUT_SIZE} (scale={scale:.4f})"
        )
        return buf.getvalue(), scale, orig_w, orig_h

    def _scale_results(self, results: dict, scale: float, orig_w: int, orig_h: int) -> dict:
        """Scale SAM3 bboxes and masks from SAM3 input coordinates back to original."""
        if scale == 1.0:
            return results

        scaled = {}
        for noun, noun_data in results.items():
            instances = []
            for inst in noun_data.get('instances', []):
                bbox = inst.get('bbox', {})
                new_inst = {
                    **inst,
                    'bbox': {
                        'x':      round(bbox.get('x', 0)      / scale),
                        'y':      round(bbox.get('y', 0)      / scale),
                        'width':  round(bbox.get('width', 0)  / scale),
                        'height': round(bbox.get('height', 0) / scale),
                    },
                }
                if 'mask_rle' in inst and 'mask_shape' in inst:
                    new_inst['mask_rle'], new_inst['mask_shape'] = self._scale_mask(
                        inst['mask_rle'], inst['mask_shape'], orig_h, orig_w, scale
                    )
                instances.append(new_inst)
            scaled[noun] = {'instances': instances}
        return scaled

    def _scale_mask(self, rle: list, mask_shape: list, orig_h: int, orig_w: int, scale: float):
        """
        Decode RLE mask from SAM3 coordinate space, crop to the actual image
        area (top-left placement means padding is on right/bottom), resize to
        original dimensions, and re-encode as RLE.

        RLE format: run lengths starting from the value of the first pixel.
        Segmentation masks universally start with background (False), so we
        decode assuming False-first.
        """
        h, w = mask_shape
        flat = np.zeros(h * w, dtype=bool)
        pos = 0
        val = False
        for count in rle:
            flat[pos:pos + count] = val
            pos += count
            val = not val
        mask = flat.reshape(h, w)

        # Crop to the scaled image area (padding is outside this region)
        scaled_h = round(orig_h * scale)
        scaled_w = round(orig_w * scale)
        mask = mask[:scaled_h, :scaled_w]

        # Resize to original dimensions using nearest-neighbour (boolean mask)
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)
        mask_resized = np.array(mask_img) > 127

        # Re-encode as RLE
        flat_out = mask_resized.flatten()
        rle_out = []
        if len(flat_out) == 0:
            return rle_out, [orig_h, orig_w]
        current = flat_out[0]
        count = 1
        for v in flat_out[1:]:
            if v == current:
                count += 1
            else:
                rle_out.append(int(count))
                current = v
                count = 1
        rle_out.append(int(count))
        return rle_out, [orig_h, orig_w]

    def _apply_subject_fallback(self, results: dict, subject_noun: str, image_id: int):
        """Populate results['_subject'] from the best available fallback noun.

        Called when the voted subject noun returned 0 instances from SAM3.

        Fallback priority:
          1. Validated noun in the same category as subject_noun, largest single bbox
          2. Any validated noun with the largest single bbox (last resort)

        Modifies results in place. Does nothing if no validated noun exists.
        """
        # Collect nouns that have at least one instance with a bbox
        validated = {
            noun: data['instances']
            for noun, data in results.items()
            if not noun.startswith('_') and data.get('instances')
        }

        if not validated:
            self.logger.info(f"sam3: no validated nouns available for subject fallback (image {image_id})")
            return

        def max_bbox_area(instances):
            best = 0
            for inst in instances:
                bbox = inst.get('bbox', {})
                area = bbox.get('width', 0) * bbox.get('height', 0)
                if area > best:
                    best = area
            return best

        # Step 1: category-matched fallback
        subject_category = categorize_nouns([subject_noun]).get(subject_noun)
        if subject_category:
            candidate_categories = categorize_nouns(list(validated.keys()))
            category_matched = {
                noun: instances
                for noun, instances in validated.items()
                if candidate_categories.get(noun) == subject_category
            }
            if category_matched:
                best_noun = max(category_matched, key=lambda n: max_bbox_area(category_matched[n]))
                results['_subject'] = {'instances': category_matched[best_noun], 'noun': best_noun}
                self.logger.info(
                    f"sam3: subject fallback (category '{subject_category}') → '{best_noun}' "
                    f"({len(category_matched[best_noun])} instance(s)) for image {image_id}"
                )
                return

        # Step 2: largest single bbox regardless of category
        best_noun = max(validated, key=lambda n: max_bbox_area(validated[n]))
        results['_subject'] = {'instances': validated[best_noun], 'noun': best_noun}
        self.logger.info(
            f"sam3: subject fallback (largest bbox) → '{best_noun}' "
            f"({len(validated[best_noun])} instance(s)) for image {image_id}"
        )

    def _compute_background_mask(self, results: dict) -> dict:
        """Union all _subject instance masks and invert to produce a background mask.

        Returns a dict with mask_rle, mask_shape, and rle_runs (for logging),
        or None if _subject is absent or has no masks.

        RLE convention: False-first (background=False, subject=True).  The
        background mask is the complement, so it starts with True.  A leading
        zero-length False run is prepended to maintain the False-first convention
        that all other masks in the system use.
        """
        instances = results.get('_subject', {}).get('instances', [])
        if not instances:
            return None

        first = instances[0]
        if 'mask_rle' not in first or 'mask_shape' not in first:
            return None

        h, w = first['mask_shape']
        union = np.zeros(h * w, dtype=bool)

        for inst in instances:
            rle = inst.get('mask_rle', [])
            shape = inst.get('mask_shape', [h, w])
            if not rle or shape[0] != h or shape[1] != w:
                continue
            flat = np.zeros(h * w, dtype=bool)
            pos = 0
            val = False
            for count in rle:
                flat[pos:pos + count] = val
                pos += count
                val = not val
            union |= flat

        background = ~union

        # Encode as RLE
        rle_out = []
        current = background[0]
        count = 1
        for v in background[1:]:
            if v == current:
                count += 1
            else:
                rle_out.append(int(count))
                current = v
                count = 1
        rle_out.append(int(count))

        # Maintain False-first convention: prepend a zero-length False run
        # if the background mask begins with True pixels (which it typically
        # will, since subject masks start with False).
        if background[0]:
            rle_out = [0] + rle_out

        return {
            'mask_rle': rle_out,
            'mask_shape': [h, w],
            'rle_runs': len(rle_out),
        }

    def _fetch_existing_results(self, image_id: int) -> tuple:
        """Return (nouns_queried, data) from any existing sam3_results row.

        Returns ([], {}) if no row exists yet.
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                "SELECT nouns_queried, data FROM sam3_results WHERE image_id = %s LIMIT 1",
                (image_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            if not row:
                return [], {}
            nouns_queried = row[0] or []
            data = row[1] if isinstance(row[1], dict) else json.loads(row[1])
            return nouns_queried, data
        except Exception as e:
            self.logger.error(f"sam3: fetch existing results failed for image {image_id}: {e}")
            return [], {}

    def _upsert_sam3_results(
        self, image_id: int, nouns: list, results: dict, processing_time: float
    ):
        """Insert or update sam3_results row for this image."""
        # Exclude probe keys (underscore-prefixed) from the instance count so
        # the stored count reflects detected foreground objects only.
        total_instances = sum(
            len(v.get('instances', [])) for k, v in results.items()
            if not k.startswith('_')
        )
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO sam3_results
                    (image_id, nouns_queried, data, instance_count, processing_time, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (image_id) DO UPDATE SET
                    nouns_queried    = EXCLUDED.nouns_queried,
                    data             = EXCLUDED.data,
                    instance_count   = EXCLUDED.instance_count,
                    processing_time  = EXCLUDED.processing_time,
                    updated_at       = NOW()
                """,
                (
                    image_id,
                    nouns,
                    json.dumps(results),
                    total_instances,
                    processing_time,
                )
            )
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"sam3: upsert failed for image {image_id}: {e}")
            raise



    def _validate_noun_consensus(self, image_id: int, results: dict):
        """Set sam3_validated=true on noun_consensus nouns that SAM3 found.

        A noun is validated if its entry in sam3 results has at least one
        instance. Nouns SAM3 missed are left untagged — no invalidation.
        """
        validated = [
            noun for noun, data in results.items()
            if len(data.get('instances', [])) > 0
        ]
        if not validated:
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
                            THEN n || '{"sam3_validated": true}'::jsonb
                            ELSE n
                        END
                    )
                    FROM jsonb_array_elements(nouns) n
                ),
                updated_at = NOW()
                WHERE image_id = %s
                """,
                (validated, image_id)
            )
            self.db_conn.commit()
            cursor.close()
            self.logger.info(
                f"sam3: validated {len(validated)} nouns for image {image_id}: {validated}"
            )
        except Exception as e:
            self.logger.error(
                f"sam3: failed to write validation for image {image_id}: {e}"
            )


    def _trigger_rembg(self, image_id: int, image_transport: dict = None):
        """Publish image_id to the rembg queue for background removal refinement."""
        try:
            queue = self._get_queue_name('system.rembg')
            dlq = f"{queue}.dlq"
            self.channel.queue_declare(queue=dlq, durable=True)
            self.channel.queue_declare(
                queue=queue, durable=True,
                arguments={'x-dead-letter-exchange': '', 'x-dead-letter-routing-key': dlq}
            )
            self.channel.basic_publish(
                exchange='',
                routing_key=queue,
                body=json.dumps({'image_id': image_id, **(image_transport or {})}),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            self.logger.info(f"sam3: triggered rembg for image {image_id}")
        except Exception as e:
            self.logger.error(f"sam3: failed to trigger rembg for image {image_id}: {e}")

if __name__ == '__main__':
    worker = Sam3Worker()
    worker.start()
