#!/usr/bin/env python3
"""
PostProcessingWorker - Base class for bbox postprocessing workers
Handles cropped image processing for colors, face, pose detection
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import json
import base64
import io
import time
import requests
from datetime import datetime
from base_worker import BaseWorker

class PostProcessingWorker(BaseWorker):
    """Base class for postprocessing workers that handle cropped images"""
    
    def __init__(self, service_name):
        super().__init__(service_name)
        
        # Queue name is handled by base_worker via config now
        # Service URL is built from config
        if self.service_host and self.service_port and self.service_endpoint:
            self.service_url = f"http://{self.service_host}:{self.service_port}{self.service_endpoint}"
        else:
            # For postprocessing services that might not have HTTP endpoints
            self.service_url = None

    def _parse_message_body(self, body):
        """Parse one JSON message and tolerate trailing junk from malformed republishes."""
        raw = body.decode('utf-8') if isinstance(body, (bytes, bytearray)) else body
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            decoder = json.JSONDecoder()
            parsed, end = decoder.raw_decode(raw)
            trailing = raw[end:].strip()
            if trailing:
                self.logger.warning(
                    f"Recovered {self.service_name} message with trailing data after JSON object: {trailing[:200]!r}"
                )
            else:
                raise exc
            return parsed

    def _is_valid_postprocessing_message(self, message):
        required_keys = ('merged_box_id', 'image_id', 'cluster_id', 'bbox', 'cropped_image_data')
        return all(key in message for key in required_keys)
    
    def connect_to_database(self):
        """Connect to DB and set autocommit=False for FK-safe transaction handling."""
        if not super().connect_to_database():
            return False
        self.db_conn.autocommit = False
        return True
    
    def process_service(self, cropped_image_data):
        """Process the cropped image with the specific service - override in subclasses"""
        raise NotImplementedError("Subclasses must implement process_service")
    
    def save_postprocessing_result(
        self,
        merged_box_id,
        image_id,
        result_data,
        bbox,
        cluster_id,
        processing_time=None,
        commit=True,
    ):
        """Save postprocessing result to database"""
        try:
            cursor = self.db_conn.cursor()

            # Store cluster_id in data so the API can look up the source bbox
            stored = dict(result_data) if result_data else {}
            if cluster_id:
                stored['cluster_id'] = cluster_id

            result_status = 'success'
            error_message = None
            if isinstance(result_data, dict):
                result_status = result_data.get('status', 'success') or 'success'
                error_message = (
                    result_data.get('error_message')
                    or result_data.get('error')
                    or result_data.get('message')
                )

            # Insert postprocessing result
            insert_started_at = time.time()
            cursor.execute("""
                INSERT INTO postprocessing (merged_box_id, image_id, service, data, status, error_message, processing_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                merged_box_id,
                image_id,
                self._get_clean_service_name(),
                json.dumps(stored),
                result_status,
                error_message,
                processing_time,
            ))
            insert_duration = time.time() - insert_started_at
            
            commit_duration = 0.0
            if commit:
                commit_started_at = time.time()
                self.db_conn.commit()
                commit_duration = time.time() - commit_started_at
            cursor.close()
            return True, {'insert': insert_duration, 'commit': commit_duration}
            
        except Exception as e:
            error_str = str(e)
            if 'postprocessing_merged_box_id_fkey' in error_str:
                # FK violation means merged_box was superseded by reharmonization - silently skip
                self.logger.info(f"Merged_box_id {merged_box_id} no longer exists (superseded by reharmonization) - skipping")
                if self.db_conn:
                    self.db_conn.rollback()
                return True, {'insert': 0.0, 'commit': 0.0}  # Return success to acknowledge the message
            else:
                self.logger.error(f"Error saving postprocessing result: {e}")
                if self.db_conn:
                    try:
                        self.db_conn.rollback()
                    except Exception:
                        pass
                return False, {'insert': 0.0, 'commit': 0.0}
    

    def process_message(self, ch, method, properties, body):
        """Process a postprocessing message - standard pattern for all postprocessing workers"""
        try:
            callback_started_at = time.time()
            # Ensure DB connection is healthy before doing any work — mirrors base_worker pattern.
            # Without this, an idle-dropped connection spins in a requeue loop instead of reconnecting.
            db_health_started_at = time.time()
            if not self.ensure_database_connection():
                self.logger.error(
                    "Database connection unavailable, rejecting message without requeue."
                )
                self._safe_nack(ch, method.delivery_tag, requeue=False)
                return
            db_health_duration = time.time() - db_health_started_at

            # Parse message
            parse_started_at = time.time()
            message = self._parse_message_body(body)
            parse_duration = time.time() - parse_started_at
            if not self._is_valid_postprocessing_message(message):
                self.logger.error(
                    f"Malformed {self.service_name} message, dropping without requeue: "
                    f"keys={sorted(message.keys())}"
                )
                self._safe_ack(ch, method.delivery_tag)
                return

            merged_box_id = message['merged_box_id']
            image_id = message['image_id']
            cluster_id = message['cluster_id']
            bbox = message['bbox']
            cropped_image_data = message['cropped_image_data']
            trace_id = message.get('trace_id')
            source_service = message.get('source_service')
            source_stage = message.get('source_stage')
            dispatch_enqueued_at = message.get('dispatch_enqueued_at')
            dispatch_received_at = message.get('dispatch_received_at')
            downstream_enqueued_at = message.get('downstream_enqueued_at')
            publisher_enqueued_at = message.get('_publisher_enqueued_at')
            publisher_started_at = message.get('_publisher_started_at')
            worker_received_at = time.time()

            def _latency_from_iso(value):
                if not value:
                    return None
                try:
                    return worker_received_at - datetime.fromisoformat(value).timestamp()
                except Exception:
                    return None

            upstream_queue_wait = _latency_from_iso(dispatch_enqueued_at)
            downstream_queue_wait = _latency_from_iso(downstream_enqueued_at)
            dispatcher_to_worker = None
            if dispatch_received_at:
                dispatcher_to_worker = _latency_from_iso(dispatch_received_at)
            publisher_local_wait = None
            if publisher_enqueued_at and publisher_started_at:
                try:
                    publisher_local_wait = (
                        datetime.fromisoformat(publisher_started_at).timestamp()
                        - datetime.fromisoformat(publisher_enqueued_at).timestamp()
                    )
                except Exception:
                    publisher_local_wait = None
            publish_to_worker = _latency_from_iso(publisher_started_at)
            callback_to_measurement = worker_received_at - callback_started_at

            self.logger.info(f"Processing {self.service_name} for {cluster_id} (merged_box_id: {merged_box_id})")

            # Process with specific service. Contract:
            # - dict => terminal service response, must be stored even if empty/no-findings/failed
            # - None => infrastructure/transport problem, retry
            start_time = time.time()
            result_data = self.process_service(cropped_image_data)
            processing_time = round(time.time() - start_time, 3)

            # Save terminal result
            if result_data is not None:
                save_started_at = time.time()
                success, save_timings = self.save_postprocessing_result(
                    merged_box_id,
                    image_id,
                    result_data,
                    bbox,
                    cluster_id,
                    processing_time,
                    commit=False,
                )
                save_duration = time.time() - save_started_at
                if success:
                    terminal_event_type = 'completed'
                    if isinstance(result_data, dict) and (result_data.get('status') or 'success') != 'success':
                        terminal_event_type = 'failed'
                    event_timings = self._record_postprocessing_event(
                        image_id=image_id,
                        merged_box_id=merged_box_id,
                        service=self._get_clean_service_name(),
                        cluster_id=cluster_id,
                        event_type=terminal_event_type,
                        source_service=source_service,
                        source_stage=source_stage,
                        data={
                            'status': (result_data.get('status') if isinstance(result_data, dict) else None),
                            'error_message': (
                                result_data.get('error_message')
                                if isinstance(result_data, dict) else None
                            ),
                        },
                        commit=True,
                    )
                    event_write_duration = event_timings['insert'] + event_timings['commit']
                    self._safe_ack(ch, method.delivery_tag)
                    timing = [
                        f"{self.service_name} image={image_id}",
                        f"cluster={cluster_id}",
                        f"service_call={processing_time:.3f}s",
                        f"save={save_duration:.3f}s",
                        f"save_insert={save_timings['insert']:.3f}s",
                        f"save_commit={save_timings['commit']:.3f}s",
                        f"event_write={event_write_duration:.3f}s",
                        f"event_insert={event_timings['insert']:.3f}s",
                        f"event_commit={event_timings['commit']:.3f}s",
                        f"total={time.time() - worker_received_at:.3f}s",
                    ]
                    if upstream_queue_wait is not None:
                        timing.insert(2, f"from_dispatch_enqueue={upstream_queue_wait:.3f}s")
                    if downstream_queue_wait is not None:
                        timing.insert(3, f"worker_queue_wait={downstream_queue_wait:.3f}s")
                    if dispatcher_to_worker is not None:
                        timing.insert(4, f"from_dispatcher_receive={dispatcher_to_worker:.3f}s")
                    if publisher_local_wait is not None:
                        timing.insert(5, f"publisher_local_wait={publisher_local_wait:.3f}s")
                    if publish_to_worker is not None:
                        timing.insert(6, f"publish_to_worker={publish_to_worker:.3f}s")
                    timing.insert(7, f"callback_overhead={callback_to_measurement:.3f}s")
                    timing.insert(8, f"db_health={db_health_duration:.3f}s")
                    timing.insert(9, f"parse={parse_duration:.3f}s")
                    if source_service or source_stage:
                        timing.insert(2, f"source={source_service or 'unknown'}:{source_stage or 'unknown'}")
                    prefix = f"[{trace_id}] " if trace_id else ""
                    self.logger.info(prefix + "processed " + " ".join(timing))
                    return True
                else:
                    self._safe_nack(ch, method.delivery_tag, requeue=True)
                    return False
            else:
                self.logger.warning(
                    f"{self.service_name} produced no terminal response for image {image_id} "
                    f"cluster={cluster_id}; requeueing"
                )
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                return False

        except Exception as e:
            error_str = str(e)
            if 'postprocessing_merged_box_id_fkey' in error_str:
                # FK violation means merged_box was superseded by reharmonization - silently skip
                self.logger.info(f"Merged_box_id {merged_box_id} no longer exists (superseded by reharmonization) - skipping")
                self._safe_ack(ch, method.delivery_tag)  # Acknowledge to remove from queue
                return True
            else:
                self.logger.error(f"Error processing {self.service_name} message: {e}")
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                return False
