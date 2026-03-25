#!/usr/bin/env python3
"""
BaseWorker - Clean base class for all ML service workers
Eliminates the SERVICE_NAME environment variable mess and provides proper inheritance
"""
import os
import sys
import json
import time
import logging
import socket
import base64
import io
import threading
import ssl
import pika
import psycopg2
import requests
import yaml
import re
from PIL import Image
from collections import deque
from functools import partial
from datetime import datetime
from dotenv import load_dotenv
from service_config import get_service_config

class BaseWorker:
    """Base class for all ML service workers"""
    
    def __init__(self, service_name, env_file='.env'):
        self.service_name = service_name
        self.load_config(env_file)
        self.setup_logging()
        self.db_conn = None
        self.channel = None
        self.jobs_completed = 0
        self.jobs_failed = 0

        # Database resilience tracking
        self.consecutive_db_failures = 0
        self.max_db_failures_before_backoff = 3
        self.db_backoff_delay = 1  # Start with 1 second
        self.max_db_backoff_delay = 60  # Max 60 seconds

        # Async publisher state — outbound publishes run on a dedicated RabbitMQ
        # I/O loop instead of blocking the consume thread.
        self._publish_thread = None
        self._publish_connection = None
        self._publish_channel = None
        self._publish_ready = threading.Event()
        self._publish_lock = threading.Lock()
        self._publish_pending = deque()
        self._sync_publish_connection = None
        self._sync_publish_channel = None
        self._running = False

        # Heartbeat thread — keeps worker_registry up to date while running
        self._heartbeat_thread = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_interval = int(os.getenv('WORKER_HEARTBEAT_INTERVAL', '30'))

        # SIGTERM handler — converts kill signal into KeyboardInterrupt for clean shutdown
        import signal
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        
    def load_config(self, env_file):
        """Load configuration from .env and service_config.yaml"""
        # Load .env file
        if not load_dotenv(env_file):
            raise ValueError(f"Could not load {env_file} file. Copy .env.example to .env and configure.")
        
        # Load service configuration using new YAML loader
        self.config = get_service_config()
        
        # Validate service name (must be in category.service format)
        service_def = self.config.get_service_config(self.service_name)
        
        # Service configuration
        self.service_host = service_def.get('host')
        self.service_port = service_def.get('port')
        self.service_endpoint = service_def.get('endpoint')
        
        # Queue configuration
        self.queue_name = self.config.get_queue_name(self.service_name)
        self.queue_host = self._get_required('QUEUE_HOST')
        self.queue_port = int(os.getenv('QUEUE_PORT', '5672'))
        self.queue_ssl = os.getenv('QUEUE_SSL', '').lower() in ('true', '1', 'yes')
        self.queue_user = self._get_required('QUEUE_USER')
        self.queue_password = self._get_required('QUEUE_PASSWORD')

        # Database configuration
        self.db_host = self._get_required('DB_HOST')
        self.db_name = self._get_required('DB_NAME')
        self.db_user = self._get_required('DB_USER')
        self.db_password = self._get_required('DB_PASSWORD')
        self.db_sslmode = os.getenv('DB_SSLMODE')
        
        # Worker configuration
        self.worker_id = f"worker_{self.service_name}_{int(time.time())}"
        # Optional per-service prefetch override from YAML; fallback to env; default 1
        self.worker_prefetch_count = int(
            service_def.get('prefetch', os.getenv('WORKER_PREFETCH_COUNT', '1'))
        )
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '30'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.retry_delay = int(os.getenv('RETRY_DELAY', '5'))
        
        # Post-processing triggers - use new config loader methods
        self.bbox_services = self.config.get_spatial_services()
        
        # Determine triggers based on service type and YAML rules
        self.is_spatial = self.config.is_spatial_service(self.service_name)
        self.is_semantic = self.config.is_semantic_service(self.service_name)
        self.is_vlm = self.config.is_vlm_service(self.service_name)
        self.enable_consensus_triggers = self.config.should_trigger_consensus(self.service_name)

        # Enable direct bbox postprocessing triggers for spatial services
        self.enable_triggers = self.is_spatial
        # Enable noun and verb consensus for VLM services
        self.enable_noun_consensus = self.is_vlm
        self.enable_verb_consensus = self.is_vlm
        
        # Performance configuration
        self.processing_delay = float(os.getenv('PROCESSING_DELAY', '0.0'))

    def _get_required(self, key):
        """Get required environment variable or raise error"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value
    
    def _get_queue_name(self, service_name):
        """Get queue name for any service using its queue_name configuration"""
        try:
            return self.config.get_queue_name(service_name)
        except ValueError:
            # For services not in config (legacy support), use service name
            return service_name
    
    def _get_queue_by_service_type(self, service_type):
        """Find service by service_type and return its queue name"""
        return self.config.get_queue_by_service_type(service_type)
    
    def _get_clean_service_name(self):
        """Get clean service name without category prefix"""
        if '.' in self.service_name:
            return self.service_name.split('.', 1)[1]  # Return part after first dot
        return self.service_name

    def _normalize_emoji(self, emoji):
        if not emoji:
            return emoji
        return re.sub(r'[\uFE00-\uFE0F\u180B-\u180D\u200D]', '', emoji)

    def _normalize_person_emoji(self, emoji):
        normalized = self._normalize_emoji(emoji)
        if normalized in ['🧑', '👩', '🧒']:
            return '🧑'
        return normalized

    def _normalize_prediction_bbox(self, prediction):
        bbox = prediction.get('bbox')
        if isinstance(bbox, list) and len(bbox) >= 4:
            return {
                'x': bbox[0],
                'y': bbox[1],
                'width': bbox[2],
                'height': bbox[3],
            }
        if isinstance(bbox, dict) and all(k in bbox for k in ('x', 'y', 'width', 'height')):
            return {
                'x': bbox['x'],
                'y': bbox['y'],
                'width': bbox['width'],
                'height': bbox['height'],
            }
        return None

    def _now_iso(self):
        return datetime.now().isoformat()

    def _crop_bbox_from_image_data(self, image_data, bbox):
        """Crop one bbox from base64 image data and return base64 JPEG bytes."""
        try:
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes))
            with img:
                crop_box = (
                    bbox['x'],
                    bbox['y'],
                    bbox['x'] + bbox['width'],
                    bbox['y'] + bbox['height'],
                )
                cropped_img = img.crop(crop_box)
                if cropped_img.mode not in ('RGB', 'L'):
                    cropped_img = cropped_img.convert('RGB')
                img_buffer = io.BytesIO()
                cropped_img.save(img_buffer, format='JPEG', quality=90)
                img_buffer.seek(0)
                return base64.b64encode(img_buffer.getvalue()).decode('latin-1')
        except Exception as e:
            self.logger.error(f"Failed to crop bbox from image data: {e}")
            return None

    def _build_postprocessing_messages(self, image_id, message, result, tier, trace_id):
        """Build direct face/pose postprocessing messages for single-spatial YOLO paths."""
        if self.service_name not in self.bbox_services:
            return []
        if len(self.bbox_services) != 1:
            return []
        if self._get_clean_service_name() != 'yolo_v8':
            return []
        if not isinstance(result, dict):
            return []

        predictions = result.get('predictions') or []
        if not predictions:
            return []

        messages = []
        for index, prediction in enumerate(predictions):
            bbox = self._normalize_prediction_bbox(prediction)
            if not bbox:
                continue

            if bbox['width'] < 8 or bbox['height'] < 8:
                continue

            label = (prediction.get('label') or '').strip().lower()
            emoji = self._normalize_person_emoji(prediction.get('emoji', ''))
            if label != 'person' and emoji != '🧑':
                continue

            cluster_id = f"yolo_person_{index}"
            cropped_image_data = self._crop_bbox_from_image_data(message.get('image_data'), bbox)
            if not cropped_image_data:
                continue

            if self.config.is_available_for_tier('postprocessing.face', tier):
                self._record_postprocessing_event(
                    image_id=image_id,
                    merged_box_id=None,
                    service='face',
                    cluster_id=cluster_id,
                    event_type='enqueued',
                    source_service=self._get_clean_service_name(),
                    source_stage='result_stored',
                    data={'bbox': bbox},
                )
                messages.append((
                    self._get_queue_name('postprocessing.face'),
                    json.dumps({
                        'merged_box_id': None,
                        'image_id': image_id,
                        'cluster_id': cluster_id,
                        'bbox': bbox,
                        'cropped_image_data': cropped_image_data,
                        'trace_id': trace_id,
                        'tier': tier,
                        'source_service': self._get_clean_service_name(),
                        'source_stage': 'result_stored',
                        'dispatch_enqueued_at': self._now_iso(),
                    }),
                ))
            if self.config.is_available_for_tier('postprocessing.pose', tier):
                self._record_postprocessing_event(
                    image_id=image_id,
                    merged_box_id=None,
                    service='pose',
                    cluster_id=cluster_id,
                    event_type='enqueued',
                    source_service=self._get_clean_service_name(),
                    source_stage='result_stored',
                    data={'bbox': bbox},
                )
                messages.append((
                    self._get_queue_name('postprocessing.pose'),
                    json.dumps({
                        'merged_box_id': None,
                        'image_id': image_id,
                        'cluster_id': cluster_id,
                        'bbox': bbox,
                        'cropped_image_data': cropped_image_data,
                        'trace_id': trace_id,
                        'tier': tier,
                        'source_service': self._get_clean_service_name(),
                        'source_stage': 'result_stored',
                        'dispatch_enqueued_at': self._now_iso(),
                    }),
                ))

        return messages

    def _declare_additional_queues(self, declare_with_dlq):
        """Override in subclasses to declare additional downstream queues on the publish channel."""
        pass

    def setup_logging(self):
        """Setup logging for this worker"""
        self.logger = logging.getLogger(self.service_name)
        self.logger.setLevel(getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper()))
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _build_queue_params(self, **overrides):
        """Build pika ConnectionParameters with optional TLS support."""
        credentials = pika.PlainCredentials(self.queue_user, self.queue_password)
        kwargs = dict(
            host=self.queue_host,
            port=self.queue_port,
            credentials=credentials,
            heartbeat=60,
            blocked_connection_timeout=300,
            connection_attempts=10,
            retry_delay=5,
            socket_timeout=10,
        )
        if self.queue_ssl:
            ssl_context = ssl.create_default_context()
            kwargs['ssl_options'] = pika.SSLOptions(ssl_context, self.queue_host)
        kwargs.update(overrides)
        return pika.ConnectionParameters(**kwargs)

    def _get_publish_declarations(self):
        """Return downstream queues this worker may publish to, including DLQs."""
        declarations = []

        def add_queue(queue_name):
            dlq_name = f"{queue_name}.dlq"
            declarations.append((dlq_name, None))
            args = {
                'x-dead-letter-exchange': '',
                'x-dead-letter-routing-key': dlq_name
            }
            ttl_env = os.getenv('QUEUE_MESSAGE_TTL_MS')
            if ttl_env and ttl_env.isdigit() and int(ttl_env) > 0:
                args['x-message-ttl'] = int(ttl_env)
            declarations.append((queue_name, args))

        if self.enable_triggers:
            if self.service_name in self.bbox_services and len(self.bbox_services) == 1:
                add_queue(self._get_queue_name('postprocessing.face'))
                add_queue(self._get_queue_name('postprocessing.pose'))
        if self.enable_consensus_triggers:
            add_queue(self._get_queue_by_service_type('consensus'))
        if self.enable_noun_consensus:
            add_queue(self._get_queue_by_service_type('noun_consensus'))
        if self.enable_verb_consensus:
            add_queue(self._get_queue_by_service_type('verb_consensus'))

        self._declare_additional_queues(lambda _channel, queue_name: add_queue(queue_name))
        return declarations

    def _connect_sync_publish_channel(self):
        """Create a dedicated publish-confirm channel owned by the consume thread."""
        if self._sync_publish_connection:
            try:
                self._sync_publish_connection.close()
            except Exception:
                pass

        self._sync_publish_connection = pika.BlockingConnection(self._build_queue_params())
        self._sync_publish_channel = self._sync_publish_connection.channel()
        self._sync_publish_channel.confirm_delivery()

        def declare_with_dlq(channel, queue_name):
            dlq_name = f"{queue_name}.dlq"
            channel.queue_declare(queue=dlq_name, durable=True)
            args = {
                'x-dead-letter-exchange': '',
                'x-dead-letter-routing-key': dlq_name
            }
            ttl_env = os.getenv('QUEUE_MESSAGE_TTL_MS')
            if ttl_env and ttl_env.isdigit() and int(ttl_env) > 0:
                args['x-message-ttl'] = int(ttl_env)
            channel.queue_declare(queue=queue_name, durable=True, arguments=args)

        if self.enable_triggers:
            if self.service_name in self.bbox_services and len(self.bbox_services) == 1:
                declare_with_dlq(self._sync_publish_channel, self._get_queue_name('postprocessing.face'))
                declare_with_dlq(self._sync_publish_channel, self._get_queue_name('postprocessing.pose'))
        if self.enable_consensus_triggers:
            declare_with_dlq(self._sync_publish_channel, self._get_queue_by_service_type('consensus'))
        if self.enable_noun_consensus:
            declare_with_dlq(self._sync_publish_channel, self._get_queue_by_service_type('noun_consensus'))
        if self.enable_verb_consensus:
            declare_with_dlq(self._sync_publish_channel, self._get_queue_by_service_type('verb_consensus'))

        self._declare_additional_queues(declare_with_dlq)

    def _publish_messages_sync_confirm(self, messages):
        """Publish required downstream messages synchronously with broker confirms.

        messages: iterable of (routing_key, body_json_string)
        """
        if not messages:
            return

        if self._sync_publish_connection is None or self._sync_publish_connection.is_closed:
            self._connect_sync_publish_channel()
        elif self._sync_publish_channel is None or self._sync_publish_channel.is_closed:
            self._connect_sync_publish_channel()

        for routing_key, body in messages:
            try:
                published = self._sync_publish_channel.basic_publish(
                    exchange='',
                    routing_key=routing_key,
                    body=body,
                    properties=pika.BasicProperties(delivery_mode=2),
                    mandatory=True,
                )
                if published is False:
                    raise RuntimeError(f"Broker did not confirm publish to {routing_key}")
            except (pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError,
                    pika.exceptions.StreamLostError) as e:
                self.logger.warning(f"Sync publish connection lost: {e}. Reconnecting...")
                self._connect_sync_publish_channel()
                published = self._sync_publish_channel.basic_publish(
                    exchange='',
                    routing_key=routing_key,
                    body=body,
                    properties=pika.BasicProperties(delivery_mode=2),
                    mandatory=True,
                )
                if published is False:
                    raise RuntimeError(f"Broker did not confirm publish to {routing_key} after reconnect")

    def _start_async_publisher(self):
        """Publisher thread target using RabbitMQ's async I/O loop."""
        while self._running:
            try:
                self._publish_ready.clear()
                self._publish_connection = pika.SelectConnection(
                    self._build_queue_params(),
                    on_open_callback=self._on_publish_connection_open,
                    on_open_error_callback=self._on_publish_connection_open_error,
                    on_close_callback=self._on_publish_connection_closed,
                )
                self._publish_connection.ioloop.start()
            except Exception as e:
                if self._running:
                    self.logger.error(f"Async publish loop failed: {e}")
            finally:
                self._publish_ready.clear()
                self._publish_channel = None
                self._publish_connection = None

            if self._running:
                time.sleep(self.retry_delay)

        self.logger.info("Async publish thread stopped")

    def _on_publish_connection_open(self, connection):
        connection.channel(on_open_callback=self._on_publish_channel_open)

    def _on_publish_connection_open_error(self, connection, error):
        self.logger.warning(f"Async publish connection open failed: {error}")
        connection.ioloop.stop()

    def _on_publish_connection_closed(self, connection, reason):
        self._publish_ready.clear()
        self._publish_channel = None
        if self._running:
            self.logger.warning(f"Async publish connection closed: {reason}")
        connection.ioloop.stop()

    def _on_publish_channel_open(self, channel):
        self._publish_channel = channel
        declarations = self._get_publish_declarations()
        self._declare_publish_queue_at_index(declarations, 0)

    def _declare_publish_queue_at_index(self, declarations, index):
        if self._publish_channel is None:
            return
        if index >= len(declarations):
            self._publish_ready.set()
            self.logger.info("Async publisher connected to RabbitMQ")
            self._flush_pending_async_publishes()
            return

        queue_name, arguments = declarations[index]
        self._publish_channel.queue_declare(
            queue=queue_name,
            durable=True,
            arguments=arguments,
            callback=lambda _frame: self._declare_publish_queue_at_index(declarations, index + 1),
        )

    def _flush_pending_async_publishes(self):
        while True:
            with self._publish_lock:
                if not self._publish_pending:
                    return
                routing_key, body, local_enqueued_at = self._publish_pending.popleft()
            self._async_publish_message(routing_key, body, local_enqueued_at)

    def _enqueue_publish(self, routing_key, body):
        """Publish through the async RabbitMQ connection without blocking the worker."""
        body = self._augment_publish_body(
            body,
            _publisher_enqueued_at=self._now_iso(),
        )
        local_enqueued_at = time.monotonic()

        if (
            self._publish_connection is not None
            and self._publish_channel is not None
            and self._publish_ready.is_set()
        ):
            try:
                self._publish_connection.ioloop.add_callback_threadsafe(
                    partial(self._async_publish_message, routing_key, body, local_enqueued_at)
                )
                return
            except Exception as e:
                self.logger.warning(f"Async publish handoff failed, buffering locally: {e}")

        with self._publish_lock:
            self._publish_pending.append((routing_key, body, local_enqueued_at))

    def _async_publish_message(self, routing_key, body, local_enqueued_at):
        if self._publish_channel is None or self._publish_channel.is_closed:
            with self._publish_lock:
                self._publish_pending.appendleft((routing_key, body, local_enqueued_at))
            return

        local_queue_wait = time.monotonic() - local_enqueued_at
        body = self._augment_publish_body(
            body,
            _publisher_started_at=self._now_iso(),
        )
        publish_started_at = time.time()
        self._publish_channel.basic_publish(
            exchange='',
            routing_key=routing_key,
            body=body,
            properties=pika.BasicProperties(delivery_mode=2),
        )
        publish_duration = time.time() - publish_started_at
        self._log_publish_timing(
            routing_key,
            body,
            local_queue_wait,
            publish_duration,
        )

    def _augment_publish_body(self, body, **fields):
        """Add publish-trace metadata to JSON object bodies.

        Non-JSON payloads are returned unchanged.
        """
        if not body or not fields:
            return body
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                return body
            for key, value in fields.items():
                payload[key] = value
            return json.dumps(payload)
        except Exception:
            return body

    def _log_publish_timing(self, routing_key, body, local_queue_wait, publish_duration, retried=False):
        """Emit publish timing when the local handoff or publish call is slow."""
        if local_queue_wait < 0.05 and publish_duration < 0.05:
            return
        image_id = None
        trace_id = None
        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                image_id = payload.get('image_id')
                trace_id = payload.get('trace_id')
        except Exception:
            pass
        bits = [
            f"routing_key={routing_key}",
            f"local_queue_wait={local_queue_wait:.3f}s",
            f"publish_call={publish_duration:.3f}s",
        ]
        if image_id is not None:
            bits.insert(1, f"image={image_id}")
        if retried:
            bits.append("retried=true")
        prefix = f"[{trace_id}] " if trace_id else ""
        self.logger.info(prefix + "publish_timing " + " ".join(bits))

    def _safe_ack(self, ch, delivery_tag):
        """Ack a message, logging instead of raising if the channel is dead.

        Uses self.channel instead of ch parameter to avoid ACKing on stale channels
        after reconnect. The ch parameter is kept for API compatibility but not used.
        """
        try:
            self.channel.basic_ack(delivery_tag=delivery_tag)
        except (pika.exceptions.AMQPChannelError, pika.exceptions.AMQPConnectionError,
                pika.exceptions.StreamLostError) as e:
            self.logger.warning(f"Channel dead during ack (message will be redelivered): {e}")
            raise  # Let it propagate to start()'s reconnect loop

    def _safe_nack(self, ch, delivery_tag, requeue=True):
        """Nack a message, swallowing channel errors since we're already in an error path.

        Uses self.channel instead of ch parameter to avoid NACKing on stale channels
        after reconnect. The ch parameter is kept for API compatibility but not used.
        """
        try:
            self.channel.basic_nack(delivery_tag=delivery_tag, requeue=requeue)
        except (pika.exceptions.AMQPChannelError, pika.exceptions.AMQPConnectionError,
                pika.exceptions.StreamLostError) as e:
            self.logger.warning(
                f"Channel dead during nack (broker will redeliver unacked message): {e}"
            )

    def job_completed_successfully(self):
        """Call this after successfully completing a job"""
        self.jobs_completed += 1
        self.logger.debug(f"Job completed successfully (total: {self.jobs_completed})")
    
    def job_failed(self, error_msg=None):
        """Call this after a job fails"""
        self.jobs_failed += 1
        self.logger.debug(f"Job failed (total failures: {self.jobs_failed})")
    
    def get_service_url(self):
        """Build the complete service URL for processing"""
        return f"http://{self.service_host}:{self.service_port}{self.service_endpoint}"
    
    def post_image_data(self, image_data_b64):
        """POST base64 encoded image data to service"""
        service_url = self.get_service_url()
        
        # Decode base64 to bytes for posting
        image_bytes = base64.b64decode(image_data_b64)
        
        # POST as multipart file upload
        files = {'file': ('image.jpg', io.BytesIO(image_bytes), 'image/jpeg')}
        try:
            response = requests.post(
                service_url,
                files=files,
                timeout=self.request_timeout
            )
            return self._coerce_terminal_http_response(response)
        except requests.RequestException as e:
            if getattr(e, 'response', None) is not None:
                return self._coerce_terminal_http_response(e.response)
            self.logger.error(f"{self.service_name} request failed before terminal response: {e}")
            return None

    def _coerce_terminal_http_response(self, response, service=None):
        """Convert an HTTP response into a terminal result payload.

        If the service answered at all, that is a terminal outcome for the current
        trigger. Non-2xx and non-JSON responses are normalized into failed result
        payloads so workers do not spin forever retrying stable service errors.
        """
        service = service or self._get_clean_service_name()
        body_text = ""
        try:
            body_text = response.text or ""
        except Exception:
            body_text = ""

        payload = None
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if not isinstance(payload, dict):
            payload = {
                'service': service,
                'status': 'failed',
                'predictions': [],
                'error_message': body_text.strip()[:500] or (
                    f"HTTP {response.status_code} with non-JSON response"
                ),
                'metadata': {},
            }
        else:
            payload.setdefault('service', service)
            if not isinstance(payload.get('metadata'), dict):
                payload['metadata'] = {}
            if response.ok:
                payload.setdefault('status', 'success')
            else:
                if payload.get('status') == 'success' or not payload.get('status'):
                    payload['status'] = 'failed'
                if not any(payload.get(key) for key in ('error_message', 'error', 'message')):
                    payload['error_message'] = (
                        body_text.strip()[:500] or f"HTTP {response.status_code}"
                    )

        payload['metadata'].setdefault('http_status', response.status_code)
        if not response.ok:
            payload['metadata'].setdefault('terminal_http_error', True)
        return payload
    
    def trigger_consensus(self, image_id, message):
        """Trigger consensus processing (async via background publish thread)"""
        if not self.enable_consensus_triggers:
            return

        self._record_service_event(
            image_id=image_id,
            service='consensus',
            event_type='enqueued',
            source_service=self._get_clean_service_name(),
            source_stage='consensus_trigger',
        )
        consensus_message = {
            'image_id': image_id,
            'image_filename': message.get('image_filename', f'image_{image_id}'),
            'service': self.service_name,
            'worker_id': self.worker_id,
            'processed_at': datetime.now().isoformat(),
            'tier': message.get('tier', 'free'),
        }

        self._enqueue_publish(
            self._get_queue_by_service_type('consensus'),
            json.dumps(consensus_message)
        )

        self.logger.debug(f"Enqueued consensus trigger for {self.service_name} image {image_id}")
    
    def trigger_noun_consensus(self, image_id, message):
        """Trigger noun consensus for VLM services (async via background publish thread)"""
        if not self.enable_noun_consensus:
            return

        self._record_service_event(
            image_id=image_id,
            service='noun_consensus',
            event_type='enqueued',
            source_service=self._get_clean_service_name(),
            source_stage='noun_consensus_trigger',
        )
        noun_consensus_message = {
            'image_id': image_id,
            'image_filename': message.get('image_filename', f'image_{image_id}'),
            'image_data': message.get('image_data'),
            'service': self.service_name,
            'worker_id': self.worker_id,
            'processed_at': datetime.now().isoformat(),
            'tier': message.get('tier', 'free'),
        }

        self._enqueue_publish(
            self._get_queue_by_service_type('noun_consensus'),
            json.dumps(noun_consensus_message)
        )

        self.logger.debug(f"Enqueued noun_consensus trigger for {self.service_name} image {image_id}")

    def trigger_verb_consensus(self, image_id, message):
        """Trigger verb consensus for VLM services (async via background publish thread)"""
        if not self.enable_verb_consensus:
            return

        self._record_service_event(
            image_id=image_id,
            service='verb_consensus',
            event_type='enqueued',
            source_service=self._get_clean_service_name(),
            source_stage='verb_consensus_trigger',
        )
        verb_consensus_message = {
            'image_id': image_id,
            'image_filename': message.get('image_filename', f'image_{image_id}'),
            'service': self.service_name,
            'worker_id': self.worker_id,
            'processed_at': datetime.now().isoformat(),
            'tier': message.get('tier', 'free'),
        }

        self._enqueue_publish(
            self._get_queue_by_service_type('verb_consensus'),
            json.dumps(verb_consensus_message)
        )

        self.logger.debug(f"Enqueued verb_consensus trigger for {self.service_name} image {image_id}")

    def _record_service_dispatch(self, image_id, service, cluster_id=None):
        """Insert a pending service_dispatch record. Best-effort; errors swallowed.

        Returns the dispatch_id (bigint PK) of the inserted row, or None on failure.
        Callers that need targeted completion tracking (e.g. progressively-triggered
        services like florence2_grounding) should embed this ID in their queue message
        and pass it to _update_service_dispatch via the dispatch_id parameter.

        On autocommit=False connections the INSERT is wrapped in a SAVEPOINT so
        that a failure (e.g. FK violation) rolls back only this INSERT and leaves
        the parent transaction alive.
        """
        try:
            total_started_at = time.time()
            cursor = self.db_conn.cursor()
            savepoint_duration = 0.0
            if not self.db_conn.autocommit:
                savepoint_started_at = time.time()
                cursor.execute("SAVEPOINT before_service_dispatch")
                savepoint_duration = time.time() - savepoint_started_at
            execute_started_at = time.time()
            cursor.execute(
                "INSERT INTO service_dispatch (image_id, service, cluster_id) VALUES (%s, %s, %s) RETURNING dispatch_id",
                (image_id, service, cluster_id),
            )
            execute_duration = time.time() - execute_started_at
            fetch_started_at = time.time()
            row = cursor.fetchone()
            fetch_duration = time.time() - fetch_started_at
            close_started_at = time.time()
            cursor.close()
            close_duration = time.time() - close_started_at
            if getattr(self, "service_name", None) == "system.harmony":
                self.logger.info(
                    f"service_dispatch record timing service={service} image={image_id} "
                    f"savepoint={savepoint_duration:.3f}s "
                    f"execute={execute_duration:.3f}s "
                    f"fetch={fetch_duration:.3f}s "
                    f"close={close_duration:.3f}s "
                    f"total={time.time() - total_started_at:.3f}s"
                )
            return row[0] if row else None
        except Exception as e:
            self.logger.warning(
                f"Failed to record service_dispatch for {service}/{image_id}: {e}"
            )
            if not self.db_conn.autocommit:
                try:
                    self.db_conn.cursor().execute("ROLLBACK TO SAVEPOINT before_service_dispatch")
                except Exception:
                    pass
            return None

    def _update_service_dispatch(self, image_id, service=None, cluster_id=None, status='complete', reason=None, dispatch_id=None):
        """Update service_dispatch to the given status. Best-effort; errors swallowed.

        dispatch_id: when provided, targets exactly this row by PK — safe for
        progressively-triggered services (e.g. florence2_grounding) where multiple
        pending rows can exist for the same image+service. Without dispatch_id the
        update bulk-clears all pending rows matching (image_id, service, cluster_id),
        which is correct for services that dispatch at most once per image.

        cluster_id=None matches rows WHERE cluster_id IS NULL (image-level services).
        Provide cluster_id for bbox-level services such as face and pose.
        service defaults to this worker's clean service name.
        reason is written to failed_reason when provided.

        On autocommit=False connections the UPDATE is committed automatically.
        """
        service = service or self._get_clean_service_name()
        try:
            total_started_at = time.time()
            cursor = self.db_conn.cursor()
            execute_started_at = time.time()
            if dispatch_id is not None:
                if reason is not None:
                    cursor.execute(
                        """UPDATE service_dispatch SET status = %s, failed_reason = %s
                           WHERE dispatch_id = %s AND status = 'pending'""",
                        (status, reason, dispatch_id),
                    )
                else:
                    cursor.execute(
                        """UPDATE service_dispatch SET status = %s
                           WHERE dispatch_id = %s AND status = 'pending'""",
                        (status, dispatch_id),
                    )
            elif cluster_id is None:
                if reason is not None:
                    cursor.execute(
                        """UPDATE service_dispatch SET status = %s, failed_reason = %s
                           WHERE image_id = %s AND service = %s
                             AND cluster_id IS NULL AND status = 'pending'""",
                        (status, reason, image_id, service),
                    )
                else:
                    cursor.execute(
                        """UPDATE service_dispatch SET status = %s
                           WHERE image_id = %s AND service = %s
                             AND cluster_id IS NULL AND status = 'pending'""",
                        (status, image_id, service),
                    )
            else:
                if reason is not None:
                    cursor.execute(
                        """UPDATE service_dispatch SET status = %s, failed_reason = %s
                           WHERE image_id = %s AND service = %s
                             AND cluster_id = %s AND status = 'pending'""",
                        (status, reason, image_id, service, cluster_id),
                    )
                else:
                    cursor.execute(
                        """UPDATE service_dispatch SET status = %s
                           WHERE image_id = %s AND service = %s
                             AND cluster_id = %s AND status = 'pending'""",
                        (status, image_id, service, cluster_id),
                    )
            execute_duration = time.time() - execute_started_at
            close_started_at = time.time()
            cursor.close()
            close_duration = time.time() - close_started_at
            commit_duration = 0.0
            if not self.db_conn.autocommit:
                commit_started_at = time.time()
                self.db_conn.commit()
                commit_duration = time.time() - commit_started_at
            if getattr(self, "service_name", None) == "system.harmony":
                self.logger.info(
                    f"service_dispatch update timing service={service} image={image_id} "
                    f"dispatch_id={dispatch_id if dispatch_id is not None else 'none'} "
                    f"cluster_id={cluster_id if cluster_id is not None else 'null'} "
                    f"execute={execute_duration:.3f}s "
                    f"close={close_duration:.3f}s "
                    f"commit={commit_duration:.3f}s "
                    f"total={time.time() - total_started_at:.3f}s"
                )
        except Exception as e:
            self.logger.warning(
                f"Failed to update service_dispatch for {service}/{image_id}: {e}"
            )
            if not self.db_conn.autocommit:
                try:
                    self.db_conn.rollback()
                except Exception:
                    pass

    def _record_postprocessing_event(
        self,
        image_id,
        service,
        cluster_id,
        event_type,
        merged_box_id=None,
        source_service=None,
        source_stage=None,
        data=None,
        commit=False,
    ):
        """Append one postprocessing event row. Best-effort and mutation-free."""
        if not cluster_id:
            return

        try:
            cursor = self.db_conn.cursor()
            if not getattr(self.db_conn, "autocommit", False):
                cursor.execute("SAVEPOINT before_postprocessing_event")
            cursor.execute(
                """
                INSERT INTO postprocessing_events (
                    image_id, merged_box_id, service, cluster_id,
                    event_type, source_service, source_stage, data
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    image_id,
                    merged_box_id,
                    service,
                    cluster_id,
                    event_type,
                    source_service,
                    source_stage,
                    json.dumps(data) if data is not None else None,
                ),
            )
            if getattr(self.db_conn, "autocommit", False) or commit:
                self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.warning(
                f"Failed to record postprocessing_event for {service}/{image_id}/{cluster_id}: {e}"
            )
            try:
                if getattr(self.db_conn, "autocommit", False):
                    self.db_conn.rollback()
                else:
                    rollback_cursor = self.db_conn.cursor()
                    rollback_cursor.execute("ROLLBACK TO SAVEPOINT before_postprocessing_event")
                    rollback_cursor.close()
            except Exception:
                pass

    def _record_service_event(
        self,
        image_id,
        service,
        event_type,
        source_service=None,
        source_stage=None,
        data=None,
        commit=False,
    ):
        """Append one image-level service event row. Best-effort and mutation-free."""
        try:
            cursor = self.db_conn.cursor()
            if not getattr(self.db_conn, "autocommit", False):
                cursor.execute("SAVEPOINT before_service_event")
            cursor.execute(
                """
                INSERT INTO service_events (
                    image_id, service, event_type, source_service, source_stage, data
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    image_id,
                    service,
                    event_type,
                    source_service,
                    source_stage,
                    json.dumps(data) if data is not None else None,
                ),
            )
            if getattr(self.db_conn, "autocommit", False) or commit:
                self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.warning(
                f"Failed to record service_event for {service}/{image_id}: {e}"
            )
            try:
                if getattr(self.db_conn, "autocommit", False):
                    self.db_conn.rollback()
                else:
                    rollback_cursor = self.db_conn.cursor()
                    rollback_cursor.execute("ROLLBACK TO SAVEPOINT before_service_event")
                    rollback_cursor.close()
            except Exception:
                pass

    def _store_terminal_service_result(
        self,
        image_id,
        payload,
        status='success',
        processing_time=None,
        service=None,
        source_trace_id=None,
    ):
        """Persist a terminal JSON result row for this service.

        Used by internal/system workers that previously treated terminal
        no-data or non-ideal outcomes as absence instead of as a real result.
        """
        service = service or self._get_clean_service_name()
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                INSERT INTO results (
                    image_id, service, source_trace_id, data, status, worker_id, processing_time
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    image_id,
                    service,
                    source_trace_id,
                    json.dumps(payload),
                    status,
                    self.worker_id,
                    processing_time,
                ),
            )
            self.db_conn.commit()
            cursor.close()
            return True
        except Exception as e:
            self.logger.error(
                f"Failed to store terminal result for {service}/{image_id}: {e}"
            )
            try:
                self.db_conn.rollback()
            except Exception:
                pass
            return False

    def _handle_sigterm(self, signum, frame):
        """Convert SIGTERM to KeyboardInterrupt so workers can clean up on windmill.sh stop."""
        raise KeyboardInterrupt("SIGTERM received")

    def _register_worker(self):
        """Register this worker in worker_registry. Called once on startup.

        - Marks any existing online row for this (service, host) as offline at its last_heartbeat
          (best approximation of when it died for unclean exits)
        - Sweeps any globally stale online rows (last_heartbeat older than 3x heartbeat interval)
        - Opportunistically deletes offline rows older than WORKER_REGISTRY_RETENTION_DAYS
        - Inserts a fresh online row for this worker
        """
        service = self._get_clean_service_name()
        host = socket.gethostname()
        stale_threshold = self._heartbeat_interval * 3
        try:
            cursor = self.db_conn.cursor()

            # Mark previous row for this (service, host) offline at its last known heartbeat
            cursor.execute("""
                UPDATE worker_registry
                SET status = 'offline', offline_at = last_heartbeat
                WHERE service = %s AND host = %s AND status = 'online'
            """, (service, host))

            # Sweep globally stale online rows from any host
            cursor.execute("""
                UPDATE worker_registry
                SET status = 'offline', offline_at = last_heartbeat
                WHERE status = 'online'
                  AND last_heartbeat < NOW() - INTERVAL '%s seconds'
            """, (stale_threshold,))

            # Insert fresh row for this worker
            cursor.execute("""
                INSERT INTO worker_registry (worker_id, service, host, started_at, last_heartbeat, status)
                VALUES (%s, %s, %s, NOW(), NOW(), 'online')
            """, (self.worker_id, service, host))

            cursor.close()
            if not self.db_conn.autocommit:
                self.db_conn.commit()
            self.logger.info(f"Registered in worker registry ({host})")
        except Exception as e:
            self.logger.warning(f"Failed to register in worker registry: {e}")

    def _heartbeat_loop(self):
        """Background thread: updates last_heartbeat every WORKER_HEARTBEAT_INTERVAL seconds.
        Marks status='offline' on clean exit so consumers can distinguish clean vs unclean shutdowns."""
        try:
            self._heartbeat_loop_inner()
        except Exception as e:
            self.logger.error(f"Heartbeat thread crashed unexpectedly: {e}", exc_info=True)

    def _heartbeat_loop_inner(self):
        """Inner heartbeat loop — separated so the outer method can catch any unexpected exit."""
        try:
            conn = self._new_db_connection(autocommit=True)
        except Exception as e:
            self.logger.warning(f"Heartbeat thread failed to connect to DB: {e}")
            return

        # Event.wait(timeout) blocks for up to timeout seconds, returns True if stop was signalled
        while not self._heartbeat_stop.wait(self._heartbeat_interval):
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE worker_registry SET last_heartbeat = NOW() WHERE worker_id = %s",
                    (self.worker_id,)
                )
                cursor.close()
                self.logger.debug("Heartbeat sent")
            except Exception as e:
                self.logger.warning(f"Heartbeat failed: {e}. Reconnecting...")
                try:
                    conn.close()
                    conn = self._new_db_connection(autocommit=True)
                except Exception as reconnect_e:
                    self.logger.error(f"Heartbeat reconnect failed: {reconnect_e}")

        # Clean shutdown — mark offline with precise offline_at timestamp
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE worker_registry SET status = 'offline', offline_at = NOW() WHERE worker_id = %s",
                (self.worker_id,)
            )
            cursor.close()
            self.logger.info("Marked offline in worker registry")
        except Exception as e:
            self.logger.warning(f"Failed to mark offline in worker registry: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _start_registry(self):
        """Register and start heartbeat thread. Call after DB connection established."""
        self._heartbeat_stop.clear()
        self._register_worker()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"{self.service_name}_heartbeat"
        )
        self._heartbeat_thread.start()

    def _stop_registry(self):
        """Signal heartbeat to stop and wait for offline marker to be written."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5)

    def after_result_stored(self, image_id, result, message):
        """Hook called after storing the ML result. Override in subclasses for
        in-place extrapolation that belongs to the same service."""
        pass

    def _cleanup(self):
        """Hook called during shutdown, after the consume loop exits but before
        the main DB connection is closed. Override in subclasses to close extra
        connections (e.g. read_db_conn) or flush other resources."""
        pass

    def _new_db_connection(self, autocommit=True):
        """Create a new PostgreSQL connection using the worker's config.
        Subclasses should use this for any additional connections (e.g. read replicas)."""
        kwargs = dict(
            host=self.db_host,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password,
            connect_timeout=10,
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=3,
        )
        if self.db_sslmode:
            kwargs['sslmode'] = self.db_sslmode
        conn = psycopg2.connect(**kwargs)
        conn.autocommit = autocommit
        return conn

    def connect_to_database(self):
        """Connect to PostgreSQL database"""
        try:
            # Close existing connection to prevent leaks on reconnect
            if self.db_conn:
                try:
                    self.db_conn.close()
                except Exception:
                    pass
            self.db_conn = self._new_db_connection(autocommit=True)
            self.logger.info(f"Connected to PostgreSQL at {self.db_host}")

            # Reset failure tracking on successful connection
            self.consecutive_db_failures = 0
            self.db_backoff_delay = 1

            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False

    def ensure_database_connection(self):
        """
        Ensure database connection is healthy, reconnect if needed.
        Returns True if connection is healthy, False otherwise.
        Implements exponential backoff on consecutive failures.

        Uses psycopg2's conn.closed attribute (0=open, 1=closed, 2=broken)
        to avoid a SELECT 1 round-trip on every call. The round-trip only
        fires when the connection looks suspect — not on the hot path.
        """
        try:
            # conn.closed == 0: open and healthy (no round-trip needed)
            # conn.closed == 1: cleanly closed
            # conn.closed == 2: broken (e.g. server went away)
            if not self.db_conn or self.db_conn.closed != 0:
                self.logger.warning("Database connection is closed or broken, reconnecting")
                return self._reconnect_database()

            # Connection is open — trust it. If the next real query fails,
            # the exception handler will reconnect then.
            if self.consecutive_db_failures > 0:
                self.consecutive_db_failures = 0
                self.db_backoff_delay = 1

            return True

        except Exception as e:
            self.logger.warning(f"Database health check failed: {e}")
            return self._reconnect_database()

    def _reconnect_database(self):
        """
        Attempt to reconnect to database with exponential backoff.
        Returns True if reconnection successful, False otherwise.
        """
        self.consecutive_db_failures += 1

        # Apply exponential backoff after consecutive failures
        if self.consecutive_db_failures >= self.max_db_failures_before_backoff:
            self.logger.warning(
                f"Database connection failed {self.consecutive_db_failures} times, "
                f"backing off for {self.db_backoff_delay}s"
            )
            time.sleep(self.db_backoff_delay)

            # Increase backoff delay (exponential with cap)
            self.db_backoff_delay = min(
                self.db_backoff_delay * 2,
                self.max_db_backoff_delay
            )

        # Attempt reconnection
        self.logger.info("Attempting database reconnection...")
        success = self.connect_to_database()

        if not success:
            self.logger.error(
                f"Database reconnection failed (attempt {self.consecutive_db_failures})"
            )

        return success
    
    def connect_to_queue(self):
        """Connect to RabbitMQ for consuming. Only declares this worker's own queue.
        Downstream queues are declared by the background publish thread."""
        try:
            self.connection = pika.BlockingConnection(self._build_queue_params())
            self.channel = self.connection.channel()

            # Declare this worker's consume queue with DLQ
            dlq_name = f"{self.queue_name}.dlq"
            self.channel.queue_declare(queue=dlq_name, durable=True)
            args = {
                'x-dead-letter-exchange': '',
                'x-dead-letter-routing-key': dlq_name
            }
            ttl_env = os.getenv('QUEUE_MESSAGE_TTL_MS')
            if ttl_env and ttl_env.isdigit() and int(ttl_env) > 0:
                args['x-message-ttl'] = int(ttl_env)
            self.channel.queue_declare(queue=self.queue_name, durable=True, arguments=args)

            self.channel.basic_qos(prefetch_count=self.worker_prefetch_count)
            self.logger.info(f"Connected to RabbitMQ at {self.queue_host}")
            self.logger.info(f"Consuming from: {self.queue_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to queue: {e}")
            return False
    
    def process_message(self, ch, method, properties, body):
        """Process a queue message - common logic for all workers"""
        image_id = None  # initialised here so the failure except blocks can reference it
        try:
            # Ensure database connection is healthy before processing
            if not self.ensure_database_connection():
                self.logger.error(
                    "Database connection unavailable, rejecting message without requeue. "
                    "Worker will retry after backoff delay."
                )
                # Reject without requeue to prevent CPU spin during DB outage
                # Message will go to DLQ after max retries
                self._safe_nack(ch, method.delivery_tag, requeue=False)
                self.job_failed("Database unavailable")
                return

            # Parse message
            message = json.loads(body)
            image_id = message['image_id']
            trace_id = message.get('trace_id')

            if trace_id:
                self.logger.debug(f"[{trace_id}] Processing {self.service_name} request for image {image_id}")
            else:
                self.logger.debug(f"Processing {self.service_name} request for image {image_id}")

            # Call ML service
            result = self.post_image_data(message['image_data'])
            if not isinstance(result, dict):
                self.logger.error(
                    f"{self.service_name} returned no terminal JSON result for image {image_id}"
                )
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                self.job_failed("No terminal JSON result")
                return

            result_status = result.get('status', 'success') or 'success'
            failed_reason = (
                result.get('error_message')
                or result.get('error')
                or result.get('message')
                or ((result.get('metadata') or {}).get('reason'))
            )

            # Store result in database
            processing_time = None
            processing_time = result.get('processing_time') or (
                result.get('metadata') or {}
            ).get('processing_time')
            source_trace_id = trace_id if trace_id else None
            cursor = self.db_conn.cursor()
            cursor.execute("""
                WITH upserted AS (
                    INSERT INTO results (
                        image_id, service, source_trace_id, data, status, worker_id, processing_time
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (image_id, service, source_trace_id)
                    WHERE source_trace_id IS NOT NULL
                    DO NOTHING
                    RETURNING image_id
                ),
                touched AS (
                    SELECT image_id FROM upserted
                    UNION ALL
                    SELECT %s
                    WHERE %s IS NOT NULL
                      AND NOT EXISTS (SELECT 1 FROM upserted)
                )
                UPDATE service_dispatch
                   SET status = %s,
                       failed_reason = %s
                 WHERE image_id = (SELECT image_id FROM touched LIMIT 1)
                   AND service = %s
                   AND cluster_id IS NULL
                   AND status = 'pending'
            """, (
                image_id,
                self._get_clean_service_name(),
                source_trace_id,
                json.dumps(result),
                result_status,
                self.worker_id,
                processing_time,
                image_id,
                source_trace_id,
                'complete' if result_status == 'success' else 'failed',
                None if result_status == 'success' else failed_reason,
                self._get_clean_service_name(),
            ))
            self.db_conn.commit()  # CRITICAL: Commit the transaction!
            cursor.close()

            if trace_id:
                self.logger.info(
                    f"[{trace_id}] stored {self.service_name} result for image {image_id} "
                    f"(status={result_status})"
                )

            if result_status != 'success':
                self._safe_ack(ch, method.delivery_tag)
                self.job_completed_successfully()
                self.logger.warning(
                    f"{self.service_name} returned terminal non-success for image {image_id}: "
                    f"{failed_reason or 'no reason provided'}"
                )
                return

            # In-place extrapolation hook: subclasses derive additional data
            # from the same result without a separate queue/worker
            self.after_result_stored(image_id, result, message)

            tier = message.get('tier', 'free')
            downstream_messages = []
            postprocessing_messages = self._build_postprocessing_messages(
                image_id=image_id,
                message=message,
                result=result,
                tier=tier,
                trace_id=trace_id,
            )

            if postprocessing_messages:
                downstream_messages.extend(postprocessing_messages)
                if trace_id:
                    self.logger.info(
                        f"[{trace_id}] queueing direct postprocessing from {self.service_name} for image {image_id}"
                    )

            if self.enable_noun_consensus:
                noun_consensus_message = {
                    'image_id': image_id,
                    'image_filename': message.get('image_filename', f'image_{image_id}'),
                    'image_data': message.get('image_data'),
                    'service': self.service_name,
                    'worker_id': self.worker_id,
                    'processed_at': datetime.now().isoformat(),
                    'tier': tier,
                }
                downstream_messages.append((
                    self._get_queue_by_service_type('noun_consensus'),
                    json.dumps(noun_consensus_message)
                ))

            if self.enable_verb_consensus:
                verb_consensus_message = {
                    'image_id': image_id,
                    'image_filename': message.get('image_filename', f'image_{image_id}'),
                    'service': self.service_name,
                    'worker_id': self.worker_id,
                    'processed_at': datetime.now().isoformat(),
                    'tier': tier,
                }
                downstream_messages.append((
                    self._get_queue_by_service_type('verb_consensus'),
                    json.dumps(verb_consensus_message)
                ))

            if self.enable_consensus_triggers:
                consensus_message = {
                    'image_id': image_id,
                    'image_filename': message.get('image_filename', f'image_{image_id}'),
                    'service': self.service_name,
                    'worker_id': self.worker_id,
                    'processed_at': datetime.now().isoformat(),
                    'tier': tier,
                }
                downstream_messages.append((
                    self._get_queue_by_service_type('consensus'),
                    json.dumps(consensus_message)
                ))

            publish_started_at = time.time()
            for routing_key, body in downstream_messages:
                self._enqueue_publish(routing_key, body)
            if trace_id and downstream_messages:
                self.logger.info(
                    f"[{trace_id}] queued {len(downstream_messages)} downstream message(s) "
                    f"from {self.service_name} for image {image_id} in {time.time() - publish_started_at:.3f}s"
                )

            # Acknowledge message
            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()

            self.logger.info(f"Successfully processed {self.service_name} request for image {image_id}")

        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            # Database connection error mid-message — requeue so the job is not lost,
            # then reconnect with backoff. The backoff delay prevents CPU spin while
            # the database is unavailable.
            self.logger.error(f"Database error processing {self.service_name} message: {e}")
            self.logger.warning("Requeueing message and attempting database reconnect...")
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self._reconnect_database()
            self.job_failed(str(e))
        except Exception as e:
            # Other errors (ML service, parsing, etc.) - requeue for retry
            self.logger.error(f"Error processing {self.service_name} message: {e}")
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self.job_failed(str(e))
    
    def start(self):
        """Start the worker"""
        self.logger.info(f"Starting {self.service_name} worker ({self.worker_id})")

        # Connect to services — retry with backoff on transient failures (e.g. DNS not yet
        # available at boot) rather than exiting immediately.
        startup_delay = 5
        while True:
            try:
                if self.connect_to_database():
                    break
            except KeyboardInterrupt:
                self.logger.info("Interrupted during startup, exiting")
                sys.exit(0)
            self.logger.warning(f"Database connection failed at startup, retrying in {startup_delay}s...")
            time.sleep(startup_delay)
            startup_delay = min(startup_delay * 2, self.max_db_backoff_delay)

        startup_delay = 5
        while True:
            try:
                if self.connect_to_queue():
                    break
            except KeyboardInterrupt:
                self.logger.info("Interrupted during startup, exiting")
                sys.exit(0)
            self.logger.warning(f"Queue connection failed at startup, retrying in {startup_delay}s...")
            time.sleep(startup_delay)
            startup_delay = min(startup_delay * 2, self.max_db_backoff_delay)

        # Start background publish thread
        self._running = True
        self._publish_thread = threading.Thread(
            target=self._start_async_publisher,
            daemon=True,
            name=f"{self.service_name}_publish"
        )
        self._publish_thread.start()

        # Register in worker registry and start heartbeat thread
        self._start_registry()

        # Start consuming with reconnect loop
        while True:
            try:
                self.channel.basic_consume(
                    queue=self.queue_name,
                    on_message_callback=self.process_message
                )
                self.logger.info("Waiting for messages. Press CTRL+C to exit")
                self.channel.start_consuming()
            except KeyboardInterrupt:
                self.logger.info("Stopping worker...")
                self._running = False
                try:
                    self.channel.stop_consuming()
                except Exception:
                    pass
                try:
                    self.connection.close()
                except Exception:
                    pass
                if self._publish_connection:
                    try:
                        self._publish_connection.ioloop.add_callback_threadsafe(
                            self._publish_connection.ioloop.stop
                        )
                    except Exception:
                        pass
                if self._publish_thread and self._publish_thread.is_alive():
                    self.logger.info("Waiting for async publisher thread to stop...")
                    self._publish_thread.join(timeout=10)
                    if self._publish_thread.is_alive():
                        self.logger.warning("Async publisher thread did not stop within timeout")
                self._stop_registry()
                break
            except (pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError,
                    pika.exceptions.StreamLostError) as e:
                self.logger.warning(f"Queue connection lost: {e}. Reconnecting...")
                time.sleep(self.retry_delay)
                if not self.connect_to_queue():
                    self.logger.warning("Reconnect failed, retrying...")
                    time.sleep(self.retry_delay)
                    continue
            finally:
                if self.db_conn and self.db_conn.closed:
                    try:
                        self.connect_to_database()
                    except Exception:
                        pass

        self._cleanup()
        if self.db_conn:
            self.db_conn.close()
        if self._sync_publish_connection:
            try:
                self._sync_publish_connection.close()
            except Exception:
                pass
        self.logger.info(f"{self.service_name} worker stopped")
