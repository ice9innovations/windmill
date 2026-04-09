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
from datetime import datetime, timezone
from dotenv import load_dotenv
from service_config import get_service_config

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.image_store import (
    get_crop,
    get_image,
    get_image_store_config,
    is_valkey_image_store_enabled,
    ping as ping_image_store,
    put_crop,
)
from core.postgres_connection import (
    close_quietly,
    commit_if_needed,
    ManagedPostgresConnection,
    PostgresConnectionConfig,
    rollback_quietly,
    create_connection,
)
from core.rabbitmq_connection import (
    declare_queue,
    ManagedRabbitMQAsyncPublisher,
    ManagedRabbitMQBlockingConnection,
    RabbitMQConnectionConfig,
)
from core.worker_registry import ManagedWorkerRegistry

class BaseWorker:
    """Base class for all ML service workers"""
    
    def __init__(self, service_name, env_file='.env'):
        self.service_name = service_name
        self.load_config(env_file)
        self.setup_logging()
        self._db_connection = ManagedPostgresConnection(
            self.db_config,
            autocommit=True,
            logger=self.logger,
            label=f"{self.service_name} main database",
        )
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
        self._sync_publish_connection = None
        self._sync_publish_channel = None
        self._running = False

        self._heartbeat_interval = int(os.getenv('WORKER_HEARTBEAT_INTERVAL', '30'))
        self._consume_queue = self._new_managed_queue_connection(label='consume queue')
        self._sync_publish_queue = self._new_managed_queue_connection(label='sync publish queue')
        self._async_publisher = ManagedRabbitMQAsyncPublisher(
            params_factory=self._build_queue_params,
            declaration_provider=self._get_publish_declarations,
            prepare_body=self._augment_publish_body,
            publish_timing_logger=self._log_publish_timing,
            logger=self.logger,
            retry_delay=self.retry_delay,
            label=f"{self.service_name}_publish",
        )
        self._registry = ManagedWorkerRegistry(
            connection_factory=self._new_db_connection,
            logger=self.logger,
            worker_id=self.worker_id,
            service=self._get_clean_service_name(),
            heartbeat_interval=self._heartbeat_interval,
            stale_threshold=self._heartbeat_interval * 3,
        )

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
        self.queue_config = RabbitMQConnectionConfig(
            host=self.queue_host,
            port=self.queue_port,
            user=self.queue_user,
            password=self.queue_password,
            use_ssl=self.queue_ssl,
            server_hostname=self.queue_host,
        )

        # Database configuration
        self.db_host = self._get_required('DB_HOST')
        self.db_name = self._get_required('DB_NAME')
        self.db_user = self._get_required('DB_USER')
        self.db_password = self._get_required('DB_PASSWORD')
        self.db_sslmode = os.getenv('DB_SSLMODE')
        self.db_config = PostgresConnectionConfig(
            host=self.db_host,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password,
            sslmode=self.db_sslmode,
        )
        
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
        # Enable direct bbox postprocessing triggers for spatial services
        self.enable_triggers = self.is_spatial
        # Enable noun and verb consensus for VLM services
        self.enable_noun_consensus = self.is_vlm
        self.enable_verb_consensus = self.is_vlm

        # Performance configuration
        self.processing_delay = float(os.getenv('PROCESSING_DELAY', '0.0'))
        self.image_store_config = get_image_store_config()

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

    def _image_transport_fields(self, message):
        if message.get('image_ref'):
            return {'image_ref': message['image_ref']}
        if message.get('image_data'):
            return {'image_data': message['image_data']}
        return {}

    def _decode_base64_field(self, encoded_value, field_name):
        if not encoded_value:
            return None
        try:
            return base64.b64decode(encoded_value)
        except Exception as e:
            self.logger.error(f"Failed to decode {field_name}: {e}")
            return None

    def _normalize_failed_reason(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=True)[:1000]
            except Exception:
                return str(value)[:1000]
        return str(value)[:1000]

    def resolve_image_bytes(self, message, required=True):
        image_ref = message.get('image_ref')
        if image_ref:
            fetch_started_at = time.time()
            image_bytes = get_image(image_ref, config=self.image_store_config, log=self.logger)
            fetch_duration = time.time() - fetch_started_at
            if image_bytes is not None:
                if self.service_name == 'system.florence2_grounding':
                    self.logger.info(
                        f"{self.service_name}: image_ref fetch image={message.get('image_id')} "
                        f"duration={fetch_duration:.3f}s bytes={len(image_bytes)}"
                    )
                return image_bytes
            self.logger.error(
                f"{self.service_name}: image_ref missing or expired for image {message.get('image_id')}: "
                f"{image_ref} (fetch_duration={fetch_duration:.3f}s)"
            )
            if not required:
                return None

        image_data = message.get('image_data')
        if image_data:
            image_bytes = self._decode_base64_field(image_data, 'image_data')
            if image_bytes is not None:
                return image_bytes

        if required:
            self.logger.error(
                f"{self.service_name}: no usable image payload for image {message.get('image_id')}"
            )
        return None

    def resolve_image_data_b64(self, message, required=True):
        image_data = message.get('image_data')
        if image_data:
            return image_data

        image_bytes = self.resolve_image_bytes(message, required=required)
        if image_bytes is None:
            return None
        return base64.b64encode(image_bytes).decode('utf-8')

    def resolve_crop_bytes(self, message, required=True):
        crop_ref = message.get('crop_ref')
        if crop_ref:
            crop_bytes = get_crop(crop_ref, config=self.image_store_config, log=self.logger)
            if crop_bytes is not None:
                return crop_bytes
            self.logger.error(
                f"{self.service_name}: crop_ref missing or expired for merged_box_id {message.get('merged_box_id')}: {crop_ref}"
            )
            if not required:
                return None

        crop_data = message.get('cropped_image_data')
        if crop_data:
            crop_bytes = self._decode_base64_field(crop_data.encode('latin-1'), 'cropped_image_data')
            if crop_bytes is not None:
                return crop_bytes

        if required:
            self.logger.error(
                f"{self.service_name}: no usable crop payload for merged_box_id {message.get('merged_box_id')}"
            )
        return None

    def build_crop_transport_fields(self, crop_bytes):
        if is_valkey_image_store_enabled():
            return {'crop_ref': put_crop(crop_bytes, config=self.image_store_config)}
        return {'cropped_image_data': base64.b64encode(crop_bytes).decode('latin-1')}

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
        """Legacy bbox postprocessing fanout is disabled."""
        return []

    def _declare_additional_queues(self, declare_queue):
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
        heartbeat = int(os.getenv('QUEUE_HEARTBEAT_SECONDS', '300'))
        blocked_timeout = int(os.getenv('QUEUE_BLOCKED_TIMEOUT_SECONDS', '600'))
        socket_timeout = int(os.getenv('QUEUE_SOCKET_TIMEOUT_SECONDS', '30'))
        connection_attempts = int(os.getenv('QUEUE_CONNECTION_ATTEMPTS', '10'))
        retry_delay = int(os.getenv('QUEUE_CONNECTION_RETRY_DELAY_SECONDS', '5'))
        tcp_options = {}
        if os.getenv('QUEUE_TCP_KEEPALIVE', 'true').lower() in ('true', '1', 'yes'):
            tcp_options[socket.SO_KEEPALIVE] = 1
            keepidle = int(os.getenv('QUEUE_TCP_KEEPIDLE_SECONDS', '60'))
            keepintvl = int(os.getenv('QUEUE_TCP_KEEPINTVL_SECONDS', '15'))
            keepcnt = int(os.getenv('QUEUE_TCP_KEEPCNT', '4'))
            if hasattr(socket, 'TCP_KEEPIDLE'):
                tcp_options[socket.TCP_KEEPIDLE] = keepidle
            if hasattr(socket, 'TCP_KEEPINTVL'):
                tcp_options[socket.TCP_KEEPINTVL] = keepintvl
            if hasattr(socket, 'TCP_KEEPCNT'):
                tcp_options[socket.TCP_KEEPCNT] = keepcnt
        return self.queue_config.build_params(
            # BlockingConnection heartbeats are serviced by the adapter loop.
            # Many workers spend long stretches inside message callbacks, so a
            # 60s heartbeat is too aggressive and causes gratuitous reconnects.
            heartbeat=heartbeat,
            blocked_connection_timeout=blocked_timeout,
            connection_attempts=connection_attempts,
            retry_delay=retry_delay,
            socket_timeout=socket_timeout,
            tcp_options=tcp_options or None,
            **overrides,
        )

    def _get_publish_declarations(self):
        """Return downstream queues this worker may publish to."""
        declarations = []

        def add_queue(queue_name):
            dlq_name = f"{queue_name}.dlq"
            declarations.append((dlq_name, None))
            args = {
                'x-dead-letter-exchange': '',
                'x-dead-letter-routing-key': dlq_name,
            }
            ttl_env = os.getenv('QUEUE_MESSAGE_TTL_MS')
            if ttl_env and ttl_env.isdigit() and int(ttl_env) > 0:
                args['x-message-ttl'] = int(ttl_env)
            declarations.append((queue_name, args))

        if self.enable_noun_consensus:
            add_queue(self._get_queue_by_service_type('noun_consensus'))
        if self.enable_verb_consensus:
            add_queue(self._get_queue_by_service_type('verb_consensus'))
        if self.config.is_available_for_tier('system.postprocessing_orchestrator', 'free') \
                or self.config.is_available_for_tier('system.postprocessing_orchestrator', 'paid') \
                or self.config.is_available_for_tier('system.postprocessing_orchestrator', 'pro'):
            add_queue(self._get_queue_by_service_type('postprocessing_orchestrator'))
        self._declare_additional_queues(add_queue)
        return declarations

    def _queue_message_ttl_ms(self):
        ttl_env = os.getenv('QUEUE_MESSAGE_TTL_MS')
        if ttl_env and ttl_env.isdigit():
            ttl_ms = int(ttl_env)
            if ttl_ms > 0:
                return ttl_ms
        return None

    def _new_managed_queue_connection(self, label='RabbitMQ queue'):
        return ManagedRabbitMQBlockingConnection(
            params_factory=self._build_queue_params,
            logger=self.logger,
            label=f"{self.service_name} {label}",
        )

    def _connect_sync_publish_channel(self):
        """Create a dedicated publish-confirm channel owned by the consume thread."""
        self._sync_publish_connection, self._sync_publish_channel = self._sync_publish_queue.connect()
        self._sync_publish_channel.confirm_delivery()
        ttl_ms = self._queue_message_ttl_ms()

        def declare_shared_queue(channel, queue_name):
            declare_queue(channel, queue_name, ttl_ms=ttl_ms)

        if self.enable_noun_consensus:
            declare_shared_queue(self._sync_publish_channel, self._get_queue_by_service_type('noun_consensus'))
        if self.enable_verb_consensus:
            declare_shared_queue(self._sync_publish_channel, self._get_queue_by_service_type('verb_consensus'))
        self._declare_additional_queues(declare_shared_queue)

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

    def _enqueue_publish(self, routing_key, body):
        """Publish through the async RabbitMQ connection without blocking the worker."""
        self._async_publisher.publish(routing_key, body)

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
        cluster_id = None
        source_service = None
        source_stage = None
        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                image_id = payload.get('image_id')
                trace_id = payload.get('trace_id')
                cluster_id = payload.get('cluster_id')
                source_service = payload.get('source_service')
                source_stage = payload.get('source_stage')
        except Exception:
            pass
        bits = [
            f"routing_key={routing_key}",
            f"local_queue_wait={local_queue_wait:.3f}s",
            f"publish_call={publish_duration:.3f}s",
        ]
        if image_id is not None:
            bits.insert(1, f"image={image_id}")
        if cluster_id is not None:
            bits.insert(2, f"cluster={cluster_id}")
        if source_service or source_stage:
            bits.insert(3, f"source={source_service or 'unknown'}:{source_stage or 'unknown'}")
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

    def _is_dispatch_terminal(self, image_id, service=None, cluster_id=None):
        """Return True when the newest matching service_dispatch row is not pending."""
        service = service or self._get_clean_service_name()
        try:
            cursor = self.db_conn.cursor()
            if cluster_id is None:
                cursor.execute(
                    """
                    SELECT status
                    FROM service_dispatch
                    WHERE image_id = %s
                      AND service = %s
                      AND cluster_id IS NULL
                    ORDER BY dispatch_id DESC
                    LIMIT 1
                    """,
                    (image_id, service),
                )
            else:
                cursor.execute(
                    """
                    SELECT status
                    FROM service_dispatch
                    WHERE image_id = %s
                      AND service = %s
                      AND cluster_id = %s
                    ORDER BY dispatch_id DESC
                    LIMIT 1
                    """,
                    (image_id, service, cluster_id),
                )
            row = cursor.fetchone()
            cursor.close()
            return bool(row and row[0] in ('complete', 'failed', 'dead-lettered', 'dead_lettered'))
        except Exception as e:
            self.logger.warning(
                f"Failed terminal-dispatch check for {service}/{image_id}/{cluster_id}: {e}"
            )
            return False

    def _store_terminal_message_failure(
        self,
        image_id,
        reason,
        trace_id=None,
        service=None,
        cluster_id=None,
        source_stage='message_validation',
    ):
        """Persist a terminal failed outcome for a consumed queue message."""
        service = service or self._get_clean_service_name()
        payload = {
            'service': service,
            'status': 'failed',
            'predictions': [],
            'error_message': reason,
            'metadata': {
                'terminal_worker_error': True,
                'terminal_reason': 'image_payload_unavailable',
            },
        }
        self._store_terminal_service_result(
            image_id=image_id,
            payload=payload,
            status='failed',
            processing_time=0.0,
            service=service,
            source_trace_id=trace_id,
            commit=True,
        )
        self._update_service_dispatch(
            image_id=image_id,
            service=service,
            cluster_id=cluster_id,
            status='failed',
            reason=reason,
        )
        self._record_service_event(
            image_id=image_id,
            service=service,
            event_type='failed',
            source_service=service,
            source_stage=source_stage,
            data={'error_message': reason},
            commit=True,
        )

    def _ack_terminal_message_failure(
        self,
        ch,
        delivery_tag,
        *,
        image_id,
        reason,
        trace_id=None,
        service=None,
        cluster_id=None,
        source_stage='message_validation',
    ):
        self._store_terminal_message_failure(
            image_id=image_id,
            reason=reason,
            trace_id=trace_id,
            service=service,
            cluster_id=cluster_id,
            source_stage=source_stage,
        )
        self._safe_ack(ch, delivery_tag)
        self.job_failed(reason)

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
    
    def post_image_bytes(self, image_bytes):
        """POST raw image bytes to service."""
        service_url = self.get_service_url()

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

    def post_image_data(self, image_data_b64):
        """POST base64 encoded image data to service."""
        image_bytes = base64.b64decode(image_data_b64)
        return self.post_image_bytes(image_bytes)

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

    def _extract_http_status(self, payload):
        """Return integer HTTP status when present on a terminal payload."""
        if not isinstance(payload, dict):
            return None
        metadata = payload.get('metadata')
        if not isinstance(metadata, dict):
            return None
        http_status = metadata.get('http_status')
        if http_status is None:
            return None
        try:
            return int(http_status)
        except (TypeError, ValueError):
            return None
    
    def trigger_noun_consensus(self, image_id, message):
        """Trigger noun consensus for VLM services (async via background publish thread)"""
        if not self.enable_noun_consensus:
            return
        trigger_started_at = time.time()
        tier = message.get('tier', 'free')
        t0 = time.time()
        is_latest, current_count, expected_count, latest_service = (
            self._is_latest_vlm_terminal_for_tier(
                image_id, tier, self._get_clean_service_name()
            )
        )
        latest_check_duration = time.time() - t0
        if not is_latest:
            self.logger.debug(
                f"Skipping noun_consensus trigger for {self.service_name} image {image_id}: "
                f"latest={latest_service!r} ({current_count}/{expected_count})"
            )
            return
        t0 = time.time()
        if not self._should_enqueue_final_consensus_service(image_id, 'noun_consensus'):
            duplicate_check_duration = time.time() - t0
            self.logger.debug(
                f"Skipping duplicate noun_consensus trigger for {self.service_name} image {image_id}"
            )
            self.logger.info(
                f"consensus_handoff service=noun_consensus image={image_id} "
                f"publisher={self._get_clean_service_name()} "
                f"latest_check={latest_check_duration:.3f}s "
                f"duplicate_check={duplicate_check_duration:.3f}s "
                f"total={time.time() - trigger_started_at:.3f}s skipped=duplicate"
            )
            return
        duplicate_check_duration = time.time() - t0

        pending_events = [{
            'image_id': image_id,
            'service': 'noun_consensus',
            'event_type': 'enqueued',
            'source_service': self._get_clean_service_name(),
            'source_stage': 'noun_consensus_trigger',
        }]
        noun_consensus_message = {
            'image_id': image_id,
            'image_filename': message.get('image_filename', f'image_{image_id}'),
            'service': self.service_name,
            'worker_id': self.worker_id,
            'processed_at': datetime.now().isoformat(),
            'tier': tier,
            'submitted_at_epoch': message.get('submitted_at_epoch'),
            'consensus_enqueued_at_epoch': time.time(),
        }
        noun_consensus_message.update(self._image_transport_fields(message))

        orchestrator_message = None
        if self.config.is_available_for_tier('system.postprocessing_orchestrator', tier):
            pending_events.append({
                'image_id': image_id,
                'service': 'postprocessing_orchestrator',
                'event_type': 'enqueued',
                'source_service': self._get_clean_service_name(),
                'source_stage': 'primary_complete_trigger',
            })
            orchestrator_message = {
                'image_id': image_id,
                'tier': tier,
                'task_type': 'primary_complete',
                'services_present': sorted(self._tier_vlm_service_names(tier)),
                'source_service': self._get_clean_service_name(),
                'submitted_at_epoch': message.get('submitted_at_epoch'),
            }

        t0 = time.time()
        event_timings = self._record_service_events_batch(pending_events, commit=True)
        record_event_duration = event_timings['insert'] + event_timings['commit']

        t0 = time.time()
        self._enqueue_publish(
            self._get_queue_by_service_type('noun_consensus'),
            json.dumps(noun_consensus_message)
        )
        if orchestrator_message is not None:
            self._enqueue_publish(
                self._get_queue_by_service_type('postprocessing_orchestrator'),
                json.dumps(orchestrator_message)
            )
        publish_duration = time.time() - t0

        self.logger.info(
            f"consensus_handoff service=noun_consensus image={image_id} "
            f"publisher={self._get_clean_service_name()} "
            f"latest_check={latest_check_duration:.3f}s "
            f"duplicate_check={duplicate_check_duration:.3f}s "
            f"record_event={record_event_duration:.3f}s "
            f"publish={publish_duration:.3f}s "
            f"total={time.time() - trigger_started_at:.3f}s"
        )

        self.logger.debug(f"Enqueued noun_consensus trigger for {self.service_name} image {image_id}")

    def trigger_verb_consensus(self, image_id, message):
        """Compatibility no-op: noun_consensus now produces verb consensus too."""
        return

    def _tier_vlm_service_names(self, tier):
        """Return clean VLM primary service names configured for the tier."""
        names = []
        for full_name in self.config.get_services_by_tier(tier):
            if not full_name.startswith('primary.'):
                continue
            if self.config.is_vlm_service(full_name):
                names.append(full_name.split('.', 1)[1])
        return names

    def _is_vlm_set_complete_for_tier(self, image_id, tier):
        """Return (ready, current_count, expected_count) for tier VLM completion."""
        try:
            vlm_services = self._tier_vlm_service_names(tier)
            expected_count = len(vlm_services)
            if expected_count == 0:
                return False, 0, 0

            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(DISTINCT service)
                FROM results
                WHERE image_id = %s
                  AND service = ANY(%s::text[])
                  AND status = 'success'
                """,
                (image_id, vlm_services),
            )
            current_count = cursor.fetchone()[0] or 0
            cursor.close()
            return current_count >= expected_count, current_count, expected_count
        except Exception as e:
            self.logger.warning(
                f"Failed VLM completeness check for image {image_id} tier={tier}: {e}"
            )
            return False, 0, 0

    def _is_latest_vlm_terminal_for_tier(self, image_id, tier, current_service):
        """Return whether the current service is the latest terminal VLM result.

        Terminal means a VLM has settled to either success or failed. This is
        stricter than mere completeness. It prevents multiple late VLMs from
        all triggering the same final consensus work after they each observe a
        fully settled tier set.
        """
        try:
            vlm_services = self._tier_vlm_service_names(tier)
            expected_count = len(vlm_services)
            if expected_count == 0:
                return False, 0, 0, None

            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                WITH scoped AS (
                    SELECT service, result_id
                    FROM results
                    WHERE image_id = %s
                      AND service = ANY(%s::text[])
                      AND status IN ('success', 'failed')
                )
                SELECT
                    (SELECT COUNT(DISTINCT service) FROM scoped) AS current_count,
                    (SELECT service FROM scoped ORDER BY result_id DESC LIMIT 1) AS latest_service
                """,
                (image_id, vlm_services),
            )
            row = cursor.fetchone()
            cursor.close()

            current_count = row[0] or 0
            latest_service = row[1]
            is_latest = (
                current_count >= expected_count and
                latest_service == current_service
            )
            return is_latest, current_count, expected_count, latest_service
        except Exception as e:
            self.logger.warning(
                f"Failed latest terminal VLM check for image {image_id} tier={tier}: {e}"
            )
            return False, 0, 0, None

    def _should_enqueue_final_consensus_service(self, image_id, service):
        """Return True when noun/verb consensus has not already been enqueued/completed."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                """
                SELECT event_type
                FROM service_events
                WHERE image_id = %s
                  AND service = %s
                ORDER BY event_id DESC
                LIMIT 1
                """,
                (image_id, service),
            )
            row = cursor.fetchone()
            cursor.close()
            if not row:
                return True
            return row[0] == 'failed'
        except Exception as e:
            self.logger.warning(
                f"Failed duplicate-consensus trigger check for image {image_id} service={service}: {e}"
            )
            return True

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
            close_quietly(cursor)
            close_duration = time.time() - close_started_at
            commit_duration = 0.0
            if commit_if_needed(self.db_conn):
                commit_started_at = time.time()
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
            rollback_quietly(self.db_conn)

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
            return {'insert': 0.0, 'commit': 0.0}

        try:
            cursor = self.db_conn.cursor()
            insert_started_at = time.time()
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
            insert_duration = time.time() - insert_started_at
            commit_duration = 0.0
            if commit_if_needed(self.db_conn, force=commit):
                commit_started_at = time.time()
                commit_duration = time.time() - commit_started_at
            close_quietly(cursor)
            return {'insert': insert_duration, 'commit': commit_duration}
        except Exception as e:
            self.logger.warning(
                f"Failed to record postprocessing_event for {service}/{image_id}/{cluster_id}: {e}"
            )
            rollback_quietly(self.db_conn)
            return {'insert': 0.0, 'commit': 0.0}

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
            insert_started_at = time.time()
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
            insert_duration = time.time() - insert_started_at
            commit_duration = 0.0
            if commit_if_needed(self.db_conn, force=commit):
                commit_started_at = time.time()
                commit_duration = time.time() - commit_started_at
            close_quietly(cursor)
            return {'insert': insert_duration, 'commit': commit_duration}
        except Exception as e:
            self.logger.warning(
                f"Failed to record service_event for {service}/{image_id}: {e}"
            )
            rollback_quietly(self.db_conn)
            return {'insert': 0.0, 'commit': 0.0}

    def _record_service_events_batch(self, events, commit=False):
        """Append multiple image-level service event rows in one DB round trip."""
        if not events:
            return {'insert': 0.0, 'commit': 0.0}
        try:
            values_sql = ", ".join(["(%s, %s, %s, %s, %s, %s)"] * len(events))
            params = []
            for event in events:
                params.extend([
                    event['image_id'],
                    event['service'],
                    event['event_type'],
                    event.get('source_service'),
                    event.get('source_stage'),
                    json.dumps(event.get('data')) if event.get('data') is not None else None,
                ])

            cursor = self.db_conn.cursor()
            insert_started_at = time.time()
            cursor.execute(
                f"""
                INSERT INTO service_events (
                    image_id, service, event_type, source_service, source_stage, data
                )
                VALUES {values_sql}
                """,
                params,
            )
            insert_duration = time.time() - insert_started_at
            commit_duration = 0.0
            if commit_if_needed(self.db_conn, force=commit):
                commit_started_at = time.time()
                commit_duration = time.time() - commit_started_at
            close_quietly(cursor)
            return {'insert': insert_duration, 'commit': commit_duration}
        except Exception as e:
            self.logger.warning(f"Failed to record batched service_events: {e}")
            rollback_quietly(self.db_conn)
            return {'insert': 0.0, 'commit': 0.0}

    def _store_terminal_service_result(
        self,
        image_id,
        payload,
        status='success',
        processing_time=None,
        service=None,
        source_trace_id=None,
        commit=True,
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
                    image_id, service, source_trace_id, data, status, http_status, worker_id, processing_time
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (image_id, service, source_trace_id)
                WHERE source_trace_id IS NOT NULL
                DO NOTHING
                """,
                (
                    image_id,
                    service,
                    source_trace_id,
                    json.dumps(payload),
                    status,
                    self._extract_http_status(payload),
                    self.worker_id,
                    processing_time,
                ),
            )
            commit_if_needed(self.db_conn, force=commit)
            close_quietly(cursor)
            return True
        except Exception as e:
            self.logger.error(
                f"Failed to store terminal result for {service}/{image_id}: {e}"
            )
            rollback_quietly(self.db_conn)
            return False

    def _handle_sigterm(self, signum, frame):
        """Convert SIGTERM to KeyboardInterrupt so workers can clean up on windmill.sh stop."""
        raise KeyboardInterrupt("SIGTERM received")

    def _start_registry(self):
        """Register and start heartbeat thread. Call after DB connection established."""
        try:
            self._registry.start(self.db_conn)
            self.logger.info(f"Registered in worker registry ({self._registry.host})")
        except Exception as e:
            self.logger.warning(f"Failed to register in worker registry: {e}")

    def _stop_registry(self):
        """Signal heartbeat to stop and wait for offline marker to be written."""
        self._registry.stop(join_timeout=5)

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
        return create_connection(self.db_config, autocommit=autocommit)

    def _new_managed_db_connection(self, autocommit=True, label='database'):
        return ManagedPostgresConnection(
            self.db_config,
            autocommit=autocommit,
            logger=self.logger,
            label=f"{self.service_name} {label}",
        )

    def _connect_main_database(self, autocommit=True):
        self._db_connection.autocommit = autocommit
        self.db_conn = self._db_connection.connect()
        self.logger.info(f"Connected to PostgreSQL at {self.db_host}")

        self.consecutive_db_failures = 0
        self.db_backoff_delay = 1
        return True

    def connect_to_database(self):
        """Connect to PostgreSQL database"""
        try:
            return self._connect_main_database(autocommit=True)
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

    def _retry_db_read_once(self, operation_name: str, fn, default):
        """Retry one safe read after a forced reconnect.

        This is intentionally limited to retry-safe read paths. Write paths need
        explicit idempotency/transaction rules and should not silently reuse this.
        """
        try:
            return fn()
        except Exception as first_error:
            self.logger.warning(
                f"{self.service_name}: {operation_name} failed on first attempt: "
                f"{first_error}; retrying after reconnect"
            )
            if not self._reconnect_database():
                self.logger.warning(
                    f"{self.service_name}: reconnect failed while retrying {operation_name}"
                )
                return default
            try:
                return fn()
            except Exception as second_error:
                self.logger.warning(
                    f"{self.service_name}: could not {operation_name}: {second_error}"
                )
                return default
    
    def connect_to_queue(self):
        """Connect to RabbitMQ for consuming. Only declares this worker's own queue.
        Downstream queues are declared by the background publish thread."""
        try:
            self.connection, self.channel = self._consume_queue.connect()
            declare_queue(
                self.channel,
                self.queue_name,
                ttl_ms=self._queue_message_ttl_ms(),
            )

            self.channel.basic_qos(prefetch_count=self.worker_prefetch_count)
            self.logger.info(f"Connected to RabbitMQ at {self.queue_host}")
            self.logger.info(f"Consuming from: {self.queue_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to queue: {e}")
            return False

    def warm_image_store_connection(self):
        """Eagerly establish the worker's Valkey/TLS client before first live message."""
        if not is_valkey_image_store_enabled():
            return True
        started_at = time.time()
        try:
            ping_image_store(config=self.image_store_config)
            self.logger.info(
                f"Connected to image store at {self.image_store_config.host}:{self.image_store_config.port} "
                f"in {time.time() - started_at:.3f}s"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to image store: {e}")
            return False
    
    def process_message(self, ch, method, properties, body):
        """Process a queue message - common logic for all workers"""
        image_id = None  # initialised here so the failure except blocks can reference it
        try:
            callback_started_at = time.time()
            # Ensure database connection is healthy before processing
            if not self.ensure_database_connection():
                self.logger.error(
                    "Database connection unavailable, requeueing message. "
                    "Worker will retry after backoff delay."
                )
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                self.job_failed("Database unavailable")
                return

            # Parse message
            parse_started_at = time.time()
            message = json.loads(body)
            parse_duration = time.time() - parse_started_at
            image_id = message['image_id']
            trace_id = message.get('trace_id')
            submitted_at = message.get('submitted_at')
            submitted_at_epoch = message.get('submitted_at_epoch')
            worker_received_at = time.time()

            def _latency_from_iso(value):
                if not value:
                    return None
                try:
                    return worker_received_at - datetime.fromisoformat(value).timestamp()
                except Exception:
                    return None

            upstream_queue_wait = None
            if submitted_at_epoch is not None:
                try:
                    upstream_queue_wait = worker_received_at - float(submitted_at_epoch)
                except Exception:
                    upstream_queue_wait = None
            if upstream_queue_wait is None:
                upstream_queue_wait = _latency_from_iso(submitted_at)

            if self._is_dispatch_terminal(image_id, self._get_clean_service_name(), cluster_id=None):
                self.logger.warning(
                    f"{self.service_name}: dispatch already terminal for image {image_id}, acking redelivery"
                )
                self._safe_ack(ch, method.delivery_tag)
                return

            if trace_id:
                self.logger.debug(f"[{trace_id}] Processing {self.service_name} request for image {image_id}")
            else:
                self.logger.debug(f"Processing {self.service_name} request for image {image_id}")

            # Call ML service
            fetch_started_at = time.time()
            image_bytes = self.resolve_image_bytes(message)
            image_fetch_duration = time.time() - fetch_started_at
            if image_bytes is None:
                failure_reason = "image_ref missing, expired, or otherwise unavailable"
                self._ack_terminal_message_failure(
                    ch,
                    method.delivery_tag,
                    image_id=image_id,
                    reason=failure_reason,
                    trace_id=trace_id,
                    service=self._get_clean_service_name(),
                    source_stage='image_transport_resolve',
                )
                return

            request_started_at = time.time()
            result = self.post_image_bytes(image_bytes)
            request_duration = time.time() - request_started_at
            request_completed_at = time.time()
            if not isinstance(result, dict):
                self.logger.error(
                    f"{self.service_name} returned no terminal JSON result for image {image_id}"
                )
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                self.job_failed("No terminal JSON result")
                return

            result_status = result.get('status', 'success') or 'success'
            failed_reason = self._normalize_failed_reason(
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
            service_name = self._get_clean_service_name()
            lifecycle_source_service = message.get('service') or service_name

            def _iso_utc(epoch_seconds):
                return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()

            received_event_data = json.dumps({
                'event_at': _iso_utc(worker_received_at),
                'trace_id': trace_id,
                'worker_id': self.worker_id,
                'submitted_at': submitted_at,
                'submitted_at_epoch': submitted_at_epoch,
                'upstream_queue_wait_seconds': round(upstream_queue_wait, 6) if upstream_queue_wait is not None else None,
                'image_fetch_duration_seconds': round(image_fetch_duration, 6),
            })
            started_event_data = json.dumps({
                'event_at': _iso_utc(request_started_at),
                'trace_id': trace_id,
                'worker_id': self.worker_id,
                'image_fetch_duration_seconds': round(image_fetch_duration, 6),
            })
            terminal_event_data = json.dumps({
                'event_at': _iso_utc(request_completed_at),
                'trace_id': trace_id,
                'worker_id': self.worker_id,
                'request_duration_seconds': round(request_duration, 6),
                'image_fetch_duration_seconds': round(image_fetch_duration, 6),
                'upstream_queue_wait_seconds': round(upstream_queue_wait, 6) if upstream_queue_wait is not None else None,
                'http_status': self._extract_http_status(result),
                'result_status': result_status,
                'processing_time_seconds': processing_time,
                'failed_reason': failed_reason,
            })
            persist_started_at = time.time()
            cursor = self.db_conn.cursor()
            cursor.execute("""
                WITH upserted AS (
                    INSERT INTO results (
                        image_id, service, source_trace_id, data, status, http_status, worker_id, processing_time
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
                ),
                updated_dispatch AS (
                    UPDATE service_dispatch
                       SET status = %s,
                           failed_reason = %s
                     WHERE image_id = (SELECT image_id FROM touched LIMIT 1)
                       AND service = %s
                       AND cluster_id IS NULL
                       AND status = 'pending'
                    RETURNING image_id
                )
                INSERT INTO service_events (
                    image_id, service, event_type, source_service, source_stage, data
                )
                SELECT
                    upserted.image_id,
                    %s,
                    event_rows.event_type,
                    %s,
                    event_rows.source_stage,
                    event_rows.data::jsonb
                FROM upserted
                CROSS JOIN (
                    VALUES
                        (%s, %s, %s),
                        (%s, %s, %s),
                        (%s, %s, %s)
                ) AS event_rows(event_type, source_stage, data)
            """, (
                image_id,
                service_name,
                source_trace_id,
                json.dumps(result),
                result_status,
                self._extract_http_status(result),
                self.worker_id,
                processing_time,
                image_id,
                source_trace_id,
                'complete' if result_status == 'success' else 'failed',
                None if result_status == 'success' else failed_reason,
                service_name,
                service_name,
                lifecycle_source_service,
                'received',
                'queue_consume',
                received_event_data,
                'started',
                'service_request_started',
                started_event_data,
                'completed' if result_status == 'success' else 'failed',
                'service_request_finished',
                terminal_event_data,
            ))
            commit_if_needed(self.db_conn, force=True)
            persist_duration = time.time() - persist_started_at
            close_quietly(cursor)

            if trace_id:
                self.logger.info(
                    f"[{trace_id}] stored {self.service_name} result for image {image_id} "
                    f"(status={result_status})"
                )

            terminal_non_success = result_status != 'success'
            if terminal_non_success:
                self.logger.warning(
                    f"{self.service_name} returned terminal non-success for image {image_id}: "
                    f"{failed_reason or 'no reason provided'}"
                )

            # In-place extrapolation hook: subclasses derive additional data
            # from the same result without a separate queue/worker
            poststore_started_at = time.time()
            if not terminal_non_success:
                self.after_result_stored(image_id, result, message)

            tier = message.get('tier', 'free')
            downstream_messages = []
            if not terminal_non_success:
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

            latest_check_started_at = time.time()
            vlm_tail_ready, current_vlm_count, expected_vlm_count, latest_vlm_service = (
                self._is_latest_vlm_terminal_for_tier(
                    image_id, tier, self._get_clean_service_name()
                )
            )
            latest_check_duration = time.time() - latest_check_started_at
            noun_duplicate_check_duration = 0.0
            consensus_event_insert_duration = 0.0
            consensus_event_commit_duration = 0.0
            enqueued_noun_consensus = False
            pending_consensus_events = []

            should_enqueue_noun_consensus = False
            if self.enable_noun_consensus and vlm_tail_ready:
                t0 = time.time()
                should_enqueue_noun_consensus = self._should_enqueue_final_consensus_service(image_id, 'noun_consensus')
                noun_duplicate_check_duration = time.time() - t0

            if self.enable_noun_consensus and should_enqueue_noun_consensus:
                pending_consensus_events.append({
                    'image_id': image_id,
                    'service': 'noun_consensus',
                    'event_type': 'enqueued',
                    'source_service': self._get_clean_service_name(),
                    'source_stage': 'noun_consensus_trigger',
                })
                noun_consensus_message = {
                    'image_id': image_id,
                    'image_filename': message.get('image_filename', f'image_{image_id}'),
                    'service': self.service_name,
                    'worker_id': self.worker_id,
                    'processed_at': datetime.now().isoformat(),
                    'tier': tier,
                    'submitted_at_epoch': message.get('submitted_at_epoch'),
                    'consensus_enqueued_at_epoch': time.time(),
                }
                noun_consensus_message.update(self._image_transport_fields(message))
                downstream_messages.append((
                    self._get_queue_by_service_type('noun_consensus'),
                    json.dumps(noun_consensus_message)
                ))
                enqueued_noun_consensus = True
            elif self.enable_noun_consensus:
                self.logger.debug(
                    f"Skipping noun_consensus enqueue for {self.service_name} image {image_id}: "
                    f"latest={latest_vlm_service!r} ({current_vlm_count}/{expected_vlm_count})"
                )

            orchestrator_duplicate_check_duration = 0.0
            enqueued_postprocessing_orchestrator = False
            should_enqueue_postprocessing_orchestrator = False
            if self.config.is_available_for_tier('system.postprocessing_orchestrator', tier) and vlm_tail_ready:
                should_enqueue_postprocessing_orchestrator = True
            if should_enqueue_postprocessing_orchestrator:
                pending_consensus_events.append({
                    'image_id': image_id,
                    'service': 'postprocessing_orchestrator',
                    'event_type': 'enqueued',
                    'source_service': self._get_clean_service_name(),
                    'source_stage': 'primary_complete_trigger',
                })
                postprocessing_message = {
                    'image_id': image_id,
                    'tier': tier,
                    'task_type': 'primary_complete',
                    'services_present': sorted(self._tier_vlm_service_names(tier)),
                    'source_service': self._get_clean_service_name(),
                    'submitted_at_epoch': message.get('submitted_at_epoch'),
                }
                downstream_messages.append((
                    self._get_queue_by_service_type('postprocessing_orchestrator'),
                    json.dumps(postprocessing_message)
                ))
                enqueued_postprocessing_orchestrator = True
            elif self.config.is_available_for_tier('system.postprocessing_orchestrator', tier):
                self.logger.debug(
                    f"Skipping postprocessing_orchestrator enqueue for {self.service_name} image {image_id}: "
                    f"latest={latest_vlm_service!r} ({current_vlm_count}/{expected_vlm_count})"
                )
            if pending_consensus_events:
                event_timings = self._record_service_events_batch(
                    pending_consensus_events,
                    commit=True,
                )
                consensus_event_insert_duration = event_timings['insert']
                consensus_event_commit_duration = event_timings['commit']
            poststore_duration = time.time() - poststore_started_at

            publish_started_at = time.time()
            for routing_key, body in downstream_messages:
                self._enqueue_publish(routing_key, body)
            publish_duration = time.time() - publish_started_at
            if enqueued_noun_consensus:
                self.logger.info(
                    f"consensus_handoff service=noun_consensus image={image_id} "
                    f"publisher={self._get_clean_service_name()} "
                    f"latest_check={latest_check_duration:.3f}s "
                    f"duplicate_check={noun_duplicate_check_duration:.3f}s "
                    f"record_event={consensus_event_insert_duration + consensus_event_commit_duration:.3f}s "
                    f"publish={publish_duration:.3f}s"
                )
            if enqueued_postprocessing_orchestrator:
                self.logger.info(
                    f"consensus_handoff service=postprocessing_orchestrator image={image_id} "
                    f"publisher={self._get_clean_service_name()} "
                    f"latest_check={latest_check_duration:.3f}s "
                    f"duplicate_check={orchestrator_duplicate_check_duration:.3f}s "
                    f"record_event={consensus_event_insert_duration + consensus_event_commit_duration:.3f}s "
                    f"publish={publish_duration:.3f}s"
                )
            if trace_id and downstream_messages:
                self.logger.info(
                    f"[{trace_id}] queued {len(downstream_messages)} downstream message(s) "
                    f"from {self.service_name} for image {image_id} in {time.time() - publish_started_at:.3f}s"
                )

            # Acknowledge message
            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()

            timing = [
                f"{self.service_name} image={image_id}",
                f"parse={parse_duration:.3f}s",
                f"fetch={image_fetch_duration:.3f}s",
                f"request={request_duration:.3f}s",
                f"persist={persist_duration:.3f}s",
                f"post_store={poststore_duration:.3f}s",
                f"publish={publish_duration:.3f}s",
                (
                    f"service_processing_time={processing_time:.3f}s"
                    if processing_time is not None else
                    "service_processing_time=None"
                ),
                f"total={time.time() - callback_started_at:.3f}s",
                "status=success",
            ]
            if upstream_queue_wait is not None:
                timing.insert(1, f"from_submit={upstream_queue_wait:.3f}s")
            self.logger.info(" ".join(timing))

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
                if self.warm_image_store_connection():
                    break
            except KeyboardInterrupt:
                self.logger.info("Interrupted during startup, exiting")
                sys.exit(0)
            self.logger.warning(f"Image store connection failed at startup, retrying in {startup_delay}s...")
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
        self._async_publisher.start()

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
                self._consume_queue.close()
                self.logger.info("Waiting for async publisher thread to stop...")
                self._async_publisher.stop(join_timeout=10)
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
        close_quietly(self.db_conn)
        self._consume_queue.close()
        self._sync_publish_queue.close()
        self.logger.info(f"{self.service_name} worker stopped")
