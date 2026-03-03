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
import queue as stdlib_queue
import ssl
import pika
import psycopg2
import requests
import yaml
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

        # Background publish thread — decouples downstream trigger publishes
        # from the consume loop so they don't block ML throughput
        self._publish_queue = stdlib_queue.Queue(maxsize=1000)
        self._publish_thread = None
        self._publish_connection = None
        self._publish_channel = None
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

        # Enable triggers for spatial services (for bbox harmonization)
        self.enable_triggers = self.is_spatial
        # Enable caption scoring for semantic services
        self.enable_caption_scoring = self.is_semantic
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

    def _connect_publish_channel(self):
        """Create a separate RabbitMQ connection and channel for background publishing.
        This connection is owned by the publish thread and never touched by the consume thread."""
        self._publish_connection = pika.BlockingConnection(self._build_queue_params())
        self._publish_channel = self._publish_connection.channel()

        # Declare all downstream queues this worker might publish to
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

        # Declare downstream queues based on what this worker triggers
        if self.enable_triggers:
            declare_with_dlq(self._publish_channel, self._get_queue_by_service_type('harmonization'))
        if self.enable_consensus_triggers:
            declare_with_dlq(self._publish_channel, self._get_queue_by_service_type('consensus'))
        if self.enable_caption_scoring:
            declare_with_dlq(self._publish_channel, self._get_queue_by_service_type('caption_score'))
        if self.enable_noun_consensus:
            declare_with_dlq(self._publish_channel, self._get_queue_by_service_type('noun_consensus'))
        if self.enable_verb_consensus:
            declare_with_dlq(self._publish_channel, self._get_queue_by_service_type('verb_consensus'))
        if self.enable_noun_consensus:
            declare_with_dlq(self._publish_channel, self._get_queue_by_service_type('sam3'))

        self.logger.info("Publish thread connected to RabbitMQ")

    def _publish_loop(self):
        """Background thread target: drains the publish queue and sends messages
        on its own dedicated pika connection."""
        try:
            self._connect_publish_channel()
        except Exception as e:
            self.logger.error(f"Publish thread failed to connect: {e}")
            return

        while self._running:
            try:
                routing_key, body = self._publish_queue.get(timeout=0.5)
            except stdlib_queue.Empty:
                # Process pika heartbeats during idle periods to prevent
                # the broker from closing the connection
                try:
                    self._publish_connection.process_data_events()
                except Exception as e:
                    self.logger.warning(f"Publish connection heartbeat failed: {e}. Reconnecting...")
                    try:
                        self._connect_publish_channel()
                    except Exception as reconnect_e:
                        self.logger.error(f"Publish thread reconnect failed: {reconnect_e}")
                continue

            try:
                self._publish_channel.basic_publish(
                    exchange='',
                    routing_key=routing_key,
                    body=body,
                    properties=pika.BasicProperties(delivery_mode=2)
                )
            except (pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError,
                    pika.exceptions.StreamLostError) as e:
                self.logger.warning(f"Publish connection lost: {e}. Reconnecting...")
                try:
                    self._connect_publish_channel()
                    # Retry the failed publish after reconnecting
                    self._publish_channel.basic_publish(
                        exchange='',
                        routing_key=routing_key,
                        body=body,
                        properties=pika.BasicProperties(delivery_mode=2)
                    )
                except Exception as retry_e:
                    self.logger.error(f"Publish retry failed after reconnect: {retry_e}")
            except Exception as e:
                self.logger.error(f"Publish error: {e}")

        # Drain remaining messages on shutdown
        drained = 0
        while not self._publish_queue.empty():
            try:
                routing_key, body = self._publish_queue.get_nowait()
                self._publish_channel.basic_publish(
                    exchange='',
                    routing_key=routing_key,
                    body=body,
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                drained += 1
            except Exception:
                break
        if drained:
            self.logger.info(f"Publish thread drained {drained} remaining messages on shutdown")

        try:
            self._publish_connection.close()
        except Exception:
            pass
        self.logger.info("Publish thread stopped")

    def _enqueue_publish(self, routing_key, body):
        """Enqueue a message for background publishing. Non-blocking — returns immediately."""
        try:
            self._publish_queue.put_nowait((routing_key, body))
        except stdlib_queue.Full:
            self.logger.error(
                f"Publish queue full ({self._publish_queue.maxsize}), "
                f"dropping message to {routing_key}"
            )

    def _safe_ack(self, ch, delivery_tag):
        """Ack a message, logging instead of raising if the channel is dead."""
        try:
            ch.basic_ack(delivery_tag=delivery_tag)
        except (pika.exceptions.AMQPChannelError, pika.exceptions.AMQPConnectionError,
                pika.exceptions.StreamLostError) as e:
            self.logger.warning(f"Channel dead during ack (message will be redelivered): {e}")
            raise  # Let it propagate to start()'s reconnect loop

    def _safe_nack(self, ch, delivery_tag, requeue=True):
        """Nack a message, swallowing channel errors since we're already in an error path."""
        try:
            ch.basic_nack(delivery_tag=delivery_tag, requeue=requeue)
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
        response = requests.post(
            service_url,
            files=files,
            timeout=self.request_timeout
        )
        response.raise_for_status()
        return response.json()
    
    def trigger_consensus(self, image_id, message):
        """Trigger consensus processing (async via background publish thread)"""
        if self.enable_consensus_triggers:
            try:
                consensus_message = {
                    'image_id': image_id,
                    'image_filename': message.get('image_filename', f'image_{image_id}'),
                    'service': self.service_name,
                    'worker_id': self.worker_id,
                    'processed_at': datetime.now().isoformat()
                }

                self._enqueue_publish(
                    self._get_queue_by_service_type('consensus'),
                    json.dumps(consensus_message)
                )

                self.logger.debug(f"Enqueued consensus trigger for {self.service_name} image {image_id}")

            except Exception as e:
                self.logger.error(f"Failed to enqueue consensus message: {e}")
    
    def trigger_caption_scoring(self, image_id, message):
        """Trigger caption scoring for caption generation services (async via background publish thread)"""
        if self.enable_caption_scoring:
            try:
                caption_score_message = {
                    'image_id': image_id,
                    'image_filename': message.get('image_filename', f'image_{image_id}'),
                    'image_data': message['image_data'],
                    'service': self.service_name,
                    'worker_id': self.worker_id,
                    'processed_at': datetime.now().isoformat()
                }

                self._enqueue_publish(
                    self._get_queue_by_service_type('caption_score'),
                    json.dumps(caption_score_message)
                )

                self.logger.debug(f"Enqueued caption scoring trigger for {self.service_name} image {image_id}")

            except Exception as e:
                self.logger.error(f"Failed to enqueue caption score message: {e}")
    
    def trigger_noun_consensus(self, image_id, message):
        """Trigger noun consensus for VLM services (async via background publish thread)"""
        if not self.enable_noun_consensus:
            return
        try:
            noun_consensus_message = {
                'image_id': image_id,
                'image_filename': message.get('image_filename', f'image_{image_id}'),
                'image_data': message.get('image_data'),
                'service': self.service_name,
                'worker_id': self.worker_id,
                'processed_at': datetime.now().isoformat()
            }

            self._enqueue_publish(
                self._get_queue_by_service_type('noun_consensus'),
                json.dumps(noun_consensus_message)
            )

            self.logger.debug(f"Enqueued noun_consensus trigger for {self.service_name} image {image_id}")

        except Exception as e:
            self.logger.error(f"Failed to enqueue noun_consensus message: {e}")

    def trigger_verb_consensus(self, image_id, message):
        """Trigger verb consensus for VLM services (async via background publish thread)"""
        if not self.enable_verb_consensus:
            return
        try:
            verb_consensus_message = {
                'image_id': image_id,
                'image_filename': message.get('image_filename', f'image_{image_id}'),
                'service': self.service_name,
                'worker_id': self.worker_id,
                'processed_at': datetime.now().isoformat()
            }

            self._enqueue_publish(
                self._get_queue_by_service_type('verb_consensus'),
                json.dumps(verb_consensus_message)
            )

            self.logger.debug(f"Enqueued verb_consensus trigger for {self.service_name} image {image_id}")

        except Exception as e:
            self.logger.error(f"Failed to enqueue verb_consensus message: {e}")

    def _record_service_dispatch(self, image_id, service, cluster_id=None):
        """Insert a pending service_dispatch record. Best-effort; errors swallowed.

        On autocommit=False connections the INSERT is wrapped in a SAVEPOINT so
        that a failure (e.g. FK violation) rolls back only this INSERT and leaves
        the parent transaction alive.
        """
        try:
            cursor = self.db_conn.cursor()
            if not self.db_conn.autocommit:
                cursor.execute("SAVEPOINT before_service_dispatch")
            cursor.execute(
                "INSERT INTO service_dispatch (image_id, service, cluster_id) VALUES (%s, %s, %s)",
                (image_id, service, cluster_id),
            )
            cursor.close()
        except Exception as e:
            self.logger.warning(
                f"Failed to record service_dispatch for {service}/{image_id}: {e}"
            )
            if not self.db_conn.autocommit:
                try:
                    self.db_conn.cursor().execute("ROLLBACK TO SAVEPOINT before_service_dispatch")
                except Exception:
                    pass

    def _update_service_dispatch(self, image_id, service=None, cluster_id=None, status='complete'):
        """Update service_dispatch to the given status. Best-effort; errors swallowed.

        cluster_id=None matches rows WHERE cluster_id IS NULL (image-level services).
        Provide cluster_id for bbox-level services such as face and pose.
        service defaults to this worker's clean service name.

        On autocommit=False connections the UPDATE is committed automatically.
        """
        service = service or self._get_clean_service_name()
        try:
            cursor = self.db_conn.cursor()
            if cluster_id is None:
                cursor.execute(
                    """UPDATE service_dispatch SET status = %s
                       WHERE image_id = %s AND service = %s
                         AND cluster_id IS NULL AND status = 'pending'""",
                    (status, image_id, service),
                )
            else:
                cursor.execute(
                    """UPDATE service_dispatch SET status = %s
                       WHERE image_id = %s AND service = %s
                         AND cluster_id = %s AND status = 'pending'""",
                    (status, image_id, service, cluster_id),
                )
            cursor.close()
            if not self.db_conn.autocommit:
                self.db_conn.commit()
        except Exception as e:
            self.logger.warning(
                f"Failed to update service_dispatch for {service}/{image_id}: {e}"
            )
            if not self.db_conn.autocommit:
                try:
                    self.db_conn.rollback()
                except Exception:
                    pass

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
        """
        try:
            # Check if connection exists and is open
            if not self.db_conn or self.db_conn.closed:
                self.logger.warning("Database connection is closed")
                return self._reconnect_database()

            # Validate connection with a test query
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()

            # Connection is healthy - reset failure tracking
            if self.consecutive_db_failures > 0:
                self.logger.info("Database connection restored")
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

            # Store result in database
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO results (image_id, service, data, status, worker_id)
                VALUES (%s, %s, %s, %s, %s)
            """, (image_id, self._get_clean_service_name(), json.dumps(result), 'success', self.worker_id))
            self.db_conn.commit()  # CRITICAL: Commit the transaction!
            cursor.close()

            # Mark service as complete in dispatch tracking — best-effort
            self._update_service_dispatch(image_id)

            # In-place extrapolation hook: subclasses derive additional data
            # from the same result without a separate queue/worker
            self.after_result_stored(image_id, result, message)

            # Trigger post-processing for bbox services (async via background publish thread)
            if self.service_name in self.bbox_services and self.enable_triggers:
                bbox_message = {
                    'image_id': image_id,
                    'image_filename': message.get('image_filename', f'image_{image_id}'),
                    'image_data': message['image_data'],
                    'trace_id': trace_id,
                    'service': self.service_name,
                    'worker_id': self.worker_id,
                    'processed_at': datetime.now().isoformat()
                }

                self._enqueue_publish(
                    self._get_queue_by_service_type('harmonization'),
                    json.dumps(bbox_message)
                )

                self.logger.debug(f"Enqueued bbox completion to harmonization")

            # Trigger caption scoring
            if self.enable_caption_scoring:
                self.trigger_caption_scoring(image_id, message)

            # Trigger noun and verb consensus for VLM services
            self.trigger_noun_consensus(image_id, message)
            self.trigger_verb_consensus(image_id, message)

            # Trigger consensus processing
            self.trigger_consensus(image_id, message)

            # Acknowledge message
            self._safe_ack(ch, method.delivery_tag)
            self.job_completed_successfully()

            self.logger.info(f"Successfully processed {self.service_name} request for image {image_id}")

        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            # Database connection errors - don't requeue to prevent CPU spin
            self.logger.error(f"Database error processing {self.service_name} message: {e}")
            self.logger.warning("Rejecting message without requeue due to database error")
            self._safe_nack(ch, method.delivery_tag, requeue=False)
            self.job_failed(str(e))
        except Exception as e:
            # Other errors (ML service, parsing, etc.) - requeue for retry
            self.logger.error(f"Error processing {self.service_name} message: {e}")
            self._safe_nack(ch, method.delivery_tag, requeue=True)
            self.job_failed(str(e))
    
    def start(self):
        """Start the worker"""
        self.logger.info(f"Starting {self.service_name} worker ({self.worker_id})")

        # Connect to services
        if not self.connect_to_database():
            sys.exit(1)
        if not self.connect_to_queue():
            sys.exit(1)

        # Start background publish thread
        self._running = True
        self._publish_thread = threading.Thread(
            target=self._publish_loop,
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
                if self._publish_thread and self._publish_thread.is_alive():
                    self.logger.info("Waiting for publish thread to drain...")
                    self._publish_thread.join(timeout=10)
                    if self._publish_thread.is_alive():
                        self.logger.warning("Publish thread did not stop within timeout")
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

        if self.db_conn:
            self.db_conn.close()
        self.logger.info(f"{self.service_name} worker stopped")