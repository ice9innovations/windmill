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
        self.queue_user = self._get_required('QUEUE_USER')
        self.queue_password = self._get_required('QUEUE_PASSWORD')
        
        # Database configuration
        self.db_host = self._get_required('DB_HOST')
        self.db_name = self._get_required('DB_NAME')
        self.db_user = self._get_required('DB_USER')
        self.db_password = self._get_required('DB_PASSWORD')
        
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
        self.enable_consensus_triggers = self.config.should_trigger_consensus(self.service_name)
        
        # Enable triggers for spatial services (for bbox harmonization)
        self.enable_triggers = self.is_spatial
        # Enable caption scoring for semantic services
        self.enable_caption_scoring = self.is_semantic
        
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
        """Trigger consensus processing"""
        if self.enable_consensus_triggers:
            try:
                # Check if connection is healthy
                if not self.channel or self.connection.is_closed:
                    self.logger.warning("RabbitMQ connection lost, reconnecting...")
                    if not self.connect_to_queue():
                        self.logger.error("Failed to reconnect to RabbitMQ for consensus message")
                        raise Exception("RabbitMQ reconnection failed")
                
                consensus_message = {
                    'image_id': image_id,
                    'image_filename': message.get('image_filename', f'image_{image_id}'),
                    'image_data': message['image_data'],  # Pass through base64 image data
                    'service': self.service_name,
                    'worker_id': self.worker_id,
                    'processed_at': datetime.now().isoformat()
                }
                
                self.channel.basic_publish(
                    exchange='',
                    routing_key=self._get_queue_by_service_type('consensus'),
                    body=json.dumps(consensus_message),
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                
                self.logger.debug(f"Published consensus trigger for {self.service_name} image {image_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to publish consensus message: {e}")
    
    def trigger_caption_scoring(self, image_id, message):
        """Trigger caption scoring for caption generation services"""
        if self.enable_caption_scoring:
            try:
                # Check if connection is healthy
                if not self.channel or self.connection.is_closed:
                    self.logger.warning("RabbitMQ connection lost, reconnecting...")
                    if not self.connect_to_queue():
                        self.logger.error("Failed to reconnect to RabbitMQ for consensus message")
                        raise Exception("RabbitMQ reconnection failed")
                
                caption_score_message = {
                    'image_id': image_id,
                    'image_filename': message.get('image_filename', f'image_{image_id}'),
                    'image_data': message['image_data'],  # Pass through base64 image data
                    'service': self.service_name,
                    'worker_id': self.worker_id,
                    'processed_at': datetime.now().isoformat()
                }
                
                self.channel.basic_publish(
                    exchange='',
                    routing_key=self._get_queue_by_service_type('caption_score'),
                    body=json.dumps(caption_score_message),
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                
                self.logger.debug(f"Published caption scoring trigger for {self.service_name} image {image_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to publish caption score message: {e}")
    
    def connect_to_database(self):
        """Connect to PostgreSQL database"""
        try:
            self.db_conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            self.db_conn.autocommit = True
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
        """Connect to RabbitMQ queue"""
        try:
            credentials = pika.PlainCredentials(self.queue_user, self.queue_password)
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=self.queue_host,
                    credentials=credentials,
                    heartbeat=60,
                    blocked_connection_timeout=300,
                    connection_attempts=10,
                    retry_delay=5,
                    socket_timeout=10
                )
            )
            self.channel = self.connection.channel()
            
            # Declare queues with DLQ (TTL optional/opt-in)
            def declare_with_dlq(channel, queue_name):
                dlq_name = f"{queue_name}.dlq"
                # Declare DLQ first
                channel.queue_declare(queue=dlq_name, durable=True)
                # Declare main queue with dead letter exchange routing to DLQ
                args = {
                    'x-dead-letter-exchange': '',
                    'x-dead-letter-routing-key': dlq_name
                }
                ttl_env = os.getenv('QUEUE_MESSAGE_TTL_MS')
                if ttl_env and ttl_env.isdigit() and int(ttl_env) > 0:
                    args['x-message-ttl'] = int(ttl_env)
                channel.queue_declare(queue=queue_name, durable=True, arguments=args)
            
            declare_with_dlq(self.channel, self.queue_name)
            if self.enable_triggers:
                declare_with_dlq(self.channel, self._get_queue_by_service_type('harmonization'))
                declare_with_dlq(self.channel, self._get_queue_by_service_type('consensus'))
            
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
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
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
                INSERT INTO results (image_id, service, data, status, result_created, worker_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (image_id, self._get_clean_service_name(), json.dumps(result), 'success', datetime.now(), self.worker_id))
            self.db_conn.commit()  # CRITICAL: Commit the transaction!
            cursor.close()

            # Trigger post-processing for bbox services
            if self.service_name in self.bbox_services and self.enable_triggers:
                # Check if connection is healthy
                if not self.channel or self.connection.is_closed:
                    self.logger.warning("RabbitMQ connection lost, reconnecting...")
                    if not self.connect_to_queue():
                        self.logger.error("Failed to reconnect to RabbitMQ for consensus message")
                        raise Exception("RabbitMQ reconnection failed")

                bbox_message = {
                    'image_id': image_id,
                    'image_filename': message.get('image_filename', f'image_{image_id}'),
                    'image_data': message['image_data'],  # Pass through base64 image data
                    'trace_id': trace_id,
                    'service': self.service_name,
                    'worker_id': self.worker_id,
                    'processed_at': datetime.now().isoformat()
                }

                self.channel.basic_publish(
                    exchange='',
                    routing_key=self._get_queue_by_service_type('harmonization'),
                    body=json.dumps(bbox_message),
                    properties=pika.BasicProperties(delivery_mode=2)
                )

                harmonization_queue = self._get_queue_by_service_type('harmonization')
                self.logger.debug(f"Published bbox completion to {harmonization_queue}")

            # Trigger caption scoring
            if self.enable_caption_scoring:
                self.trigger_caption_scoring(image_id, message)

            # Trigger consensus processing
            self.trigger_consensus(image_id, message)

            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            self.job_completed_successfully()

            self.logger.info(f"Successfully processed {self.service_name} request for image {image_id}")

        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            # Database connection errors - don't requeue to prevent CPU spin
            self.logger.error(f"Database error processing {self.service_name} message: {e}")
            self.logger.warning("Rejecting message without requeue due to database error")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            self.job_failed(str(e))
        except Exception as e:
            # Other errors (ML service, parsing, etc.) - requeue for retry
            self.logger.error(f"Error processing {self.service_name} message: {e}")
            # Reject and requeue for transient errors
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            self.job_failed(str(e))
    
    def start(self):
        """Start the worker"""
        self.logger.info(f"Starting {self.service_name} worker ({self.worker_id})")
        
        # Connect to services
        if not self.connect_to_database():
            sys.exit(1)
        if not self.connect_to_queue():
            sys.exit(1)
        
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
                try:
                    self.channel.stop_consuming()
                except Exception:
                    pass
                try:
                    self.connection.close()
                except Exception:
                    pass
                break
            except (pika.exceptions.AMQPConnectionError, pika.exceptions.StreamLostError, pika.exceptions.ChannelClosedByBroker) as e:
                self.logger.warning(f"Queue connection lost/recovered needed: {e}. Reconnecting...")
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