#!/usr/bin/env python3
"""
Generic Queue Worker - Configurable worker for any ML service
Loads configuration from .env file to determine which service to run
"""
import os
import sys
import json
import time
import logging
import socket
import pika
import psycopg2
import mysql.connector
import requests
from datetime import datetime
from dotenv import load_dotenv

class WorkerConfig:
    """Load and validate worker configuration"""
    
    def __init__(self, env_file='.env'):
        # Load .env file
        if not load_dotenv(env_file):
            raise ValueError(f"Could not load {env_file} file. Copy .env.example to .env and configure.")
        
        # Load service definitions
        with open('service_config.json', 'r') as f:
            self.service_definitions = json.load(f)['services']
        
        # Required configuration
        self.service_name = self._get_required('SERVICE_NAME')
        self.validate_service_name()
        
        # Service configuration (from service_config.json)
        service_def = self.service_definitions[self.service_name]
        self.service_host = service_def['host']
        self.service_port = service_def['port']
        self.service_endpoint = service_def['endpoint']
        
        # Queue configuration
        self.queue_name = os.getenv('QUEUE_NAME', f"queue_{self.service_name}")
        self.queue_host = self._get_required('QUEUE_HOST')
        self.queue_user = self._get_required('QUEUE_USER')
        self.queue_password = self._get_required('QUEUE_PASSWORD')
        
        # Database configuration
        self.db_host = self._get_required('DB_HOST')
        self.db_name = self._get_required('DB_NAME')
        self.db_user = self._get_required('DB_USER')
        self.db_password = self._get_required('DB_PASSWORD')
        
        # Worker configuration
        self.worker_id = os.getenv('WORKER_ID', f"worker_{self.service_name}_{int(time.time())}")
        self.worker_prefetch_count = int(os.getenv('WORKER_PREFETCH_COUNT', '1'))
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '30'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.retry_delay = int(os.getenv('RETRY_DELAY', '5'))
        
        # Post-processing triggers
        self.bbox_services = ['yolov8', 'rtdetr', 'detectron2']
        self.enable_triggers = service_def.get('enable_triggers', False)
        
        # Logging configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        self.log_format = os.getenv('LOG_FORMAT', 'detailed')
        
        # Performance configuration
        self.processing_delay = float(os.getenv('PROCESSING_DELAY', '0.0'))
        self.heartbeat_interval = int(os.getenv('HEARTBEAT_INTERVAL', '60'))
        
        # Monitoring configuration  
        self.enable_monitoring = os.getenv('ENABLE_MONITORING', 'false').lower() == 'true'
        if self.enable_monitoring:
            self.monitoring_db_host = self._get_required('MONITORING_DB_HOST')
            self.monitoring_db_user = self._get_required('MONITORING_DB_USER') 
            self.monitoring_db_password = self._get_required('MONITORING_DB_PASSWORD')
            self.monitoring_db_name = self._get_required('MONITORING_DB_NAME')
        else:
            self.monitoring_db_host = None
            self.monitoring_db_user = None
            self.monitoring_db_password = None
            self.monitoring_db_name = None
    
    def _get_required(self, key):
        """Get required environment variable or raise error"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value
    
    def validate_service_name(self):
        """Validate service name exists in service definitions"""
        if self.service_name not in self.service_definitions:
            available = ', '.join(self.service_definitions.keys())
            raise ValueError(f"Unknown service '{self.service_name}'. Available: {available}")
    
    def get_service_url(self, image_url):
        """Build the complete service URL for processing"""
        return f"http://{self.service_host}:{self.service_port}{self.service_endpoint}?url={image_url}"

class WorkerMonitoring:
    """Handles worker monitoring and heartbeats"""
    
    def __init__(self, config):
        self.config = config
        self.mysql_conn = None
        self.last_heartbeat = 0
        self.jobs_completed = 0
        self.start_time = time.time()
        self.hostname = socket.gethostname()
        
        if config.enable_monitoring:
            self.connect_to_monitoring_db()
    
    def connect_to_monitoring_db(self):
        """Connect to MySQL monitoring database"""
        try:
            self.mysql_conn = mysql.connector.connect(
                host=self.config.monitoring_db_host,
                user=self.config.monitoring_db_user,
                password=self.config.monitoring_db_password,
                database=self.config.monitoring_db_name,
                autocommit=True
            )
            self.send_heartbeat('starting')
            return True
        except Exception as e:
            print(f"Warning: Could not connect to monitoring database: {e}")
            return False
    
    def maybe_send_heartbeat(self):
        """Send heartbeat if enough time has passed (every 2 minutes)"""
        if not self.config.enable_monitoring or not self.mysql_conn:
            return
        
        if time.time() - self.last_heartbeat > 120:  # 2 minutes
            self.send_heartbeat('alive')
    
    def send_heartbeat(self, status, error_msg=None):
        """Send heartbeat to monitoring database"""
        if not self.config.enable_monitoring or not self.mysql_conn:
            return
        
        try:
            # Calculate processing rate
            runtime_minutes = max((time.time() - self.start_time) / 60, 0.1)  # Avoid division by zero
            jobs_per_minute = self.jobs_completed / runtime_minutes
            
            cursor = self.mysql_conn.cursor()
            cursor.execute("""
                INSERT INTO worker_heartbeats 
                (worker_id, service_name, node_hostname, status, jobs_completed, 
                 jobs_per_minute, last_job_time, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                self.config.worker_id,
                self.config.service_name,
                self.hostname,
                status,
                self.jobs_completed,
                round(jobs_per_minute, 2),
                datetime.now() if self.jobs_completed > 0 else None,
                error_msg
            ))
            
            self.last_heartbeat = time.time()
            
        except Exception as e:
            print(f"Warning: Failed to send heartbeat: {e}")
    
    def increment_job_count(self):
        """Increment completed job count"""
        self.jobs_completed += 1
    
    def worker_stopping(self):
        """Called when worker is shutting down"""
        if self.config.enable_monitoring and self.mysql_conn:
            self.send_heartbeat('stopping')
            self.mysql_conn.close()

class GenericWorker:
    """Generic ML service worker"""
    
    def __init__(self):
        self.config = WorkerConfig()
        self.setup_logging()
        self.monitoring = WorkerMonitoring(self.config)
        self.connection = None
        self.channel = None
        self.db_conn = None
        self.logger.info(f"Initialized {self.config.service_name.upper()} worker ({self.config.worker_id})")
    
    def setup_logging(self):
        """Configure logging based on config"""
        if self.config.log_format == 'detailed':
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            log_format = '%(levelname)s: %(message)s'
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=log_format
        )
        self.logger = logging.getLogger(f'worker.{self.config.service_name}')
    
    def connect_to_rabbitmq(self):
        """Connect to RabbitMQ with retry logic"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                credentials = pika.PlainCredentials(self.config.queue_user, self.config.queue_password)
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(
                        host=self.config.queue_host,
                        credentials=credentials,
                        heartbeat=600,  # 10 minutes
                        blocked_connection_timeout=300,  # 5 minutes
                        socket_timeout=30,
                        retry_delay=2,
                        connection_attempts=3
                    )
                )
                self.channel = self.connection.channel()
                
                # Declare the queue (create if it doesn't exist)
                self.channel.queue_declare(queue=self.config.queue_name, durable=True)
                
                # Declare post-processing queues if triggers are enabled
                if self.config.enable_triggers:
                    self.channel.queue_declare(queue='queue_bbox_merge', durable=True)
                    self.channel.queue_declare(queue='queue_spatial_enrichment', durable=True)
                    self.channel.queue_declare(queue='queue_consensus', durable=True)
                
                # Set prefetch count for fair distribution
                self.channel.basic_qos(prefetch_count=self.config.worker_prefetch_count)
                
                self.logger.info(f"Connected to RabbitMQ at {self.config.queue_host}, queue: {self.config.queue_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to connect to RabbitMQ (attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    continue
                return False
    
    def connect_to_database(self):
        """Connect to PostgreSQL database"""
        try:
            # Close existing connection if any
            if hasattr(self, 'db_conn') and self.db_conn:
                try:
                    self.db_conn.close()
                except:
                    pass
            
            self.db_conn = psycopg2.connect(
                host=self.config.db_host,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password
            )
            self.logger.info(f"Connected to PostgreSQL at {self.config.db_host}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def ensure_database_connection(self):
        """Ensure database connection is healthy, reconnect if needed"""
        try:
            if not self.db_conn or self.db_conn.closed:
                self.logger.info("Database connection closed, reconnecting...")
                return self.connect_to_database()
            else:
                # Test connection with simple query
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                return True
        except Exception as e:
            self.logger.warning(f"Database connection unhealthy: {e}")
            self.logger.info("Reconnecting to database...")
            return self.connect_to_database()
    
    def process_image(self, image_data):
        """Process image through the configured ML service"""
        self.logger.info(f"Processing: {image_data['image_filename']}")
        
        service_url = self.config.get_service_url(image_data['image_url'])
        self.logger.debug(f"Calling: {service_url}")
        
        start_time = time.time()
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.get(service_url, timeout=self.config.request_timeout)
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    # Add queue processing metadata
                    if 'metadata' in result:
                        result['metadata']['queue_processing_time'] = processing_time
                        result['metadata']['worker_id'] = self.config.worker_id
                        result['metadata']['attempt'] = attempt + 1
                    
                    self.logger.info(f"Completed {image_data['image_filename']} in {processing_time:.2f}s")
                    return result
                    
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    self.logger.warning(f"Service error (attempt {attempt + 1}): {error_msg}")
                    
                    if attempt == self.config.max_retries - 1:
                        return {
                            "service": self.config.service_name,
                            "status": "error",
                            "predictions": [],
                            "error": {"message": error_msg},
                            "metadata": {
                                "processing_time": processing_time,
                                "worker_id": self.config.worker_id,
                                "attempts": attempt + 1
                            }
                        }
                    
                    time.sleep(self.config.retry_delay)
                    
            except requests.exceptions.RequestException as e:
                processing_time = time.time() - start_time
                error_msg = f"Network error: {str(e)}"
                self.logger.warning(f"Network error (attempt {attempt + 1}): {error_msg}")
                
                if attempt == self.config.max_retries - 1:
                    return {
                        "service": self.config.service_name,
                        "status": "error",
                        "predictions": [],
                        "error": {"message": error_msg},
                        "metadata": {
                            "processing_time": processing_time,
                            "worker_id": self.config.worker_id,
                            "attempts": attempt + 1
                        }
                    }
                
                time.sleep(self.config.retry_delay)
    
    def save_result_to_database(self, image_data, result):
        """Save processing result to database with retry logic"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # Ensure healthy database connection
                if not self.ensure_database_connection():
                    self.logger.error("Could not establish database connection")
                    if attempt < max_retries - 1:
                        self.logger.info(f"Retrying database save (attempt {attempt + 1}/{max_retries})")
                        time.sleep(2)
                        continue
                    raise Exception("Failed to establish database connection")
                
                cursor = self.db_conn.cursor()
                
                cursor.execute("""
                    INSERT INTO results (image_id, service, data, status, processing_time, result_created, worker_id)
                    VALUES (%s, %s, %s, %s, %s, NOW(), %s)
                """, (
                    image_data['image_id'],
                    self.config.service_name,
                    json.dumps(result),
                    result['status'],
                    result['metadata'].get('processing_time', 0),
                    self.config.worker_id
                ))
                
                self.db_conn.commit()
                cursor.close()
                self.logger.debug(f"Saved result for {image_data['image_filename']}")
                
                if attempt > 0:
                    self.logger.info(f"Successfully saved result after {attempt + 1} attempts")
                
                return  # Success, exit retry loop
                
            except Exception as e:
                self.logger.error(f"Database error (attempt {attempt + 1}/{max_retries}): {e}")
                
                # Rollback transaction on error
                try:
                    if self.db_conn:
                        self.db_conn.rollback()
                except:
                    pass
                
                # If not the last attempt, wait and retry
                if attempt < max_retries - 1:
                    self.logger.info(f"Will retry database save after 2 seconds...")
                    time.sleep(2)
                    continue
                    
                # Last attempt failed, re-raise the exception
                raise
    
    def trigger_post_processing(self, image_data, result):
        """Trigger post-processing via queue messages (proper architecture)"""
        if not self.config.enable_triggers:
            return
            
        # Publish to bbox merge queue if this is a bbox service
        if self.config.service_name in self.config.bbox_services:
            try:
                bbox_message = {
                    'image_id': image_data['image_id'],
                    'image_filename': image_data.get('image_filename', f'image_{image_data["image_id"]}'),
                    'service': self.config.service_name,
                    'worker_id': self.config.worker_id,
                    'processed_at': datetime.now().isoformat()
                }
                
                # Publish to bbox merge queue
                self.channel.basic_publish(
                    exchange='',
                    routing_key='queue_bbox_merge',
                    body=json.dumps(bbox_message),
                    properties=pika.BasicProperties(
                        delivery_mode=2  # Make message persistent
                    )
                )
                
                self.logger.debug(f"Published bbox completion to queue_bbox_merge: {bbox_message}")
                
            except Exception as e:
                self.logger.error(f"Failed to publish bbox completion message: {e}")
        
        # Publish to consensus queue if this is a non-bbox primary service (colors, blip, clip, etc.)
        elif self.config.service_name not in self.config.bbox_services:
            try:
                consensus_message = {
                    'image_id': image_data['image_id'],
                    'image_filename': image_data.get('image_filename', f'image_{image_data["image_id"]}'),
                    'service': self.config.service_name,
                    'worker_id': self.config.worker_id,
                    'processed_at': datetime.now().isoformat()
                }
                
                # Publish to consensus queue
                self.channel.basic_publish(
                    exchange='',
                    routing_key='queue_consensus',
                    body=json.dumps(consensus_message),
                    properties=pika.BasicProperties(
                        delivery_mode=2  # Make message persistent
                    )
                )
                
                self.logger.debug(f"Published {self.config.service_name} completion to queue_consensus: {consensus_message}")
                
            except Exception as e:
                self.logger.error(f"Failed to publish consensus completion message: {e}")
        
        # Publish to caption score queue if this is BLIP or LLaMa (caption generation services)
        if self.config.service_name in ['blip', 'ollama']:
            try:
                caption_score_message = {
                    'image_id': image_data['image_id'],
                    'image_filename': image_data.get('image_filename', f'image_{image_data["image_id"]}'),
                    'service': self.config.service_name,
                    'worker_id': self.config.worker_id,
                    'processed_at': datetime.now().isoformat()
                }
                
                # Publish to caption score queue
                self.channel.basic_publish(
                    exchange='',
                    routing_key='queue_caption_score',
                    body=json.dumps(caption_score_message),
                    properties=pika.BasicProperties(
                        delivery_mode=2  # Make message persistent
                    )
                )
                
                self.logger.debug(f"Published {self.config.service_name} completion to queue_caption_score: {caption_score_message}")
                
            except Exception as e:
                self.logger.error(f"Failed to publish caption score message: {e}")
    
    def process_message(self, ch, method, properties, body):
        """Process a single queue message"""
        try:
            # Parse job data
            image_data = json.loads(body.decode('utf-8'))
            self.logger.debug(f"Received job: {image_data['image_filename']}")
            
            # Add processing delay if configured (for testing/throttling)
            if self.config.processing_delay > 0:
                time.sleep(self.config.processing_delay)
            
            # Process the image
            result = self.process_image(image_data)
            
            # Save to database
            self.save_result_to_database(image_data, result)
            
            # Trigger post-processing if enabled
            self.trigger_post_processing(image_data, result)
            
            # Update monitoring
            self.monitoring.increment_job_count()
            self.monitoring.maybe_send_heartbeat()
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            self.logger.info(f"Completed job: {image_data['image_filename']}")
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            # Send error status to monitoring
            self.monitoring.send_heartbeat('error', str(e))
            # Reject and requeue for retry
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    def start_consuming(self):
        """Start consuming messages from the queue with reconnection logic"""
        self.logger.info(f"Starting {self.config.service_name.upper()} worker")
        self.logger.info(f"Queue: {self.config.queue_name}")
        self.logger.info(f"Service URL: http://{self.config.service_host}:{self.config.service_port}{self.config.service_endpoint}")
        
        while True:
            try:
                # Setup message consumer
                self.channel.basic_consume(
                    queue=self.config.queue_name,
                    on_message_callback=self.process_message
                )
                
                self.logger.info("Waiting for messages. Press CTRL+C to exit")
                self.channel.start_consuming()
                
            except KeyboardInterrupt:
                self.logger.info("Stopping worker...")
                if self.channel:
                    self.channel.stop_consuming()
                break
                
            except Exception as e:
                self.logger.error(f"Connection lost during consuming: {e}")
                self.logger.info("Attempting to reconnect...")
                
                # Close existing connections
                try:
                    if self.connection and not self.connection.is_closed:
                        self.connection.close()
                except:
                    pass
                
                # Try to reconnect
                if self.connect_to_rabbitmq():
                    self.logger.info("Reconnected successfully, resuming...")
                    continue
                else:
                    self.logger.error("Failed to reconnect, stopping worker")
                    break
        
        # Clean shutdown
        self.monitoring.worker_stopping()
        
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        if self.db_conn:
            self.db_conn.close()
        self.logger.info("Worker stopped")
    
    def run(self):
        """Main worker entry point"""
        # Connect to infrastructure
        if not self.connect_to_rabbitmq():
            return 1
        
        if not self.connect_to_database():
            return 1
        
        # Start processing
        self.start_consuming()
        return 0

def main():
    """Main entry point"""
    try:
        worker = GenericWorker()
        return worker.run()
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Worker error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())