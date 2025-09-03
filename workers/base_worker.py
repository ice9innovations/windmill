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
import pika
import psycopg2
import mysql.connector
import requests
from datetime import datetime
from dotenv import load_dotenv

class BaseWorker:
    """Base class for all ML service workers"""
    
    def __init__(self, service_name, env_file='.env'):
        self.service_name = service_name
        self.load_config(env_file)
        self.setup_logging()
        self.db_conn = None
        self.monitoring_conn = None
        self.channel = None
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.start_time = time.time()
        
    def load_config(self, env_file):
        """Load configuration from .env and service_config.json"""
        # Load .env file
        if not load_dotenv(env_file):
            raise ValueError(f"Could not load {env_file} file. Copy .env.example to .env and configure.")
        
        # Load service definitions
        with open('service_config.json', 'r') as f:
            self.service_definitions = json.load(f)['services']
        
        # Validate service name
        if self.service_name not in self.service_definitions:
            available = ', '.join(self.service_definitions.keys())
            raise ValueError(f"Unknown service '{self.service_name}'. Available: {available}")
        
        # Service configuration
        service_def = self.service_definitions[self.service_name]
        self.service_host = service_def['host']
        self.service_port = service_def['port']
        self.service_endpoint = service_def['endpoint']
        
        # Queue configuration
        self.queue_name = f"queue_{self.service_name}"
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
        self.worker_prefetch_count = int(os.getenv('WORKER_PREFETCH_COUNT', '1'))
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '30'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.retry_delay = int(os.getenv('RETRY_DELAY', '5'))
        
        # Post-processing triggers - dynamically load spatial services from config
        self.bbox_services = [
            service_name for service_name, config in self.service_definitions.items()
            if 'spatial' in config.get('service_type', '').split(',')
        ]
        self.enable_triggers = service_def.get('enable_triggers', False)
        self.enable_consensus_triggers = service_def.get('enable_consensus_triggers', False)
        self.enable_caption_scoring = service_def.get('enable_caption_scoring', False)
        
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

    def _get_required(self, key):
        """Get required environment variable or raise error"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value
    
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
    
    def get_service_url(self, image_url):
        """Build the complete service URL for processing"""
        return f"http://{self.service_host}:{self.service_port}{self.service_endpoint}?url={image_url}"
    
    def trigger_consensus(self, image_id):
        """Trigger consensus processing"""
        if self.enable_consensus_triggers:
            try:
                consensus_message = {
                    'image_id': image_id,
                    'image_filename': f'image_{image_id}',
                    'service': self.service_name,
                    'worker_id': self.worker_id,
                    'processed_at': datetime.now().isoformat()
                }
                
                self.channel.basic_publish(
                    exchange='',
                    routing_key='queue_consensus',
                    body=json.dumps(consensus_message),
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                
                self.logger.debug(f"Published consensus trigger for {self.service_name} image {image_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to publish consensus message: {e}")
    
    def trigger_caption_scoring(self, image_id):
        """Trigger caption scoring for caption generation services"""
        if self.enable_triggers and self.enable_caption_scoring:
            try:
                caption_score_message = {
                    'image_id': image_id,
                    'image_filename': f'image_{image_id}',
                    'service': self.service_name,
                    'worker_id': self.worker_id,
                    'processed_at': datetime.now().isoformat()
                }
                
                self.channel.basic_publish(
                    exchange='',
                    routing_key='queue_caption_score',
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
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def connect_to_queue(self):
        """Connect to RabbitMQ queue"""
        try:
            credentials = pika.PlainCredentials(self.queue_user, self.queue_password)
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=self.queue_host, credentials=credentials)
            )
            self.channel = self.connection.channel()
            
            # Declare queues
            self.channel.queue_declare(queue=self.queue_name, durable=True)
            if self.enable_triggers:
                self.channel.queue_declare(queue='queue_bbox_merge', durable=True)
                self.channel.queue_declare(queue='queue_consensus', durable=True)
            
            self.channel.basic_qos(prefetch_count=self.worker_prefetch_count)
            self.logger.info(f"Connected to RabbitMQ at {self.queue_host}")
            self.logger.info(f"Consuming from: {self.queue_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to queue: {e}")
            return False
    
    def process_message(self, ch, method, properties, body):
        """Process a queue message - override in subclasses"""
        raise NotImplementedError("Subclasses must implement process_message")
    
    def start(self):
        """Start the worker"""
        self.logger.info(f"Starting {self.service_name} worker ({self.worker_id})")
        
        # Connect to services
        if not self.connect_to_database():
            sys.exit(1)
        if not self.connect_to_queue():
            sys.exit(1)
        
        # Start consuming
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.process_message
        )
        
        self.logger.info("Waiting for messages. Press CTRL+C to exit")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.logger.info("Stopping worker...")
            self.channel.stop_consuming()
            self.connection.close()