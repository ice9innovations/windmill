#!/usr/bin/env python3
"""
Caption Score Worker - Scores captions against images using CLIP
Triggered by BLIP and LLaMa services to evaluate caption quality
"""
import os
import json
import time
import logging
import socket
import psycopg2
import pika
import requests
from datetime import datetime
from dotenv import load_dotenv

class CaptionScoreWorker:
    """Worker that scores captions against images using CLIP similarity"""
    
    def __init__(self):
        # Load configuration
        if not load_dotenv():
            raise ValueError("Could not load .env file")
        
        # Load service definitions
        with open('service_config.json', 'r') as f:
            self.service_definitions = json.load(f)['services']
        
        # Database configuration
        self.db_host = self._get_required('DB_HOST')
        self.db_name = self._get_required('DB_NAME')
        self.db_user = self._get_required('DB_USER')
        self.db_password = self._get_required('DB_PASSWORD')
        
        # CLIP score service configuration
        clip_config = self.service_definitions.get('caption_score', {})
        self.clip_score_url = f"http://{clip_config.get('host', 'localhost')}:{clip_config.get('port', 7778)}{clip_config.get('endpoint', '/score')}"
        
        # Worker configuration
        self.worker_id = os.getenv('WORKER_ID', f'caption_score_worker_{int(time.time())}')
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '30'))
        
        # Queue configuration
        self.queue_name = os.getenv('CAPTION_SCORE_QUEUE_NAME', 'queue_caption_score')
        self.queue_host = self._get_required('QUEUE_HOST')
        self.queue_user = self._get_required('QUEUE_USER')
        self.queue_password = self._get_required('QUEUE_PASSWORD')
        
        # Services that trigger caption scoring
        self.trigger_services = ['blip', 'ollama']  # ollama = llama in service config
        
        # Logging
        self.setup_logging()
        self.db_conn = None
        self.read_db_conn = None
        
    def setup_logging(self):
        """Configure logging"""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('caption_score_worker')
    
    def _get_required(self, key):
        """Get required environment variable with no fallback"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value
    
    def connect_to_database(self):
        """Connect to PostgreSQL database"""
        try:
            # Close existing connections if any
            if hasattr(self, 'db_conn') and self.db_conn:
                try:
                    self.db_conn.close()
                except:
                    pass
            if hasattr(self, 'read_db_conn') and self.read_db_conn:
                try:
                    self.read_db_conn.close()
                except:
                    pass
            
            # Main connection for transactions (write operations)
            self.db_conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            self.db_conn.autocommit = False
            
            # Read-only connection for queries
            self.read_db_conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            self.read_db_conn.autocommit = True
            
            self.logger.info(f"Connected to PostgreSQL at {self.db_host}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def ensure_database_connection(self):
        """Ensure database connections are healthy, reconnect if needed"""
        reconnect_needed = False
        
        # Check main connection
        try:
            if not self.db_conn or self.db_conn.closed:
                reconnect_needed = True
            else:
                # Test connection with simple query
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
        except Exception as e:
            self.logger.warning(f"Main database connection unhealthy: {e}")
            reconnect_needed = True
            
        # Check read connection  
        try:
            if not self.read_db_conn or self.read_db_conn.closed:
                reconnect_needed = True
            else:
                # Test connection with simple query
                cursor = self.read_db_conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
        except Exception as e:
            self.logger.warning(f"Read database connection unhealthy: {e}")
            reconnect_needed = True
            
        if reconnect_needed:
            self.logger.info("Reconnecting to database...")
            return self.connect_to_database()
            
        return True
    
    def connect_to_rabbitmq(self):
        """Connect to RabbitMQ for queue-based processing"""
        try:
            credentials = pika.PlainCredentials(self.queue_user, self.queue_password)
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=self.queue_host,
                    credentials=credentials
                )
            )
            self.channel = self.connection.channel()
            
            # Declare the queue
            self.channel.queue_declare(queue=self.queue_name, durable=True)
            
            # Set prefetch count for fair distribution
            self.channel.basic_qos(prefetch_count=1)
            
            self.logger.info(f"Connected to RabbitMQ at {self.queue_host}, queue: {self.queue_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False
    
    def get_image_captions(self, image_id):
        """Get captions generated by BLIP and LLaMa for this image"""
        try:
            # Ensure healthy database connection
            if not self.ensure_database_connection():
                self.logger.error("Could not establish database connection")
                return []
                
            cursor = self.read_db_conn.cursor()
            
            # Get analysis data from BLIP and LLaMa services  
            query = """
                SELECT service, data 
                FROM results 
                WHERE image_id = %s 
                AND service IN ('blip', 'ollama')
                AND status = 'success'
                AND data IS NOT NULL
            """
            
            cursor.execute(query, (image_id,))
            results = cursor.fetchall()
            cursor.close()
            
            captions = []
            
            for service, data in results:
                try:
                    # data is already JSONB in PostgreSQL, so it should be a dict
                    if isinstance(data, str):
                        data = json.loads(data)
                    
                    if service == 'blip':
                        # Extract BLIP caption
                        predictions = data.get('predictions', [])
                        if predictions:
                            text = predictions[0].get('text', '')
                            if text:
                                captions.append({
                                    'service': 'blip',
                                    'caption': text,
                                    'confidence': 1.0  # BLIP doesn't provide confidence
                                })
                    
                    elif service == 'ollama':
                        # Extract LLaMa caption
                        predictions = data.get('predictions', [])
                        if predictions:
                            text = predictions[0].get('text', '')
                            if text:
                                captions.append({
                                    'service': 'ollama',
                                    'caption': text,
                                    'confidence': 1.0  # LLaMa doesn't provide confidence
                                })
                
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    self.logger.warning(f"Failed to parse {service} data for image {image_id}: {e}")
            
            return captions
            
        except Exception as e:
            self.logger.error(f"Error getting captions for image {image_id}: {e}")
            return []
    
    def score_caption_against_image(self, image_path, caption):
        """Score a caption against an image using CLIP"""
        try:
            # Call CLIP score endpoint
            params = {
                'file': image_path,
                'caption': caption
            }
            
            response = requests.get(
                self.clip_score_url,
                params=params,
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'success':
                    return result.get('similarity_score', 0.0)
                else:
                    self.logger.warning(f"CLIP score API error: {result.get('error', {}).get('message', 'Unknown error')}")
                    return None
            else:
                self.logger.error(f"CLIP score API returned {response.status_code}: {response.text}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error calling CLIP score API: {e}")
            return None
    
    def save_caption_scores(self, image_id, caption_scores):
        """Save caption similarity scores to postprocessing table with retry logic"""
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
                    return False
                
                cursor = self.db_conn.cursor()
                
                # Delete old caption scores for this image (atomic replacement)
                cursor.execute("""
                    DELETE FROM postprocessing 
                    WHERE image_id = %s AND service = 'caption_score'
                """, (image_id,))
                
                # Insert new caption scores
                score_data = {
                    'caption_scores': caption_scores,
                    'total_captions': len(caption_scores),
                    'processing_algorithm': 'clip_similarity_v1',
                    'processed_at': datetime.now().isoformat()
                }
                
                cursor.execute("""
                    INSERT INTO postprocessing (image_id, service, data, status)
                    VALUES (%s, %s, %s, %s)
                """, (
                    image_id,
                    'caption_score',
                    json.dumps(score_data),
                    'success'
                ))
                
                self.db_conn.commit()
                cursor.close()
                
                if attempt > 0:
                    self.logger.info(f"Successfully saved caption scores after {attempt + 1} attempts")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error saving caption scores (attempt {attempt + 1}/{max_retries}): {e}")
                
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
                    
                return False
        
        return False
    
    def process_queue_message(self, ch, method, properties, body):
        """Process a message from the caption score queue"""
        try:
            # Parse the message
            message_data = json.loads(body.decode('utf-8'))
            image_id = message_data['image_id']
            image_filename = message_data.get('image_filename', f'image_{image_id}')
            
            self.logger.info(f"Processing caption scoring for: {image_filename}")
            
            # Get image path
            cursor = self.read_db_conn.cursor()
            cursor.execute("SELECT image_path FROM images WHERE image_id = %s", (image_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if not result:
                self.logger.error(f"Image not found: {image_id}")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            
            image_path = result[0]
            
            # Get captions from BLIP and LLaMa
            captions = self.get_image_captions(image_id)
            
            if not captions:
                self.logger.debug(f"No captions found for {image_filename}")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            
            # Score each caption against the image
            caption_scores = []
            
            for caption_info in captions:
                caption = caption_info['caption']
                service = caption_info['service']
                
                similarity_score = self.score_caption_against_image(image_path, caption)
                
                if similarity_score is not None:
                    caption_scores.append({
                        'service': service,
                        'caption': caption,
                        'original_confidence': caption_info['confidence'],
                        'similarity_score': similarity_score,
                        'scored_at': datetime.now().isoformat()
                    })
                    
                    self.logger.info(f"{service} caption score: {similarity_score:.3f} for '{caption[:50]}...'")
            
            # Save results
            if caption_scores:
                if self.save_caption_scores(image_id, caption_scores):
                    self.logger.info(f"Saved {len(caption_scores)} caption scores for {image_filename}")
                else:
                    self.logger.error(f"Failed to save caption scores for {image_filename}")
            
            # Acknowledge the message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            self.logger.error(f"Error processing queue message: {e}")
            # Reject and requeue for retry
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    def run(self):
        """Main entry point - queue-based processing"""
        if not self.connect_to_database():
            return 1
        
        if not self.connect_to_rabbitmq():
            return 1
        
        self.logger.info(f"Starting caption score worker ({self.worker_id})")
        self.logger.info(f"Listening on queue: {self.queue_name}")
        self.logger.info(f"CLIP score endpoint: {self.clip_score_url}")
        self.logger.info(f"Trigger services: {', '.join(self.trigger_services)}")
        
        # Setup message consumer
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.process_queue_message
        )
        
        self.logger.info("Waiting for caption scoring messages. Press CTRL+C to exit")
        
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.logger.info("Stopping caption score worker...")
            self.channel.stop_consuming()
        finally:
            if self.connection and not self.connection.is_closed:
                self.connection.close()
            if self.db_conn:
                self.db_conn.close()
            if self.read_db_conn:
                self.read_db_conn.close()
            self.logger.info("Caption score worker stopped")
        
        return 0

def main():
    """Main entry point"""
    try:
        worker = CaptionScoreWorker()
        return worker.run()
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Caption score worker error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())