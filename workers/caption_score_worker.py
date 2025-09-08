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
from base_worker import BaseWorker

class CaptionScoreWorker(BaseWorker):
    """Worker that scores captions against images using CLIP similarity"""
    
    def __init__(self):
        # Initialize with caption score service type
        super().__init__('postprocessing.caption_score')
        
        # CLIP score service configuration - find by service_type
        caption_score_services = self.config.get_service_group('postprocessing.caption_score[]')
        service_config = self.config.get_service_config(caption_score_services[0])
        self.clip_score_url = f"http://{service_config['host']}:{service_config['port']}{service_config['endpoint']}"
        
        # Services that trigger caption scoring
        self.trigger_services = self.config.get_service_group('primary.semantic[]')
        
        # Caption score worker needs separate read connection for queries
        self.read_db_conn = None
        
    
    def connect_to_database(self):
        """Connect to PostgreSQL database with dual connections"""
        # Call parent to set up main connection
        if not super().connect_to_database():
            return False
            
        try:
            # Set up transaction mode for main connection
            self.db_conn.autocommit = False
            
            # Create separate read connection for queries
            self.read_db_conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            self.read_db_conn.autocommit = True  # Auto-commit for read queries
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set up dual database connections: {e}")
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
    
    
    def get_service_caption(self, image_id, service_name):
        """Get caption from a specific service for this image"""
        try:
            # Ensure healthy database connection
            if not self.ensure_database_connection():
                self.logger.error("Could not establish database connection")
                return None
            
            cursor = self.read_db_conn.cursor()
            
            # Get result from the specific service
            cursor.execute("""
                SELECT data
                FROM results 
                WHERE image_id = %s 
                AND service = %s
                AND status = 'success'
                ORDER BY result_created DESC
                LIMIT 1
            """, (image_id, service_name))
            
            row = cursor.fetchone()
            cursor.close()
            
            if not row:
                return None
                
            try:
                # Parse JSON data if it's a string
                if isinstance(row[0], str):
                    result_data = json.loads(row[0])
                else:
                    result_data = row[0]
                
                # Extract predictions
                predictions = result_data.get('predictions', [])
                if predictions:
                    # For captioning services, extract the caption text
                    # Services store captions as either 'caption' or 'text' fields
                    caption_text = predictions[0].get('caption', predictions[0].get('text', '')).strip()
                    if caption_text:
                        return {
                            'service': service_name,
                            'caption': caption_text,
                            'original_confidence': predictions[0].get('confidence', 1.0)
                        }
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                self.logger.warning(f"Failed to parse {service_name} data for image {image_id}: {e}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting caption from {service_name} for image {image_id}: {e}")
            return None

    def score_caption_against_image(self, image_data, caption):
        """Score a caption against an image using CLIP"""
        try:
            import base64
            import io
            
            # Decode base64 image data to bytes
            image_bytes = base64.b64decode(image_data)
            
            # Call CLIP score endpoint using multipart file upload + form data
            files = {'file': ('image.jpg', io.BytesIO(image_bytes), 'image/jpeg')}
            data = {'caption': caption}
            
            response = requests.post(
                self.clip_score_url,
                files=files,
                data=data,
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
    

    def save_individual_caption_score(self, image_id, caption_score):
        """Save individual caption score to database"""
        max_retries = 3
        
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
                
                # Insert caption score for this specific service
                score_data = {
                    'caption_score': caption_score,
                    'processing_algorithm': 'clip_similarity_v1',
                    'processed_at': datetime.now().isoformat()
                }
                
                cursor.execute("""
                    INSERT INTO postprocessing (image_id, service, data, status)
                    VALUES (%s, %s, %s, %s)
                """, (
                    image_id,
                    f"caption_score_{caption_score['service']}",  # e.g. "caption_score_blip"
                    json.dumps(score_data),
                    'success'
                ))
                
                self.db_conn.commit()
                cursor.close()
                
                if attempt > 0:
                    self.logger.info(f"Successfully saved caption score after {attempt + 1} attempts")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error saving caption score (attempt {attempt + 1}/{max_retries}): {e}")
                
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
    
    def process_queue_message(self, ch, method, properties, body):
        """Process a message from the caption score queue"""
        try:
            # Parse the message
            message_data = json.loads(body.decode('utf-8'))
            image_id = message_data['image_id']
            service_name = message_data['service']
            image_filename = message_data.get('image_filename', f'image_{image_id}')
            image_data = message_data.get('image_data')  # Base64 encoded image data
            
            self.logger.info(f"Processing caption scoring for {service_name}: {image_filename}")
            
            # Validate image data is present
            if not image_data:
                self.logger.error(f"No image_data in message for image {image_id}")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            
            # Get caption from the specific service that triggered this
            caption_data = self.get_service_caption(image_id, service_name)
            
            if not caption_data:
                self.logger.debug(f"No caption found from {service_name} for {image_filename}")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
            
            # Score the individual caption against the image
            caption = caption_data['caption']
            service = caption_data['service']
            
            similarity_score = self.score_caption_against_image(image_data, caption)
            
            if similarity_score is not None:
                caption_score = {
                    'service': service,
                    'caption': caption,
                    'original_confidence': caption_data['original_confidence'],
                    'similarity_score': similarity_score,
                    'scored_at': datetime.now().isoformat()
                }
                
                self.logger.info(f"{service} caption score: {similarity_score:.3f} for '{caption[:50]}...'")
                
                # Save individual caption score
                if self.save_individual_caption_score(image_id, caption_score):
                    self.logger.info(f"Saved caption score for {service}: {image_filename}")
                else:
                    self.logger.error(f"Failed to save caption score for {service}: {image_filename}")
            else:
                self.logger.warning(f"Failed to score caption from {service} for {image_filename}")
            
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
        
        if not self.connect_to_queue():
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