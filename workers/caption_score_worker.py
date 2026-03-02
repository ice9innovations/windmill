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
        
        # CLIP score service configuration - image-text similarity at clip_score endpoint
        clip_score_services = self.config.get_service_group('postprocessing.clip_score[]')
        service_config = self.config.get_service_config(clip_score_services[0])
        self.clip_score_url = f"http://{service_config['host']}:{service_config['port']}{service_config['endpoint']}"
        
        # Services that trigger caption scoring - use actual service names from database
        semantic_services = self.config.get_service_group('primary.semantic[]')
        # Extract just the service names (blip, ollama) from hierarchical names (primary.blip, primary.ollama)
        self.trigger_services = [service.split('.')[-1] for service in semantic_services]
        
        # Caption score worker needs separate read connection for queries
        self.read_db_conn = None
        
    
    def connect_to_database(self):
        """Connect to PostgreSQL database with dual connections"""
        if hasattr(self, 'read_db_conn') and self.read_db_conn:
            try:
                self.read_db_conn.close()
            except Exception:
                pass

        # Call parent to set up main connection
        if not super().connect_to_database():
            return False

        try:
            # Set up transaction mode for main connection
            self.db_conn.autocommit = False

            # Create separate read connection for queries
            self.read_db_conn = self._new_db_connection(autocommit=True)
            
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
                        }
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                self.logger.warning(f"Failed to parse {service_name} data for image {image_id}: {e}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting caption from {service_name} for image {image_id}: {e}")
            return None

    def score_caption_against_image(self, image_data, caption):
        """Score a caption against an image using CLIP.

        Returns a dict with:
            similarity_score  - cosine similarity float
            image_embedding   - 768-dim normalized float list (ViT-L/14)
            caption_embedding - 768-dim normalized float list for the caption text
        Returns None on failure.
        """
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
                    return {
                        'similarity_score': result.get('similarity_score', 0.0),
                        'image_embedding': result.get('image_embedding'),
                        'caption_embedding': result.get('text_embedding'),
                    }
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
        """Save individual caption score and caption embedding to postprocessing table"""
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

                # Store caption embedding alongside the score so it travels with the caption
                score_data = {
                    'caption_score': caption_score,
                    'processing_algorithm': 'clip_similarity_v1',
                    'processed_at': datetime.now().isoformat()
                }
                if caption_score.get('caption_embedding') is not None:
                    score_data['caption_embedding'] = caption_score['caption_embedding']

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

                try:
                    if self.db_conn:
                        self.db_conn.rollback()
                except:
                    pass

                if attempt < max_retries - 1:
                    self.logger.info(f"Will retry database save after 2 seconds...")
                    time.sleep(2)
                    continue

        return False

    def save_image_embedding(self, image_id, image_embedding):
        """Write the CLIP image embedding to the images table, once per image.

        Uses an idempotent UPDATE that only fires when image_clip_embedding is NULL,
        so concurrent caption score workers for the same image are safe.
        The embedding list is serialized to pgvector's text format and cast in SQL.
        """
        max_retries = 3

        for attempt in range(max_retries):
            try:
                if not self.ensure_database_connection():
                    self.logger.error("Could not establish database connection for image embedding save")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return False

                cursor = self.db_conn.cursor()

                # json.dumps produces '[1.0, 2.0, ...]' — valid pgvector text input
                embedding_str = json.dumps(image_embedding)

                cursor.execute("""
                    UPDATE images
                    SET image_clip_embedding = %s::vector
                    WHERE image_id = %s AND image_clip_embedding IS NULL
                """, (embedding_str, image_id))

                self.db_conn.commit()
                cursor.close()

                return True

            except Exception as e:
                self.logger.error(f"Error saving image embedding (attempt {attempt + 1}/{max_retries}): {e}")

                try:
                    if self.db_conn:
                        self.db_conn.rollback()
                except:
                    pass

                if attempt < max_retries - 1:
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
                self._safe_ack(ch, method.delivery_tag)
                return
            
            # Get caption from the specific service that triggered this - extract just service name from hierarchical name
            actual_service_name = service_name.split('.')[-1]  # primary.blip -> blip
            caption_data = self.get_service_caption(image_id, actual_service_name)
            
            if not caption_data:
                self.logger.debug(f"No caption found from {service_name} for {image_filename}")
                self._safe_ack(ch, method.delivery_tag)
                return
            
            # Score the individual caption against the image
            caption = caption_data['caption']
            service = caption_data['service']

            clip_result = self.score_caption_against_image(image_data, caption)

            if clip_result is None:
                self.logger.error(f"CLIP service failed to score caption from {service} for {image_filename}")
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                return

            similarity_score = clip_result['similarity_score']
            self.logger.info(f"{service} caption score: {similarity_score:.3f} for '{caption[:50]}...'")

            caption_score = {
                'service': service,
                'caption': caption,
                'similarity_score': similarity_score,
                'caption_embedding': clip_result.get('caption_embedding'),
                'scored_at': datetime.now().isoformat()
            }

            # Save caption score (includes caption embedding) — only ack if this succeeds
            if not self.save_individual_caption_score(image_id, caption_score):
                self.logger.error(f"Caption score database save failed for {service}: {image_filename}")
                self._safe_nack(ch, method.delivery_tag, requeue=True)
                return

            self.logger.info(f"Saved caption score for {service}: {image_filename}")

            # Save image embedding once — idempotent, so safe to call on every message
            image_embedding = clip_result.get('image_embedding')
            if image_embedding is not None:
                if not self.save_image_embedding(image_id, image_embedding):
                    # Non-fatal: caption score is already committed; log and continue
                    self.logger.warning(f"Image embedding save failed for {image_filename} (image_id={image_id}); will be written on next caption score")

            # Only acknowledge message when caption score is durably saved
            self._safe_ack(ch, method.delivery_tag)
            
        except Exception as e:
            self.logger.error(f"Error processing queue message: {e}")
            # Reject and requeue for retry
            self._safe_nack(ch, method.delivery_tag, requeue=True)
    
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
        
        # Consume with reconnect loop
        while True:
            try:
                self.channel.basic_consume(
                    queue=self.queue_name,
                    on_message_callback=self.process_queue_message
                )
                self.logger.info("Waiting for caption scoring messages. Press CTRL+C to exit")
                self.channel.start_consuming()
            except KeyboardInterrupt:
                self.logger.info("Stopping caption score worker...")
                try:
                    self.channel.stop_consuming()
                except Exception:
                    pass
                break
            except (pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError,
                    pika.exceptions.StreamLostError) as e:
                self.logger.warning(f"Queue connection lost: {e}. Reconnecting...")
                time.sleep(self.retry_delay)
                if not self.connect_to_queue():
                    self.logger.warning("Reconnect failed, retrying...")
                    time.sleep(self.retry_delay)
                    continue

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