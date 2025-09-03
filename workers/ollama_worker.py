#!/usr/bin/env python3
"""
OllamaWorker - LLM vision analysis service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import json
import time
import requests
import pika
from datetime import datetime
from base_worker import BaseWorker

class OllamaWorker(BaseWorker):
    """Worker for Ollama LLM vision analysis service"""
    
    def __init__(self):
        super().__init__('ollama')
    
    def process_message(self, ch, method, properties, body):
        """Process an Ollama LLM vision request"""
        try:
            # Parse message
            message = json.loads(body)
            image_id = message['image_id']
            image_url = message['image_url']
            
            self.logger.debug(f"Processing Ollama request for image {image_id}")
            
            # Call Ollama service
            service_url = self.get_service_url(image_url)
            response = requests.get(service_url, timeout=self.request_timeout)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Store result in database
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO results (image_id, service, data, status, result_created, worker_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (image_id, self.service_name, json.dumps(result), 'success', datetime.now(), self.worker_id))
            cursor.close()
            
            # Trigger caption scoring and consensus processing
            self.trigger_caption_scoring(image_id)
            self.trigger_consensus(image_id)
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            self.jobs_completed += 1
            
            self.logger.info(f"Successfully processed Ollama vision analysis for image {image_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing Ollama message: {e}")
            # Reject and requeue
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            self.jobs_failed += 1

if __name__ == "__main__":
    worker = OllamaWorker()
    worker.start()