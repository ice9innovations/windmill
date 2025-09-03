#!/usr/bin/env python3
"""
MetadataWorker - metadata ML service worker
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

class MetadataWorker(BaseWorker):
    """Worker for metadata ML service"""
    
    def __init__(self):
        super().__init__('metadata')
    
    def process_message(self, ch, method, properties, body):
        """Process a metadata ML request"""
        try:
            # Parse message
            message = json.loads(body)
            image_id = message['image_id']
            image_url = message['image_url']
            
            self.logger.debug(f"Processing metadata request for image {image_id}")
            
            # Call metadata service
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
            
            # Trigger post-processing for bbox services
            if self.service_name in self.bbox_services and self.enable_triggers:
                bbox_message = {
                    'image_id': image_id,
                    'image_filename': message.get('image_filename', f'image_{image_id}'),
                    'service': self.service_name,
                    'worker_id': self.worker_id,
                    'processed_at': datetime.now().isoformat()
                }
                
                self.channel.basic_publish(
                    exchange='',
                    routing_key='queue_bbox_merge',
                    body=json.dumps(bbox_message),
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                
                self.logger.debug(f"Published bbox completion to queue_bbox_merge")
            
            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            self.job_completed_successfully()
            
            self.logger.info(f"Successfully processed metadata request for image {image_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing metadata message: {e}")
            # Reject and requeue
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            self.job_failed(str(e))

if __name__ == "__main__":
    worker = MetadataWorker()
    worker.start()
