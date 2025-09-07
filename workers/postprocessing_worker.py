#!/usr/bin/env python3
"""
PostProcessingWorker - Base class for bbox postprocessing workers
Handles cropped image processing for colors, face, pose detection
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import json
import base64
import io
import requests
from datetime import datetime
from base_worker import BaseWorker

class PostProcessingWorker(BaseWorker):
    """Base class for postprocessing workers that handle cropped images"""
    
    def __init__(self, service_name, service_url, service_port=None):
        super().__init__(service_name)
        
        # Override queue name for postprocessing - use config-based naming
        if service_name in self.service_definitions:
            configured_queue = self.service_definitions[service_name].get('queue_name', service_name)
            self.queue_name = f"postprocessing_{configured_queue}"
        else:
            # Fallback for services not in config
            self.queue_name = f"postprocessing_{service_name}"
        
        # Service configuration for postprocessing
        if service_port:
            self.service_url = f"http://localhost:{service_port}/v3/analyze"
        else:
            self.service_url = service_url
    
    def start(self):
        """Start the postprocessing worker - setup database transactions"""
        # Call parent start to establish connections
        super().start()
        
        # Configure database for transactions (needed for FK handling)
        if self.db_conn:
            self.db_conn.autocommit = False
    
    def process_service(self, cropped_image_data):
        """Process the cropped image with the specific service - override in subclasses"""
        raise NotImplementedError("Subclasses must implement process_service")
    
    def save_postprocessing_result(self, merged_box_id, image_id, result_data, bbox, cluster_id):
        """Save postprocessing result to database"""
        try:
            cursor = self.db_conn.cursor()
            
            # Insert postprocessing result
            cursor.execute("""
                INSERT INTO postprocessing (merged_box_id, image_id, service, data, status, result_created)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                merged_box_id, 
                image_id, 
                self.service_name, 
                json.dumps(result_data) if result_data else json.dumps({}),
                'success',
                datetime.now()
            ))
            
            self.db_conn.commit()
            cursor.close()
            return True
            
        except Exception as e:
            error_str = str(e)
            if 'postprocessing_merged_box_id_fkey' in error_str:
                # FK violation means merged_box was superseded by reharmonization - silently skip
                self.logger.info(f"Merged_box_id {merged_box_id} no longer exists (superseded by reharmonization) - skipping")
                if self.db_conn:
                    self.db_conn.rollback()
                return True  # Return success to acknowledge the message
            else:
                self.logger.error(f"Error saving postprocessing result: {e}")
                if self.db_conn:
                    self.db_conn.rollback()
                return False
    
    def process_message(self, ch, method, properties, body):
        """Process a postprocessing message - standard pattern for all postprocessing workers"""
        try:
            # Parse message
            message = json.loads(body.decode('utf-8'))
            merged_box_id = message['merged_box_id']
            image_id = message['image_id']
            cluster_id = message['cluster_id']
            bbox = message['bbox']
            cropped_image_data = message['cropped_image_data']
            
            self.logger.info(f"Processing {self.service_name} for {cluster_id} (merged_box_id: {merged_box_id})")
            
            # Process with specific service
            result_data = self.process_service(cropped_image_data)
            
            # Save result
            if result_data:
                success = self.save_postprocessing_result(
                    merged_box_id, image_id, result_data, bbox, cluster_id
                )
                if success:
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    self.logger.info(f"Successfully processed {self.service_name} for {cluster_id}")
                else:
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            else:
                # Save empty result to avoid reprocessing
                self.save_postprocessing_result(
                    merged_box_id, image_id, None, bbox, cluster_id
                )
                ch.basic_ack(delivery_tag=method.delivery_tag)
                self.logger.warning(f"No {self.service_name} data for {cluster_id}, saved empty result")
                
        except Exception as e:
            error_str = str(e)
            if 'postprocessing_merged_box_id_fkey' in error_str:
                # FK violation means merged_box was superseded by reharmonization - silently skip
                self.logger.info(f"Merged_box_id {merged_box_id} no longer exists (superseded by reharmonization) - skipping")
                ch.basic_ack(delivery_tag=method.delivery_tag)  # Acknowledge to remove from queue
            else:
                self.logger.error(f"Error processing {self.service_name} message: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)