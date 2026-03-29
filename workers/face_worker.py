#!/usr/bin/env python3
"""
BboxFaceWorker - Face detection on cropped person bounding boxes
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import base64
import io
import json
import requests
from postprocessing_worker import PostProcessingWorker
from base_worker import BaseWorker

class BboxFaceWorker(PostProcessingWorker):
    """Worker for face detection on cropped person bounding boxes"""
    
    def __init__(self):
        super().__init__('primary.face')
        self.current_bbox = None  # Store current person bbox for coordinate transformation
    
    def process_message(self, ch, method, properties, body):
        """Handle both legacy bbox messages and new primary image messages."""
        try:
            message = self._parse_message_body(body)
            if 'cropped_image_data' in message or 'crop_ref' in message:
                self.current_bbox = message.get('bbox', {})
                return super().process_message(ch, method, properties, body)

            self.current_bbox = None
            return BaseWorker.process_message(self, ch, method, properties, body)

        except Exception as e:
            self.logger.error(f"Error in face message processing: {e}")
            self._safe_nack(ch, method.delivery_tag, requeue=True)

    def process_service(self, cropped_image_bytes):
        """Process face detection on cropped image"""
        try:
            # Call face service
            files = {'file': ('bbox_crop.jpg', io.BytesIO(cropped_image_bytes), 'image/jpeg')}
            response = requests.post(
                self.service_url,
                files=files,
                timeout=self.request_timeout
            )
            
            face_data = self._coerce_terminal_http_response(response)
            if face_data.get('status') == 'success' and face_data.get('predictions'):
                return self.transform_face_coordinates_to_full_image(face_data)
            return face_data
            
        except Exception as e:
            self.logger.error(f"Error processing face detection: {e}")
            return None
    
    def transform_face_coordinates_to_full_image(self, face_data):
        """Transform face coordinates from crop-relative to full-image-relative"""
        try:
            # Get the original person bounding box from the stored message
            if not self.current_bbox:
                self.logger.warning("No person bbox available, cannot transform face coordinates")
                return face_data
            
            person_x = self.current_bbox.get('x', 0)
            person_y = self.current_bbox.get('y', 0)
            
            transformed_data = face_data.copy()
            
            for prediction in transformed_data.get('predictions', []):
                # Transform bounding box coordinates
                if 'bbox' in prediction:
                    bbox = prediction['bbox']
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        crop_x, crop_y, crop_w, crop_h = bbox[0], bbox[1], bbox[2], bbox[3]
                        # Transform to full image coordinates
                        prediction['bbox'] = [
                            person_x + crop_x,  # x position on full image
                            person_y + crop_y,  # y position on full image  
                            crop_w,             # width stays the same
                            crop_h              # height stays the same
                        ]
                
                # Transform keypoint coordinates  
                if 'keypoints' in prediction:
                    keypoints = prediction['keypoints']
                    transformed_keypoints = {}
                    for keypoint_name, coords in keypoints.items():
                        if isinstance(coords, list) and len(coords) >= 2:
                            crop_x, crop_y = coords[0], coords[1]
                            transformed_keypoints[keypoint_name] = [
                                person_x + crop_x,  # x position on full image
                                person_y + crop_y   # y position on full image
                            ]
                    prediction['keypoints'] = transformed_keypoints
            
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Error transforming face coordinates: {e}")
            # Return original data if transformation fails
            return face_data

if __name__ == "__main__":
    worker = BboxFaceWorker()
    worker.start()
