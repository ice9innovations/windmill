#!/usr/bin/env python3
"""
BboxFaceWorker - Face detection on cropped person bounding boxes
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import base64
import io
import requests
from postprocessing_worker import PostProcessingWorker

class BboxFaceWorker(PostProcessingWorker):
    """Worker for face detection on cropped person bounding boxes"""
    
    def __init__(self):
        super().__init__('postprocessing.face')
        self.current_bbox = None  # Store current person bbox for coordinate transformation
    
    def process_message(self, ch, method, properties, body):
        """Override to store bbox data for coordinate transformation"""
        try:
            import json
            # Parse message
            message = json.loads(body.decode('utf-8'))
            self.current_bbox = message.get('bbox', {})
            
            # Call parent process_message which will call our process_service
            return super().process_message(ch, method, properties, body)
            
        except Exception as e:
            self.logger.error(f"Error in face message processing: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    def process_service(self, cropped_image_data):
        """Process face detection on cropped image"""
        try:
            # Decode base64 image data
            image_bytes = base64.b64decode(cropped_image_data.encode('latin-1'))
            
            # Call face service
            files = {'file': ('bbox_crop.jpg', io.BytesIO(image_bytes), 'image/jpeg')}
            response = requests.post(
                self.service_url,
                files=files,
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                face_data = response.json()
                if face_data.get('status') == 'success' and face_data.get('predictions') and len(face_data.get('predictions', [])) > 0:
                    # Transform coordinates from crop-relative to full-image-relative
                    transformed_data = self.transform_face_coordinates_to_full_image(face_data)
                    return transformed_data
            
            self.logger.warning(f"Face service returned status {response.status_code}: {response.text[:200]}")
            return None
            
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