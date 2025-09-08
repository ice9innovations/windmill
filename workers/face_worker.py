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
                if face_data.get('status') == 'success' and face_data.get('predictions'):
                    return face_data
            
            self.logger.warning(f"Face service returned status {response.status_code}: {response.text[:200]}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing face detection: {e}")
            return None

if __name__ == "__main__":
    worker = BboxFaceWorker()
    worker.start()