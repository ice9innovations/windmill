#!/usr/bin/env python3
"""
BboxPoseWorker - Pose estimation on cropped person bounding boxes
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import base64
import io
import requests
from postprocessing_worker import PostProcessingWorker

class BboxPoseWorker(PostProcessingWorker):
    """Worker for pose estimation on cropped person bounding boxes"""
    
    def __init__(self):
        super().__init__('pose', None, service_port=7786)
    
    def process_service(self, cropped_image_data):
        """Process pose estimation on cropped image"""
        try:
            # Decode base64 image data
            image_bytes = base64.b64decode(cropped_image_data.encode('latin-1'))
            
            # Call pose service
            files = {'file': ('bbox_crop.jpg', io.BytesIO(image_bytes), 'image/jpeg')}
            response = requests.post(
                self.service_url,
                files=files,
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                pose_data = response.json()
                if pose_data.get('status') == 'success' and pose_data.get('predictions'):
                    return pose_data
            
            self.logger.warning(f"Pose service returned status {response.status_code}: {response.text[:200]}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing pose estimation: {e}")
            return None

if __name__ == "__main__":
    worker = BboxPoseWorker()
    worker.start()