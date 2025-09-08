#!/usr/bin/env python3
"""
BboxColorsWorker - Color analysis on cropped bounding boxes
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import base64
import io
import requests
from postprocessing_worker import PostProcessingWorker

class BboxColorsWorker(PostProcessingWorker):
    """Worker for color analysis on cropped bounding boxes"""
    
    def __init__(self):
        super().__init__('colors')
    
    def process_service(self, cropped_image_data):
        """Process color analysis on cropped image"""
        try:
            # Decode base64 image data
            image_bytes = base64.b64decode(cropped_image_data.encode('latin-1'))
            
            # Call colors service
            files = {'file': ('bbox_crop.jpg', io.BytesIO(image_bytes), 'image/jpeg')}
            response = requests.post(
                self.service_url,
                files=files,
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                colors_data = response.json()
                if colors_data.get('status') == 'success' and colors_data.get('predictions'):
                    return colors_data
            
            self.logger.warning(f"Colors service returned status {response.status_code}: {response.text[:200]}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing colors: {e}")
            return None

if __name__ == "__main__":
    worker = BboxColorsWorker()
    worker.start()
