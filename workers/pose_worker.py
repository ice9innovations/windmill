#!/usr/bin/env python3
"""
BboxPoseWorker - Pose estimation on cropped person bounding boxes
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

class BboxPoseWorker(PostProcessingWorker):
    """Worker for pose estimation on cropped person bounding boxes"""

    def __init__(self):
        super().__init__('primary.pose')
        self.current_bbox = None

    def process_message(self, ch, method, properties, body):
        """Handle both legacy bbox messages and new primary image messages."""
        try:
            message = self._parse_message_body(body)
            if 'cropped_image_data' in message:
                self.current_bbox = message.get('bbox', {})
                return super().process_message(ch, method, properties, body)
            self.current_bbox = None
            return BaseWorker.process_message(self, ch, method, properties, body)
        except Exception:
            self.current_bbox = {}
            return super().process_message(ch, method, properties, body)

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
                timeout=(5, 5)  # (connection timeout, read timeout)
            )

            pose_data = self._coerce_terminal_http_response(response)
            if pose_data.get('status') == 'success' and pose_data.get('predictions'):
                return self._transform_landmarks_to_full_image(pose_data)
            return pose_data

        except requests.exceptions.Timeout as e:
            self.logger.error(f"Pose service timeout after 5 seconds: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing pose estimation: {e}")
            return None

    def _transform_landmarks_to_full_image(self, pose_data):
        """Transform MediaPipe normalized landmarks (0-1 relative to crop) to
        full-image pixel coordinates by applying the person bbox offset."""
        if not self.current_bbox:
            self.logger.warning("No person bbox available, cannot transform pose landmarks")
            return pose_data

        crop_x = self.current_bbox.get('x', 0)
        crop_y = self.current_bbox.get('y', 0)
        crop_w = self.current_bbox.get('width', 1)
        crop_h = self.current_bbox.get('height', 1)

        result = pose_data.copy()
        for prediction in result.get('predictions', []):
            if 'landmarks' not in prediction:
                continue
            transformed = {}
            for name, coords in prediction['landmarks'].items():
                if isinstance(coords, dict) and 'x' in coords and 'y' in coords:
                    transformed[name] = {
                        'x': crop_x + coords['x'] * crop_w,
                        'y': crop_y + coords['y'] * crop_h,
                        'z': coords.get('z', 0),
                        'visibility': coords.get('visibility', 0),
                    }
                else:
                    transformed[name] = coords
            prediction['landmarks'] = transformed
        return result

if __name__ == "__main__":
    worker = BboxPoseWorker()
    worker.start()
