#!/usr/bin/env python3
"""
Yolov8Worker - yolov8 ML service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class Yolov8Worker(BaseWorker):
    """Worker for yolov8 ML service"""
    
    def __init__(self):
        super().__init__('yolo_v8')

if __name__ == "__main__":
    worker = Yolov8Worker()
    worker.start()
