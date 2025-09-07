#!/usr/bin/env python3
"""
Yolo_365Worker - YOLOv11 Object365 detection service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class Yolo_365Worker(BaseWorker):
    """Worker for YOLOv11 Object365 detection service"""
    
    def __init__(self):
        super().__init__('primary.yolo_365')

if __name__ == "__main__":
    worker = Yolo_365Worker()
    worker.start()