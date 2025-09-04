#!/usr/bin/env python3
"""
Yolo_oi7Worker - YOLOv8 Open Images v7 detection service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class Yolo_oi7Worker(BaseWorker):
    """Worker for YOLOv8 Open Images v7 detection service"""
    
    def __init__(self):
        super().__init__('yolo_oi7')

if __name__ == "__main__":
    worker = Yolo_oi7Worker()
    worker.start()