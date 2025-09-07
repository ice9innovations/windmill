#!/usr/bin/env python3
"""
Detectron2Worker - detectron2 ML service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class Detectron2Worker(BaseWorker):
    """Worker for detectron2 ML service"""
    
    def __init__(self):
        super().__init__('primary.detectron2')

if __name__ == "__main__":
    worker = Detectron2Worker()
    worker.start()
