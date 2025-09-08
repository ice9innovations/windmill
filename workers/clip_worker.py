#!/usr/bin/env python3
"""
ClipWorker - CLIP image classification service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class ClipWorker(BaseWorker):
    """Worker for CLIP image classification service"""
    
    def __init__(self):
        super().__init__('clip')

if __name__ == "__main__":
    worker = ClipWorker()
    worker.start()
