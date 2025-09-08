#!/usr/bin/env python3
"""
BlipWorker - BLIP image captioning service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class BlipWorker(BaseWorker):
    """Worker for BLIP image captioning service"""
    
    def __init__(self):
        super().__init__('blip')

if __name__ == "__main__":
    worker = BlipWorker()
    worker.start()
