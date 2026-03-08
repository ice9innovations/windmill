#!/usr/bin/env python3
"""
Florence2Worker - Florence-2 VLM service worker (MORE_DETAILED_CAPTION)
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class Florence2Worker(BaseWorker):
    """Worker for Florence-2 paragraph captioning"""

    def __init__(self):
        super().__init__('primary.florence2')

if __name__ == "__main__":
    worker = Florence2Worker()
    worker.start()
