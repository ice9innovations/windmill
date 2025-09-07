#!/usr/bin/env python3
"""
MetadataWorker - metadata ML service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class MetadataWorker(BaseWorker):
    """Worker for metadata ML service"""
    
    def __init__(self):
        super().__init__('primary.metadata')

if __name__ == "__main__":
    worker = MetadataWorker()
    worker.start()
