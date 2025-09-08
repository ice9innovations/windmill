#!/usr/bin/env python3
"""
RtdetrWorker - rtdetr ML service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class RtdetrWorker(BaseWorker):
    """Worker for rtdetr ML service"""
    
    def __init__(self):
        super().__init__('primary.rtdetr')

if __name__ == "__main__":
    worker = RtdetrWorker()
    worker.start()
