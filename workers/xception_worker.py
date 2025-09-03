#!/usr/bin/env python3
"""
XceptionWorker - xception ML service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class XceptionWorker(BaseWorker):
    """Worker for xception ML service"""
    
    def __init__(self):
        super().__init__('xception')

if __name__ == "__main__":
    worker = XceptionWorker()
    worker.start()
