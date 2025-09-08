#!/usr/bin/env python3
"""
ColorsWorker - colors ML service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class ColorsWorker(BaseWorker):
    """Worker for colors ML service"""
    
    def __init__(self):
        super().__init__('colors')

if __name__ == "__main__":
    worker = ColorsWorker()
    worker.start()
