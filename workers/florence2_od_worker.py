#!/usr/bin/env python3
"""
Florence2OdWorker - Florence-2 open-vocabulary object detection worker (OD)
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class Florence2OdWorker(BaseWorker):
    """Worker for Florence-2 open-vocabulary object detection"""

    def __init__(self):
        super().__init__('primary.florence2_od')

if __name__ == "__main__":
    worker = Florence2OdWorker()
    worker.start()
