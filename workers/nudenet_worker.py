#!/usr/bin/env python3
"""
NudenetWorker - nudenet ML service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class NudenetWorker(BaseWorker):
    """Worker for nudenet ML service"""

    def __init__(self):
        super().__init__('primary.nudenet')

if __name__ == "__main__":
    worker = NudenetWorker()
    worker.start()
