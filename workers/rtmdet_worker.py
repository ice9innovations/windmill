#!/usr/bin/env python3
"""
RtmdetWorker - rtmdet ML service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class RtmdetWorker(BaseWorker):
    """Worker for rtmdet ML service"""

    def __init__(self):
        super().__init__('primary.rtmdet')

if __name__ == "__main__":
    worker = RtmdetWorker()
    worker.start()