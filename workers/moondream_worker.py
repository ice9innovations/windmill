#!/usr/bin/env python3
"""
MoondreamWorker - Moondream VLM service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class MoondreamWorker(BaseWorker):
    """Worker for Moondream VLM service"""

    def __init__(self):
        super().__init__('primary.moondream')

if __name__ == "__main__":
    worker = MoondreamWorker()
    worker.start()
