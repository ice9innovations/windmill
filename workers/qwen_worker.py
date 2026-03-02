#!/usr/bin/env python3
"""
QwenWorker - Qwen VL ML service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class QwenWorker(BaseWorker):
    """Worker for Qwen VL ML service"""

    def __init__(self):
        super().__init__('primary.qwen')

if __name__ == "__main__":
    worker = QwenWorker()
    worker.start()
