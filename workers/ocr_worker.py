#!/usr/bin/env python3
"""
OcrWorker - ocr ML service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class OcrWorker(BaseWorker):
    """Worker for ocr ML service"""
    
    def __init__(self):
        super().__init__('primary.ocr')

if __name__ == "__main__":
    worker = OcrWorker()
    worker.start()
