#!/usr/bin/env python3
"""
Nsfw2Worker - nsfw2 ML service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class Nsfw2Worker(BaseWorker):
    """Worker for nsfw2 ML service"""
    
    def __init__(self):
        super().__init__('nsfw2')

if __name__ == "__main__":
    worker = Nsfw2Worker()
    worker.start()