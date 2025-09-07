#!/usr/bin/env python3
"""
OllamaWorker - ollama ML service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class OllamaWorker(BaseWorker):
    """Worker for ollama ML service"""
    
    def __init__(self):
        super().__init__('primary.ollama')

if __name__ == "__main__":
    worker = OllamaWorker()
    worker.start()
