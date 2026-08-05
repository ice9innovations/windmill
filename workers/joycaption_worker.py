#!/usr/bin/env python3
"""
JoyCaptionWorker - JoyCaption VLM service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker


class JoyCaptionWorker(BaseWorker):
    """Worker for JoyCaption image captioning service"""

    def __init__(self):
        super().__init__('primary.joycaption')


if __name__ == "__main__":
    worker = JoyCaptionWorker()
    worker.start()
