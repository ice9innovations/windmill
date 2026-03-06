#!/usr/bin/env python3
"""
QrWorker - QR code and barcode scanner ML service worker
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker

class QrWorker(BaseWorker):
    """Worker for QR code and barcode scanner service"""

    def __init__(self):
        super().__init__('primary.qr')

if __name__ == "__main__":
    worker = QrWorker()
    worker.start()
