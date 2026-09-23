#!/usr/bin/env python3
"""NSFWSegmentationWorker - NSFW segmentation ML service worker."""
import os
import sys

sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker


class NSFWSegmentationWorker(BaseWorker):
    """Worker for the NSFW segmentation service."""

    def __init__(self):
        super().__init__('primary.nsfw-segmentation')


if __name__ == "__main__":
    worker = NSFWSegmentationWorker()
    worker.start()
