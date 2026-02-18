#!/usr/bin/env python3
"""
SpeciesNetWorker - Google SpeciesNet wildlife species identification

SpeciesNet is a spatial service (returns bounding box detections) but does not
participate in consensus yet (no emoji mapping). Configured via service_config.yaml
with service_type: spatial and consensus: false.
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from base_worker import BaseWorker


class SpeciesNetWorker(BaseWorker):
    """Worker for SpeciesNet wildlife species identification service"""

    def __init__(self):
        super().__init__('primary.speciesnet')


if __name__ == "__main__":
    worker = SpeciesNetWorker()
    worker.start()
