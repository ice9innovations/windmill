#!/bin/bash
mkdir -p logs pids

echo "Starting post-processing workers..."
./start_worker_bbox_merger.sh
./start_worker_consensus.sh
./start_worker_spatial_enrichment.sh

echo "All post-processing workers started. Check logs/ for output."
