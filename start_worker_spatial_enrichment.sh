#!/bin/bash
python spatial_enrichment_worker.py > logs/spatial_enrichment_worker.log 2>&1 &
echo $! > pids/spatial_enrichment_worker.pid
echo "Started spatial enrichment worker (PID: $!)"
