#!/bin/bash
export SERVICE_NAME=metadata
python generic_worker.py > logs/metadata_worker.log 2>&1 &
echo $! > pids/metadata_worker.pid
echo "Started metadata worker (PID: $!)"

