#!/bin/bash
export SERVICE_NAME=inception_v3
python generic_worker.py > logs/inception_v3_worker.log 2>&1 &
echo $! > pids/inception_v3_worker.pid
echo "Started Inception_v3 worker (PID: $!)"

