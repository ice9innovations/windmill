#!/bin/bash
export SERVICE_NAME=xception
python generic_worker.py > logs/xception_worker.log 2>&1 &
echo $! > pids/xception_worker.pid
echo "Started xception worker (PID: $!)"

