#!/bin/bash
export SERVICE_NAME=rtdetr
python generic_worker.py > logs/rtdetr_worker.log 2>&1 &
echo $! > pids/rtdetr_worker.pid
echo "Started RT-DETR worker (PID: $!)"

