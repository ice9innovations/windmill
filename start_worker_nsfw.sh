#!/bin/bash
export SERVICE_NAME=nsfw2
python generic_worker.py > logs/nsfw2_worker.log 2>&1 &
echo $! > pids/nsfw2_worker.pid
echo "Started NSFW2 worker (PID: $!)"

