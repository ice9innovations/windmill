#!/bin/bash
export SERVICE_NAME=ollama
python generic_worker.py > logs/ollama_worker.log 2>&1 &
echo $! > pids/ollama_worker.pid
echo "Started ollama worker (PID: $!)"

