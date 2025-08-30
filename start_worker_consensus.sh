#!/bin/bash

# Ensure directories exist
mkdir -p logs pids

# Stop any existing consensus worker
if [ -f pids/consensus_worker.pid ]; then
    PID=$(cat pids/consensus_worker.pid)
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping existing consensus worker (PID: $PID)"
        kill "$PID"
        sleep 2
    fi
    rm -f pids/consensus_worker.pid
fi

# Start consensus worker with better error handling
nohup python consensus_worker.py >> logs/consensus_worker.log 2>&1 &
PID=$!

# Save PID
echo $PID > pids/consensus_worker.pid

# Verify it started successfully
sleep 2
if kill -0 "$PID" 2>/dev/null; then
    echo "Started consensus worker (PID: $PID)"
else
    echo "Failed to start consensus worker"
    rm -f pids/consensus_worker.pid
    exit 1
fi
