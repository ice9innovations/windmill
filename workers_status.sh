#!/bin/bash
# status_workers.sh - Check status of windmill workers
# Usage: ./workers_status.sh [worker_name]
#   - No args: Show status of all workers
#   - worker_name: Show status of specific worker (e.g. xception, blip, etc.)

WORKER_NAME="$1"

if [ -n "$WORKER_NAME" ]; then
    echo "📊 Checking status of worker: $WORKER_NAME..."
else
    echo "📊 Checking status of all windmill workers..."
fi
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

running_count=0
dead_count=0

for pidfile in pids/*.pid; do
    if [ -f "$pidfile" ]; then
        PID=$(cat "$pidfile")
        SERVICE=$(basename "$pidfile" .pid)
        
        # Skip if specific worker requested and this isn't it
        # Strip _worker suffix for comparison (e.g. "ollama_worker" -> "ollama")
        service_name="${SERVICE%_worker}"
        if [ -n "$WORKER_NAME" ] && [ "$service_name" != "$WORKER_NAME" ]; then
            continue
        fi
        if kill -0 $PID 2>/dev/null; then
            echo -e "${GREEN}✅ $SERVICE is running (PID: $PID)${NC}"
            running_count=$((running_count + 1))
        else
            echo -e "${RED}❌ $SERVICE is dead (PID: $PID)${NC}"
            rm "$pidfile"
            dead_count=$((dead_count + 1))
        fi
    fi
done

echo
echo "📊 Status summary:"
echo -e "  ${GREEN}✅ Running: $running_count workers${NC}"
if [ $dead_count -gt 0 ]; then
    echo -e "  ${RED}❌ Dead: $dead_count workers${NC}"
fi

# Check if specific worker was requested but not found
if [ -n "$WORKER_NAME" ] && [ $running_count -eq 0 ] && [ $dead_count -eq 0 ]; then
    echo -e "  ${YELLOW}⚠️  Worker '$WORKER_NAME' not found${NC}"
    echo "Available workers:"
    for pidfile in pids/*.pid; do
        if [ -f "$pidfile" ]; then
            echo "  - $(basename "$pidfile" .pid)"
        fi
    done
fi
