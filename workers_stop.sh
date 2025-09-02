#!/bin/bash
# stop_workers.sh - Gracefully stop windmill workers
# Usage: ./workers_stop.sh [worker_name]
#   - No args: Stop all workers
#   - worker_name: Stop specific worker (e.g. xception, blip, etc.)

WORKER_NAME="$1"

if [ -n "$WORKER_NAME" ]; then
    echo "🛑 Stopping worker: $WORKER_NAME..."
else
    echo "🛑 Stopping all windmill workers..."
fi

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

stopped_count=0
failed_count=0

for pidfile in pids/*.pid; do
    if [ -f "$pidfile" ]; then
        PID=$(cat "$pidfile")
        worker_name=$(basename "$pidfile" .pid)
        
        # Skip if specific worker requested and this isn't it
        # Strip _worker suffix for comparison (e.g. "ollama_worker" -> "ollama")
        service_name="${worker_name%_worker}"
        if [ -n "$WORKER_NAME" ] && [ "$service_name" != "$WORKER_NAME" ]; then
            continue
        fi
        
        # Check if process is still running
        if ! kill -0 "$PID" 2>/dev/null; then
            echo -e "${YELLOW}🧹 Worker $worker_name (PID: $PID) not running - cleaning up PID file${NC}"
            rm "$pidfile"
            continue
        fi
        
        echo "🛑 Stopping $worker_name (PID: $PID)..."
        
        # Try graceful shutdown first (SIGTERM)
        kill "$PID" 2>/dev/null
        
        # Wait up to 5 seconds for graceful shutdown
        for i in {1..5}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo -e "  ${GREEN}✅ $worker_name stopped gracefully${NC}"
                rm "$pidfile"
                stopped_count=$((stopped_count + 1))
                break
            fi
            sleep 1
        done
        
        # If still running, force kill
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "  ${YELLOW}⚡ Force killing $worker_name...${NC}"
            kill -9 "$PID" 2>/dev/null
            sleep 1
            
            if ! kill -0 "$PID" 2>/dev/null; then
                echo -e "  ${GREEN}✅ $worker_name force stopped${NC}"
                rm "$pidfile"
                stopped_count=$((stopped_count + 1))
            else
                echo -e "  ${RED}❌ Failed to stop $worker_name${NC}"
                failed_count=$((failed_count + 1))
            fi
        fi
    fi
done

echo
echo "🏁 Shutdown complete:"
echo -e "  ${GREEN}✅ Stopped: $stopped_count workers${NC}"
if [ $failed_count -gt 0 ]; then
    echo -e "  ${RED}❌ Failed: $failed_count workers${NC}"
fi

# Check if specific worker was requested but not found
if [ -n "$WORKER_NAME" ] && [ $stopped_count -eq 0 ] && [ $failed_count -eq 0 ]; then
    echo -e "  ${YELLOW}⚠️  Worker '$WORKER_NAME' not found${NC}"
    echo "Available workers:"
    for pidfile in pids/*.pid; do
        if [ -f "$pidfile" ]; then
            echo "  - $(basename "$pidfile" .pid)"
        fi
    done
fi

# Clean up any orphaned PID files
find pids/ -name "*.pid" -type f 2>/dev/null | while read -r pidfile; do
    PID=$(cat "$pidfile" 2>/dev/null)
    if [ -n "$PID" ] && ! kill -0 "$PID" 2>/dev/null; then
        rm "$pidfile"
        echo -e "${YELLOW}🧹 Cleaned up orphaned PID file: $(basename "$pidfile")${NC}"
    fi
done

echo "🌾 The windmill has stopped."