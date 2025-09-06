#!/bin/bash
# Simple, reliable worker management (no wrapper scripts)
# Usage: ./windmill.sh {start|stop|restart|status}

ACTION="$1"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Dynamically build worker list from available worker files
get_all_workers() {
    # Get all worker files and extract service names, excluding base classes
    for worker_file in workers/*_worker.py; do
        if [ -f "$worker_file" ]; then
            # Extract worker name: workers/blip_worker.py -> blip
            worker_name=$(basename "$worker_file" "_worker.py")
            
            # Skip base classes and utilities
            if [[ "$worker_name" != "base" && "$worker_name" != "generic" && "$worker_name" != "postprocessing" ]]; then
                echo "$worker_name"
            fi
        fi
    done
}

start_all() {
    echo "🚀 Starting all workers..."
    mkdir -p logs
    
    # Start all workers - now they all use the same clean pattern
    for worker in $(get_all_workers); do
        echo "  Starting $worker..."
        nohup python workers/${worker}_worker.py > logs/${worker}_worker.log 2>&1 &
    done
    
    sleep 2
    echo -e "${GREEN}✅ Started all workers${NC}"
}

stop_all() {
    echo "🛑 Stopping all workers..."
    
    # Stop each worker individually - now they can all be targeted precisely!
    for worker in $(get_all_workers); do
        if pkill -f "workers/${worker}_worker.py" 2>/dev/null; then
            echo "  ✅ Stopped $worker"
        fi
    done
    
    # Wait and verify everything is dead
    sleep 1
    remaining=$(ps aux | grep "python workers" | grep -v grep | wc -l)
    if [ "$remaining" -gt 0 ]; then
        echo "  ⚠️  Force killing remaining $remaining processes..."
        pkill -9 -f "python workers/" 2>/dev/null
        sleep 1
    fi
    
    echo -e "${GREEN}✅ All workers stopped${NC}"
}

status_all() {
    echo "📊 Worker Status:"
    echo "===================="
    
    # Check all workers - unified clean approach
    for worker in $(get_all_workers); do
        if pgrep -f "workers/${worker}_worker.py" >/dev/null 2>&1; then
            local pid=$(pgrep -f "workers/${worker}_worker.py")
            echo -e "${GREEN}✅ $worker${NC} (PID: $pid)"
        else
            echo -e "${RED}❌ $worker${NC} (not running)"
        fi
    done
}

stop_worker() {
    local worker="$1"
    if pkill -f "workers/${worker}_worker.py" 2>/dev/null; then
        echo "✅ Stopped $worker"
        return 0
    else
        echo "❌ $worker was not running"
        return 1
    fi
}

start_worker() {
    local worker="$1"
    
    # Check if worker file exists
    if [ ! -f "workers/${worker}_worker.py" ]; then
        echo "❌ ERROR: workers/${worker}_worker.py does not exist"
        return 1
    fi
    
    echo "  Starting $worker..."
    nohup python workers/${worker}_worker.py > logs/${worker}_worker.log 2>&1 &
    echo "✅ Started $worker"
}

case "$ACTION" in
    start)
        if [ -n "$2" ]; then
            # Start individual worker: ./windmill.sh start blip
            echo "🚀 Starting $2..."
            mkdir -p logs
            start_worker "$2"
        else
            # Start all workers
            start_all
        fi
        ;;
    stop)
        if [ -n "$2" ]; then
            # Stop individual worker: ./windmill.sh stop blip
            echo "🛑 Stopping $2..."
            stop_worker "$2"
        else
            # Stop all workers
            stop_all
        fi
        ;;
    restart)
        if [ -n "$2" ]; then
            # Restart individual worker: ./windmill.sh restart ollama
            echo "🔄 Restarting $2..."
            stop_worker "$2"
            sleep 1
            start_worker "$2"
            echo ""
            status_all
        else
            # Restart all workers
            echo "🔄 Restarting all workers..."
            stop_all
            sleep 2
            start_all
            echo ""
            status_all
        fi
        ;;
    status)
        status_all
        ;;
    *)
        echo "Usage: $0 {start [worker]|stop [worker]|restart [worker]|status}"
        echo ""
        echo "Examples:"
        echo "  $0 start          # Start all workers"
        echo "  $0 start ollama   # Start just the ollama worker"
        echo "  $0 stop           # Stop all workers" 
        echo "  $0 stop ollama    # Stop just the ollama worker"
        echo "  $0 restart        # Restart all workers"
        echo "  $0 restart ollama # Restart just the ollama worker"
        echo "  $0 status         # Show status of all workers"
        exit 1
        ;;
esac
