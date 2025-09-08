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
            # Extract worker name from full filename
            basename_file=$(basename "$worker_file" ".py")
            
            # Skip base classes and utilities
            if [[ "$basename_file" != "base_worker" && "$basename_file" != "postprocessing_worker" && "$basename_file" != "service_config" ]]; then
                echo "$basename_file"
            fi
        fi
    done | sort -u
}

start_all() {
    echo "🚀 Starting all workers..."
    mkdir -p logs
    
    # Start all workers - now they all use the same clean pattern
    for worker in $(get_all_workers); do
        echo "  Starting $worker..."
        nohup python workers/${worker}.py > logs/${worker}.log 2>&1 &
    done
    
    sleep 2
    echo -e "${GREEN}✅ Started all workers${NC}"
}

stop_all() {
    echo "🛑 Stopping all workers..."
    
    # Stop each worker individually - now they can all be targeted precisely!
    for worker in $(get_all_workers); do
        if pkill -f "workers/${worker}.py" 2>/dev/null; then
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
        if pgrep -f "workers/${worker}.py" >/dev/null 2>&1; then
            local pid=$(pgrep -f "workers/${worker}.py")
            echo -e "${GREEN}✅ $worker${NC} (PID: $pid)"
        else
            echo -e "${RED}❌ $worker${NC} (not running)"
        fi
    done
}

stop_worker() {
    local worker="$1"
    local worker_file=""
    
    # Map service names to actual worker files (same mapping as start_worker)
    case "$worker" in
        "harmony")
            worker_file="workers/harmony_worker.py"
            ;;
        "consensus")
            worker_file="workers/consensus_worker.py"
            ;;
        "caption_score")
            worker_file="workers/caption_score_worker.py"
            ;;
        "colors_post")
            worker_file="workers/colors_post_worker.py"
            ;;
        "face")
            worker_file="workers/face_worker.py"
            ;;
        "pose")
            worker_file="workers/pose_worker.py"
            ;;
        *)
            # For service workers, try the standard pattern first
            if [ -f "workers/${worker}_worker.py" ]; then
                worker_file="workers/${worker}_worker.py"
            elif [ -f "workers/${worker}.py" ]; then
                worker_file="workers/${worker}.py"
            else
                worker_file="workers/${worker}.py"  # fallback
            fi
            ;;
    esac
    
    if pkill -f "$worker_file" 2>/dev/null; then
        echo "✅ Stopped $worker"
        return 0
    else
        echo "❌ $worker was not running"
        return 1
    fi
}

start_worker() {
    local worker="$1"
    local worker_file=""
    
    # Map service names to actual worker files
    case "$worker" in
        "harmony")
            worker_file="workers/harmony_worker.py"
            ;;
        "consensus")
            worker_file="workers/consensus_worker.py"
            ;;
        "caption_score")
            worker_file="workers/caption_score_worker.py"
            ;;
        "colors_post")
            worker_file="workers/colors_post_worker.py"
            ;;
        "face")
            worker_file="workers/face_worker.py"
            ;;
        "pose")
            worker_file="workers/pose_worker.py"
            ;;
        *)
            # For service workers, try the standard pattern first
            if [ -f "workers/${worker}_worker.py" ]; then
                worker_file="workers/${worker}_worker.py"
            elif [ -f "workers/${worker}.py" ]; then
                worker_file="workers/${worker}.py"
            fi
            ;;
    esac
    
    # Check if worker file exists
    if [ ! -f "$worker_file" ]; then
        echo "❌ ERROR: $worker_file does not exist"
        return 1
    fi
    
    echo "  Starting $worker..."
    nohup python $worker_file > logs/${worker}.log 2>&1 &
    echo "✅ Started $worker"
}

case "$ACTION" in
    start)
        if [ -n "$2" ]; then
            # Start individual worker: ./windmill.sh start blip
            if [ "$2" = "all" ]; then
                echo "🚀 Starting all workers..."
                start_all
            else
                echo "🚀 Starting $2..."
                mkdir -p logs
                start_worker "$2"
            fi
        else
            # Start all workers
            start_all
        fi
        ;;
    stop)
        if [ -n "$2" ]; then
            # Stop individual worker: ./windmill.sh stop blip
            if [ "$2" = "all" ]; then
                echo "🛑 Stopping all workers..."
                stop_all
            else
                echo "🛑 Stopping $2..."
                stop_worker "$2"
            fi
        else
            # Stop all workers
            stop_all
        fi
        ;;
    restart)
        if [ -n "$2" ]; then
            # Restart individual worker: ./windmill.sh restart ollama
            if [ "$2" = "all" ]; then
                echo "🔄 Restarting all workers..."
                stop_all
                sleep 2
                start_all
                echo ""
                status_all
            else
                echo "🔄 Restarting $2..."
                stop_worker "$2"
                sleep 1
                start_worker "$2"
                echo ""
                status_all
            fi
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
