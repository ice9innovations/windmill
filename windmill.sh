#!/bin/bash
# Simple, reliable worker management (no wrapper scripts)
# Usage: ./windmill.sh {start|stop|restart|status}

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR"
source windmill_venv/bin/activate

load_dotenv_key() {
    local key="$1"
    if [ -f ".env" ]; then
        grep -E "^${key}=" ".env" | tail -n 1 | cut -d= -f2-
    fi
}

if [ -z "${WINDMILL_WORKER_CAPACITY+x}" ]; then
    WINDMILL_WORKER_CAPACITY="$(load_dotenv_key "WINDMILL_WORKER_CAPACITY")"
fi
if [ -z "${WINDMILL_ENABLED_WORKERS+x}" ]; then
    WINDMILL_ENABLED_WORKERS="$(load_dotenv_key "WINDMILL_ENABLED_WORKERS")"
fi

ACTION="$1"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Per-machine state file — tracks which workers are enabled on this machine.
# Managed automatically by start/stop; never committed to git.
STATE_FILE=".windmill_state"
SCHEDULER_STATUS_FILE=".windmill_scheduler_status"

state_add() {
    local name="$1"
    touch "$STATE_FILE"
    if ! grep -qx "$name" "$STATE_FILE"; then
        echo "$name" >> "$STATE_FILE"
    fi
}

state_remove() {
    local name="$1"
    if [ -f "$STATE_FILE" ]; then
        grep -vx "$name" "$STATE_FILE" > "${STATE_FILE}.tmp" || true
        mv "${STATE_FILE}.tmp" "$STATE_FILE"
    fi
}

get_enabled_workers() {
    if [ -n "$WINDMILL_ENABLED_WORKERS" ]; then
        echo "$WINDMILL_ENABLED_WORKERS" | tr ',' '\n' | tr ' ' '\n' | sed '/^$/d'
    elif [ -f "$STATE_FILE" ] && [ -s "$STATE_FILE" ]; then
        cat "$STATE_FILE"
    fi
}

normalize_worker_name() {
    local worker="$1"
    case "$worker" in
        "caption_score")
            worker="caption_score_worker"
            ;;
        "colors_post")
            worker="colors_post_worker"
            ;;
        "face")
            worker="face_worker"
            ;;
        "pose")
            worker="pose_worker"
            ;;
    esac
    if [[ "$worker" == *.py ]]; then
        worker="$(basename "$worker" ".py")"
    fi
    if [[ "$worker" != *_worker ]]; then
        worker="${worker}_worker"
    fi
    echo "$worker"
}

worker_enabled_for_scheduler() {
    local target
    target="$(normalize_worker_name "$1")"
    for enabled_worker in $(get_enabled_workers); do
        if [ "$(normalize_worker_name "$enabled_worker")" = "$target" ]; then
            return 0
        fi
    done
    return 1
}

bootstrap_state_from_running() {
    echo "  No state file found — scanning running processes to build initial state..."
    local found=0
    for worker in $(get_all_workers); do
        if pgrep -f "workers/${worker}.py" >/dev/null 2>&1; then
            state_add "$worker"
            echo "  Found running: $worker"
            found=$((found + 1))
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo -e "${YELLOW}  No running workers found. Use './windmill.sh start <worker>' to enable workers.${NC}"
    else
        echo "  State file initialized with $found worker(s)."
    fi
}

# Dynamically build worker list from available worker files
get_all_workers() {
    # Get all worker files and extract service names, excluding base classes
    for worker_file in workers/*_worker.py; do
        if [ -f "$worker_file" ]; then
            # Extract worker name from full filename
            basename_file=$(basename "$worker_file" ".py")
            
            # Skip base classes and utilities
            if [[ "$basename_file" != "base_worker" && "$basename_file" != "machine_scheduler_worker" && "$basename_file" != "postprocessing_worker" && "$basename_file" != "db_worker" && "$basename_file" != "service_config" ]]; then
                echo "$basename_file"
            fi
        fi
    done | sort -u
}

capacity_mode_enabled() {
    [ -n "${WINDMILL_WORKER_CAPACITY:-}" ]
}

start_scheduler() {
    mkdir -p logs
    if pgrep -f "workers/machine_scheduler_worker.py" >/dev/null 2>&1; then
        local pid=$(pgrep -f "workers/machine_scheduler_worker.py")
        echo -e "${GREEN}✅ machine_scheduler_worker${NC} already running (PID: $pid)"
        return 0
    fi
    rm -f "$SCHEDULER_STATUS_FILE"
    echo "  Starting machine scheduler with WINDMILL_WORKER_CAPACITY=$WINDMILL_WORKER_CAPACITY..."
    nohup python workers/machine_scheduler_worker.py >> logs/machine_scheduler_worker.log 2>&1 &
    wait_for_scheduler_status
    echo -e "${GREEN}✅ Started machine scheduler${NC}"
}

stop_scheduler() {
    if pkill -f "workers/machine_scheduler_worker.py" 2>/dev/null; then
        local attempts=0
        while [ "$attempts" -lt 20 ]; do
            if ! pgrep -f "workers/machine_scheduler_worker.py" >/dev/null 2>&1; then
                rm -f "$SCHEDULER_STATUS_FILE"
                return 0
            fi
            sleep 0.25
            attempts=$((attempts + 1))
        done
        pkill -9 -f "workers/machine_scheduler_worker.py" 2>/dev/null
        rm -f "$SCHEDULER_STATUS_FILE"
    fi
}

wait_for_scheduler_status() {
    if ! capacity_mode_enabled; then
        return
    fi
    local attempts=0
    while [ "$attempts" -lt 40 ]; do
        if [ -f "$SCHEDULER_STATUS_FILE" ]; then
            python - "$SCHEDULER_STATUS_FILE" <<'PY'
import json
import sys
import time

try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    age = time.time() - float(data.get("updated_at_epoch", 0) or 0)
    if data.get("state") == "running" and age <= 5:
        raise SystemExit(0)
except Exception:
    pass
raise SystemExit(1)
PY
            if [ "$?" -eq 0 ]; then
                return
            fi
        fi
        sleep 0.25
        attempts=$((attempts + 1))
    done
    echo -e "${YELLOW}  Scheduler started, but status is not fresh yet. Check logs/machine_scheduler_worker.log if this persists.${NC}"
}

start_all() {
    mkdir -p logs

    # Bootstrap state from running processes if no state file exists yet
    if [ ! -f "$STATE_FILE" ]; then
        bootstrap_state_from_running
    fi

    local enabled
    enabled=$(get_enabled_workers)

    if [ -z "$enabled" ]; then
        echo -e "${YELLOW}⚠️  No enabled workers on this machine.${NC}"
        echo "    Use './windmill.sh start <worker>' or WINDMILL_ENABLED_WORKERS to enable workers."
        return
    fi

    if capacity_mode_enabled; then
        echo "🚦 Capacity mode enabled; starting one machine scheduler."
        start_scheduler
        return
    fi

    echo "🚀 Starting enabled workers..."
    for worker in $enabled; do
        start_worker "$worker"
    done

    sleep 2
    echo -e "${GREEN}✅ Started enabled workers${NC}"
}

stop_all() {
    echo "🛑 Stopping all workers..."
    if pgrep -f "workers/machine_scheduler_worker.py" >/dev/null 2>&1; then
        stop_scheduler
        echo "  ✅ Stopped machine_scheduler_worker"
    fi
    
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
    local scheduler_running=0
    if pgrep -f "workers/machine_scheduler_worker.py" >/dev/null 2>&1; then
        scheduler_running=1
        local scheduler_pid=$(pgrep -f "workers/machine_scheduler_worker.py")
        echo -e "${GREEN}✅ machine_scheduler_worker${NC} (PID: $scheduler_pid, capacity: ${WINDMILL_WORKER_CAPACITY:-unset})"
        print_scheduler_status
    else
        echo -e "${RED}❌ machine_scheduler_worker${NC} (not running, capacity: ${WINDMILL_WORKER_CAPACITY:-unset})"
        if [ -f "$SCHEDULER_STATUS_FILE" ]; then
            print_scheduler_status
        fi
    fi
    
    # Check all workers - unified clean approach
    for worker in $(get_all_workers); do
        if capacity_mode_enabled && worker_enabled_for_scheduler "$worker"; then
            if [ "$scheduler_running" -eq 1 ]; then
                echo -e "${GREEN}✅ $worker${NC} (managed by machine_scheduler_worker)"
            else
                echo -e "${YELLOW}⚠️  $worker${NC} (enabled for scheduler, scheduler not running)"
            fi
        elif pgrep -f "workers/${worker}.py" >/dev/null 2>&1; then
            local pid=$(pgrep -f "workers/${worker}.py")
            echo -e "${GREEN}✅ $worker${NC} (PID: $pid)"
        else
            echo -e "${RED}❌ $worker${NC} (not running)"
        fi
    done
}

print_scheduler_status() {
    if [ ! -f "$SCHEDULER_STATUS_FILE" ]; then
        echo "   scheduler status: not available yet"
        return
    fi
    python - "$SCHEDULER_STATUS_FILE" <<'PY'
import json
import sys
import time

path = sys.argv[1]
try:
    with open(path) as f:
        data = json.load(f)
except Exception as exc:
    print(f"   scheduler status: unreadable ({exc})")
    raise SystemExit(0)

age = time.time() - float(data.get("updated_at_epoch", 0) or 0)
state = data.get("state", "unknown")
capacity = data.get("capacity", "unknown")
stale = " stale" if age > 10 else ""
print(f"   scheduler: state={state} capacity={capacity} updated={age:.1f}s ago{stale}")

managed = data.get("managed") or []
if managed:
    names = ", ".join(
        f"{item.get('worker')}[{item.get('queue')}]"
        for item in managed
    )
    print(f"   managed: {names}")
else:
    print("   managed: none")

for slot in data.get("slots") or []:
    slot_id = slot.get("slot")
    failed = slot.get("failed")
    job = slot.get("current_job")
    if failed:
        print(f"   slot {slot_id}: failed ({failed})")
    elif job:
        runtime = time.time() - float(job.get("started_at_epoch", time.time()))
        print(
            f"   slot {slot_id}: active "
            f"{job.get('service')}[{job.get('queue')}] {runtime:.1f}s"
        )
    else:
        print(f"   slot {slot_id}: idle")
PY
}

stop_worker() {
    local worker="$1"
    local worker_file=""
    
    # Map service names to actual worker files (same mapping as start_worker)
    case "$worker" in
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
            # Accept both full name (noun_consensus_worker) and short name (noun_consensus)
            if [ -f "workers/${worker}.py" ]; then
                worker_file="workers/${worker}.py"
            elif [ -f "workers/${worker}_worker.py" ]; then
                worker_file="workers/${worker}_worker.py"
            else
                echo "❌ ERROR: Unknown worker '$worker'"
                return 1
            fi
            ;;
    esac
    
    local canonical=$(basename "$worker_file" ".py")

    if capacity_mode_enabled; then
        state_remove "$canonical"
        echo "✅ Unregistered $canonical from machine_scheduler_worker"
        if pgrep -f "workers/machine_scheduler_worker.py" >/dev/null 2>&1; then
            echo "  Restarting machine scheduler to apply enabled-worker changes..."
            stop_scheduler
            start_scheduler
        fi
        return 0
    fi

    local pids=$(pgrep -f "$worker_file")
    if [ -n "$pids" ]; then
        echo "  Stopping PIDs: $pids"
        kill $pids 2>/dev/null        # SIGTERM — gives worker chance to mark offline
        sleep 4                       # Grace period for clean shutdown
        local remaining=$(pgrep -f "$worker_file")
        if [ -n "$remaining" ]; then
            kill -9 $remaining 2>/dev/null  # Force kill if still alive
            sleep 1
        fi
        state_remove "$canonical"
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
            # Accept both full name (noun_consensus_worker) and short name (noun_consensus)
            if [ -f "workers/${worker}.py" ]; then
                worker_file="workers/${worker}.py"
            elif [ -f "workers/${worker}_worker.py" ]; then
                worker_file="workers/${worker}_worker.py"
            else
                echo "❌ ERROR: Unknown worker '$worker'"
                return 1
            fi
            ;;
    esac
    
    # Check if worker file exists
    if [ ! -f "$worker_file" ]; then
        echo "❌ ERROR: $worker_file does not exist"
        return 1
    fi

    # Extract actual worker name from file path for consistent logging
    local log_name=$(basename "$worker_file" ".py")

    state_add "$log_name"

    if capacity_mode_enabled; then
        echo "  Capacity mode enabled; registering $worker for scheduler."
        start_scheduler
        return 0
    fi

    echo "  Starting $worker..."
    nohup python $worker_file >> logs/${log_name}.log 2>&1 &
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
                wait_for_scheduler_status
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
            wait_for_scheduler_status
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
