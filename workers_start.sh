#!/bin/bash
# workers_start.sh - Start windmill workers
# Usage: ./workers_start.sh [worker_name]
#   - No args: Start all workers
#   - worker_name: Start specific worker (e.g. xception, blip, etc.)

WORKER_NAME="$1"

# Create necessary directories
mkdir -p logs pids

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Define worker mappings (service_name -> script_name)
declare -A WORKER_SCRIPTS=(
    ["blip"]="start_worker_blip.sh"
    ["clip"]="start_worker_clip.sh"
    ["colors"]="start_worker_colors.sh"
    ["detectron"]="start_worker_detectron.sh"
    ["inception"]="start_worker_inception.sh"
    ["metadata"]="start_worker_metadata.sh"
    ["nsfw"]="start_worker_nsfw.sh"
    ["ocr"]="start_worker_ocr.sh"
    ["ollama"]="start_worker_ollama.sh"
    ["rtdetr"]="start_worker_rtdetr.sh"
    ["yolo"]="start_worker_yolo.sh"
    ["bbox_merger"]="start_worker_bbox_merger.sh"
    ["caption_score"]="start_worker_caption_score.sh"
    ["consensus"]="start_worker_consensus.sh"
    ["spatial_enrichment"]="start_worker_spatial_enrichment.sh"
)

start_worker() {
    local worker_name="$1"
    local script_name="$2"
    
    if [ -f "$script_name" ]; then
        echo "🚀 Starting $worker_name worker..."
        ./"$script_name"
        if [ $? -eq 0 ]; then
            echo -e "  ${GREEN}✅ $worker_name started successfully${NC}"
            return 0
        else
            echo -e "  ${RED}❌ Failed to start $worker_name${NC}"
            return 1
        fi
    else
        echo -e "  ${RED}❌ Script not found: $script_name${NC}"
        return 1
    fi
}

started_count=0
failed_count=0

if [ -n "$WORKER_NAME" ]; then
    # Start specific worker
    echo "🚀 Starting worker: $WORKER_NAME..."
    echo
    
    if [[ -n "${WORKER_SCRIPTS[$WORKER_NAME]}" ]]; then
        if start_worker "$WORKER_NAME" "${WORKER_SCRIPTS[$WORKER_NAME]}"; then
            started_count=1
        else
            failed_count=1
        fi
    else
        echo -e "${RED}❌ Unknown worker: $WORKER_NAME${NC}"
        echo "Available workers:"
        for worker in "${!WORKER_SCRIPTS[@]}"; do
            echo "  - $worker"
        done | sort
        exit 1
    fi
    
else
    # Start all workers
    echo "🚀 Starting all windmill workers..."
    echo
    
    # Primary workers
    for worker_name in blip clip colors detectron inception metadata nsfw ocr ollama rtdetr yolo; do
        if start_worker "$worker_name" "${WORKER_SCRIPTS[$worker_name]}"; then
            started_count=$((started_count + 1))
        else
            failed_count=$((failed_count + 1))
        fi
    done
    
    # Start postprocessing workers
    echo
    echo "🔄 Starting postprocessing workers..."
    ./start_all_postprocessing.sh
    if [ $? -eq 0 ]; then
        started_count=$((started_count + 4))  # Assume 4 postprocessing workers
    else
        failed_count=$((failed_count + 1))
    fi
fi

echo
echo "🏁 Startup complete:"
echo -e "  ${GREEN}✅ Started: $started_count workers${NC}"
if [ $failed_count -gt 0 ]; then
    echo -e "  ${RED}❌ Failed: $failed_count workers${NC}"
fi
echo
echo "📁 Check logs/ directory for worker output."