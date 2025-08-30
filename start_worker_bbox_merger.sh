# start_bbox_merger.sh
#!/bin/bash
python bbox_merger_worker.py > logs/bbox_merger.log 2>&1 &
echo $! > pids/bbox_merger.pid
echo "Started bbox merger (PID: $!)"
