# start_yolov8_worker.sh  
#!/bin/bash
export SERVICE_NAME=clip
python generic_worker.py > logs/clip_worker.log 2>&1 &
echo $! > pids/clip_worker.pid
echo "Started CLIP worker (PID: $!)"

