# start_yolov8_worker.sh  
#!/bin/bash
export SERVICE_NAME=yolov8
python generic_worker.py > logs/yolov8_worker.log 2>&1 &
echo $! > pids/yolov8_worker.pid
echo "Started YOLOv8 worker (PID: $!)"

