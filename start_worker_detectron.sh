# start_yolov8_worker.sh  
#!/bin/bash
export SERVICE_NAME=detectron2
python generic_worker.py > logs/detectron2_worker.log 2>&1 &
echo $! > pids/detectron2_worker.pid
echo "Started Detectron2 worker (PID: $!)"

