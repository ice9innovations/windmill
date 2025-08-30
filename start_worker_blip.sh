# start_yolov8_worker.sh  
#!/bin/bash
export SERVICE_NAME=blip
python generic_worker.py > logs/blip_worker.log 2>&1 &
echo $! > pids/blip_worker.pid
echo "Started BLIP worker (PID: $!)"

