# start_colors_worker.sh
#!/bin/bash
export SERVICE_NAME=colors
python generic_worker.py > logs/colors_worker.log 2>&1 &
echo $! > pids/colors_worker.pid
echo "Started colors worker (PID: $!)"
