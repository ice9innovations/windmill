#!/bin/bash
python caption_score_worker.py > logs/caption_score_worker.log 2>&1 &
echo $! > pids/caption_score_worker.pid
echo "Started Caption Score worker (PID: $!)"