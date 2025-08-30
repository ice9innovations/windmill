#!/bin/bash
export SERVICE_NAME=ocr
python generic_worker.py > logs/ocr_worker.log 2>&1 &
echo $! > pids/ocr_worker.pid
echo "Started OCR worker (PID: $!)"

