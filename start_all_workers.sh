# start_all_workers.sh
#!/bin/bash
mkdir -p logs pids

./start_worker_blip.sh
./start_worker_clip.sh
./start_worker_colors.sh
./start_worker_detectron.sh
./start_worker_inception.sh
./start_worker_metadata.sh
./start_worker_nsfw.sh
./start_worker_ocr.sh
./start_worker_ollama.sh
./start_worker_rtdetr.sh
./start_worker_yolo.sh

./start_all_postprocessing.sh

echo "All workers started. Check logs/ directory for output."
