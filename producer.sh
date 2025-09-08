#!/bin/bash
# Producer script - Submit jobs to ML service queues
# Usage: ./producer.sh [options]

# Change to the workers directory where producer.py is located
cd "$(dirname "$0")/workers"

# Run the producer with all passed arguments
python producer.py "$@"
