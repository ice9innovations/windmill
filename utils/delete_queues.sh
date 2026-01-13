#!/bin/bash
# Delete windmill queues using RabbitMQ Management API

set -e

# Load .env file
if [ -f .env ]; then
    source .env
else
    echo "Error: .env file not found"
    exit 1
fi

echo "Deleting windmill queues on $QUEUE_HOST..."

# Get list of queues and delete windmill ones
curl -s -u "$QUEUE_USER:$QUEUE_PASSWORD" "http://$QUEUE_HOST:15672/api/queues" | \
jq -r '.[].name' | \
while read queue; do
    if [ ! -z "$queue" ]; then
        echo "Deleting: $queue"
        curl -s -u "$QUEUE_USER:$QUEUE_PASSWORD" -X DELETE "http://$QUEUE_HOST:15672/api/queues/%2F/$queue"
    fi
done

echo "Done."