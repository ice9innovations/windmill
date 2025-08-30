#!/bin/bash

# Generic Worker Setup Script
# Helps configure and start a worker for a specific ML service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Generic Queue Worker Setup${NC}"
echo "This script helps you configure and start a worker for Animal Farm ML services."
echo

# Check if service name provided
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Usage: $0 <service_name> [worker_id]${NC}"
    echo
    echo "Available services:"
    python3 generic_producer.py --list-services | grep "^  " | head -12
    echo
    echo "Examples:"
    echo "  $0 colors                    # Start colors worker with auto-generated ID"
    echo "  $0 blip worker_blip_box2     # Start BLIP worker with specific ID"
    exit 1
fi

SERVICE_NAME=$1
WORKER_ID=${2:-"worker_${SERVICE_NAME}_$(hostname)"}

echo -e "${BLUE}📋 Configuration:${NC}"
echo "  Service: $SERVICE_NAME"
echo "  Worker ID: $WORKER_ID"
echo

# Check if service is valid
if ! python3 generic_producer.py --list-services | grep -q "^  $SERVICE_NAME"; then
    echo -e "${RED}❌ Error: Unknown service '$SERVICE_NAME'${NC}"
    echo
    echo "Available services:"
    python3 generic_producer.py --list-services | grep "^  " | head -12
    exit 1
fi

# Create .env file
ENV_FILE=".env"
EXAMPLE_FILE=".env.${SERVICE_NAME}.example"

echo -e "${YELLOW}🔧 Setting up configuration...${NC}"

# Use service-specific example if it exists
if [ -f "$EXAMPLE_FILE" ]; then
    echo "  Using template: $EXAMPLE_FILE"
    cp "$EXAMPLE_FILE" "$ENV_FILE"
else
    echo "  Using generic template: .env.example"
    cp ".env.example" "$ENV_FILE"
    
    # Update service name in generic template
    sed -i "s/SERVICE_NAME=.*/SERVICE_NAME=$SERVICE_NAME/" "$ENV_FILE"
fi

# Update worker ID
sed -i "s/WORKER_ID=.*/WORKER_ID=$WORKER_ID/" "$ENV_FILE"

echo -e "${GREEN}✅ Configuration file created: $ENV_FILE${NC}"
echo

# Check dependencies
echo -e "${YELLOW}📦 Checking dependencies...${NC}"
if ! python3 -c "import pika, psycopg2, requests, dotenv" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Installing missing dependencies...${NC}"
    pip3 install -r requirements.txt
    echo -e "${GREEN}✅ Dependencies installed${NC}"
else
    echo -e "${GREEN}✅ Dependencies already installed${NC}"
fi
echo

# Test configuration
echo -e "${YELLOW}🧪 Testing configuration...${NC}"
if python3 test_generic_system.py >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Configuration test passed${NC}"
else
    echo -e "${YELLOW}⚠️  Configuration test had warnings (continuing anyway)${NC}"
fi
echo

# Show next steps
echo -e "${BLUE}🎯 Ready to start worker!${NC}"
echo
echo "To start the worker:"
echo -e "  ${GREEN}python3 generic_worker.py${NC}"
echo
echo "To monitor progress:"
echo -e "  ${GREEN}# Load .env and run query${NC}"
echo -e "  ${GREEN}source .env && PGPASSWORD=\$DB_PASSWORD psql -h \$DB_HOST -U \$DB_USER -d \$DB_NAME -c \"SELECT service, COUNT(*) FROM results GROUP BY service;\"${NC}"
echo
echo "To submit test jobs:"
echo -e "  ${GREEN}python3 generic_producer.py --services $SERVICE_NAME --limit 10${NC}"
echo

# Ask if user wants to start worker now
read -p "Start the worker now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}🚀 Starting $SERVICE_NAME worker...${NC}"
    echo "Press Ctrl+C to stop"
    echo
    python3 generic_worker.py
else
    echo -e "${BLUE}👍 Worker ready! Run 'python3 generic_worker.py' when you're ready.${NC}"
fi