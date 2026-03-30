#!/bin/bash
#
# Clear Database Script - Clears all active derived data tables while preserving images
# This script safely clears all ML processing results and derived data while keeping the base images table intact.
#

# Load environment variables
source .env

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🧹 Windmill Database Cleanup Script${NC}"
echo "=================================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ Error: .env file not found${NC}"
    echo "Please ensure .env file exists with database configuration"
    exit 1
fi

# Verify required environment variables
if [ -z "$DB_HOST" ] || [ -z "$DB_NAME" ] || [ -z "$DB_USER" ] || [ -z "$DB_PASSWORD" ]; then
    echo -e "${RED}❌ Error: Missing required database environment variables${NC}"
    echo "Required: DB_HOST, DB_NAME, DB_USER, DB_PASSWORD"
    exit 1
fi

echo "🎯 Target Database: $DB_USER@$DB_HOST/$DB_NAME"
echo ""

# Function to terminate stuck database connections
terminate_stuck_connections() {
    echo "🔍 Checking for stuck database connections..."
    
    # Get stuck connections (idle in transaction for more than 5 minutes)
    STUCK_PIDS=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c "
        SELECT pid FROM pg_stat_activity 
        WHERE state = 'idle in transaction' 
        AND state_change < NOW() - INTERVAL '5 minutes'
        AND pid != pg_backend_pid();
    " | tr -d ' ')
    
    if [ -n "$STUCK_PIDS" ]; then
        echo -e "${YELLOW}⚠️  Found stuck database connections, terminating...${NC}"
        for pid in $STUCK_PIDS; do
            if [ -n "$pid" ]; then
                echo "   Terminating connection $pid"
                PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "SELECT pg_terminate_backend($pid);" > /dev/null
            fi
        done
        echo -e "${GREEN}✅ Terminated stuck connections${NC}"
    else
        echo -e "${GREEN}✅ No stuck connections found${NC}"
    fi
}

# Function to clear database tables
clear_tables() {
    echo ""
    echo "🗑️  Clearing derived data tables..."
    echo "   (Preserving images table with $(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM images;" | tr -d ' ') images)"
    
    # Clear tables in order that respects foreign key constraints
    # postprocessing -> merged_boxes -> everything else
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
        DELETE FROM postprocessing;
        DELETE FROM merged_boxes;  
        DELETE FROM results;
    " > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Successfully cleared all derived data tables${NC}"
    else
        echo -e "${RED}❌ Error clearing tables${NC}"
        return 1
    fi
}

# Function to verify cleanup
verify_cleanup() {
    echo ""
    echo "📊 Verifying cleanup results..."
    
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c "
        SELECT 
            'images' as table_name, 
            COUNT(*) as count,
            CASE WHEN COUNT(*) > 0 THEN '✅ Preserved' ELSE '❌ Empty' END as status
        FROM images
        UNION ALL
        SELECT 
            'results' as table_name, 
            COUNT(*) as count,
            CASE WHEN COUNT(*) = 0 THEN '✅ Cleared' ELSE '❌ Has Data' END as status
        FROM results  
        UNION ALL
        SELECT 
            'merged_boxes' as table_name, 
            COUNT(*) as count,
            CASE WHEN COUNT(*) = 0 THEN '✅ Cleared' ELSE '❌ Has Data' END as status
        FROM merged_boxes
        UNION ALL
        SELECT 
            'postprocessing' as table_name, 
            COUNT(*) as count,
            CASE WHEN COUNT(*) = 0 THEN '✅ Cleared' ELSE '❌ Has Data' END as status
        FROM postprocessing
        ORDER BY table_name;
    "
}

# Main execution
echo "Starting database cleanup process..."
echo ""

# Step 1: Terminate any stuck connections
terminate_stuck_connections

# Step 2: Clear the tables
clear_tables
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Database cleanup failed${NC}"
    exit 1
fi

# Step 3: Verify results
verify_cleanup

echo ""
echo -e "${GREEN}🎉 Database cleanup completed successfully!${NC}"
echo "   • Images table preserved"
echo "   • All derived data tables cleared"
echo "   • Ready for fresh ML processing"
