-- Windmill Monitoring Database Schema for PostgreSQL
-- Separate monitoring database to isolate monitoring queries from ML processing
-- Migrated from MySQL monitoring database to eliminate dual-database platform overhead
-- Based on current MySQL worker_heartbeats table structure

-- Create monitoring database (run as superuser)
-- CREATE DATABASE monitoring OWNER animal_farm_user;

-- Worker heartbeats table: Track worker health and performance
CREATE TABLE worker_heartbeats (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    worker_id VARCHAR(50),
    service_name VARCHAR(50),
    node_hostname VARCHAR(50),
    status VARCHAR(20) CHECK (status IN ('alive', 'starting', 'stopping', 'error')),
    jobs_completed INTEGER,
    jobs_per_minute REAL,
    last_job_time TIMESTAMP,
    queue_depth INTEGER,
    error_message TEXT
);

-- Indexes for monitoring queries
CREATE INDEX idx_worker_heartbeats_worker_time ON worker_heartbeats(worker_id, timestamp);
CREATE INDEX idx_worker_heartbeats_service_time ON worker_heartbeats(service_name, timestamp);
CREATE INDEX idx_worker_heartbeats_hostname_time ON worker_heartbeats(node_hostname, timestamp);
CREATE INDEX idx_worker_heartbeats_status_time ON worker_heartbeats(status, timestamp);

-- Grant permissions to application user
GRANT ALL PRIVILEGES ON worker_heartbeats TO animal_farm_user;
GRANT ALL PRIVILEGES ON SEQUENCE worker_heartbeats_id_seq TO animal_farm_user;