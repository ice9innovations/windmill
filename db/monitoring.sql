-- Current Windmill Monitoring Database Schema
-- MySQL schema documentation for existing monitoring tables
-- This documents the actual current state as of 2025-09-03

-- Use monitoring database
USE monitoring;

-- Worker Heartbeats Table: Current monitoring implementation
-- Tracks worker health and performance metrics across the distributed system
-- Current record count: 1920 records (from 2025-08-28 to 2025-09-03)
CREATE TABLE `worker_heartbeats` (
  `id` int NOT NULL AUTO_INCREMENT,
  `timestamp` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  `worker_id` varchar(50) DEFAULT NULL,
  `service_name` varchar(50) DEFAULT NULL,
  `node_hostname` varchar(50) DEFAULT NULL,
  `status` enum('alive','starting','stopping','error') DEFAULT NULL,
  `jobs_completed` int DEFAULT NULL,
  `jobs_per_minute` float DEFAULT NULL,
  `last_job_time` timestamp NULL DEFAULT NULL,
  `queue_depth` int DEFAULT NULL,
  `error_message` text,
  PRIMARY KEY (`id`),
  KEY `idx_worker_time` (`worker_id`,`timestamp`),
  KEY `idx_service_time` (`service_name`,`timestamp`)
) ENGINE=InnoDB AUTO_INCREMENT=1921 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Database: monitoring
-- Tables: 1 (worker_heartbeats only)
-- Views: None
-- Stored Procedures: None
-- Users: worker_monitor with full privileges