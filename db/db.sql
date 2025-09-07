-- Windmill Database Schema
-- PostgreSQL schema for Animal Farm distributed ML processing pipeline
-- Exported from production database 2025-09-07

-- Drop existing tables in dependency order (for clean reinstalls)
DROP TABLE IF EXISTS postprocessing CASCADE;
DROP TABLE IF EXISTS consensus CASCADE;
DROP TABLE IF EXISTS merged_boxes CASCADE;
DROP TABLE IF EXISTS results CASCADE;
DROP TABLE IF EXISTS images CASCADE;

-- Images table: Source images for ML processing
CREATE TABLE images (
    image_id BIGSERIAL PRIMARY KEY,
    image_filename VARCHAR(255),
    image_path TEXT,
    image_url TEXT,
    image_group VARCHAR(255),
    image_created TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

-- Create indexes for images table
CREATE INDEX idx_images_group ON images(image_group);

-- Results table: ML service results for each image
CREATE TABLE results (
    result_id BIGSERIAL PRIMARY KEY,
    image_id BIGINT NOT NULL,
    service VARCHAR(255) NOT NULL,
    data JSONB NOT NULL,
    worker_id VARCHAR(50),
    worker_hostname VARCHAR(100),
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    result_created TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    processing_time DOUBLE PRECISION
);

-- Create indexes for results table
CREATE INDEX idx_results_created ON results(result_created);
CREATE INDEX idx_results_image_service ON results(image_id, service);
CREATE INDEX idx_results_status ON results(status);
CREATE INDEX idx_results_worker ON results(worker_id);

-- Create foreign key constraint for results
ALTER TABLE results ADD CONSTRAINT results_image_id_fkey 
    FOREIGN KEY (image_id) REFERENCES images(image_id);

-- Merged boxes table: Harmonized bounding box results
CREATE TABLE merged_boxes (
    merged_id BIGSERIAL PRIMARY KEY,
    image_id BIGINT NOT NULL,
    source_result_ids BIGINT[],
    merged_data JSONB NOT NULL,
    created TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    processing_time DOUBLE PRECISION
);

-- Create indexes for merged_boxes table
CREATE INDEX idx_merged_boxes_image ON merged_boxes(image_id);
CREATE INDEX idx_merged_boxes_created ON merged_boxes(created);

-- Create foreign key constraint for merged_boxes
ALTER TABLE merged_boxes ADD CONSTRAINT merged_boxes_image_id_fkey 
    FOREIGN KEY (image_id) REFERENCES images(image_id);

-- Consensus table: Voting consensus results across all services
CREATE TABLE consensus (
    consensus_id BIGSERIAL PRIMARY KEY,
    image_id BIGINT,
    consensus_data JSONB NOT NULL,
    consensus_created TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    processing_time DOUBLE PRECISION
);

-- Create indexes for consensus table
CREATE INDEX idx_consensus_created ON consensus(consensus_created);
CREATE INDEX idx_consensus_image_created ON consensus(image_id, consensus_created DESC);

-- Create foreign key constraint for consensus
ALTER TABLE consensus ADD CONSTRAINT consensus_image_id_fkey 
    FOREIGN KEY (image_id) REFERENCES images(image_id);

-- Postprocessing table: Postprocessing results on cropped bbox regions
CREATE TABLE postprocessing (
    post_id BIGSERIAL PRIMARY KEY,
    image_id BIGINT NOT NULL,
    merged_box_id BIGINT,
    service VARCHAR(255) NOT NULL,
    data JSONB NOT NULL,
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    result_created TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    processing_time DOUBLE PRECISION
);

-- Create indexes for postprocessing table
CREATE INDEX idx_postprocessing_image ON postprocessing(image_id);
CREATE INDEX idx_postprocessing_merged_box ON postprocessing(merged_box_id);
CREATE INDEX idx_postprocessing_status ON postprocessing(status);

-- Create foreign key constraints for postprocessing
ALTER TABLE postprocessing ADD CONSTRAINT postprocessing_image_id_fkey 
    FOREIGN KEY (image_id) REFERENCES images(image_id);
ALTER TABLE postprocessing ADD CONSTRAINT postprocessing_merged_box_id_fkey 
    FOREIGN KEY (merged_box_id) REFERENCES merged_boxes(merged_id);