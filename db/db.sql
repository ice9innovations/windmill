-- Windmill Database Schema
-- PostgreSQL schema for Animal Farm distributed ML processing pipeline
-- Creates tables for image processing, ML results, bbox harmonization, and consensus

-- Drop existing tables in dependency order (for clean reinstalls)
DROP TABLE IF EXISTS postprocessing CASCADE;
DROP TABLE IF EXISTS consensus CASCADE;
DROP TABLE IF EXISTS merged_boxes CASCADE;
DROP TABLE IF EXISTS results CASCADE;
DROP TABLE IF EXISTS images CASCADE;

-- Images table: Source images for ML processing
CREATE TABLE images (
    image_id SERIAL PRIMARY KEY,
    image_filename VARCHAR(255) NOT NULL,
    image_path TEXT,
    image_url TEXT,
    image_created TIMESTAMP DEFAULT NOW(),
    image_group VARCHAR(255),  -- Used by populate_images.py for grouping (e.g. 'coco2017')
    
    -- Ensure either path or URL is provided
    CONSTRAINT check_image_source CHECK (image_path IS NOT NULL OR image_url IS NOT NULL)
);

-- Results table: ML service results for each image
CREATE TABLE results (
    result_id SERIAL PRIMARY KEY,
    image_id INT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    service VARCHAR(50) NOT NULL,
    data JSONB NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    processing_time FLOAT,
    result_created TIMESTAMP DEFAULT NOW(),
    worker_id VARCHAR(50),
    
    -- Ensure valid status values
    CONSTRAINT check_status CHECK (status IN ('pending', 'processing', 'success', 'error', 'timeout'))
);

-- Merged boxes table: Harmonized bounding boxes from detection services
-- Uses DELETE+INSERT pattern for atomic updates during re-harmonization
CREATE TABLE merged_boxes (
    merged_id SERIAL PRIMARY KEY,
    image_id INT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    source_result_ids INT[] NOT NULL,  -- Array of result_ids that contributed to this merge
    merged_data JSONB NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'success',
    created TIMESTAMP DEFAULT NOW(),
    worker_id VARCHAR(50),
    
    -- Ensure non-empty source array
    CONSTRAINT check_source_results CHECK (array_length(source_result_ids, 1) > 0)
);

-- Consensus table: V3 voting algorithm results across ALL services
-- Uses DELETE+INSERT pattern for atomic updates during re-consensus
CREATE TABLE consensus (
    consensus_id SERIAL PRIMARY KEY,
    image_id INT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    consensus_data JSONB NOT NULL,
    processing_time FLOAT,
    consensus_created TIMESTAMP DEFAULT NOW(),
    worker_id VARCHAR(50)
);

-- Postprocessing table: Spatial analysis results on cropped bounding boxes
-- Uses INSERT-only pattern for parallel worker safety
CREATE TABLE postprocessing (
    postprocessing_id SERIAL PRIMARY KEY,
    image_id INT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    merged_box_id INT REFERENCES merged_boxes(merged_id) ON DELETE CASCADE,
    service VARCHAR(50) NOT NULL,  -- 'colors', 'face', 'pose', 'caption_score_blip', etc.
    data JSONB NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'success',
    result_created TIMESTAMP DEFAULT NOW(),
    worker_id VARCHAR(50),
    
    -- Ensure valid postprocessing services
    CONSTRAINT check_postprocessing_service CHECK (
        service IN ('colors', 'face', 'pose', 'caption_score_blip', 'caption_score_ollama')
    )
);

-- Performance Indexes
-- Core query patterns for workers and monitoring

-- Results table indexes
CREATE INDEX idx_results_image_service ON results(image_id, service);
CREATE INDEX idx_results_service_status ON results(service, status);
CREATE INDEX idx_results_created ON results(result_created);
CREATE INDEX idx_results_status ON results(status);

-- Merged boxes indexes  
CREATE INDEX idx_merged_boxes_image ON merged_boxes(image_id);
CREATE INDEX idx_merged_boxes_created ON merged_boxes(created);

-- Consensus indexes
CREATE INDEX idx_consensus_image ON consensus(image_id);
CREATE INDEX idx_consensus_created ON consensus(consensus_created);

-- Postprocessing indexes
CREATE INDEX idx_postprocessing_image_service ON postprocessing(image_id, service);
CREATE INDEX idx_postprocessing_merged_box ON postprocessing(merged_box_id);
CREATE INDEX idx_postprocessing_service ON postprocessing(service);
CREATE INDEX idx_postprocessing_created ON postprocessing(result_created);

-- Images indexes
CREATE INDEX idx_images_filename ON images(image_filename);
CREATE INDEX idx_images_created ON images(image_created);



-- Grant permissions for application user
-- Note: Replace 'animal_farm_user' with your actual database user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO animal_farm_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO animal_farm_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO animal_farm_user;

-- Schema creation confirmation
DO $$ 
BEGIN
    RAISE NOTICE 'Windmill PostgreSQL database schema created successfully!';
    RAISE NOTICE 'Tables: images, results, merged_boxes, consensus, postprocessing';
    RAISE NOTICE 'Worker coordination: timestamp-based polling (no triggers)';
END $$;