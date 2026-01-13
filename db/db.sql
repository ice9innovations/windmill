-- Windmill Database Schema
-- PostgreSQL schema for Animal Farm distributed ML processing pipeline
-- Exported from production database 2025-09-07

-- Drop existing tables in dependency order (for clean reinstalls)
DROP TABLE IF EXISTS content_analysis CASCADE;
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

-- Postprocessing table: Postprocessing results on cropped bbox regions and image-level analysis
CREATE TABLE postprocessing (
    post_id BIGSERIAL PRIMARY KEY,
    image_id BIGINT NOT NULL,
    merged_box_id BIGINT,  -- NULL for image-level analysis (e.g. caption scoring)
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

-- Content Analysis table: Semantic-spatial content understanding and scene classification
CREATE TABLE content_analysis (
    analysis_id BIGSERIAL PRIMARY KEY,
    image_id BIGINT NOT NULL,

    -- Gender breakdown
    gender_breakdown JSONB,  -- {female_nudity: bool, male_nudity: bool, mixed_gender: bool, confidence: {female: float, male: float}}

    -- Anatomy detected
    anatomy_exposed TEXT[],  -- ['female_breast', 'male_genitalia', 'buttocks', ...]

    -- Scene classification
    scene_type TEXT,  -- 'simple_nudity', 'sexually_explicit', 'artistic_nudity', 'softcore_pornography', 'breastfeeding'
    intimacy_level TEXT,  -- 'solo', 'intimate', 'explicit_sexual', 'group'

    -- Activity detection
    activities_detected TEXT[],  -- ['sexual_intercourse', 'oral_sex', 'ffm_threesome', 'breastfeeding', ...]

    -- Spatial relationships
    spatial_relationships JSONB,  -- [{type: 'genital_overlap', bbox1: ..., bbox2: ..., iou: float}]

    -- Person deduplication
    person_bboxes_raw INT,  -- Original person bbox count before deduplication
    person_bboxes_deduplicated INT,  -- After removing containment duplicates
    containment_relationships JSONB,  -- [{contained_bbox: ..., containing_bbox: ..., confidence: ...}]

    -- Semantic validation
    semantic_validation JSONB,  -- {corroborated: bool, conflicts: [...], confidence: float}
    vlm_hallucinations JSONB,  -- [{vlm: 'ollama', hallucinated_gender: 'male', spatial_evidence: 'female', ...}]

    -- People analysis
    people_count INT,  -- Deduplicated count (actual number of distinct people)
    person_attributions JSONB,  -- [{bbox_ids: [1,2], gender: 'female', spatial_markers: [...], semantic_agreement: [...]}]

    -- Full analysis output
    full_analysis JSONB,

    -- Metadata
    analysis_version TEXT,
    created TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),

    UNIQUE(image_id)
);

-- Create indexes for content_analysis table
CREATE INDEX idx_content_analysis_scene_type ON content_analysis(scene_type);
CREATE INDEX idx_content_analysis_intimacy ON content_analysis(intimacy_level);
CREATE INDEX idx_content_analysis_image ON content_analysis(image_id);
CREATE INDEX idx_content_analysis_created ON content_analysis(created);

-- Create foreign key constraint for content_analysis
ALTER TABLE content_analysis ADD CONSTRAINT content_analysis_image_id_fkey
    FOREIGN KEY (image_id) REFERENCES images(image_id);