-- Windmill Database Schema
-- PostgreSQL schema for Animal Farm distributed ML processing pipeline
-- Exported from production database 2025-09-07

-- Enable pgvector extension for CLIP embedding storage and similarity search
CREATE EXTENSION IF NOT EXISTS vector;

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
    services_submitted TEXT[],
    image_created TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    -- CLIP image embedding (ViT-L/14, 768-dimensional, normalized)
    -- Written once on first caption score; used for image similarity search
    image_clip_embedding vector(768),
    -- Perceptual hash (pHash) for duplicate detection across formats/resolutions
    -- 16-char hex string; compare via Hamming distance (0-4 bits = same image)
    image_phash VARCHAR(16)
);

-- Create indexes for images table
CREATE INDEX idx_images_group ON images(image_group);
CREATE INDEX idx_images_filename ON images(image_filename);

-- HNSW index for fast approximate cosine similarity search across image embeddings
CREATE INDEX idx_images_clip_embedding ON images USING hnsw (image_clip_embedding vector_cosine_ops);
-- Index for fast exact pHash lookups
CREATE INDEX idx_images_phash ON images(image_phash);

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
    processing_time DOUBLE PRECISION,
    created TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),

    -- Live schema also has: framing_analysis JSONB, face_correlations JSONB, nsfw2_correlation JSONB
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

-- Verb Consensus table: Cross-VLM verb agreement and per-service SVO triples
-- No SAM3 integration — verbs have no spatial grounding.
CREATE TABLE IF NOT EXISTS verb_consensus (
    verb_consensus_id BIGSERIAL PRIMARY KEY,
    image_id          BIGINT NOT NULL REFERENCES images(image_id),
    verbs             JSONB  NOT NULL DEFAULT '[]', -- array of collapsed verb objects
    svo_triples       JSONB  NOT NULL DEFAULT '{}', -- {service: [[s,v,o], ...]} per-service reference
    services_present  TEXT[] NOT NULL DEFAULT '{}', -- which VLMs contributed
    service_count     INTEGER NOT NULL DEFAULT 0,
    processing_time   DOUBLE PRECISION,
    created_at        TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at        TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT verb_consensus_image_id_unique UNIQUE (image_id)
);

CREATE INDEX IF NOT EXISTS idx_verb_consensus_image ON verb_consensus(image_id);

-- Noun Consensus table: Cross-VLM noun synonym collapsing and vote counting
-- Progressive: upserted on each VLM completion; sam3_validated flags preserved across upserts.
CREATE TABLE IF NOT EXISTS noun_consensus (
    noun_consensus_id BIGSERIAL PRIMARY KEY,
    image_id          BIGINT NOT NULL REFERENCES images(image_id),
    nouns             JSONB  NOT NULL DEFAULT '[]', -- array of collapsed noun objects with sam3_validated flags
    category_tally    JSONB  NOT NULL DEFAULT '[]', -- [{category, vote_count, services, nouns:[...]}]
    services_present  TEXT[] NOT NULL DEFAULT '{}', -- which VLMs contributed
    service_count     INTEGER NOT NULL DEFAULT 0,
    processing_time   DOUBLE PRECISION,
    created_at        TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at        TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT noun_consensus_image_id_unique UNIQUE (image_id)
);

CREATE INDEX IF NOT EXISTS idx_noun_consensus_image ON noun_consensus(image_id);

-- Caption Summary table: LLM-synthesized single caption from all VLM outputs
-- Triggered by SAM3 completion; requires >= 2 VLM captions to write a row.
CREATE TABLE IF NOT EXISTS caption_summary (
    caption_summary_id BIGSERIAL PRIMARY KEY,
    image_id           BIGINT  NOT NULL REFERENCES images(image_id),
    summary_caption    TEXT    NOT NULL,
    model              TEXT    NOT NULL,
    services_present   TEXT[]  NOT NULL DEFAULT '{}',
    service_count      INTEGER NOT NULL DEFAULT 0,
    processing_time    DOUBLE PRECISION,
    created_at         TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at         TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT caption_summary_image_id_unique UNIQUE (image_id)
);

CREATE INDEX IF NOT EXISTS idx_caption_summary_image ON caption_summary(image_id);

-- Rembg Results table: Alpha mattes produced by the rembg background removal service
-- Written on-demand when a consumer requests a matte via ice9-api.
-- premasked=true means the SAM3 _subject mask was applied before rembg ran.
CREATE TABLE IF NOT EXISTS rembg_results (
    rembg_id        BIGSERIAL PRIMARY KEY,
    image_id        BIGINT NOT NULL REFERENCES images(image_id),
    png_b64         TEXT NOT NULL,           -- base64-encoded grayscale alpha matte PNG
    shape           JSONB NOT NULL,          -- [height, width]
    model           TEXT,                    -- model name reported by the rembg service
    premasked       BOOLEAN NOT NULL DEFAULT FALSE,
    processing_time DOUBLE PRECISION,
    created_at      TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at      TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT rembg_results_image_id_unique UNIQUE (image_id)
);

CREATE INDEX IF NOT EXISTS idx_rembg_results_image ON rembg_results(image_id);

-- Service Dispatch table: unified lifecycle tracking for all dispatched services.
-- Primary services written at submission time by api.py; secondary services (face/pose/sam3)
-- written at dispatch time by harmony_worker and noun_consensus_worker.
-- Status updated to 'complete' by workers when result is written.
-- 'stale' is computed at read time by api.py for pending rows past their age threshold.
-- Enables accurate per-service status without relying on result rows, which can't
-- distinguish "dispatched+no results" from "dispatched+crashed" from "never dispatched".
CREATE TABLE IF NOT EXISTS service_dispatch (
    dispatch_id   BIGSERIAL PRIMARY KEY,
    image_id      BIGINT NOT NULL REFERENCES images(image_id),
    service       TEXT NOT NULL,     -- service short name, e.g. 'blip', 'face', 'pose', 'sam3'
    cluster_id    TEXT,              -- bbox cluster_id for face/pose; NULL for image-level services
    dispatched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending, complete, failed, dead-lettered
    failed_reason TEXT                              -- error message or DLQ death reason; NULL on success
);

CREATE INDEX IF NOT EXISTS idx_service_dispatch_image ON service_dispatch (image_id);

-- Worker registry: tracks worker uptime history across all machines.
-- Workers INSERT on startup and UPDATE last_heartbeat while running.
-- status='offline' is written on clean shutdown or stale detection; offline_at records when.
-- Rows are never deleted — history is retained for pipeline gap diagnostics.
CREATE TABLE IF NOT EXISTS worker_registry (
    worker_id      TEXT PRIMARY KEY,           -- unique per process, e.g. worker_primary.yolo_v8_1772502927
    service        TEXT NOT NULL,              -- clean service name, e.g. 'yolo_v8', 'harmony', 'face'
    host           TEXT NOT NULL,              -- hostname of the machine running this worker
    started_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    offline_at     TIMESTAMPTZ,                -- set on clean shutdown or stale detection; NULL while online
    status         TEXT NOT NULL DEFAULT 'online'  -- 'online', 'offline'
);

CREATE INDEX IF NOT EXISTS idx_worker_registry_service ON worker_registry (service);
CREATE INDEX IF NOT EXISTS idx_worker_registry_status  ON worker_registry (status);