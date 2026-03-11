-- Windmill Database Schema
-- PostgreSQL reference schema for Animal Farm distributed ML processing pipeline
-- Last synced against live database: 2026-03-05
--
-- !! WARNING: DO NOT RUN THIS FILE AGAINST A LIVE DATABASE !!
-- The DROP TABLE statements below use CASCADE and will destroy all data.
-- This file is a reference document only. Apply schema changes as individual
-- ALTER TABLE migrations on the live database.

-- Enable pgvector extension for CLIP embedding storage and similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing tables in dependency order (for clean reinstalls ONLY)
DROP TABLE IF EXISTS content_analysis CASCADE;
DROP TABLE IF EXISTS postprocessing CASCADE;
DROP TABLE IF EXISTS sam3_results CASCADE;
DROP TABLE IF EXISTS rembg_results CASCADE;
DROP TABLE IF EXISTS noun_consensus CASCADE;
DROP TABLE IF EXISTS verb_consensus CASCADE;
DROP TABLE IF EXISTS caption_summary CASCADE;
DROP TABLE IF EXISTS service_dispatch CASCADE;
DROP TABLE IF EXISTS consensus CASCADE;
DROP TABLE IF EXISTS merged_boxes CASCADE;
DROP TABLE IF EXISTS results CASCADE;
DROP TABLE IF EXISTS images CASCADE;
DROP TABLE IF EXISTS worker_registry CASCADE;
DROP TABLE IF EXISTS worker_availability_log CASCADE;
DROP TABLE IF EXISTS conceptnet_edges CASCADE;


-- ---------------------------------------------------------------------------
-- Core pipeline tables
-- ---------------------------------------------------------------------------

-- Images table: Source image metadata submitted for ML processing
CREATE TABLE images (
    image_id              BIGSERIAL PRIMARY KEY,
    image_filename        VARCHAR(255),
    image_path            TEXT,
    image_url             TEXT,
    image_group           VARCHAR(255),
    services_submitted    TEXT[],
    image_created         TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    -- CLIP image embedding (ViT-L/14, 768-dimensional, normalized)
    -- Written once on first caption score; used for image similarity search
    image_clip_embedding  vector(768),
    -- Customer tier at submission time; gates which downstream services fire
    tier                  VARCHAR(20) DEFAULT 'free'
);

CREATE INDEX idx_images_group ON images(image_group);
CREATE INDEX idx_images_filename ON images(image_filename);
-- HNSW index for fast approximate cosine similarity search across image embeddings
CREATE INDEX idx_images_clip_embedding ON images USING hnsw (image_clip_embedding vector_cosine_ops);


-- Results table: Raw ML service results per image (INSERT-only)
CREATE TABLE results (
    result_id        BIGSERIAL PRIMARY KEY,
    image_id         BIGINT NOT NULL REFERENCES images(image_id),
    service          VARCHAR(255) NOT NULL,
    data             JSONB NOT NULL,
    worker_id        VARCHAR(50),
    worker_hostname  VARCHAR(100),
    status           VARCHAR(20) NOT NULL,
    error_message    TEXT,
    retry_count      INTEGER DEFAULT 0,
    result_created   TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    processing_time  DOUBLE PRECISION
);

CREATE INDEX idx_results_created ON results(result_created);
CREATE INDEX idx_results_image_service ON results(image_id, service);
CREATE INDEX idx_results_status ON results(status);
CREATE INDEX idx_results_worker ON results(worker_id);


-- Merged boxes table: Harmonized bounding boxes from spatial services (DELETE+INSERT per image)
-- worker_id tracks which harmony_worker instance wrote the rows.
-- status is always 'success'; rows that fail harmonization are not written.
CREATE TABLE merged_boxes (
    merged_id          INTEGER PRIMARY KEY,  -- SERIAL, not BIGSERIAL (legacy)
    image_id           BIGINT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    source_result_ids  BIGINT[] NOT NULL,
    merged_data        JSONB NOT NULL,
    worker_id          VARCHAR(50),
    status             VARCHAR(20) NOT NULL DEFAULT 'success',
    created            TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_merged_boxes_image ON merged_boxes(image_id);
CREATE INDEX idx_merged_boxes_status ON merged_boxes(status);
CREATE INDEX idx_merged_boxes_worker ON merged_boxes(worker_id);


-- Consensus table: V3 voting consensus across all ML results (DELETE+INSERT per image)
CREATE TABLE consensus (
    consensus_id       BIGSERIAL PRIMARY KEY,
    image_id           BIGINT NOT NULL REFERENCES images(image_id),
    consensus_data     JSONB NOT NULL,
    consensus_created  TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    processing_time    DOUBLE PRECISION
);

CREATE INDEX idx_consensus_created ON consensus(consensus_created);
CREATE INDEX idx_consensus_image ON consensus(image_id);


-- Postprocessing table: Per-bbox and image-level postprocessing results (INSERT-only)
-- merged_box_id is NULL for image-level postprocessing (e.g. caption scoring).
CREATE TABLE postprocessing (
    post_id          BIGSERIAL PRIMARY KEY,
    image_id         BIGINT NOT NULL REFERENCES images(image_id),
    merged_box_id    BIGINT REFERENCES merged_boxes(merged_id),
    service          VARCHAR(255) NOT NULL,
    data             JSONB,
    status           VARCHAR(20) NOT NULL,
    result_created   TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    error_message    TEXT,
    processing_time  DOUBLE PRECISION,
    retry_count      INTEGER DEFAULT 0
);

CREATE INDEX idx_postprocessing_image_service ON postprocessing(image_id, service);
CREATE INDEX idx_postprocessing_merged_box ON postprocessing(merged_box_id);
CREATE INDEX idx_postprocessing_created ON postprocessing(result_created);


-- Content Analysis table: Semantic-spatial scene understanding (UPSERT per image)
CREATE TABLE content_analysis (
    analysis_id                BIGSERIAL PRIMARY KEY,
    image_id                   BIGINT NOT NULL REFERENCES images(image_id),

    -- Canonical schema: all analysis data in full_analysis JSONB
    -- Contains: anatomy_exposed, gender_breakdown, person_attributions,
    --           activity_analysis (scene_type, intimacy_level, activities, spatial_relationships),
    --           semantic_validations, gender_vote, vlm_hallucinations,
    --           framing_analysis, face_correlations, nsfw2_correlation,
    --           person_deduplication (raw_count, deduplicated_count, containments),
    --           keyword_extraction, spatial_gender_inference
    full_analysis              JSONB,

    -- Metadata
    analysis_version           TEXT,
    created                    TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    processing_time            DOUBLE PRECISION,

    UNIQUE(image_id)
);

CREATE INDEX idx_content_analysis_image ON content_analysis(image_id);
CREATE INDEX idx_content_analysis_created ON content_analysis(created);


-- Noun Consensus table: Cross-VLM noun synonym collapsing and category tally (UPSERT)
-- Progressive: upserted on each VLM completion; sam3_validated flags preserved across upserts.
CREATE TABLE IF NOT EXISTS noun_consensus (
    noun_consensus_id  BIGSERIAL PRIMARY KEY,
    image_id           BIGINT NOT NULL REFERENCES images(image_id),
    nouns              JSONB NOT NULL DEFAULT '[]',   -- collapsed noun objects with sam3_validated flags
    category_tally     JSONB NOT NULL DEFAULT '[]',   -- [{category, vote_count, services, nouns:[...]}]
    services_present   TEXT[] NOT NULL DEFAULT '{}',
    service_count      INTEGER NOT NULL DEFAULT 0,
    processing_time    DOUBLE PRECISION,
    created_at         TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at         TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT noun_consensus_image_id_unique UNIQUE (image_id)
);

CREATE INDEX IF NOT EXISTS idx_noun_consensus_image ON noun_consensus(image_id);


-- Verb Consensus table: Cross-VLM verb agreement and SVO triples (UPSERT)
-- No spatial grounding — verbs are not linked to bboxes.
CREATE TABLE IF NOT EXISTS verb_consensus (
    verb_consensus_id  BIGSERIAL PRIMARY KEY,
    image_id           BIGINT NOT NULL REFERENCES images(image_id),
    verbs              JSONB NOT NULL DEFAULT '[]',   -- array of collapsed verb objects
    svo_triples        JSONB NOT NULL DEFAULT '{}',   -- {service: [[s,v,o], ...]} per-service reference
    services_present   TEXT[] NOT NULL DEFAULT '{}',
    service_count      INTEGER NOT NULL DEFAULT 0,
    processing_time    DOUBLE PRECISION,
    created_at         TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at         TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT verb_consensus_image_id_unique UNIQUE (image_id)
);

CREATE INDEX IF NOT EXISTS idx_verb_consensus_image ON verb_consensus(image_id);


-- Caption Summary table: LLM-synthesized caption from all VLM outputs (UPSERT)
-- Triggered after noun/verb consensus; requires >= 2 VLM captions.
CREATE TABLE IF NOT EXISTS caption_summary (
    caption_summary_id  BIGSERIAL PRIMARY KEY,
    image_id            BIGINT NOT NULL REFERENCES images(image_id),
    summary_caption     TEXT NOT NULL,
    model               TEXT NOT NULL,
    services_present    TEXT[] NOT NULL DEFAULT '{}',
    service_count       INTEGER NOT NULL DEFAULT 0,
    processing_time     DOUBLE PRECISION,
    created_at          TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at          TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT caption_summary_image_id_unique UNIQUE (image_id)
);

CREATE INDEX IF NOT EXISTS idx_caption_summary_image ON caption_summary(image_id);


-- SAM3 Results table: SAM3 segmentation masks per detected noun (UPSERT)
-- id and image_id are INTEGER (not BIGINT) — legacy from original schema.
-- processing_time is NUMERIC(10,3) not DOUBLE PRECISION.
-- dispatched_face_pose tracks how many face/pose jobs were dispatched from SAM3 results.
CREATE TABLE IF NOT EXISTS sam3_results (
    id                    INTEGER PRIMARY KEY,
    image_id              INTEGER NOT NULL REFERENCES images(image_id),
    nouns_queried         TEXT[] NOT NULL,
    data                  JSONB NOT NULL,
    instance_count        INTEGER NOT NULL DEFAULT 0,
    processing_time       NUMERIC(10,3),
    created_at            TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at            TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    dispatched_face_pose  INTEGER DEFAULT 0,
    CONSTRAINT sam3_results_image_id_key UNIQUE (image_id)
);

CREATE INDEX IF NOT EXISTS idx_sam3_results_image_id ON sam3_results(image_id);


-- Rembg Results table: Alpha mattes from background removal (UPSERT)
-- Written on-demand when a consumer requests a matte via ice9-api.
-- premasked=true means the SAM3 subject mask was applied before rembg ran.
CREATE TABLE IF NOT EXISTS rembg_results (
    rembg_id         BIGSERIAL PRIMARY KEY,
    image_id         BIGINT NOT NULL REFERENCES images(image_id),
    png_b64          TEXT NOT NULL,      -- base64-encoded grayscale alpha matte PNG
    shape            JSONB NOT NULL,     -- [height, width]
    model            TEXT,              -- model name reported by rembg service
    premasked        BOOLEAN NOT NULL DEFAULT FALSE,
    processing_time  DOUBLE PRECISION,
    created_at       TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at       TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT rembg_results_image_id_unique UNIQUE (image_id)
);

CREATE INDEX IF NOT EXISTS idx_rembg_results_image ON rembg_results(image_id);


-- Service Dispatch table: Unified job lifecycle tracking for all dispatched services.
-- Primary services written at submission time by api.py.
-- Secondary services (face/pose/sam3) written at dispatch time by harmony/noun_consensus workers.
-- status: pending → complete | failed | dead-lettered
-- failed_reason: error message (workers) or x-death reason (dlq_worker); NULL on success.
-- Note: sequence and FK constraint names retain 'secondary_dispatch' prefix from original naming.
CREATE TABLE IF NOT EXISTS service_dispatch (
    dispatch_id    BIGSERIAL PRIMARY KEY,  -- sequence: secondary_dispatch_dispatch_id_seq
    image_id       BIGINT NOT NULL REFERENCES images(image_id),
    service        TEXT NOT NULL,          -- service short name, e.g. 'blip', 'face', 'sam3'
    cluster_id     TEXT,                   -- bbox cluster_id for face/pose; NULL for image-level
    dispatched_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    status         TEXT NOT NULL DEFAULT 'pending',  -- pending, complete, failed, dead-lettered
    failed_reason  TEXT                   -- error detail or DLQ death reason; NULL on success
);

CREATE INDEX IF NOT EXISTS idx_service_dispatch_image ON service_dispatch(image_id);


-- ---------------------------------------------------------------------------
-- Infrastructure and monitoring tables
-- ---------------------------------------------------------------------------

-- Worker Registry table: Live worker heartbeat tracking across all machines.
-- Workers INSERT on startup and UPDATE last_heartbeat while running.
-- Rows are never deleted — history is retained for pipeline gap diagnostics.
CREATE TABLE IF NOT EXISTS worker_registry (
    worker_id       TEXT PRIMARY KEY,   -- unique per process, e.g. 'yolo_v8_worker_1772502927'
    service         TEXT NOT NULL,      -- clean service name, e.g. 'yolo_v8', 'harmony'
    host            TEXT NOT NULL,
    started_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_heartbeat  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    offline_at      TIMESTAMP WITH TIME ZONE,  -- set on clean shutdown or stale sweep
    status          TEXT NOT NULL DEFAULT 'online'  -- 'online', 'offline'
);

CREATE INDEX IF NOT EXISTS idx_worker_registry_service ON worker_registry(service);
CREATE INDEX IF NOT EXISTS idx_worker_registry_status  ON worker_registry(status);


-- Worker Availability Log table: Time-series availability snapshots per service.
-- Written by availability_snapshot_worker on a regular interval.
CREATE TABLE IF NOT EXISTS worker_availability_log (
    log_id       BIGSERIAL PRIMARY KEY,
    recorded_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    service      TEXT NOT NULL,
    is_available BOOLEAN NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_wal_recorded_at ON worker_availability_log(recorded_at);
CREATE INDEX IF NOT EXISTS idx_wal_service     ON worker_availability_log(service);
CREATE INDEX IF NOT EXISTS idx_wal_service_at  ON worker_availability_log(service, recorded_at);


-- ---------------------------------------------------------------------------
-- Reference / bulk-loaded data tables
-- ---------------------------------------------------------------------------

-- ConceptNet Edges table: Subset of ConceptNet loaded for noun synonym collapsing.
-- Loaded via utils/load_conceptnet.sh; used at startup to build in-memory frozenset.
-- No primary key — data is immutable bulk-loaded reference data.
CREATE TABLE IF NOT EXISTS conceptnet_edges (
    relation   TEXT NOT NULL,
    start_uri  TEXT NOT NULL,
    end_uri    TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_cn_rel       ON conceptnet_edges(relation);
CREATE INDEX IF NOT EXISTS idx_cn_start_rel ON conceptnet_edges(start_uri, relation);
CREATE INDEX IF NOT EXISTS idx_cn_end_rel   ON conceptnet_edges(end_uri, relation);


