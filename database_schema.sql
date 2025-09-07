-- Windmill Database Schema
-- Generated: 2025-09-06
-- Database: animal_farm

-- TABLES OVERVIEW
-- images: Core image metadata and tracking
-- results: ML service results for each image/service combination
-- merged_boxes: Harmonized bounding boxes from detection services
-- consensus: Voting consensus results across all services
-- postprocessing: Specialized processing results (face, pose, bbox_colors)

-- =============================================================================
-- IMAGES TABLE
-- =============================================================================

-- Note: Schema for images table not captured - would need explicit \d images command

-- =============================================================================
-- RESULTS TABLE - Core ML service results
-- =============================================================================

CREATE TABLE public.results (
    result_id bigint NOT NULL DEFAULT nextval('results_result_id_seq'::regclass),
    image_id bigint NOT NULL,
    service character varying(255) NOT NULL,
    data jsonb NOT NULL,
    worker_id character varying(50),
    worker_hostname character varying(100),
    status character varying(20) NOT NULL,
    error_message text,
    retry_count integer DEFAULT 0,
    result_created timestamp without time zone DEFAULT now(),
    processing_time double precision
);

-- Indexes
CREATE INDEX idx_results_created ON public.results USING btree (result_created);
CREATE INDEX idx_results_image_service ON public.results USING btree (image_id, service);
CREATE INDEX idx_results_status ON public.results USING btree (status);
CREATE INDEX idx_results_worker ON public.results USING btree (worker_id);

-- Foreign keys
ALTER TABLE ONLY public.results
    ADD CONSTRAINT results_image_id_fkey FOREIGN KEY (image_id) REFERENCES public.images(image_id);

-- =============================================================================
-- MERGED_BOXES TABLE - Harmonized bounding boxes (DELETE+INSERT pattern)
-- =============================================================================

CREATE TABLE public.merged_boxes (
    merged_id bigint NOT NULL DEFAULT nextval('merged_boxes_merged_id_seq'::regclass),
    image_id bigint NOT NULL,
    source_result_ids bigint[] NOT NULL,
    merged_data jsonb NOT NULL,
    worker_id character varying(50),
    status character varying(20) NOT NULL,
    created timestamp without time zone DEFAULT now()
);

-- Indexes
CREATE INDEX idx_merged_boxes_image ON public.merged_boxes USING btree (image_id);
CREATE INDEX idx_merged_boxes_status ON public.merged_boxes USING btree (status);
CREATE INDEX idx_merged_boxes_worker ON public.merged_boxes USING btree (worker_id);

-- Foreign keys
ALTER TABLE ONLY public.merged_boxes
    ADD CONSTRAINT merged_boxes_image_id_fkey FOREIGN KEY (image_id) REFERENCES public.images(image_id);

-- =============================================================================
-- CONSENSUS TABLE - Voting consensus results (DELETE+INSERT pattern)
-- =============================================================================

CREATE TABLE public.consensus (
    consensus_id bigint NOT NULL DEFAULT nextval('consensus_consensus_id_seq'::regclass),
    image_id bigint,
    consensus_data jsonb NOT NULL,
    consensus_created timestamp without time zone DEFAULT now(),
    processing_time double precision
);

-- Indexes
CREATE INDEX idx_consensus_created ON public.consensus USING btree (consensus_created);
CREATE INDEX idx_consensus_image_created ON public.consensus USING btree (image_id, consensus_created DESC);

-- Foreign keys
ALTER TABLE ONLY public.consensus
    ADD CONSTRAINT consensus_image_id_fkey FOREIGN KEY (image_id) REFERENCES public.images(image_id);

-- =============================================================================
-- POSTPROCESSING TABLE - Specialized processing results
-- =============================================================================

CREATE TABLE public.postprocessing (
    post_id bigint NOT NULL DEFAULT nextval('postprocessing_post_id_seq'::regclass),
    image_id bigint NOT NULL,
    merged_box_id bigint,
    service character varying(255) NOT NULL,
    data jsonb NOT NULL,
    status character varying(20) NOT NULL,
    error_message text,
    retry_count integer DEFAULT 0,
    result_created timestamp without time zone DEFAULT now(),
    processing_time double precision
);

-- Indexes
CREATE INDEX idx_postprocessing_image ON public.postprocessing USING btree (image_id);
CREATE INDEX idx_postprocessing_merged_box ON public.postprocessing USING btree (merged_box_id);
CREATE INDEX idx_postprocessing_status ON public.postprocessing USING btree (status);

-- Foreign keys
ALTER TABLE ONLY public.postprocessing
    ADD CONSTRAINT postprocessing_image_id_fkey FOREIGN KEY (image_id) REFERENCES public.images(image_id);
ALTER TABLE ONLY public.postprocessing
    ADD CONSTRAINT postprocessing_merged_box_id_fkey FOREIGN KEY (merged_box_id) REFERENCES public.merged_boxes(merged_id);

-- =============================================================================
-- KEY PROCESSING PATTERNS
-- =============================================================================

-- 1. RESULTS: INSERT-only pattern for service results
-- 2. MERGED_BOXES: DELETE+INSERT pattern for clean JOINs (removes old harmonizations)  
-- 3. CONSENSUS: DELETE+INSERT pattern for clean JOINs (removes old consensus)
-- 4. POSTPROCESSING: INSERT-only pattern for parallel worker safety

-- =============================================================================
-- FOREIGN KEY DEPENDENCY ORDER FOR DELETIONS
-- =============================================================================

-- 1. Delete postprocessing (references merged_boxes)
-- 2. Delete consensus (references images only)  
-- 3. Delete merged_boxes (references images only)
-- 4. Delete results (references images only)
-- 5. Delete images (root table)