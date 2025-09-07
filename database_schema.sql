-- Complete Windmill Database Schema
-- Generated: 2025-09-07
-- Database: animal_farm_dev

-- =============================================================================
-- SEQUENCES
-- =============================================================================

CREATE SEQUENCE public.images_image_id_seq;
CREATE SEQUENCE public.results_result_id_seq;
CREATE SEQUENCE public.merged_boxes_merged_id_seq;
CREATE SEQUENCE public.consensus_consensus_id_seq;
CREATE SEQUENCE public.postprocessing_post_id_seq;

-- =============================================================================
-- IMAGES TABLE - Core image metadata and tracking
-- =============================================================================

CREATE TABLE public.images (
    image_id bigint NOT NULL DEFAULT nextval('images_image_id_seq'::regclass),
    image_filename character varying(255),
    image_path text,
    image_url text,
    image_group character varying(255),
    image_created timestamp without time zone DEFAULT now()
);

-- Primary key and indexes
ALTER TABLE ONLY public.images
    ADD CONSTRAINT images_pkey PRIMARY KEY (image_id);

CREATE INDEX idx_images_group ON public.images USING btree (image_group);

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

-- Primary key and indexes
ALTER TABLE ONLY public.results
    ADD CONSTRAINT results_pkey PRIMARY KEY (result_id);

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
    harmonized_data jsonb NOT NULL,
    bbox_created timestamp without time zone DEFAULT now()
);

-- Primary key and indexes
ALTER TABLE ONLY public.merged_boxes
    ADD CONSTRAINT merged_boxes_pkey PRIMARY KEY (merged_id);

CREATE INDEX idx_merged_boxes_created ON public.merged_boxes USING btree (bbox_created);
CREATE INDEX idx_merged_boxes_image ON public.merged_boxes USING btree (image_id);

-- Foreign keys
ALTER TABLE ONLY public.merged_boxes
    ADD CONSTRAINT merged_boxes_image_id_fkey FOREIGN KEY (image_id) REFERENCES public.images(image_id);

-- =============================================================================
-- CONSENSUS TABLE - Voting consensus results (DELETE+INSERT pattern)
-- =============================================================================

CREATE TABLE public.consensus (
    consensus_id bigint NOT NULL DEFAULT nextval('consensus_consensus_id_seq'::regclass),
    image_id bigint NOT NULL,
    consensus_data jsonb NOT NULL,
    consensus_created timestamp without time zone DEFAULT now()
);

-- Primary key and indexes
ALTER TABLE ONLY public.consensus
    ADD CONSTRAINT consensus_pkey PRIMARY KEY (consensus_id);

CREATE INDEX idx_consensus_created ON public.consensus USING btree (consensus_created);
CREATE INDEX idx_consensus_image ON public.consensus USING btree (image_id);

-- Foreign keys
ALTER TABLE ONLY public.consensus
    ADD CONSTRAINT consensus_image_id_fkey FOREIGN KEY (image_id) REFERENCES public.images(image_id);

-- =============================================================================
-- POSTPROCESSING TABLE - Bbox postprocessing results (INSERT-only pattern)
-- =============================================================================

CREATE TABLE public.postprocessing (
    post_id bigint NOT NULL DEFAULT nextval('postprocessing_post_id_seq'::regclass),
    merged_box_id bigint NOT NULL,
    image_id bigint NOT NULL,
    service character varying(255) NOT NULL,
    data jsonb,
    status character varying(20) NOT NULL,
    result_created timestamp without time zone DEFAULT now()
);

-- Primary key and indexes
ALTER TABLE ONLY public.postprocessing
    ADD CONSTRAINT postprocessing_pkey PRIMARY KEY (post_id);

CREATE INDEX idx_postprocessing_created ON public.postprocessing USING btree (result_created);
CREATE INDEX idx_postprocessing_image_service ON public.postprocessing USING btree (image_id, service);
CREATE INDEX idx_postprocessing_merged_box ON public.postprocessing USING btree (merged_box_id);

-- Foreign keys
ALTER TABLE ONLY public.postprocessing
    ADD CONSTRAINT postprocessing_image_id_fkey FOREIGN KEY (image_id) REFERENCES public.images(image_id);

ALTER TABLE ONLY public.postprocessing
    ADD CONSTRAINT postprocessing_merged_box_id_fkey FOREIGN KEY (merged_box_id) REFERENCES public.merged_boxes(merged_id);
