-- Verb Consensus Table
-- Stores cross-VLM verb agreement results and per-service SVO triples for each image.
-- Populated by verb_consensus_worker; updated progressively as VLMs complete.
-- No SAM3 integration — verbs have no spatial grounding.

CREATE TABLE IF NOT EXISTS verb_consensus (
    verb_consensus_id BIGSERIAL PRIMARY KEY,
    image_id          BIGINT NOT NULL REFERENCES images(image_id),
    verbs             JSONB  NOT NULL DEFAULT '[]', -- array of collapsed verb objects
    svo_triples       JSONB  NOT NULL DEFAULT '{}', -- {service: [[s,v,o], ...]} per-service reference
    services_present  TEXT[] NOT NULL DEFAULT '{}', -- which VLMs contributed
    service_count     INTEGER NOT NULL DEFAULT 0,
    created_at        TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at        TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT verb_consensus_image_id_unique UNIQUE (image_id)
);

CREATE INDEX IF NOT EXISTS idx_verb_consensus_image ON verb_consensus(image_id);
