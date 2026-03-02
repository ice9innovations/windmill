-- Noun Consensus Table
-- Stores cross-VLM noun agreement results for each image.
-- Populated by noun_consensus_worker; updated progressively as VLMs complete.

CREATE TABLE IF NOT EXISTS noun_consensus (
    noun_consensus_id BIGSERIAL PRIMARY KEY,
    image_id          BIGINT NOT NULL REFERENCES images(image_id),
    nouns             JSONB  NOT NULL,          -- array of collapsed noun objects
    services_present  TEXT[] NOT NULL DEFAULT '{}', -- which VLMs contributed
    service_count     INTEGER NOT NULL DEFAULT 0,
    created_at        TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at        TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT noun_consensus_image_id_unique UNIQUE (image_id)
);

CREATE INDEX IF NOT EXISTS idx_noun_consensus_image ON noun_consensus(image_id);
