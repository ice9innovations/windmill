-- Caption Summary Table
-- Stores a single synthesized caption per image, produced by calling an LLM
-- (Qwen via Ollama by default) with all VLM captions, noun consensus, and
-- verb consensus as input.
--
-- Populated by caption_summary_worker; triggered by SAM3 completion.
-- Not written if fewer than 2 VLM captions are present for the image.

CREATE TABLE IF NOT EXISTS caption_summary (
    caption_summary_id BIGSERIAL PRIMARY KEY,
    image_id           BIGINT  NOT NULL REFERENCES images(image_id),
    summary_caption    TEXT    NOT NULL,
    model              TEXT    NOT NULL,
    services_present   TEXT[]  NOT NULL DEFAULT '{}',
    service_count      INTEGER NOT NULL DEFAULT 0,
    created_at         TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    updated_at         TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
    CONSTRAINT caption_summary_image_id_unique UNIQUE (image_id)
);

CREATE INDEX IF NOT EXISTS idx_caption_summary_image ON caption_summary(image_id);
