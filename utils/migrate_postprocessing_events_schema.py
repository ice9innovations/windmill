#!/usr/bin/env python3
"""Create the postprocessing_events table used for append-only child-work breadcrumbs."""

import os
import psycopg2


def _connect():
    kwargs = {
        "host": os.environ["DB_HOST"],
        "database": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
    }
    sslmode = os.getenv("DB_SSLMODE")
    if sslmode:
        kwargs["sslmode"] = sslmode
    return psycopg2.connect(**kwargs)


SQL = [
    """
    CREATE TABLE IF NOT EXISTS postprocessing_events (
        event_id         BIGSERIAL PRIMARY KEY,
        image_id         BIGINT NOT NULL REFERENCES images(image_id),
        merged_box_id    BIGINT REFERENCES merged_boxes(merged_id),
        service          VARCHAR(255) NOT NULL,
        cluster_id       TEXT NOT NULL,
        event_type       VARCHAR(20) NOT NULL,
        source_service   VARCHAR(255),
        source_stage     VARCHAR(255),
        data             JSONB,
        created_at       TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_postprocessing_events_image_service
        ON postprocessing_events(image_id, service)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_postprocessing_events_image_cluster
        ON postprocessing_events(image_id, cluster_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_postprocessing_events_created
        ON postprocessing_events(created_at)
    """,
]


def main():
    conn = _connect()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            for statement in SQL:
                cur.execute(statement)
        print("postprocessing_events schema ready")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
