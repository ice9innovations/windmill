#!/usr/bin/env python3
"""
Create the append-only service_events table used for internal/progressive service breadcrumbs.

Usage:
  python utils/migrate_service_events_schema.py
"""

import os
import sys

import psycopg2
from dotenv import load_dotenv


def required_env(primary_key, fallback_key=None):
    value = os.getenv(primary_key)
    if not value and fallback_key:
        value = os.getenv(fallback_key)
    if not value:
        names = primary_key if not fallback_key else f"{primary_key} or {fallback_key}"
        raise ValueError(f"Required environment variable {names} not set")
    return value


def main():
    load_dotenv(".env")
    conn = psycopg2.connect(
        dbname=required_env("DB_NAME", "POSTGRES_DB"),
        user=required_env("DB_USER", "POSTGRES_USER"),
        password=required_env("DB_PASSWORD", "POSTGRES_PASSWORD"),
        host=required_env("DB_HOST", "POSTGRES_HOST"),
        port=os.getenv("DB_PORT", os.getenv("POSTGRES_PORT", "5432")),
        sslmode=os.getenv("DB_SSLMODE"),
        connect_timeout=10,
    )
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS service_events (
            event_id BIGSERIAL PRIMARY KEY,
            image_id BIGINT NOT NULL REFERENCES images(image_id),
            service VARCHAR(255) NOT NULL,
            event_type VARCHAR(20) NOT NULL,
            source_service VARCHAR(255),
            source_stage VARCHAR(255),
            data JSONB,
            created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_service_events_image_service
        ON service_events(image_id, service)
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_service_events_created
        ON service_events(created_at)
        """
    )

    cur.close()
    conn.close()
    print("service_events schema ready")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Failed to migrate service_events schema: {e}", file=sys.stderr)
        sys.exit(1)
