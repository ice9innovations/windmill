#!/usr/bin/env python3
"""
Add original/normalized image dimension columns to the images table.

Usage:
  python utils/migrate_original_image_dimensions_schema.py
"""
import os
import sys

import psycopg2
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()

    connect_kwargs = dict(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    if os.getenv("DB_SSLMODE"):
        connect_kwargs["sslmode"] = os.getenv("DB_SSLMODE")

    missing = [key for key in ("host", "database", "user", "password") if not connect_kwargs.get(key)]
    if missing:
        print(f"Missing required DB settings: {', '.join(missing)}")
        return 1

    statements = [
        """
        ALTER TABLE images
        ADD COLUMN IF NOT EXISTS original_image_width INTEGER
        """,
        """
        ALTER TABLE images
        ADD COLUMN IF NOT EXISTS original_image_height INTEGER
        """,
        """
        ALTER TABLE images
        ADD COLUMN IF NOT EXISTS normalized_image_width INTEGER
        """,
        """
        ALTER TABLE images
        ADD COLUMN IF NOT EXISTS normalized_image_height INTEGER
        """,
    ]

    conn = psycopg2.connect(**connect_kwargs)
    try:
        conn.autocommit = False
        cur = conn.cursor()
        for stmt in statements:
            cur.execute(stmt)
        conn.commit()
        cur.close()
        print("Added original/normalized image dimension columns to images table.")
        return 0
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
