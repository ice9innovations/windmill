#!/usr/bin/env python3
"""
Add targeted service_dispatch indexes for pending-row updates.

Usage:
  python utils/migrate_service_dispatch_indexes.py
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
        CREATE INDEX IF NOT EXISTS idx_service_dispatch_pending_image_service_nullcluster
        ON service_dispatch (image_id, service)
        WHERE cluster_id IS NULL AND status = 'pending'
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_service_dispatch_pending_image_service_cluster
        ON service_dispatch (image_id, service, cluster_id)
        WHERE status = 'pending'
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
        print("Added targeted pending-row indexes to service_dispatch.")
        return 0
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
