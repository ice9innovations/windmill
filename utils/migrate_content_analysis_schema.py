#!/usr/bin/env python3
"""
Migrate content_analysis table to canonical full_analysis schema.

Backfills anatomy_exposed, gender_breakdown, and person_attributions
from top-level columns into the full_analysis JSONB structure.

Related issue: content-analysis-duplicate-schema-writes.md
"""
import os
import sys
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(project_root, '.env'))

def connect_db():
    """Connect to PostgreSQL database"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        sslmode=os.getenv('DB_SSLMODE', 'prefer')
    )

def migrate_row(cursor, row):
    """Migrate a single row by adding missing fields to full_analysis"""
    image_id = row['image_id']
    full_analysis = row['full_analysis']

    # Skip if full_analysis is NULL (shouldn't happen based on our check)
    if not full_analysis:
        print(f"  SKIP: image {image_id} - full_analysis is NULL")
        return False

    # Check if already migrated
    if all(k in full_analysis for k in ['anatomy_exposed', 'gender_breakdown', 'person_attributions']):
        return False  # Already has all three fields

    # Add missing fields from top-level columns
    modified = False

    if 'anatomy_exposed' not in full_analysis:
        full_analysis['anatomy_exposed'] = row['anatomy_exposed']
        modified = True

    if 'gender_breakdown' not in full_analysis:
        full_analysis['gender_breakdown'] = row['gender_breakdown']
        modified = True

    if 'person_attributions' not in full_analysis:
        full_analysis['person_attributions'] = row['person_attributions']
        modified = True

    if not modified:
        return False

    # Update the row
    cursor.execute(
        "UPDATE content_analysis SET full_analysis = %s WHERE image_id = %s",
        (json.dumps(full_analysis), image_id)
    )

    return True

def main():
    print("Content Analysis Schema Migration")
    print("=" * 60)
    print()

    # Connect to database
    print("Connecting to database...")
    conn = connect_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Count total rows
    cursor.execute("SELECT COUNT(*) as total FROM content_analysis")
    total = cursor.fetchone()['total']
    print(f"Total rows: {total}")
    print()

    # Check how many need migration
    cursor.execute("""
        SELECT COUNT(*) as needs_migration
        FROM content_analysis
        WHERE full_analysis IS NOT NULL
        AND (
            full_analysis->>'anatomy_exposed' IS NULL
            OR full_analysis->>'gender_breakdown' IS NULL
            OR full_analysis->>'person_attributions' IS NULL
        )
    """)
    needs_migration = cursor.fetchone()['needs_migration']
    print(f"Rows needing migration: {needs_migration}")
    print()

    if needs_migration == 0:
        print("No migration needed. All rows already have the required fields.")
        cursor.close()
        conn.close()
        return

    # Confirm before proceeding
    response = input(f"Migrate {needs_migration} rows? [y/N]: ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        cursor.close()
        conn.close()
        return

    print()
    print("Starting migration...")
    print()

    # Create new cursor for fetching rows
    fetch_cursor = conn.cursor(cursor_factory=RealDictCursor)
    fetch_cursor.execute("""
        SELECT image_id, anatomy_exposed, gender_breakdown, person_attributions, full_analysis
        FROM content_analysis
        WHERE full_analysis IS NOT NULL
        ORDER BY image_id
    """)

    migrated = 0
    skipped = 0
    errors = 0

    for row in fetch_cursor:
        try:
            if migrate_row(cursor, row):
                migrated += 1
                if migrated % 100 == 0:
                    print(f"  Migrated {migrated} rows...")
                    conn.commit()  # Commit in batches
            else:
                skipped += 1
        except Exception as e:
            errors += 1
            print(f"  ERROR: image {row['image_id']} - {e}")
            conn.rollback()

    # Final commit
    conn.commit()

    print()
    print("=" * 60)
    print("Migration complete:")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped:  {skipped}")
    print(f"  Errors:   {errors}")
    print()

    # Verify migration
    cursor.execute("""
        SELECT COUNT(*) as still_needs_migration
        FROM content_analysis
        WHERE full_analysis IS NOT NULL
        AND (
            full_analysis->>'anatomy_exposed' IS NULL
            OR full_analysis->>'gender_breakdown' IS NULL
            OR full_analysis->>'person_attributions' IS NULL
        )
    """)
    still_needs = cursor.fetchone()['still_needs_migration']

    if still_needs > 0:
        print(f"WARNING: {still_needs} rows still need migration (check errors above)")
        sys.exit(1)
    else:
        print("✓ All rows migrated successfully")

    fetch_cursor.close()
    cursor.close()
    conn.close()

if __name__ == '__main__':
    main()
