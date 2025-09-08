#!/usr/bin/env python3
"""
Import images into the images table from a CSV file.

CSV columns (header required):
  image_id, image_filename, image_url, image_group

Usage:
  DB_HOST=... DB_NAME=... DB_USER=... DB_PASSWORD=... \
  python utils/import_images_csv.py --csv /path/to/images.csv
"""

import os
import csv
import argparse
import psycopg2
from dotenv import load_dotenv


def import_csv(csv_path: str) -> int:
    load_dotenv()

    db_host = os.getenv('DB_HOST')
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    if not db_host or not db_name or not db_user or not db_password:
        print('Missing DB_HOST/DB_NAME/DB_USER/DB_PASSWORD in environment')
        return 1

    conn = psycopg2.connect(host=db_host, database=db_name, user=db_user, password=db_password)
    cur = conn.cursor()

    inserted = 0
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get('image_id')
            filename = row.get('image_filename')
            url = row.get('image_url')
            group = row.get('image_group')
            if not (filename and url):
                continue
            cur.execute(
                """
                INSERT INTO images (image_id, image_filename, image_url, image_group)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (image_id) DO NOTHING
                """,
                (int(image_id) if image_id else None, filename, url, group)
            )
            inserted += cur.rowcount

    conn.commit()
    cur.close()
    conn.close()

    print(f"Inserted {inserted} rows from {csv_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Import images from CSV into images table')
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    args = parser.parse_args()
    return import_csv(args.csv)


if __name__ == '__main__':
    raise SystemExit(main())


