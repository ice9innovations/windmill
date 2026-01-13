#!/usr/bin/env python3
"""
Populate the images table from val2017 directory
"""
import os
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

# Note: images must be populated from a URL to support cloud deployments
BASE_URL="http://192.168.0.121/val2017" # no trailing slash

def connect_to_database():
    """Connect to PostgreSQL using .env credentials"""
    # Load environment variables
    if not load_dotenv():
        raise ValueError("Could not load .env file")

    def get_required(key):
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value

    return psycopg2.connect(
        host=get_required('DB_HOST'),
        database=get_required('DB_NAME'),
        user=get_required('DB_USER'),
        password=get_required('DB_PASSWORD')
    )

def get_val_files(val_dir):
    """Get all image files from val2017 directory"""
    val_path = Path(val_dir)
    if not val_path.exists():
        print(f"❌ Directory not found: {val_dir}")
        return []

    # Get JPG/JPEG files only
    image_files = []
    for ext in ['*.jpg', '*.jpeg']:
        image_files.extend(val_path.glob(ext))

    print(f"📁 Found {len(image_files)} image files")
    return image_files

def populate_images_table(db_conn, image_files):
    """Insert all images into the database"""
    cursor = db_conn.cursor()

    # First, let's see how many images are already in the table
    cursor.execute("SELECT COUNT(*) FROM images")
    existing_count = cursor.fetchone()[0]
    print(f"📊 Existing images in database: {existing_count}")

    inserted_count = 0
    batch_size = 1000

    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i + batch_size]

        # Prepare batch insert
        values = []
        for image_file in batch:
            filename = image_file.name
            local_path = str(image_file)  # Local path
            url = f"{BASE_URL}/{filename}"
            image_group = "val2017"

            values.append((filename, local_path, url, image_group))

        # Batch insert (simple - no duplicate checking)
        insert_query = """
            INSERT INTO images (image_filename, image_path, image_url, image_group)
            VALUES %s
        """

        try:
            from psycopg2.extras import execute_values
            execute_values(cursor, insert_query, values, template=None)
            db_conn.commit()

            inserted_count += cursor.rowcount
            print(f"✅ Inserted batch {i//batch_size + 1}: {cursor.rowcount} new images")

        except Exception as e:
            print(f"❌ Error inserting batch {i//batch_size + 1}: {e}")
            db_conn.rollback()

    print(f"🎉 Total images inserted: {inserted_count}")

    # Final count
    cursor.execute("SELECT COUNT(*) FROM images")
    final_count = cursor.fetchone()[0]
    print(f"📊 Total images in database: {final_count}")

def main():
    val_dir = "/home/sd/animal-farm/val2017"

    print("🚀 Populating images table from val2017 directory")
    print(f"📂 Source directory: {val_dir}")

    # Get image files
    image_files = get_val_files(val_dir)
    if not image_files:
        return

    # Connect to database
    try:
        db_conn = connect_to_database()
        print("✅ Connected to PostgreSQL")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return

    # Populate table
    populate_images_table(db_conn, image_files)

    # Close connection
    db_conn.close()
    print("✅ Database connection closed")

if __name__ == "__main__":
    main()