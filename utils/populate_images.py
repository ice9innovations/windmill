#!/usr/bin/env python3
"""
Populate the images table from webp directory
"""
import os
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

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

def get_webp_files(webp_dir):
    """Get all webp files from directory"""
    webp_path = Path(webp_dir)
    if not webp_path.exists():
        print(f"❌ Directory not found: {webp_dir}")
        return []
    
    webp_files = list(webp_path.glob("*.webp"))
    print(f"📁 Found {len(webp_files)} webp files")
    return webp_files

def populate_images_table(db_conn, webp_files):
    """Insert all images into the database"""
    cursor = db_conn.cursor()
    
    # First, let's see how many images are already in the table
    cursor.execute("SELECT COUNT(*) FROM images")
    existing_count = cursor.fetchone()[0]
    print(f"📊 Existing images in database: {existing_count}")
    
    inserted_count = 0
    batch_size = 1000
    
    for i in range(0, len(webp_files), batch_size):
        batch = webp_files[i:i + batch_size]
        
        # Prepare batch insert
        values = []
        for webp_file in batch:
            filename = webp_file.name
            local_path = str(webp_file)  # Local path (will be irrelevant for k1)
            url = f"http://peach.local/coco/webp/{filename}"
            image_group = "coco2017"
            
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
    webp_dir = "/home/sd/animal-farm/webp"
    
    print("🚀 Populating images table from webp directory")
    print(f"📂 Source directory: {webp_dir}")
    
    # Get webp files
    webp_files = get_webp_files(webp_dir)
    if not webp_files:
        return
    
    # Connect to database
    try:
        db_conn = connect_to_database()
        print("✅ Connected to PostgreSQL")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return
    
    # Populate table
    populate_images_table(db_conn, webp_files)
    
    # Close connection
    db_conn.close()
    print("✅ Database connection closed")

if __name__ == "__main__":
    main()