#!/usr/bin/env python3
"""
Clear all processing results for a specific image group.
This deletes results, merged_boxes, consensus, and postprocessing for the group,
but keeps the images in the images table so they can be reprocessed.

Usage:
    python utils/clear_image_group.py <image_group>

Example:
    python utils/clear_image_group.py nudenet_test
"""
import os
import sys
import psycopg2
from dotenv import load_dotenv

def clear_image_group(image_group):
    """Clear all processing results for an image group"""
    load_dotenv()

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

    cursor = conn.cursor()

    # First, get count of images in this group
    cursor.execute("""
        SELECT COUNT(*) FROM images WHERE image_group = %s
    """, (image_group,))
    image_count = cursor.fetchone()[0]

    if image_count == 0:
        print(f"❌ No images found in group '{image_group}'")
        cursor.close()
        conn.close()
        return False

    print(f"📊 Found {image_count:,} images in group '{image_group}'")
    print()

    # Get current counts
    cursor.execute("""
        SELECT COUNT(*)
        FROM results r
        JOIN images i ON r.image_id = i.image_id
        WHERE i.image_group = %s
    """, (image_group,))
    results_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM merged_boxes mb
        JOIN images i ON mb.image_id = i.image_id
        WHERE i.image_group = %s
    """, (image_group,))
    merged_boxes_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM consensus c
        JOIN images i ON c.image_id = i.image_id
        WHERE i.image_group = %s
    """, (image_group,))
    consensus_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*)
        FROM postprocessing p
        JOIN images i ON p.image_id = i.image_id
        WHERE i.image_group = %s
    """, (image_group,))
    postprocessing_count = cursor.fetchone()[0]

    print(f"Current data for group '{image_group}':")
    print(f"  Results: {results_count:,}")
    print(f"  Merged boxes: {merged_boxes_count:,}")
    print(f"  Consensus: {consensus_count:,}")
    print(f"  Postprocessing: {postprocessing_count:,}")
    print()

    if results_count == 0 and merged_boxes_count == 0 and consensus_count == 0 and postprocessing_count == 0:
        print("✅ No data to clear - group is already clean")
        cursor.close()
        conn.close()
        return True

    # Confirm with user
    response = input(f"⚠️  This will DELETE all processing results for '{image_group}'. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Aborted")
        cursor.close()
        conn.close()
        return False

    print()
    print("🗑️  Clearing data...")

    # Delete in order respecting foreign keys
    # postprocessing -> consensus -> merged_boxes -> results

    if postprocessing_count > 0:
        cursor.execute("""
            DELETE FROM postprocessing p
            USING images i
            WHERE p.image_id = i.image_id
            AND i.image_group = %s
        """, (image_group,))
        deleted = cursor.rowcount
        print(f"  ✅ Deleted {deleted:,} postprocessing records")

    if consensus_count > 0:
        cursor.execute("""
            DELETE FROM consensus c
            USING images i
            WHERE c.image_id = i.image_id
            AND i.image_group = %s
        """, (image_group,))
        deleted = cursor.rowcount
        print(f"  ✅ Deleted {deleted:,} consensus records")

    if merged_boxes_count > 0:
        cursor.execute("""
            DELETE FROM merged_boxes mb
            USING images i
            WHERE mb.image_id = i.image_id
            AND i.image_group = %s
        """, (image_group,))
        deleted = cursor.rowcount
        print(f"  ✅ Deleted {deleted:,} merged_boxes records")

    if results_count > 0:
        cursor.execute("""
            DELETE FROM results r
            USING images i
            WHERE r.image_id = i.image_id
            AND i.image_group = %s
        """, (image_group,))
        deleted = cursor.rowcount
        print(f"  ✅ Deleted {deleted:,} results records")

    conn.commit()
    cursor.close()
    conn.close()

    print()
    print(f"✅ Successfully cleared all data for group '{image_group}'")
    print(f"📝 Images remain in database and can be reprocessed with:")
    print(f"   ./producer.sh --group {image_group} --limit <N>")

    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python clear_image_group.py <image_group>")
        print("\nExample:")
        print("  python clear_image_group.py nudenet_test")
        sys.exit(1)

    image_group = sys.argv[1]
    success = clear_image_group(image_group)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
