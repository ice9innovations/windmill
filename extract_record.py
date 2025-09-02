#!/usr/bin/env python3
"""
Extract Record Script - Extracts complete ML processing record for a single image
This script pulls all related data from the database for a specific image_id and creates a comprehensive JSON file for review.
"""

import os
import sys
import json
import psycopg2
import argparse
from datetime import datetime
from dotenv import load_dotenv

class RecordExtractor:
    """Extract complete processing record for a single image"""
    
    def __init__(self):
        # Load environment variables
        if not load_dotenv('.env'):
            raise ValueError("Could not load .env file. Please ensure .env exists with database configuration.")
        
        # Database configuration
        self.db_host = self._get_required('DB_HOST')
        self.db_name = self._get_required('DB_NAME')
        self.db_user = self._get_required('DB_USER')
        self.db_password = self._get_required('DB_PASSWORD')
        
        self.conn = None
    
    def _get_required(self, key):
        """Get required environment variable"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not found in .env file")
        return value
    
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            print(f"✅ Connected to {self.db_user}@{self.db_host}/{self.db_name}")
        except Exception as e:
            raise Exception(f"Failed to connect to database: {e}")
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def get_image_info(self, image_id):
        """Get basic image information"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                SELECT image_id, image_filename, image_path, image_url, image_group, image_created
                FROM images 
                WHERE image_id = %s
            """, (image_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
                
            return {
                'image_id': row[0],
                'image_filename': row[1],
                'image_path': row[2],
                'image_url': row[3],
                'image_group': row[4],
                'image_created': row[5].isoformat() if row[5] else None
            }
    
    def get_ml_results(self, image_id):
        """Get all ML service results for the image"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                SELECT result_id, service, data, status, processing_time, 
                       result_created, worker_id, error_message
                FROM results 
                WHERE image_id = %s
                ORDER BY service, result_created
            """, (image_id,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'result_id': row[0],
                    'service': row[1],
                    'data': row[2],  # Already JSON from JSONB column
                    'status': row[3],
                    'processing_time': float(row[4]) if row[4] else None,
                    'result_created': row[5].isoformat() if row[5] else None,
                    'worker_id': row[6],
                    'error_message': row[7]
                })
            
            return results
    
    def get_merged_boxes(self, image_id):
        """Get harmonized bounding box results (filters out single-service boxes)"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                SELECT merged_id, source_result_ids, merged_data, status, 
                       created, worker_id
                FROM merged_boxes 
                WHERE image_id = %s
                ORDER BY created
            """, (image_id,))
            
            merged_boxes = []
            for row in cursor.fetchall():
                merged_data = row[2]  # Already JSON from JSONB column
                
                # Filter out single-service merged boxes (detection_count < 2)
                detection_count = merged_data.get('detection_count', 0)
                if detection_count < 2:
                    continue  # Skip single-service boxes
                
                merged_boxes.append({
                    'merged_id': row[0],
                    'source_result_ids': row[1],  # Array of result IDs
                    'merged_data': merged_data,
                    'status': row[3],
                    'created': row[4].isoformat() if row[4] else None,
                    'worker_id': row[5]
                })
            
            return merged_boxes
    
    def get_consensus(self, image_id):
        """Get voting consensus results"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                SELECT consensus_id, consensus_data, processing_time, 
                       consensus_created
                FROM consensus 
                WHERE image_id = %s
                ORDER BY consensus_created
            """, (image_id,))
            
            consensus_results = []
            for row in cursor.fetchall():
                consensus_results.append({
                    'consensus_id': row[0],
                    'consensus_data': row[1],  # Already JSON from JSONB column
                    'processing_time': float(row[2]) if row[2] else None,
                    'consensus_created': row[3].isoformat() if row[3] else None
                })
            
            return consensus_results
    
    def get_spatial_enrichments(self, image_id):
        """Get spatial enrichment results (face/pose/colors on bounding boxes)"""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                SELECT post_id, merged_box_id, service, data, 
                       status, result_created, error_message, processing_time
                FROM postprocessing 
                WHERE image_id = %s
                ORDER BY merged_box_id, service, result_created
            """, (image_id,))
            
            enrichments = []
            for row in cursor.fetchall():
                enrichments.append({
                    'post_id': row[0],
                    'merged_box_id': row[1],
                    'service': row[2],
                    'data': row[3],  # Already JSON from JSONB column
                    'status': row[4],
                    'result_created': row[5].isoformat() if row[5] else None,
                    'error_message': row[6],
                    'processing_time': float(row[7]) if row[7] else None
                })
            
            return enrichments
    
    def get_processing_summary(self, image_id):
        """Get processing summary statistics"""
        with self.conn.cursor() as cursor:
            # Count results by service and status
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT service) as total_services,
                    COUNT(*) FILTER (WHERE status = 'success') as successful_results,
                    COUNT(*) FILTER (WHERE status = 'error') as failed_results,
                    COUNT(*) as total_results,
                    MIN(result_created) as first_result,
                    MAX(result_created) as last_result
                FROM results 
                WHERE image_id = %s
            """, (image_id,))
            
            row = cursor.fetchone()
            if not row:
                return {}
            
            # Get service breakdown
            cursor.execute("""
                SELECT service, status, COUNT(*) as count
                FROM results 
                WHERE image_id = %s
                GROUP BY service, status
                ORDER BY service, status
            """, (image_id,))
            
            service_breakdown = {}
            for service_row in cursor.fetchall():
                service = service_row[0]
                status = service_row[1]
                count = service_row[2]
                
                if service not in service_breakdown:
                    service_breakdown[service] = {}
                service_breakdown[service][status] = count
            
            return {
                'total_services': row[0],
                'successful_results': row[1],
                'failed_results': row[2],
                'total_results': row[3],
                'first_result': row[4].isoformat() if row[4] else None,
                'last_result': row[5].isoformat() if row[5] else None,
                'service_breakdown': service_breakdown
            }
    
    def extract_complete_record(self, image_id):
        """Extract complete processing record for an image"""
        print(f"🔍 Extracting complete record for image_id: {image_id}")
        
        # Get image info
        image_info = self.get_image_info(image_id)
        if not image_info:
            raise ValueError(f"Image with ID {image_id} not found in database")
        
        print(f"📷 Found image: {image_info['image_filename']}")
        
        # Extract all related data
        record = {
            'extraction_info': {
                'extracted_at': datetime.now().isoformat(),
                'image_id': image_id,
                'extractor': 'windmill-extract-record-script'
            },
            'image': image_info,
            'ml_results': self.get_ml_results(image_id),
            'merged_boxes': self.get_merged_boxes(image_id),
            'consensus': self.get_consensus(image_id),
            'spatial_enrichments': self.get_spatial_enrichments(image_id),
            'processing_summary': self.get_processing_summary(image_id)
        }
        
        # Print summary
        summary = record['processing_summary']
        print(f"📊 Processing Summary:")
        print(f"   • Total services: {summary.get('total_services', 0)}")
        print(f"   • Successful results: {summary.get('successful_results', 0)}")
        print(f"   • Failed results: {summary.get('failed_results', 0)}")
        print(f"   • Merged boxes: {len(record['merged_boxes'])}")
        print(f"   • Consensus results: {len(record['consensus'])}")
        print(f"   • Spatial enrichments: {len(record['spatial_enrichments'])}")
        
        return record
    
    def save_json_file(self, record, output_file):
        """Save record to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(record, f, indent=2, default=str)
        
        print(f"💾 Record saved to: {output_file}")
        
        # Print file size
        file_size = os.path.getsize(output_file)
        if file_size > 1024 * 1024:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size} bytes"
        
        print(f"📁 File size: {size_str}")

def list_available_images(extractor, limit=20):
    """List images that have processing results (not all 118k images)"""
    print(f"📋 Images with processing results (showing first {limit}):")
    print()
    
    with extractor.conn.cursor() as cursor:
        cursor.execute("""
            SELECT 
                i.image_id,
                i.image_filename,
                COUNT(r.result_id) as result_count,
                COUNT(DISTINCT r.service) as service_count,
                COUNT(mb.merged_id) as merged_boxes_count,
                COUNT(c.consensus_id) as consensus_count,
                MAX(r.result_created) as last_processed
            FROM images i
            INNER JOIN results r ON i.image_id = r.image_id
            LEFT JOIN merged_boxes mb ON i.image_id = mb.image_id
            LEFT JOIN consensus c ON i.image_id = c.image_id
            GROUP BY i.image_id, i.image_filename
            ORDER BY last_processed DESC
            LIMIT %s
        """, (limit,))
        
        rows = cursor.fetchall()
        
        if not rows:
            print("No images with processing results found.")
            print("(All processing data may have been cleared)")
            return
        
        print("ID      | Filename                  | Results | Services | Boxes | Consensus | Last Processed")
        print("--------|---------------------------|---------|----------|-------|-----------|----------------")
        
        for row in rows:
            image_id, filename, results, services, boxes, consensus, last_processed = row
            filename_short = (filename[:20] + '...') if len(filename) > 23 else filename
            last_processed_str = last_processed.strftime("%m/%d %H:%M") if last_processed else "Unknown"
            print(f"{image_id:7} | {filename_short:25} | {results:7} | {services:8} | {boxes:5} | {consensus:9} | {last_processed_str}")

def main():
    parser = argparse.ArgumentParser(description='Extract complete ML processing record for a single image')
    parser.add_argument('image_id', nargs='?', type=int, help='Image ID to extract')
    parser.add_argument('--output', '-o', help='Output JSON file (default: record_<image_id>.json)')
    parser.add_argument('--list', '-l', action='store_true', help='List available images')
    parser.add_argument('--limit', type=int, default=20, help='Limit for --list (default: 20)')
    
    args = parser.parse_args()
    
    # Initialize extractor
    try:
        extractor = RecordExtractor()
        extractor.connect()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    try:
        if args.list:
            list_available_images(extractor, args.limit)
        elif args.image_id:
            # Extract record
            record = extractor.extract_complete_record(args.image_id)
            
            # Determine output file
            if args.output:
                output_file = args.output
            else:
                output_file = f"record_{args.image_id}.json"
            
            # Save to file
            extractor.save_json_file(record, output_file)
            
            print(f"✅ Extraction completed successfully!")
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        extractor.disconnect()

if __name__ == '__main__':
    main()