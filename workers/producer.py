#!/usr/bin/env python3
"""
Generic Queue Producer - Submit jobs to multiple service queues
Configurable producer that can send images to any combination of ML services
"""
import os
import uuid
import sys
import json
import time
import pika
import psycopg2
import argparse
import base64
import requests
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import io

class ProducerConfig:
    """Load producer configuration"""
    
    def __init__(self, env_file='../.env'):
        # Load .env file (optional for producer)
        load_dotenv(env_file)
        
        # Load service configuration using new YAML loader
        from service_config import get_service_config
        self.config = get_service_config('../service_config.yaml')
        
        # Queue configuration
        self.queue_host = self._get_required('QUEUE_HOST')
        self.queue_user = self._get_required('QUEUE_USER')
        self.queue_password = self._get_required('QUEUE_PASSWORD')
        
        # Database configuration
        self.db_host = self._get_required('DB_HOST')
        self.db_name = self._get_required('DB_NAME')
        self.db_user = self._get_required('DB_USER')
        self.db_password = self._get_required('DB_PASSWORD')
    
    def _get_required(self, key):
        """Get required environment variable with no fallback"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value
    
    def get_available_services(self):
        """Get list of available service names"""
        primary_services = list(self.config.get_services_by_category('primary').keys())
        return [name.split('.', 1)[1] for name in primary_services]  # Remove 'primary.' prefix for CLI
    
    def get_services_by_category(self, category):
        """Get services that belong to a specific category"""
        services = self.config.get_services_by_category(category)
        return [name.split('.', 1)[1] for name in services.keys()]  # Remove category prefix for CLI
    
    def get_primary_services(self):
        """Get services that run on whole images (excludes postprocessing-only services)"""
        return self.get_services_by_category('primary')
    
    def get_queue_name(self, service_name):
        """Get queue name for a service (expects service_name without category prefix)"""
        full_service_name = f'primary.{service_name}'
        return self.config.get_queue_name(full_service_name)

class GenericProducer:
    """Generic producer for submitting jobs to ML service queues"""
    
    def __init__(self):
        self.config = ProducerConfig()
        self.connection = None
        self.channel = None
        self.db_conn = None
        print(f"🚀 Generic Queue Producer initialized")
    
    def extract_image_dimensions(self, image_bytes):
        """Extract width and height from image bytes using PIL"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return image.width, image.height
        except Exception as e:
            print(f"⚠️  Failed to extract image dimensions: {e}")
            return -1, -1
    
    def connect_to_rabbitmq(self):
        """Connect to RabbitMQ"""
        try:
            credentials = pika.PlainCredentials(self.config.queue_user, self.config.queue_password)
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=self.config.queue_host,
                    credentials=credentials
                )
            )
            self.channel = self.connection.channel()
            print(f"✅ Connected to RabbitMQ at {self.config.queue_host}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to RabbitMQ: {e}")
            return False
    
    def connect_to_database(self):
        """Connect to PostgreSQL database"""
        try:
            self.db_conn = psycopg2.connect(
                host=self.config.db_host,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password
            )
            print(f"✅ Connected to PostgreSQL at {self.config.db_host}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            return False
    
    def create_queues(self, service_names):
        """Create queues for the specified services"""
        created_queues = []
        
        # Helper to declare a queue with DLQ/TTL args matching workers
        def declare_with_dlq(channel, queue_name):
            dlq_name = f"{queue_name}.dlq"
            channel.queue_declare(queue=dlq_name, durable=True)
            args = {
                'x-dead-letter-exchange': '',
                'x-dead-letter-routing-key': dlq_name,
                'x-max-length': int(os.getenv('QUEUE_MAX_LENGTH', '100000'))
            }
            ttl_env = os.getenv('QUEUE_MESSAGE_TTL_MS')
            if ttl_env and ttl_env.isdigit() and int(ttl_env) > 0:
                args['x-message-ttl'] = int(ttl_env)
            channel.queue_declare(queue=queue_name, durable=True, arguments=args)

        for service_name in service_names:
            try:
                queue_name = self.config.get_queue_name(service_name)
                declare_with_dlq(self.channel, queue_name)
                created_queues.append(queue_name)
                
            except ValueError as e:
                print(f"❌ Error creating queue for {service_name}: {e}")
                continue
        
        if created_queues:
            print(f"✅ Created/verified queues: {', '.join(created_queues)}")
        
        return created_queues
    
    def get_images_from_database(self, limit=None, image_group=None, resume=False, image_ids_file=None):
        """Get image cursor from database, optionally filtered by group, resuming, or specific image IDs"""
        try:
            cursor = self.db_conn.cursor()
            
            # Build query with optional group filter and resume functionality
            base_query = """
                SELECT image_id, image_filename, image_path, image_url 
                FROM images 
            """
            
            conditions = []
            params = []
            
            # Handle specific image IDs from file
            if image_ids_file:
                with open(image_ids_file, 'r') as f:
                    image_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
                if image_ids:
                    placeholders = ','.join(['%s'] * len(image_ids))
                    conditions.append(f"image_id IN ({placeholders})")
                    params.extend(image_ids)
                    print(f"📋 Processing {len(image_ids)} specific image IDs from {image_ids_file}")
                else:
                    print(f"❌ No valid image IDs found in {image_ids_file}")
                    return None
            
            elif image_group:
                conditions.append("image_group = %s")
                params.append(image_group)
            
            # Resume functionality: skip images that already have results
            if resume:
                # Find the highest image_id that already has results
                resume_cursor = self.db_conn.cursor()
                resume_query = "SELECT MAX(image_id) FROM results"
                if image_group:
                    resume_query += " WHERE image_id IN (SELECT image_id FROM images WHERE image_group = %s)"
                    resume_cursor.execute(resume_query, (image_group,))
                else:
                    resume_cursor.execute(resume_query)
                
                max_processed_id = resume_cursor.fetchone()[0]
                resume_cursor.close()
                
                if max_processed_id:
                    conditions.append("image_id > %s")
                    params.append(max_processed_id)
                    print(f"🔄 Resuming from image_id > {max_processed_id} (last processed)")
                else:
                    print(f"🔄 No previous results found, starting from beginning")
            
            if conditions:
                base_query += "WHERE " + " AND ".join(conditions) + " "
            
            base_query += "ORDER BY image_id"
            
            if limit:
                base_query += " LIMIT %s"
                params.append(limit)
            
            cursor.execute(base_query, params)
            
            # Return count and cursor for streaming
            row_count = cursor.rowcount if cursor.rowcount > 0 else "unknown"
            group_info = f" from group '{image_group}'" if image_group else ""
            resume_info = " (resuming)" if resume else ""
            print(f"📋 Found {row_count} images{group_info}{resume_info} in database")
            return cursor
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            return None
    
    def process_image_row(self, row):
        """Process a single database row into image data"""
        image_data = {
            "image_id": row[0],
            "image_filename": row[1],
            "submitted_at": datetime.now().isoformat()
        }
        
        # Try URL first (deployment-agnostic), fallback to local path
        image_url = row[3] if row[3] else None
        image_path = row[2] if row[2] else None
        
        # Prefer URL for deployment-agnostic access
        if image_url:
            try:
                print(f"🔗 Fetching image from URL: {image_url}")
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                image_bytes = response.content
                
                # Extract dimensions and add image data
                width, height = self.extract_image_dimensions(image_bytes)
                image_data["image_data"] = base64.b64encode(image_bytes).decode('utf-8')
                image_data["image_width"] = width
                image_data["image_height"] = height
                return image_data
            except Exception as e:
                print(f"⚠️  Failed to fetch image from URL {image_url}: {e}")
                # Fall through to try local path
        
        # Fallback to local file path
        if image_path and os.path.exists(image_path):
            try:
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                    
                    # Extract dimensions and add image data
                    width, height = self.extract_image_dimensions(image_bytes)
                    image_data["image_data"] = base64.b64encode(image_bytes).decode('utf-8')
                    image_data["image_width"] = width
                    image_data["image_height"] = height
                return image_data
            except Exception as e:
                print(f"⚠️  Failed to read image {image_path}: {e}")
                return None
        
        print(f"⚠️  No valid image source found - URL: {image_url}, Path: {image_path}")
        return None
    
    def submit_job(self, queue_name, image_data):
        """Submit a single job to a queue"""
        try:
            message = json.dumps(image_data)
            
            self.channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=message,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                )
            )
            return True
            
        except Exception as e:
            print(f"❌ Error submitting to {queue_name}: {e}")
            return False
    
    def submit_jobs(self, service_names, cursor, delay=0.01):
        """Submit jobs for multiple services and images using streaming cursor"""
        submitted_jobs = 0
        processed_images = 0
        
        print(f"📤 Starting streaming job submission ({len(service_names)} services)")
        
        # Stream through images one at a time
        for row in cursor:
            # Process single image into memory
            image_data = self.process_image_row(row)
            if not image_data:
                print(f"⚠️  Skipping image {row[0]} - failed to process")
                continue
                
            processed_images += 1
            # Create a trace id for this image's batch of jobs
            trace_id = str(uuid.uuid4())
            jobs_this_image = 0
            
            # Submit jobs for this image to all services
            for service_name in service_names:
                try:
                    queue_name = self.config.get_queue_name(service_name)
                    
                    # Add service-specific metadata to job
                    job_data = image_data.copy()
                    job_data['trace_id'] = trace_id
                    job_data['service_name'] = service_name
                    job_data['queue_name'] = queue_name
                    
                    if self.submit_job(queue_name, job_data):
                        submitted_jobs += 1
                        jobs_this_image += 1
                        if submitted_jobs % 100 == 0:
                            print(f"📤 Submitted {submitted_jobs} jobs (processed {processed_images} images, avg {submitted_jobs/processed_images:.1f} jobs/image)...")
                    else:
                        print(f"❌ Failed to submit job for image {image_data['image_id']} to service {service_name}")
                    
                    # Small delay to prevent overwhelming the queue
                    if delay > 0:
                        time.sleep(delay)
                        
                except ValueError as e:
                    print(f"❌ Skipping {service_name}: {e}")
                    continue
            
            # Debug if jobs per image isn't exactly the number of services
            if jobs_this_image != len(service_names):
                print(f"⚠️  Image {image_data['image_id']}: submitted {jobs_this_image}/{len(service_names)} jobs")
            
            # Clear image data from memory after processing
            image_data = None
        
        print(f"✅ Submitted {submitted_jobs} jobs from {processed_images} images successfully")
        return submitted_jobs
    
    def close_connections(self):
        """Close all connections"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        if self.db_conn:
            self.db_conn.close()

def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(description='Submit jobs to ML service queues')
    
    parser.add_argument('--limit', '-l', 
                       type=int, 
                       help='Limit number of images to process')
    
    parser.add_argument('--delay', '-d', 
                       type=float, 
                       default=0.01, 
                       help='Delay between jobs in seconds (default: 0.01)')
    
    parser.add_argument('--list-services', 
                       action='store_true', 
                       help='List available services and exit')
    
    parser.add_argument('--group', '-g',
                       help='Process only images from specific group (e.g., coco2017)')
    
    parser.add_argument('--no-resume', 
                       action='store_true',
                       help='Process all images including those with existing results (dangerous - can create duplicates)')
    
    parser.add_argument('--image-ids-file',
                       help='File containing image IDs to reprocess (one per line)')
    
    args = parser.parse_args()
    
    try:
        producer = GenericProducer()
        
        # List services option
        if args.list_services:
            print("Available services by category:")
            print()
            
            # Primary services (whole image processing)
            primary_services = producer.config.get_primary_services()
            print("📋 PRIMARY (whole image processing):")
            for service in sorted(primary_services):
                full_service_name = f'primary.{service}'
                service_config = producer.config.config.get_service_config(full_service_name)
                port = service_config.get('port', 'N/A')
                print(f"  {service:12} (port {port})")
            
            print()
            
            # Postprocessing-only services (bbox region processing)
            postprocessing_services = producer.config.get_services_by_category('postprocessing')
            print("🔍 POSTPROCESSING (bbox region processing via postprocessing workers):")
            for service in sorted(postprocessing_services):
                full_service_name = f'postprocessing.{service}'
                service_config = producer.config.config.get_service_config(full_service_name)
                port = service_config.get('port', 'N/A')
                print(f"  {service:12} (port {port})")
            
            print()
            print("Service sets:")
            print("  all          = primary services (recommended for whole image processing)")
            print("  primary      = primary services (same as 'all')")  
            print("  postprocessing = face + pose services (handled by postprocessing workers, not for direct use)")
            print("  full_catalog = all services including postprocessing")
            
            return 0
        
        # Connect to infrastructure
        if not producer.connect_to_rabbitmq():
            return 1
        
        if not producer.connect_to_database():
            return 1
        
        # Use all primary services only
        service_names = producer.config.get_primary_services()
        
        print(f"🎯 Target services: {', '.join(service_names)}")
        
        # Create queues
        created_queues = producer.create_queues(service_names)
        if not created_queues:
            print("❌ No queues could be created")
            return 1
        
        # Get images cursor for streaming (resume by default, unless --no-resume specified)
        resume = not args.no_resume  
        cursor = producer.get_images_from_database(args.limit, args.group, resume, args.image_ids_file)
        if not cursor:
            print("❌ No images found in database")
            return 1
        
        # Submit jobs using streaming
        submitted = producer.submit_jobs(service_names, cursor, args.delay)
        
        print(f"🎉 Processing complete! {submitted} jobs submitted.")
        
        producer.close_connections()
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())