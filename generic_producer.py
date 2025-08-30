#!/usr/bin/env python3
"""
Generic Queue Producer - Submit jobs to multiple service queues
Configurable producer that can send images to any combination of ML services
"""
import os
import sys
import json
import time
import pika
import psycopg2
import argparse
from datetime import datetime
from dotenv import load_dotenv

class ProducerConfig:
    """Load producer configuration"""
    
    def __init__(self, env_file='.env'):
        # Load .env file (optional for producer)
        load_dotenv(env_file)
        
        # Load service definitions
        with open('service_config.json', 'r') as f:
            self.service_definitions = json.load(f)['services']
        
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
        return list(self.service_definitions.keys())
    
    def get_services_by_category(self, category):
        """Get services that belong to a specific category"""
        return [
            service_name for service_name, service_config in self.service_definitions.items()
            if service_config.get('category') == category
        ]
    
    def get_primary_services(self):
        """Get services that run on whole images (excludes spatial_only)"""
        return self.get_services_by_category('primary')
    
    def get_queue_name(self, service_name):
        """Get queue name for a service"""
        if service_name not in self.service_definitions:
            raise ValueError(f"Unknown service: {service_name}")
        prefix = self.service_definitions[service_name].get('queue_prefix', 'queue_')
        return f"{prefix}{service_name}"

class GenericProducer:
    """Generic producer for submitting jobs to ML service queues"""
    
    def __init__(self):
        self.config = ProducerConfig()
        self.connection = None
        self.channel = None
        self.db_conn = None
        print(f"🚀 Generic Queue Producer initialized")
    
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
        
        for service_name in service_names:
            try:
                queue_name = self.config.get_queue_name(service_name)
                self.channel.queue_declare(queue=queue_name, durable=True)
                created_queues.append(queue_name)
                
            except ValueError as e:
                print(f"❌ Error creating queue for {service_name}: {e}")
                continue
        
        if created_queues:
            print(f"✅ Created/verified queues: {', '.join(created_queues)}")
        
        return created_queues
    
    def get_images_from_database(self, limit=None, image_group=None):
        """Get images from database, optionally filtered by group"""
        try:
            cursor = self.db_conn.cursor()
            
            # Build query with optional group filter
            base_query = """
                SELECT image_id, image_filename, image_path, image_url 
                FROM images 
            """
            
            conditions = []
            params = []
            
            if image_group:
                conditions.append("image_group = %s")
                params.append(image_group)
            
            if conditions:
                base_query += "WHERE " + " AND ".join(conditions) + " "
            
            base_query += "ORDER BY image_id"
            
            if limit:
                base_query += " LIMIT %s"
                params.append(limit)
            
            cursor.execute(base_query, params)
            
            images = []
            for row in cursor.fetchall():
                images.append({
                    "image_id": row[0],
                    "image_filename": row[1],
                    "image_path": row[2],
                    "image_url": row[3],
                    "submitted_at": datetime.now().isoformat()
                })
            
            group_info = f" from group '{image_group}'" if image_group else ""
            print(f"📋 Retrieved {len(images)} images{group_info} from database")
            return images
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            return []
    
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
    
    def submit_jobs(self, service_names, images, delay=0.01):
        """Submit jobs for multiple services and images"""
        total_jobs = len(service_names) * len(images)
        submitted_jobs = 0
        
        print(f"📤 Submitting {total_jobs} jobs ({len(images)} images × {len(service_names)} services)")
        
        # Submit jobs for each image to each service
        for image in images:
            for service_name in service_names:
                try:
                    queue_name = self.config.get_queue_name(service_name)
                    
                    # Add service-specific metadata to job
                    job_data = image.copy()
                    job_data['service_name'] = service_name
                    job_data['queue_name'] = queue_name
                    
                    if self.submit_job(queue_name, job_data):
                        submitted_jobs += 1
                        if submitted_jobs % 100 == 0:
                            print(f"📤 Submitted {submitted_jobs}/{total_jobs} jobs...")
                    
                    # Small delay to prevent overwhelming the queue
                    if delay > 0:
                        time.sleep(delay)
                        
                except ValueError as e:
                    print(f"❌ Skipping {service_name}: {e}")
                    continue
        
        print(f"✅ Submitted {submitted_jobs}/{total_jobs} jobs successfully")
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
    
    parser.add_argument('--services', '-s', 
                       help='Services to use: all, primary, spatial_only, full_catalog, or comma-separated list (default: all)', 
                       default='all')
    
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
                desc = producer.config.service_definitions[service]['description']
                port = producer.config.service_definitions[service]['port']
                print(f"  {service:12} (port {port}): {desc}")
            
            print()
            
            # Spatial-only services (bbox region processing)
            spatial_services = producer.config.get_services_by_category('spatial_only')
            print("🔍 SPATIAL_ONLY (bbox region processing via spatial_enrichment_worker):")
            for service in sorted(spatial_services):
                desc = producer.config.service_definitions[service]['description']
                port = producer.config.service_definitions[service]['port']
                print(f"  {service:12} (port {port}): {desc}")
            
            print()
            print("Service sets:")
            print("  all          = primary services (recommended for whole image processing)")
            print("  primary      = primary services (same as 'all')")  
            print("  spatial_only = face + pose services (not recommended for direct use)")
            print("  full_catalog = all services including spatial_only")
            
            return 0
        
        # Connect to infrastructure
        if not producer.connect_to_rabbitmq():
            return 1
        
        if not producer.connect_to_database():
            return 1
        
        # Determine services to use
        if args.services == 'all':
            # 'all' now means primary services only (excludes face/pose)
            service_names = producer.config.get_primary_services()
        elif args.services == 'primary':
            service_names = producer.config.get_primary_services()
        elif args.services == 'spatial_only':
            service_names = producer.config.get_services_by_category('spatial_only')
        elif args.services == 'full_catalog':
            # Full catalog includes ALL services (including face/pose)
            service_names = producer.config.get_available_services()
        else:
            service_names = [s.strip() for s in args.services.split(',')]
        
        print(f"🎯 Target services: {', '.join(service_names)}")
        
        # Create queues
        created_queues = producer.create_queues(service_names)
        if not created_queues:
            print("❌ No queues could be created")
            return 1
        
        # Get images
        images = producer.get_images_from_database(args.limit, args.group)
        if not images:
            print("❌ No images found in database")
            return 1
        
        # Submit jobs
        submitted = producer.submit_jobs(service_names, images, args.delay)
        
        print(f"🎉 Processing complete! {submitted} jobs submitted.")
        
        producer.close_connections()
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())