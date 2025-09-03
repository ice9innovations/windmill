#!/usr/bin/env python3
"""
Worker Monitoring Script - Check health and status of distributed workers
"""
import os
import mysql.connector
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

class WorkerMonitor:
    """Monitor worker health and performance"""
    
    def __init__(self):
        load_dotenv()
        
        # Get required monitoring configuration
        def get_required(key):
            value = os.getenv(key)
            if not value:
                raise ValueError(f"Required environment variable {key} not set")
            return value
        
        self.mysql_conn = mysql.connector.connect(
            host=get_required('MONITORING_DB_HOST'),
            user=get_required('MONITORING_DB_USER'),
            password=get_required('MONITORING_DB_PASSWORD'),
            database=get_required('MONITORING_DB_NAME')
        )
    
    def find_dead_workers(self, minutes_silent=5):
        """Find workers that haven't sent heartbeat in X minutes"""
        cursor = self.mysql_conn.cursor()
        
        query = """
            SELECT worker_id, service_name, node_hostname, 
                   MAX(timestamp) as last_seen,
                   TIMESTAMPDIFF(MINUTE, MAX(timestamp), NOW()) as minutes_silent,
                   MAX(jobs_completed) as total_jobs
            FROM worker_heartbeats 
            WHERE timestamp > DATE_SUB(NOW(), INTERVAL 1 HOUR)
            GROUP BY worker_id, service_name, node_hostname
            HAVING minutes_silent > %s
            ORDER BY minutes_silent DESC
        """
        
        cursor.execute(query, (minutes_silent,))
        results = cursor.fetchall()
        cursor.close()
        
        return results
    
    def get_service_performance(self, minutes_ago=10):
        """Get processing performance by service"""
        cursor = self.mysql_conn.cursor()
        
        query = """
            SELECT service_name, 
                   AVG(jobs_per_minute) as avg_rate,
                   COUNT(DISTINCT worker_id) as active_workers,
                   SUM(jobs_completed) as total_jobs,
                   MAX(timestamp) as last_update
            FROM worker_heartbeats 
            WHERE timestamp > DATE_SUB(NOW(), INTERVAL %s MINUTE)
            AND status = 'alive'
            GROUP BY service_name
            ORDER BY avg_rate DESC
        """
        
        cursor.execute(query, (minutes_ago,))
        results = cursor.fetchall()
        cursor.close()
        
        return results
    
    def get_node_distribution(self):
        """Get worker distribution across nodes"""
        cursor = self.mysql_conn.cursor()
        
        query = """
            SELECT node_hostname,
                   COUNT(DISTINCT worker_id) as worker_count,
                   GROUP_CONCAT(DISTINCT service_name ORDER BY service_name) as services,
                   MAX(timestamp) as last_update
            FROM worker_heartbeats 
            WHERE timestamp > DATE_SUB(NOW(), INTERVAL 10 MINUTE)
            AND status = 'alive'
            GROUP BY node_hostname
            ORDER BY worker_count DESC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        
        return results
    
    def get_recent_errors(self, minutes_ago=60):
        """Get recent worker errors"""
        cursor = self.mysql_conn.cursor()
        
        query = """
            SELECT timestamp, worker_id, service_name, node_hostname, error_message
            FROM worker_heartbeats 
            WHERE status = 'error'
            AND timestamp > DATE_SUB(NOW(), INTERVAL %s MINUTE)
            ORDER BY timestamp DESC
            LIMIT 20
        """
        
        cursor.execute(query, (minutes_ago,))
        results = cursor.fetchall()
        cursor.close()
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Monitor distributed worker health')
    parser.add_argument('--dead-threshold', '-d', type=int, default=5,
                       help='Minutes without heartbeat to consider worker dead (default: 5)')
    parser.add_argument('--performance-window', '-p', type=int, default=10, 
                       help='Minutes window for performance stats (default: 10)')
    parser.add_argument('--errors-window', '-e', type=int, default=60,
                       help='Minutes window for error reporting (default: 60)')
    
    args = parser.parse_args()
    
    try:
        monitor = WorkerMonitor()
        
        print("🔍 Worker Health Monitor")
        print("=" * 50)
        
        # Dead workers
        dead_workers = monitor.find_dead_workers(args.dead_threshold)
        if dead_workers:
            print(f"\n❌ Dead Workers (silent > {args.dead_threshold} min):")
            for worker_id, service, hostname, last_seen, silent_mins, jobs in dead_workers:
                print(f"  {worker_id} ({service}) on {hostname}")
                print(f"    Last seen: {last_seen} ({silent_mins} min ago)")
                print(f"    Completed jobs: {jobs}")
        else:
            print(f"\n✅ All workers alive (heartbeat within {args.dead_threshold} min)")
        
        # Service performance
        print(f"\n📊 Service Performance (last {args.performance_window} min):")
        performance = monitor.get_service_performance(args.performance_window)
        if performance:
            print("  Service        | Workers | Jobs/Min | Total Jobs | Last Update")
            print("  " + "-" * 65)
            for service, rate, workers, total, last_update in performance:
                rate_str = f"{rate:.1f}" if rate else "0.0"
                print(f"  {service:14} |   {workers:3}   |  {rate_str:6} |    {total:6} | {last_update}")
        else:
            print("  No active workers found")
        
        # Node distribution  
        print(f"\n🏠 Worker Distribution by Node:")
        nodes = monitor.get_node_distribution()
        if nodes:
            for hostname, worker_count, services, last_update in nodes:
                print(f"  {hostname}: {worker_count} workers")
                print(f"    Services: {services}")
                print(f"    Last update: {last_update}")
        else:
            print("  No nodes with active workers")
        
        # Recent errors
        print(f"\n⚠️  Recent Errors (last {args.errors_window} min):")
        errors = monitor.get_recent_errors(args.errors_window)
        if errors:
            for timestamp, worker_id, service, hostname, error_msg in errors:
                print(f"  {timestamp}: {worker_id} ({service}) on {hostname}")
                print(f"    Error: {error_msg}")
        else:
            print("  No recent errors")
        
        return 0
        
    except Exception as e:
        print(f"❌ Monitoring error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())