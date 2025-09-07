#!/usr/bin/env python3
"""
Service configuration loader for nested YAML structure
Handles parsing and flattening the new service_config.yaml format
"""
import yaml
import os
from typing import Dict, Any, List

class ServiceConfig:
    """Service configuration loader for nested YAML structure"""
    
    def __init__(self, config_file='service_config.yaml'):
        self.config_file = config_file
        self.raw_config = None
        self.load_config()
    
    def load_config(self):
        """Load and parse the YAML configuration file"""
        try:
            with open(self.config_file, 'r') as f:
                self.raw_config = yaml.safe_load(f)
            
            if not self.raw_config or 'services' not in self.raw_config:
                raise ValueError(f"Invalid config file {self.config_file}: missing 'services' section")
            
        except FileNotFoundError:
            raise ValueError(f"Config file {self.config_file} not found")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {self.config_file}: {e}")
    
    def get_service_config(self, service_full_name: str) -> Dict[str, Any]:
        """Get configuration for a specific service using category.service format"""
        if '.' not in service_full_name:
            raise ValueError(f"Service name must be in format 'category.service', got: '{service_full_name}'")
        
        category, service_name = service_full_name.split('.', 1)
        
        if category not in self.raw_config['services']:
            available_categories = ', '.join(self.raw_config['services'].keys())
            raise ValueError(f"Unknown category '{category}'. Available: {available_categories}")
        
        if service_name not in self.raw_config['services'][category]:
            available_services = ', '.join(self.raw_config['services'][category].keys())
            raise ValueError(f"Unknown service '{service_name}' in category '{category}'. Available: {available_services}")
        
        service_config = self.raw_config['services'][category][service_name].copy()
        service_config['category'] = category
        
        # Default queue_name to service_name if not specified
        if 'queue_name' not in service_config:
            service_config['queue_name'] = service_name
            
        return service_config
    
    def get_services_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get all services in a specific category"""
        if category not in self.raw_config['services']:
            return {}
        
        services = {}
        for service_name, service_config in self.raw_config['services'][category].items():
            config = service_config.copy() if service_config else {}
            config['category'] = category
            if 'queue_name' not in config:
                config['queue_name'] = service_name
            services[f"{category}.{service_name}"] = config
        
        return services
    
    def get_services_by_type(self, service_type: str) -> Dict[str, Dict[str, Any]]:
        """Get all services with a specific service_type (handles CSV)"""
        matching_services = {}
        
        for category, services in self.raw_config['services'].items():
            for service_name, service_config in services.items():
                if not service_config:
                    continue
                    
                config_types = service_config.get('service_type', '')
                # Split on comma and strip whitespace
                types = [t.strip() for t in config_types.split(',') if t.strip()]
                
                if service_type in types:
                    full_name = f"{category}.{service_name}"
                    config = service_config.copy()
                    config['category'] = category
                    if 'queue_name' not in config:
                        config['queue_name'] = service_name
                    matching_services[full_name] = config
        
        return matching_services
    
    def get_queue_name(self, service_full_name: str) -> str:
        """Get queue name for a service"""
        service_config = self.get_service_config(service_full_name)
        return service_config.get('queue_name', service_full_name.split('.', 1)[1])
    
    def get_queue_by_service_type(self, service_type: str) -> str:
        """Find first service by service_type and return its queue name"""
        services = self.get_services_by_type(service_type)
        if not services:
            raise ValueError(f"No service found with service_type: {service_type}")
        
        # Return first match
        service_full_name = next(iter(services.keys()))
        return self.get_queue_name(service_full_name)
    
    def get_service_group(self, group_spec: str) -> List[str]:
        """Get list of service names using category.service_type[] notation
        
        Examples:
            primary.spatial[] -> ['primary.yolov8', 'primary.rtdetr', ...]  
            postprocessing.specialized[] -> ['postprocessing.face', 'postprocessing.pose']
            system.consensus[] -> ['system.consensus']
        """
        if not group_spec.endswith('[]'):
            raise ValueError(f"Group specification must end with [], got: '{group_spec}'")
        
        # Parse category.service_type[]
        base_spec = group_spec[:-2]  # Remove []
        if '.' not in base_spec:
            raise ValueError(f"Group specification must be in format 'category.service_type[]', got: '{group_spec}'")
        
        category, service_type = base_spec.split('.', 1)
        
        # Find matching services
        matching_services = []
        if category not in self.raw_config['services']:
            return matching_services
        
        for service_name, service_config in self.raw_config['services'][category].items():
            if not service_config:
                continue
                
            config_types = service_config.get('service_type', '')
            # Split on comma and strip whitespace
            types = [t.strip() for t in config_types.split(',') if t.strip()]
            
            if service_type in types:
                matching_services.append(f"{category}.{service_name}")
        
        return sorted(matching_services)
    
    def should_trigger_consensus(self, service_full_name: str) -> bool:
        """Check if service should trigger consensus (default True unless consensus: false)"""
        service_config = self.get_service_config(service_full_name)
        return service_config.get('consensus', True)
    
    def is_spatial_service(self, service_full_name: str) -> bool:
        """Check if service is a spatial (bbox) service"""
        services = self.get_services_by_type('spatial')
        return service_full_name in services
    
    def is_semantic_service(self, service_full_name: str) -> bool:
        """Check if service is a semantic (captioning) service"""  
        services = self.get_services_by_type('semantic')
        return service_full_name in services
    
    def get_spatial_services(self) -> List[str]:
        """Get list of all spatial service names"""
        return list(self.get_services_by_type('spatial').keys())

# Global singleton instance
_config_instance = None

def get_service_config(config_file='service_config.yaml') -> ServiceConfig:
    """Get singleton service config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ServiceConfig(config_file)
    return _config_instance

def reload_service_config(config_file='service_config.yaml'):
    """Force reload of service configuration"""
    global _config_instance
    _config_instance = ServiceConfig(config_file)
    return _config_instance