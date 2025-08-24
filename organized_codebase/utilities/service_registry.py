#!/usr/bin/env python3
"""
Service Registry - Agent A Hour 5
Central service registration and management system

Registers core services with the integrated architecture framework
using DependencyContainer with proper layer separation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Type, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Import architecture framework
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.architecture_integration import (
    get_architecture_framework,
    ServiceRegistration,
    ArchitecturalLayer,
    LifetimeScope
)

# Import dashboard monitoring (with fallback)
try:
    from web.dashboard.architecture_monitor import get_architecture_monitor
except ImportError:
    def get_architecture_monitor():
        return None


class ServiceType(Enum):
    """Types of services in the system"""
    ANALYTICS = "analytics"
    MONITORING = "monitoring"
    DASHBOARD = "dashboard"
    TEMPLATE = "template"
    INTELLIGENCE = "intelligence"
    SECURITY = "security"
    TESTING = "testing"
    OPTIMIZATION = "optimization"


@dataclass
class ServiceDefinition:
    """Definition of a service to be registered"""
    name: str
    service_type: ServiceType
    interface_type: Optional[Type] = None
    implementation_type: Optional[Type] = None
    instance: Optional[Any] = None
    layer: ArchitecturalLayer = ArchitecturalLayer.APPLICATION
    lifetime: LifetimeScope = LifetimeScope.SINGLETON
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


class CoreServiceRegistry:
    """
    Central registry for all core services
    
    Manages service registration, discovery, and lifecycle using the
    integrated architecture framework with proper layer separation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Get architecture components
        self.framework = get_architecture_framework()
        self.monitor = get_architecture_monitor()  # May be None if dashboard unavailable
        
        # Service definitions
        self.service_definitions: Dict[str, ServiceDefinition] = {}
        self.registered_services: Dict[str, ServiceRegistration] = {}
        
        # Registration statistics
        self.registration_stats = {
            'total_attempted': 0,
            'successful': 0,
            'failed': 0,
            'by_layer': {},
            'by_type': {},
            'start_time': datetime.now()
        }
        
        self.logger.info("Core Service Registry initialized")
    
    def define_service(self, definition: ServiceDefinition) -> bool:
        """Define a service for registration"""
        try:
            self.service_definitions[definition.name] = definition
            self.logger.debug(f"Defined service: {definition.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to define service {definition.name}: {e}")
            return False
    
    def register_service(self, service_name: str) -> bool:
        """Register a defined service with the architecture framework"""
        definition = self.service_definitions.get(service_name)
        if not definition:
            self.logger.error(f"Service definition not found: {service_name}")
            return False
        
        try:
            self.registration_stats['total_attempted'] += 1
            
            # Determine what to register
            if definition.instance:
                # Register instance
                success = self._register_instance(definition)
            elif definition.implementation_type:
                # Register type mapping
                success = self._register_type_mapping(definition)
            else:
                # Register factory or auto-resolve
                success = self._register_auto_service(definition)
            
            if success:
                self.registration_stats['successful'] += 1
                self._update_stats(definition, success=True)
                self.logger.info(f"Successfully registered service: {service_name}")
            else:
                self.registration_stats['failed'] += 1
                self._update_stats(definition, success=False)
                
            return success
            
        except Exception as e:
            self.registration_stats['failed'] += 1
            self._update_stats(definition, success=False)
            self.logger.error(f"Failed to register service {service_name}: {e}")
            return False
    
    def _register_instance(self, definition: ServiceDefinition) -> bool:
        """Register service instance"""
        interface_type = definition.interface_type or type(definition.instance)
        
        return self.framework.register_service(
            service_name=definition.name,
            service_type=interface_type,
            implementation=definition.instance,
            layer=definition.layer,
            lifetime=definition.lifetime
        )
    
    def _register_type_mapping(self, definition: ServiceDefinition) -> bool:
        """Register service type mapping"""
        interface_type = definition.interface_type or definition.implementation_type
        
        return self.framework.register_service(
            service_name=definition.name,
            service_type=interface_type,
            implementation=definition.implementation_type,
            layer=definition.layer,
            lifetime=definition.lifetime
        )
    
    def _register_auto_service(self, definition: ServiceDefinition) -> bool:
        """Register auto-resolved service"""
        # For services without explicit implementation, create a placeholder
        class AutoService:
            __name__ = f"AutoService_{definition.name}"
            
            def __init__(self):
                self.name = definition.name
                self.service_type = definition.service_type
                self.metadata = definition.metadata
                
            def __repr__(self):
                return f"<AutoService: {self.name}>"
        
        # Set the class name dynamically
        AutoService.__name__ = f"AutoService_{definition.name}"
        auto_instance = AutoService()
        
        return self.framework.register_service(
            service_name=definition.name,
            service_type=AutoService,
            implementation=auto_instance,
            layer=definition.layer,
            lifetime=definition.lifetime
        )
    
    def _update_stats(self, definition: ServiceDefinition, success: bool):
        """Update registration statistics"""
        # Update layer stats
        layer_key = definition.layer.value
        if layer_key not in self.registration_stats['by_layer']:
            self.registration_stats['by_layer'][layer_key] = {'success': 0, 'failed': 0}
        
        if success:
            self.registration_stats['by_layer'][layer_key]['success'] += 1
        else:
            self.registration_stats['by_layer'][layer_key]['failed'] += 1
        
        # Update type stats
        type_key = definition.service_type.value
        if type_key not in self.registration_stats['by_type']:
            self.registration_stats['by_type'][type_key] = {'success': 0, 'failed': 0}
        
        if success:
            self.registration_stats['by_type'][type_key]['success'] += 1
        else:
            self.registration_stats['by_type'][type_key]['failed'] += 1
    
    def register_all_core_services(self) -> Dict[str, bool]:
        """Register all defined core services"""
        results = {}
        
        self.logger.info("Starting core services registration...")
        
        # Register in dependency order (infrastructure first, then application, etc.)
        registration_order = [
            ArchitecturalLayer.INFRASTRUCTURE,
            ArchitecturalLayer.APPLICATION,
            ArchitecturalLayer.PRESENTATION
        ]
        
        for layer in registration_order:
            layer_services = [
                name for name, def_ in self.service_definitions.items()
                if def_.layer == layer
            ]
            
            self.logger.info(f"Registering {len(layer_services)} services in {layer.value} layer")
            
            for service_name in layer_services:
                results[service_name] = self.register_service(service_name)
        
        success_count = sum(1 for r in results.values() if r)
        total_count = len(results)
        
        self.logger.info(f"Service registration complete: {success_count}/{total_count} successful")
        
        return results
    
    def get_registration_report(self) -> Dict[str, Any]:
        """Get comprehensive service registration report"""
        total_time = (datetime.now() - self.registration_stats['start_time']).total_seconds()
        
        return {
            'summary': {
                'total_attempted': self.registration_stats['total_attempted'],
                'successful': self.registration_stats['successful'],
                'failed': self.registration_stats['failed'],
                'success_rate': (
                    self.registration_stats['successful'] / 
                    max(self.registration_stats['total_attempted'], 1)
                ),
                'total_time_seconds': total_time
            },
            'by_layer': self.registration_stats['by_layer'],
            'by_type': self.registration_stats['by_type'],
            'service_definitions': len(self.service_definitions),
            'registered_services': len(self.registered_services),
            'timestamp': datetime.now().isoformat()
        }


def initialize_core_services() -> CoreServiceRegistry:
    """Initialize and define all core services"""
    registry = CoreServiceRegistry()
    
    # Define Dashboard Services (Presentation Layer)
    monitor = get_architecture_monitor()
    if monitor:
        registry.define_service(ServiceDefinition(
            name="architecture_monitor",
            service_type=ServiceType.DASHBOARD,
            instance=monitor,
            layer=ArchitecturalLayer.PRESENTATION,
            lifetime=LifetimeScope.SINGLETON,
            metadata={"component": "dashboard", "real_time": True}
        ))
    
    # Define Monitoring Services (Infrastructure Layer)
    registry.define_service(ServiceDefinition(
        name="realtime_monitor",
        service_type=ServiceType.MONITORING,
        layer=ArchitecturalLayer.INFRASTRUCTURE,
        lifetime=LifetimeScope.SINGLETON,
        metadata={"websocket_enabled": True}
    ))
    
    # Define Analytics Services (Application Layer)
    registry.define_service(ServiceDefinition(
        name="personal_analytics",
        service_type=ServiceType.ANALYTICS,
        layer=ArchitecturalLayer.APPLICATION,
        lifetime=LifetimeScope.SCOPED,
        metadata={"data_source": "user_activity"}
    ))
    
    # Define Template Services (Application Layer)
    registry.define_service(ServiceDefinition(
        name="template_processor",
        service_type=ServiceType.TEMPLATE,
        layer=ArchitecturalLayer.APPLICATION,
        lifetime=LifetimeScope.SINGLETON,
        metadata={"template_engine": "modular"}
    ))
    
    registry.define_service(ServiceDefinition(
        name="webapp_generator",
        service_type=ServiceType.TEMPLATE,
        layer=ArchitecturalLayer.APPLICATION,
        lifetime=LifetimeScope.TRANSIENT,
        metadata={"generator_type": "webapp"}
    ))
    
    registry.define_service(ServiceDefinition(
        name="api_generator",
        service_type=ServiceType.TEMPLATE,
        layer=ArchitecturalLayer.APPLICATION,
        lifetime=LifetimeScope.TRANSIENT,
        metadata={"generator_type": "api"}
    ))
    
    # Define Intelligence Services (Application Layer)
    registry.define_service(ServiceDefinition(
        name="predictive_intelligence",
        service_type=ServiceType.INTELLIGENCE,
        layer=ArchitecturalLayer.APPLICATION,
        lifetime=LifetimeScope.SINGLETON,
        metadata={"intelligence_type": "predictive", "ml_enabled": True}
    ))
    
    registry.define_service(ServiceDefinition(
        name="code_predictor",
        service_type=ServiceType.INTELLIGENCE,
        layer=ArchitecturalLayer.APPLICATION,
        lifetime=LifetimeScope.SINGLETON,
        metadata={"prediction_type": "code_analysis"}
    ))
    
    # Define Security Services (Infrastructure Layer)
    registry.define_service(ServiceDefinition(
        name="security_scanner",
        service_type=ServiceType.SECURITY,
        layer=ArchitecturalLayer.INFRASTRUCTURE,
        lifetime=LifetimeScope.SINGLETON,
        metadata={"scan_type": "unified"}
    ))
    
    # Define Testing Services (Infrastructure Layer)
    registry.define_service(ServiceDefinition(
        name="test_generator",
        service_type=ServiceType.TESTING,
        layer=ArchitecturalLayer.INFRASTRUCTURE,
        lifetime=LifetimeScope.TRANSIENT,
        metadata={"generation_type": "automated"}
    ))
    
    # Define Optimization Services (Application Layer)
    registry.define_service(ServiceDefinition(
        name="performance_optimizer",
        service_type=ServiceType.OPTIMIZATION,
        layer=ArchitecturalLayer.APPLICATION,
        lifetime=LifetimeScope.SINGLETON,
        metadata={"optimization_type": "performance"}
    ))
    
    registry.logger.info(f"Defined {len(registry.service_definitions)} core services")
    
    return registry


# Global registry instance
_service_registry: Optional[CoreServiceRegistry] = None


def get_service_registry() -> CoreServiceRegistry:
    """Get global service registry instance"""
    global _service_registry
    if _service_registry is None:
        _service_registry = initialize_core_services()
    return _service_registry


def register_all_services() -> Dict[str, Any]:
    """Register all core services and return results"""
    registry = get_service_registry()
    results = registry.register_all_core_services()
    report = registry.get_registration_report()
    
    return {
        'registration_results': results,
        'report': report
    }