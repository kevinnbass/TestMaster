#!/usr/bin/env python3
"""
Hexagonal Architecture Layer Separation - Agent A
Implements clean layer management with proper separation of concerns

This module provides LayerManager and supporting classes for enforcing
hexagonal architecture principles with proper layer isolation and
dependency direction validation.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Protocol
import importlib.util
import re


class ArchitecturalLayer(Enum):
    """Hexagonal architecture layers"""
    DOMAIN = "domain"
    APPLICATION = "application" 
    INFRASTRUCTURE = "infrastructure"
    PRESENTATION = "presentation"


@dataclass
class LayerComponent:
    """Represents a component within an architectural layer"""
    name: str
    layer: ArchitecturalLayer
    path: Path
    dependencies: Set[str] = field(default_factory=set)
    interfaces: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    
    
@dataclass
class LayerViolation:
    """Represents a violation of layer separation principles"""
    component: str
    violation_type: str
    source_layer: ArchitecturalLayer
    target_layer: ArchitecturalLayer
    description: str
    severity: str = "medium"


@dataclass
class ArchitectureValidationResult:
    """Results of architectural layer validation"""
    compliance_score: float
    layer_health: Dict[ArchitecturalLayer, float]
    violations: List[LayerViolation]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class LayerAdapter(Protocol):
    """Protocol for layer adapters"""
    
    def adapt(self, data: Any) -> Any:
        """Adapt data between layers"""
        ...
    
    def validate(self, data: Any) -> bool:
        """Validate data for layer requirements"""
        ...


class LayerManager:
    """
    Hexagonal architecture layer management
    
    Manages layer separation, dependency injection, and architectural compliance
    for clean architecture implementation.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Layer structure
        self.layers = {
            ArchitecturalLayer.DOMAIN: self._create_domain_layer(),
            ArchitecturalLayer.APPLICATION: self._create_application_layer(), 
            ArchitecturalLayer.INFRASTRUCTURE: self._create_infrastructure_layer(),
            ArchitecturalLayer.PRESENTATION: self._create_presentation_layer()
        }
        
        # Adapters and ports
        self.adapters: Dict[str, LayerAdapter] = {}
        self.ports: Dict[str, Any] = {}
        
        # Component registry
        self.components: Dict[str, LayerComponent] = {}
        
        # Validation rules
        self.dependency_rules = self._initialize_dependency_rules()

    def _create_domain_layer(self) -> Dict[str, Any]:
        """Create domain layer configuration"""
        return {
            'name': 'Domain',
            'description': 'Business logic and entities',
            'allowed_dependencies': [],  # Domain depends on nothing
            'components': {},
            'patterns': ['entity', 'value_object', 'aggregate', 'domain_service']
        }
    
    def _create_application_layer(self) -> Dict[str, Any]:
        """Create application layer configuration"""
        return {
            'name': 'Application',
            'description': 'Use cases and application services',
            'allowed_dependencies': [ArchitecturalLayer.DOMAIN],
            'components': {},
            'patterns': ['use_case', 'application_service', 'command', 'query']
        }
    
    def _create_infrastructure_layer(self) -> Dict[str, Any]:
        """Create infrastructure layer configuration"""
        return {
            'name': 'Infrastructure', 
            'description': 'External interfaces and implementations',
            'allowed_dependencies': [ArchitecturalLayer.DOMAIN, ArchitecturalLayer.APPLICATION],
            'components': {},
            'patterns': ['repository', 'api_client', 'database', 'external_service']
        }
    
    def _create_presentation_layer(self) -> Dict[str, Any]:
        """Create presentation layer configuration"""
        return {
            'name': 'Presentation',
            'description': 'User interface and controllers',
            'allowed_dependencies': [ArchitecturalLayer.APPLICATION],
            'components': {},
            'patterns': ['controller', 'view', 'presenter', 'api_endpoint']
        }

    def _initialize_dependency_rules(self) -> Dict[ArchitecturalLayer, Set[ArchitecturalLayer]]:
        """Initialize allowed dependency directions"""
        return {
            ArchitecturalLayer.DOMAIN: set(),
            ArchitecturalLayer.APPLICATION: {ArchitecturalLayer.DOMAIN},
            ArchitecturalLayer.INFRASTRUCTURE: {ArchitecturalLayer.DOMAIN, ArchitecturalLayer.APPLICATION},
            ArchitecturalLayer.PRESENTATION: {ArchitecturalLayer.APPLICATION}
        }

    def register_adapter(self, layer: str, adapter: LayerAdapter) -> bool:
        """Register adapter for specific layer"""
        try:
            self.adapters[layer] = adapter
            self._validate_adapter_interface(adapter)
            self.logger.info(f"Registered adapter for layer: {layer}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register adapter for {layer}: {e}")
            return False

    def _validate_adapter_interface(self, adapter: LayerAdapter) -> bool:
        """Validate adapter implements required interface"""
        required_methods = ['adapt', 'validate']
        for method in required_methods:
            if not hasattr(adapter, method):
                raise ValueError(f"Adapter missing required method: {method}")
        return True

    def create_service_interface(self, service_name: str) -> object:
        """Create clean service interface with dependency injection"""
        interface = self._generate_service_interface(service_name)
        self.ports[service_name] = interface
        self.logger.info(f"Created service interface: {service_name}")
        return interface

    def _generate_service_interface(self, service_name: str) -> Dict[str, Any]:
        """Generate service interface configuration"""
        return {
            'name': service_name,
            'type': 'service_interface',
            'methods': [],
            'dependencies': [],
            'created_at': datetime.now(),
            'layer': None  # Will be determined during registration
        }

    def validate_architecture_integrity(self) -> ArchitectureValidationResult:
        """Validate hexagonal architecture compliance"""
        violations = []
        layer_health = {}
        
        # Check layer isolation
        isolation_violations = self._check_layer_isolation()
        violations.extend(isolation_violations)
        
        # Validate dependency directions
        dependency_violations = self._validate_dependency_directions() 
        violations.extend(dependency_violations)
        
        # Check interface compliance
        interface_violations = self._check_interface_compliance()
        violations.extend(interface_violations)
        
        # Calculate layer health scores
        for layer in ArchitecturalLayer:
            layer_violations = [v for v in violations if v.source_layer == layer]
            health_score = max(0.0, 1.0 - (len(layer_violations) * 0.1))
            layer_health[layer] = health_score
        
        # Overall compliance score
        if layer_health:
            compliance_score = sum(layer_health.values()) / len(layer_health)
        else:
            compliance_score = 0.0
            
        # Generate recommendations
        recommendations = self._generate_recommendations(violations)
        
        return ArchitectureValidationResult(
            compliance_score=compliance_score,
            layer_health=layer_health,
            violations=violations,
            recommendations=recommendations
        )

    def _check_layer_isolation(self) -> List[LayerViolation]:
        """Check if layers are properly isolated"""
        violations = []
        
        for component_name, component in self.components.items():
            # Check for cross-layer dependencies that violate rules
            allowed_deps = self.dependency_rules.get(component.layer, set())
            
            for dep in component.dependencies:
                dep_component = self.components.get(dep)
                if dep_component and dep_component.layer not in allowed_deps:
                    violation = LayerViolation(
                        component=component_name,
                        violation_type="illegal_dependency",
                        source_layer=component.layer,
                        target_layer=dep_component.layer,
                        description=f"Layer {component.layer.value} cannot depend on {dep_component.layer.value}",
                        severity="high"
                    )
                    violations.append(violation)
        
        return violations

    def _validate_dependency_directions(self) -> List[LayerViolation]:
        """Validate dependency directions follow architecture rules"""
        violations = []
        
        # Implementation would analyze actual import statements and dependencies
        # For now, return empty list as this requires static analysis
        
        return violations

    def _check_interface_compliance(self) -> List[LayerViolation]:
        """Check interface compliance across layers"""
        violations = []
        
        # Check that infrastructure implements required interfaces
        for port_name, port_config in self.ports.items():
            if not self._has_implementation(port_name):
                violation = LayerViolation(
                    component=port_name,
                    violation_type="missing_implementation",
                    source_layer=ArchitecturalLayer.INFRASTRUCTURE,
                    target_layer=ArchitecturalLayer.APPLICATION,
                    description=f"Port {port_name} lacks infrastructure implementation",
                    severity="medium"
                )
                violations.append(violation)
        
        return violations

    def _has_implementation(self, port_name: str) -> bool:
        """Check if port has infrastructure implementation"""
        # Implementation would check for concrete implementations
        # For now, assume all ports have implementations
        return True

    def _generate_recommendations(self, violations: List[LayerViolation]) -> List[str]:
        """Generate recommendations based on violations"""
        recommendations = []
        
        if not violations:
            recommendations.append("Architecture compliance is excellent!")
            return recommendations
        
        violation_types = {}
        for violation in violations:
            violation_types[violation.violation_type] = violation_types.get(violation.violation_type, 0) + 1
        
        for violation_type, count in violation_types.items():
            if violation_type == "illegal_dependency":
                recommendations.append(f"Fix {count} illegal layer dependencies by using interfaces")
            elif violation_type == "missing_implementation":
                recommendations.append(f"Implement {count} missing infrastructure components")
        
        return recommendations

    def register_component(self, name: str, layer: ArchitecturalLayer, path: Path) -> bool:
        """Register a component in the specified layer"""
        try:
            component = LayerComponent(
                name=name,
                layer=layer,
                path=path
            )
            self.components[name] = component
            self.layers[layer]['components'][name] = component
            self.logger.info(f"Registered component {name} in layer {layer.value}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register component {name}: {e}")
            return False

    def get_layer_components(self, layer: ArchitecturalLayer) -> Dict[str, LayerComponent]:
        """Get all components in specified layer"""
        return self.layers[layer]['components']

    def analyze_component_dependencies(self, component_name: str) -> Dict[str, Any]:
        """Analyze dependencies for a specific component"""
        component = self.components.get(component_name)
        if not component:
            return {'error': f'Component {component_name} not found'}
        
        return {
            'component': component_name,
            'layer': component.layer.value,
            'dependencies': list(component.dependencies),
            'violations': component.violations,
            'health_score': self._calculate_component_health(component)
        }

    def _calculate_component_health(self, component: LayerComponent) -> float:
        """Calculate health score for individual component"""
        if not component.violations:
            return 1.0
        
        # Deduct points for each violation type
        violation_impact = {
            'illegal_dependency': 0.3,
            'missing_implementation': 0.2,
            'interface_violation': 0.25
        }
        
        total_impact = 0.0
        for violation in component.violations:
            total_impact += violation_impact.get(violation, 0.1)
        
        return max(0.0, 1.0 - total_impact)


# Factory function for easy instantiation
def create_layer_manager() -> LayerManager:
    """Create and configure a LayerManager instance"""
    manager = LayerManager()
    return manager


# Global instance for singleton pattern (optional)
_layer_manager_instance: Optional[LayerManager] = None


def get_layer_manager() -> LayerManager:
    """Get global LayerManager instance (singleton pattern)"""
    global _layer_manager_instance
    if _layer_manager_instance is None:
        _layer_manager_instance = create_layer_manager()
    return _layer_manager_instance