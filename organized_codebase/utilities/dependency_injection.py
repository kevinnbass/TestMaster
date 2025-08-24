#!/usr/bin/env python3
"""
Dependency Injection Container - Agent A
Advanced dependency injection with lifecycle management

Implements comprehensive dependency injection container with support for
singleton, transient, and scoped lifetimes, automatic resolution, and
clean architecture integration.
"""

import inspect
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, get_type_hints
from dataclasses import dataclass, field
from threading import Lock
import weakref


T = TypeVar('T')


class LifetimeScope(Enum):
    """Dependency lifetime management scopes"""
    SINGLETON = "singleton"      # Single instance for application lifetime
    TRANSIENT = "transient"      # New instance every time
    SCOPED = "scoped"           # Single instance per scope
    REQUEST = "request"          # Single instance per request (web apps)


class InjectionStrategy(Enum):
    """Dependency injection strategies"""
    CONSTRUCTOR = "constructor"   # Constructor injection
    PROPERTY = "property"        # Property injection  
    METHOD = "method"           # Method injection
    FACTORY = "factory"         # Factory-based creation


@dataclass
class DependencyRegistration:
    """Registration information for a dependency"""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    lifetime: LifetimeScope = LifetimeScope.TRANSIENT
    strategy: InjectionStrategy = InjectionStrategy.CONSTRUCTOR
    dependencies: List[Type] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)


@dataclass 
class ResolutionContext:
    """Context information for dependency resolution"""
    requested_type: Type
    resolution_path: List[Type] = field(default_factory=list)
    scope_data: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None


class DependencyResolutionError(Exception):
    """Exception raised when dependency resolution fails"""
    pass


class CircularDependencyError(DependencyResolutionError):
    """Exception raised when circular dependencies are detected"""
    pass


class IDependencyResolver(ABC):
    """Interface for custom dependency resolvers"""
    
    @abstractmethod
    def can_resolve(self, service_type: Type) -> bool:
        """Check if resolver can handle the service type"""
        pass
    
    @abstractmethod
    def resolve(self, service_type: Type, container: 'DependencyContainer') -> Any:
        """Resolve the service instance"""
        pass


class DependencyContainer:
    """
    Advanced dependency injection container
    
    Provides comprehensive dependency injection with lifecycle management,
    automatic resolution, and integration with hexagonal architecture.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._lock = Lock()
        
        # Registration storage
        self._registrations: Dict[Type, DependencyRegistration] = {}
        self._instances: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        
        # Resolution helpers
        self._resolvers: List[IDependencyResolver] = []
        self._resolution_stack: List[Type] = []
        
        # Lifecycle management
        self._singletons: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._scopes: Dict[str, Dict[Type, Any]] = {}
        
        # Built-in registrations
        self._register_built_ins()

    def _register_built_ins(self):
        """Register built-in services"""
        # Register self for dependency injection of the container
        self.register_instance(DependencyContainer, self)
        self.register_instance(type(self), self)

    # Registration Methods
    
    def register_transient(self, service_type: Type[T], implementation_type: Type[T] = None) -> 'DependencyContainer':
        """Register service with transient lifetime (new instance each time)"""
        return self._register_service(
            service_type, 
            implementation_type or service_type,
            LifetimeScope.TRANSIENT
        )

    def register_singleton(self, service_type: Type[T], implementation_type: Type[T] = None) -> 'DependencyContainer':
        """Register service with singleton lifetime (single instance)"""
        return self._register_service(
            service_type,
            implementation_type or service_type, 
            LifetimeScope.SINGLETON
        )

    def register_scoped(self, service_type: Type[T], implementation_type: Type[T] = None) -> 'DependencyContainer':
        """Register service with scoped lifetime (single instance per scope)"""
        return self._register_service(
            service_type,
            implementation_type or service_type,
            LifetimeScope.SCOPED
        )

    def register_instance(self, service_type: Type[T], instance: T) -> 'DependencyContainer':
        """Register specific instance as singleton"""
        with self._lock:
            registration = DependencyRegistration(
                service_type=service_type,
                instance=instance,
                lifetime=LifetimeScope.SINGLETON,
                strategy=InjectionStrategy.CONSTRUCTOR
            )
            self._registrations[service_type] = registration
            self._instances[service_type] = instance
            
            self.logger.debug(f"Registered instance for {service_type.__name__}")
            return self

    def register_factory(self, service_type: Type[T], factory: Callable[..., T], 
                        lifetime: LifetimeScope = LifetimeScope.TRANSIENT) -> 'DependencyContainer':
        """Register factory function for service creation"""
        with self._lock:
            registration = DependencyRegistration(
                service_type=service_type,
                factory=factory,
                lifetime=lifetime,
                strategy=InjectionStrategy.FACTORY
            )
            
            # Analyze factory dependencies
            registration.dependencies = self._analyze_factory_dependencies(factory)
            self._registrations[service_type] = registration
            
            self.logger.debug(f"Registered factory for {service_type.__name__}")
            return self

    def _register_service(self, service_type: Type, implementation_type: Type, 
                         lifetime: LifetimeScope) -> 'DependencyContainer':
        """Internal service registration"""
        with self._lock:
            registration = DependencyRegistration(
                service_type=service_type,
                implementation_type=implementation_type,
                lifetime=lifetime,
                strategy=InjectionStrategy.CONSTRUCTOR
            )
            
            # Analyze constructor dependencies
            registration.dependencies = self._analyze_constructor_dependencies(implementation_type)
            self._registrations[service_type] = registration
            
            self.logger.debug(f"Registered {service_type.__name__} -> {implementation_type.__name__} ({lifetime.value})")
            return self

    def _analyze_constructor_dependencies(self, implementation_type: Type) -> List[Type]:
        """Analyze constructor parameters for dependency injection"""
        try:
            signature = inspect.signature(implementation_type.__init__)
            type_hints = get_type_hints(implementation_type.__init__)
            
            dependencies = []
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                    
                # Get type hint if available
                param_type = type_hints.get(param_name)
                if param_type:
                    dependencies.append(param_type)
                    
            return dependencies
            
        except Exception as e:
            self.logger.warning(f"Could not analyze dependencies for {implementation_type.__name__}: {e}")
            return []

    def _analyze_factory_dependencies(self, factory: Callable) -> List[Type]:
        """Analyze factory function parameters for dependency injection"""
        try:
            signature = inspect.signature(factory)
            type_hints = get_type_hints(factory)
            
            dependencies = []
            for param_name, param in signature.parameters.items():
                param_type = type_hints.get(param_name)
                if param_type:
                    dependencies.append(param_type)
                    
            return dependencies
            
        except Exception as e:
            self.logger.warning(f"Could not analyze factory dependencies: {e}")
            return []

    # Resolution Methods
    
    def resolve(self, service_type: Type[T], context: Optional[ResolutionContext] = None) -> T:
        """Resolve service instance with dependency injection"""
        if context is None:
            context = ResolutionContext(requested_type=service_type)
        
        # Check for circular dependencies
        if service_type in context.resolution_path:
            path = " -> ".join([t.__name__ for t in context.resolution_path + [service_type]])
            raise CircularDependencyError(f"Circular dependency detected: {path}")
        
        try:
            context.resolution_path.append(service_type)
            return self._resolve_internal(service_type, context)
            
        except Exception as e:
            if isinstance(e, (DependencyResolutionError, CircularDependencyError)):
                raise
            raise DependencyResolutionError(f"Failed to resolve {service_type.__name__}: {e}")
            
        finally:
            if service_type in context.resolution_path:
                context.resolution_path.remove(service_type)

    def _resolve_internal(self, service_type: Type[T], context: ResolutionContext) -> T:
        """Internal resolution logic"""
        # Check if already registered
        registration = self._registrations.get(service_type)
        if not registration:
            # Try custom resolvers
            for resolver in self._resolvers:
                if resolver.can_resolve(service_type):
                    return resolver.resolve(service_type, self)
            
            # Auto-registration for concrete classes
            if not inspect.isabstract(service_type) and inspect.isclass(service_type):
                self.register_transient(service_type)
                registration = self._registrations[service_type]
            else:
                raise DependencyResolutionError(f"No registration found for {service_type.__name__}")
        
        # Handle different lifetime scopes
        if registration.lifetime == LifetimeScope.SINGLETON:
            return self._resolve_singleton(service_type, registration, context)
        elif registration.lifetime == LifetimeScope.SCOPED:
            return self._resolve_scoped(service_type, registration, context)
        else:  # TRANSIENT
            return self._create_instance(registration, context)

    def _resolve_singleton(self, service_type: Type[T], registration: DependencyRegistration, 
                          context: ResolutionContext) -> T:
        """Resolve singleton instance"""
        # Check if instance already exists
        if service_type in self._instances:
            return self._instances[service_type]
        
        with self._lock:
            # Double-check after acquiring lock
            if service_type in self._instances:
                return self._instances[service_type]
            
            # Create new singleton instance
            instance = self._create_instance(registration, context)
            self._instances[service_type] = instance
            return instance

    def _resolve_scoped(self, service_type: Type[T], registration: DependencyRegistration,
                       context: ResolutionContext) -> T:
        """Resolve scoped instance"""
        scope_id = context.request_id or "default"
        
        if scope_id not in self._scoped_instances:
            self._scoped_instances[scope_id] = {}
        
        scope_instances = self._scoped_instances[scope_id]
        if service_type in scope_instances:
            return scope_instances[service_type]
        
        # Create new scoped instance
        instance = self._create_instance(registration, context)
        scope_instances[service_type] = instance
        return instance

    def _create_instance(self, registration: DependencyRegistration, context: ResolutionContext) -> Any:
        """Create new instance based on registration"""
        if registration.instance is not None:
            return registration.instance
            
        if registration.factory is not None:
            return self._create_from_factory(registration, context)
        
        if registration.implementation_type is not None:
            return self._create_from_constructor(registration, context)
        
        raise DependencyResolutionError(f"No creation strategy for {registration.service_type.__name__}")

    def _create_from_factory(self, registration: DependencyRegistration, context: ResolutionContext) -> Any:
        """Create instance using factory function"""
        factory_args = []
        
        # Resolve factory dependencies
        for dependency_type in registration.dependencies:
            dependency = self._resolve_internal(dependency_type, context)
            factory_args.append(dependency)
        
        return registration.factory(*factory_args)

    def _create_from_constructor(self, registration: DependencyRegistration, context: ResolutionContext) -> Any:
        """Create instance using constructor injection"""
        constructor_args = []
        
        # Resolve constructor dependencies
        for dependency_type in registration.dependencies:
            dependency = self._resolve_internal(dependency_type, context)
            constructor_args.append(dependency)
        
        return registration.implementation_type(*constructor_args)

    # Utility Methods
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if service type is registered"""
        return service_type in self._registrations

    def add_resolver(self, resolver: IDependencyResolver) -> 'DependencyContainer':
        """Add custom dependency resolver"""
        self._resolvers.append(resolver)
        self.logger.debug(f"Added custom resolver: {type(resolver).__name__}")
        return self

    def create_scope(self, scope_id: str) -> 'DependencyScope':
        """Create new dependency scope"""
        return DependencyScope(self, scope_id)

    def clear_scope(self, scope_id: str) -> None:
        """Clear scoped instances for specific scope"""
        if scope_id in self._scoped_instances:
            del self._scoped_instances[scope_id]
            self.logger.debug(f"Cleared scope: {scope_id}")

    def get_registrations(self) -> Dict[Type, DependencyRegistration]:
        """Get all current registrations (for debugging)"""
        return self._registrations.copy()

    def validate_registrations(self) -> List[str]:
        """Validate all registrations can be resolved"""
        issues = []
        
        for service_type, registration in self._registrations.items():
            try:
                # Attempt to resolve without creating instances
                context = ResolutionContext(requested_type=service_type)
                self._validate_resolution_path(service_type, context)
            except Exception as e:
                issues.append(f"{service_type.__name__}: {e}")
        
        return issues

    def _validate_resolution_path(self, service_type: Type, context: ResolutionContext) -> None:
        """Validate that service type can be resolved without circular dependencies"""
        if service_type in context.resolution_path:
            path = " -> ".join([t.__name__ for t in context.resolution_path + [service_type]])
            raise CircularDependencyError(f"Circular dependency: {path}")
        
        registration = self._registrations.get(service_type)
        if not registration:
            return  # Will be auto-registered or handled by resolvers
        
        context.resolution_path.append(service_type)
        try:
            for dependency_type in registration.dependencies:
                self._validate_resolution_path(dependency_type, context)
        finally:
            context.resolution_path.pop()


class DependencyScope:
    """Manages scoped dependency lifetimes"""
    
    def __init__(self, container: DependencyContainer, scope_id: str):
        self.container = container
        self.scope_id = scope_id

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container.clear_scope(self.scope_id)

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve service within this scope"""
        context = ResolutionContext(
            requested_type=service_type,
            request_id=self.scope_id
        )
        return self.container.resolve(service_type, context)


# Global container instance (optional)
_global_container: Optional[DependencyContainer] = None


def get_container() -> DependencyContainer:
    """Get global dependency container instance"""
    global _global_container
    if _global_container is None:
        _global_container = DependencyContainer()
    return _global_container


def configure_container() -> DependencyContainer:
    """Configure and return dependency container"""
    container = DependencyContainer()
    
    # Add any default configurations here
    
    return container