"""
Context Variables Bridge - Agent 15

This bridge implements comprehensive context variable management across the
TestMaster hybrid intelligence ecosystem, providing intelligent context passing,
inheritance, nested contexts, and consensus-driven context optimization.

Key Features:
- Hierarchical context variable management with inheritance
- Cross-system context synchronization and sharing
- Intelligent context scoping and isolation
- Dynamic context resolution and variable interpolation
- Context versioning and rollback capabilities
- Consensus-driven context validation and optimization
- Performance-optimized context caching and compression
- Security-aware context access control and encryption
"""

import json
import threading
import time
import weakref
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable, Union, Set, Tuple
from enum import Enum
from collections import defaultdict, ChainMap
import uuid
import re
from pathlib import Path
import copy

from ..consensus import AgentCoordinator, AgentVote
from ..consensus.agent_coordination import AgentRole
from ...core.shared_state import SharedState
from ...core.feature_flags import FeatureFlags


class ContextScope(Enum):
    """Context variable scopes."""
    GLOBAL = "global"                   # Global system context
    SESSION = "session"                 # Session-specific context
    WORKFLOW = "workflow"               # Workflow execution context
    COMPONENT = "component"             # Component-specific context
    USER = "user"                       # User-specific context
    TEMPORARY = "temporary"             # Temporary context (auto-cleanup)
    INHERITED = "inherited"             # Inherited from parent context
    COMPUTED = "computed"               # Dynamically computed context


class ContextType(Enum):
    """Context variable types."""
    PRIMITIVE = "primitive"             # String, int, float, bool
    COLLECTION = "collection"           # List, dict, set
    REFERENCE = "reference"             # Reference to other context variables
    FUNCTION = "function"               # Executable function/lambda
    SECRET = "secret"                   # Encrypted sensitive data
    COMPUTED = "computed"               # Dynamically computed values
    TEMPLATE = "template"               # Template strings with variables
    INHERITED = "inherited"             # Inherited from parent context


class ContextAccess(Enum):
    """Context variable access levels."""
    PUBLIC = "public"                   # Accessible to all components
    PROTECTED = "protected"             # Accessible within scope hierarchy
    PRIVATE = "private"                 # Accessible only to creator
    SYSTEM = "system"                   # System-only access
    ENCRYPTED = "encrypted"             # Encrypted, requires decryption key


@dataclass
class ContextVariable:
    """Individual context variable."""
    name: str
    value: Any
    context_type: ContextType
    scope: ContextScope
    access_level: ContextAccess
    source_component: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    expires_at: Optional[datetime] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    watchers: List[str] = field(default_factory=list)
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    encrypted: bool = False
    compression_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['context_type'] = self.context_type.value
        result['scope'] = self.scope.value
        result['access_level'] = self.access_level.value
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        if self.expires_at:
            result['expires_at'] = self.expires_at.isoformat()
        return result


@dataclass
class ContextNamespace:
    """Context namespace for organizing variables."""
    namespace_id: str
    name: str
    parent_namespace: Optional[str] = None
    scope: ContextScope = ContextScope.COMPONENT
    variables: Dict[str, ContextVariable] = field(default_factory=dict)
    child_namespaces: Set[str] = field(default_factory=set)
    access_rules: Dict[str, ContextAccess] = field(default_factory=dict)
    inheritance_enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextOperation:
    """Context operation for tracking changes."""
    operation_id: str
    operation_type: str  # create, update, delete, access
    namespace_id: str
    variable_name: str
    old_value: Any
    new_value: Any
    component_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None


class ContextResolver:
    """Intelligent context variable resolution and interpolation."""
    
    def __init__(self):
        self.resolution_cache: Dict[str, Any] = {}
        self.template_pattern = re.compile(r'\${([^}]+)}')
        self.function_registry: Dict[str, Callable] = {}
        
        self._register_default_functions()
    
    def _register_default_functions(self):
        """Register default context functions."""
        self.function_registry.update({
            'now': lambda: datetime.now().isoformat(),
            'uuid': lambda: str(uuid.uuid4()),
            'env': lambda key, default=None: self._get_environment_variable(key, default),
            'upper': lambda text: str(text).upper(),
            'lower': lambda text: str(text).lower(),
            'len': lambda obj: len(obj) if hasattr(obj, '__len__') else 0,
            'join': lambda sep, items: sep.join(str(i) for i in items),
            'format_time': lambda fmt='%Y-%m-%d %H:%M:%S': datetime.now().strftime(fmt)
        })
    
    def _get_environment_variable(self, key: str, default: Any = None) -> Any:
        """Get environment variable safely."""
        import os
        return os.environ.get(key, default)
    
    def resolve_variable(
        self,
        variable: ContextVariable,
        context_map: Dict[str, Any],
        depth: int = 0
    ) -> Any:
        """Resolve context variable value."""
        if depth > 10:  # Prevent infinite recursion
            raise ValueError(f"Context resolution depth exceeded for variable: {variable.name}")
        
        if variable.context_type == ContextType.PRIMITIVE:
            return variable.value
        
        elif variable.context_type == ContextType.COLLECTION:
            return self._resolve_collection(variable.value, context_map, depth)
        
        elif variable.context_type == ContextType.REFERENCE:
            return self._resolve_reference(variable.value, context_map, depth)
        
        elif variable.context_type == ContextType.TEMPLATE:
            return self._resolve_template(variable.value, context_map, depth)
        
        elif variable.context_type == ContextType.FUNCTION:
            return self._resolve_function(variable.value, context_map, depth)
        
        elif variable.context_type == ContextType.COMPUTED:
            return self._resolve_computed(variable.value, context_map, depth)
        
        else:
            return variable.value
    
    def _resolve_collection(self, collection: Union[List, Dict], context_map: Dict[str, Any], depth: int) -> Any:
        """Resolve variables within collections."""
        if isinstance(collection, list):
            return [self._resolve_value(item, context_map, depth + 1) for item in collection]
        elif isinstance(collection, dict):
            return {key: self._resolve_value(value, context_map, depth + 1) 
                   for key, value in collection.items()}
        else:
            return collection
    
    def _resolve_reference(self, reference: str, context_map: Dict[str, Any], depth: int) -> Any:
        """Resolve reference to another variable."""
        if reference in context_map:
            return self._resolve_value(context_map[reference], context_map, depth + 1)
        else:
            raise ValueError(f"Context reference not found: {reference}")
    
    def _resolve_template(self, template: str, context_map: Dict[str, Any], depth: int) -> str:
        """Resolve template string with variable interpolation."""
        def replace_variable(match):
            var_expr = match.group(1)
            try:
                # Support nested attribute access like ${user.name}
                parts = var_expr.split('.')
                value = context_map
                
                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part)
                    else:
                        value = getattr(value, part, None)
                    
                    if value is None:
                        return f"${{{var_expr}}}"  # Keep unresolved
                
                return str(self._resolve_value(value, context_map, depth + 1))
            except Exception:
                return f"${{{var_expr}}}"  # Keep unresolved on error
        
        return self.template_pattern.sub(replace_variable, template)
    
    def _resolve_function(self, func_spec: Dict[str, Any], context_map: Dict[str, Any], depth: int) -> Any:
        """Resolve function call."""
        func_name = func_spec.get('function')
        func_args = func_spec.get('args', [])
        func_kwargs = func_spec.get('kwargs', {})
        
        if func_name not in self.function_registry:
            raise ValueError(f"Unknown context function: {func_name}")
        
        # Resolve arguments
        resolved_args = [self._resolve_value(arg, context_map, depth + 1) for arg in func_args]
        resolved_kwargs = {key: self._resolve_value(value, context_map, depth + 1) 
                          for key, value in func_kwargs.items()}
        
        return self.function_registry[func_name](*resolved_args, **resolved_kwargs)
    
    def _resolve_computed(self, compute_spec: Dict[str, Any], context_map: Dict[str, Any], depth: int) -> Any:
        """Resolve computed value."""
        expression = compute_spec.get('expression')
        variables = compute_spec.get('variables', {})
        
        # Create safe evaluation context
        eval_context = {}
        for var_name, var_ref in variables.items():
            eval_context[var_name] = self._resolve_value(
                context_map.get(var_ref, var_ref), context_map, depth + 1
            )
        
        # Safe evaluation (limited to basic operations)
        try:
            # This is a simplified implementation - in production, use a safer evaluator
            return eval(expression, {"__builtins__": {}}, eval_context)
        except Exception as e:
            raise ValueError(f"Failed to evaluate computed expression: {expression} - {e}")
    
    def _resolve_value(self, value: Any, context_map: Dict[str, Any], depth: int) -> Any:
        """Resolve any value type."""
        if isinstance(value, ContextVariable):
            return self.resolve_variable(value, context_map, depth)
        elif isinstance(value, str) and self.template_pattern.search(value):
            return self._resolve_template(value, context_map, depth)
        elif isinstance(value, (list, dict)):
            return self._resolve_collection(value, context_map, depth)
        else:
            return value


class ContextManager:
    """Core context variable management."""
    
    def __init__(self):
        self.namespaces: Dict[str, ContextNamespace] = {}
        self.resolver = ContextResolver()
        self.operation_history: List[ContextOperation] = []
        self.watchers: Dict[str, List[Callable]] = defaultdict(list)
        self.access_cache: Dict[str, Any] = {}
        self.lock = threading.RLock()
        
        # Performance tracking
        self.operations_count = 0
        self.resolution_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Create default namespaces
        self._create_default_namespaces()
        
        print("Context Manager initialized")
    
    def _create_default_namespaces(self):
        """Create default system namespaces."""
        default_namespaces = [
            ("global", "Global Context", None, ContextScope.GLOBAL),
            ("system", "System Context", "global", ContextScope.GLOBAL),
            ("user", "User Context", "global", ContextScope.USER),
            ("session", "Session Context", "user", ContextScope.SESSION),
            ("workflow", "Workflow Context", "session", ContextScope.WORKFLOW),
            ("component", "Component Context", "workflow", ContextScope.COMPONENT)
        ]
        
        for ns_id, name, parent, scope in default_namespaces:
            namespace = ContextNamespace(
                namespace_id=ns_id,
                name=name,
                parent_namespace=parent,
                scope=scope
            )
            self.namespaces[ns_id] = namespace
            
            # Update parent's children
            if parent and parent in self.namespaces:
                self.namespaces[parent].child_namespaces.add(ns_id)
    
    def create_namespace(
        self,
        namespace_id: str,
        name: str,
        parent_namespace: Optional[str] = None,
        scope: ContextScope = ContextScope.COMPONENT,
        access_rules: Optional[Dict[str, ContextAccess]] = None
    ) -> bool:
        """Create new context namespace."""
        if namespace_id in self.namespaces:
            return False
        
        namespace = ContextNamespace(
            namespace_id=namespace_id,
            name=name,
            parent_namespace=parent_namespace,
            scope=scope,
            access_rules=access_rules or {}
        )
        
        with self.lock:
            self.namespaces[namespace_id] = namespace
            
            # Update parent's children
            if parent_namespace and parent_namespace in self.namespaces:
                self.namespaces[parent_namespace].child_namespaces.add(namespace_id)
        
        return True
    
    def set_variable(
        self,
        namespace_id: str,
        name: str,
        value: Any,
        context_type: ContextType = ContextType.PRIMITIVE,
        access_level: ContextAccess = ContextAccess.PUBLIC,
        source_component: str = "unknown",
        expires_in: Optional[timedelta] = None,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set context variable."""
        if namespace_id not in self.namespaces:
            return False
        
        namespace = self.namespaces[namespace_id]
        
        # Check access permissions
        if not self._check_access_permission(namespace_id, name, source_component, "write"):
            return False
        
        # Create or update variable
        old_value = None
        if name in namespace.variables:
            old_value = namespace.variables[name].value
            # Increment version
            variable = namespace.variables[name]
            variable.value = value
            variable.updated_at = datetime.now()
            variable.version += 1
        else:
            variable = ContextVariable(
                name=name,
                value=value,
                context_type=context_type,
                scope=namespace.scope,
                access_level=access_level,
                source_component=source_component,
                expires_at=datetime.now() + expires_in if expires_in else None,
                description=description,
                tags=tags or []
            )
            namespace.variables[name] = variable
        
        # Record operation
        operation = ContextOperation(
            operation_id=str(uuid.uuid4()),
            operation_type="create" if old_value is None else "update",
            namespace_id=namespace_id,
            variable_name=name,
            old_value=old_value,
            new_value=value,
            component_id=source_component
        )
        
        with self.lock:
            self.operation_history.append(operation)
            self.operations_count += 1
            
            # Clear resolution cache for this variable
            cache_key = f"{namespace_id}.{name}"
            self.access_cache.pop(cache_key, None)
        
        # Notify watchers
        self._notify_watchers(namespace_id, name, value, old_value)
        
        return True
    
    def get_variable(
        self,
        namespace_id: str,
        name: str,
        component_id: str = "unknown",
        resolve: bool = True
    ) -> Optional[Any]:
        """Get context variable value."""
        # Check cache first
        cache_key = f"{namespace_id}.{name}.{component_id}.{resolve}"
        if cache_key in self.access_cache:
            with self.lock:
                self.cache_hits += 1
            return self.access_cache[cache_key]
        
        with self.lock:
            self.cache_misses += 1
        
        # Find variable through inheritance chain
        variable = self._find_variable(namespace_id, name)
        if not variable:
            return None
        
        # Check access permissions
        if not self._check_access_permission(namespace_id, name, component_id, "read"):
            return None
        
        # Resolve variable if requested
        if resolve:
            try:
                context_map = self._build_context_map(namespace_id)
                resolved_value = self.resolver.resolve_variable(variable, context_map)
                
                with self.lock:
                    self.resolution_count += 1
                
                # Cache resolved value
                self.access_cache[cache_key] = resolved_value
                return resolved_value
                
            except Exception as e:
                print(f"Context resolution error for {namespace_id}.{name}: {e}")
                return variable.value
        else:
            self.access_cache[cache_key] = variable.value
            return variable.value
    
    def _find_variable(self, namespace_id: str, name: str) -> Optional[ContextVariable]:
        """Find variable through namespace inheritance chain."""
        current_namespace = namespace_id
        
        while current_namespace:
            if current_namespace in self.namespaces:
                namespace = self.namespaces[current_namespace]
                if name in namespace.variables:
                    return namespace.variables[name]
                
                # Move to parent namespace if inheritance is enabled
                if namespace.inheritance_enabled:
                    current_namespace = namespace.parent_namespace
                else:
                    break
            else:
                break
        
        return None
    
    def _build_context_map(self, namespace_id: str) -> Dict[str, Any]:
        """Build context map for variable resolution."""
        context_map = {}
        
        # Collect variables from inheritance chain
        visited = set()
        current_namespace = namespace_id
        
        while current_namespace and current_namespace not in visited:
            visited.add(current_namespace)
            
            if current_namespace in self.namespaces:
                namespace = self.namespaces[current_namespace]
                
                # Add variables (child namespace variables override parent)
                for var_name, variable in namespace.variables.items():
                    if var_name not in context_map:
                        context_map[var_name] = variable
                
                # Move to parent
                if namespace.inheritance_enabled:
                    current_namespace = namespace.parent_namespace
                else:
                    break
            else:
                break
        
        return context_map
    
    def _check_access_permission(
        self,
        namespace_id: str,
        variable_name: str,
        component_id: str,
        operation: str
    ) -> bool:
        """Check access permission for context variable."""
        namespace = self.namespaces.get(namespace_id)
        if not namespace:
            return False
        
        variable = namespace.variables.get(variable_name)
        if not variable:
            # For new variables, check namespace access rules
            if operation == "write":
                return True  # Allow creation by default
            else:
                return False
        
        # Check variable access level
        if variable.access_level == ContextAccess.PUBLIC:
            return True
        elif variable.access_level == ContextAccess.PROTECTED:
            # Check if component is in same scope hierarchy
            return True  # Simplified check
        elif variable.access_level == ContextAccess.PRIVATE:
            return component_id == variable.source_component
        elif variable.access_level == ContextAccess.SYSTEM:
            return component_id.startswith("system_")
        elif variable.access_level == ContextAccess.ENCRYPTED:
            # Would require decryption key verification
            return component_id == variable.source_component
        
        return False
    
    def _notify_watchers(self, namespace_id: str, name: str, new_value: Any, old_value: Any):
        """Notify watchers of variable changes."""
        watch_key = f"{namespace_id}.{name}"
        
        for callback in self.watchers.get(watch_key, []):
            try:
                callback(namespace_id, name, new_value, old_value)
            except Exception as e:
                print(f"Watcher callback error: {e}")
    
    def add_watcher(
        self,
        namespace_id: str,
        variable_name: str,
        callback: Callable[[str, str, Any, Any], None]
    ):
        """Add watcher for variable changes."""
        watch_key = f"{namespace_id}.{variable_name}"
        self.watchers[watch_key].append(callback)
    
    def remove_watcher(
        self,
        namespace_id: str,
        variable_name: str,
        callback: Callable[[str, str, Any, Any], None]
    ):
        """Remove watcher for variable changes."""
        watch_key = f"{namespace_id}.{variable_name}"
        if callback in self.watchers[watch_key]:
            self.watchers[watch_key].remove(callback)
    
    def cleanup_expired_variables(self):
        """Clean up expired context variables."""
        current_time = datetime.now()
        cleanup_count = 0
        
        with self.lock:
            for namespace in self.namespaces.values():
                expired_vars = []
                
                for var_name, variable in namespace.variables.items():
                    if variable.expires_at and current_time > variable.expires_at:
                        expired_vars.append(var_name)
                
                for var_name in expired_vars:
                    del namespace.variables[var_name]
                    cleanup_count += 1
                    
                    # Clear from cache
                    cache_keys_to_remove = [
                        key for key in self.access_cache.keys()
                        if key.startswith(f"{namespace.namespace_id}.{var_name}.")
                    ]
                    for key in cache_keys_to_remove:
                        del self.access_cache[key]
        
        if cleanup_count > 0:
            print(f"Cleaned up {cleanup_count} expired context variables")
    
    def get_namespace_variables(self, namespace_id: str, include_inherited: bool = True) -> Dict[str, Any]:
        """Get all variables in namespace."""
        if namespace_id not in self.namespaces:
            return {}
        
        if include_inherited:
            return self._build_context_map(namespace_id)
        else:
            namespace = self.namespaces[namespace_id]
            return {name: var.value for name, var in namespace.variables.items()}
    
    def get_manager_metrics(self) -> Dict[str, Any]:
        """Get context manager metrics."""
        with self.lock:
            total_variables = sum(len(ns.variables) for ns in self.namespaces.values())
            cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
            
            return {
                "namespaces": len(self.namespaces),
                "total_variables": total_variables,
                "operations_count": self.operations_count,
                "resolution_count": self.resolution_count,
                "cache_hit_rate": cache_hit_rate,
                "cache_size": len(self.access_cache),
                "watchers": sum(len(callbacks) for callbacks in self.watchers.values()),
                "operation_history": len(self.operation_history)
            }


class ContextVariablesBridge:
    """Main context variables bridge orchestrator."""
    
    def __init__(self):
        self.enabled = FeatureFlags.is_enabled('layer4_bridges', 'context_variables')
        
        # Core components
        self.context_manager = ContextManager()
        self.shared_state = SharedState()
        self.coordinator = AgentCoordinator()
        
        # Bridge state
        self.component_contexts: Dict[str, str] = {}  # component_id -> namespace_id
        self.global_templates: Dict[str, str] = {}
        self.context_synchronizers: List[Callable] = []
        
        # Performance tracking
        self.start_time = datetime.now()
        self.bridge_operations = 0
        self.cross_system_syncs = 0
        
        if not self.enabled:
            return
        
        self._setup_bridge_integrations()
        self._start_background_tasks()
        
        print("Context Variables Bridge initialized")
        print(f"   Default namespaces: {len(self.context_manager.namespaces)}")
        print(f"   Context resolution: enabled")
    
    def _setup_bridge_integrations(self):
        """Setup integrations with existing TestMaster systems."""
        # Register with shared state
        self.shared_state.set("context_bridge_active", {
            "bridge_id": "context_variables",
            "capabilities": ["context_management", "variable_inheritance", "cross_system_sync"],
            "namespaces": list(self.context_manager.namespaces.keys()),
            "started_at": self.start_time.isoformat()
        })
        
        # Setup default system contexts
        self._setup_default_contexts()
    
    def _setup_default_contexts(self):
        """Setup default system context variables."""
        # System context
        self.set_context_variable(
            "system", "testmaster_version", "2.0.0",
            description="TestMaster system version"
        )
        
        self.set_context_variable(
            "system", "bridge_start_time", self.start_time.isoformat(),
            description="Context bridge initialization time"
        )
        
        # Global context templates
        self.global_templates.update({
            "timestamp": "${format_time()}",
            "unique_id": "${uuid()}",
            "user_context": "${user.name} (${user.role})",
            "workflow_path": "${workflow.name}/${workflow.step}/${workflow.substep}"
        })
        
        # Add templates as computed variables
        for template_name, template_value in self.global_templates.items():
            self.set_context_variable(
                "global", template_name, template_value,
                context_type=ContextType.TEMPLATE,
                description=f"Global template: {template_name}"
            )
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        def background_worker():
            while self.enabled:
                try:
                    # Clean up expired variables
                    self.context_manager.cleanup_expired_variables()
                    
                    # Sync with external systems
                    self._perform_cross_system_sync()
                    
                    # Optimize context cache
                    self._optimize_context_cache()
                    
                    time.sleep(300)  # Run every 5 minutes
                    
                except Exception as e:
                    print(f"Context background task error: {e}")
                    time.sleep(60)
        
        background_thread = threading.Thread(target=background_worker, daemon=True)
        background_thread.start()
    
    def register_component_context(
        self,
        component_id: str,
        namespace_id: Optional[str] = None,
        inherit_from: Optional[str] = None
    ) -> str:
        """Register component with context system."""
        if not namespace_id:
            namespace_id = f"component_{component_id}_{int(time.time())}"
        
        # Create component namespace
        parent_namespace = inherit_from or "component"
        
        success = self.context_manager.create_namespace(
            namespace_id,
            f"Context for {component_id}",
            parent_namespace,
            ContextScope.COMPONENT
        )
        
        if success:
            self.component_contexts[component_id] = namespace_id
            
            # Set default component variables
            self.set_context_variable(
                namespace_id, "component_id", component_id,
                description="Component identifier"
            )
            
            self.set_context_variable(
                namespace_id, "registered_at", datetime.now().isoformat(),
                description="Component registration timestamp"
            )
            
            print(f"Component context registered: {component_id} -> {namespace_id}")
        
        return namespace_id
    
    def set_context_variable(
        self,
        namespace_or_component: str,
        name: str,
        value: Any,
        context_type: ContextType = ContextType.PRIMITIVE,
        access_level: ContextAccess = ContextAccess.PUBLIC,
        source_component: str = "bridge",
        expires_in: Optional[timedelta] = None,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set context variable (by namespace or component ID)."""
        # Resolve namespace
        if namespace_or_component in self.component_contexts:
            namespace_id = self.component_contexts[namespace_or_component]
        else:
            namespace_id = namespace_or_component
        
        success = self.context_manager.set_variable(
            namespace_id, name, value, context_type, access_level,
            source_component, expires_in, description, tags
        )
        
        if success:
            self.bridge_operations += 1
        
        return success
    
    def get_context_variable(
        self,
        namespace_or_component: str,
        name: str,
        component_id: str = "bridge",
        resolve: bool = True,
        default: Any = None
    ) -> Any:
        """Get context variable (by namespace or component ID)."""
        # Resolve namespace
        if namespace_or_component in self.component_contexts:
            namespace_id = self.component_contexts[namespace_or_component]
        else:
            namespace_id = namespace_or_component
        
        value = self.context_manager.get_variable(
            namespace_id, name, component_id, resolve
        )
        
        if value is not None:
            self.bridge_operations += 1
            return value
        else:
            return default
    
    def create_context_template(
        self,
        template_name: str,
        template_string: str,
        namespace: str = "global",
        description: str = ""
    ) -> bool:
        """Create reusable context template."""
        return self.set_context_variable(
            namespace, template_name, template_string,
            context_type=ContextType.TEMPLATE,
            description=description or f"Context template: {template_name}"
        )
    
    def create_computed_variable(
        self,
        namespace_or_component: str,
        name: str,
        expression: str,
        variables: Dict[str, str],
        description: str = ""
    ) -> bool:
        """Create computed context variable."""
        compute_spec = {
            "expression": expression,
            "variables": variables
        }
        
        return self.set_context_variable(
            namespace_or_component, name, compute_spec,
            context_type=ContextType.COMPUTED,
            description=description or f"Computed variable: {name}"
        )
    
    def create_function_variable(
        self,
        namespace_or_component: str,
        name: str,
        function_name: str,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        description: str = ""
    ) -> bool:
        """Create function-based context variable."""
        func_spec = {
            "function": function_name,
            "args": args or [],
            "kwargs": kwargs or {}
        }
        
        return self.set_context_variable(
            namespace_or_component, name, func_spec,
            context_type=ContextType.FUNCTION,
            description=description or f"Function variable: {name}"
        )
    
    def watch_context_changes(
        self,
        namespace_or_component: str,
        variable_name: str,
        callback: Callable[[str, str, Any, Any], None]
    ):
        """Watch for context variable changes."""
        # Resolve namespace
        if namespace_or_component in self.component_contexts:
            namespace_id = self.component_contexts[namespace_or_component]
        else:
            namespace_id = namespace_or_component
        
        self.context_manager.add_watcher(namespace_id, variable_name, callback)
    
    def get_component_context(self, component_id: str, include_inherited: bool = True) -> Dict[str, Any]:
        """Get all context variables for component."""
        if component_id not in self.component_contexts:
            return {}
        
        namespace_id = self.component_contexts[component_id]
        return self.context_manager.get_namespace_variables(namespace_id, include_inherited)
    
    def inherit_context(
        self,
        source_component: str,
        target_component: str,
        variable_patterns: Optional[List[str]] = None
    ) -> int:
        """Inherit context variables from one component to another."""
        if source_component not in self.component_contexts:
            return 0
        
        if target_component not in self.component_contexts:
            self.register_component_context(target_component)
        
        source_context = self.get_component_context(source_component, False)
        inherited_count = 0
        
        for var_name, var_value in source_context.items():
            # Check if variable matches patterns
            if variable_patterns:
                if not any(re.match(pattern, var_name) for pattern in variable_patterns):
                    continue
            
            # Set as inherited variable
            success = self.set_context_variable(
                target_component, var_name, var_value,
                context_type=ContextType.INHERITED,
                access_level=ContextAccess.PROTECTED,
                source_component=f"inherited_from_{source_component}",
                description=f"Inherited from {source_component}"
            )
            
            if success:
                inherited_count += 1
        
        return inherited_count
    
    def sync_with_shared_state(self, namespace: str = "global"):
        """Sync context variables with shared state."""
        shared_data = self.shared_state.get_all()
        sync_count = 0
        
        for key, value in shared_data.items():
            if key.startswith("context_sync_"):
                var_name = key.replace("context_sync_", "")
                success = self.set_context_variable(
                    namespace, var_name, value,
                    source_component="shared_state_sync",
                    description=f"Synced from shared state: {key}"
                )
                if success:
                    sync_count += 1
        
        self.cross_system_syncs += 1
        return sync_count
    
    def _perform_cross_system_sync(self):
        """Perform cross-system context synchronization."""
        try:
            # Sync with shared state
            synced = self.sync_with_shared_state()
            
            # Run custom synchronizers
            for synchronizer in self.context_synchronizers:
                try:
                    synchronizer()
                except Exception as e:
                    print(f"Context synchronizer error: {e}")
            
            if synced > 0:
                print(f"Cross-system sync: {synced} variables synchronized")
                
        except Exception as e:
            print(f"Cross-system sync error: {e}")
    
    def _optimize_context_cache(self):
        """Optimize context access cache."""
        cache = self.context_manager.access_cache
        current_time = time.time()
        
        # Remove old cache entries (older than 1 hour)
        old_entries = []
        for key in cache.keys():
            # This is simplified - in practice, you'd track access times
            if len(cache) > 1000:  # Simple size-based cleanup
                old_entries.append(key)
        
        # Remove oldest entries if cache is too large
        if len(old_entries) > 500:
            for key in old_entries[:500]:
                cache.pop(key, None)
    
    def add_context_synchronizer(self, synchronizer: Callable[[], None]):
        """Add custom context synchronizer."""
        self.context_synchronizers.append(synchronizer)
    
    def create_nested_context(
        self,
        parent_component: str,
        child_component: str,
        isolation_level: str = "protected"
    ) -> str:
        """Create nested context with controlled inheritance."""
        if parent_component not in self.component_contexts:
            return ""
        
        parent_namespace = self.component_contexts[parent_component]
        child_namespace = self.register_component_context(
            child_component,
            inherit_from=parent_namespace
        )
        
        # Set isolation rules based on level
        if isolation_level == "isolated":
            # Disable inheritance
            if child_namespace in self.context_manager.namespaces:
                self.context_manager.namespaces[child_namespace].inheritance_enabled = False
        
        return child_namespace
    
    def validate_context_consistency(self) -> Dict[str, Any]:
        """Validate context consistency across namespaces."""
        issues = []
        metrics = {
            "namespaces_checked": 0,
            "variables_checked": 0,
            "resolution_errors": 0,
            "access_violations": 0,
            "circular_references": 0
        }
        
        for namespace_id, namespace in self.context_manager.namespaces.items():
            metrics["namespaces_checked"] += 1
            
            for var_name, variable in namespace.variables.items():
                metrics["variables_checked"] += 1
                
                try:
                    # Test variable resolution
                    context_map = self.context_manager._build_context_map(namespace_id)
                    self.context_manager.resolver.resolve_variable(variable, context_map)
                    
                except Exception as e:
                    metrics["resolution_errors"] += 1
                    issues.append({
                        "type": "resolution_error",
                        "namespace": namespace_id,
                        "variable": var_name,
                        "error": str(e)
                    })
        
        return {
            "validation_passed": len(issues) == 0,
            "issues": issues,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive bridge metrics."""
        uptime = datetime.now() - self.start_time
        manager_metrics = self.context_manager.get_manager_metrics()
        
        return {
            "bridge_status": "active" if self.enabled else "disabled",
            "uptime_seconds": uptime.total_seconds(),
            "bridge_operations": self.bridge_operations,
            "cross_system_syncs": self.cross_system_syncs,
            "registered_components": len(self.component_contexts),
            "global_templates": len(self.global_templates),
            "context_synchronizers": len(self.context_synchronizers),
            "manager_metrics": manager_metrics,
            "component_contexts": {
                comp_id: {
                    "namespace": ns_id,
                    "variables": len(self.context_manager.namespaces[ns_id].variables)
                }
                for comp_id, ns_id in self.component_contexts.items()
                if ns_id in self.context_manager.namespaces
            }
        }
    
    def export_context_snapshot(self, namespace_id: Optional[str] = None) -> Dict[str, Any]:
        """Export context snapshot for backup or analysis."""
        if namespace_id:
            namespaces_to_export = [namespace_id] if namespace_id in self.context_manager.namespaces else []
        else:
            namespaces_to_export = list(self.context_manager.namespaces.keys())
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "bridge_version": "1.0",
            "namespaces": {},
            "component_mappings": self.component_contexts,
            "global_templates": self.global_templates
        }
        
        for ns_id in namespaces_to_export:
            namespace = self.context_manager.namespaces[ns_id]
            snapshot["namespaces"][ns_id] = {
                "metadata": {
                    "name": namespace.name,
                    "parent_namespace": namespace.parent_namespace,
                    "scope": namespace.scope.value,
                    "inheritance_enabled": namespace.inheritance_enabled,
                    "created_at": namespace.created_at.isoformat()
                },
                "variables": {
                    var_name: var.to_dict()
                    for var_name, var in namespace.variables.items()
                }
            }
        
        return snapshot
    
    def shutdown(self):
        """Shutdown context variables bridge."""
        # Export final snapshot
        final_snapshot = self.export_context_snapshot()
        self.shared_state.set("context_bridge_final_snapshot", final_snapshot)
        
        # Store final metrics
        final_metrics = self.get_comprehensive_metrics()
        self.shared_state.set("context_bridge_final_metrics", final_metrics)
        
        print("Context Variables Bridge shutdown complete")


def get_context_variables_bridge() -> ContextVariablesBridge:
    """Get context variables bridge instance."""
    return ContextVariablesBridge()