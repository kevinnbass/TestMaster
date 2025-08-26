"""
Test Integration Hub - Seamless module integration system for TestMaster framework

This hub provides:
- Dynamic module discovery and loading
- Inter-module communication and data flow
- Plugin architecture for extensibility
- Configuration management and validation
"""
import asyncio
import importlib
import inspect
import logging
import os
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref
import sys

# Mock Framework Imports for Testing
import pytest
from unittest.mock import Mock, patch, MagicMock
import unittest

class ModuleType(Enum):
    EXTRACTOR = "extractor"
    GENERATOR = "generator"
    ANALYZER = "analyzer"
    EXECUTOR = "executor"
    REPORTER = "reporter"
    ORCHESTRATOR = "orchestrator"

class IntegrationStatus(Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DEPRECATED = "deprecated"

@dataclass
class ModuleManifest:
    """Module metadata and capabilities descriptor"""
    name: str
    version: str
    module_type: ModuleType
    entry_point: str
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    configuration_schema: Dict[str, Any] = field(default_factory=dict)
    api_endpoints: List[str] = field(default_factory=list)
    data_formats: Dict[str, str] = field(default_factory=dict)
    compatibility_version: str = "1.0"

@dataclass
class ModuleInstance:
    """Runtime module instance container"""
    manifest: ModuleManifest
    instance: Any
    status: IntegrationStatus
    load_time: float
    error_message: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    event_handlers: Dict[str, List[Callable]] = field(default_factory=dict)

class EventBus:
    """Inter-module event communication system"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
    def subscribe(self, event_type: str, handler: Callable) -> str:
        """Subscribe to event type with handler"""
        with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            
            # Use weak references to prevent memory leaks
            weak_handler = weakref.WeakMethod(handler) if hasattr(handler, '__self__') else weakref.ref(handler)
            self.subscribers[event_type].append(weak_handler)
            
            return f"{event_type}_{len(self.subscribers[event_type])}"
    
    def unsubscribe(self, event_type: str, handler: Callable) -> bool:
        """Unsubscribe handler from event type"""
        with self._lock:
            if event_type not in self.subscribers:
                return False
                
            # Remove handler references
            original_count = len(self.subscribers[event_type])
            self.subscribers[event_type] = [
                ref for ref in self.subscribers[event_type] 
                if ref() is not handler
            ]
            
            return len(self.subscribers[event_type]) < original_count
    
    def emit(self, event_type: str, data: Dict[str, Any]) -> int:
        """Emit event to all subscribers"""
        with self._lock:
            if event_type not in self.subscribers:
                return 0
            
            event_data = {
                "type": event_type,
                "timestamp": asyncio.get_event_loop().time(),
                "data": data
            }
            
            self.event_history.append(event_data)
            if len(self.event_history) > 1000:  # Limit history size
                self.event_history = self.event_history[-500:]
            
            successful_calls = 0
            dead_refs = []
            
            for weak_ref in self.subscribers[event_type]:
                handler = weak_ref()
                if handler is None:
                    dead_refs.append(weak_ref)
                    continue
                    
                try:
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.create_task(handler(event_data))
                    else:
                        handler(event_data)
                    successful_calls += 1
                except Exception as e:
                    logging.error(f"Event handler error for {event_type}: {e}")
            
            # Clean up dead references
            for dead_ref in dead_refs:
                self.subscribers[event_type].remove(dead_ref)
            
            return successful_calls

class ConfigurationManager:
    """Centralized configuration management"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("config")
        self.configurations: Dict[str, Dict[str, Any]] = {}
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self._load_configurations()
        
    def _load_configurations(self) -> None:
        """Load all configuration files"""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True)
            return
            
        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    self.configurations[config_file.stem] = config_data
            except Exception as e:
                logging.error(f"Failed to load config {config_file}: {e}")
    
    def register_schema(self, module_name: str, schema: Dict[str, Any]) -> None:
        """Register configuration schema for module"""
        self.schemas[module_name] = schema
        
        # Validate existing configuration against schema
        if module_name in self.configurations:
            self._validate_configuration(module_name, self.configurations[module_name])
    
    def get_configuration(self, module_name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get configuration for module"""
        return self.configurations.get(module_name, default or {})
    
    def set_configuration(self, module_name: str, config: Dict[str, Any]) -> bool:
        """Set configuration for module with validation"""
        if module_name in self.schemas:
            if not self._validate_configuration(module_name, config):
                return False
        
        self.configurations[module_name] = config
        self._save_configuration(module_name, config)
        return True
    
    def _validate_configuration(self, module_name: str, config: Dict[str, Any]) -> bool:
        """Validate configuration against schema"""
        schema = self.schemas.get(module_name)
        if not schema:
            return True  # No schema to validate against
        
        # Simple validation - can be extended with jsonschema
        for required_field in schema.get("required", []):
            if required_field not in config:
                logging.error(f"Missing required field {required_field} in {module_name} config")
                return False
        
        return True
    
    def _save_configuration(self, module_name: str, config: Dict[str, Any]) -> None:
        """Save configuration to file"""
        config_file = self.config_dir / f"{module_name}.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save config {module_name}: {e}")

class ModuleLoader:
    """Dynamic module loading and management"""
    
    def __init__(self, module_paths: List[Path]):
        self.module_paths = module_paths
        self.loaded_modules: Dict[str, ModuleInstance] = {}
        self.manifests: Dict[str, ModuleManifest] = {}
        self._discovery_cache: Dict[str, float] = {}  # Path -> last_modified
        
    def discover_modules(self) -> List[ModuleManifest]:
        """Discover all available modules"""
        discovered_manifests = []
        
        for module_path in self.module_paths:
            if not module_path.exists():
                continue
                
            for py_file in module_path.rglob("*.py"):
                if py_file.name.startswith("test_") or py_file.name.startswith("__"):
                    continue
                    
                try:
                    manifest = self._extract_manifest_from_file(py_file)
                    if manifest:
                        discovered_manifests.append(manifest)
                        self.manifests[manifest.name] = manifest
                except Exception as e:
                    logging.debug(f"Could not extract manifest from {py_file}: {e}")
        
        return discovered_manifests
    
    def _extract_manifest_from_file(self, file_path: Path) -> Optional[ModuleManifest]:
        """Extract module manifest from Python file"""
        try:
            # Read file and look for manifest or docstring patterns
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for TestMaster module patterns
            if "TestMaster" not in content:
                return None
            
            # Extract module type from filename patterns
            module_type = self._infer_module_type(file_path.name)
            if not module_type:
                return None
            
            # Create basic manifest
            manifest = ModuleManifest(
                name=file_path.stem,
                version="1.0.0",
                module_type=module_type,
                entry_point=str(file_path.relative_to(file_path.parents[2])).replace('\\', '.').replace('/', '.')[:-3]
            )
            
            # Extract capabilities from class and function names
            manifest.capabilities = self._extract_capabilities(content)
            
            return manifest
            
        except Exception as e:
            logging.debug(f"Manifest extraction failed for {file_path}: {e}")
            return None
    
    def _infer_module_type(self, filename: str) -> Optional[ModuleType]:
        """Infer module type from filename"""
        type_patterns = {
            ModuleType.EXTRACTOR: ["extractor", "extract", "pattern"],
            ModuleType.GENERATOR: ["generator", "generate", "template"],
            ModuleType.ANALYZER: ["analyzer", "analyze", "intelligence"],
            ModuleType.EXECUTOR: ["executor", "execute", "runner"],
            ModuleType.REPORTER: ["reporter", "report", "analytics"],
            ModuleType.ORCHESTRATOR: ["orchestrator", "orchestrate", "coordinator"]
        }
        
        filename_lower = filename.lower()
        for module_type, patterns in type_patterns.items():
            if any(pattern in filename_lower for pattern in patterns):
                return module_type
        
        return None
    
    def _extract_capabilities(self, content: str) -> List[str]:
        """Extract module capabilities from code content"""
        capabilities = []
        
        # Simple pattern matching for common capabilities
        capability_patterns = {
            "ast_parsing": ["ast.", "parse", "AST"],
            "test_generation": ["generate_test", "create_test"],
            "pattern_analysis": ["analyze_pattern", "pattern_match"],
            "multi_language": ["language", "cross_lang"],
            "async_support": ["async def", "await ", "asyncio"],
            "configuration": ["config", "settings"],
            "reporting": ["report", "metrics", "analytics"],
            "caching": ["cache", "memoize"],
            "validation": ["validate", "check", "verify"]
        }
        
        for capability, patterns in capability_patterns.items():
            if any(pattern in content for pattern in patterns):
                capabilities.append(capability)
        
        return capabilities
    
    def load_module(self, module_name: str) -> Optional[ModuleInstance]:
        """Load and instantiate module"""
        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name]
        
        if module_name not in self.manifests:
            logging.error(f"Module {module_name} not found in manifests")
            return None
        
        manifest = self.manifests[module_name]
        
        try:
            # Import the module
            module = importlib.import_module(manifest.entry_point)
            
            # Find the main class to instantiate
            main_class = self._find_main_class(module, manifest.module_type)
            if not main_class:
                raise ImportError(f"No suitable main class found in {module_name}")
            
            # Instantiate the class
            instance = main_class()
            
            # Create module instance wrapper
            module_instance = ModuleInstance(
                manifest=manifest,
                instance=instance,
                status=IntegrationStatus.LOADED,
                load_time=asyncio.get_event_loop().time()
            )
            
            self.loaded_modules[module_name] = module_instance
            return module_instance
            
        except Exception as e:
            error_instance = ModuleInstance(
                manifest=manifest,
                instance=None,
                status=IntegrationStatus.ERROR,
                load_time=asyncio.get_event_loop().time(),
                error_message=str(e)
            )
            self.loaded_modules[module_name] = error_instance
            logging.error(f"Failed to load module {module_name}: {e}")
            return error_instance
    
    def _find_main_class(self, module: Any, module_type: ModuleType) -> Optional[Type]:
        """Find the main class in a module based on type"""
        type_class_patterns = {
            ModuleType.EXTRACTOR: ["Extractor", "Extract", "Pattern"],
            ModuleType.GENERATOR: ["Generator", "Generate", "Template"],
            ModuleType.ANALYZER: ["Analyzer", "Analyze", "Intelligence"],
            ModuleType.EXECUTOR: ["Executor", "Execute", "Runner"],
            ModuleType.REPORTER: ["Reporter", "Report", "Analytics"],
            ModuleType.ORCHESTRATOR: ["Orchestrator", "Orchestrate", "Coordinator"]
        }
        
        patterns = type_class_patterns.get(module_type, [])
        
        # Find classes that match the patterns
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if any(pattern in name for pattern in patterns):
                # Ensure it's defined in this module (not imported)
                if obj.__module__ == module.__name__:
                    return obj
        
        # Fallback: return the first class defined in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__:
                return obj
        
        return None
    
    def unload_module(self, module_name: str) -> bool:
        """Unload module and clean up resources"""
        if module_name not in self.loaded_modules:
            return False
        
        module_instance = self.loaded_modules[module_name]
        
        # Call cleanup method if available
        if hasattr(module_instance.instance, 'cleanup'):
            try:
                module_instance.instance.cleanup()
            except Exception as e:
                logging.error(f"Error during {module_name} cleanup: {e}")
        
        # Remove from loaded modules
        del self.loaded_modules[module_name]
        
        return True

class TestIntegrationHub:
    """Central integration hub for TestMaster framework"""
    
    def __init__(self, module_paths: Optional[List[Path]] = None, config_dir: Optional[Path] = None):
        self.module_paths = module_paths or [Path("core/testing")]
        self.event_bus = EventBus()
        self.config_manager = ConfigurationManager(config_dir)
        self.module_loader = ModuleLoader(self.module_paths)
        self.integration_status = IntegrationStatus.UNLOADED
        self._integration_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def initialize(self) -> bool:
        """Initialize the integration hub"""
        try:
            # Discover all available modules
            discovered_modules = self.module_loader.discover_modules()
            logging.info(f"Discovered {len(discovered_modules)} modules")
            
            # Register configuration schemas
            for manifest in discovered_modules:
                if manifest.configuration_schema:
                    self.config_manager.register_schema(manifest.name, manifest.configuration_schema)
            
            self.integration_status = IntegrationStatus.LOADED
            
            # Emit initialization event
            self.event_bus.emit("hub.initialized", {
                "module_count": len(discovered_modules),
                "timestamp": asyncio.get_event_loop().time()
            })
            
            return True
            
        except Exception as e:
            logging.error(f"Hub initialization failed: {e}")
            self.integration_status = IntegrationStatus.ERROR
            return False
    
    def load_module(self, module_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """Load and configure a module"""
        try:
            # Set configuration if provided
            if config:
                if not self.config_manager.set_configuration(module_name, config):
                    return False
            
            # Load the module
            module_instance = self.module_loader.load_module(module_name)
            if not module_instance or module_instance.status == IntegrationStatus.ERROR:
                return False
            
            # Configure the module
            module_config = self.config_manager.get_configuration(module_name)
            if hasattr(module_instance.instance, 'configure'):
                module_instance.instance.configure(module_config)
            
            # Register event handlers if the module supports them
            if hasattr(module_instance.instance, 'get_event_handlers'):
                handlers = module_instance.instance.get_event_handlers()
                for event_type, handler in handlers.items():
                    self.event_bus.subscribe(event_type, handler)
            
            module_instance.status = IntegrationStatus.ACTIVE
            
            # Emit module loaded event
            self.event_bus.emit("module.loaded", {
                "module_name": module_name,
                "module_type": module_instance.manifest.module_type.value,
                "capabilities": module_instance.manifest.capabilities
            })
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to load module {module_name}: {e}")
            return False
    
    def get_module(self, module_name: str) -> Optional[Any]:
        """Get loaded module instance"""
        if module_name in self.module_loader.loaded_modules:
            module_instance = self.module_loader.loaded_modules[module_name]
            if module_instance.status == IntegrationStatus.ACTIVE:
                return module_instance.instance
        return None
    
    def list_modules(self, module_type: Optional[ModuleType] = None) -> List[ModuleManifest]:
        """List available modules, optionally filtered by type"""
        modules = list(self.module_loader.manifests.values())
        if module_type:
            modules = [m for m in modules if m.module_type == module_type]
        return modules
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        return {
            "hub_status": self.integration_status.value,
            "total_modules": len(self.module_loader.manifests),
            "loaded_modules": len(self.module_loader.loaded_modules),
            "active_modules": len([m for m in self.module_loader.loaded_modules.values() if m.status == IntegrationStatus.ACTIVE]),
            "event_subscribers": len(self.event_bus.subscribers),
            "configurations": len(self.config_manager.configurations)
        }


# Comprehensive Test Suite
class TestIntegrationHub(unittest.TestCase):
    
    def setUp(self):
        self.hub = TestIntegrationHub()
        
    def test_hub_initialization(self):
        """Test hub initialization"""
        success = self.hub.initialize()
        self.assertTrue(success)
        self.assertEqual(self.hub.integration_status, IntegrationStatus.LOADED)
    
    def test_event_bus_subscription(self):
        """Test event bus functionality"""
        received_events = []
        
        def event_handler(event_data):
            received_events.append(event_data)
        
        # Subscribe to event
        subscription_id = self.hub.event_bus.subscribe("test.event", event_handler)
        self.assertIsNotNone(subscription_id)
        
        # Emit event
        count = self.hub.event_bus.emit("test.event", {"message": "test"})
        self.assertEqual(count, 1)
        self.assertEqual(len(received_events), 1)
        
    def test_configuration_management(self):
        """Test configuration management"""
        config = {"timeout": 30, "retries": 3}
        
        # Set configuration
        success = self.hub.config_manager.set_configuration("test_module", config)
        self.assertTrue(success)
        
        # Get configuration
        retrieved_config = self.hub.config_manager.get_configuration("test_module")
        self.assertEqual(retrieved_config, config)
        
    def test_module_manifest_creation(self):
        """Test module manifest creation"""
        manifest = ModuleManifest(
            name="test_analyzer",
            version="1.0.0",
            module_type=ModuleType.ANALYZER,
            entry_point="core.testing.test_analyzer",
            capabilities=["ast_parsing", "pattern_analysis"]
        )
        
        self.assertEqual(manifest.name, "test_analyzer")
        self.assertEqual(manifest.module_type, ModuleType.ANALYZER)
        self.assertIn("ast_parsing", manifest.capabilities)
        
    def test_integration_status_tracking(self):
        """Test integration status tracking"""
        status = self.hub.get_integration_status()
        
        self.assertIn("hub_status", status)
        self.assertIn("total_modules", status)
        self.assertIn("loaded_modules", status)
        self.assertIn("active_modules", status)


if __name__ == "__main__":
    # Demo usage
    hub = TestIntegrationHub()
    
    if hub.initialize():
        print("TestMaster Integration Hub initialized successfully")
        
        # List discovered modules
        modules = hub.list_modules()
        print(f"Discovered {len(modules)} modules:")
        for module in modules:
            print(f"  - {module.name} ({module.module_type.value})")
        
        # Get status
        status = hub.get_integration_status()
        print(f"Hub Status: {status}")
        
    else:
        print("Failed to initialize Integration Hub")
    
    # Run tests
    pytest.main([__file__, "-v"])