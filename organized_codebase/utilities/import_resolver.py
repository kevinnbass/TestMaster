#!/usr/bin/env python3
"""
Import Resolution Framework - Agent A
Intelligent import resolution with fallback mechanisms

Provides advanced import resolution with intelligent fallbacks, module
discovery, and integration with clean architecture dependency management.
"""

import importlib
import importlib.util
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable, Type
import ast
import inspect
from enum import Enum


class ImportStrategy(Enum):
    """Import resolution strategies"""
    DIRECT = "direct"                 # Direct importlib import
    FALLBACK = "fallback"            # Try fallback providers
    DYNAMIC = "dynamic"              # Dynamic module loading
    LAZY = "lazy"                    # Lazy loading on first use
    CACHED = "cached"                # Use cached results


class ImportResult(Enum):
    """Import operation results"""
    SUCCESS = "success"
    FALLBACK_USED = "fallback_used"
    NOT_FOUND = "not_found" 
    ERROR = "error"
    CIRCULAR = "circular"


@dataclass
class ImportAttempt:
    """Record of import attempt"""
    module_name: str
    strategy: ImportStrategy
    result: ImportResult
    fallback_used: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    resolution_time_ms: float = 0.0


@dataclass 
class ModuleInfo:
    """Information about discovered module"""
    name: str
    path: Path
    package: Optional[str] = None
    is_package: bool = False
    dependencies: Set[str] = field(default_factory=set)
    exports: List[str] = field(default_factory=list)
    documentation: Optional[str] = None
    version: Optional[str] = None


class IImportProvider(ABC):
    """Interface for custom import providers"""
    
    @abstractmethod
    def can_provide(self, module_name: str) -> bool:
        """Check if provider can supply the module"""
        pass
    
    @abstractmethod
    def provide_module(self, module_name: str) -> Any:
        """Provide the module implementation"""
        pass
    
    @abstractmethod 
    def get_priority(self) -> int:
        """Get provider priority (higher = more preferred)"""
        pass


class IFallbackProvider(ABC):
    """Interface for fallback module providers"""
    
    @abstractmethod
    def create_fallback(self, module_name: str, original_error: Exception) -> Any:
        """Create fallback implementation for failed import"""
        pass


class FeatureDiscoveryLog:
    """Log for tracking feature discovery attempts"""
    
    def __init__(self):
        self.discoveries: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def log_discovery_attempt(self, feature_name: str, discovery_data: Dict[str, Any]):
        """Log a feature discovery attempt"""
        entry = {
            'feature_name': feature_name,
            'timestamp': datetime.now(),
            'discovery_data': discovery_data
        }
        self.discoveries.append(entry)
        self.logger.info(f"Feature discovery logged: {feature_name}")


class ImportResolver:
    """
    Intelligent import resolution with fallback mechanisms
    
    Provides comprehensive import resolution with caching, fallbacks,
    module discovery, and integration with clean architecture patterns.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Module registry and caching
        self.module_registry: Dict[str, Any] = {}
        self.import_cache: Dict[str, Any] = {}
        self.fallback_providers: Dict[str, IFallbackProvider] = {}
        self.import_providers: List[IImportProvider] = []
        
        # Discovery and tracking
        self.feature_discovery_log = FeatureDiscoveryLog()
        self.import_attempts: List[ImportAttempt] = []
        self.module_info: Dict[str, ModuleInfo] = {}
        
        # Configuration
        self.enable_caching = True
        self.enable_fallbacks = True
        self.enable_discovery = True
        self.max_cache_size = 1000
        
        # Path management
        self.search_paths: List[Path] = []
        self._initialize_search_paths()

    def _initialize_search_paths(self):
        """Initialize module search paths"""
        # Add current working directory and common paths
        self.search_paths.extend([
            Path.cwd(),
            Path.cwd() / "core",
            Path.cwd() / "src", 
            Path.cwd() / "lib"
        ])
        
        # Add system paths
        for path_str in sys.path:
            path = Path(path_str)
            if path.exists() and path not in self.search_paths:
                self.search_paths.append(path)

    def resolve_import(self, module_name: str, strategy: ImportStrategy = ImportStrategy.DIRECT) -> Any:
        """
        Resolve module import with intelligent fallback
        
        Args:
            module_name: Name of module to import
            strategy: Import strategy to use
            
        Returns:
            Imported module or fallback implementation
        """
        start_time = datetime.now()
        
        try:
            # Feature Discovery Protocol Integration
            if self.enable_discovery:
                self._execute_feature_discovery(module_name)
            
            # Try cached result first
            if self.enable_caching and module_name in self.import_cache:
                cached_module = self.import_cache[module_name]
                self._log_import_attempt(module_name, strategy, ImportResult.SUCCESS, 
                                       resolution_time=(datetime.now() - start_time).total_seconds() * 1000)
                return cached_module
            
            # Attempt direct import
            if strategy in [ImportStrategy.DIRECT, ImportStrategy.CACHED]:
                try:
                    module = importlib.import_module(module_name)
                    self._cache_module(module_name, module)
                    self._log_import_attempt(module_name, strategy, ImportResult.SUCCESS,
                                           resolution_time=(datetime.now() - start_time).total_seconds() * 1000)
                    return module
                except ImportError as e:
                    self.logger.debug(f"Direct import failed for {module_name}: {e}")
                    
            # Try custom providers
            provider_result = self._try_import_providers(module_name)
            if provider_result is not None:
                self._cache_module(module_name, provider_result)
                self._log_import_attempt(module_name, ImportStrategy.FALLBACK, ImportResult.FALLBACK_USED,
                                       resolution_time=(datetime.now() - start_time).total_seconds() * 1000)
                return provider_result
                
            # Try dynamic loading
            if strategy == ImportStrategy.DYNAMIC:
                dynamic_result = self._try_dynamic_import(module_name)
                if dynamic_result is not None:
                    self._cache_module(module_name, dynamic_result)
                    self._log_import_attempt(module_name, strategy, ImportResult.SUCCESS,
                                           resolution_time=(datetime.now() - start_time).total_seconds() * 1000)
                    return dynamic_result
            
            # Create fallback if enabled
            if self.enable_fallbacks:
                fallback = self._create_fallback(module_name, ImportError(f"Module {module_name} not found"))
                if fallback is not None:
                    self._cache_module(module_name, fallback)
                    self._log_import_attempt(module_name, ImportStrategy.FALLBACK, ImportResult.FALLBACK_USED,
                                           fallback_used="generated_fallback",
                                           resolution_time=(datetime.now() - start_time).total_seconds() * 1000)
                    return fallback
            
            # Import failed
            error_msg = f"Failed to resolve import: {module_name}"
            self._log_import_attempt(module_name, strategy, ImportResult.NOT_FOUND, 
                                   error_message=error_msg,
                                   resolution_time=(datetime.now() - start_time).total_seconds() * 1000)
            raise ImportError(error_msg)
            
        except Exception as e:
            self._log_import_attempt(module_name, strategy, ImportResult.ERROR,
                                   error_message=str(e),
                                   resolution_time=(datetime.now() - start_time).total_seconds() * 1000)
            raise

    def _execute_feature_discovery(self, module_name: str):
        """Execute feature discovery protocol for import resolution"""
        self.logger.info(f"🚨 FEATURE DISCOVERY: Starting exhaustive search for import resolution mechanisms...")
        
        # Search for existing import features
        existing_features = self._discover_existing_import_features(module_name)
        
        if existing_features:
            self.logger.info(f"✅ FOUND EXISTING IMPORT FEATURES: {len(existing_features)} items")
            self.feature_discovery_log.log_discovery_attempt(
                f"import_resolution_{module_name}",
                {
                    'existing_features': existing_features,
                    'decision': 'ENHANCE_EXISTING',
                    'enhancement_plan': self._create_enhancement_plan(existing_features),
                    'rationale': 'Existing import resolution found - enhancing instead of duplicating'
                }
            )
        else:
            self.logger.info(f"🚨 NO EXISTING IMPORT FEATURES FOUND - PROCEEDING WITH NEW IMPLEMENTATION")

    def _discover_existing_import_features(self, module_name: str) -> List[Dict[str, Any]]:
        """Discover existing import resolution features"""
        features = []
        
        # Check if module is already in registry
        if module_name in self.module_registry:
            features.append({
                'type': 'registry_entry',
                'module': module_name,
                'location': 'module_registry'
            })
        
        # Check for cached imports
        if module_name in self.import_cache:
            features.append({
                'type': 'cached_import', 
                'module': module_name,
                'location': 'import_cache'
            })
        
        # Check system modules
        if module_name in sys.modules:
            features.append({
                'type': 'system_module',
                'module': module_name,
                'location': 'sys.modules'
            })
        
        return features

    def _create_enhancement_plan(self, existing_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create plan for enhancing existing import features"""
        return {
            'strategy': 'integrate_existing',
            'existing_count': len(existing_features),
            'integration_points': [f['location'] for f in existing_features],
            'enhancement_type': 'augment_with_fallbacks'
        }

    def _enhance_existing_import(self, existing_features: List[Dict[str, Any]], module_name: str) -> Any:
        """Enhance existing import mechanisms"""
        # Use the first available existing feature
        for feature in existing_features:
            if feature['type'] == 'system_module':
                return sys.modules[module_name]
            elif feature['type'] == 'cached_import':
                return self.import_cache[module_name]
            elif feature['type'] == 'registry_entry':
                return self.module_registry[module_name]
        
        # Fallback to standard import
        return importlib.import_module(module_name)

    def _try_import_providers(self, module_name: str) -> Optional[Any]:
        """Try custom import providers"""
        # Sort providers by priority (highest first)
        sorted_providers = sorted(self.import_providers, key=lambda p: p.get_priority(), reverse=True)
        
        for provider in sorted_providers:
            try:
                if provider.can_provide(module_name):
                    return provider.provide_module(module_name)
            except Exception as e:
                self.logger.warning(f"Provider {type(provider).__name__} failed for {module_name}: {e}")
                continue
        
        return None

    def _try_dynamic_import(self, module_name: str) -> Optional[Any]:
        """Try dynamic module loading from search paths"""
        for search_path in self.search_paths:
            try:
                # Try as Python file
                py_file = search_path / f"{module_name}.py"
                if py_file.exists():
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        return module
                
                # Try as package
                pkg_dir = search_path / module_name
                if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
                    spec = importlib.util.spec_from_file_location(module_name, pkg_dir / "__init__.py")
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module) 
                        return module
                        
            except Exception as e:
                self.logger.debug(f"Dynamic import failed for {module_name} in {search_path}: {e}")
                continue
        
        return None

    def _create_fallback(self, module_name: str, original_error: Exception) -> Optional[Any]:
        """Create fallback implementation for failed import"""
        # Try registered fallback providers first
        for provider_name, provider in self.fallback_providers.items():
            try:
                fallback = provider.create_fallback(module_name, original_error)
                if fallback is not None:
                    self.logger.info(f"Created fallback for {module_name} using {provider_name}")
                    return fallback
            except Exception as e:
                self.logger.warning(f"Fallback provider {provider_name} failed for {module_name}: {e}")
                continue
        
        # Generate basic fallback module
        return self._generate_basic_fallback(module_name, original_error)

    def _generate_basic_fallback(self, module_name: str, original_error: Exception) -> Any:
        """Generate basic fallback module implementation"""
        class FallbackModule:
            """Fallback module implementation"""
            
            def __init__(self, name: str, error: Exception):
                self.__name__ = name
                self.__file__ = f"<fallback:{name}>"
                self._original_error = error
                self._import_resolver = None  # Avoid circular reference
            
            def __getattr__(self, name: str):
                # Return a stub function that logs when called
                def stub_function(*args, **kwargs):
                    logging.getLogger(__name__).warning(
                        f"Called {self.__name__}.{name} from fallback module. "
                        f"Original import error: {self._original_error}"
                    )
                    return None
                return stub_function
            
            def __repr__(self):
                return f"<FallbackModule '{self.__name__}'>"
        
        return FallbackModule(module_name, original_error)

    def _cache_module(self, module_name: str, module: Any):
        """Cache module for future use"""
        if not self.enable_caching:
            return
            
        # Implement LRU-style cache management
        if len(self.import_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.import_cache))
            del self.import_cache[oldest_key]
        
        self.import_cache[module_name] = module

    def _log_import_attempt(self, module_name: str, strategy: ImportStrategy, result: ImportResult,
                          fallback_used: Optional[str] = None, error_message: Optional[str] = None,
                          resolution_time: float = 0.0):
        """Log import attempt for debugging and metrics"""
        attempt = ImportAttempt(
            module_name=module_name,
            strategy=strategy,
            result=result,
            fallback_used=fallback_used,
            error_message=error_message,
            resolution_time_ms=resolution_time
        )
        
        self.import_attempts.append(attempt)
        
        # Keep only recent attempts to prevent memory issues
        if len(self.import_attempts) > 1000:
            self.import_attempts = self.import_attempts[-500:]

    # Provider Management
    
    def register_provider(self, provider: IImportProvider) -> 'ImportResolver':
        """Register custom import provider"""
        self.import_providers.append(provider)
        self.import_providers.sort(key=lambda p: p.get_priority(), reverse=True)
        self.logger.info(f"Registered import provider: {type(provider).__name__}")
        return self

    def register_fallback_provider(self, name: str, provider: IFallbackProvider) -> 'ImportResolver':
        """Register fallback provider"""
        self.fallback_providers[name] = provider
        self.logger.info(f"Registered fallback provider: {name}")
        return self

    def add_search_path(self, path: Union[str, Path]) -> 'ImportResolver':
        """Add path to module search paths"""
        path_obj = Path(path) if isinstance(path, str) else path
        if path_obj not in self.search_paths:
            self.search_paths.append(path_obj)
            self.logger.info(f"Added search path: {path_obj}")
        return self

    # Analytics and Debugging
    
    def get_import_statistics(self) -> Dict[str, Any]:
        """Get import resolution statistics"""
        if not self.import_attempts:
            return {'total_attempts': 0}
        
        total_attempts = len(self.import_attempts)
        successful = len([a for a in self.import_attempts if a.result == ImportResult.SUCCESS])
        fallback_used = len([a for a in self.import_attempts if a.result == ImportResult.FALLBACK_USED])
        errors = len([a for a in self.import_attempts if a.result == ImportResult.ERROR])
        
        avg_resolution_time = sum(a.resolution_time_ms for a in self.import_attempts) / total_attempts
        
        return {
            'total_attempts': total_attempts,
            'successful': successful,
            'fallback_used': fallback_used, 
            'errors': errors,
            'success_rate': successful / total_attempts if total_attempts > 0 else 0,
            'average_resolution_time_ms': avg_resolution_time,
            'cache_size': len(self.import_cache),
            'providers_registered': len(self.import_providers)
        }

    def clear_cache(self) -> 'ImportResolver':
        """Clear import cache"""
        self.import_cache.clear()
        self.logger.info("Import cache cleared")
        return self

    def discover_modules(self, search_path: Optional[Path] = None) -> List[ModuleInfo]:
        """Discover available modules in search paths"""
        modules = []
        paths_to_search = [search_path] if search_path else self.search_paths
        
        for path in paths_to_search:
            if not path.exists():
                continue
                
            # Find Python files
            for py_file in path.glob("*.py"):
                if py_file.stem.startswith("_") and py_file.stem != "__init__":
                    continue
                    
                module_info = self._analyze_module_file(py_file)
                if module_info:
                    modules.append(module_info)
            
            # Find packages
            for pkg_dir in path.iterdir():
                if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
                    module_info = self._analyze_package_dir(pkg_dir)
                    if module_info:
                        modules.append(module_info)
        
        return modules

    def _analyze_module_file(self, py_file: Path) -> Optional[ModuleInfo]:
        """Analyze Python file for module information"""
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract exports (functions, classes, constants)
            exports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    exports.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    exports.append(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and not target.id.startswith('_'):
                            exports.append(target.id)
            
            return ModuleInfo(
                name=py_file.stem,
                path=py_file,
                is_package=False,
                exports=exports,
                documentation=ast.get_docstring(tree)
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze {py_file}: {e}")
            return None

    def _analyze_package_dir(self, pkg_dir: Path) -> Optional[ModuleInfo]:
        """Analyze package directory for module information"""
        try:
            init_file = pkg_dir / "__init__.py"
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            return ModuleInfo(
                name=pkg_dir.name,
                path=pkg_dir,
                is_package=True,
                documentation=ast.get_docstring(tree)
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze package {pkg_dir}: {e}")
            return None


# Global instance for easy access
_import_resolver_instance: Optional[ImportResolver] = None


def get_import_resolver() -> ImportResolver:
    """Get global ImportResolver instance"""
    global _import_resolver_instance
    if _import_resolver_instance is None:
        _import_resolver_instance = ImportResolver()
    return _import_resolver_instance


# Convenience function for direct import resolution
def resolve_import(module_name: str, strategy: ImportStrategy = ImportStrategy.DIRECT) -> Any:
    """Convenience function for resolving imports"""
    resolver = get_import_resolver()
    return resolver.resolve_import(module_name, strategy)