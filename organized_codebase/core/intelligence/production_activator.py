#!/usr/bin/env python3
"""
Production Intelligence Activator - Agent A Hour 5
Moves production intelligence modules from PRODUCTION_PACKAGES to active deployment

Integrates predictive intelligence modules with the architecture framework
and activates them for production use.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import importlib.util
import sys

# Import architecture framework
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.architecture_integration import (
    get_architecture_framework,
    ArchitecturalLayer,
    LifetimeScope
)
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.service_registry import (
    get_service_registry,
    ServiceDefinition,
    ServiceType
)


class ActivationStatus(Enum):
    """Status of module activation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ModuleActivation:
    """Result of module activation"""
    module_name: str
    source_path: Path
    target_path: Path
    status: ActivationStatus
    error_message: Optional[str] = None
    dependencies: List[str] = None
    activation_time: datetime = None
    
    def __post_init__(self):
        if self.activation_time is None:
            self.activation_time = datetime.now()
        if self.dependencies is None:
            self.dependencies = []


class ProductionIntelligenceActivator:
    """
    Production Intelligence Module Activator
    
    Moves intelligence modules from PRODUCTION_PACKAGES to active deployment
    and integrates them with the architecture framework.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Architecture components
        self.framework = get_architecture_framework()
        self.service_registry = get_service_registry()
        
        # Path configuration
        self.production_source = Path("PRODUCTION_PACKAGES/TestMaster_Production_v20250821_200633/core/intelligence")
        self.target_base = Path("core/intelligence")
        
        # Module mapping
        self.priority_modules = {
            'predictive_intelligence_core.py': {
                'target': 'predictive/intelligence_core.py',
                'layer': ArchitecturalLayer.APPLICATION,
                'service_type': ServiceType.INTELLIGENCE
            },
            'predictive_types.py': {
                'target': 'predictive/types.py',
                'layer': ArchitecturalLayer.DOMAIN,
                'service_type': ServiceType.INTELLIGENCE
            },
            'code_predictor.py': {
                'target': 'predictive/code_predictor.py',
                'layer': ArchitecturalLayer.APPLICATION,
                'service_type': ServiceType.INTELLIGENCE
            },
            'language_bridge.py': {
                'target': 'predictive/language_bridge.py',
                'layer': ArchitecturalLayer.INFRASTRUCTURE,
                'service_type': ServiceType.INTELLIGENCE
            },
            'pattern_detector.py': {
                'target': 'analysis/pattern_detector.py',
                'layer': ArchitecturalLayer.APPLICATION,
                'service_type': ServiceType.ANALYTICS
            },
            'meta_orchestrator_core.py': {
                'target': 'orchestration/meta_orchestrator.py',
                'layer': ArchitecturalLayer.APPLICATION,
                'service_type': ServiceType.INTELLIGENCE
            }
        }
        
        # Activation tracking
        self.activation_results: List[ModuleActivation] = []
        self.activated_services: List[str] = []
        
        # Statistics
        self.stats = {
            'modules_discovered': 0,
            'modules_activated': 0,
            'modules_failed': 0,
            'services_registered': 0,
            'start_time': datetime.now()
        }
        
        self.logger.info("Production Intelligence Activator initialized")
    
    def discover_production_modules(self) -> List[Path]:
        """Discover available production modules"""
        modules = []
        
        if not self.production_source.exists():
            self.logger.warning(f"Production source not found: {self.production_source}")
            return modules
        
        try:
            # Find Python modules in production directory
            for module_file in self.production_source.rglob("*.py"):
                if module_file.name.startswith("__"):
                    continue
                    
                # Check if it's in our priority list or has intelligence keywords
                if (module_file.name in self.priority_modules or
                    any(keyword in module_file.name.lower() for keyword in 
                        ['intelligence', 'predictor', 'analyzer', 'orchestrator'])):
                    modules.append(module_file)
            
            self.stats['modules_discovered'] = len(modules)
            self.logger.info(f"Discovered {len(modules)} production modules")
            
        except Exception as e:
            self.logger.error(f"Failed to discover production modules: {e}")
        
        return modules
    
    def activate_production_modules(self) -> List[ModuleActivation]:
        """Activate discovered production modules"""
        self.logger.info("Starting production module activation...")
        
        modules = self.discover_production_modules()
        
        # Process priority modules first
        for module_path in modules:
            try:
                result = self._activate_single_module(module_path)
                self.activation_results.append(result)
                
                if result.status == ActivationStatus.COMPLETED:
                    self.stats['modules_activated'] += 1
                else:
                    self.stats['modules_failed'] += 1
                    
            except Exception as e:
                error_result = ModuleActivation(
                    module_name=module_path.name,
                    source_path=module_path,
                    target_path=Path("unknown"),
                    status=ActivationStatus.FAILED,
                    error_message=str(e)
                )
                self.activation_results.append(error_result)
                self.stats['modules_failed'] += 1
        
        # Register activated modules as services
        self._register_activated_services()
        
        success_count = len([r for r in self.activation_results if r.status == ActivationStatus.COMPLETED])
        total_count = len(self.activation_results)
        
        self.logger.info(f"Module activation complete: {success_count}/{total_count} successful")
        
        return self.activation_results
    
    def _activate_single_module(self, module_path: Path) -> ModuleActivation:
        """Activate a single production module"""
        module_name = module_path.name
        
        try:
            # Determine target path
            if module_name in self.priority_modules:
                config = self.priority_modules[module_name]
                target_path = self.target_base / config['target']
            else:
                # Default placement for non-priority modules
                relative_path = module_path.relative_to(self.production_source)
                target_path = self.target_base / "production" / relative_path
            
            # Create target directory
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if target already exists and is newer
            if target_path.exists():
                source_mtime = module_path.stat().st_mtime
                target_mtime = target_path.stat().st_mtime
                
                if target_mtime >= source_mtime:
                    return ModuleActivation(
                        module_name=module_name,
                        source_path=module_path,
                        target_path=target_path,
                        status=ActivationStatus.SKIPPED,
                        error_message="Target is newer or same age as source"
                    )
            
            # Copy module to target
            shutil.copy2(module_path, target_path)
            
            # Validate activation
            if self._validate_activation(target_path):
                self.logger.info(f"Activated module: {module_name} -> {target_path}")
                
                return ModuleActivation(
                    module_name=module_name,
                    source_path=module_path,
                    target_path=target_path,
                    status=ActivationStatus.COMPLETED,
                    dependencies=self._analyze_dependencies(target_path)
                )
            else:
                return ModuleActivation(
                    module_name=module_name,
                    source_path=module_path,
                    target_path=target_path,
                    status=ActivationStatus.FAILED,
                    error_message="Validation failed after activation"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to activate {module_name}: {e}")
            return ModuleActivation(
                module_name=module_name,
                source_path=module_path,
                target_path=target_path if 'target_path' in locals() else Path("unknown"),
                status=ActivationStatus.FAILED,
                error_message=str(e)
            )
    
    def _validate_activation(self, target_path: Path) -> bool:
        """Validate that module activation was successful"""
        try:
            # Check file exists and is readable
            if not target_path.exists() or not target_path.is_file():
                return False
            
            # Try to parse as Python module
            with open(target_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic syntax validation
            try:
                compile(content, str(target_path), 'exec')
                return True
            except SyntaxError:
                return False
                
        except Exception as e:
            self.logger.warning(f"Validation failed for {target_path}: {e}")
            return False
    
    def _analyze_dependencies(self, module_path: Path) -> List[str]:
        """Analyze module dependencies"""
        dependencies = []
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple import extraction
            import re
            
            # Find import statements
            import_patterns = [
                r'^from\s+([^\s]+)\s+import',
                r'^import\s+([^\s,]+)'
            ]
            
            for line in content.split('\n'):
                line = line.strip()
                for pattern in import_patterns:
                    match = re.match(pattern, line)
                    if match:
                        module_name = match.group(1).split('.')[0]
                        if module_name not in dependencies:
                            dependencies.append(module_name)
        
        except Exception as e:
            self.logger.warning(f"Failed to analyze dependencies for {module_path}: {e}")
        
        return dependencies
    
    def _register_activated_services(self):
        """Register activated modules as services"""
        try:
            for result in self.activation_results:
                if result.status != ActivationStatus.COMPLETED:
                    continue
                
                # Get module configuration
                config = self.priority_modules.get(result.module_name, {
                    'layer': ArchitecturalLayer.APPLICATION,
                    'service_type': ServiceType.INTELLIGENCE
                })
                
                # Generate service name
                service_name = result.module_name.replace('.py', '').replace('_', '-')
                
                # Define service
                self.service_registry.define_service(ServiceDefinition(
                    name=service_name,
                    service_type=config['service_type'],
                    layer=config['layer'],
                    lifetime=LifetimeScope.SINGLETON,
                    metadata={
                        'module_path': str(result.target_path),
                        'activated_from': str(result.source_path),
                        'activation_time': result.activation_time.isoformat(),
                        'dependencies': result.dependencies,
                        'production_module': True
                    }
                ))
                
                self.activated_services.append(service_name)
            
            self.stats['services_registered'] = len(self.activated_services)
            self.logger.info(f"Registered {len(self.activated_services)} activated modules as services")
            
        except Exception as e:
            self.logger.error(f"Failed to register activated services: {e}")
    
    def test_activated_modules(self) -> Dict[str, Any]:
        """Test activated modules for basic functionality"""
        test_results = {}
        
        for result in self.activation_results:
            if result.status != ActivationStatus.COMPLETED:
                test_results[result.module_name] = {
                    'status': 'skipped',
                    'reason': 'Module not activated'
                }
                continue
            
            try:
                # Attempt to import the module
                module_spec = importlib.util.spec_from_file_location(
                    result.module_name.replace('.py', ''),
                    result.target_path
                )
                
                if module_spec and module_spec.loader:
                    module = importlib.util.module_from_spec(module_spec)
                    
                    # Add to sys.modules temporarily for testing
                    original_module = sys.modules.get(module_spec.name)
                    sys.modules[module_spec.name] = module
                    
                    try:
                        module_spec.loader.exec_module(module)
                        test_results[result.module_name] = {
                            'status': 'success',
                            'classes': [name for name in dir(module) if not name.startswith('_') and hasattr(getattr(module, name), '__class__')],
                            'functions': [name for name in dir(module) if not name.startswith('_') and callable(getattr(module, name))]
                        }
                    finally:
                        # Restore original module
                        if original_module is not None:
                            sys.modules[module_spec.name] = original_module
                        elif module_spec.name in sys.modules:
                            del sys.modules[module_spec.name]
                else:
                    test_results[result.module_name] = {
                        'status': 'failed',
                        'error': 'Could not create module spec'
                    }
                    
            except Exception as e:
                test_results[result.module_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return test_results
    
    def get_activation_report(self) -> Dict[str, Any]:
        """Get comprehensive activation report"""
        successful = [r for r in self.activation_results if r.status == ActivationStatus.COMPLETED]
        failed = [r for r in self.activation_results if r.status == ActivationStatus.FAILED]
        skipped = [r for r in self.activation_results if r.status == ActivationStatus.SKIPPED]
        
        total_time = (datetime.now() - self.stats['start_time']).total_seconds()
        
        return {
            'summary': {
                'modules_discovered': self.stats['modules_discovered'],
                'modules_activated': len(successful),
                'modules_failed': len(failed),
                'modules_skipped': len(skipped),
                'services_registered': self.stats['services_registered'],
                'success_rate': len(successful) / max(len(self.activation_results), 1),
                'total_time_seconds': total_time
            },
            'successful_activations': [
                {
                    'module_name': r.module_name,
                    'target_path': str(r.target_path),
                    'dependencies': len(r.dependencies)
                }
                for r in successful
            ],
            'failed_activations': [
                {
                    'module_name': r.module_name,
                    'error_message': r.error_message
                }
                for r in failed
            ],
            'activated_services': self.activated_services,
            'production_source': str(self.production_source),
            'target_base': str(self.target_base),
            'timestamp': datetime.now().isoformat()
        }


# Global activator instance
_production_activator: Optional[ProductionIntelligenceActivator] = None


def get_production_activator() -> ProductionIntelligenceActivator:
    """Get global production activator instance"""
    global _production_activator
    if _production_activator is None:
        _production_activator = ProductionIntelligenceActivator()
    return _production_activator


def activate_production_intelligence() -> Dict[str, Any]:
    """Activate all production intelligence modules"""
    activator = get_production_activator()
    
    # Activate modules
    results = activator.activate_production_modules()
    
    # Test activated modules
    test_results = activator.test_activated_modules()
    
    # Get comprehensive report
    report = activator.get_activation_report()
    
    return {
        'activation_results': results,
        'test_results': test_results,
        'report': report
    }