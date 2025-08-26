"""
Configuration Migration Utilities
==================================

Tools to help migrate from scattered Config classes to unified configuration system.
Provides backward compatibility and migration assistance.

Author: Agent E - Infrastructure Consolidation
"""

import ast
import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass

from .unified_config import (
    get_config_manager, ConfigCategory, 
    APIConfig, SecurityConfig, MonitoringConfig, 
    CachingConfig, TestingConfig, MLConfig,
    InfrastructureConfig, IntegrationConfig
)

logger = logging.getLogger(__name__)


@dataclass
class MigrationResult:
    """Result of configuration migration."""
    files_scanned: int
    config_classes_found: int
    migrations_needed: List[str]
    backward_compatible: List[str]
    warnings: List[str]


class ConfigMigrationTool:
    """
    Tool to help migrate scattered Config classes to unified system.
    
    Analyzes codebase for Config class usage and provides migration guidance.
    """
    
    def __init__(self, root_path: Optional[Path] = None):
        self.root_path = root_path or Path.cwd()
        self.config_manager = get_config_manager()
        
        # Known config class patterns to look for
        self.config_patterns = {
            r'class.*APIConfig.*:', ConfigCategory.API,
            r'class.*SecurityConfig.*:', ConfigCategory.SECURITY,
            r'class.*MonitoringConfig.*:', ConfigCategory.MONITORING,
            r'class.*CachingConfig.*:', ConfigCategory.CACHING,
            r'class.*TestingConfig.*:', ConfigCategory.TESTING,
            r'class.*MLConfig.*:', ConfigCategory.ML,
            r'class.*InfrastructureConfig.*:', ConfigCategory.INFRASTRUCTURE,
            r'class.*IntegrationConfig.*:', ConfigCategory.INTEGRATION,
            r'class.*Config\(.*\):', 'generic',
            r'@dataclass.*class.*Config:', 'dataclass_config',
        }
        
        # Import patterns that need migration
        self.import_patterns = [
            r'from.*config.*import.*Config',
            r'import.*config.*Config',
            r'from.*\.config import',
        ]
    
    def analyze_codebase(self) -> MigrationResult:
        """Analyze codebase for configuration usage."""
        result = MigrationResult(
            files_scanned=0,
            config_classes_found=0,
            migrations_needed=[],
            backward_compatible=[],
            warnings=[]
        )
        
        # Find all Python files
        python_files = list(self.root_path.rglob("*.py"))
        result.files_scanned = len(python_files)
        
        for file_path in python_files:
            try:
                self._analyze_file(file_path, result)
            except Exception as e:
                logger.warning(f"Could not analyze {file_path}: {e}")
                result.warnings.append(f"Failed to analyze {file_path}: {e}")
        
        logger.info(f"Analysis complete: {result.config_classes_found} config classes found")
        return result
    
    def _analyze_file(self, file_path: Path, result: MigrationResult):
        """Analyze a single file for configuration usage."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='cp1252', errors='ignore') as f:
                    content = f.read()
            except Exception:
                logger.warning(f"Could not read file {file_path} due to encoding issues")
                result.warnings.append(f"Could not read {file_path}: encoding issues")
                return
        
        # Look for config class definitions
        for pattern, category in self.config_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            
            if matches:
                result.config_classes_found += len(matches)
                relative_path = file_path.relative_to(self.root_path)
                
                if isinstance(category, ConfigCategory):
                    # Specific category found
                    migration_note = f"{relative_path}: Found {category.value} config class - migrate to unified_config.{category.value}"
                    result.migrations_needed.append(migration_note)
                else:
                    # Generic config class
                    migration_note = f"{relative_path}: Generic config class found - review for migration"
                    result.migrations_needed.append(migration_note)
        
        # Look for import statements that might need updating
        for pattern in self.import_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            
            if matches:
                relative_path = file_path.relative_to(self.root_path)
                import_note = f"{relative_path}: Config import found - update to use unified_config"
                result.backward_compatible.append(import_note)
    
    def create_compatibility_layer(self) -> str:
        """Create backward compatibility imports."""
        compatibility_code = '''
"""
Backward Compatibility Layer for Configuration
==============================================

Provides backward compatibility for existing config imports.
Gradually migrate to use unified_config directly.
"""

# Import unified config
from .unified_config import (
    get_config_manager, ConfigCategory,
    APIConfig as UnifiedAPIConfig,
    SecurityConfig as UnifiedSecurityConfig,
    MonitoringConfig as UnifiedMonitoringConfig,
    CachingConfig as UnifiedCachingConfig,
    TestingConfig as UnifiedTestingConfig,
    MLConfig as UnifiedMLConfig,
    InfrastructureConfig as UnifiedInfrastructureConfig,
    IntegrationConfig as UnifiedIntegrationConfig
)

# Backward compatible aliases
APIConfig = UnifiedAPIConfig
SecurityConfig = UnifiedSecurityConfig  
MonitoringConfig = UnifiedMonitoringConfig
CachingConfig = UnifiedCachingConfig
TestingConfig = UnifiedTestingConfig
MLConfig = UnifiedMLConfig
InfrastructureConfig = UnifiedInfrastructureConfig
IntegrationConfig = UnifiedIntegrationConfig

# Convenience functions for backward compatibility
def get_api_config():
    """Get API configuration (backward compatible)."""
    return get_config_manager().get_api_config()

def get_security_config():
    """Get security configuration (backward compatible).""" 
    return get_config_manager().get_security_config()

def get_monitoring_config():
    """Get monitoring configuration (backward compatible)."""
    return get_config_manager().get_monitoring_config()

def get_caching_config():
    """Get caching configuration (backward compatible)."""
    return get_config_manager().get_caching_config()

def get_testing_config():
    """Get testing configuration (backward compatible)."""
    return get_config_manager().get_testing_config()

def get_ml_config():
    """Get ML configuration (backward compatible)."""
    return get_config_manager().get_ml_config()

def get_infrastructure_config():
    """Get infrastructure configuration (backward compatible)."""
    return get_config_manager().get_infrastructure_config()

def get_integration_config():
    """Get integration configuration (backward compatible)."""
    return get_config_manager().get_integration_config()

# Legacy config instance (for modules that expect config objects)
class LegacyConfigBridge:
    """Bridge for legacy config usage patterns."""
    
    @property
    def api(self):
        return get_api_config()
    
    @property  
    def security(self):
        return get_security_config()
    
    @property
    def monitoring(self):
        return get_monitoring_config()
    
    @property
    def caching(self):
        return get_caching_config()
    
    @property
    def testing(self):
        return get_testing_config()
    
    @property
    def ml(self):
        return get_ml_config()
    
    @property
    def infrastructure(self):
        return get_infrastructure_config()
    
    @property
    def integration(self):
        return get_integration_config()

# Legacy instance for backward compatibility
config = LegacyConfigBridge()

__all__ = [
    'APIConfig', 'SecurityConfig', 'MonitoringConfig', 'CachingConfig',
    'TestingConfig', 'MLConfig', 'InfrastructureConfig', 'IntegrationConfig',
    'get_api_config', 'get_security_config', 'get_monitoring_config',
    'get_caching_config', 'get_testing_config', 'get_ml_config',
    'get_infrastructure_config', 'get_integration_config',
    'config'
]
'''
        return compatibility_code
    
    def generate_migration_report(self, result: MigrationResult) -> str:
        """Generate detailed migration report."""
        report = f"""
Configuration Migration Report
=============================
Generated: {result.files_scanned} files scanned

Summary:
- Config classes found: {result.config_classes_found}
- Migrations needed: {len(result.migrations_needed)}
- Backward compatible imports: {len(result.backward_compatible)}
- Warnings: {len(result.warnings)}

Migration Actions Needed:
"""
        
        for migration in result.migrations_needed:
            report += f"  • {migration}\n"
        
        if result.backward_compatible:
            report += "\nBackward Compatible Updates:\n"
            for update in result.backward_compatible:
                report += f"  • {update}\n"
        
        if result.warnings:
            report += "\nWarnings:\n"
            for warning in result.warnings:
                report += f"  ⚠ {warning}\n"
        
        report += f"""
Migration Strategy:
==================
1. Create backward compatibility layer in config/__init__.py
2. Update imports gradually: from config import APIConfig -> from config.unified_config import get_api_config
3. Replace config class instantiation with config manager calls
4. Test each module after migration
5. Remove compatibility layer once all modules migrated

Unified Config Benefits:
======================
✅ Single source of truth for all configurations
✅ Environment-based configuration profiles  
✅ Type-safe configuration access
✅ Built-in validation
✅ Hot reload capability
✅ Centralized configuration management
"""
        
        return report


def run_migration_analysis(root_path: Optional[Path] = None) -> MigrationResult:
    """Run configuration migration analysis."""
    tool = ConfigMigrationTool(root_path)
    return tool.analyze_codebase()


def create_backward_compatibility():
    """Create backward compatibility layer."""
    tool = ConfigMigrationTool()
    compatibility_code = tool.create_compatibility_layer()
    
    # Write to config/__init__.py
    config_init = Path("config") / "__init__.py"
    with open(config_init, 'w') as f:
        f.write(compatibility_code)
    
    logger.info("Created backward compatibility layer in config/__init__.py")


if __name__ == "__main__":
    # Run migration analysis
    result = run_migration_analysis()
    
    # Generate report
    tool = ConfigMigrationTool()
    report = tool.generate_migration_report(result)
    
    # Save report
    with open("config_migration_report.txt", "w") as f:
        f.write(report)
    
    print("Migration analysis complete. See config_migration_report.txt")