#!/usr/bin/env python3
"""
Intelligence Dependency Resolver - Agent A Hour 8
Comprehensive dependency resolution and installation system for intelligence modules

Resolves missing dependencies identified in Hour 7 validation and creates
intelligent installation strategies for optimal module activation.
"""

import logging
import subprocess
import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import json
import ast
import re

# Import validation system from Hour 7
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.core.intelligence.cc_1.module_validator import get_module_validator, ValidationStatus


class DependencyStatus(Enum):
    """Dependency resolution status"""
    AVAILABLE = "available"
    MISSING = "missing" 
    INSTALLED = "installed"
    FAILED = "failed"
    OPTIONAL = "optional"
    CONFLICTED = "conflicted"


class InstallationPriority(Enum):
    """Installation priority levels"""
    CRITICAL = "critical"      # Required for core functionality
    HIGH = "high"             # Needed for most modules
    MEDIUM = "medium"         # Useful for enhanced features
    LOW = "low"              # Optional enhancements
    DEVELOPMENT = "development"  # Dev/testing only


@dataclass
class DependencyInfo:
    """Information about a dependency"""
    name: str
    version_spec: Optional[str] = None
    import_name: Optional[str] = None  # Different from package name
    description: str = ""
    priority: InstallationPriority = InstallationPriority.MEDIUM
    alternatives: List[str] = None
    system_requirements: List[str] = None
    installation_notes: str = ""
    
    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []
        if self.system_requirements is None:
            self.system_requirements = []
        if self.import_name is None:
            self.import_name = self.name


@dataclass
class ResolutionResult:
    """Result of dependency resolution"""
    dependency_name: str
    status: DependencyStatus
    installed_version: Optional[str] = None
    installation_command: Optional[str] = None
    error_message: Optional[str] = None
    resolution_time: datetime = None
    
    def __post_init__(self):
        if self.resolution_time is None:
            self.resolution_time = datetime.now()


@dataclass
class InstallationPlan:
    """Comprehensive installation plan"""
    total_dependencies: int
    critical_dependencies: List[str]
    high_priority_dependencies: List[str]  
    medium_priority_dependencies: List[str]
    low_priority_dependencies: List[str]
    estimated_time_minutes: int
    installation_order: List[str]
    batch_commands: List[str]
    post_install_validation: List[str]
    rollback_plan: List[str]


class IntelligenceDependencyResolver:
    """
    Intelligence Dependency Resolver
    
    Comprehensive system for resolving, installing, and managing dependencies
    for the intelligence module platform based on validation results.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Dependency registry
        self.dependency_registry = self._build_dependency_registry()
        
        # Resolution tracking
        self.resolution_results: List[ResolutionResult] = []
        self.installed_packages: Set[str] = set()
        
        # Configuration
        self.requirements_file = Path("requirements.txt")
        self.cache_file = Path("data/dependency_cache.json")
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Module validator integration
        self.validator = get_module_validator()
        
        # Statistics
        self.stats = {
            'dependencies_analyzed': 0,
            'dependencies_resolved': 0,
            'dependencies_installed': 0,
            'dependencies_failed': 0,
            'modules_enabled': 0,
            'start_time': datetime.now()
        }
        
        self.logger.info("Intelligence Dependency Resolver initialized")
    
    def _build_dependency_registry(self) -> Dict[str, DependencyInfo]:
        """Build comprehensive dependency registry for intelligence modules"""
        registry = {}
        
        # Critical ML/AI Dependencies
        ml_deps = {
            'numpy': DependencyInfo(
                name='numpy',
                version_spec='>=1.21.0',
                description='Fundamental package for scientific computing',
                priority=InstallationPriority.CRITICAL,
                installation_notes='Core dependency for all numerical operations'
            ),
            'pandas': DependencyInfo(
                name='pandas',
                version_spec='>=1.3.0', 
                description='Data manipulation and analysis library',
                priority=InstallationPriority.CRITICAL,
                installation_notes='Essential for data processing modules'
            ),
            'scikit-learn': DependencyInfo(
                name='scikit-learn',
                version_spec='>=1.0.0',
                import_name='sklearn',
                description='Machine learning library',
                priority=InstallationPriority.HIGH,
                installation_notes='Required for predictive analytics and ML modules'
            ),
            'tensorflow': DependencyInfo(
                name='tensorflow',
                version_spec='>=2.8.0',
                description='Deep learning framework',
                priority=InstallationPriority.HIGH,
                alternatives=['torch'],
                installation_notes='For deep learning and neural network modules'
            ),
            'torch': DependencyInfo(
                name='torch',
                version_spec='>=1.10.0',
                description='PyTorch deep learning framework',
                priority=InstallationPriority.HIGH,
                alternatives=['tensorflow'],
                installation_notes='Alternative deep learning framework'
            )
        }
        
        # Natural Language Processing
        nlp_deps = {
            'nltk': DependencyInfo(
                name='nltk',
                version_spec='>=3.6.0',
                description='Natural language processing toolkit',
                priority=InstallationPriority.MEDIUM,
                installation_notes='For text analysis and semantic modules'
            ),
            'spacy': DependencyInfo(
                name='spacy',
                version_spec='>=3.4.0',
                description='Advanced NLP library',
                priority=InstallationPriority.MEDIUM,
                system_requirements=['en_core_web_sm model'],
                installation_notes='Requires language model download: python -m spacy download en_core_web_sm'
            ),
            'transformers': DependencyInfo(
                name='transformers',
                version_spec='>=4.15.0',
                description='State-of-the-art NLP transformers',
                priority=InstallationPriority.MEDIUM,
                installation_notes='For advanced language understanding modules'
            )
        }
        
        # Data Visualization
        viz_deps = {
            'matplotlib': DependencyInfo(
                name='matplotlib',
                version_spec='>=3.5.0',
                description='Plotting and visualization library',
                priority=InstallationPriority.MEDIUM,
                installation_notes='For chart generation and data visualization'
            ),
            'plotly': DependencyInfo(
                name='plotly',
                version_spec='>=5.5.0',
                description='Interactive plotting library',
                priority=InstallationPriority.LOW,
                installation_notes='For interactive dashboard visualizations'
            ),
            'seaborn': DependencyInfo(
                name='seaborn',
                version_spec='>=0.11.0',
                description='Statistical visualization library',
                priority=InstallationPriority.LOW,
                installation_notes='For statistical plots and analytics'
            )
        }
        
        # Network and Graph Analysis
        graph_deps = {
            'networkx': DependencyInfo(
                name='networkx',
                version_spec='>=2.6.0',
                description='Graph analysis library',
                priority=InstallationPriority.MEDIUM,
                installation_notes='For dependency graph analysis and network algorithms'
            ),
            'igraph': DependencyInfo(
                name='igraph',
                version_spec='>=0.9.0',
                import_name='igraph',
                description='Fast graph analysis library',
                priority=InstallationPriority.LOW,
                alternatives=['networkx'],
                installation_notes='High-performance alternative to NetworkX'
            )
        }
        
        # Web Framework Dependencies  
        web_deps = {
            'flask': DependencyInfo(
                name='flask',
                version_spec='>=2.0.0',
                description='Web framework',
                priority=InstallationPriority.HIGH,
                installation_notes='Required for API endpoints and dashboard'
            ),
            'websockets': DependencyInfo(
                name='websockets',
                version_spec='>=10.0',
                description='WebSocket implementation',
                priority=InstallationPriority.HIGH,
                installation_notes='Required for real-time streaming'
            ),
            'requests': DependencyInfo(
                name='requests',
                version_spec='>=2.25.0',
                description='HTTP library',
                priority=InstallationPriority.HIGH,
                installation_notes='For API integrations and external requests'
            )
        }
        
        # Testing and Development
        dev_deps = {
            'pytest': DependencyInfo(
                name='pytest',
                version_spec='>=6.2.0',
                description='Testing framework',
                priority=InstallationPriority.DEVELOPMENT,
                installation_notes='For module testing and validation'
            ),
            'black': DependencyInfo(
                name='black',
                version_spec='>=22.0.0',
                description='Code formatter',
                priority=InstallationPriority.DEVELOPMENT,
                installation_notes='For code style consistency'
            )
        }
        
        # Combine all registries
        registry.update(ml_deps)
        registry.update(nlp_deps)
        registry.update(viz_deps)
        registry.update(graph_deps)
        registry.update(web_deps)
        registry.update(dev_deps)
        
        return registry
    
    def analyze_module_dependencies(self, module_path: Path) -> List[str]:
        """Analyze module file to extract dependency requirements"""
        dependencies = []
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find imports
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dep_name = alias.name.split('.')[0]
                        if dep_name not in ['sys', 'os', 'pathlib', 'datetime', 'json']:  # Skip standard library
                            dependencies.append(dep_name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dep_name = node.module.split('.')[0]
                        if dep_name not in ['sys', 'os', 'pathlib', 'datetime', 'json']:
                            dependencies.append(dep_name)
            
            # Remove duplicates
            dependencies = list(set(dependencies))
            
        except Exception as e:
            self.logger.warning(f"Could not analyze dependencies for {module_path}: {e}")
        
        return dependencies
    
    def check_dependency_availability(self, dependency_name: str) -> ResolutionResult:
        """Check if a dependency is available and get its status"""
        import_name = dependency_name
        
        # Check if we have registry info
        if dependency_name in self.dependency_registry:
            dep_info = self.dependency_registry[dependency_name]
            import_name = dep_info.import_name
        
        try:
            # Try to import the package
            spec = importlib.util.find_spec(import_name)
            if spec is not None:
                # Package is available, try to get version
                try:
                    module = importlib.import_module(import_name)
                    version = getattr(module, '__version__', 'unknown')
                    
                    return ResolutionResult(
                        dependency_name=dependency_name,
                        status=DependencyStatus.AVAILABLE,
                        installed_version=version
                    )
                except Exception:
                    return ResolutionResult(
                        dependency_name=dependency_name,
                        status=DependencyStatus.AVAILABLE,
                        installed_version='unknown'
                    )
            else:
                return ResolutionResult(
                    dependency_name=dependency_name,
                    status=DependencyStatus.MISSING,
                    installation_command=self._get_install_command(dependency_name)
                )
        
        except Exception as e:
            return ResolutionResult(
                dependency_name=dependency_name,
                status=DependencyStatus.FAILED,
                error_message=str(e)
            )
    
    def resolve_all_dependencies(self) -> Dict[str, ResolutionResult]:
        """Resolve all dependencies for intelligence modules"""
        self.logger.info("Starting comprehensive dependency resolution...")
        
        all_results = {}
        
        # Get all intelligence modules
        intelligence_modules = self.validator.discover_intelligence_modules()
        
        # Collect all dependencies
        all_dependencies = set()
        
        for module_path in intelligence_modules:
            module_deps = self.analyze_module_dependencies(module_path)
            all_dependencies.update(module_deps)
        
        # Add registry dependencies
        all_dependencies.update(self.dependency_registry.keys())
        
        # Resolve each dependency
        for dep_name in all_dependencies:
            result = self.check_dependency_availability(dep_name)
            all_results[dep_name] = result
            self.resolution_results.append(result)
            
            self.stats['dependencies_analyzed'] += 1
            if result.status == DependencyStatus.AVAILABLE:
                self.stats['dependencies_resolved'] += 1
        
        self.logger.info(f"Dependency resolution complete: {len(all_results)} dependencies analyzed")
        return all_results
    
    def create_installation_plan(self, resolution_results: Dict[str, ResolutionResult]) -> InstallationPlan:
        """Create comprehensive installation plan for missing dependencies"""
        
        missing_deps = [
            name for name, result in resolution_results.items()
            if result.status == DependencyStatus.MISSING
        ]
        
        # Categorize by priority
        critical_deps = []
        high_priority_deps = []
        medium_priority_deps = []
        low_priority_deps = []
        
        for dep_name in missing_deps:
            if dep_name in self.dependency_registry:
                dep_info = self.dependency_registry[dep_name]
                priority = dep_info.priority
                
                if priority == InstallationPriority.CRITICAL:
                    critical_deps.append(dep_name)
                elif priority == InstallationPriority.HIGH:
                    high_priority_deps.append(dep_name)
                elif priority == InstallationPriority.MEDIUM:
                    medium_priority_deps.append(dep_name)
                else:
                    low_priority_deps.append(dep_name)
            else:
                medium_priority_deps.append(dep_name)  # Default to medium
        
        # Create installation order (critical first)
        installation_order = critical_deps + high_priority_deps + medium_priority_deps + low_priority_deps
        
        # Create batch commands
        batch_commands = []
        if installation_order:
            # Use requirements.txt if it exists
            if self.requirements_file.exists():
                batch_commands.append(f"{sys.executable} -m pip install -r {self.requirements_file}")
            else:
                # Create individual install commands
                batch_size = 5
                for i in range(0, len(installation_order), batch_size):
                    batch = installation_order[i:i + batch_size]
                    batch_commands.append(f"{sys.executable} -m pip install " + " ".join(batch))
        
        # Estimate time (rough estimate: 30s per dependency)
        estimated_time = len(missing_deps) * 0.5
        
        # Post-install validation
        validation_commands = [
            f"{sys.executable} -c \"import {dep}\"" for dep in installation_order[:10]  # Validate first 10
        ]
        
        # Rollback plan
        rollback_commands = [
            f"{sys.executable} -m pip uninstall -y " + " ".join(installation_order)
        ]
        
        plan = InstallationPlan(
            total_dependencies=len(missing_deps),
            critical_dependencies=critical_deps,
            high_priority_dependencies=high_priority_deps,
            medium_priority_dependencies=medium_priority_deps,
            low_priority_dependencies=low_priority_deps,
            estimated_time_minutes=int(estimated_time),
            installation_order=installation_order,
            batch_commands=batch_commands,
            post_install_validation=validation_commands,
            rollback_plan=rollback_commands
        )
        
        return plan
    
    def execute_installation_plan(self, plan: InstallationPlan, dry_run: bool = True) -> Dict[str, Any]:
        """Execute the installation plan"""
        results = {
            'executed_commands': [],
            'successful_installs': [],
            'failed_installs': [],
            'execution_time': 0,
            'dry_run': dry_run
        }
        
        if dry_run:
            self.logger.info("DRY RUN: Installation plan simulation")
            for command in plan.batch_commands:
                self.logger.info(f"Would execute: {command}")
                results['executed_commands'].append(command)
            return results
        
        start_time = datetime.now()
        self.logger.info("Executing installation plan...")
        
        for command in plan.batch_commands:
            try:
                self.logger.info(f"Executing: {command}")
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                
                if result.returncode == 0:
                    self.logger.info(f"Successfully executed: {command}")
                    results['successful_installs'].append(command)
                    results['executed_commands'].append(command)
                else:
                    self.logger.error(f"Failed to execute: {command}")
                    self.logger.error(f"Error: {result.stderr}")
                    results['failed_installs'].append({
                        'command': command,
                        'error': result.stderr
                    })
            
            except subprocess.TimeoutExpired:
                self.logger.error(f"Timeout executing: {command}")
                results['failed_installs'].append({
                    'command': command,
                    'error': 'Installation timeout'
                })
            
            except Exception as e:
                self.logger.error(f"Exception executing {command}: {e}")
                results['failed_installs'].append({
                    'command': command,
                    'error': str(e)
                })
        
        results['execution_time'] = (datetime.now() - start_time).total_seconds()
        self.stats['dependencies_installed'] = len(results['successful_installs'])
        self.stats['dependencies_failed'] = len(results['failed_installs'])
        
        return results
    
    def _get_install_command(self, dependency_name: str) -> str:
        """Get appropriate install command for dependency"""
        if dependency_name in self.dependency_registry:
            dep_info = self.dependency_registry[dependency_name]
            if dep_info.version_spec:
                return f"{sys.executable} -m pip install {dependency_name}{dep_info.version_spec}"
        
        return f"{sys.executable} -m pip install {dependency_name}"
    
    def validate_post_installation(self) -> Dict[str, Any]:
        """Validate dependencies after installation and re-run module validation"""
        self.logger.info("Running post-installation validation...")
        
        # Re-resolve dependencies
        resolution_results = self.resolve_all_dependencies()
        
        # Count improvements
        available_count = sum(1 for result in resolution_results.values() 
                            if result.status == DependencyStatus.AVAILABLE)
        missing_count = sum(1 for result in resolution_results.values()
                          if result.status == DependencyStatus.MISSING)
        
        # Re-run module validation
        validation_reports = self.validator.validate_all_modules()
        
        # Calculate module operational improvement
        operational_modules = sum(1 for report in validation_reports
                                if report.overall_status == ValidationStatus.PASSED)
        warning_modules = sum(1 for report in validation_reports
                            if report.overall_status == ValidationStatus.WARNING)
        
        self.stats['modules_enabled'] = operational_modules
        
        validation_summary = {
            'dependencies_available': available_count,
            'dependencies_missing': missing_count,
            'dependency_resolution_rate': available_count / max(len(resolution_results), 1) * 100,
            'modules_operational': operational_modules,
            'modules_warning': warning_modules,
            'modules_failed': len(validation_reports) - operational_modules - warning_modules,
            'module_operational_rate': operational_modules / max(len(validation_reports), 1) * 100,
            'overall_improvement': True if operational_modules > 0 else False
        }
        
        return validation_summary
    
    def get_dependency_report(self) -> Dict[str, Any]:
        """Generate comprehensive dependency resolution report"""
        resolution_results = self.resolve_all_dependencies()
        installation_plan = self.create_installation_plan(resolution_results)
        
        # Category breakdown
        status_breakdown = {}
        for status in DependencyStatus:
            status_breakdown[status.value] = sum(
                1 for result in resolution_results.values()
                if result.status == status
            )
        
        # Priority breakdown for missing dependencies
        priority_breakdown = {
            'critical': len(installation_plan.critical_dependencies),
            'high': len(installation_plan.high_priority_dependencies),
            'medium': len(installation_plan.medium_priority_dependencies),
            'low': len(installation_plan.low_priority_dependencies)
        }
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_dependencies': len(resolution_results),
                'available_dependencies': status_breakdown.get('available', 0),
                'missing_dependencies': status_breakdown.get('missing', 0),
                'failed_dependencies': status_breakdown.get('failed', 0),
                'resolution_rate': status_breakdown.get('available', 0) / max(len(resolution_results), 1) * 100
            },
            'status_breakdown': status_breakdown,
            'priority_breakdown': priority_breakdown,
            'installation_plan': asdict(installation_plan),
            'statistics': self.stats,
            'recommendations': self._generate_recommendations(resolution_results, installation_plan)
        }
        
        return report
    
    def _generate_recommendations(self, resolution_results: Dict[str, ResolutionResult], 
                                plan: InstallationPlan) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        missing_count = sum(1 for result in resolution_results.values()
                          if result.status == DependencyStatus.MISSING)
        
        if missing_count > 20:
            recommendations.append(
                "High number of missing dependencies detected. Consider using virtual environment."
            )
        
        if len(plan.critical_dependencies) > 0:
            recommendations.append(
                f"Install {len(plan.critical_dependencies)} critical dependencies first for core functionality."
            )
        
        if plan.estimated_time_minutes > 10:
            recommendations.append(
                f"Installation will take approximately {plan.estimated_time_minutes} minutes. Run during maintenance window."
            )
        
        recommendations.append("Run dry-run first to preview installation commands.")
        recommendations.append("Consider creating system backup before major dependency installation.")
        
        return recommendations


# Global dependency resolver instance
_dependency_resolver: Optional[IntelligenceDependencyResolver] = None


def get_dependency_resolver() -> IntelligenceDependencyResolver:
    """Get global dependency resolver instance"""
    global _dependency_resolver
    if _dependency_resolver is None:
        _dependency_resolver = IntelligenceDependencyResolver()
    return _dependency_resolver


def resolve_intelligence_dependencies() -> Dict[str, Any]:
    """Resolve all intelligence dependencies and create installation plan"""
    resolver = get_dependency_resolver()
    return resolver.get_dependency_report()


if __name__ == "__main__":
    # Run dependency resolution if called directly
    logging.basicConfig(level=logging.INFO)
    
    resolver = get_dependency_resolver()
    report = resolver.get_dependency_report()
    
    print(f"\nDependency Resolution Report:")
    print(f"Total dependencies: {report['summary']['total_dependencies']}")
    print(f"Available: {report['summary']['available_dependencies']}")
    print(f"Missing: {report['summary']['missing_dependencies']}")
    print(f"Resolution rate: {report['summary']['resolution_rate']:.1f}%")
    
    if report['summary']['missing_dependencies'] > 0:
        plan = report['installation_plan']
        print(f"\nInstallation Plan:")
        print(f"Critical: {len(plan['critical_dependencies'])}")
        print(f"High Priority: {len(plan['high_priority_dependencies'])}")
        print(f"Medium Priority: {len(plan['medium_priority_dependencies'])}")
        print(f"Estimated time: {plan['estimated_time_minutes']} minutes")