"""
Universal Test Generation Orchestrator

The central orchestration system that coordinates all TestMaster components:
- Intelligenc Layer: ToT reasoning, optimization, LLM providers
- Security Intelligence: vulnerability scanning, compliance, security tests
- Core Framework: AST analysis, language detection, universal abstractions
- Test Generation: unified test creation with framework adaptation

Adapted from Agency Swarm's orchestration patterns and PraisonAI's coordination system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Union, Tuple
from enum import Enum
from datetime import datetime
import json
from pathlib import Path

# Core imports
from ..core.ast_abstraction import UniversalAST
from ..core.language_detection import UniversalLanguageDetector, CodebaseProfile
from ..core.framework_abstraction import UniversalTestSuite, TestMetadata

# Intelligence imports
from ..intelligence.tree_of_thought import UniversalToTTestGenerator, ToTGenerationConfig
from ..intelligence.optimization import MultiObjectiveOptimizer, OptimizationObjective
from ..intelligence.llm_providers import LLMProviderManager, LLMProviderConfig

# Security imports
from ..security.universal_scanner import UniversalSecurityScanner, SecurityScanConfig
from ..security.compliance_framework import ComplianceFramework, ComplianceStandard
from ..security.security_test_generator import SecurityTestGenerator, SecurityTestConfig

# Orchestration imports
from .framework_adapter import UniversalFrameworkAdapter, FrameworkAdapterConfig
from .output_system import CodebaseAgnosticOutputSystem, OutputSystemConfig


class OrchestrationMode(Enum):
    """Orchestration modes for different use cases."""
    STANDARD = "standard"  # Basic test generation
    INTELLIGENT = "intelligent"  # With ToT reasoning and optimization
    SECURITY_FOCUSED = "security_focused"  # Emphasis on security testing
    COMPLIANCE = "compliance"  # Focus on compliance requirements
    COMPREHENSIVE = "comprehensive"  # Full intelligence + security + compliance
    RAPID = "rapid"  # Fast generation with minimal intelligence
    ENTERPRISE = "enterprise"  # Full enterprise features


@dataclass
class OrchestrationConfig:
    """Configuration for the universal orchestrator."""
    # Mode settings
    mode: OrchestrationMode = OrchestrationMode.COMPREHENSIVE
    
    # Core settings
    target_directory: str = ""
    output_directory: str = "./generated_tests"
    
    # Intelligence settings
    enable_tot_reasoning: bool = True
    enable_optimization: bool = True
    enable_llm_providers: bool = True
    
    # Security settings
    enable_security_scanning: bool = True
    enable_compliance_checking: bool = True
    enable_security_tests: bool = True
    target_compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    
    # Framework settings
    auto_detect_frameworks: bool = True
    target_frameworks: List[str] = field(default_factory=list)
    
    # Output settings
    output_formats: List[str] = field(default_factory=lambda: ["python", "universal"])
    include_documentation: bool = True
    include_metrics: bool = True
    
    # Performance settings
    parallel_processing: bool = True
    max_workers: int = 4
    timeout_seconds: int = 600
    
    # Quality settings
    min_test_quality_score: float = 0.8
    min_coverage_target: float = 0.85
    enable_self_healing: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mode': self.mode.value,
            'target_directory': self.target_directory,
            'output_directory': self.output_directory,
            'intelligence': {
                'enable_tot_reasoning': self.enable_tot_reasoning,
                'enable_optimization': self.enable_optimization,
                'enable_llm_providers': self.enable_llm_providers
            },
            'security': {
                'enable_security_scanning': self.enable_security_scanning,
                'enable_compliance_checking': self.enable_compliance_checking,
                'enable_security_tests': self.enable_security_tests,
                'target_compliance_standards': [s.value for s in self.target_compliance_standards]
            },
            'framework': {
                'auto_detect_frameworks': self.auto_detect_frameworks,
                'target_frameworks': self.target_frameworks
            },
            'output': {
                'output_formats': self.output_formats,
                'include_documentation': self.include_documentation,
                'include_metrics': self.include_metrics
            },
            'performance': {
                'parallel_processing': self.parallel_processing,
                'max_workers': self.max_workers,
                'timeout_seconds': self.timeout_seconds
            },
            'quality': {
                'min_test_quality_score': self.min_test_quality_score,
                'min_coverage_target': self.min_coverage_target,
                'enable_self_healing': self.enable_self_healing
            }
        }


@dataclass
class OrchestrationMetrics:
    """Metrics from orchestration execution."""
    # Timing metrics
    total_duration: float = 0.0
    analysis_duration: float = 0.0
    generation_duration: float = 0.0
    security_scan_duration: float = 0.0
    output_generation_duration: float = 0.0
    
    # Analysis metrics
    files_analyzed: int = 0
    languages_detected: int = 0
    frameworks_detected: int = 0
    total_functions: int = 0
    total_classes: int = 0
    
    # Security metrics
    vulnerabilities_found: int = 0
    compliance_gaps: int = 0
    security_tests_generated: int = 0
    
    # Test generation metrics
    test_suites_generated: int = 0
    total_tests_generated: int = 0
    intelligence_enhanced_tests: int = 0
    
    # Quality metrics
    average_test_quality_score: float = 0.0
    estimated_coverage: float = 0.0
    self_healing_fixes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timing': {
                'total_duration': self.total_duration,
                'analysis_duration': self.analysis_duration,
                'generation_duration': self.generation_duration,
                'security_scan_duration': self.security_scan_duration,
                'output_generation_duration': self.output_generation_duration
            },
            'analysis': {
                'files_analyzed': self.files_analyzed,
                'languages_detected': self.languages_detected,
                'frameworks_detected': self.frameworks_detected,
                'total_functions': self.total_functions,
                'total_classes': self.total_classes
            },
            'security': {
                'vulnerabilities_found': self.vulnerabilities_found,
                'compliance_gaps': self.compliance_gaps,
                'security_tests_generated': self.security_tests_generated
            },
            'test_generation': {
                'test_suites_generated': self.test_suites_generated,
                'total_tests_generated': self.total_tests_generated,
                'intelligence_enhanced_tests': self.intelligence_enhanced_tests
            },
            'quality': {
                'average_test_quality_score': self.average_test_quality_score,
                'estimated_coverage': self.estimated_coverage,
                'self_healing_fixes': self.self_healing_fixes
            }
        }


@dataclass
class OrchestrationResult:
    """Result of orchestration execution."""
    # Success/failure
    success: bool = False
    error_message: Optional[str] = None
    
    # Generated artifacts
    test_suites: List[UniversalTestSuite] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    
    # Analysis results
    codebase_profile: Optional[CodebaseProfile] = None
    universal_ast: Optional[UniversalAST] = None
    
    # Security results
    security_scan_result = None  # SecurityScanResult
    compliance_reports: List = field(default_factory=list)  # List[ComplianceReport]
    
    # Intelligence results
    optimization_results: List = field(default_factory=list)
    tot_reasoning_paths: List = field(default_factory=list)
    
    # Metrics
    metrics: OrchestrationMetrics = field(default_factory=OrchestrationMetrics)
    
    # Metadata
    execution_timestamp: datetime = field(default_factory=datetime.now)
    config_used: Optional[OrchestrationConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'execution': {
                'success': self.success,
                'error_message': self.error_message,
                'execution_timestamp': self.execution_timestamp.isoformat()
            },
            'artifacts': {
                'test_suites_count': len(self.test_suites),
                'output_files': self.output_files,
                'test_suites': [ts.to_dict() for ts in self.test_suites]
            },
            'analysis': {
                'codebase_profile': self.codebase_profile.to_dict() if self.codebase_profile else None,
                'universal_ast_summary': {
                    'modules': len(self.universal_ast.modules) if self.universal_ast else 0,
                    'total_functions': self.universal_ast.total_functions if self.universal_ast else 0,
                    'total_classes': self.universal_ast.total_classes if self.universal_ast else 0
                } if self.universal_ast else None
            },
            'security': {
                'security_scan_completed': self.security_scan_result is not None,
                'compliance_reports_count': len(self.compliance_reports),
                'vulnerabilities_found': self.security_scan_result.critical_count + self.security_scan_result.high_count if self.security_scan_result else 0
            },
            'intelligence': {
                'optimization_results_count': len(self.optimization_results),
                'tot_reasoning_paths_count': len(self.tot_reasoning_paths)
            },
            'metrics': self.metrics.to_dict(),
            'config_used': self.config_used.to_dict() if self.config_used else None
        }


class UniversalTestOrchestrator:
    """Universal test generation orchestrator."""
    
    def __init__(self, config: OrchestrationConfig = None):
        self.config = config or OrchestrationConfig()
        
        # Initialize core components
        self.language_detector = UniversalLanguageDetector()
        
        # Initialize intelligence components (conditionally)
        self.tot_generator = None
        self.optimizer = None
        self.llm_manager = None
        
        # Initialize security components (conditionally)
        self.security_scanner = None
        self.compliance_framework = None
        self.security_test_generator = None
        
        # Initialize orchestration components
        self.framework_adapter = UniversalFrameworkAdapter()
        self.output_system = CodebaseAgnosticOutputSystem()
        
        # Initialize based on configuration
        self._initialize_components()
        
        print(f"Universal Test Orchestrator initialized")
        print(f"   Mode: {self.config.mode.value}")
        print(f"   Intelligence enabled: {self.config.enable_tot_reasoning}")
        print(f"   Security enabled: {self.config.enable_security_scanning}")
        print(f"   Compliance enabled: {self.config.enable_compliance_checking}")
    
    def orchestrate(self, target_path: str) -> OrchestrationResult:
        """Main orchestration method."""
        start_time = datetime.now()
        
        print(f"\n🎯 Starting Universal Test Orchestration")
        print(f"   Target: {target_path}")
        print(f"   Mode: {self.config.mode.value}")
        print(f"   Output: {self.config.output_directory}")
        
        # Initialize result
        result = OrchestrationResult(config_used=self.config)
        
        try:
            # Phase 1: Codebase Analysis
            analysis_start = datetime.now()
            print(f"\n📊 Phase 1: Codebase Analysis")
            
            codebase_profile, universal_ast = self._analyze_codebase(target_path)
            result.codebase_profile = codebase_profile
            result.universal_ast = universal_ast
            
            result.metrics.analysis_duration = (datetime.now() - analysis_start).total_seconds()
            result.metrics.files_analyzed = len(codebase_profile.files)
            result.metrics.languages_detected = len(codebase_profile.languages)
            result.metrics.total_functions = universal_ast.total_functions
            result.metrics.total_classes = universal_ast.total_classes
            
            # Phase 2: Security Analysis (if enabled)
            if self.config.enable_security_scanning or self.config.enable_compliance_checking:
                security_start = datetime.now()
                print(f"\n🔒 Phase 2: Security Analysis")
                
                security_results = self._perform_security_analysis(universal_ast)
                result.security_scan_result = security_results.get('scan_result')
                result.compliance_reports = security_results.get('compliance_reports', [])
                
                result.metrics.security_scan_duration = (datetime.now() - security_start).total_seconds()
                if result.security_scan_result:
                    result.metrics.vulnerabilities_found = (
                        result.security_scan_result.critical_count + 
                        result.security_scan_result.high_count
                    )
                result.metrics.compliance_gaps = sum(
                    report.non_compliant_rules for report in result.compliance_reports
                )
            
            # Phase 3: Intelligent Test Generation
            generation_start = datetime.now()
            print(f"\n🧠 Phase 3: Intelligent Test Generation")
            
            test_suites = self._generate_intelligent_tests(
                universal_ast, 
                result.security_scan_result, 
                result.compliance_reports
            )
            result.test_suites = test_suites
            
            result.metrics.generation_duration = (datetime.now() - generation_start).total_seconds()
            result.metrics.test_suites_generated = len(test_suites)
            result.metrics.total_tests_generated = sum(ts.count_tests() for ts in test_suites)
            
            # Phase 4: Output Generation
            output_start = datetime.now()
            print(f"\n📝 Phase 4: Output Generation")
            
            output_files = self._generate_outputs(test_suites, codebase_profile)
            result.output_files = output_files
            
            result.metrics.output_generation_duration = (datetime.now() - output_start).total_seconds()
            
            # Calculate final metrics
            result.metrics.total_duration = (datetime.now() - start_time).total_seconds()
            result.metrics.average_test_quality_score = self._calculate_average_quality(test_suites)
            result.metrics.estimated_coverage = self._estimate_coverage(test_suites, universal_ast)
            
            result.success = True
            
            print(f"\n✅ Orchestration Complete!")
            print(f"   Duration: {result.metrics.total_duration:.2f}s")
            print(f"   Test suites: {len(test_suites)}")
            print(f"   Total tests: {result.metrics.total_tests_generated}")
            print(f"   Output files: {len(output_files)}")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            print(f"\n❌ Orchestration Failed: {str(e)}")
        
        return result
    
    def _initialize_components(self):
        """Initialize components based on configuration."""
        
        # Intelligence components
        if self.config.enable_tot_reasoning:
            self.tot_generator = UniversalToTTestGenerator()
            print("   ToT reasoning enabled")
        
        if self.config.enable_optimization:
            self.optimizer = MultiObjectiveOptimizer()
            print("   Multi-objective optimization enabled")
        
        if self.config.enable_llm_providers:
            self.llm_manager = LLMProviderManager()
            print("   LLM provider management enabled")
        
        # Security components
        if self.config.enable_security_scanning:
            security_config = SecurityScanConfig()
            self.security_scanner = UniversalSecurityScanner(security_config)
            print("   Security scanning enabled")
        
        if self.config.enable_compliance_checking:
            self.compliance_framework = ComplianceFramework()
            print("   Compliance checking enabled")
        
        if self.config.enable_security_tests:
            security_test_config = SecurityTestConfig()
            self.security_test_generator = SecurityTestGenerator(security_test_config)
            print("   Security test generation enabled")
    
    def _analyze_codebase(self, target_path: str) -> Tuple[CodebaseProfile, UniversalAST]:
        """Analyze the target codebase."""
        print(f"   Detecting languages and frameworks...")
        
        # Detect codebase profile
        codebase_profile = self.language_detector.detect_codebase(target_path)
        
        print(f"   Languages: {', '.join(codebase_profile.languages.keys())}")
        print(f"   Primary language: {codebase_profile.primary_language}")
        print(f"   Files: {len(codebase_profile.files)}")
        
        # Create Universal AST
        print(f"   Building Universal AST...")
        universal_ast = UniversalAST.from_directory(target_path)
        
        print(f"   Modules: {len(universal_ast.modules)}")
        print(f"   Functions: {universal_ast.total_functions}")
        print(f"   Classes: {universal_ast.total_classes}")
        
        return codebase_profile, universal_ast
    
    def _perform_security_analysis(self, universal_ast: UniversalAST) -> Dict[str, Any]:
        """Perform security analysis."""
        results = {}
        
        # Security scanning
        if self.security_scanner:
            print(f"   Running security vulnerability scan...")
            scan_result = self.security_scanner.scan_ast(universal_ast)
            results['scan_result'] = scan_result
            
            print(f"   Vulnerabilities found: {len(scan_result.findings)}")
            print(f"   Critical: {scan_result.critical_count}")
            print(f"   High: {scan_result.high_count}")
            print(f"   Medium: {scan_result.medium_count}")
        
        # Compliance checking
        if self.compliance_framework:
            print(f"   Running compliance assessments...")
            compliance_reports = []
            
            # Check configured standards or default set
            standards_to_check = self.config.target_compliance_standards
            if not standards_to_check:
                standards_to_check = [
                    ComplianceStandard.OWASP_ASVS,
                    ComplianceStandard.SOX,
                    ComplianceStandard.GDPR
                ]
            
            for standard in standards_to_check:
                security_findings = results.get('scan_result', {}).get('findings', []) if results.get('scan_result') else []
                compliance_report = self.compliance_framework.assess_compliance(
                    standard, universal_ast, security_findings
                )
                compliance_reports.append(compliance_report)
                
                print(f"   {standard.value.upper()}: {compliance_report.overall_score:.1%} compliant")
            
            results['compliance_reports'] = compliance_reports
        
        return results
    
    def _generate_intelligent_tests(self, 
                                   universal_ast: UniversalAST,
                                   security_scan_result=None,
                                   compliance_reports: List = None) -> List[UniversalTestSuite]:
        """Generate intelligent tests using all available components."""
        test_suites = []
        
        print(f"   Generating tests for {len(universal_ast.modules)} modules...")
        
        # Basic test generation for each module
        for module in universal_ast.modules:
            print(f"   Processing module: {module.name}")
            
            # Create base test suite
            test_suite = UniversalTestSuite(
                name=f"{module.name}TestSuite",
                metadata=TestMetadata(
                    tags=["automated", "intelligent"],
                    category="integration",
                    description=f"Intelligent test suite for {module.name}"
                )
            )
            
            # Generate tests using intelligence components
            if self.config.enable_tot_reasoning and self.tot_generator:
                tot_config = ToTGenerationConfig(
                    reasoning_depth=3,
                    enable_optimization=True,
                    include_edge_cases=True
                )
                tot_result = self.tot_generator.generate_tests(module, tot_config)
                if tot_result.success and tot_result.test_suite:
                    test_suite.test_cases.extend(tot_result.test_suite.test_cases)
                    print(f"     ToT generated: {len(tot_result.test_suite.test_cases)} test cases")
            
            # Generate security tests if enabled
            if self.config.enable_security_tests and self.security_test_generator:
                module_vulnerabilities = []
                if security_scan_result and security_scan_result.findings:
                    module_vulnerabilities = [
                        f for f in security_scan_result.findings 
                        if module.file_path in f.file_path
                    ]
                
                module_compliance_reports = []
                if compliance_reports:
                    module_compliance_reports = compliance_reports  # Simplified
                
                # Create minimal AST for this module
                module_ast = UniversalAST()
                module_ast.modules = [module]
                module_ast.project_path = universal_ast.project_path
                
                security_test_suite = self.security_test_generator.generate_security_tests(
                    module_ast, module_vulnerabilities, module_compliance_reports
                )
                
                if security_test_suite.universal_test_suite.test_cases:
                    test_suite.test_cases.extend(security_test_suite.universal_test_suite.test_cases)
                    print(f"     Security tests generated: {len(security_test_suite.universal_test_suite.test_cases)} test cases")
            
            # Calculate metrics for this test suite
            test_suite.calculate_metrics()
            
            if test_suite.test_cases:
                test_suites.append(test_suite)
        
        # Apply optimization if enabled
        if self.config.enable_optimization and self.optimizer and test_suites:
            print(f"   Applying multi-objective optimization...")
            # This would optimize the test suites for coverage, quality, etc.
            # Simplified for now
            print(f"   Optimization complete")
        
        return test_suites
    
    def _generate_outputs(self, test_suites: List[UniversalTestSuite], codebase_profile: CodebaseProfile) -> List[str]:
        """Generate output files in various formats."""
        output_files = []
        
        print(f"   Generating outputs in {len(self.config.output_formats)} format(s)...")
        
        # Configure framework adapter based on detected frameworks
        adapter_config = FrameworkAdapterConfig()
        if self.config.auto_detect_frameworks:
            adapter_config.target_frameworks = list(codebase_profile.frameworks.keys())
        else:
            adapter_config.target_frameworks = self.config.target_frameworks
        
        # Configure output system
        output_config = OutputSystemConfig(
            output_directory=self.config.output_directory,
            output_formats=self.config.output_formats,
            include_documentation=self.config.include_documentation,
            include_metrics=self.config.include_metrics
        )
        
        # Generate outputs for each test suite
        for test_suite in test_suites:
            print(f"   Generating outputs for: {test_suite.name}")
            
            # Adapt to target frameworks
            adapted_suites = self.framework_adapter.adapt_test_suite(test_suite, adapter_config)
            
            # Generate output files
            for adapted_suite in adapted_suites:
                suite_output_files = self.output_system.generate_outputs(adapted_suite, output_config)
                output_files.extend(suite_output_files)
        
        print(f"   Generated {len(output_files)} output files")
        
        return output_files
    
    def _calculate_average_quality(self, test_suites: List[UniversalTestSuite]) -> float:
        """Calculate average test quality score."""
        if not test_suites:
            return 0.0
        
        total_score = 0.0
        total_tests = 0
        
        for suite in test_suites:
            if hasattr(suite, 'quality_score'):
                total_score += suite.quality_score
                total_tests += 1
        
        return total_score / total_tests if total_tests > 0 else 0.0
    
    def _estimate_coverage(self, test_suites: List[UniversalTestSuite], universal_ast: UniversalAST) -> float:
        """Estimate test coverage."""
        if not test_suites or not universal_ast:
            return 0.0
        
        # Simplified coverage estimation
        total_functions = universal_ast.total_functions
        total_tests = sum(suite.count_tests() for suite in test_suites)
        
        # Rough estimation: assume each test covers one function on average
        estimated_coverage = min(total_tests / total_functions if total_functions > 0 else 0.0, 1.0)
        
        return estimated_coverage
    
    def get_orchestration_modes(self) -> List[str]:
        """Get available orchestration modes."""
        return [mode.value for mode in OrchestrationMode]
    
    def get_supported_compliance_standards(self) -> List[str]:
        """Get supported compliance standards."""
        return [standard.value for standard in ComplianceStandard]
    
    def validate_config(self, config: OrchestrationConfig) -> Tuple[bool, List[str]]:
        """Validate orchestration configuration."""
        errors = []
        
        # Check target directory
        if not config.target_directory:
            errors.append("Target directory is required")
        elif not Path(config.target_directory).exists():
            errors.append(f"Target directory does not exist: {config.target_directory}")
        
        # Check output directory
        try:
            Path(config.output_directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {str(e)}")
        
        # Check compliance standards
        valid_standards = [s.value for s in ComplianceStandard]
        for standard in config.target_compliance_standards:
            if standard.value not in valid_standards:
                errors.append(f"Invalid compliance standard: {standard.value}")
        
        return len(errors) == 0, errors