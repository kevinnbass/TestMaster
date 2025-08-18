"""
TestMaster Unified Entry Point

Main entry point for the TestMaster Unified Intelligence System.
Provides access to all TestMaster capabilities through a single interface.

Usage:
    python -m testmaster --help
    python -m testmaster orchestrate --target ./codebase
    python -m testmaster security-scan --target ./codebase
    python -m testmaster compliance --target ./codebase --standard OWASP_ASVS
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .orchestration import (
    UniversalTestOrchestrator,
    OrchestrationConfig,
    OrchestrationMode
)
from .security import (
    UniversalSecurityScanner,
    SecurityScanConfig,
    ComplianceFramework,
    ComplianceStandard
)
from .intelligence import (
    UniversalHierarchicalTestGenerator,
    HierarchicalPlanningConfig,
    MultiObjectiveOptimizer,
    LLMProviderManager
)
from .core import (
    UniversalLanguageDetector,
    UniversalAST
)


def create_orchestration_config(args) -> OrchestrationConfig:
    """Create orchestration configuration from arguments."""
    
    # Map string mode to enum
    mode_mapping = {
        'standard': OrchestrationMode.STANDARD,
        'intelligent': OrchestrationMode.INTELLIGENT,
        'security_focused': OrchestrationMode.SECURITY_FOCUSED,
        'compliance': OrchestrationMode.COMPLIANCE,
        'comprehensive': OrchestrationMode.COMPREHENSIVE,
        'rapid': OrchestrationMode.RAPID,
        'enterprise': OrchestrationMode.ENTERPRISE
    }
    
    mode = mode_mapping.get(args.mode, OrchestrationMode.COMPREHENSIVE)
    
    # Parse compliance standards
    compliance_standards = []
    if args.compliance_standards:
        for standard_name in args.compliance_standards:
            try:
                standard = ComplianceStandard(standard_name.lower())
                compliance_standards.append(standard)
            except ValueError:
                print(f"Warning: Unknown compliance standard: {standard_name}")
    
    config = OrchestrationConfig(
        mode=mode,
        target_directory=args.target,
        output_directory=args.output or f"{args.target}_testmaster_output",
        
        # Intelligence settings
        enable_hierarchical_planning=args.enable_intelligence,
        enable_optimization=args.enable_optimization,
        enable_llm_providers=args.enable_llm,
        
        # Security settings
        enable_security_scanning=args.enable_security,
        enable_compliance_checking=args.enable_compliance,
        enable_security_tests=args.enable_security_tests,
        target_compliance_standards=compliance_standards,
        
        # Framework settings
        auto_detect_frameworks=args.auto_detect_frameworks,
        target_frameworks=args.frameworks or [],
        
        # Output settings
        output_formats=args.output_formats or ["python", "universal"],
        include_documentation=args.include_docs,
        include_metrics=args.include_metrics,
        
        # Performance settings
        parallel_processing=args.parallel,
        max_workers=args.workers,
        timeout_seconds=args.timeout,
        
        # Quality settings
        min_test_quality_score=args.min_quality,
        min_coverage_target=args.min_coverage,
        enable_self_healing=args.enable_self_healing
    )
    
    return config


def cmd_orchestrate(args):
    """Run unified orchestration."""
    print("🎯 TestMaster Unified Orchestration")
    print("=" * 50)
    
    # Create configuration
    config = create_orchestration_config(args)
    
    # Initialize and run orchestrator
    orchestrator = UniversalTestOrchestrator(config)
    
    # Validate configuration
    is_valid, errors = orchestrator.validate_config(config)
    if not is_valid:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    # Run orchestration
    result = orchestrator.orchestrate(args.target)
    
    # Display results
    if result.success:
        print(f"\\n✅ Orchestration completed successfully!")
        print(f"   Duration: {result.metrics.total_duration:.2f}s")
        print(f"   Test suites: {len(result.test_suites)}")
        print(f"   Output files: {len(result.output_files)}")
        print(f"   Output directory: {config.output_directory}")
    else:
        print(f"\\n❌ Orchestration failed: {result.error_message}")
        return False
    
    return True


def cmd_security_scan(args):
    """Run security scanning."""
    print("🔒 TestMaster Security Scan")
    print("=" * 40)
    
    # Initialize scanner
    config = SecurityScanConfig()
    scanner = UniversalSecurityScanner(config)
    
    # Run scan
    result = scanner.scan_directory(args.target)
    
    # Display results
    print(f"\\n📊 Security Scan Results:")
    print(f"   Files scanned: {result.total_files_scanned}")
    print(f"   Vulnerabilities found: {len(result.findings)}")
    print(f"   Critical: {result.critical_count}")
    print(f"   High: {result.high_count}")
    print(f"   Medium: {result.medium_count}")
    print(f"   Low: {result.low_count}")
    print(f"   Risk score: {result.get_risk_score():.1f}/100")
    
    # Output detailed findings if requested
    if args.detailed and result.findings:
        print(f"\\n🔍 Detailed Findings:")
        for i, finding in enumerate(result.findings[:10], 1):  # Show first 10
            print(f"   {i}. {finding.title}")
            print(f"      File: {finding.file_path}:{finding.line_number}")
            print(f"      Severity: {finding.severity.value}")
            print(f"      Type: {finding.type.value}")
            if finding.recommendation:
                print(f"      Recommendation: {finding.recommendation}")
            print()
        
        if len(result.findings) > 10:
            print(f"   ... and {len(result.findings) - 10} more findings")
    
    return True


def cmd_compliance_check(args):
    """Run compliance checking."""
    print("📋 TestMaster Compliance Check")
    print("=" * 45)
    
    # Parse standard
    try:
        standard = ComplianceStandard(args.standard.lower())
    except ValueError:
        print(f"❌ Unknown compliance standard: {args.standard}")
        print(f"Available standards: {[s.value for s in ComplianceStandard]}")
        return False
    
    # Initialize framework
    compliance = ComplianceFramework()
    
    # Build AST
    print(f"Analyzing codebase...")
    universal_ast = UniversalAST.from_directory(args.target)
    
    # Run assessment
    report = compliance.assess_compliance(standard, universal_ast)
    
    # Display results
    print(f"\\n📊 Compliance Assessment Results:")
    print(f"   Standard: {standard.value.upper()}")
    print(f"   Overall score: {report.overall_score:.1%}")
    print(f"   Overall status: {report.overall_status.value}")
    print(f"   Total rules: {report.total_rules}")
    print(f"   Compliant: {report.compliant_rules}")
    print(f"   Non-compliant: {report.non_compliant_rules}")
    print(f"   Partially compliant: {report.partially_compliant_rules}")
    
    # Show remediation plan if available
    if args.detailed and report.remediation_plan:
        print(f"\\n💡 Remediation Plan:")
        for i, item in enumerate(report.remediation_plan[:5], 1):  # Show first 5
            print(f"   {i}. {item['title']}")
            print(f"      Priority: {item['priority']}")
            print(f"      Effort: {item['estimated_effort']}")
            print(f"      Deadline: {item['deadline']}")
            print()
    
    return True


def cmd_intelligence_test(args):
    """Run intelligence-enhanced test generation."""
    print("🧠 TestMaster Intelligence Test Generation")
    print("=" * 50)
    
    # Initialize HTP generator
    config = HierarchicalPlanningConfig(
        planning_depth=args.reasoning_depth,
        enable_optimization=args.enable_optimization,
        include_edge_cases=True
    )
    generator = UniversalHierarchicalTestGenerator(config)
    
    # Build AST
    print(f"Analyzing codebase...")
    universal_ast = UniversalAST.from_directory(args.target)
    
    # Generate tests for each module
    total_tests = 0
    for module in universal_ast.modules[:5]:  # Limit for demo
        print(f"\\nGenerating tests for: {module.name}")
        result = generator.generate_tests(module, config)
        
        if result.success:
            test_count = result.test_suite.count_tests() if result.test_suite else 0
            print(f"   ✓ Generated {test_count} tests")
            total_tests += test_count
        else:
            print(f"   ❌ Failed: {result.error_message}")
    
    print(f"\\n📊 Intelligence Test Generation Complete:")
    print(f"   Total tests generated: {total_tests}")
    
    return True


def cmd_analyze(args):
    """Analyze codebase."""
    print("📊 TestMaster Codebase Analysis")
    print("=" * 40)
    
    # Initialize detector
    detector = UniversalLanguageDetector()
    
    # Detect codebase
    profile = detector.detect_codebase(args.target)
    
    # Display results
    print(f"\\n📈 Analysis Results:")
    print(f"   Primary language: {profile.primary_language}")
    print(f"   Languages detected: {len(profile.languages)}")
    for lang, info in profile.languages.items():
        print(f"      {lang}: {info['file_count']} files, {info['line_count']} lines")
    
    print(f"   Frameworks detected: {len(profile.frameworks)}")
    for framework, info in profile.frameworks.items():
        print(f"      {framework}: confidence {info['confidence']:.1%}")
    
    print(f"   Total files: {len(profile.files)}")
    print(f"   File types: {len(profile.file_types)}")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TestMaster Unified Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Orchestrate command
    orchestrate_parser = subparsers.add_parser('orchestrate', help='Run unified orchestration')
    orchestrate_parser.add_argument('--target', required=True, help='Target directory')
    orchestrate_parser.add_argument('--output', help='Output directory')
    orchestrate_parser.add_argument('--mode', choices=['standard', 'intelligent', 'security_focused', 'compliance', 'comprehensive', 'rapid', 'enterprise'], default='comprehensive')
    orchestrate_parser.add_argument('--enable-intelligence', action='store_true', default=True)
    orchestrate_parser.add_argument('--enable-optimization', action='store_true', default=True)
    orchestrate_parser.add_argument('--enable-llm', action='store_true', default=True)
    orchestrate_parser.add_argument('--enable-security', action='store_true', default=True)
    orchestrate_parser.add_argument('--enable-compliance', action='store_true', default=True)
    orchestrate_parser.add_argument('--enable-security-tests', action='store_true', default=True)
    orchestrate_parser.add_argument('--compliance-standards', nargs='*', help='Compliance standards to check')
    orchestrate_parser.add_argument('--auto-detect-frameworks', action='store_true', default=True)
    orchestrate_parser.add_argument('--frameworks', nargs='*', help='Target frameworks')
    orchestrate_parser.add_argument('--output-formats', nargs='*', help='Output formats')
    orchestrate_parser.add_argument('--include-docs', action='store_true', default=True)
    orchestrate_parser.add_argument('--include-metrics', action='store_true', default=True)
    orchestrate_parser.add_argument('--parallel', action='store_true', default=True)
    orchestrate_parser.add_argument('--workers', type=int, default=4)
    orchestrate_parser.add_argument('--timeout', type=int, default=600)
    orchestrate_parser.add_argument('--min-quality', type=float, default=0.8)
    orchestrate_parser.add_argument('--min-coverage', type=float, default=0.85)
    orchestrate_parser.add_argument('--enable-self-healing', action='store_true', default=True)
    
    # Security scan command
    security_parser = subparsers.add_parser('security-scan', help='Run security scanning')
    security_parser.add_argument('--target', required=True, help='Target directory')
    security_parser.add_argument('--detailed', action='store_true', help='Show detailed findings')
    
    # Compliance check command
    compliance_parser = subparsers.add_parser('compliance', help='Run compliance checking')
    compliance_parser.add_argument('--target', required=True, help='Target directory')
    compliance_parser.add_argument('--standard', required=True, help='Compliance standard')
    compliance_parser.add_argument('--detailed', action='store_true', help='Show detailed results')
    
    # Intelligence test command
    intelligence_parser = subparsers.add_parser('intelligence-test', help='Run intelligence test generation')
    intelligence_parser.add_argument('--target', required=True, help='Target directory')
    intelligence_parser.add_argument('--reasoning-depth', type=int, default=3, help='Hierarchical planning depth')
    intelligence_parser.add_argument('--enable-optimization', action='store_true', default=True)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze codebase')
    analyze_parser.add_argument('--target', required=True, help='Target directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return False
    
    # Validate target directory
    if hasattr(args, 'target') and not Path(args.target).exists():
        print(f"❌ Target directory does not exist: {args.target}")
        return False
    
    # Dispatch to command handlers
    commands = {
        'orchestrate': cmd_orchestrate,
        'security-scan': cmd_security_scan,
        'compliance': cmd_compliance_check,
        'intelligence-test': cmd_intelligence_test,
        'analyze': cmd_analyze
    }
    
    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)