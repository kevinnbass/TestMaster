#!/usr/bin/env python3
"""
TestMaster Unified Orchestration Example

Demonstrates the complete hybrid intelligence integration with unified orchestration.
This example shows how all components work together:
- Intelligence Layer (ToT, Optimization, LLM Providers)
- Security Intelligence (Scanning, Compliance, Security Tests)
- Universal Orchestration (Framework Adaptation, Output System)

Usage:
    python unified_orchestration_example.py --target ./path/to/codebase
    python unified_orchestration_example.py --demo
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add testmaster to path
sys.path.insert(0, str(Path(__file__).parent))

from testmaster.orchestration import (
    UniversalTestOrchestrator,
    OrchestrationConfig,
    OrchestrationMode
)
from testmaster.security import ComplianceStandard


def create_comprehensive_config(target_directory: str, output_directory: str = None) -> OrchestrationConfig:
    """Create a comprehensive orchestration configuration."""
    
    if output_directory is None:
        output_directory = f"{target_directory}_testmaster_output"
    
    config = OrchestrationConfig(
        # Mode - Use comprehensive for full functionality
        mode=OrchestrationMode.COMPREHENSIVE,
        
        # Core settings
        target_directory=target_directory,
        output_directory=output_directory,
        
        # Intelligence settings - Enable all advanced features
        enable_tot_reasoning=True,
        enable_optimization=True,
        enable_llm_providers=True,
        
        # Security settings - Full security analysis
        enable_security_scanning=True,
        enable_compliance_checking=True,
        enable_security_tests=True,
        target_compliance_standards=[
            ComplianceStandard.OWASP_ASVS,
            ComplianceStandard.SOX,
            ComplianceStandard.GDPR,
            ComplianceStandard.PCI_DSS
        ],
        
        # Framework settings - Auto-detect and adapt
        auto_detect_frameworks=True,
        target_frameworks=["pytest", "unittest", "jest", "junit"],
        
        # Output settings - Multiple formats
        output_formats=["python", "javascript", "universal", "markdown"],
        include_documentation=True,
        include_metrics=True,
        
        # Performance settings
        parallel_processing=True,
        max_workers=4,
        timeout_seconds=600,
        
        # Quality settings
        min_test_quality_score=0.8,
        min_coverage_target=0.85,
        enable_self_healing=True
    )
    
    return config


def run_enterprise_orchestration(target_directory: str, output_directory: str = None) -> bool:
    """Run enterprise-grade orchestration with full features."""
    
    print("🚀 TestMaster Unified Orchestration - Enterprise Mode")
    print("=" * 60)
    
    # Create comprehensive configuration
    config = create_comprehensive_config(target_directory, output_directory)
    
    print(f"📋 Configuration:")
    print(f"   Mode: {config.mode.value}")
    print(f"   Target: {config.target_directory}")
    print(f"   Output: {config.output_directory}")
    print(f"   Intelligence: ToT={config.enable_tot_reasoning}, Opt={config.enable_optimization}, LLM={config.enable_llm_providers}")
    print(f"   Security: Scan={config.enable_security_scanning}, Compliance={config.enable_compliance_checking}, Tests={config.enable_security_tests}")
    print(f"   Frameworks: {config.target_frameworks}")
    print(f"   Formats: {config.output_formats}")
    print()
    
    # Initialize orchestrator
    orchestrator = UniversalTestOrchestrator(config)
    
    # Validate configuration
    print("🔍 Validating configuration...")
    is_valid, errors = orchestrator.validate_config(config)
    
    if not is_valid:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    print("✅ Configuration validated successfully")
    print()
    
    # Run orchestration
    print("🎯 Starting unified orchestration...")
    result = orchestrator.orchestrate(target_directory)
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 ORCHESTRATION RESULTS")
    print("=" * 60)
    
    if result.success:
        print("✅ Orchestration completed successfully!")
        print()
        
        # Analysis results
        print("📈 Analysis Results:")
        print(f"   Files analyzed: {result.metrics.files_analyzed}")
        print(f"   Languages detected: {result.metrics.languages_detected}")
        print(f"   Frameworks detected: {result.metrics.frameworks_detected}")
        print(f"   Functions found: {result.metrics.total_functions}")
        print(f"   Classes found: {result.metrics.total_classes}")
        print()
        
        # Security results
        if result.security_scan_result:
            print("🔒 Security Analysis:")
            print(f"   Vulnerabilities found: {result.metrics.vulnerabilities_found}")
            print(f"   Critical: {result.security_scan_result.critical_count}")
            print(f"   High: {result.security_scan_result.high_count}")
            print(f"   Medium: {result.security_scan_result.medium_count}")
            print(f"   Risk score: {result.security_scan_result.get_risk_score():.1f}/100")
        
        if result.compliance_reports:
            print(f"   Compliance reports: {len(result.compliance_reports)}")
            for report in result.compliance_reports:
                print(f"   {report.standard.value.upper()}: {report.overall_score:.1%} compliant")
        print()
        
        # Test generation results
        print("🧠 Test Generation:")
        print(f"   Test suites generated: {result.metrics.test_suites_generated}")
        print(f"   Total tests generated: {result.metrics.total_tests_generated}")
        print(f"   Intelligence-enhanced tests: {result.metrics.intelligence_enhanced_tests}")
        print(f"   Security tests generated: {result.metrics.security_tests_generated}")
        print(f"   Average quality score: {result.metrics.average_test_quality_score:.1%}")
        print(f"   Estimated coverage: {result.metrics.estimated_coverage:.1%}")
        print()
        
        # Output results
        print("📝 Output Generation:")
        print(f"   Output files generated: {len(result.output_files)}")
        for output_file in result.output_files[:10]:  # Show first 10 files
            print(f"   - {output_file}")
        if len(result.output_files) > 10:
            print(f"   ... and {len(result.output_files) - 10} more files")
        print()
        
        # Performance metrics
        print("⚡ Performance Metrics:")
        print(f"   Total duration: {result.metrics.total_duration:.2f}s")
        print(f"   Analysis time: {result.metrics.analysis_duration:.2f}s")
        print(f"   Security scan time: {result.metrics.security_scan_duration:.2f}s")
        print(f"   Generation time: {result.metrics.generation_duration:.2f}s")
        print(f"   Output time: {result.metrics.output_generation_duration:.2f}s")
        print()
        
        # Recommendations
        print("💡 Recommendations:")
        
        if result.metrics.estimated_coverage < 0.8:
            print("   - Consider generating additional tests to improve coverage")
        
        if result.metrics.vulnerabilities_found > 0:
            print("   - Review and address identified security vulnerabilities")
        
        if result.compliance_reports:
            non_compliant = [r for r in result.compliance_reports if r.overall_score < 0.8]
            if non_compliant:
                print("   - Address compliance gaps in the following standards:")
                for report in non_compliant:
                    print(f"     * {report.standard.value.upper()}: {report.overall_score:.1%}")
        
        if result.metrics.average_test_quality_score < 0.8:
            print("   - Review generated tests and improve quality")
        
        print()
        
    else:
        print("❌ Orchestration failed!")
        print(f"Error: {result.error_message}")
        return False
    
    return True


def run_demo_orchestration():
    """Run a demonstration with a sample project structure."""
    
    print("🎬 TestMaster Demo - Creating Sample Project")
    print("=" * 50)
    
    # Create a sample project structure
    demo_dir = Path("./testmaster_demo")
    demo_dir.mkdir(exist_ok=True)
    
    # Create sample Python files
    sample_files = {
        "main.py": '''"""
Sample application for TestMaster demo.
"""

def calculate_total(items):
    """Calculate total of numeric items."""
    return sum(item for item in items if isinstance(item, (int, float)))

def validate_user_input(user_input):
    """Validate user input - potential security issue."""
    # This has a potential SQL injection vulnerability
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    return query

class UserManager:
    """Manages user operations."""
    
    def __init__(self):
        self.users = []
    
    def add_user(self, username, password):
        """Add a new user - hardcoded credentials issue."""
        # Hardcoded admin password - security vulnerability
        if username == "admin" and password == "admin123":
            return True
        
        user = {"username": username, "password": password}
        self.users.append(user)
        return True
    
    def authenticate_user(self, username, password):
        """Authenticate user credentials."""
        for user in self.users:
            if user["username"] == username and user["password"] == password:
                return True
        return False
''',
        
        "utils.py": '''"""
Utility functions for the application.
"""

import os
import subprocess

def process_file(filename):
    """Process a file - potential security issue."""
    # Command injection vulnerability
    result = os.system(f"cat {filename}")
    return result

def encrypt_data(data, key = os.getenv('KEY')):
    """Encrypt data - weak crypto."""
    # Weak encryption implementation
    encrypted = ""
    for char in data:
        encrypted += chr(ord(char) ^ ord(key[0]))
    return encrypted

def log_activity(activity):
    """Log user activity."""
    with open("activity.log", "a") as f:
        f.write(f"{activity}\\n")
''',
        
        "api.py": '''"""
API endpoints for the application.
"""

from flask import Flask, request

app = Flask(__name__)

@app.route("/api/user", methods=["POST"])
def create_user():
    """Create user endpoint - potential XSS."""
    username = request.form.get("username", "")
    # XSS vulnerability - no input sanitization
    return f"<h1>Welcome {username}!</h1>"

@app.route("/api/data", methods=["GET"])
def get_data():
    """Get data endpoint - potential information disclosure."""
    # Sensitive data exposure
    config = {
        "database_password": "super_secret_123",
        "api_key": "sk-1234567890abcdef",
        "debug": True
    }
    return config

if __name__ == "__main__":
    app.run(debug=True)
'''
    }
    
    # Write sample files
    for filename, content in sample_files.items():
        file_path = demo_dir / filename
        file_path.write_text(content)
        print(f"   Created: {file_path}")
    
    print()
    
    # Run orchestration on demo project
    success = run_enterprise_orchestration(str(demo_dir))
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print(f"📁 Demo project: {demo_dir.absolute()}")
        print(f"📁 Generated tests: {demo_dir.absolute()}_testmaster_output")
        print("\nThe demo showcased:")
        print("   ✓ Codebase analysis and language detection")
        print("   ✓ Security vulnerability scanning")
        print("   ✓ Compliance assessment")
        print("   ✓ Intelligence-enhanced test generation")
        print("   ✓ Multi-format output generation")
        print("   ✓ Comprehensive documentation")
    
    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TestMaster Unified Orchestration Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_orchestration_example.py --demo
  python unified_orchestration_example.py --target ./my_project
  python unified_orchestration_example.py --target ./my_project --output ./test_output
        """
    )
    
    parser.add_argument(
        "--target",
        type=str,
        help="Target directory to analyze and generate tests for"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for generated tests (default: {target}_testmaster_output)"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration with sample project"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "intelligent", "security_focused", "compliance", "comprehensive", "rapid", "enterprise"],
        default="comprehensive",
        help="Orchestration mode (default: comprehensive)"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        return run_demo_orchestration()
    
    elif args.target:
        if not Path(args.target).exists():
            print(f"❌ Target directory does not exist: {args.target}")
            return False
        
        return run_enterprise_orchestration(args.target, args.output)
    
    else:
        parser.print_help()
        print("\n💡 Try: python unified_orchestration_example.py --demo")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)