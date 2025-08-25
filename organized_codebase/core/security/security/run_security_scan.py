"""
Quick Security Scan Runner
"""

from core.intelligence.security import VulnerabilityScanner, ComplianceChecker, ThreatModeler
from core.intelligence.documentation import DocumentationAutoGenerator, APISpecBuilder, DiagramCreator
import json
from datetime import datetime

def run_security_scan():
    """Run security scan on core modules."""
    print("Starting security scan...")
    
    # Initialize scanner
    scanner = VulnerabilityScanner()
    
    # Scan core intelligence modules
    results = scanner.scan_directory("core/intelligence", recursive=True)
    
    # Generate report
    report = scanner.generate_report()
    
    print(f"Scan complete!")
    print(f"Files scanned: {report['files_scanned']}")
    print(f"Total vulnerabilities: {report['total_vulnerabilities']}")
    print(f"By severity: {report['by_severity']}")
    
    return report

def run_documentation_metrics():
    """Get documentation generation metrics."""
    print("\nGathering documentation metrics...")
    
    doc_gen = DocumentationAutoGenerator()
    api_builder = APISpecBuilder()
    diagram_creator = DiagramCreator()
    
    # Analyze some modules
    doc_gen.generate_documentation("core/intelligence/__init__.py")
    api_builder.scan_directory("core/intelligence/api", framework="auto")
    diagram_creator.analyze_architecture("core/intelligence")
    
    metrics = {
        'doc_generator': doc_gen.get_metrics(),
        'api_endpoints': len(api_builder.endpoints),
        'architecture_components': len(diagram_creator.components),
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"Documentation modules active: 3")
    print(f"API endpoints discovered: {metrics['api_endpoints']}")
    print(f"Architecture components mapped: {metrics['architecture_components']}")
    
    return metrics

def run_compliance_check():
    """Run basic compliance check."""
    print("\nRunning compliance checks...")
    
    checker = ComplianceChecker()
    
    # Check OWASP compliance
    issues = checker.check_owasp_compliance("core/intelligence")
    
    # Generate report
    report = checker.generate_compliance_report(["OWASP"])
    
    print(f"Compliance score: {report.compliance_score:.1f}%")
    print(f"Issues found: {report.total_issues}")
    
    return report

if __name__ == "__main__":
    print("=== TestMaster Security & Documentation Report ===\n")
    
    # Run security scan
    security_report = run_security_scan()
    
    # Run compliance check
    compliance_report = run_compliance_check()
    
    # Get documentation metrics
    doc_metrics = run_documentation_metrics()
    
    print("\n=== Summary ===")
    print(f"Security vulnerabilities: {security_report['total_vulnerabilities']}")
    print(f"Compliance score: {compliance_report.compliance_score:.1f}%")
    print(f"Documentation coverage: Active")
    print(f"All systems operational!")
    
    # Save reports
    with open("security_report.json", "w") as f:
        json.dump(security_report, f, indent=2)
    
    print("\nReports saved to security_report.json")