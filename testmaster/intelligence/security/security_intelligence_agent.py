"""
Security Intelligence Agent

Intelligent security-focused test generation that combines vulnerability scanning,
OWASP compliance checking, and hierarchical test planning for comprehensive
security testing coverage.
"""

import re
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from enum import Enum
from pathlib import Path
from datetime import datetime

from ...security.universal_scanner import UniversalSecurityScanner, VulnerabilityType, SeverityLevel
from ...security.compliance_framework import ComplianceFramework, ComplianceStandard
from ..hierarchical_planning import (
    HierarchicalTestPlanner, 
    PlanningNode, 
    TestPlanGenerator, 
    TestPlanEvaluator,
    EvaluationCriteria,
    get_best_planner
)
from ..consensus import AgentCoordinator, AgentVote
from ..consensus.agent_coordination import AgentRole
from ...core.shared_state import get_shared_state, cache_test_result, get_cached_test_result


class SecurityTestStrategy(Enum):
    """Security testing strategies."""
    VULNERABILITY_FOCUSED = "vulnerability_focused"
    COMPLIANCE_DRIVEN = "compliance_driven"
    THREAT_MODEL_BASED = "threat_model_based"
    PENETRATION_TESTING = "penetration_testing"
    SECURE_CODE_REVIEW = "secure_code_review"


@dataclass
class SecurityTestPlan:
    """Plan for security testing."""
    strategy: SecurityTestStrategy
    target_vulnerabilities: List[VulnerabilityType]
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    test_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    expected_coverage: float = 0.8
    priority_score: float = 0.5
    estimated_time: float = 60.0


@dataclass
class SecurityFinding:
    """Security analysis finding."""
    vulnerability_type: VulnerabilityType
    severity: SeverityLevel
    location: str
    description: str
    impact: str
    remediation: str
    test_scenarios: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecurityPlanGenerator(TestPlanGenerator):
    """Generates security-focused test plans."""
    
    def __init__(self, scanner: UniversalSecurityScanner = None):
        self.scanner = scanner or UniversalSecurityScanner()
        self.compliance_framework = ComplianceFramework()
        
        # OWASP Top 10 patterns
        self.owasp_patterns = {
            VulnerabilityType.SQL_INJECTION: [
                "SELECT * FROM users WHERE id = '{}'",
                "INSERT INTO table VALUES ('{}')",
                "DELETE FROM table WHERE condition = '{}'"
            ],
            VulnerabilityType.XSS: [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "onload=alert('xss')"
            ],
            VulnerabilityType.COMMAND_INJECTION: [
                "; rm -rf /",
                "| cat /etc/passwd",
                "&& whoami"
            ]
        }
        
        print("Security Plan Generator initialized")
        print(f"   OWASP patterns loaded: {len(self.owasp_patterns)}")
    
    def generate(self, parent_node: PlanningNode, context: Dict[str, Any]) -> List[PlanningNode]:
        """Generate security test plans."""
        
        module_path = context.get('module_path', '')
        module_analysis = context.get('module_analysis', {})
        
        # Scan for vulnerabilities first
        scan_results = self._analyze_security_context(module_path, module_analysis)
        
        # Generate different security strategies
        strategies = self._determine_security_strategies(scan_results, context)
        
        children = []
        for i, strategy in enumerate(strategies):
            plan = self._create_security_plan(strategy, scan_results, module_analysis)
            
            child_node = PlanningNode(
                id=f"{parent_node.id}_security_{i}",
                content={
                    'strategy': strategy.value,
                    'security_plan': plan,
                    'target_vulnerabilities': [v.value for v in plan.target_vulnerabilities],
                    'compliance_standards': [s.value for s in plan.compliance_standards],
                    'test_scenarios': plan.test_scenarios,
                    'expected_coverage': plan.expected_coverage,
                    'estimated_time': plan.estimated_time,
                    'priority_score': plan.priority_score
                },
                depth=parent_node.depth + 1
            )
            
            children.append(child_node)
        
        return children
    
    def _analyze_security_context(self, module_path: str, module_analysis: Dict) -> List[SecurityFinding]:
        """Analyze module for security vulnerabilities."""
        findings = []
        
        # Get source code for analysis
        source_code = module_analysis.get('source_code', '')
        functions = module_analysis.get('functions', [])
        
        # Scan for common vulnerability patterns
        for vuln_type, patterns in self.owasp_patterns.items():
            for pattern in patterns:
                if self._check_vulnerability_pattern(source_code, pattern, vuln_type):
                    finding = SecurityFinding(
                        vulnerability_type=vuln_type,
                        severity=self._assess_severity(vuln_type, source_code),
                        location=module_path,
                        description=f"Potential {vuln_type.value} vulnerability detected",
                        impact=self._assess_impact(vuln_type),
                        remediation=self._get_remediation(vuln_type),
                        confidence=0.7  # Pattern-based detection
                    )
                    findings.append(finding)
        
        # Check for security-sensitive functions
        security_functions = ['eval', 'exec', 'open', 'subprocess', 'os.system']
        for func_name in security_functions:
            if func_name in source_code:
                finding = SecurityFinding(
                    vulnerability_type=VulnerabilityType.INJECTION,
                    severity=SeverityLevel.HIGH,
                    location=module_path,
                    description=f"Security-sensitive function '{func_name}' detected",
                    impact="Could allow code injection or file system access",
                    remediation=f"Validate inputs to {func_name} and use safer alternatives",
                    confidence=0.9
                )
                findings.append(finding)
        
        return findings
    
    def _check_vulnerability_pattern(self, source_code: str, pattern: str, vuln_type: VulnerabilityType) -> bool:
        """Check if source code contains vulnerability pattern."""
        # Simple pattern matching - in production this would be more sophisticated
        if vuln_type == VulnerabilityType.SQL_INJECTION:
            return 'SELECT' in source_code.upper() and '{}' in source_code
        elif vuln_type == VulnerabilityType.XSS:
            return '<script>' in source_code.lower()
        elif vuln_type == VulnerabilityType.COMMAND_INJECTION:
            return any(cmd in source_code for cmd in ['os.system', 'subprocess.call', 'exec'])
        
        return False
    
    def _assess_severity(self, vuln_type: VulnerabilityType, source_code: str) -> SeverityLevel:
        """Assess severity of vulnerability."""
        critical_types = [VulnerabilityType.SQL_INJECTION, VulnerabilityType.COMMAND_INJECTION]
        high_types = [VulnerabilityType.XSS, VulnerabilityType.AUTHENTICATION_BYPASS]
        
        if vuln_type in critical_types:
            return SeverityLevel.CRITICAL
        elif vuln_type in high_types:
            return SeverityLevel.HIGH
        else:
            return SeverityLevel.MEDIUM
    
    def _assess_impact(self, vuln_type: VulnerabilityType) -> str:
        """Assess impact of vulnerability."""
        impact_map = {
            VulnerabilityType.SQL_INJECTION: "Database compromise, data theft, unauthorized access",
            VulnerabilityType.XSS: "Session hijacking, credential theft, malicious script execution",
            VulnerabilityType.COMMAND_INJECTION: "Remote code execution, system compromise",
            VulnerabilityType.AUTHENTICATION_BYPASS: "Unauthorized access to protected resources",
            VulnerabilityType.PATH_TRAVERSAL: "Unauthorized file access, information disclosure"
        }
        return impact_map.get(vuln_type, "Security vulnerability with potential for exploitation")
    
    def _get_remediation(self, vuln_type: VulnerabilityType) -> str:
        """Get remediation advice for vulnerability."""
        remediation_map = {
            VulnerabilityType.SQL_INJECTION: "Use parameterized queries, input validation, ORM frameworks",
            VulnerabilityType.XSS: "Output encoding, Content Security Policy, input validation",
            VulnerabilityType.COMMAND_INJECTION: "Input validation, avoid system calls, use safe APIs",
            VulnerabilityType.AUTHENTICATION_BYPASS: "Implement proper authentication checks, session management",
            VulnerabilityType.PATH_TRAVERSAL: "Path validation, restrict file access, use allowlists"
        }
        return remediation_map.get(vuln_type, "Follow secure coding practices and validate all inputs")
    
    def _determine_security_strategies(self, findings: List[SecurityFinding], context: Dict) -> List[SecurityTestStrategy]:
        """Determine appropriate security testing strategies."""
        strategies = []
        
        # Always include vulnerability-focused if findings exist
        if findings:
            strategies.append(SecurityTestStrategy.VULNERABILITY_FOCUSED)
        
        # Add compliance-driven for certain contexts
        if context.get('compliance_required', False):
            strategies.append(SecurityTestStrategy.COMPLIANCE_DRIVEN)
        
        # Add threat modeling for complex modules
        complexity = context.get('module_analysis', {}).get('complexity', 0)
        if complexity > 10:
            strategies.append(SecurityTestStrategy.THREAT_MODEL_BASED)
        
        # Add secure code review for all modules
        strategies.append(SecurityTestStrategy.SECURE_CODE_REVIEW)
        
        # Default strategy if none selected
        if not strategies:
            strategies.append(SecurityTestStrategy.VULNERABILITY_FOCUSED)
        
        return strategies
    
    def _create_security_plan(self, strategy: SecurityTestStrategy, 
                            findings: List[SecurityFinding], 
                            module_analysis: Dict) -> SecurityTestPlan:
        """Create security test plan for strategy."""
        
        if strategy == SecurityTestStrategy.VULNERABILITY_FOCUSED:
            return self._create_vulnerability_plan(findings)
        elif strategy == SecurityTestStrategy.COMPLIANCE_DRIVEN:
            return self._create_compliance_plan(findings)
        elif strategy == SecurityTestStrategy.THREAT_MODEL_BASED:
            return self._create_threat_model_plan(module_analysis)
        elif strategy == SecurityTestStrategy.SECURE_CODE_REVIEW:
            return self._create_code_review_plan(module_analysis)
        else:
            return self._create_default_security_plan()
    
    def _create_vulnerability_plan(self, findings: List[SecurityFinding]) -> SecurityTestPlan:
        """Create vulnerability-focused test plan."""
        target_vulns = list(set(f.vulnerability_type for f in findings))
        
        test_scenarios = []
        for finding in findings:
            scenarios = self._generate_test_scenarios(finding)
            test_scenarios.extend(scenarios)
        
        return SecurityTestPlan(
            strategy=SecurityTestStrategy.VULNERABILITY_FOCUSED,
            target_vulnerabilities=target_vulns,
            test_scenarios=test_scenarios,
            expected_coverage=0.9,
            priority_score=0.8,
            estimated_time=len(findings) * 15.0
        )
    
    def _create_compliance_plan(self, findings: List[SecurityFinding]) -> SecurityTestPlan:
        """Create compliance-driven test plan."""
        return SecurityTestPlan(
            strategy=SecurityTestStrategy.COMPLIANCE_DRIVEN,
            target_vulnerabilities=[VulnerabilityType.AUTHENTICATION_BYPASS, VulnerabilityType.AUTHORIZATION_FAILURE],
            compliance_standards=[ComplianceStandard.OWASP_ASVS, ComplianceStandard.PCI_DSS],
            test_scenarios=[
                {"test": "authentication_bypass", "description": "Test authentication mechanisms"},
                {"test": "authorization_check", "description": "Verify access controls"},
                {"test": "data_protection", "description": "Test sensitive data handling"}
            ],
            expected_coverage=0.85,
            priority_score=0.9,
            estimated_time=45.0
        )
    
    def _create_threat_model_plan(self, module_analysis: Dict) -> SecurityTestPlan:
        """Create threat model-based test plan."""
        return SecurityTestPlan(
            strategy=SecurityTestStrategy.THREAT_MODEL_BASED,
            target_vulnerabilities=[VulnerabilityType.INJECTION, VulnerabilityType.INSECURE_DESIGN],
            test_scenarios=[
                {"test": "threat_modeling", "description": "Identify potential attack vectors"},
                {"test": "attack_simulation", "description": "Simulate common attacks"},
                {"test": "defense_validation", "description": "Validate security controls"}
            ],
            expected_coverage=0.7,
            priority_score=0.7,
            estimated_time=60.0
        )
    
    def _create_code_review_plan(self, module_analysis: Dict) -> SecurityTestPlan:
        """Create secure code review plan."""
        return SecurityTestPlan(
            strategy=SecurityTestStrategy.SECURE_CODE_REVIEW,
            target_vulnerabilities=[VulnerabilityType.HARDCODED_CREDENTIALS, VulnerabilityType.WEAK_RANDOMNESS],
            test_scenarios=[
                {"test": "credential_scan", "description": "Scan for hardcoded credentials"},
                {"test": "crypto_review", "description": "Review cryptographic implementations"},
                {"test": "input_validation", "description": "Check input validation mechanisms"}
            ],
            expected_coverage=0.6,
            priority_score=0.5,
            estimated_time=30.0
        )
    
    def _create_default_security_plan(self) -> SecurityTestPlan:
        """Create default security test plan."""
        return SecurityTestPlan(
            strategy=SecurityTestStrategy.VULNERABILITY_FOCUSED,
            target_vulnerabilities=[VulnerabilityType.INJECTION, VulnerabilityType.XSS],
            test_scenarios=[
                {"test": "basic_injection", "description": "Test for injection vulnerabilities"},
                {"test": "xss_check", "description": "Check for XSS vulnerabilities"}
            ],
            expected_coverage=0.5,
            priority_score=0.4,
            estimated_time=20.0
        )
    
    def _generate_test_scenarios(self, finding: SecurityFinding) -> List[Dict[str, Any]]:
        """Generate test scenarios for a security finding."""
        scenarios = []
        
        vuln_type = finding.vulnerability_type
        
        if vuln_type == VulnerabilityType.SQL_INJECTION:
            scenarios.extend([
                {"test": "sql_injection_basic", "payload": "' OR '1'='1", "description": "Basic SQL injection test"},
                {"test": "sql_injection_union", "payload": "' UNION SELECT * FROM users--", "description": "Union-based SQL injection"},
                {"test": "sql_injection_blind", "payload": "' AND 1=1--", "description": "Blind SQL injection test"}
            ])
        elif vuln_type == VulnerabilityType.XSS:
            scenarios.extend([
                {"test": "xss_reflected", "payload": "<script>alert('xss')</script>", "description": "Reflected XSS test"},
                {"test": "xss_stored", "payload": "<img src=x onerror=alert('xss')>", "description": "Stored XSS test"},
                {"test": "xss_dom", "payload": "javascript:alert('xss')", "description": "DOM-based XSS test"}
            ])
        elif vuln_type == VulnerabilityType.COMMAND_INJECTION:
            scenarios.extend([
                {"test": "cmd_injection_basic", "payload": "; ls -la", "description": "Basic command injection"},
                {"test": "cmd_injection_pipe", "payload": "| cat /etc/passwd", "description": "Pipe-based command injection"}
            ])
        
        return scenarios


class SecurityPlanEvaluator(TestPlanEvaluator):
    """Evaluates security test plans."""
    
    def __init__(self):
        self.evaluation_weights = {
            'security_coverage': 0.4,
            'vulnerability_detection': 0.3,
            'compliance_alignment': 0.2,
            'implementation_feasibility': 0.1
        }
        
        print("Security Plan Evaluator initialized")
    
    def evaluate(self, node: PlanningNode, criteria: List[EvaluationCriteria]) -> float:
        """Evaluate security test plan."""
        plan_content = node.content
        security_plan = plan_content.get('security_plan')
        
        if not security_plan:
            return 0.0
        
        # Calculate individual scores
        coverage_score = self._evaluate_coverage(security_plan)
        detection_score = self._evaluate_detection_capability(security_plan)
        compliance_score = self._evaluate_compliance(security_plan)
        feasibility_score = self._evaluate_feasibility(security_plan)
        
        # Store individual scores
        node.update_score('security_coverage', coverage_score)
        node.update_score('vulnerability_detection', detection_score)
        node.update_score('compliance_alignment', compliance_score)
        node.update_score('implementation_feasibility', feasibility_score)
        
        # Calculate weighted aggregate
        aggregate = (
            coverage_score * self.evaluation_weights['security_coverage'] +
            detection_score * self.evaluation_weights['vulnerability_detection'] +
            compliance_score * self.evaluation_weights['compliance_alignment'] +
            feasibility_score * self.evaluation_weights['implementation_feasibility']
        )
        
        return aggregate
    
    def _evaluate_coverage(self, plan: SecurityTestPlan) -> float:
        """Evaluate security coverage."""
        # More vulnerability types = better coverage
        vuln_coverage = min(1.0, len(plan.target_vulnerabilities) / 10.0)
        
        # More test scenarios = better coverage
        scenario_coverage = min(1.0, len(plan.test_scenarios) / 15.0)
        
        # Expected coverage from plan
        expected_coverage = plan.expected_coverage
        
        return (vuln_coverage + scenario_coverage + expected_coverage) / 3.0
    
    def _evaluate_detection_capability(self, plan: SecurityTestPlan) -> float:
        """Evaluate vulnerability detection capability."""
        # Critical vulnerabilities get higher scores
        critical_vulns = [
            VulnerabilityType.SQL_INJECTION,
            VulnerabilityType.COMMAND_INJECTION,
            VulnerabilityType.AUTHENTICATION_BYPASS
        ]
        
        critical_count = sum(1 for v in plan.target_vulnerabilities if v in critical_vulns)
        critical_score = min(1.0, critical_count / 3.0)
        
        # Strategy effectiveness
        strategy_scores = {
            SecurityTestStrategy.VULNERABILITY_FOCUSED: 0.9,
            SecurityTestStrategy.PENETRATION_TESTING: 0.85,
            SecurityTestStrategy.THREAT_MODEL_BASED: 0.8,
            SecurityTestStrategy.COMPLIANCE_DRIVEN: 0.7,
            SecurityTestStrategy.SECURE_CODE_REVIEW: 0.6
        }
        
        strategy_score = strategy_scores.get(plan.strategy, 0.5)
        
        return (critical_score + strategy_score) / 2.0
    
    def _evaluate_compliance(self, plan: SecurityTestPlan) -> float:
        """Evaluate compliance alignment."""
        if plan.compliance_standards:
            return 1.0  # Has compliance standards
        
        # Check if strategy supports compliance
        compliance_strategies = [
            SecurityTestStrategy.COMPLIANCE_DRIVEN,
            SecurityTestStrategy.SECURE_CODE_REVIEW
        ]
        
        if plan.strategy in compliance_strategies:
            return 0.7
        
        return 0.3
    
    def _evaluate_feasibility(self, plan: SecurityTestPlan) -> float:
        """Evaluate implementation feasibility."""
        # Time feasibility
        time_score = 1.0
        if plan.estimated_time > 120:
            time_score = 0.5
        elif plan.estimated_time > 60:
            time_score = 0.7
        
        # Scenario complexity
        scenario_count = len(plan.test_scenarios)
        complexity_score = 1.0
        if scenario_count > 20:
            complexity_score = 0.6
        elif scenario_count > 10:
            complexity_score = 0.8
        
        return (time_score + complexity_score) / 2.0


class VulnerabilityTestGenerator:
    """Generates specific tests for detected vulnerabilities."""
    
    def __init__(self):
        self.test_templates = self._load_test_templates()
        print("Vulnerability Test Generator initialized")
    
    def generate_vulnerability_tests(self, finding: SecurityFinding) -> str:
        """Generate specific tests for a vulnerability."""
        vuln_type = finding.vulnerability_type
        template = self.test_templates.get(vuln_type)
        
        if not template:
            return self._generate_generic_test(finding)
        
        return template.format(
            location=finding.location,
            description=finding.description,
            severity=finding.severity.value
        )
    
    def _load_test_templates(self) -> Dict[VulnerabilityType, str]:
        """Load test templates for different vulnerability types."""
        return {
            VulnerabilityType.SQL_INJECTION: '''
def test_sql_injection_protection_{location}():
    """Test SQL injection protection for {location}."""
    # Test basic SQL injection
    malicious_input = "' OR '1'='1"
    result = process_input(malicious_input)
    assert result is None or "error" in str(result).lower()
    
    # Test union-based injection
    union_payload = "' UNION SELECT * FROM users--"
    result = process_input(union_payload)
    assert result is None or "error" in str(result).lower()
''',
            VulnerabilityType.XSS: '''
def test_xss_protection_{location}():
    """Test XSS protection for {location}."""
    # Test script injection
    xss_payload = "<script>alert('xss')</script>"
    result = process_input(xss_payload)
    assert "<script>" not in str(result)
    
    # Test event handler injection
    event_payload = "<img src=x onerror=alert('xss')>"
    result = process_input(event_payload)
    assert "onerror" not in str(result)
''',
            VulnerabilityType.COMMAND_INJECTION: '''
def test_command_injection_protection_{location}():
    """Test command injection protection for {location}."""
    # Test command chaining
    cmd_payload = "; ls -la"
    with pytest.raises((ValueError, SecurityError)):
        execute_command(cmd_payload)
    
    # Test pipe injection
    pipe_payload = "| cat /etc/passwd"
    with pytest.raises((ValueError, SecurityError)):
        execute_command(pipe_payload)
'''
        }
    
    def _generate_generic_test(self, finding: SecurityFinding) -> str:
        """Generate generic security test."""
        return f'''
def test_security_{finding.vulnerability_type.value}():
    """Test for {finding.vulnerability_type.value} vulnerability."""
    # {finding.description}
    # Severity: {finding.severity.value}
    # Remediation: {finding.remediation}
    
    # TODO: Implement specific test for {finding.vulnerability_type.value}
    assert True  # Placeholder
'''


class OWASPComplianceChecker:
    """Checks OWASP compliance and generates compliance tests."""
    
    def __init__(self):
        self.owasp_top_10 = [
            VulnerabilityType.INJECTION,
            VulnerabilityType.AUTHENTICATION_BYPASS,
            VulnerabilityType.SENSITIVE_DATA_EXPOSURE,
            VulnerabilityType.XSS,
            VulnerabilityType.SECURITY_MISCONFIGURATION,
            VulnerabilityType.VULNERABLE_COMPONENTS,
            VulnerabilityType.AUTHORIZATION_FAILURE,
            VulnerabilityType.LOGGING_MONITORING_FAILURE,
            VulnerabilityType.SSRF,
            VulnerabilityType.CRYPTO_FAILURE
        ]
        
        print("OWASP Compliance Checker initialized")
        print(f"   Monitoring {len(self.owasp_top_10)} OWASP Top 10 categories")
    
    def check_compliance(self, findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Check OWASP compliance based on findings."""
        found_vulns = {f.vulnerability_type for f in findings}
        
        compliance_score = 0.0
        covered_categories = 0
        
        category_status = {}
        for vuln_type in self.owasp_top_10:
            if vuln_type in found_vulns:
                category_status[vuln_type.value] = "VULNERABLE"
            else:
                category_status[vuln_type.value] = "COMPLIANT"
                covered_categories += 1
        
        compliance_score = covered_categories / len(self.owasp_top_10)
        
        return {
            "compliance_score": compliance_score,
            "categories_compliant": covered_categories,
            "total_categories": len(self.owasp_top_10),
            "category_status": category_status,
            "recommendations": self._generate_compliance_recommendations(found_vulns)
        }
    
    def _generate_compliance_recommendations(self, found_vulns: Set[VulnerabilityType]) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        for vuln_type in found_vulns:
            if vuln_type == VulnerabilityType.INJECTION:
                recommendations.append("Implement parameterized queries and input validation")
            elif vuln_type == VulnerabilityType.XSS:
                recommendations.append("Implement output encoding and Content Security Policy")
            elif vuln_type == VulnerabilityType.AUTHENTICATION_BYPASS:
                recommendations.append("Strengthen authentication mechanisms and session management")
        
        return recommendations


class SecurityIntelligenceAgent:
    """Main security intelligence agent for coordinated security testing."""
    
    def __init__(self, coordinator: AgentCoordinator = None):
        self.coordinator = coordinator
        self.shared_state = get_shared_state()
        
        # Initialize components
        self.security_scanner = UniversalSecurityScanner()
        self.plan_generator = SecurityPlanGenerator(self.security_scanner)
        self.plan_evaluator = SecurityPlanEvaluator()
        self.test_generator = VulnerabilityTestGenerator()
        self.compliance_checker = OWASPComplianceChecker()
        
        # Register with coordinator if provided
        if self.coordinator:
            self.coordinator.register_agent(
                "security_intelligence_agent",
                AgentRole.SECURITY_SCANNER,
                weight=1.2,  # Higher weight for security expertise
                specialization=["vulnerability_detection", "owasp_compliance", "security_testing"]
            )
        
        print("Security Intelligence Agent initialized")
        print("   Components: scanner, planner, evaluator, test generator, compliance checker")
    
    def analyze_security(self, module_path: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive security analysis of a module."""
        context = context or {}
        
        print(f"\n🔒 Security Analysis: {module_path}")
        
        # Check cache first
        cached_result = get_cached_test_result(f"security_{module_path}")
        if cached_result:
            print("Using cached security analysis")
            return cached_result
        
        start_time = time.time()
        
        # 1. Scan for vulnerabilities
        scan_results = self.security_scanner.scan_file(Path(module_path))
        
        # 2. Convert to findings
        findings = self._convert_scan_results(scan_results, module_path)
        
        # 3. Check OWASP compliance
        compliance_report = self.compliance_checker.check_compliance(findings)
        
        # 4. Generate security test plan using hierarchical planning
        security_plan = self._generate_security_plan(module_path, findings, context)
        
        # 5. Generate specific vulnerability tests
        vulnerability_tests = []
        for finding in findings:
            test_code = self.test_generator.generate_vulnerability_tests(finding)
            vulnerability_tests.append({
                'vulnerability': finding.vulnerability_type.value,
                'severity': finding.severity.value,
                'test_code': test_code
            })
        
        analysis_time = time.time() - start_time
        
        result = {
            'module_path': module_path,
            'findings': [self._finding_to_dict(f) for f in findings],
            'vulnerability_count': len(findings),
            'highest_severity': self._get_highest_severity(findings),
            'compliance_report': compliance_report,
            'security_plan': security_plan,
            'vulnerability_tests': vulnerability_tests,
            'analysis_time': analysis_time,
            'recommendations': self._generate_recommendations(findings),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache result
        cache_test_result(f"security_{module_path}", result, 85.0)
        
        print(f"   Found {len(findings)} security issues")
        print(f"   OWASP compliance: {compliance_report['compliance_score']:.1%}")
        print(f"   Analysis time: {analysis_time:.2f}s")
        
        return result
    
    def _convert_scan_results(self, scan_results: Dict, module_path: str) -> List[SecurityFinding]:
        """Convert scanner results to security findings."""
        findings = []
        
        vulnerabilities = scan_results.get('vulnerabilities', [])
        for vuln in vulnerabilities:
            finding = SecurityFinding(
                vulnerability_type=VulnerabilityType(vuln.get('type', 'injection')),
                severity=SeverityLevel(vuln.get('severity', 'medium')),
                location=module_path,
                description=vuln.get('description', 'Security vulnerability detected'),
                impact=vuln.get('impact', 'Potential security risk'),
                remediation=vuln.get('remediation', 'Follow secure coding practices'),
                confidence=vuln.get('confidence', 0.8)
            )
            findings.append(finding)
        
        return findings
    
    def _generate_security_plan(self, module_path: str, findings: List[SecurityFinding], context: Dict) -> Dict[str, Any]:
        """Generate security test plan using hierarchical planning."""
        try:
            # Use hierarchical planner for security planning
            planner = get_best_planner(prefer_llm=False)  # Use template-based for security
            planner.plan_generator = self.plan_generator
            planner.plan_evaluator = self.plan_evaluator
            
            # Create planning context
            planning_context = {
                'module_path': module_path,
                'security_findings': findings,
                'vulnerability_count': len(findings),
                **context
            }
            
            # Initial security plan
            initial_plan = {
                'objective': 'comprehensive_security_testing',
                'module_path': module_path,
                'findings_count': len(findings),
                'priority': 'high' if findings else 'medium'
            }
            
            # Execute planning
            planning_tree = planner.plan(initial_plan, planning_context)
            best_plan = planning_tree.get_best_plan()
            
            if best_plan:
                return best_plan[-1].content
            else:
                return self._create_fallback_security_plan(findings)
                
        except Exception as e:
            print(f"Security planning failed: {e}")
            return self._create_fallback_security_plan(findings)
    
    def _create_fallback_security_plan(self, findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Create fallback security plan."""
        return {
            'strategy': 'vulnerability_focused',
            'target_vulnerabilities': [f.vulnerability_type.value for f in findings],
            'estimated_coverage': 0.7,
            'estimated_time': len(findings) * 10.0,
            'test_scenarios': [
                {'test': 'vulnerability_scan', 'description': 'Scan for security vulnerabilities'},
                {'test': 'input_validation', 'description': 'Test input validation'}
            ]
        }
    
    def _finding_to_dict(self, finding: SecurityFinding) -> Dict[str, Any]:
        """Convert security finding to dictionary."""
        return {
            'vulnerability_type': finding.vulnerability_type.value,
            'severity': finding.severity.value,
            'location': finding.location,
            'description': finding.description,
            'impact': finding.impact,
            'remediation': finding.remediation,
            'confidence': finding.confidence,
            'test_scenarios': finding.test_scenarios
        }
    
    def _get_highest_severity(self, findings: List[SecurityFinding]) -> str:
        """Get highest severity from findings."""
        if not findings:
            return 'none'
        
        severity_order = {
            SeverityLevel.CRITICAL: 4,
            SeverityLevel.HIGH: 3,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.LOW: 1,
            SeverityLevel.INFO: 0
        }
        
        highest = max(findings, key=lambda f: severity_order[f.severity])
        return highest.severity.value
    
    def _generate_recommendations(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        vuln_types = {f.vulnerability_type for f in findings}
        
        if VulnerabilityType.SQL_INJECTION in vuln_types:
            recommendations.append("Implement parameterized queries and input validation")
        
        if VulnerabilityType.XSS in vuln_types:
            recommendations.append("Implement output encoding and Content Security Policy")
        
        if VulnerabilityType.COMMAND_INJECTION in vuln_types:
            recommendations.append("Avoid system calls and validate all inputs")
        
        if not recommendations:
            recommendations.append("Follow secure coding practices and conduct regular security reviews")
        
        return recommendations
    
    def coordinate_security_consensus(self, task_description: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Coordinate with other agents for security consensus."""
        if not self.coordinator:
            return None
        
        # Create coordination task
        task_id = self.coordinator.create_coordination_task(
            description=task_description,
            required_roles={AgentRole.SECURITY_SCANNER, AgentRole.QUALITY_ASSESSOR},
            context=context
        )
        
        # Submit security assessment vote
        security_score = context.get('security_score', 0.5)
        self.coordinator.submit_vote(
            task_id=task_id,
            agent_id="security_intelligence_agent",
            choice=security_score,
            confidence=0.9,
            reasoning="Security analysis based on vulnerability scan and OWASP compliance"
        )
        
        # Wait for consensus (in real implementation, this would be event-driven)
        time.sleep(2)
        
        result = self.coordinator.get_coordination_result(task_id)
        return result.to_dict() if result else None


def test_security_intelligence():
    """Test the security intelligence agent."""
    print("\n" + "="*60)
    print("Testing Security Intelligence Agent")
    print("="*60)
    
    # Create test context
    test_module = '''
def login(username, password):
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    return execute_query(query)

def display_message(message):
    return f"<div>{message}</div>"
    
def run_command(cmd):
    import os
    return os.system(cmd)
'''
    
    context = {
        'module_path': 'vulnerable_module.py',
        'module_analysis': {
            'source_code': test_module,
            'functions': ['login', 'display_message', 'run_command'],
            'complexity': 15
        }
    }
    
    # Create security agent
    agent = SecurityIntelligenceAgent()
    
    # Perform security analysis
    print("\n1. Performing security analysis...")
    result = agent.analyze_security('vulnerable_module.py', context)
    
    print(f"\n2. Analysis Results:")
    print(f"   Vulnerabilities found: {result['vulnerability_count']}")
    print(f"   Highest severity: {result['highest_severity']}")
    print(f"   OWASP compliance: {result['compliance_report']['compliance_score']:.1%}")
    print(f"   Analysis time: {result['analysis_time']:.2f}s")
    
    # Show findings
    if result['findings']:
        print(f"\n3. Security Findings:")
        for i, finding in enumerate(result['findings'][:3], 1):
            print(f"   {i}. {finding['vulnerability_type']} ({finding['severity']})")
            print(f"      {finding['description']}")
    
    # Show recommendations
    print(f"\n4. Recommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print("\n✅ Security Intelligence Agent test completed!")
    return True


if __name__ == "__main__":
    test_security_intelligence()