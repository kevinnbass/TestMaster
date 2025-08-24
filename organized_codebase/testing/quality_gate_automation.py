"""
Quality Gate Automation System for TestMaster
Enterprise quality gates with automated decision making and enforcement
"""

import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

class GateType(Enum):
    """Types of quality gates"""
    COVERAGE_GATE = "coverage_gate"
    SECURITY_GATE = "security_gate"
    PERFORMANCE_GATE = "performance_gate"
    RELIABILITY_GATE = "reliability_gate"
    COMPLIANCE_GATE = "compliance_gate"
    CUSTOM_GATE = "custom_gate"

class GateAction(Enum):
    """Actions for gate enforcement"""
    BLOCK = "block"
    WARN = "warn"
    APPROVE = "approve"
    ESCALATE = "escalate"
    CONDITIONAL_APPROVE = "conditional_approve"

class GateStatus(Enum):
    """Gate evaluation status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PENDING = "pending"
    SKIPPED = "skipped"

@dataclass
class QualityRule:
    """Individual quality rule definition"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    operator: str  # ">=", "<=", "==", "!=", ">", "<"
    threshold_value: float
    severity: str  # "critical", "high", "medium", "low"
    weight: float
    enabled: bool = True
    custom_validator: Optional[str] = None

@dataclass
class GateDefinition:
    """Complete quality gate definition"""
    gate_id: str
    name: str
    gate_type: GateType
    description: str
    rules: List[QualityRule]
    enforcement_action: GateAction
    escalation_rules: Dict[str, Any]
    bypass_conditions: List[str]
    notification_settings: Dict[str, Any]
    enabled: bool = True

@dataclass
class RuleEvaluation:
    """Result of rule evaluation"""
    rule_id: str
    rule_name: str
    metric_value: float
    threshold_value: float
    operator: str
    passed: bool
    severity: str
    weight: float
    impact_score: float
    message: str

@dataclass
class GateEvaluation:
    """Complete gate evaluation result"""
    gate_id: str
    gate_name: str
    gate_type: GateType
    status: GateStatus
    overall_score: float
    rule_evaluations: List[RuleEvaluation]
    enforcement_action: GateAction
    blocking_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    evaluation_time: float
    metadata: Dict[str, Any]

@dataclass
class QualityGateResult:
    """Complete quality gate assessment"""
    timestamp: float
    overall_status: GateStatus
    gates_passed: int
    gates_failed: int
    gates_warned: int
    gate_evaluations: List[GateEvaluation]
    final_action: GateAction
    blocking_gates: List[str]
    approval_required: bool
    escalation_triggered: bool
    summary_message: str

class QualityGateAutomation:
    """Enterprise quality gate automation system"""
    
    def __init__(self):
        self.gate_definitions: Dict[str, GateDefinition] = {}
        self.evaluation_history: List[QualityGateResult] = []
        self.custom_validators: Dict[str, Callable] = {}
        self.notification_handlers: Dict[str, Callable] = {}
        self.escalation_handlers: Dict[str, Callable] = {}
        
        # Initialize default gates
        self._initialize_default_gates()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_default_gates(self):
        """Initialize standard quality gates"""
        
        # Coverage Gate
        coverage_rules = [
            QualityRule(
                rule_id="line_coverage_rule",
                name="Line Coverage Threshold",
                description="Minimum line coverage percentage",
                metric_name="line_coverage",
                operator=">=",
                threshold_value=80.0,
                severity="high",
                weight=0.4
            ),
            QualityRule(
                rule_id="branch_coverage_rule",
                name="Branch Coverage Threshold",
                description="Minimum branch coverage percentage",
                metric_name="branch_coverage",
                operator=">=",
                threshold_value=75.0,
                severity="high",
                weight=0.3
            ),
            QualityRule(
                rule_id="function_coverage_rule",
                name="Function Coverage Threshold",
                description="Minimum function coverage percentage",
                metric_name="function_coverage",
                operator=">=",
                threshold_value=90.0,
                severity="medium",
                weight=0.3
            )
        ]
        
        coverage_gate = GateDefinition(
            gate_id="coverage_gate",
            name="Test Coverage Gate",
            gate_type=GateType.COVERAGE_GATE,
            description="Ensures adequate test coverage",
            rules=coverage_rules,
            enforcement_action=GateAction.BLOCK,
            escalation_rules={"escalate_to": "qa_lead", "after_failures": 3},
            bypass_conditions=["emergency_release", "hotfix"],
            notification_settings={"notify_teams": ["dev", "qa"], "channels": ["email", "slack"]}
        )
        
        # Security Gate
        security_rules = [
            QualityRule(
                rule_id="security_coverage_rule",
                name="Security Test Coverage",
                description="Minimum security test coverage",
                metric_name="security_test_coverage",
                operator=">=",
                threshold_value=70.0,
                severity="critical",
                weight=0.5
            ),
            QualityRule(
                rule_id="vulnerability_count_rule", 
                name="Known Vulnerabilities",
                description="Maximum allowed vulnerabilities",
                metric_name="vulnerability_count",
                operator="<=",
                threshold_value=0.0,
                severity="critical",
                weight=0.5
            )
        ]
        
        security_gate = GateDefinition(
            gate_id="security_gate",
            name="Security Quality Gate",
            gate_type=GateType.SECURITY_GATE,
            description="Ensures security standards compliance",
            rules=security_rules,
            enforcement_action=GateAction.BLOCK,
            escalation_rules={"escalate_to": "security_team", "after_failures": 1},
            bypass_conditions=["security_approved"],
            notification_settings={"notify_teams": ["security", "dev"], "priority": "high"}
        )
        
        # Performance Gate
        performance_rules = [
            QualityRule(
                rule_id="test_execution_time_rule",
                name="Test Execution Time",
                description="Maximum test suite execution time",
                metric_name="test_execution_time",
                operator="<=",
                threshold_value=1800.0,  # 30 minutes
                severity="medium",
                weight=0.6
            ),
            QualityRule(
                rule_id="test_pass_rate_rule",
                name="Test Pass Rate",
                description="Minimum test pass rate",
                metric_name="test_pass_rate",
                operator=">=",
                threshold_value=95.0,
                severity="high",
                weight=0.4
            )
        ]
        
        performance_gate = GateDefinition(
            gate_id="performance_gate",
            name="Performance Quality Gate",
            gate_type=GateType.PERFORMANCE_GATE,
            description="Ensures performance standards",
            rules=performance_rules,
            enforcement_action=GateAction.WARN,
            escalation_rules={"escalate_to": "performance_team", "after_failures": 5},
            bypass_conditions=["performance_approved"],
            notification_settings={"notify_teams": ["dev", "performance"]}
        )
        
        # Reliability Gate
        reliability_rules = [
            QualityRule(
                rule_id="flaky_test_rate_rule",
                name="Flaky Test Rate",
                description="Maximum flaky test percentage",
                metric_name="flaky_test_rate",
                operator="<=",
                threshold_value=5.0,
                severity="medium",
                weight=0.7
            ),
            QualityRule(
                rule_id="test_reliability_score_rule",
                name="Test Reliability Score",
                description="Minimum test reliability score",
                metric_name="test_reliability_score",
                operator=">=",
                threshold_value=85.0,
                severity="medium",
                weight=0.3
            )
        ]
        
        reliability_gate = GateDefinition(
            gate_id="reliability_gate",
            name="Reliability Quality Gate",
            gate_type=GateType.RELIABILITY_GATE,
            description="Ensures test reliability standards",
            rules=reliability_rules,
            enforcement_action=GateAction.WARN,
            escalation_rules={"escalate_to": "qa_lead", "after_failures": 7},
            bypass_conditions=["reliability_waiver"],
            notification_settings={"notify_teams": ["qa", "dev"]}
        )
        
        # Store gates
        self.gate_definitions[coverage_gate.gate_id] = coverage_gate
        self.gate_definitions[security_gate.gate_id] = security_gate
        self.gate_definitions[performance_gate.gate_id] = performance_gate
        self.gate_definitions[reliability_gate.gate_id] = reliability_gate
    
    def evaluate_quality_gates(self, metrics: Dict[str, float], 
                              context: Optional[Dict[str, Any]] = None) -> QualityGateResult:
        """Evaluate all quality gates against provided metrics"""
        start_time = time.time()
        context = context or {}
        
        gate_evaluations = []
        gates_passed = 0
        gates_failed = 0
        gates_warned = 0
        blocking_gates = []
        approval_required = False
        escalation_triggered = False
        
        # Evaluate each enabled gate
        for gate_def in self.gate_definitions.values():
            if not gate_def.enabled:
                continue
            
            # Check bypass conditions
            if self._check_bypass_conditions(gate_def, context):
                self.logger.info(f"Gate {gate_def.gate_id} bypassed due to conditions")
                continue
            
            gate_eval = self._evaluate_single_gate(gate_def, metrics, context)
            gate_evaluations.append(gate_eval)
            
            # Update counters
            if gate_eval.status == GateStatus.PASSED:
                gates_passed += 1
            elif gate_eval.status == GateStatus.FAILED:
                gates_failed += 1
                if gate_eval.enforcement_action == GateAction.BLOCK:
                    blocking_gates.append(gate_eval.gate_id)
            elif gate_eval.status == GateStatus.WARNING:
                gates_warned += 1
            
            # Check for approval requirements
            if gate_eval.enforcement_action == GateAction.ESCALATE:
                escalation_triggered = True
                approval_required = True
            elif gate_eval.enforcement_action == GateAction.CONDITIONAL_APPROVE:
                approval_required = True
        
        # Determine overall status and final action
        overall_status, final_action = self._determine_overall_status(gate_evaluations, blocking_gates)
        
        # Generate summary message
        summary_message = self._generate_summary_message(
            overall_status, gates_passed, gates_failed, gates_warned, blocking_gates
        )
        
        evaluation_time = time.time() - start_time
        
        result = QualityGateResult(
            timestamp=time.time(),
            overall_status=overall_status,
            gates_passed=gates_passed,
            gates_failed=gates_failed,
            gates_warned=gates_warned,
            gate_evaluations=gate_evaluations,
            final_action=final_action,
            blocking_gates=blocking_gates,
            approval_required=approval_required,
            escalation_triggered=escalation_triggered,
            summary_message=summary_message
        )
        
        # Store in history
        self.evaluation_history.append(result)
        if len(self.evaluation_history) > 1000:  # Keep last 1000 evaluations
            self.evaluation_history = self.evaluation_history[-1000:]
        
        # Handle notifications and escalations
        self._handle_notifications(result)
        if escalation_triggered:
            self._handle_escalations(result)
        
        self.logger.info(f"Quality gates evaluation completed in {evaluation_time:.2f}s: {overall_status.value}")
        
        return result
    
    def _evaluate_single_gate(self, gate_def: GateDefinition, metrics: Dict[str, float],
                            context: Dict[str, Any]) -> GateEvaluation:
        """Evaluate a single quality gate"""
        rule_evaluations = []
        total_weight = 0.0
        weighted_score = 0.0
        blocking_issues = []
        warnings = []
        
        for rule in gate_def.rules:
            if not rule.enabled:
                continue
            
            rule_eval = self._evaluate_rule(rule, metrics, context)
            rule_evaluations.append(rule_eval)
            
            # Calculate weighted score
            total_weight += rule.weight
            if rule_eval.passed:
                weighted_score += rule.weight
            
            # Collect issues
            if not rule_eval.passed:
                if rule.severity in ["critical", "high"]:
                    blocking_issues.append(rule_eval.message)
                else:
                    warnings.append(rule_eval.message)
        
        # Calculate overall gate score
        overall_score = (weighted_score / total_weight * 100) if total_weight > 0 else 0.0
        
        # Determine gate status
        if len(blocking_issues) > 0:
            status = GateStatus.FAILED
            enforcement_action = gate_def.enforcement_action
        elif len(warnings) > 0:
            status = GateStatus.WARNING
            enforcement_action = GateAction.WARN
        else:
            status = GateStatus.PASSED
            enforcement_action = GateAction.APPROVE
        
        # Generate recommendations
        recommendations = self._generate_gate_recommendations(rule_evaluations, gate_def)
        
        return GateEvaluation(
            gate_id=gate_def.gate_id,
            gate_name=gate_def.name,
            gate_type=gate_def.gate_type,
            status=status,
            overall_score=overall_score,
            rule_evaluations=rule_evaluations,
            enforcement_action=enforcement_action,
            blocking_issues=blocking_issues,
            warnings=warnings,
            recommendations=recommendations,
            evaluation_time=time.time(),
            metadata={"context": context}
        )
    
    def _evaluate_rule(self, rule: QualityRule, metrics: Dict[str, float],
                      context: Dict[str, Any]) -> RuleEvaluation:
        """Evaluate a single quality rule"""
        
        # Get metric value
        metric_value = metrics.get(rule.metric_name, 0.0)
        
        # Apply custom validator if exists
        if rule.custom_validator and rule.custom_validator in self.custom_validators:
            validator = self.custom_validators[rule.custom_validator]
            passed = validator(metric_value, rule.threshold_value, context)
        else:
            # Standard operator evaluation
            passed = self._evaluate_operator(metric_value, rule.operator, rule.threshold_value)
        
        # Calculate impact score
        impact_score = self._calculate_rule_impact(rule, metric_value, passed)
        
        # Generate message
        if passed:
            message = f"✓ {rule.name}: {metric_value} {rule.operator} {rule.threshold_value}"
        else:
            message = f"✗ {rule.name}: {metric_value} does not meet {rule.operator} {rule.threshold_value}"
        
        return RuleEvaluation(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            metric_value=metric_value,
            threshold_value=rule.threshold_value,
            operator=rule.operator,
            passed=passed,
            severity=rule.severity,
            weight=rule.weight,
            impact_score=impact_score,
            message=message
        )
    
    def _evaluate_operator(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate operator comparison"""
        if operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == "==":
            return abs(value - threshold) < 0.001  # Float comparison
        elif operator == "!=":
            return abs(value - threshold) >= 0.001
        else:
            raise ValueError(f"Unsupported operator: {operator}")
    
    def _calculate_rule_impact(self, rule: QualityRule, metric_value: float, passed: bool) -> float:
        """Calculate impact score for rule failure"""
        if passed:
            return 0.0
        
        # Calculate deviation from threshold
        if rule.operator in [">=", ">"]:
            deviation = max(0, rule.threshold_value - metric_value) / rule.threshold_value
        elif rule.operator in ["<=", "<"]:
            deviation = max(0, metric_value - rule.threshold_value) / rule.threshold_value
        else:
            deviation = abs(metric_value - rule.threshold_value) / rule.threshold_value
        
        # Weight by severity and rule weight
        severity_weights = {"critical": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4}
        severity_weight = severity_weights.get(rule.severity, 0.5)
        
        impact_score = deviation * severity_weight * rule.weight
        return min(1.0, impact_score)
    
    def _check_bypass_conditions(self, gate_def: GateDefinition, context: Dict[str, Any]) -> bool:
        """Check if gate should be bypassed"""
        for condition in gate_def.bypass_conditions:
            if condition in context.get("bypass_flags", []):
                return True
            if context.get(condition, False):
                return True
        return False
    
    def _determine_overall_status(self, gate_evaluations: List[GateEvaluation],
                                blocking_gates: List[str]) -> Tuple[GateStatus, GateAction]:
        """Determine overall status and final action"""
        if blocking_gates:
            return GateStatus.FAILED, GateAction.BLOCK
        
        # Check for warnings
        warning_count = sum(1 for gate in gate_evaluations if gate.status == GateStatus.WARNING)
        if warning_count > 0:
            return GateStatus.WARNING, GateAction.WARN
        
        # Check for escalations
        escalation_count = sum(1 for gate in gate_evaluations 
                             if gate.enforcement_action == GateAction.ESCALATE)
        if escalation_count > 0:
            return GateStatus.PENDING, GateAction.ESCALATE
        
        # All gates passed
        return GateStatus.PASSED, GateAction.APPROVE
    
    def _generate_summary_message(self, overall_status: GateStatus, gates_passed: int,
                                gates_failed: int, gates_warned: int,
                                blocking_gates: List[str]) -> str:
        """Generate human-readable summary message"""
        total_gates = gates_passed + gates_failed + gates_warned
        
        if overall_status == GateStatus.PASSED:
            return f"✓ All {total_gates} quality gates passed successfully"
        elif overall_status == GateStatus.FAILED:
            return f"✗ {len(blocking_gates)} blocking gates failed: {', '.join(blocking_gates)}"
        elif overall_status == GateStatus.WARNING:
            return f"⚠ {gates_warned} gates have warnings, {gates_passed} passed"
        else:
            return f"⏳ Quality gates pending approval: {gates_passed} passed, {gates_failed} failed, {gates_warned} warned"
    
    def _generate_gate_recommendations(self, rule_evaluations: List[RuleEvaluation],
                                     gate_def: GateDefinition) -> List[str]:
        """Generate recommendations for gate improvement"""
        recommendations = []
        
        # Recommendations for failed rules
        failed_rules = [rule for rule in rule_evaluations if not rule.passed]
        for rule in failed_rules[:3]:  # Top 3 failed rules
            if rule.severity in ["critical", "high"]:
                recommendations.append(f"Urgent: Address {rule.rule_name} - currently {rule.metric_value}")
            else:
                recommendations.append(f"Improve {rule.rule_name} to meet threshold {rule.threshold_value}")
        
        # General recommendations based on gate type
        if gate_def.gate_type == GateType.COVERAGE_GATE:
            low_coverage_rules = [rule for rule in failed_rules if "coverage" in rule.rule_name.lower()]
            if low_coverage_rules:
                recommendations.append("Add tests to increase coverage in critical areas")
        
        elif gate_def.gate_type == GateType.SECURITY_GATE:
            security_failures = [rule for rule in failed_rules if rule.severity == "critical"]
            if security_failures:
                recommendations.append("Security review required before deployment")
        
        elif gate_def.gate_type == GateType.PERFORMANCE_GATE:
            performance_failures = [rule for rule in failed_rules if "time" in rule.rule_name.lower()]
            if performance_failures:
                recommendations.append("Optimize test execution performance")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _handle_notifications(self, result: QualityGateResult):
        """Handle notifications for gate results"""
        # This would integrate with actual notification systems
        for gate_eval in result.gate_evaluations:
            gate_def = self.gate_definitions.get(gate_eval.gate_id)
            if not gate_def:
                continue
            
            if gate_eval.status == GateStatus.FAILED:
                self._send_notification(gate_def, gate_eval, "failure")
            elif gate_eval.status == GateStatus.WARNING:
                self._send_notification(gate_def, gate_eval, "warning")
    
    def _send_notification(self, gate_def: GateDefinition, gate_eval: GateEvaluation, 
                          notification_type: str):
        """Send notification (placeholder for actual implementation)"""
        self.logger.info(f"Notification: {gate_def.name} {notification_type}")
        
        # Would integrate with email, Slack, Teams, etc.
        notification_data = {
            "gate_name": gate_def.name,
            "status": gate_eval.status.value,
            "score": gate_eval.overall_score,
            "issues": gate_eval.blocking_issues + gate_eval.warnings,
            "recommendations": gate_eval.recommendations
        }
        
        # Send to configured notification handlers
        for handler_name in gate_def.notification_settings.get("channels", []):
            if handler_name in self.notification_handlers:
                self.notification_handlers[handler_name](notification_data)
    
    def _handle_escalations(self, result: QualityGateResult):
        """Handle escalation workflows"""
        for gate_eval in result.gate_evaluations:
            if gate_eval.enforcement_action == GateAction.ESCALATE:
                gate_def = self.gate_definitions.get(gate_eval.gate_id)
                if gate_def:
                    self._escalate_gate_failure(gate_def, gate_eval)
    
    def _escalate_gate_failure(self, gate_def: GateDefinition, gate_eval: GateEvaluation):
        """Escalate gate failure to appropriate team"""
        escalation_target = gate_def.escalation_rules.get("escalate_to", "default")
        
        escalation_data = {
            "gate_name": gate_def.name,
            "gate_id": gate_def.gate_id,
            "failure_details": gate_eval.blocking_issues,
            "recommendations": gate_eval.recommendations,
            "severity": "high" if gate_eval.enforcement_action == GateAction.BLOCK else "medium"
        }
        
        # Send to escalation handler
        if escalation_target in self.escalation_handlers:
            self.escalation_handlers[escalation_target](escalation_data)
        
        self.logger.warning(f"Escalated {gate_def.name} failure to {escalation_target}")
    
    def add_custom_gate(self, gate_definition: GateDefinition):
        """Add custom quality gate"""
        self.gate_definitions[gate_definition.gate_id] = gate_definition
        self.logger.info(f"Added custom gate: {gate_definition.name}")
    
    def add_custom_validator(self, validator_name: str, validator_func: Callable):
        """Add custom validation function"""
        self.custom_validators[validator_name] = validator_func
        self.logger.info(f"Added custom validator: {validator_name}")
    
    def add_notification_handler(self, handler_name: str, handler_func: Callable):
        """Add notification handler"""
        self.notification_handlers[handler_name] = handler_func
        self.logger.info(f"Added notification handler: {handler_name}")
    
    def add_escalation_handler(self, handler_name: str, handler_func: Callable):
        """Add escalation handler"""
        self.escalation_handlers[handler_name] = handler_func
        self.logger.info(f"Added escalation handler: {handler_name}")
    
    def get_gate_statistics(self) -> Dict[str, Any]:
        """Get quality gate statistics"""
        if not self.evaluation_history:
            return {"status": "No evaluations performed"}
        
        recent_evaluations = self.evaluation_history[-50:]  # Last 50 evaluations
        
        total_evaluations = len(recent_evaluations)
        passed_count = sum(1 for eval in recent_evaluations if eval.overall_status == GateStatus.PASSED)
        failed_count = sum(1 for eval in recent_evaluations if eval.overall_status == GateStatus.FAILED)
        warning_count = sum(1 for eval in recent_evaluations if eval.overall_status == GateStatus.WARNING)
        
        # Gate-specific statistics
        gate_stats = {}
        for gate_id in self.gate_definitions.keys():
            gate_evaluations = []
            for eval in recent_evaluations:
                gate_eval = next((g for g in eval.gate_evaluations if g.gate_id == gate_id), None)
                if gate_eval:
                    gate_evaluations.append(gate_eval)
            
            if gate_evaluations:
                avg_score = sum(g.overall_score for g in gate_evaluations) / len(gate_evaluations)
                pass_rate = sum(1 for g in gate_evaluations if g.status == GateStatus.PASSED) / len(gate_evaluations)
                
                gate_stats[gate_id] = {
                    "average_score": avg_score,
                    "pass_rate": pass_rate,
                    "total_evaluations": len(gate_evaluations)
                }
        
        return {
            "total_evaluations": total_evaluations,
            "overall_pass_rate": passed_count / total_evaluations,
            "failure_rate": failed_count / total_evaluations,
            "warning_rate": warning_count / total_evaluations,
            "gate_statistics": gate_stats,
            "most_recent_evaluation": asdict(self.evaluation_history[-1]) if self.evaluation_history else None
        }
    
    def export_gate_configuration(self) -> str:
        """Export gate configuration as JSON"""
        config_data = {
            "gate_definitions": {gid: asdict(gate) for gid, gate in self.gate_definitions.items()},
            "custom_validators": list(self.custom_validators.keys()),
            "notification_handlers": list(self.notification_handlers.keys()),
            "escalation_handlers": list(self.escalation_handlers.keys())
        }
        return json.dumps(config_data, indent=2, default=str)
    
    def import_gate_configuration(self, config_json: str):
        """Import gate configuration from JSON"""
        try:
            config_data = json.loads(config_json)
            
            # Import gate definitions
            for gate_id, gate_data in config_data.get("gate_definitions", {}).items():
                # Convert back to dataclass (simplified)
                rules = [QualityRule(**rule_data) for rule_data in gate_data["rules"]]
                gate_data["rules"] = rules
                gate_data["gate_type"] = GateType(gate_data["gate_type"])
                gate_data["enforcement_action"] = GateAction(gate_data["enforcement_action"])
                
                gate_def = GateDefinition(**gate_data)
                self.gate_definitions[gate_id] = gate_def
            
            self.logger.info("Gate configuration imported successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to import gate configuration: {e}")
            raise
    
    def simulate_gate_evaluation(self, test_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Simulate gate evaluation for testing purposes"""
        result = self.evaluate_quality_gates(test_metrics)
        
        return {
            "simulation_result": asdict(result),
            "would_block_deployment": result.final_action == GateAction.BLOCK,
            "requires_approval": result.approval_required,
            "blocking_gates": result.blocking_gates,
            "summary": result.summary_message
        }