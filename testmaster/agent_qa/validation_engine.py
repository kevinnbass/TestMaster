"""
Validation Engine for TestMaster Agent QA

Validates agent outputs against rules and expectations.
"""

import threading
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import re

from ..core.feature_flags import FeatureFlags

class ValidationType(Enum):
    """Types of validation."""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    FORMAT = "format"
    CONTENT = "content"
    STRUCTURE = "structure"
    PERFORMANCE = "performance"

@dataclass
class ValidationRule:
    """Validation rule definition."""
    name: str
    type: ValidationType
    description: str
    validator: Callable[[Any], bool]
    error_message: str
    severity: str = "error"  # error, warning, info

@dataclass
class ValidationIssue:
    """Validation issue details."""
    rule_name: str
    severity: str
    message: str
    location: str = ""
    suggestion: str = ""

@dataclass
class ValidationResult:
    """Validation result."""
    agent_id: str
    passed: bool
    score: float
    issues: List[ValidationIssue]
    total_checks: int = 0
    passed_checks: int = 0
    
    def add_issue(self, issue: ValidationIssue):
        """Add validation issue."""
        self.issues.append(issue)

class ValidationEngine:
    """Validation engine for agent outputs."""
    
    def __init__(self):
        self.enabled = FeatureFlags.is_enabled('layer1_test_foundation', 'agent_qa')
        self.lock = threading.RLock()
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.validation_history: Dict[str, List[ValidationResult]] = {}
        
        if not self.enabled:
            return
        
        # Initialize default validation rules
        self._setup_default_rules()
        
        print("Validation engine initialized")
        print(f"   Default rule categories: {list(self.validation_rules.keys())}")
    
    def _setup_default_rules(self):
        """Setup default validation rules."""
        # Syntax validation rules
        syntax_rules = [
            ValidationRule(
                name="no_syntax_errors",
                type=ValidationType.SYNTAX,
                description="Check for syntax errors",
                validator=lambda x: self._check_syntax(x),
                error_message="Syntax errors detected"
            ),
            ValidationRule(
                name="proper_indentation",
                type=ValidationType.SYNTAX,
                description="Check for proper indentation",
                validator=lambda x: self._check_indentation(x),
                error_message="Improper indentation detected"
            )
        ]
        
        # Format validation rules
        format_rules = [
            ValidationRule(
                name="valid_json",
                type=ValidationType.FORMAT,
                description="Check if output is valid JSON when expected",
                validator=lambda x: self._check_json_format(x),
                error_message="Invalid JSON format"
            ),
            ValidationRule(
                name="consistent_naming",
                type=ValidationType.FORMAT,
                description="Check for consistent naming conventions",
                validator=lambda x: self._check_naming_consistency(x),
                error_message="Inconsistent naming conventions"
            )
        ]
        
        # Content validation rules
        content_rules = [
            ValidationRule(
                name="no_empty_output",
                type=ValidationType.CONTENT,
                description="Check that output is not empty",
                validator=lambda x: self._check_not_empty(x),
                error_message="Output is empty"
            ),
            ValidationRule(
                name="relevant_content",
                type=ValidationType.CONTENT,
                description="Check content relevance",
                validator=lambda x: self._check_content_relevance(x),
                error_message="Content appears irrelevant"
            )
        ]
        
        # Performance validation rules
        performance_rules = [
            ValidationRule(
                name="response_time",
                type=ValidationType.PERFORMANCE,
                description="Check response time",
                validator=lambda x: self._check_response_time(x),
                error_message="Response time exceeds threshold",
                severity="warning"
            )
        ]
        
        self.validation_rules = {
            "syntax": syntax_rules,
            "format": format_rules,
            "content": content_rules,
            "performance": performance_rules
        }
    
    def validate_output(
        self,
        agent_id: str,
        output: Any,
        expected: Any = None,
        validation_rules: List[ValidationRule] = None
    ) -> ValidationResult:
        """
        Validate agent output against rules.
        
        Args:
            agent_id: Agent identifier
            output: Output to validate
            expected: Expected output for comparison
            validation_rules: Custom validation rules
            
        Returns:
            Validation result with issues and score
        """
        if not self.enabled:
            return ValidationResult(agent_id, False, 0.0, [])
        
        result = ValidationResult(agent_id, True, 1.0, [])
        
        # Use custom rules if provided, otherwise use default rules
        rules_to_check = validation_rules or []
        if not rules_to_check:
            # Use all default rules
            for rule_category in self.validation_rules.values():
                rules_to_check.extend(rule_category)
        
        result.total_checks = len(rules_to_check)
        
        # Run validation rules
        for rule in rules_to_check:
            try:
                if rule.validator(output):
                    result.passed_checks += 1
                else:
                    issue = ValidationIssue(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=rule.error_message,
                        suggestion=self._get_suggestion(rule.name)
                    )
                    result.add_issue(issue)
            except Exception as e:
                # Rule execution failed
                issue = ValidationIssue(
                    rule_name=rule.name,
                    severity="error",
                    message=f"Validation rule failed: {str(e)}",
                    suggestion="Check rule implementation"
                )
                result.add_issue(issue)
        
        # Compare with expected output if provided
        if expected is not None:
            similarity_score = self._calculate_similarity(output, expected)
            if similarity_score < 0.7:  # Threshold for similarity
                issue = ValidationIssue(
                    rule_name="output_similarity",
                    severity="warning",
                    message=f"Output similarity too low: {similarity_score:.2f}",
                    suggestion="Review output against expectations"
                )
                result.add_issue(issue)
        
        # Calculate final score
        result.score = self._calculate_validation_score(result)
        result.passed = result.score >= 0.7  # Pass threshold
        
        # Store in history
        with self.lock:
            if agent_id not in self.validation_history:
                self.validation_history[agent_id] = []
            self.validation_history[agent_id].append(result)
        
        print(f"Validation completed for {agent_id}: {result.score:.2f} ({result.passed_checks}/{result.total_checks} checks passed)")
        
        return result
    
    def _check_syntax(self, output: Any) -> bool:
        """Check for syntax errors."""
        if isinstance(output, str):
            # Basic Python syntax check
            try:
                compile(output, '<string>', 'exec')
                return True
            except SyntaxError:
                return False
        return True  # Non-string outputs pass syntax check
    
    def _check_indentation(self, output: Any) -> bool:
        """Check for proper indentation."""
        if isinstance(output, str):
            lines = output.split('\n')
            indent_levels = []
            for line in lines:
                if line.strip():  # Skip empty lines
                    indent = len(line) - len(line.lstrip())
                    indent_levels.append(indent)
            
            # Check for consistent indentation (multiples of 4 or 2)
            if indent_levels:
                base_indent = min(i for i in indent_levels if i > 0) if any(i > 0 for i in indent_levels) else 4
                return all(indent % base_indent == 0 for indent in indent_levels)
        return True
    
    def _check_json_format(self, output: Any) -> bool:
        """Check if output is valid JSON when it should be."""
        if isinstance(output, str) and (output.strip().startswith('{') or output.strip().startswith('[')):
            try:
                import json
                json.loads(output)
                return True
            except json.JSONDecodeError:
                return False
        return True  # Non-JSON-like outputs pass
    
    def _check_naming_consistency(self, output: Any) -> bool:
        """Check for consistent naming conventions."""
        if isinstance(output, str):
            # Check for consistent variable naming (snake_case vs camelCase)
            snake_case_pattern = re.compile(r'[a-z_][a-z0-9_]*')
            camel_case_pattern = re.compile(r'[a-z][a-zA-Z0-9]*')
            
            words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', output)
            if words:
                snake_case_count = sum(1 for word in words if snake_case_pattern.fullmatch(word))
                camel_case_count = sum(1 for word in words if camel_case_pattern.fullmatch(word))
                
                # Consistent if one style dominates (>70%)
                total = len(words)
                return (snake_case_count / total > 0.7) or (camel_case_count / total > 0.7)
        return True
    
    def _check_not_empty(self, output: Any) -> bool:
        """Check that output is not empty."""
        if output is None:
            return False
        if isinstance(output, str):
            return len(output.strip()) > 0
        if isinstance(output, (list, dict)):
            return len(output) > 0
        return True
    
    def _check_content_relevance(self, output: Any) -> bool:
        """Check content relevance (simplified heuristic)."""
        if isinstance(output, str):
            # Basic relevance check - contains meaningful content
            words = output.split()
            return len(words) >= 3  # At least 3 words
        return True
    
    def _check_response_time(self, output: Any) -> bool:
        """Check response time (mock implementation)."""
        # In real implementation, this would check actual response time
        return True  # Always pass for now
    
    def _calculate_similarity(self, output: Any, expected: Any) -> float:
        """Calculate similarity between output and expected."""
        if type(output) != type(expected):
            return 0.0
        
        if isinstance(output, str) and isinstance(expected, str):
            # Simple string similarity
            output_words = set(output.lower().split())
            expected_words = set(expected.lower().split())
            
            if not expected_words:
                return 1.0 if not output_words else 0.0
            
            intersection = output_words.intersection(expected_words)
            union = output_words.union(expected_words)
            return len(intersection) / len(union) if union else 1.0
        
        # For other types, simple equality check
        return 1.0 if output == expected else 0.0
    
    def _calculate_validation_score(self, result: ValidationResult) -> float:
        """Calculate overall validation score."""
        if result.total_checks == 0:
            return 1.0
        
        base_score = result.passed_checks / result.total_checks
        
        # Apply penalties for errors vs warnings
        error_penalty = sum(0.1 for issue in result.issues if issue.severity == "error")
        warning_penalty = sum(0.05 for issue in result.issues if issue.severity == "warning")
        
        final_score = base_score - error_penalty - warning_penalty
        return max(0.0, min(1.0, final_score))
    
    def _get_suggestion(self, rule_name: str) -> str:
        """Get suggestion for fixing validation issue."""
        suggestions = {
            "no_syntax_errors": "Review code syntax and fix errors",
            "proper_indentation": "Use consistent indentation (4 spaces recommended)",
            "valid_json": "Ensure JSON syntax is correct",
            "consistent_naming": "Use consistent naming convention (snake_case or camelCase)",
            "no_empty_output": "Provide meaningful output content",
            "relevant_content": "Ensure content is relevant to the task",
            "response_time": "Optimize performance to reduce response time"
        }
        return suggestions.get(rule_name, "Review and fix the issue")
    
    def add_custom_rule(self, category: str, rule: ValidationRule):
        """Add custom validation rule."""
        if category not in self.validation_rules:
            self.validation_rules[category] = []
        self.validation_rules[category].append(rule)
        print(f"Added custom validation rule: {rule.name} to {category}")
    
    def get_validation_history(self, agent_id: str) -> List[ValidationResult]:
        """Get validation history for an agent."""
        with self.lock:
            return self.validation_history.get(agent_id, [])

def get_validation_engine() -> ValidationEngine:
    """Get validation engine instance."""
    return ValidationEngine()