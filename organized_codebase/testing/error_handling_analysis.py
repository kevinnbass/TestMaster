"""
Error Handling and Recovery Pattern Analysis
=============================================

Implements comprehensive error handling analysis:
- Retry logic pattern detection
- Circuit breaker implementation analysis
- Exception handling completeness
- Error propagation tracking
- Logging completeness assessment
- Input validation analysis
- Graceful degradation patterns
- Error recovery strategies
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter

from .base_analyzer import BaseAnalyzer


class ErrorHandlingAnalyzer(BaseAnalyzer):
    """Analyzer for error handling and recovery patterns."""
    
    def __init__(self, base_path: Path):
        super().__init__(base_path)
        
        # Retry patterns
        self.retry_patterns = {
            "exponential_backoff": "Exponential delay between retries",
            "linear_backoff": "Linear delay between retries",
            "immediate_retry": "No delay between retries",
            "jittered_backoff": "Randomized delay to prevent thundering herd"
        }
        
        # Circuit breaker states
        self.circuit_breaker_states = ["closed", "open", "half_open"]
        
        # Common exceptions to handle
        self.common_exceptions = {
            "network": ["ConnectionError", "TimeoutError", "RequestException"],
            "file": ["FileNotFoundError", "PermissionError", "IOError"],
            "data": ["ValueError", "KeyError", "IndexError", "AttributeError"],
            "system": ["MemoryError", "OSError", "SystemError"]
        }
        
        # Validation patterns
        self.validation_patterns = {
            "input_sanitization": ["strip", "escape", "validate"],
            "type_checking": ["isinstance", "type", "hasattr"],
            "boundary_checking": ["min", "max", "range", "bounds"],
            "format_validation": ["regex", "pattern", "format"]
        }
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive error handling analysis."""
        print("[INFO] Analyzing Error Handling Patterns...")
        
        results = {
            "retry_patterns": self._analyze_retry_patterns(),
            "circuit_breakers": self._analyze_circuit_breakers(),
            "exception_handling": self._analyze_exception_handling(),
            "error_propagation": self._analyze_error_propagation(),
            "logging_completeness": self._analyze_logging_completeness(),
            "validation_logic": self._analyze_validation_logic(),
            "graceful_degradation": self._analyze_graceful_degradation(),
            "recovery_strategies": self._analyze_recovery_strategies(),
            "error_metrics": self._calculate_error_handling_metrics()
        }
        
        print(f"  [OK] Analyzed {len(results)} error handling aspects")
        return results
    
    def _analyze_retry_patterns(self) -> Dict[str, Any]:
        """Analyze retry logic implementations."""
        retry_analysis = {
            "retry_implementations": [],
            "backoff_strategies": defaultdict(int),
            "retry_conditions": [],
            "max_retry_configs": [],
            "retry_best_practices": {}
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect retry decorators
                retry_decorators = self._detect_retry_decorators(tree, content)
                for decorator in retry_decorators:
                    decorator["file"] = file_key
                    retry_analysis["retry_implementations"].append(decorator)
                
                # Detect manual retry loops
                manual_retries = self._detect_manual_retry_loops(tree)
                for retry in manual_retries:
                    retry["file"] = file_key
                    retry_analysis["retry_implementations"].append(retry)
                
                # Analyze backoff strategies
                backoff = self._analyze_backoff_strategies(tree, content)
                for strategy, count in backoff.items():
                    retry_analysis["backoff_strategies"][strategy] += count
                
                # Detect retry conditions
                conditions = self._detect_retry_conditions(tree)
                retry_analysis["retry_conditions"].extend(conditions)
                
                # Extract max retry configurations
                max_configs = self._extract_max_retry_configs(tree, content)
                retry_analysis["max_retry_configs"].extend(max_configs)
                
            except:
                continue
        
        # Analyze retry best practices
        retry_analysis["retry_best_practices"] = self._check_retry_best_practices(retry_analysis)
        
        return retry_analysis
    
    def _detect_retry_decorators(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect retry decorator usage."""
        retry_decorators = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    decorator_name = ""
                    
                    if isinstance(decorator, ast.Name):
                        decorator_name = decorator.id
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            decorator_name = decorator.func.id
                        elif isinstance(decorator.func, ast.Attribute):
                            decorator_name = decorator.func.attr
                    
                    if "retry" in decorator_name.lower():
                        retry_info = {
                            "function": node.name,
                            "line": node.lineno,
                            "decorator": decorator_name,
                            "type": "decorator",
                            "config": self._extract_retry_config(decorator)
                        }
                        retry_decorators.append(retry_info)
        
        return retry_decorators
    
    def _detect_manual_retry_loops(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect manual retry loop implementations."""
        manual_retries = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For) or isinstance(node, ast.While):
                # Check for retry pattern indicators
                if self._is_retry_loop(node):
                    retry_info = {
                        "line": node.lineno,
                        "type": "manual_loop",
                        "loop_type": type(node).__name__,
                        "has_try_except": self._has_try_except(node),
                        "has_break": self._has_break(node),
                        "has_delay": self._has_sleep_call(node),
                        "risk": self._assess_retry_risk(node)
                    }
                    manual_retries.append(retry_info)
        
        return manual_retries
    
    def _is_retry_loop(self, loop_node: ast.AST) -> bool:
        """Check if a loop implements retry logic."""
        # Look for retry indicators
        has_try = False
        has_exception_handling = False
        has_counter = False
        
        for node in ast.walk(loop_node):
            if isinstance(node, ast.Try):
                has_try = True
                has_exception_handling = len(node.handlers) > 0
            
            # Check for retry counter
            if isinstance(node, ast.Name):
                if "retry" in node.id.lower() or "attempt" in node.id.lower():
                    has_counter = True
        
        return has_try and has_exception_handling
    
    def _analyze_circuit_breakers(self) -> Dict[str, Any]:
        """Analyze circuit breaker implementations."""
        circuit_analysis = {
            "circuit_breakers": [],
            "state_management": [],
            "threshold_configs": [],
            "timeout_configs": [],
            "fallback_mechanisms": []
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect circuit breaker patterns
                breakers = self._detect_circuit_breakers(tree, content)
                for breaker in breakers:
                    breaker["file"] = file_key
                    circuit_analysis["circuit_breakers"].append(breaker)
                
                # Analyze state management
                states = self._analyze_circuit_states(tree, content)
                circuit_analysis["state_management"].extend(states)
                
                # Extract threshold configurations
                thresholds = self._extract_threshold_configs(tree, content)
                circuit_analysis["threshold_configs"].extend(thresholds)
                
                # Detect fallback mechanisms
                fallbacks = self._detect_fallback_mechanisms(tree)
                for fallback in fallbacks:
                    fallback["file"] = file_key
                    circuit_analysis["fallback_mechanisms"].append(fallback)
                
            except:
                continue
        
        return circuit_analysis
    
    def _detect_circuit_breakers(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect circuit breaker implementations."""
        circuit_breakers = []
        
        # Look for circuit breaker libraries
        if "pybreaker" in content or "circuit_breaker" in content.lower():
            circuit_breakers.append({
                "type": "library",
                "library": "pybreaker" if "pybreaker" in content else "custom",
                "detected": True
            })
        
        # Look for custom implementations
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if "circuit" in node.name.lower() or "breaker" in node.name.lower():
                    # Analyze circuit breaker class
                    cb_info = {
                        "class": node.name,
                        "line": node.lineno,
                        "type": "custom",
                        "has_states": self._has_circuit_states(node),
                        "has_threshold": self._has_threshold_logic(node),
                        "has_timeout": self._has_timeout_logic(node)
                    }
                    circuit_breakers.append(cb_info)
        
        return circuit_breakers
    
    def _analyze_exception_handling(self) -> Dict[str, Any]:
        """Analyze exception handling completeness."""
        exception_analysis = {
            "try_except_blocks": [],
            "exception_types": defaultdict(int),
            "bare_excepts": [],
            "exception_chains": [],
            "coverage_score": 0
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Analyze try-except blocks
                for node in ast.walk(tree):
                    if isinstance(node, ast.Try):
                        try_info = {
                            "file": file_key,
                            "line": node.lineno,
                            "handlers": len(node.handlers),
                            "has_else": len(node.orelse) > 0,
                            "has_finally": len(node.finalbody) > 0,
                            "exception_types": []
                        }
                        
                        # Analyze exception handlers
                        for handler in node.handlers:
                            if handler.type:
                                if isinstance(handler.type, ast.Name):
                                    exc_type = handler.type.id
                                    try_info["exception_types"].append(exc_type)
                                    exception_analysis["exception_types"][exc_type] += 1
                                elif isinstance(handler.type, ast.Tuple):
                                    # Multiple exception types
                                    for exc in handler.type.elts:
                                        if isinstance(exc, ast.Name):
                                            try_info["exception_types"].append(exc.id)
                                            exception_analysis["exception_types"][exc.id] += 1
                            else:
                                # Bare except
                                exception_analysis["bare_excepts"].append({
                                    "file": file_key,
                                    "line": handler.lineno,
                                    "risk": "high",
                                    "recommendation": "Specify exception type"
                                })
                        
                        exception_analysis["try_except_blocks"].append(try_info)
                
                # Detect exception chains (raise from)
                chains = self._detect_exception_chains(tree)
                for chain in chains:
                    chain["file"] = file_key
                    exception_analysis["exception_chains"].append(chain)
                
            except:
                continue
        
        # Calculate exception handling coverage
        exception_analysis["coverage_score"] = self._calculate_exception_coverage(exception_analysis)
        
        return exception_analysis
    
    def _analyze_error_propagation(self) -> Dict[str, Any]:
        """Analyze how errors propagate through the system."""
        propagation_analysis = {
            "error_bubbling": [],
            "error_transformation": [],
            "error_suppression": [],
            "propagation_paths": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect error bubbling (re-raising)
                bubbling = self._detect_error_bubbling(tree)
                for bubble in bubbling:
                    bubble["file"] = file_key
                    propagation_analysis["error_bubbling"].append(bubble)
                
                # Detect error transformation
                transformations = self._detect_error_transformation(tree)
                for transform in transformations:
                    transform["file"] = file_key
                    propagation_analysis["error_transformation"].append(transform)
                
                # Detect error suppression
                suppressions = self._detect_error_suppression(tree)
                for suppression in suppressions:
                    suppression["file"] = file_key
                    propagation_analysis["error_suppression"].append(suppression)
                
            except:
                continue
        
        return propagation_analysis
    
    def _detect_error_bubbling(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect error re-raising patterns."""
        bubbling_patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise):
                # Check if it's a bare raise (re-raising)
                if node.exc is None:
                    bubbling_patterns.append({
                        "line": node.lineno,
                        "type": "bare_raise",
                        "pattern": "re-raising current exception"
                    })
                # Check for raise from (exception chaining)
                elif node.cause is not None:
                    bubbling_patterns.append({
                        "line": node.lineno,
                        "type": "raise_from",
                        "pattern": "exception chaining",
                        "preserves_context": True
                    })
        
        return bubbling_patterns
    
    def _detect_error_transformation(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect error transformation patterns."""
        transformations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    # Check if handler raises different exception
                    for inner in ast.walk(handler):
                        if isinstance(inner, ast.Raise):
                            if inner.exc and handler.type:
                                transformations.append({
                                    "line": handler.lineno,
                                    "from_exception": self._get_exception_name(handler.type),
                                    "to_exception": self._get_exception_name(inner.exc),
                                    "pattern": "exception_transformation"
                                })
        
        return transformations
    
    def _detect_error_suppression(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect error suppression patterns."""
        suppressions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    # Check if handler is empty or just passes
                    if len(handler.body) == 1:
                        if isinstance(handler.body[0], ast.Pass):
                            suppressions.append({
                                "line": handler.lineno,
                                "pattern": "silent_suppression",
                                "risk": "high",
                                "recommendation": "Log the error at minimum"
                            })
                        elif isinstance(handler.body[0], ast.Expr):
                            if isinstance(handler.body[0].value, ast.Constant):
                                if handler.body[0].value.value is None:
                                    suppressions.append({
                                        "line": handler.lineno,
                                        "pattern": "suppression_with_none",
                                        "risk": "medium"
                                    })
        
        return suppressions
    
    def _analyze_logging_completeness(self) -> Dict[str, Any]:
        """Analyze logging completeness in error handling."""
        logging_analysis = {
            "error_logging": [],
            "log_levels": defaultdict(int),
            "unlogged_errors": [],
            "structured_logging": [],
            "logging_coverage": 0
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect error logging
                error_logs = self._detect_error_logging(tree, content)
                for log in error_logs:
                    log["file"] = file_key
                    logging_analysis["error_logging"].append(log)
                
                # Analyze log levels
                levels = self._analyze_log_levels(tree, content)
                for level, count in levels.items():
                    logging_analysis["log_levels"][level] += count
                
                # Detect unlogged errors
                unlogged = self._detect_unlogged_errors(tree)
                for error in unlogged:
                    error["file"] = file_key
                    logging_analysis["unlogged_errors"].append(error)
                
                # Detect structured logging
                if self._uses_structured_logging(content):
                    logging_analysis["structured_logging"].append({
                        "file": file_key,
                        "detected": True
                    })
                
            except:
                continue
        
        # Calculate logging coverage
        logging_analysis["logging_coverage"] = self._calculate_logging_coverage(logging_analysis)
        
        return logging_analysis
    
    def _detect_error_logging(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect error logging patterns."""
        error_logs = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    has_logging = False
                    log_level = None
                    
                    for inner in ast.walk(handler):
                        if isinstance(inner, ast.Call):
                            if isinstance(inner.func, ast.Attribute):
                                if inner.func.attr in ["error", "exception", "critical", "warning"]:
                                    has_logging = True
                                    log_level = inner.func.attr
                                    
                                    error_logs.append({
                                        "line": inner.lineno,
                                        "level": log_level,
                                        "in_handler": True,
                                        "includes_traceback": log_level == "exception"
                                    })
                    
                    if not has_logging:
                        # Handler without logging
                        error_logs.append({
                            "line": handler.lineno,
                            "pattern": "missing_error_log",
                            "risk": "medium",
                            "recommendation": "Add error logging"
                        })
        
        return error_logs
    
    def _analyze_validation_logic(self) -> Dict[str, Any]:
        """Analyze input validation and sanitization."""
        validation_analysis = {
            "input_validation": [],
            "type_checking": [],
            "boundary_checking": [],
            "sanitization": [],
            "validation_coverage": {}
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect input validation
                validations = self._detect_input_validation(tree, content)
                for validation in validations:
                    validation["file"] = file_key
                    validation_analysis["input_validation"].append(validation)
                
                # Detect type checking
                type_checks = self._detect_type_checking(tree)
                for check in type_checks:
                    check["file"] = file_key
                    validation_analysis["type_checking"].append(check)
                
                # Detect boundary checking
                boundary_checks = self._detect_boundary_checking(tree)
                for check in boundary_checks:
                    check["file"] = file_key
                    validation_analysis["boundary_checking"].append(check)
                
                # Detect sanitization
                sanitization = self._detect_sanitization(tree, content)
                for san in sanitization:
                    san["file"] = file_key
                    validation_analysis["sanitization"].append(san)
                
            except:
                continue
        
        # Calculate validation coverage
        validation_analysis["validation_coverage"] = self._calculate_validation_coverage(validation_analysis)
        
        return validation_analysis
    
    def _detect_input_validation(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect input validation patterns."""
        validations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check function parameters
                for stmt in node.body:
                    if isinstance(stmt, ast.If):
                        # Check for validation patterns
                        if self._is_validation_check(stmt):
                            validations.append({
                                "function": node.name,
                                "line": stmt.lineno,
                                "type": "parameter_validation",
                                "has_error_handling": self._has_validation_error_handling(stmt)
                            })
                
                # Check for decorator-based validation
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        if "validate" in decorator.id.lower():
                            validations.append({
                                "function": node.name,
                                "line": node.lineno,
                                "type": "decorator_validation",
                                "decorator": decorator.id
                            })
        
        return validations
    
    def _is_validation_check(self, if_node: ast.If) -> bool:
        """Check if an if statement is a validation check."""
        # Look for common validation patterns
        for node in ast.walk(if_node.test):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["isinstance", "hasattr", "callable"]:
                        return True
            elif isinstance(node, ast.Compare):
                # Boundary checks
                for op in node.ops:
                    if isinstance(op, (ast.Lt, ast.Gt, ast.LtE, ast.GtE)):
                        return True
        return False
    
    def _analyze_graceful_degradation(self) -> Dict[str, Any]:
        """Analyze graceful degradation patterns."""
        degradation_analysis = {
            "fallback_values": [],
            "default_behaviors": [],
            "feature_flags": [],
            "degradation_strategies": []
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect fallback values
                fallbacks = self._detect_fallback_values(tree)
                for fallback in fallbacks:
                    fallback["file"] = file_key
                    degradation_analysis["fallback_values"].append(fallback)
                
                # Detect default behaviors
                defaults = self._detect_default_behaviors(tree)
                for default in defaults:
                    default["file"] = file_key
                    degradation_analysis["default_behaviors"].append(default)
                
                # Detect feature flags
                if self._has_feature_flags(content):
                    degradation_analysis["feature_flags"].append({
                        "file": file_key,
                        "detected": True
                    })
                
            except:
                continue
        
        return degradation_analysis
    
    def _detect_fallback_values(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect fallback value patterns."""
        fallbacks = []
        
        for node in ast.walk(tree):
            # Try-except with fallback value
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    for stmt in handler.body:
                        if isinstance(stmt, ast.Assign):
                            fallbacks.append({
                                "line": stmt.lineno,
                                "pattern": "exception_fallback",
                                "type": "value_assignment"
                            })
                        elif isinstance(stmt, ast.Return):
                            if isinstance(stmt.value, ast.Constant):
                                fallbacks.append({
                                    "line": stmt.lineno,
                                    "pattern": "exception_return_default",
                                    "default_value": stmt.value.value
                                })
            
            # Or operator for fallback
            elif isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.Or):
                    fallbacks.append({
                        "line": node.lineno,
                        "pattern": "or_fallback",
                        "type": "short_circuit"
                    })
        
        return fallbacks
    
    def _analyze_recovery_strategies(self) -> Dict[str, Any]:
        """Analyze error recovery strategies."""
        recovery_analysis = {
            "recovery_patterns": [],
            "compensation_logic": [],
            "rollback_mechanisms": [],
            "recovery_effectiveness": {}
        }
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                content = self._get_file_content(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Detect recovery patterns
                patterns = self._detect_recovery_patterns(tree, content)
                for pattern in patterns:
                    pattern["file"] = file_key
                    recovery_analysis["recovery_patterns"].append(pattern)
                
                # Detect compensation logic
                compensations = self._detect_compensation_logic(tree)
                for comp in compensations:
                    comp["file"] = file_key
                    recovery_analysis["compensation_logic"].append(comp)
                
                # Detect rollback mechanisms
                rollbacks = self._detect_rollback_mechanisms(tree, content)
                for rollback in rollbacks:
                    rollback["file"] = file_key
                    recovery_analysis["rollback_mechanisms"].append(rollback)
                
            except:
                continue
        
        # Assess recovery effectiveness
        recovery_analysis["recovery_effectiveness"] = self._assess_recovery_effectiveness(recovery_analysis)
        
        return recovery_analysis
    
    def _detect_recovery_patterns(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect error recovery patterns."""
        patterns = []
        
        # Saga pattern
        if "saga" in content.lower() or "compensate" in content.lower():
            patterns.append({
                "pattern": "saga",
                "type": "distributed_transaction"
            })
        
        # Checkpoint/restart pattern
        if "checkpoint" in content.lower() or "savepoint" in content.lower():
            patterns.append({
                "pattern": "checkpoint_restart",
                "type": "state_recovery"
            })
        
        # Retry with different strategy
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                # Multiple except blocks might indicate different strategies
                if len(node.handlers) > 2:
                    patterns.append({
                        "line": node.lineno,
                        "pattern": "multi_strategy_recovery",
                        "strategies": len(node.handlers)
                    })
        
        return patterns
    
    def _calculate_error_handling_metrics(self) -> Dict[str, Any]:
        """Calculate overall error handling metrics."""
        retry_analysis = self._analyze_retry_patterns()
        exception_analysis = self._analyze_exception_handling()
        logging_analysis = self._analyze_logging_completeness()
        validation_analysis = self._analyze_validation_logic()
        
        metrics = {
            "retry_implementations": len(retry_analysis["retry_implementations"]),
            "bare_except_count": len(exception_analysis["bare_excepts"]),
            "unlogged_errors": len(logging_analysis["unlogged_errors"]),
            "validation_coverage": validation_analysis["validation_coverage"].get("score", 0),
            "error_handling_score": 0,
            "critical_issues": [],
            "recommendations": []
        }
        
        # Calculate overall score
        score = 100
        score -= metrics["bare_except_count"] * 5
        score -= metrics["unlogged_errors"] * 3
        score += metrics["retry_implementations"] * 2
        score += metrics["validation_coverage"]
        metrics["error_handling_score"] = max(0, min(100, score))
        
        # Identify critical issues
        if metrics["bare_except_count"] > 5:
            metrics["critical_issues"].append("Too many bare except clauses")
        
        if metrics["unlogged_errors"] > 10:
            metrics["critical_issues"].append("Many errors are not being logged")
        
        if metrics["validation_coverage"] < 50:
            metrics["critical_issues"].append("Insufficient input validation")
        
        # Generate recommendations
        if metrics["retry_implementations"] == 0:
            metrics["recommendations"].append("Implement retry logic for transient failures")
        
        if metrics["bare_except_count"] > 0:
            metrics["recommendations"].append("Replace bare excepts with specific exception types")
        
        if metrics["unlogged_errors"] > 0:
            metrics["recommendations"].append("Add logging to all error handlers")
        
        return metrics
    
    # Helper methods
    def _extract_retry_config(self, decorator: ast.AST) -> Dict[str, Any]:
        """Extract retry configuration from decorator."""
        config = {}
        
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                if keyword.arg == "max_retries":
                    if isinstance(keyword.value, ast.Constant):
                        config["max_retries"] = keyword.value.value
                elif keyword.arg == "backoff":
                    if isinstance(keyword.value, ast.Constant):
                        config["backoff"] = keyword.value.value
                elif keyword.arg == "delay":
                    if isinstance(keyword.value, ast.Constant):
                        config["delay"] = keyword.value.value
        
        return config
    
    def _has_try_except(self, node: ast.AST) -> bool:
        """Check if node contains try-except."""
        for inner in ast.walk(node):
            if isinstance(inner, ast.Try):
                return True
        return False
    
    def _has_break(self, node: ast.AST) -> bool:
        """Check if node contains break statement."""
        for inner in ast.walk(node):
            if isinstance(inner, ast.Break):
                return True
        return False
    
    def _has_sleep_call(self, node: ast.AST) -> bool:
        """Check if node contains sleep call."""
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                if isinstance(inner.func, ast.Attribute):
                    if inner.func.attr == "sleep":
                        return True
                elif isinstance(inner.func, ast.Name):
                    if inner.func.id == "sleep":
                        return True
        return False
    
    def _assess_retry_risk(self, node: ast.AST) -> str:
        """Assess risk level of retry implementation."""
        if not self._has_break(node):
            return "high"  # Infinite retry risk
        if not self._has_sleep_call(node):
            return "medium"  # No backoff
        return "low"
    
    def _analyze_backoff_strategies(self, tree: ast.AST, content: str) -> Dict[str, int]:
        """Analyze backoff strategies used."""
        strategies = defaultdict(int)
        
        # Exponential backoff
        if "exponential" in content.lower() or "2 **" in content or "2**" in content:
            strategies["exponential"] += 1
        
        # Linear backoff
        if "linear" in content.lower():
            strategies["linear"] += 1
        
        # Jittered backoff
        if "jitter" in content.lower() or "random" in content.lower():
            strategies["jittered"] += 1
        
        return strategies
    
    def _detect_retry_conditions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect conditions that trigger retries."""
        conditions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if handler.type:
                        exc_name = self._get_exception_name(handler.type)
                        conditions.append({
                            "exception": exc_name,
                            "line": handler.lineno,
                            "retryable": exc_name in ["TimeoutError", "ConnectionError", "RequestException"]
                        })
        
        return conditions
    
    def _extract_max_retry_configs(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Extract max retry configurations."""
        configs = []
        
        # Look for max_retries assignments
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if "retry" in target.id.lower() and "max" in target.id.lower():
                            if isinstance(node.value, ast.Constant):
                                configs.append({
                                    "variable": target.id,
                                    "value": node.value.value,
                                    "line": node.lineno
                                })
        
        return configs
    
    def _check_retry_best_practices(self, retry_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Check retry best practices compliance."""
        practices = {
            "has_max_retries": len(retry_analysis["max_retry_configs"]) > 0,
            "uses_backoff": len(retry_analysis["backoff_strategies"]) > 0,
            "handles_specific_exceptions": True,  # Simplified
            "score": 0
        }
        
        score = 0
        if practices["has_max_retries"]:
            score += 30
        if practices["uses_backoff"]:
            score += 40
        if practices["handles_specific_exceptions"]:
            score += 30
        
        practices["score"] = score
        
        return practices
    
    def _has_circuit_states(self, class_node: ast.ClassDef) -> bool:
        """Check if class has circuit breaker states."""
        for node in class_node.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if "state" in target.id.lower():
                            return True
        return False
    
    def _has_threshold_logic(self, class_node: ast.ClassDef) -> bool:
        """Check if class has threshold logic."""
        for node in ast.walk(class_node):
            if isinstance(node, ast.Name):
                if "threshold" in node.id.lower() or "failure" in node.id.lower():
                    return True
        return False
    
    def _has_timeout_logic(self, class_node: ast.ClassDef) -> bool:
        """Check if class has timeout logic."""
        for node in ast.walk(class_node):
            if isinstance(node, ast.Name):
                if "timeout" in node.id.lower():
                    return True
        return False
    
    def _analyze_circuit_states(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Analyze circuit breaker state management."""
        states = []
        
        # Look for state transitions
        if "open" in content.lower() and "closed" in content.lower():
            states.append({
                "pattern": "state_machine",
                "states_detected": ["open", "closed"]
            })
        
        if "half_open" in content.lower() or "half-open" in content.lower():
            states.append({
                "pattern": "three_state_circuit",
                "states_detected": ["closed", "open", "half_open"]
            })
        
        return states
    
    def _extract_threshold_configs(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Extract threshold configurations."""
        configs = []
        
        # Look for failure threshold
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if "threshold" in target.id.lower():
                            if isinstance(node.value, ast.Constant):
                                configs.append({
                                    "parameter": target.id,
                                    "value": node.value.value,
                                    "line": node.lineno
                                })
        
        return configs
    
    def _detect_fallback_mechanisms(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect fallback mechanisms."""
        fallbacks = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if "fallback" in node.name.lower():
                    fallbacks.append({
                        "function": node.name,
                        "line": node.lineno,
                        "type": "fallback_function"
                    })
        
        return fallbacks
    
    def _detect_exception_chains(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect exception chaining patterns."""
        chains = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise):
                if node.cause is not None:
                    chains.append({
                        "line": node.lineno,
                        "pattern": "exception_chaining",
                        "preserves_context": True
                    })
        
        return chains
    
    def _calculate_exception_coverage(self, exception_analysis: Dict[str, Any]) -> float:
        """Calculate exception handling coverage score."""
        total_blocks = len(exception_analysis["try_except_blocks"])
        bare_excepts = len(exception_analysis["bare_excepts"])
        
        if total_blocks == 0:
            return 0
        
        coverage = ((total_blocks - bare_excepts) / total_blocks) * 100
        return coverage
    
    def _get_exception_name(self, node: ast.AST) -> str:
        """Get exception name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
        return "Unknown"
    
    def _analyze_log_levels(self, tree: ast.AST, content: str) -> Dict[str, int]:
        """Analyze log levels used."""
        levels = defaultdict(int)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ["debug", "info", "warning", "error", "critical", "exception"]:
                        levels[node.func.attr] += 1
        
        return levels
    
    def _detect_unlogged_errors(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect errors that are not logged."""
        unlogged = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    has_logging = False
                    
                    for inner in ast.walk(handler):
                        if isinstance(inner, ast.Call):
                            if isinstance(inner.func, ast.Attribute):
                                if inner.func.attr in ["debug", "info", "warning", "error", "critical", "exception"]:
                                    has_logging = True
                                    break
                    
                    if not has_logging:
                        unlogged.append({
                            "line": handler.lineno,
                            "exception": self._get_exception_name(handler.type) if handler.type else "all",
                            "risk": "medium"
                        })
        
        return unlogged
    
    def _uses_structured_logging(self, content: str) -> bool:
        """Check if code uses structured logging."""
        return "structlog" in content or "json" in content and "logger" in content
    
    def _calculate_logging_coverage(self, logging_analysis: Dict[str, Any]) -> float:
        """Calculate logging coverage score."""
        total_errors = len(logging_analysis["error_logging"])
        unlogged = len(logging_analysis["unlogged_errors"])
        
        if total_errors + unlogged == 0:
            return 0
        
        coverage = (total_errors / (total_errors + unlogged)) * 100
        return coverage
    
    def _detect_type_checking(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect type checking patterns."""
        type_checks = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == "isinstance":
                        type_checks.append({
                            "line": node.lineno,
                            "function": "isinstance",
                            "pattern": "runtime_type_check"
                        })
                    elif node.func.id == "type":
                        type_checks.append({
                            "line": node.lineno,
                            "function": "type",
                            "pattern": "type_comparison"
                        })
        
        return type_checks
    
    def _detect_boundary_checking(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect boundary checking patterns."""
        boundary_checks = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                # Check for range comparisons
                for op in node.ops:
                    if isinstance(op, (ast.Lt, ast.Gt, ast.LtE, ast.GtE)):
                        boundary_checks.append({
                            "line": node.lineno,
                            "operator": type(op).__name__,
                            "pattern": "boundary_check"
                        })
        
        return boundary_checks
    
    def _detect_sanitization(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect input sanitization patterns."""
        sanitization = []
        
        # Look for sanitization functions
        sanitize_functions = ["strip", "escape", "sanitize", "clean", "validate"]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in sanitize_functions:
                        sanitization.append({
                            "line": node.lineno,
                            "method": node.func.attr,
                            "pattern": "method_sanitization"
                        })
                elif isinstance(node.func, ast.Name):
                    if any(san in node.func.id.lower() for san in sanitize_functions):
                        sanitization.append({
                            "line": node.lineno,
                            "function": node.func.id,
                            "pattern": "function_sanitization"
                        })
        
        return sanitization
    
    def _calculate_validation_coverage(self, validation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate validation coverage metrics."""
        total_validations = (
            len(validation_analysis["input_validation"]) +
            len(validation_analysis["type_checking"]) +
            len(validation_analysis["boundary_checking"])
        )
        
        score = min(100, total_validations * 5)  # Simple scoring
        
        return {
            "score": score,
            "total_validations": total_validations,
            "has_sanitization": len(validation_analysis["sanitization"]) > 0
        }
    
    def _has_validation_error_handling(self, if_node: ast.If) -> bool:
        """Check if validation has error handling."""
        # Check if validation raises or returns error
        for node in ast.walk(if_node):
            if isinstance(node, ast.Raise):
                return True
            elif isinstance(node, ast.Return):
                return True
        return False
    
    def _detect_default_behaviors(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect default behavior patterns."""
        defaults = []
        
        for node in ast.walk(tree):
            # Function default parameters
            if isinstance(node, ast.FunctionDef):
                if node.args.defaults:
                    defaults.append({
                        "function": node.name,
                        "line": node.lineno,
                        "pattern": "default_parameters",
                        "count": len(node.args.defaults)
                    })
            
            # Dictionary get with default
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == "get" and len(node.args) > 1:
                        defaults.append({
                            "line": node.lineno,
                            "pattern": "dict_get_default"
                        })
        
        return defaults
    
    def _has_feature_flags(self, content: str) -> bool:
        """Check if code uses feature flags."""
        return "feature_flag" in content.lower() or "feature_toggle" in content.lower()
    
    def _detect_compensation_logic(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Detect compensation logic patterns."""
        compensations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if "compensate" in node.name.lower() or "rollback" in node.name.lower():
                    compensations.append({
                        "function": node.name,
                        "line": node.lineno,
                        "pattern": "compensation_function"
                    })
        
        return compensations
    
    def _detect_rollback_mechanisms(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect rollback mechanisms."""
        rollbacks = []
        
        # Database rollback
        if "rollback" in content.lower():
            rollbacks.append({
                "pattern": "transaction_rollback",
                "detected": True
            })
        
        # State rollback
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if "rollback" in node.name.lower() or "undo" in node.name.lower():
                    rollbacks.append({
                        "function": node.name,
                        "line": node.lineno,
                        "pattern": "rollback_function"
                    })
        
        return rollbacks
    
    def _assess_recovery_effectiveness(self, recovery_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess effectiveness of recovery strategies."""
        effectiveness = {
            "score": 0,
            "has_retry": len(recovery_analysis["recovery_patterns"]) > 0,
            "has_compensation": len(recovery_analysis["compensation_logic"]) > 0,
            "has_rollback": len(recovery_analysis["rollback_mechanisms"]) > 0
        }
        
        score = 0
        if effectiveness["has_retry"]:
            score += 30
        if effectiveness["has_compensation"]:
            score += 35
        if effectiveness["has_rollback"]:
            score += 35
        
        effectiveness["score"] = score
        
        return effectiveness