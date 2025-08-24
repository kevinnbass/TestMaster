"""
        )
        
        # Store in history
        with self.lock:
            if agent_id not in self.inspection_history:
                self.inspection_history[agent_id] = []
            self.inspection_history[agent_id].append(report)
        
        inspection_time = time.time() - start_time
        self._log(f"Quality inspection completed for {agent_id}: {overall_score:.2f} ({status}) in {inspection_time*1000:.1f}ms")
        
        return report
    
    def _check_syntax_quality(self, agent_id: str, test_cases: List[Dict[str, Any]] = None) -> QualityMetric:
        """Check syntax quality of agent operations."""
        # Simulate syntax validation - in real implementation would analyze actual code
        syntax_score = 0.92  # Mock score
        threshold = self.quality_standards["syntax_score"]
        
        return QualityMetric(
            name="syntax_validation",
            value=syntax_score,
            threshold=threshold,
            status="pass" if syntax_score >= threshold else "fail",
            details={
                "test_cases_validated": len(test_cases) if test_cases else 0,
                "syntax_errors": 1 if syntax_score < threshold else 0,
                "code_style_issues": 2
            }
        )
    
    def _check_semantic_quality(self, agent_id: str, test_cases: List[Dict[str, Any]] = None) -> QualityMetric:
        """Check semantic quality of agent operations."""
        # Simulate semantic analysis
        semantic_score = 0.88
        threshold = 0.8
        
        return QualityMetric(
            name="semantic_analysis",
            value=semantic_score,
            threshold=threshold,
            status="pass" if semantic_score >= threshold else "fail",
            details={
                "logic_consistency": 0.9,
                "variable_usage": 0.85,
                "flow_analysis": 0.9
            }
        )
    
    def _check_performance_quality(self, agent_id: str) -> QualityMetric:
        """Check performance quality of agent operations."""
        # Simulate performance testing
        performance_score = 0.83
        threshold = self.quality_standards["performance_score"]
        
        return QualityMetric(
            name="performance_test",
            value=performance_score,
            threshold=threshold,
            status="pass" if performance_score >= threshold else "fail",
            details={
                "response_time_ms": 150.0,
                "memory_usage_mb": 45.2,
                "cpu_efficiency": 0.85
            }
        )
    
    def _check_security_quality(self, agent_id: str) -> QualityMetric:
        """Check security quality of agent operations."""
        # Simulate security scanning
        security_score = 0.96
        threshold = self.quality_standards["security_score"]
        
        return QualityMetric(
            name="security_scan",
            value=security_score,
            threshold=threshold,
            status="pass" if security_score >= threshold else "fail",
            details={
                "vulnerabilities_found": 0,
                "security_patterns": ["input_validation", "output_sanitization"],
                "compliance_score": 0.98
            }
        )
    
    def _check_reliability_quality(self, agent_id: str) -> QualityMetric:
        """Check reliability quality of agent operations."""
        # Simulate reliability testing
        reliability_score = 0.87
        threshold = self.quality_standards["reliability_score"]
        
        return QualityMetric(
            name="reliability_test",
            value=reliability_score,
            threshold=threshold,
            status="pass" if reliability_score >= threshold else "fail",
            details={
                "error_rate": 0.02,
                "uptime_percentage": 99.5,
                "fault_tolerance": 0.9
            }
        )
    
    def _calculate_inspection_score(self, metrics: List[QualityMetric]) -> float:
        """Calculate overall quality score from inspection metrics."""
        if not metrics:
            return 0.0
        
        # Weighted average of metric scores
        weights = {
            "syntax_validation": 0.2,
            "semantic_analysis": 0.25,
            "performance_test": 0.2,
            "security_scan": 0.25,
            "reliability_test": 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            weight = weights.get(metric.name, 0.1)
            total_score += metric.value * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_status(self, overall_score: float) -> str:
        """Determine quality status from overall score."""
        if overall_score >= 0.95:
            return QualityLevel.EXCELLENT.value
        elif overall_score >= 0.85:
            return QualityLevel.GOOD.value
        elif overall_score >= 0.7:
            return QualityLevel.SATISFACTORY.value
        elif overall_score >= 0.5:
            return QualityLevel.POOR.value
        else:
            return QualityLevel.CRITICAL.value
    
    def _generate_recommendations(self, metrics: List[QualityMetric]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        for metric in metrics:
            if metric.status == "fail":
                if metric.name == "syntax_validation":
                    recommendations.append("Fix syntax errors and improve code style")
                elif metric.name == "performance_test":
                    recommendations.append("Optimize performance bottlenecks")
                elif metric.name == "security_scan":
                    recommendations.append("Address security vulnerabilities")
                elif metric.name == "reliability_test":
                    recommendations.append("Improve error handling and fault tolerance")
        
        if not recommendations:
            recommendations.append("Maintain current quality standards")
        
        return recommendations
    
    # =========================================================================
    # Output Validation
    # =========================================================================
    
    def validate_output(
        self,
        agent_id: str,
        output: Any,
        expected: Any = None,
        validation_rules: List[ValidationRule] = None
    ) -> ValidationResult:
        """
        Validate agent output against rules and expectations.
        
        Args:
            agent_id: Agent identifier
            output: Output to validate
            expected: Expected output for comparison
            validation_rules: Custom validation rules
            
        Returns:
            Validation result with issues and score
        """
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
        
        self._log(f"Validation completed for {agent_id}: {result.score:.2f} ({result.passed_checks}/{result.total_checks} checks passed)")
        
        return result
    
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
    
    # =========================================================================
    # Validation Rule Implementations
    # =========================================================================
    
    def _check_syntax(self, output: Any) -> bool:
        """Check for syntax errors in code output."""
        if isinstance(output, str):
            # Basic Python syntax check
            try:
                compile(output, '<string>', 'exec')
                return True
            except SyntaxError:
                return False
        return True  # Non-string outputs pass syntax check
    
    def _check_indentation(self, output: Any) -> bool:
        """Check for proper indentation in code output."""
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
    
    # =========================================================================
    # Quality Scoring
    # =========================================================================
    
    def calculate_score(
        self,
        agent_id: str,
        quality_metrics: List[QualityMetric],
        custom_weights: Dict[str, float] = None
    ) -> QualityScore:
        """
        Calculate comprehensive quality score for an agent.
        
        Args:
            agent_id: Agent identifier
            quality_metrics: List of quality metrics
            custom_weights: Custom weights for categories
            
        Returns:
            Quality score with detailed breakdown
        """
        # Use custom weights if provided
        weights = self._apply_custom_weights(custom_weights) if custom_weights else self.score_weights
        
        # Calculate scores by category
        breakdown = []
        for weight in weights:
            category_score = self._calculate_category_score(quality_metrics, weight.category)
            weighted_score = category_score * weight.weight
            
            breakdown.append(ScoreBreakdown(
                category=weight.category,
                score=category_score,
                weight=weight.weight,
                weighted_score=weighted_score,
                details=self._get_category_details(quality_metrics, weight.category)
            ))
        
        # Calculate overall score
        overall_score = sum(b.weighted_score for b in breakdown)
        
        # Determine status
        status = self._determine_score_status(overall_score)
        
        # Calculate percentile
        percentile = self._calculate_percentile(agent_id, overall_score)
        
        score_result = QualityScore(
            agent_id=agent_id,
            overall_score=overall_score,
            breakdown=breakdown,
            status=status,
            percentile=percentile
        )
        
        # Store in history
        with self.lock:
            if agent_id not in self.scoring_history:
                self.scoring_history[agent_id] = []
            self.scoring_history[agent_id].append(score_result)
        
        self._log(f"Quality score calculated for {agent_id}: {overall_score:.3f} ({score_result.grade}) - {status}")
        
        return score_result
    
    def _calculate_category_score(self, metrics: List[QualityMetric], category: ScoreCategory) -> float:
        """Calculate score for a specific category."""
        category_metrics = self._filter_metrics_by_category(metrics, category)
        
        if not category_metrics:
            return 0.8  # Default score if no metrics available
        
        # Calculate weighted average of category metrics
        total_score = 0.0
        total_weight = 0.0
        
        for metric in category_metrics:
            weight = self._get_metric_weight(metric.name, category)
            total_score += metric.value * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _filter_metrics_by_category(self, metrics: List[QualityMetric], category: ScoreCategory) -> List[QualityMetric]:
        """Filter metrics by category."""
        category_mapping = {
            ScoreCategory.FUNCTIONALITY: ["syntax_validation", "semantic_analysis"],
            ScoreCategory.RELIABILITY: ["reliability_test", "error_handling"],
            ScoreCategory.PERFORMANCE: ["performance_test", "response_time"],
            ScoreCategory.SECURITY: ["security_scan", "vulnerability_check"],
            ScoreCategory.MAINTAINABILITY: ["code_quality", "documentation"],
            ScoreCategory.USABILITY: ["usability_test", "accessibility"]
        }
        
        relevant_names = category_mapping.get(category, [])
        return [m for m in metrics if m.name in relevant_names or category.value in m.name]
    
    def _get_metric_weight(self, metric_name: str, category: ScoreCategory) -> float:
        """Get weight for a specific metric within a category."""
        metric_weights = {
            "syntax_validation": 0.6,
            "semantic_analysis": 0.4,
            "performance_test": 0.7,
            "response_time": 0.3,
            "security_scan": 1.0,
            "reliability_test": 1.0
        }
        return metric_weights.get(metric_name, 1.0)
    
    def _get_category_details(self, metrics: List[QualityMetric], category: ScoreCategory) -> Dict[str, Any]:
        """Get detailed information for a category."""
        category_metrics = self._filter_metrics_by_category(metrics, category)
        
        return {
            "metrics_count": len(category_metrics),
            "metrics_passed": sum(1 for m in category_metrics if m.status == "pass"),
            "average_score": sum(m.value for m in category_metrics) / len(category_metrics) if category_metrics else 0.0,
            "metric_names": [m.name for m in category_metrics]
        }
    
    def _apply_custom_weights(self, custom_weights: Dict[str, float]) -> List[ScoreWeight]:
        """Apply custom weights to scoring categories."""
        updated_weights = []
        
        for weight in self.score_weights:
            new_weight_value = custom_weights.get(weight.category.value, weight.weight)
            updated_weights.append(ScoreWeight(
                category=weight.category,
                weight=new_weight_value,
                description=weight.description
            ))
        
        # Normalize weights to sum to 1.0
        total_weight = sum(w.weight for w in updated_weights)
        if total_weight > 0:
            for weight in updated_weights:
                weight.weight = weight.weight / total_weight
        
        return updated_weights
    
    def _determine_score_status(self, overall_score: float) -> str:
        """Determine status from overall score."""
        if overall_score >= 0.9:
            return "excellent"
        elif overall_score >= 0.8:
            return "good"
        elif overall_score >= 0.7:
