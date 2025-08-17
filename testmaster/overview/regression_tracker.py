"""
Regression Tracking & Predictive Failure Detection

Advanced regression analysis with pattern recognition and
predictive failure detection for proactive issue management.

Features:
- Historical failure pattern analysis
- Regression frequency tracking and trends
- Predictive failure detection using pattern recognition
- Root cause correlation analysis
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import pickle
from collections import defaultdict, Counter
import hashlib

from ..core.layer_manager import requires_layer


class RegressionType(Enum):
    """Types of regressions."""
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    COMPATIBILITY = "compatibility"
    INTEGRATION = "integration"
    SECURITY = "security"
    DATA_INTEGRITY = "data_integrity"
    UI_UX = "ui_ux"


class FailureCategory(Enum):
    """Categories of test failures."""
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    ASSERTION_FAILURE = "assertion_failure"
    EXCEPTION_RAISED = "exception_raised"
    TIMEOUT = "timeout"
    RESOURCE_ERROR = "resource_error"
    LOGIC_ERROR = "logic_error"
    ENVIRONMENT_ERROR = "environment_error"


class FailureSeverity(Enum):
    """Severity levels for failures."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PredictionConfidence(Enum):
    """Confidence levels for predictions."""
    VERY_LOW = "very_low"
    LOW = "low" 
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class FailureRecord:
    """Record of a test failure."""
    failure_id: str
    test_file: str
    test_function: str
    module_under_test: str
    
    # Failure details
    failure_category: FailureCategory
    error_message: str
    stack_trace: str = ""
    
    # Context
    failure_time: datetime = field(default_factory=datetime.now)
    git_commit: Optional[str] = None
    branch_name: Optional[str] = None
    environment: str = "unknown"
    
    # Classification
    regression_type: Optional[RegressionType] = None
    severity: FailureSeverity = FailureSeverity.MEDIUM
    
    # Resolution
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_method: Optional[str] = None
    resolution_notes: Optional[str] = None
    
    # Metadata
    affected_files: List[str] = field(default_factory=list)
    related_failures: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class RegressionPattern:
    """A pattern of recurring regressions."""
    pattern_id: str
    pattern_name: str
    description: str
    
    # Pattern characteristics
    failure_categories: List[FailureCategory] = field(default_factory=list)
    affected_modules: List[str] = field(default_factory=list)
    common_triggers: List[str] = field(default_factory=list)
    
    # Pattern statistics
    occurrence_count: int = 0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    frequency_score: float = 0.0
    
    # Pattern context
    typical_environment: Optional[str] = None
    seasonal_pattern: bool = False
    time_based_pattern: bool = False
    
    # Resolution patterns
    common_resolutions: List[str] = field(default_factory=list)
    average_resolution_time: float = 0.0
    success_rate: float = 0.0
    
    # Prediction data
    prediction_indicators: List[str] = field(default_factory=list)
    confidence_level: PredictionConfidence = PredictionConfidence.MEDIUM


@dataclass
class FailurePrediction:
    """Prediction of potential failure."""
    prediction_id: str
    target_module: str
    predicted_failure_type: FailureCategory
    
    # Prediction details
    confidence: PredictionConfidence
    probability_score: float
    time_window: str = "24_hours"
    
    # Evidence
    contributing_patterns: List[str] = field(default_factory=list)
    risk_indicators: List[str] = field(default_factory=list)
    historical_evidence: List[str] = field(default_factory=list)
    
    # Context
    predicted_at: datetime = field(default_factory=datetime.now)
    triggering_event: Optional[str] = None
    environmental_factors: List[str] = field(default_factory=list)
    
    # Preventive actions
    recommended_actions: List[str] = field(default_factory=list)
    monitoring_suggestions: List[str] = field(default_factory=list)
    
    # Outcome tracking
    prediction_outcome: Optional[bool] = None  # True if prediction was correct
    actual_failure: Optional[str] = None
    validation_time: Optional[datetime] = None


@dataclass
class RegressionSummary:
    """Summary of regression analysis."""
    total_failures: int = 0
    resolved_failures: int = 0
    active_patterns: int = 0
    high_risk_modules: List[str] = field(default_factory=list)
    
    # Trend analysis
    failure_trend: str = "stable"  # increasing, decreasing, stable
    resolution_trend: str = "stable"
    average_resolution_time: float = 0.0
    
    # Predictions
    active_predictions: List[FailurePrediction] = field(default_factory=list)
    prediction_accuracy: float = 0.0
    
    # Time period
    analysis_period_days: int = 30
    analysis_start: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=30))
    analysis_end: datetime = field(default_factory=datetime.now)


class RegressionTracker:
    """
    Regression tracking and predictive failure detection system.
    
    Analyzes historical failure patterns, tracks regression trends,
    and provides predictive failure detection capabilities.
    """
    
    @requires_layer("layer3_orchestration", "regression_tracking")
    def __init__(self, data_dir: str = ".testmaster_regressions"):
        """
        Initialize regression tracker.
        
        Args:
            data_dir: Directory for regression data storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "failures").mkdir(exist_ok=True)
        (self.data_dir / "patterns").mkdir(exist_ok=True)
        (self.data_dir / "predictions").mkdir(exist_ok=True)
        (self.data_dir / "models").mkdir(exist_ok=True)
        
        # Data storage
        self._failure_records: Dict[str, FailureRecord] = {}
        self._regression_patterns: Dict[str, RegressionPattern] = {}
        self._predictions: Dict[str, FailurePrediction] = {}
        
        # Pattern recognition
        self._pattern_extractors = []
        self._prediction_models = {}
        
        # Statistics
        self._stats = {
            'total_failures_tracked': 0,
            'patterns_discovered': 0,
            'predictions_made': 0,
            'prediction_accuracy': 0.0,
            'last_analysis': None
        }
        
        # Load existing data
        self._load_historical_data()
        
        # Setup pattern extractors
        self._setup_pattern_extractors()
        
        print("📈 Regression tracker initialized")
        print(f"   📁 Data directory: {self.data_dir}")
        print(f"   📊 Loaded {len(self._failure_records)} historical failures")
    
    def record_failure(self, test_file: str, test_function: str, error_message: str,
                      stack_trace: str = "", module_under_test: str = "",
                      git_commit: str = None, environment: str = "unknown") -> str:
        """
        Record a test failure.
        
        Args:
            test_file: Path to test file
            test_function: Name of failing test function
            error_message: Error message
            stack_trace: Stack trace if available
            module_under_test: Module being tested
            git_commit: Git commit hash
            environment: Test environment
            
        Returns:
            Failure ID
        """
        failure_id = self._generate_failure_id(test_file, test_function, error_message)
        
        # Classify failure
        failure_category = self._classify_failure(error_message, stack_trace)
        severity = self._assess_failure_severity(error_message, stack_trace)
        
        # Create failure record
        failure_record = FailureRecord(
            failure_id=failure_id,
            test_file=test_file,
            test_function=test_function,
            module_under_test=module_under_test or self._infer_module_under_test(test_file),
            failure_category=failure_category,
            error_message=error_message,
            stack_trace=stack_trace,
            git_commit=git_commit,
            environment=environment,
            severity=severity,
            affected_files=self._extract_affected_files(stack_trace)
        )
        
        # Classify regression type
        failure_record.regression_type = self._classify_regression_type(failure_record)
        
        # Store failure
        self._failure_records[failure_id] = failure_record
        self._stats['total_failures_tracked'] += 1
        
        # Persist failure
        self._persist_failure(failure_record)
        
        # Update patterns
        self._update_patterns(failure_record)
        
        # Generate predictions
        self._generate_predictions_for_failure(failure_record)
        
        print(f"📈 Recorded failure: {test_function} ({failure_category.value})")
        return failure_id
    
    def mark_failure_resolved(self, failure_id: str, resolution_method: str,
                            resolution_notes: str = "") -> bool:
        """
        Mark a failure as resolved.
        
        Args:
            failure_id: Failure ID
            resolution_method: How the failure was resolved
            resolution_notes: Additional notes
            
        Returns:
            True if marked successfully
        """
        if failure_id not in self._failure_records:
            return False
        
        failure = self._failure_records[failure_id]
        failure.resolved = True
        failure.resolution_time = datetime.now()
        failure.resolution_method = resolution_method
        failure.resolution_notes = resolution_notes
        
        # Update patterns with resolution data
        self._update_pattern_resolutions(failure)
        
        # Persist updated failure
        self._persist_failure(failure)
        
        print(f"✅ Marked failure resolved: {failure_id}")
        return True
    
    def analyze_regression_trends(self, days: int = 30) -> RegressionSummary:
        """
        Analyze regression trends over specified period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Regression analysis summary
        """
        print(f"📊 Analyzing regression trends ({days} days)...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Filter failures in time period
        period_failures = [
            failure for failure in self._failure_records.values()
            if start_time <= failure.failure_time <= end_time
        ]
        
        # Calculate metrics
        total_failures = len(period_failures)
        resolved_failures = len([f for f in period_failures if f.resolved])
        
        # Calculate trends
        failure_trend = self._calculate_failure_trend(period_failures)
        resolution_trend = self._calculate_resolution_trend(period_failures)
        
        # Calculate average resolution time
        resolved_with_time = [f for f in period_failures if f.resolved and f.resolution_time]
        avg_resolution_time = 0.0
        if resolved_with_time:
            resolution_times = [
                (f.resolution_time - f.failure_time).total_seconds() / 3600
                for f in resolved_with_time
            ]
            avg_resolution_time = sum(resolution_times) / len(resolution_times)
        
        # Identify high-risk modules
        high_risk_modules = self._identify_high_risk_modules(period_failures)
        
        # Get active predictions
        active_predictions = [p for p in self._predictions.values() 
                            if not p.prediction_outcome]
        
        # Calculate prediction accuracy
        prediction_accuracy = self._calculate_prediction_accuracy()
        
        summary = RegressionSummary(
            total_failures=total_failures,
            resolved_failures=resolved_failures,
            active_patterns=len(self._regression_patterns),
            high_risk_modules=high_risk_modules,
            failure_trend=failure_trend,
            resolution_trend=resolution_trend,
            average_resolution_time=avg_resolution_time,
            active_predictions=active_predictions,
            prediction_accuracy=prediction_accuracy,
            analysis_period_days=days,
            analysis_start=start_time,
            analysis_end=end_time
        )
        
        print(f"📊 Analysis complete: {total_failures} failures, {len(high_risk_modules)} high-risk modules")
        return summary
    
    def predict_potential_failures(self, target_modules: List[str] = None,
                                 time_window: str = "24_hours") -> List[FailurePrediction]:
        """
        Predict potential failures based on patterns.
        
        Args:
            target_modules: Specific modules to analyze (all if None)
            time_window: Prediction time window
            
        Returns:
            List of failure predictions
        """
        print("🔮 Generating failure predictions...")
        
        predictions = []
        
        # Get modules to analyze
        modules_to_analyze = target_modules or self._get_all_modules()
        
        for module in modules_to_analyze:
            # Get historical failures for module
            module_failures = [f for f in self._failure_records.values() 
                             if f.module_under_test == module]
            
            if not module_failures:
                continue
            
            # Check each pattern for prediction applicability
            for pattern in self._regression_patterns.values():
                if module in pattern.affected_modules:
                    prediction = self._generate_prediction_from_pattern(
                        module, pattern, time_window
                    )
                    
                    if prediction and prediction.confidence.value != "very_low":
                        predictions.append(prediction)
                        self._predictions[prediction.prediction_id] = prediction
                        self._stats['predictions_made'] += 1
        
        # Sort by probability score
        predictions.sort(key=lambda p: p.probability_score, reverse=True)
        
        print(f"🔮 Generated {len(predictions)} predictions")
        return predictions
    
    def validate_prediction(self, prediction_id: str, actual_outcome: bool,
                          actual_failure_id: str = None) -> bool:
        """
        Validate a prediction with actual outcome.
        
        Args:
            prediction_id: Prediction ID
            actual_outcome: True if prediction was correct
            actual_failure_id: ID of actual failure if occurred
            
        Returns:
            True if validation recorded
        """
        if prediction_id not in self._predictions:
            return False
        
        prediction = self._predictions[prediction_id]
        prediction.prediction_outcome = actual_outcome
        prediction.actual_failure = actual_failure_id
        prediction.validation_time = datetime.now()
        
        # Update prediction accuracy
        self._update_prediction_accuracy()
        
        # Persist updated prediction
        self._persist_prediction(prediction)
        
        print(f"✅ Validated prediction {prediction_id}: {'correct' if actual_outcome else 'incorrect'}")
        return True
    
    def _generate_failure_id(self, test_file: str, test_function: str, error_message: str) -> str:
        """Generate unique failure ID."""
        content = f"{test_file}:{test_function}:{error_message[:100]}"
        hash_obj = hashlib.md5(content.encode())
        timestamp = int(datetime.now().timestamp())
        return f"failure_{timestamp}_{hash_obj.hexdigest()[:8]}"
    
    def _classify_failure(self, error_message: str, stack_trace: str) -> FailureCategory:
        """Classify failure based on error message and stack trace."""
        error_lower = error_message.lower()
        trace_lower = stack_trace.lower()
        
        # Syntax errors
        if any(keyword in error_lower for keyword in ['syntaxerror', 'invalid syntax']):
            return FailureCategory.SYNTAX_ERROR
        
        # Import errors
        elif any(keyword in error_lower for keyword in ['importerror', 'modulenotfound', 'no module named']):
            return FailureCategory.IMPORT_ERROR
        
        # Assertion failures
        elif any(keyword in error_lower for keyword in ['assertionerror', 'assertion failed']):
            return FailureCategory.ASSERTION_FAILURE
        
        # Timeouts
        elif any(keyword in error_lower for keyword in ['timeout', 'time out', 'timed out']):
            return FailureCategory.TIMEOUT
        
        # Resource errors
        elif any(keyword in error_lower for keyword in ['memory', 'disk', 'permission', 'resource']):
            return FailureCategory.RESOURCE_ERROR
        
        # Environment errors
        elif any(keyword in error_lower for keyword in ['environment', 'config', 'setup']):
            return FailureCategory.ENVIRONMENT_ERROR
        
        # Exception raised
        elif any(keyword in error_lower for keyword in ['exception', 'error']):
            return FailureCategory.EXCEPTION_RAISED
        
        # Default to logic error
        else:
            return FailureCategory.LOGIC_ERROR
    
    def _assess_failure_severity(self, error_message: str, stack_trace: str) -> FailureSeverity:
        """Assess severity of failure."""
        error_lower = error_message.lower()
        
        # Critical keywords
        if any(keyword in error_lower for keyword in [
            'critical', 'fatal', 'segmentation', 'memory', 'security'
        ]):
            return FailureSeverity.CRITICAL
        
        # High severity keywords
        elif any(keyword in error_lower for keyword in [
            'data loss', 'corruption', 'authentication', 'authorization'
        ]):
            return FailureSeverity.HIGH
        
        # Low severity keywords
        elif any(keyword in error_lower for keyword in [
            'warning', 'deprecated', 'minor', 'formatting'
        ]):
            return FailureSeverity.LOW
        
        # Default to medium
        else:
            return FailureSeverity.MEDIUM
    
    def _infer_module_under_test(self, test_file: str) -> str:
        """Infer module under test from test file name."""
        test_path = Path(test_file)
        test_name = test_path.stem
        
        # Remove test prefixes/suffixes
        if test_name.startswith('test_'):
            return test_name[5:]
        elif test_name.endswith('_test'):
            return test_name[:-5]
        elif test_name.endswith('_tests'):
            return test_name[:-6]
        
        return test_name
    
    def _extract_affected_files(self, stack_trace: str) -> List[str]:
        """Extract affected files from stack trace."""
        files = []
        
        # Look for file paths in stack trace
        file_pattern = r'File "([^"]+)"'
        matches = re.findall(file_pattern, stack_trace)
        
        for match in matches:
            if match.endswith('.py') and not match.startswith('<'):
                files.append(match)
        
        return list(set(files))  # Remove duplicates
    
    def _classify_regression_type(self, failure: FailureRecord) -> RegressionType:
        """Classify the type of regression."""
        error_lower = failure.error_message.lower()
        module_lower = failure.module_under_test.lower()
        
        # Security regressions
        if any(keyword in error_lower or keyword in module_lower 
               for keyword in ['auth', 'security', 'permission', 'token']):
            return RegressionType.SECURITY
        
        # Performance regressions
        elif any(keyword in error_lower for keyword in ['timeout', 'slow', 'performance']):
            return RegressionType.PERFORMANCE
        
        # Integration regressions
        elif any(keyword in error_lower for keyword in ['connection', 'api', 'service', 'external']):
            return RegressionType.INTEGRATION
        
        # Data integrity regressions
        elif any(keyword in error_lower for keyword in ['data', 'database', 'corruption', 'integrity']):
            return RegressionType.DATA_INTEGRITY
        
        # Compatibility regressions
        elif any(keyword in error_lower for keyword in ['version', 'compatibility', 'deprecated']):
            return RegressionType.COMPATIBILITY
        
        # UI/UX regressions
        elif any(keyword in module_lower for keyword in ['ui', 'view', 'frontend', 'interface']):
            return RegressionType.UI_UX
        
        # Default to functional
        else:
            return RegressionType.FUNCTIONAL
    
    def _setup_pattern_extractors(self):
        """Setup pattern extraction functions."""
        
        def extract_error_message_patterns():
            """Extract patterns from error messages."""
            error_groups = defaultdict(list)
            
            for failure in self._failure_records.values():
                # Normalize error message
                normalized = re.sub(r'\d+', 'NUMBER', failure.error_message)
                normalized = re.sub(r"'[^']*'", 'STRING', normalized)
                normalized = re.sub(r'"[^"]*"', 'STRING', normalized)
                
                error_groups[normalized].append(failure)
            
            # Create patterns for recurring error messages
            for normalized_error, failures in error_groups.items():
                if len(failures) >= 3:  # At least 3 occurrences
                    self._create_pattern_from_failures(failures, f"Error: {normalized_error[:50]}")
        
        def extract_module_patterns():
            """Extract patterns by affected modules."""
            module_groups = defaultdict(list)
            
            for failure in self._failure_records.values():
                module_groups[failure.module_under_test].append(failure)
            
            # Create patterns for frequently failing modules
            for module, failures in module_groups.items():
                if len(failures) >= 5:  # At least 5 failures
                    self._create_pattern_from_failures(failures, f"Module: {module}")
        
        def extract_temporal_patterns():
            """Extract time-based patterns."""
            # Group by hour of day
            hour_groups = defaultdict(list)
            day_groups = defaultdict(list)
            
            for failure in self._failure_records.values():
                hour_groups[failure.failure_time.hour].append(failure)
                day_groups[failure.failure_time.weekday()].append(failure)
            
            # Find peak failure times
            for hour, failures in hour_groups.items():
                if len(failures) >= 10:
                    pattern = self._create_pattern_from_failures(
                        failures, f"Time Pattern: Hour {hour}"
                    )
                    if pattern:
                        pattern.time_based_pattern = True
        
        self._pattern_extractors = [
            extract_error_message_patterns,
            extract_module_patterns,
            extract_temporal_patterns
        ]
    
    def _update_patterns(self, failure: FailureRecord):
        """Update patterns with new failure."""
        # Run pattern extractors
        for extractor in self._pattern_extractors:
            try:
                extractor()
            except Exception as e:
                print(f"⚠️ Error in pattern extraction: {e}")
    
    def _create_pattern_from_failures(self, failures: List[FailureRecord], 
                                    pattern_name: str) -> Optional[RegressionPattern]:
        """Create a regression pattern from a group of failures."""
        if len(failures) < 2:
            return None
        
        pattern_id = f"pattern_{hashlib.md5(pattern_name.encode()).hexdigest()[:8]}"
        
        # Check if pattern already exists
        if pattern_id in self._regression_patterns:
            pattern = self._regression_patterns[pattern_id]
            pattern.occurrence_count = len(failures)
            pattern.last_seen = max(f.failure_time for f in failures)
            return pattern
        
        # Extract pattern characteristics
        failure_categories = list(set(f.failure_category for f in failures))
        affected_modules = list(set(f.module_under_test for f in failures))
        
        # Find common triggers (common words in error messages)
        all_words = []
        for failure in failures:
            words = re.findall(r'\w+', failure.error_message.lower())
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        common_triggers = [word for word, count in word_counts.most_common(5) 
                          if count >= len(failures) * 0.5]
        
        # Calculate frequency score
        time_span = (max(f.failure_time for f in failures) - 
                    min(f.failure_time for f in failures)).days + 1
        frequency_score = len(failures) / max(time_span, 1)
        
        # Create pattern
        pattern = RegressionPattern(
            pattern_id=pattern_id,
            pattern_name=pattern_name,
            description=f"Pattern based on {len(failures)} similar failures",
            failure_categories=failure_categories,
            affected_modules=affected_modules,
            common_triggers=common_triggers,
            occurrence_count=len(failures),
            first_seen=min(f.failure_time for f in failures),
            last_seen=max(f.failure_time for f in failures),
            frequency_score=frequency_score
        )
        
        # Extract resolution patterns
        resolved_failures = [f for f in failures if f.resolved]
        if resolved_failures:
            resolution_methods = [f.resolution_method for f in resolved_failures 
                                if f.resolution_method]
            pattern.common_resolutions = list(set(resolution_methods))
            
            resolution_times = [
                (f.resolution_time - f.failure_time).total_seconds() / 3600
                for f in resolved_failures if f.resolution_time
            ]
            if resolution_times:
                pattern.average_resolution_time = sum(resolution_times) / len(resolution_times)
            
            pattern.success_rate = len(resolved_failures) / len(failures)
        
        self._regression_patterns[pattern_id] = pattern
        self._stats['patterns_discovered'] += 1
        
        # Persist pattern
        self._persist_pattern(pattern)
        
        return pattern
    
    def _update_pattern_resolutions(self, failure: FailureRecord):
        """Update patterns with resolution data."""
        for pattern in self._regression_patterns.values():
            if (failure.module_under_test in pattern.affected_modules and
                failure.failure_category in pattern.failure_categories):
                
                if failure.resolution_method and failure.resolution_method not in pattern.common_resolutions:
                    pattern.common_resolutions.append(failure.resolution_method)
                
                # Recalculate success rate and average resolution time
                related_failures = [f for f in self._failure_records.values()
                                  if (f.module_under_test in pattern.affected_modules and
                                      f.failure_category in pattern.failure_categories)]
                
                resolved = [f for f in related_failures if f.resolved]
                if resolved:
                    pattern.success_rate = len(resolved) / len(related_failures)
                    
                    resolution_times = [
                        (f.resolution_time - f.failure_time).total_seconds() / 3600
                        for f in resolved if f.resolution_time
                    ]
                    if resolution_times:
                        pattern.average_resolution_time = sum(resolution_times) / len(resolution_times)
                
                # Persist updated pattern
                self._persist_pattern(pattern)
    
    def _generate_predictions_for_failure(self, failure: FailureRecord):
        """Generate predictions based on new failure."""
        # Look for patterns that might predict similar failures
        for pattern in self._regression_patterns.values():
            if (failure.module_under_test in pattern.affected_modules and
                pattern.frequency_score > 0.1):  # Only active patterns
                
                # Check if this indicates a trend
                recent_failures = [f for f in self._failure_records.values()
                                 if (f.failure_time > datetime.now() - timedelta(days=7) and
                                     f.module_under_test == failure.module_under_test)]
                
                if len(recent_failures) >= 2:
                    # Generate prediction for continued issues
                    prediction = self._generate_prediction_from_pattern(
                        failure.module_under_test, pattern, "24_hours"
                    )
                    
                    if prediction:
                        self._predictions[prediction.prediction_id] = prediction
                        self._persist_prediction(prediction)
    
    def _generate_prediction_from_pattern(self, module: str, pattern: RegressionPattern,
                                        time_window: str) -> Optional[FailurePrediction]:
        """Generate prediction from a pattern."""
        # Calculate confidence based on pattern characteristics
        confidence_score = 0.0
        
        # Frequency contribution
        if pattern.frequency_score > 0.5:
            confidence_score += 30
        elif pattern.frequency_score > 0.2:
            confidence_score += 20
        elif pattern.frequency_score > 0.1:
            confidence_score += 10
        
        # Success rate contribution (inverse - lower success rate = higher prediction confidence)
        confidence_score += (1 - pattern.success_rate) * 25
        
        # Occurrence count contribution
        if pattern.occurrence_count >= 10:
            confidence_score += 20
        elif pattern.occurrence_count >= 5:
            confidence_score += 10
        
        # Recent activity boost
        days_since_last = (datetime.now() - pattern.last_seen).days
        if days_since_last <= 1:
            confidence_score += 25
        elif days_since_last <= 7:
            confidence_score += 15
        elif days_since_last <= 30:
            confidence_score += 5
        
        # Map confidence score to enum
        if confidence_score >= 80:
            confidence = PredictionConfidence.VERY_HIGH
        elif confidence_score >= 65:
            confidence = PredictionConfidence.HIGH
        elif confidence_score >= 45:
            confidence = PredictionConfidence.MEDIUM
        elif confidence_score >= 25:
            confidence = PredictionConfidence.LOW
        else:
            confidence = PredictionConfidence.VERY_LOW
        
        # Don't create very low confidence predictions
        if confidence == PredictionConfidence.VERY_LOW:
            return None
        
        # Determine most likely failure type
        if pattern.failure_categories:
            predicted_failure_type = max(pattern.failure_categories, 
                                       key=lambda cat: len([f for f in self._failure_records.values()
                                                          if f.failure_category == cat and f.module_under_test == module]))
        else:
            predicted_failure_type = FailureCategory.LOGIC_ERROR
        
        # Generate prediction ID
        prediction_id = f"pred_{int(datetime.now().timestamp())}_{module}_{pattern.pattern_id}"
        
        # Create prediction
        prediction = FailurePrediction(
            prediction_id=prediction_id,
            target_module=module,
            predicted_failure_type=predicted_failure_type,
            confidence=confidence,
            probability_score=confidence_score,
            time_window=time_window,
            contributing_patterns=[pattern.pattern_id],
            risk_indicators=pattern.common_triggers,
            historical_evidence=[f"Pattern {pattern.pattern_name} has {pattern.occurrence_count} occurrences"],
            recommended_actions=self._generate_preventive_actions(pattern),
            monitoring_suggestions=[f"Monitor {module} for {predicted_failure_type.value} indicators"]
        )
        
        return prediction
    
    def _generate_preventive_actions(self, pattern: RegressionPattern) -> List[str]:
        """Generate preventive actions based on pattern."""
        actions = []
        
        # Based on common resolutions
        for resolution in pattern.common_resolutions:
            if resolution:
                actions.append(f"Consider applying: {resolution}")
        
        # Based on failure categories
        for category in pattern.failure_categories:
            if category == FailureCategory.IMPORT_ERROR:
                actions.append("Check import dependencies and paths")
            elif category == FailureCategory.ASSERTION_FAILURE:
                actions.append("Review test assertions and expected values")
            elif category == FailureCategory.TIMEOUT:
                actions.append("Optimize performance or increase timeout limits")
            elif category == FailureCategory.RESOURCE_ERROR:
                actions.append("Check system resources and limits")
        
        # General actions
        actions.extend([
            "Run comprehensive test suite",
            "Review recent code changes",
            "Check environment configuration"
        ])
        
        return actions[:5]  # Limit to top 5 actions
    
    def _calculate_failure_trend(self, failures: List[FailureRecord]) -> str:
        """Calculate failure trend over time."""
        if len(failures) < 4:
            return "stable"
        
        # Sort by time
        sorted_failures = sorted(failures, key=lambda f: f.failure_time)
        
        # Split into two halves
        mid_point = len(sorted_failures) // 2
        first_half = sorted_failures[:mid_point]
        second_half = sorted_failures[mid_point:]
        
        # Calculate rates (failures per day)
        first_half_days = (first_half[-1].failure_time - first_half[0].failure_time).days + 1
        second_half_days = (second_half[-1].failure_time - second_half[0].failure_time).days + 1
        
        first_rate = len(first_half) / max(first_half_days, 1)
        second_rate = len(second_half) / max(second_half_days, 1)
        
        # Determine trend
        if second_rate > first_rate * 1.2:
            return "increasing"
        elif second_rate < first_rate * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_resolution_trend(self, failures: List[FailureRecord]) -> str:
        """Calculate resolution trend over time."""
        resolved_failures = [f for f in failures if f.resolved and f.resolution_time]
        
        if len(resolved_failures) < 4:
            return "stable"
        
        # Calculate resolution times
        resolution_times = [
            (f.resolution_time - f.failure_time).total_seconds() / 3600
            for f in resolved_failures
        ]
        
        # Sort by failure time
        sorted_data = sorted(zip(resolved_failures, resolution_times), key=lambda x: x[0].failure_time)
        times = [time for _, time in sorted_data]
        
        # Split into halves and compare
        mid_point = len(times) // 2
        first_half_avg = sum(times[:mid_point]) / mid_point
        second_half_avg = sum(times[mid_point:]) / (len(times) - mid_point)
        
        if second_half_avg > first_half_avg * 1.2:
            return "increasing"  # Taking longer to resolve
        elif second_half_avg < first_half_avg * 0.8:
            return "decreasing"  # Resolving faster
        else:
            return "stable"
    
    def _identify_high_risk_modules(self, failures: List[FailureRecord]) -> List[str]:
        """Identify modules with high failure risk."""
        module_failures = defaultdict(list)
        
        for failure in failures:
            module_failures[failure.module_under_test].append(failure)
        
        # Calculate risk scores
        risk_scores = {}
        for module, module_fails in module_failures.items():
            # Failure count score
            count_score = len(module_fails)
            
            # Severity score
            severity_score = sum(
                4 if f.severity == FailureSeverity.CRITICAL else
                3 if f.severity == FailureSeverity.HIGH else
                2 if f.severity == FailureSeverity.MEDIUM else 1
                for f in module_fails
            )
            
            # Unresolved score
            unresolved_count = len([f for f in module_fails if not f.resolved])
            unresolved_score = unresolved_count * 2
            
            # Recent activity score
            recent_failures = [f for f in module_fails 
                             if f.failure_time > datetime.now() - timedelta(days=7)]
            recent_score = len(recent_failures) * 3
            
            total_score = count_score + severity_score + unresolved_score + recent_score
            risk_scores[module] = total_score
        
        # Sort by risk score and return top modules
        sorted_modules = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return modules with significantly high risk
        if sorted_modules:
            max_score = sorted_modules[0][1]
            threshold = max_score * 0.6  # 60% of highest score
            return [module for module, score in sorted_modules if score >= threshold]
        
        return []
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate overall prediction accuracy."""
        validated_predictions = [p for p in self._predictions.values() 
                               if p.prediction_outcome is not None]
        
        if not validated_predictions:
            return 0.0
        
        correct_predictions = [p for p in validated_predictions if p.prediction_outcome]
        accuracy = len(correct_predictions) / len(validated_predictions) * 100
        
        self._stats['prediction_accuracy'] = accuracy
        return accuracy
    
    def _update_prediction_accuracy(self):
        """Update prediction accuracy statistics."""
        self._stats['prediction_accuracy'] = self._calculate_prediction_accuracy()
    
    def _get_all_modules(self) -> List[str]:
        """Get all modules from failure records."""
        modules = set()
        for failure in self._failure_records.values():
            if failure.module_under_test:
                modules.add(failure.module_under_test)
        return list(modules)
    
    def _load_historical_data(self):
        """Load historical failure data."""
        try:
            # Load failures
            failures_dir = self.data_dir / "failures"
            for failure_file in failures_dir.glob("*.json"):
                with open(failure_file, 'r') as f:
                    data = json.load(f)
                    failure = self._deserialize_failure(data)
                    self._failure_records[failure.failure_id] = failure
            
            # Load patterns
            patterns_dir = self.data_dir / "patterns"
            for pattern_file in patterns_dir.glob("*.json"):
                with open(pattern_file, 'r') as f:
                    data = json.load(f)
                    pattern = self._deserialize_pattern(data)
                    self._regression_patterns[pattern.pattern_id] = pattern
            
            # Load predictions
            predictions_dir = self.data_dir / "predictions"
            for pred_file in predictions_dir.glob("*.json"):
                with open(pred_file, 'r') as f:
                    data = json.load(f)
                    prediction = self._deserialize_prediction(data)
                    self._predictions[prediction.prediction_id] = prediction
                    
        except Exception as e:
            print(f"⚠️ Error loading historical data: {e}")
    
    def _persist_failure(self, failure: FailureRecord):
        """Persist failure record."""
        try:
            file_path = self.data_dir / "failures" / f"{failure.failure_id}.json"
            with open(file_path, 'w') as f:
                json.dump(self._serialize_failure(failure), f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Error persisting failure: {e}")
    
    def _persist_pattern(self, pattern: RegressionPattern):
        """Persist regression pattern."""
        try:
            file_path = self.data_dir / "patterns" / f"{pattern.pattern_id}.json"
            with open(file_path, 'w') as f:
                json.dump(self._serialize_pattern(pattern), f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Error persisting pattern: {e}")
    
    def _persist_prediction(self, prediction: FailurePrediction):
        """Persist failure prediction."""
        try:
            file_path = self.data_dir / "predictions" / f"{prediction.prediction_id}.json"
            with open(file_path, 'w') as f:
                json.dump(self._serialize_prediction(prediction), f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Error persisting prediction: {e}")
    
    def _serialize_failure(self, failure: FailureRecord) -> Dict[str, Any]:
        """Serialize failure record for persistence."""
        data = {
            'failure_id': failure.failure_id,
            'test_file': failure.test_file,
            'test_function': failure.test_function,
            'module_under_test': failure.module_under_test,
            'failure_category': failure.failure_category.value,
            'error_message': failure.error_message,
            'stack_trace': failure.stack_trace,
            'failure_time': failure.failure_time.isoformat(),
            'git_commit': failure.git_commit,
            'branch_name': failure.branch_name,
            'environment': failure.environment,
            'regression_type': failure.regression_type.value if failure.regression_type else None,
            'severity': failure.severity.value,
            'resolved': failure.resolved,
            'resolution_time': failure.resolution_time.isoformat() if failure.resolution_time else None,
            'resolution_method': failure.resolution_method,
            'resolution_notes': failure.resolution_notes,
            'affected_files': failure.affected_files,
            'related_failures': failure.related_failures,
            'tags': failure.tags
        }
        return data
    
    def _deserialize_failure(self, data: Dict[str, Any]) -> FailureRecord:
        """Deserialize failure record from persistence."""
        failure = FailureRecord(
            failure_id=data['failure_id'],
            test_file=data['test_file'],
            test_function=data['test_function'],
            module_under_test=data['module_under_test'],
            failure_category=FailureCategory(data['failure_category']),
            error_message=data['error_message'],
            stack_trace=data.get('stack_trace', ''),
            failure_time=datetime.fromisoformat(data['failure_time']),
            git_commit=data.get('git_commit'),
            branch_name=data.get('branch_name'),
            environment=data.get('environment', 'unknown'),
            regression_type=RegressionType(data['regression_type']) if data.get('regression_type') else None,
            severity=FailureSeverity(data.get('severity', 'medium')),
            resolved=data.get('resolved', False),
            resolution_method=data.get('resolution_method'),
            resolution_notes=data.get('resolution_notes'),
            affected_files=data.get('affected_files', []),
            related_failures=data.get('related_failures', []),
            tags=data.get('tags', [])
        )
        
        if data.get('resolution_time'):
            failure.resolution_time = datetime.fromisoformat(data['resolution_time'])
        
        return failure
    
    def _serialize_pattern(self, pattern: RegressionPattern) -> Dict[str, Any]:
        """Serialize regression pattern for persistence."""
        return {
            'pattern_id': pattern.pattern_id,
            'pattern_name': pattern.pattern_name,
            'description': pattern.description,
            'failure_categories': [cat.value for cat in pattern.failure_categories],
            'affected_modules': pattern.affected_modules,
            'common_triggers': pattern.common_triggers,
            'occurrence_count': pattern.occurrence_count,
            'first_seen': pattern.first_seen.isoformat(),
            'last_seen': pattern.last_seen.isoformat(),
            'frequency_score': pattern.frequency_score,
            'typical_environment': pattern.typical_environment,
            'seasonal_pattern': pattern.seasonal_pattern,
            'time_based_pattern': pattern.time_based_pattern,
            'common_resolutions': pattern.common_resolutions,
            'average_resolution_time': pattern.average_resolution_time,
            'success_rate': pattern.success_rate,
            'prediction_indicators': pattern.prediction_indicators,
            'confidence_level': pattern.confidence_level.value
        }
    
    def _deserialize_pattern(self, data: Dict[str, Any]) -> RegressionPattern:
        """Deserialize regression pattern from persistence."""
        return RegressionPattern(
            pattern_id=data['pattern_id'],
            pattern_name=data['pattern_name'],
            description=data['description'],
            failure_categories=[FailureCategory(cat) for cat in data.get('failure_categories', [])],
            affected_modules=data.get('affected_modules', []),
            common_triggers=data.get('common_triggers', []),
            occurrence_count=data.get('occurrence_count', 0),
            first_seen=datetime.fromisoformat(data['first_seen']),
            last_seen=datetime.fromisoformat(data['last_seen']),
            frequency_score=data.get('frequency_score', 0.0),
            typical_environment=data.get('typical_environment'),
            seasonal_pattern=data.get('seasonal_pattern', False),
            time_based_pattern=data.get('time_based_pattern', False),
            common_resolutions=data.get('common_resolutions', []),
            average_resolution_time=data.get('average_resolution_time', 0.0),
            success_rate=data.get('success_rate', 0.0),
            prediction_indicators=data.get('prediction_indicators', []),
            confidence_level=PredictionConfidence(data.get('confidence_level', 'medium'))
        )
    
    def _serialize_prediction(self, prediction: FailurePrediction) -> Dict[str, Any]:
        """Serialize failure prediction for persistence."""
        return {
            'prediction_id': prediction.prediction_id,
            'target_module': prediction.target_module,
            'predicted_failure_type': prediction.predicted_failure_type.value,
            'confidence': prediction.confidence.value,
            'probability_score': prediction.probability_score,
            'time_window': prediction.time_window,
            'contributing_patterns': prediction.contributing_patterns,
            'risk_indicators': prediction.risk_indicators,
            'historical_evidence': prediction.historical_evidence,
            'predicted_at': prediction.predicted_at.isoformat(),
            'triggering_event': prediction.triggering_event,
            'environmental_factors': prediction.environmental_factors,
            'recommended_actions': prediction.recommended_actions,
            'monitoring_suggestions': prediction.monitoring_suggestions,
            'prediction_outcome': prediction.prediction_outcome,
            'actual_failure': prediction.actual_failure,
            'validation_time': prediction.validation_time.isoformat() if prediction.validation_time else None
        }
    
    def _deserialize_prediction(self, data: Dict[str, Any]) -> FailurePrediction:
        """Deserialize failure prediction from persistence."""
        prediction = FailurePrediction(
            prediction_id=data['prediction_id'],
            target_module=data['target_module'],
            predicted_failure_type=FailureCategory(data['predicted_failure_type']),
            confidence=PredictionConfidence(data['confidence']),
            probability_score=data.get('probability_score', 0.0),
            time_window=data.get('time_window', '24_hours'),
            contributing_patterns=data.get('contributing_patterns', []),
            risk_indicators=data.get('risk_indicators', []),
            historical_evidence=data.get('historical_evidence', []),
            predicted_at=datetime.fromisoformat(data['predicted_at']),
            triggering_event=data.get('triggering_event'),
            environmental_factors=data.get('environmental_factors', []),
            recommended_actions=data.get('recommended_actions', []),
            monitoring_suggestions=data.get('monitoring_suggestions', []),
            prediction_outcome=data.get('prediction_outcome'),
            actual_failure=data.get('actual_failure')
        )
        
        if data.get('validation_time'):
            prediction.validation_time = datetime.fromisoformat(data['validation_time'])
        
        return prediction
    
    def get_regression_statistics(self) -> Dict[str, Any]:
        """Get regression tracking statistics."""
        total_failures = len(self._failure_records)
        resolved_failures = len([f for f in self._failure_records.values() if f.resolved])
        active_predictions = len([p for p in self._predictions.values() if not p.prediction_outcome])
        
        # Category distribution
        category_counts = {}
        for category in FailureCategory:
            category_counts[category.value] = len([
                f for f in self._failure_records.values()
                if f.failure_category == category
            ])
        
        # Severity distribution
        severity_counts = {}
        for severity in FailureSeverity:
            severity_counts[severity.value] = len([
                f for f in self._failure_records.values()
                if f.severity == severity
            ])
        
        return {
            "total_failures": total_failures,
            "resolved_failures": resolved_failures,
            "resolution_rate": (resolved_failures / max(total_failures, 1)) * 100,
            "active_patterns": len(self._regression_patterns),
            "total_predictions": len(self._predictions),
            "active_predictions": active_predictions,
            "prediction_accuracy": self._stats['prediction_accuracy'],
            "category_distribution": category_counts,
            "severity_distribution": severity_counts,
            "statistics": dict(self._stats)
        }
    
    def export_regression_report(self, output_path: str = "regression_analysis_report.json"):
        """Export comprehensive regression analysis report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_regression_statistics(),
            "failures": [],
            "patterns": [],
            "predictions": []
        }
        
        # Add failure summaries
        for failure in self._failure_records.values():
            failure_data = {
                "failure_id": failure.failure_id,
                "test_function": failure.test_function,
                "module_under_test": failure.module_under_test,
                "failure_category": failure.failure_category.value,
                "severity": failure.severity.value,
                "failure_time": failure.failure_time.isoformat(),
                "resolved": failure.resolved,
                "resolution_method": failure.resolution_method,
                "regression_type": failure.regression_type.value if failure.regression_type else None
            }
            report["failures"].append(failure_data)
        
        # Add pattern summaries
        for pattern in self._regression_patterns.values():
            pattern_data = {
                "pattern_id": pattern.pattern_id,
                "pattern_name": pattern.pattern_name,
                "occurrence_count": pattern.occurrence_count,
                "frequency_score": pattern.frequency_score,
                "affected_modules": pattern.affected_modules,
                "success_rate": pattern.success_rate,
                "average_resolution_time": pattern.average_resolution_time
            }
            report["patterns"].append(pattern_data)
        
        # Add prediction summaries
        for prediction in self._predictions.values():
            pred_data = {
                "prediction_id": prediction.prediction_id,
                "target_module": prediction.target_module,
                "predicted_failure_type": prediction.predicted_failure_type.value,
                "confidence": prediction.confidence.value,
                "probability_score": prediction.probability_score,
                "predicted_at": prediction.predicted_at.isoformat(),
                "prediction_outcome": prediction.prediction_outcome
            }
            report["predictions"].append(pred_data)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📄 Regression analysis report exported to {output_path}")
        except Exception as e:
            print(f"⚠️ Error exporting regression report: {e}")


# Convenience functions for regression tracking
def track_test_failure(test_file: str, test_function: str, error_message: str, 
                      tracker: RegressionTracker) -> str:
    """Track a test failure."""
    return tracker.record_failure(test_file, test_function, error_message)


def get_high_risk_predictions(tracker: RegressionTracker, max_count: int = 10) -> List[FailurePrediction]:
    """Get high-risk failure predictions."""
    predictions = tracker.predict_potential_failures()
    high_risk = [p for p in predictions if p.confidence.value in ['high', 'very_high']]
    return high_risk[:max_count]


def analyze_module_regression_risk(module_path: str, tracker: RegressionTracker) -> Dict[str, Any]:
    """Analyze regression risk for a specific module."""
    summary = tracker.analyze_regression_trends()
    
    module_failures = [f for f in tracker._failure_records.values() 
                      if f.module_under_test == module_path]
    
    if not module_failures:
        return {"risk_level": "low", "failure_count": 0}
    
    recent_failures = [f for f in module_failures 
                      if f.failure_time > datetime.now() - timedelta(days=30)]
    
    unresolved_count = len([f for f in module_failures if not f.resolved])
    
    risk_score = len(recent_failures) * 2 + unresolved_count * 3
    
    if risk_score >= 10:
        risk_level = "critical"
    elif risk_score >= 6:
        risk_level = "high"
    elif risk_score >= 3:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return {
        "risk_level": risk_level,
        "total_failures": len(module_failures),
        "recent_failures": len(recent_failures),
        "unresolved_failures": unresolved_count,
        "risk_score": risk_score
    }