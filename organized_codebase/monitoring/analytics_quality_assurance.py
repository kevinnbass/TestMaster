"""
Analytics Data Quality Assurance System
=======================================

Comprehensive data quality monitoring, integrity checks, and automated 
remediation to ensure analytics reliability and accuracy.

Author: TestMaster Team
"""

import logging
import time
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

logger = logging.getLogger(__name__)

class QualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"

class IntegrityStatus(Enum):
    VALID = "valid"
    CORRUPTED = "corrupted"
    MISSING = "missing"
    INCONSISTENT = "inconsistent"

@dataclass
class QualityMetric:
    """Represents a data quality metric."""
    name: str
    value: float
    threshold: float
    status: QualityLevel
    description: str
    timestamp: datetime

@dataclass
class IntegrityCheck:
    """Represents a data integrity check result."""
    check_name: str
    status: IntegrityStatus
    expected_value: Any
    actual_value: Any
    error_message: Optional[str] = None
    timestamp: Optional[datetime] = None

class AnalyticsQualityAssurance:
    """
    Comprehensive data quality assurance system for analytics.
    """
    
    def __init__(self, check_interval: int = 30):
        """
        Initialize the quality assurance system.
        
        Args:
            check_interval: Interval between quality checks in seconds
        """
        self.check_interval = check_interval
        
        # Quality metrics tracking
        self.quality_metrics = defaultdict(deque)
        self.quality_history = deque(maxlen=1000)
        self.integrity_checks = deque(maxlen=500)
        
        # Quality thresholds
        self.quality_thresholds = {
            'completeness': 0.95,
            'accuracy': 0.98,
            'consistency': 0.95,
            'timeliness': 0.90,
            'validity': 0.99,
            'uniqueness': 0.98
        }
        
        # Data validation rules
        self.validation_rules = {}
        self.custom_validators = {}
        
        # Quality monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Quality alerts
        self.quality_alerts = deque(maxlen=100)
        self.alert_callbacks = []
        
        # Data fingerprinting for integrity
        self.data_fingerprints = {}
        self.fingerprint_history = deque(maxlen=1000)
        
        # Performance tracking
        self.quality_stats = {
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'integrity_violations': 0,
            'quality_score_history': deque(maxlen=100),
            'start_time': datetime.now()
        }
        
        # Setup default validation rules
        self._setup_default_rules()
        
        logger.info("Analytics Quality Assurance system initialized")
    
    def start_monitoring(self):
        """Start continuous quality monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Quality assurance monitoring started")
    
    def stop_monitoring(self):
        """Stop quality monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Quality assurance monitoring stopped")
    
    def assess_data_quality(self, data: Dict[str, Any], 
                           data_source: str = "unknown") -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment.
        
        Args:
            data: Data to assess
            data_source: Source identifier for the data
        
        Returns:
            Quality assessment results
        """
        start_time = time.time()
        self.quality_stats['total_checks'] += 1
        
        try:
            # Perform quality checks
            completeness = self._check_completeness(data)
            accuracy = self._check_accuracy(data)
            consistency = self._check_consistency(data)
            timeliness = self._check_timeliness(data)
            validity = self._check_validity(data)
            uniqueness = self._check_uniqueness(data, data_source)
            
            # Calculate overall quality score
            quality_scores = {
                'completeness': completeness,
                'accuracy': accuracy,
                'consistency': consistency,
                'timeliness': timeliness,
                'validity': validity,
                'uniqueness': uniqueness
            }
            
            overall_score = statistics.mean(quality_scores.values())
            quality_level = self._determine_quality_level(overall_score)
            
            # Create quality metrics
            quality_metrics = []
            for metric_name, score in quality_scores.items():
                threshold = self.quality_thresholds.get(metric_name, 0.9)
                status = QualityLevel.GOOD if score >= threshold else QualityLevel.POOR
                
                quality_metrics.append(QualityMetric(
                    name=metric_name,
                    value=score,
                    threshold=threshold,
                    status=status,
                    description=f"{metric_name.title()} quality assessment",
                    timestamp=datetime.now()
                ))
            
            # Perform integrity checks
            integrity_results = self._perform_integrity_checks(data, data_source)
            
            # Generate data fingerprint
            fingerprint = self._generate_fingerprint(data)
            self._track_fingerprint(fingerprint, data_source)
            
            # Create assessment result
            assessment = {
                'data_source': data_source,
                'timestamp': datetime.now().isoformat(),
                'overall_quality_score': overall_score,
                'quality_level': quality_level.value,
                'quality_scores': quality_scores,
                'quality_metrics': [
                    {
                        'name': m.name,
                        'value': m.value,
                        'threshold': m.threshold,
                        'status': m.status.value,
                        'description': m.description
                    } for m in quality_metrics
                ],
                'integrity_checks': [
                    {
                        'check_name': c.check_name,
                        'status': c.status.value,
                        'expected_value': c.expected_value,
                        'actual_value': c.actual_value,
                        'error_message': c.error_message
                    } for c in integrity_results
                ],
                'data_fingerprint': fingerprint,
                'assessment_duration_ms': (time.time() - start_time) * 1000,
                'recommendations': self._generate_recommendations(quality_scores, integrity_results)
            }
            
            # Track quality history
            self.quality_history.append(assessment)
            self.quality_stats['quality_score_history'].append(overall_score)
            
            # Check for quality alerts
            self._check_quality_alerts(assessment)
            
            if overall_score >= 0.8:
                self.quality_stats['passed_checks'] += 1
            else:
                self.quality_stats['failed_checks'] += 1
            
            return assessment
            
        except Exception as e:
            self.quality_stats['failed_checks'] += 1
            logger.error(f"Quality assessment failed: {e}")
            return {
                'data_source': data_source,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'overall_quality_score': 0.0,
                'quality_level': QualityLevel.CRITICAL.value
            }
    
    def _check_completeness(self, data: Dict[str, Any]) -> float:
        """Check data completeness."""
        if not data:
            return 0.0
        
        total_fields = 0
        complete_fields = 0
        
        def count_fields(obj, path=""):
            nonlocal total_fields, complete_fields
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    total_fields += 1
                    if value is not None and value != "" and value != []:
                        complete_fields += 1
                    
                    if isinstance(value, (dict, list)):
                        count_fields(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        count_fields(item, f"{path}[{i}]" if path else f"[{i}]")
        
        count_fields(data)
        
        return complete_fields / max(total_fields, 1)
    
    def _check_accuracy(self, data: Dict[str, Any]) -> float:
        """Check data accuracy using validation rules."""
        if not data:
            return 0.0
        
        total_validations = 0
        passed_validations = 0
        
        def validate_recursive(obj, path=""):
            nonlocal total_validations, passed_validations
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check if we have validation rules for this field
                    if current_path in self.validation_rules:
                        total_validations += 1
                        rule = self.validation_rules[current_path]
                        
                        try:
                            if rule(value):
                                passed_validations += 1
                        except Exception:
                            pass  # Validation failed
                    
                    # Check numeric ranges
                    if isinstance(value, (int, float)):
                        total_validations += 1
                        if self._is_reasonable_numeric_value(key, value):
                            passed_validations += 1
                    
                    if isinstance(value, (dict, list)):
                        validate_recursive(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        validate_recursive(item, f"{path}[{i}]" if path else f"[{i}]")
        
        validate_recursive(data)
        
        # If no validations were performed, assume high accuracy
        return passed_validations / max(total_validations, 1) if total_validations > 0 else 0.95
    
    def _check_consistency(self, data: Dict[str, Any]) -> float:
        """Check data consistency."""
        if not data:
            return 0.0
        
        consistency_checks = 0
        consistency_passes = 0
        
        # Check timestamp consistency
        timestamps = []
        def collect_timestamps(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if 'timestamp' in key.lower() or 'time' in key.lower():
                        if isinstance(value, str):
                            try:
                                timestamps.append(datetime.fromisoformat(value.replace('Z', '+00:00')))
                            except ValueError:
                                pass
                    elif isinstance(value, (dict, list)):
                        collect_timestamps(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        collect_timestamps(item)
        
        collect_timestamps(data)
        
        if len(timestamps) > 1:
            consistency_checks += 1
            # Check if timestamps are within reasonable range (e.g., within last hour)
            time_range = max(timestamps) - min(timestamps)
            if time_range <= timedelta(hours=1):
                consistency_passes += 1
        
        # Check numerical consistency
        numeric_values = defaultdict(list)
        def collect_numeric_values(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if isinstance(value, (int, float)):
                        numeric_values[key.lower()].append(value)
                    elif isinstance(value, (dict, list)):
                        collect_numeric_values(value, current_path)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        collect_numeric_values(item, path)
        
        collect_numeric_values(data)
        
        for field_name, values in numeric_values.items():
            if len(values) > 1:
                consistency_checks += 1
                # Check coefficient of variation
                if statistics.mean(values) > 0:
                    cv = statistics.stdev(values) / statistics.mean(values)
                    if cv < 0.5:  # Low variability indicates consistency
                        consistency_passes += 1
        
        return consistency_passes / max(consistency_checks, 1) if consistency_checks > 0 else 0.9
    
    def _check_timeliness(self, data: Dict[str, Any]) -> float:
        """Check data timeliness."""
        if not data:
            return 0.0
        
        timestamps = []
        def collect_timestamps(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if 'timestamp' in key.lower():
                        if isinstance(value, str):
                            try:
                                timestamps.append(datetime.fromisoformat(value.replace('Z', '+00:00')))
                            except ValueError:
                                pass
                    elif isinstance(value, (dict, list)):
                        collect_timestamps(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        collect_timestamps(item)
        
        collect_timestamps(data)
        
        if not timestamps:
            return 0.8  # No timestamps found, assume reasonable timeliness
        
        now = datetime.now()
        most_recent = max(timestamps)
        age_minutes = (now - most_recent.replace(tzinfo=None)).total_seconds() / 60
        
        # Timeliness score decreases with age
        if age_minutes <= 5:
            return 1.0
        elif age_minutes <= 30:
            return 0.9
        elif age_minutes <= 60:
            return 0.7
        elif age_minutes <= 300:  # 5 hours
            return 0.5
        else:
            return 0.3
    
    def _check_validity(self, data: Dict[str, Any]) -> float:
        """Check data validity."""
        if not data:
            return 0.0
        
        validity_checks = 0
        validity_passes = 0
        
        def validate_recursive(obj):
            nonlocal validity_checks, validity_passes
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    validity_checks += 1
                    
                    # Type validation
                    if self._is_valid_type(key, value):
                        validity_passes += 1
                    
                    if isinstance(value, (dict, list)):
                        validate_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        validate_recursive(item)
        
        validate_recursive(data)
        
        return validity_passes / max(validity_checks, 1)
    
    def _check_uniqueness(self, data: Dict[str, Any], data_source: str) -> float:
        """Check data uniqueness."""
        fingerprint = self._generate_fingerprint(data)
        
        # Check against recent fingerprints for the same source
        recent_fingerprints = [
            fp for fp in self.fingerprint_history 
            if fp['source'] == data_source and 
            datetime.now() - fp['timestamp'] <= timedelta(minutes=30)
        ]
        
        if not recent_fingerprints:
            return 1.0  # First data point, assume unique
        
        # Check for duplicates
        duplicate_count = sum(1 for fp in recent_fingerprints if fp['fingerprint'] == fingerprint)
        
        # Uniqueness score decreases with duplicates
        uniqueness = max(0.0, 1.0 - (duplicate_count * 0.2))
        
        return uniqueness
    
    def _perform_integrity_checks(self, data: Dict[str, Any], 
                                 data_source: str) -> List[IntegrityCheck]:
        """Perform comprehensive integrity checks."""
        checks = []
        
        # Check required fields
        required_fields = ['timestamp']
        for field in required_fields:
            if field in data:
                checks.append(IntegrityCheck(
                    check_name=f"required_field_{field}",
                    status=IntegrityStatus.VALID,
                    expected_value="present",
                    actual_value="present"
                ))
            else:
                checks.append(IntegrityCheck(
                    check_name=f"required_field_{field}",
                    status=IntegrityStatus.MISSING,
                    expected_value="present",
                    actual_value="missing",
                    error_message=f"Required field '{field}' is missing"
                ))
        
        # Check data structure integrity
        if isinstance(data, dict):
            checks.append(IntegrityCheck(
                check_name="data_structure",
                status=IntegrityStatus.VALID,
                expected_value="dict",
                actual_value="dict"
            ))
        else:
            checks.append(IntegrityCheck(
                check_name="data_structure",
                status=IntegrityStatus.CORRUPTED,
                expected_value="dict",
                actual_value=type(data).__name__,
                error_message="Data is not a dictionary"
            ))
        
        # Check for null values in critical fields
        critical_fields = ['system_metrics', 'test_analytics']
        for field in critical_fields:
            if field in data and data[field] is None:
                checks.append(IntegrityCheck(
                    check_name=f"null_check_{field}",
                    status=IntegrityStatus.CORRUPTED,
                    expected_value="not null",
                    actual_value="null",
                    error_message=f"Critical field '{field}' is null"
                ))
        
        return checks
    
    def _generate_fingerprint(self, data: Dict[str, Any]) -> str:
        """Generate a fingerprint for data integrity tracking."""
        # Create a normalized representation of the data
        normalized = json.dumps(data, sort_keys=True, default=str)
        
        # Generate SHA-256 hash
        fingerprint = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        
        return fingerprint
    
    def _track_fingerprint(self, fingerprint: str, data_source: str):
        """Track data fingerprint for integrity monitoring."""
        self.fingerprint_history.append({
            'fingerprint': fingerprint,
            'source': data_source,
            'timestamp': datetime.now()
        })
    
    def _is_reasonable_numeric_value(self, field_name: str, value: float) -> bool:
        """Check if a numeric value is reasonable."""
        field_lower = field_name.lower()
        
        # Percentage fields should be 0-100
        if 'percent' in field_lower:
            return 0 <= value <= 100
        
        # CPU usage should be 0-100
        if 'cpu' in field_lower and 'usage' in field_lower:
            return 0 <= value <= 100
        
        # Memory should be positive
        if 'memory' in field_lower:
            return value >= 0
        
        # Counts should be non-negative
        if any(word in field_lower for word in ['count', 'total', 'num']):
            return value >= 0
        
        # General sanity check for extremely large or small values
        return -1e10 <= value <= 1e10
    
    def _is_valid_type(self, field_name: str, value: Any) -> bool:
        """Check if value type is valid for the field."""
        field_lower = field_name.lower()
        
        # Timestamp fields should be strings
        if 'timestamp' in field_lower or 'time' in field_lower:
            return isinstance(value, str)
        
        # Count fields should be numeric
        if any(word in field_lower for word in ['count', 'total', 'num']):
            return isinstance(value, (int, float))
        
        # Percentage fields should be numeric
        if 'percent' in field_lower:
            return isinstance(value, (int, float))
        
        # Boolean fields
        if any(word in field_lower for word in ['active', 'enabled', 'success']):
            return isinstance(value, bool)
        
        return True  # Default to valid
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level from score."""
        if score >= 0.95:
            return QualityLevel.EXCELLENT
        elif score >= 0.85:
            return QualityLevel.GOOD
        elif score >= 0.70:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.50:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def _generate_recommendations(self, quality_scores: Dict[str, float], 
                                 integrity_checks: List[IntegrityCheck]) -> List[str]:
        """Generate recommendations for improving data quality."""
        recommendations = []
        
        # Quality-based recommendations
        for metric, score in quality_scores.items():
            threshold = self.quality_thresholds.get(metric, 0.9)
            if score < threshold:
                if metric == 'completeness':
                    recommendations.append("Ensure all required fields are populated")
                elif metric == 'accuracy':
                    recommendations.append("Review data validation rules and fix invalid values")
                elif metric == 'consistency':
                    recommendations.append("Check for inconsistent data patterns and standardize formats")
                elif metric == 'timeliness':
                    recommendations.append("Ensure data is collected and processed more frequently")
                elif metric == 'validity':
                    recommendations.append("Validate data types and formats before processing")
                elif metric == 'uniqueness':
                    recommendations.append("Check for duplicate data and implement deduplication")
        
        # Integrity-based recommendations
        for check in integrity_checks:
            if check.status != IntegrityStatus.VALID:
                recommendations.append(f"Fix integrity issue: {check.error_message}")
        
        return recommendations
    
    def _check_quality_alerts(self, assessment: Dict[str, Any]):
        """Check for quality alerts and trigger callbacks."""
        quality_score = assessment.get('overall_quality_score', 0)
        quality_level = assessment.get('quality_level', 'critical')
        
        if quality_score < 0.7 or quality_level in ['poor', 'critical']:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': 'quality_degradation',
                'severity': 'high' if quality_score < 0.5 else 'medium',
                'message': f"Data quality degraded: {quality_score:.2f} ({quality_level})",
                'data_source': assessment.get('data_source', 'unknown'),
                'assessment': assessment
            }
            
            self.quality_alerts.append(alert)
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def _setup_default_rules(self):
        """Setup default validation rules."""
        # CPU usage validation
        self.validation_rules['cpu.usage_percent'] = lambda x: isinstance(x, (int, float)) and 0 <= x <= 100
        self.validation_rules['system_metrics.cpu.usage_percent'] = lambda x: isinstance(x, (int, float)) and 0 <= x <= 100
        
        # Memory validation
        self.validation_rules['memory.percent'] = lambda x: isinstance(x, (int, float)) and 0 <= x <= 100
        self.validation_rules['system_metrics.memory.percent'] = lambda x: isinstance(x, (int, float)) and 0 <= x <= 100
        
        # Test analytics validation
        self.validation_rules['test_analytics.total_tests'] = lambda x: isinstance(x, int) and x >= 0
        self.validation_rules['test_analytics.passed'] = lambda x: isinstance(x, int) and x >= 0
        self.validation_rules['test_analytics.failed'] = lambda x: isinstance(x, int) and x >= 0
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                time.sleep(self.check_interval)
                
                if not self.monitoring_active:
                    break
                
                # Perform periodic quality analysis
                self._perform_periodic_analysis()
                
            except Exception as e:
                logger.error(f"Quality monitoring error: {e}")
    
    def _perform_periodic_analysis(self):
        """Perform periodic quality analysis."""
        try:
            # Analyze quality trends
            if len(self.quality_stats['quality_score_history']) >= 5:
                recent_scores = list(self.quality_stats['quality_score_history'])[-5:]
                trend = "improving" if recent_scores[-1] > recent_scores[0] else "declining"
                
                if trend == "declining" and statistics.mean(recent_scores) < 0.8:
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'type': 'quality_trend',
                        'severity': 'medium',
                        'message': f"Quality trend is {trend}: {statistics.mean(recent_scores):.2f}",
                        'trend': trend,
                        'average_score': statistics.mean(recent_scores)
                    }
                    self.quality_alerts.append(alert)
            
            # Clean up old data
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.fingerprint_history = deque([
                fp for fp in self.fingerprint_history 
                if fp['timestamp'] > cutoff_time
            ], maxlen=1000)
            
        except Exception as e:
            logger.error(f"Periodic analysis failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a callback for quality alerts."""
        self.alert_callbacks.append(callback)
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get overall quality summary."""
        uptime = (datetime.now() - self.quality_stats['start_time']).total_seconds()
        
        recent_assessments = list(self.quality_history)[-10:] if self.quality_history else []
        avg_quality = statistics.mean([
            a.get('overall_quality_score', 0) for a in recent_assessments
        ]) if recent_assessments else 0
        
        return {
            'total_assessments': len(self.quality_history),
            'total_checks': self.quality_stats['total_checks'],
            'passed_checks': self.quality_stats['passed_checks'],
            'failed_checks': self.quality_stats['failed_checks'],
            'success_rate': (self.quality_stats['passed_checks'] / 
                           max(self.quality_stats['total_checks'], 1)) * 100,
            'average_quality_score': avg_quality,
            'recent_alerts': len([
                a for a in self.quality_alerts 
                if datetime.now() - datetime.fromisoformat(a['timestamp']) <= timedelta(hours=1)
            ]),
            'integrity_violations': self.quality_stats['integrity_violations'],
            'uptime_seconds': uptime,
            'monitoring_active': self.monitoring_active,
            'quality_thresholds': self.quality_thresholds,
            'validation_rules_count': len(self.validation_rules)
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent quality alerts."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.quality_alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
    
    def shutdown(self):
        """Shutdown the quality assurance system."""
        self.stop_monitoring()
        logger.info("Analytics Quality Assurance system shutdown")