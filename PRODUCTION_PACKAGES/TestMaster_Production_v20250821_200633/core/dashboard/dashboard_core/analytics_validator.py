"""
Analytics Data Validator
========================

Validates and ensures quality of analytics data before storage and processing.
Provides data integrity checks and anomaly detection.

Author: TestMaster Team
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import statistics

logger = logging.getLogger(__name__)

class AnalyticsValidator:
    """
    Validates analytics data for quality, consistency, and anomalies.
    """
    
    def __init__(self, history_window: int = 100):
        """
        Initialize the analytics validator.
        
        Args:
            history_window: Number of historical data points to keep for validation
        """
        self.history_window = history_window
        
        # Historical data for validation
        self.performance_history = deque(maxlen=history_window)
        self.test_history = deque(maxlen=history_window)
        self.system_history = deque(maxlen=history_window)
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'anomalies_detected': 0,
            'data_quality_score': 100.0
        }
        
        logger.info("Analytics Validator initialized")
    
    def validate_performance_metrics(self, metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate performance metrics for consistency and anomalies.
        
        Args:
            metrics: Performance metrics to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Required fields validation
        required_fields = ['cpu_usage', 'memory_usage_mb', 'disk_usage_percent', 'timestamp']
        for field in required_fields:
            if field not in metrics:
                issues.append(f"Missing required field: {field}")
        
        # Range validation
        if 'cpu_usage' in metrics:
            cpu = metrics['cpu_usage']
            if not isinstance(cpu, (int, float)) or cpu < 0 or cpu > 100:
                issues.append(f"Invalid CPU usage: {cpu} (must be 0-100)")
        
        if 'memory_usage_mb' in metrics:
            memory = metrics['memory_usage_mb']
            if not isinstance(memory, (int, float)) or memory < 0:
                issues.append(f"Invalid memory usage: {memory} (must be >= 0)")
        
        if 'disk_usage_percent' in metrics:
            disk = metrics['disk_usage_percent']
            if not isinstance(disk, (int, float)) or disk < 0 or disk > 100:
                issues.append(f"Invalid disk usage: {disk} (must be 0-100)")
        
        if 'network_kb_s' in metrics:
            network = metrics['network_kb_s']
            if not isinstance(network, (int, float)) or network < 0:
                issues.append(f"Invalid network usage: {network} (must be >= 0)")
        
        # Anomaly detection
        if self.performance_history and not issues:
            anomalies = self._detect_performance_anomalies(metrics)
            if anomalies:
                issues.extend([f"Anomaly detected: {anomaly}" for anomaly in anomalies])
        
        # Add to history if valid
        if not issues:
            self.performance_history.append(metrics)
        
        return len(issues) == 0, issues
    
    def validate_test_metrics(self, metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate test metrics for consistency.
        
        Args:
            metrics: Test metrics to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Required fields validation
        required_fields = ['total_tests', 'passed', 'failed', 'coverage_percent']
        for field in required_fields:
            if field not in metrics:
                issues.append(f"Missing required field: {field}")
        
        # Logical consistency validation
        if all(field in metrics for field in ['total_tests', 'passed', 'failed', 'skipped']):
            total = metrics['total_tests']
            passed = metrics['passed']
            failed = metrics['failed']
            skipped = metrics.get('skipped', 0)
            
            if passed + failed + skipped != total:
                issues.append(f"Test counts don't add up: {passed}+{failed}+{skipped} != {total}")
        
        # Range validation
        if 'coverage_percent' in metrics:
            coverage = metrics['coverage_percent']
            if not isinstance(coverage, (int, float)) or coverage < 0 or coverage > 100:
                issues.append(f"Invalid coverage: {coverage} (must be 0-100)")
        
        # Non-negative validation
        for field in ['total_tests', 'passed', 'failed', 'skipped']:
            if field in metrics:
                value = metrics[field]
                if not isinstance(value, int) or value < 0:
                    issues.append(f"Invalid {field}: {value} (must be non-negative integer)")
        
        # Add to history if valid
        if not issues:
            self.test_history.append(metrics)
        
        return len(issues) == 0, issues
    
    def validate_system_metrics(self, metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate system metrics for consistency.
        
        Args:
            metrics: System metrics to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # CPU validation
        if 'cpu' in metrics:
            cpu_data = metrics['cpu']
            if 'usage_percent' in cpu_data:
                cpu_usage = cpu_data['usage_percent']
                if not isinstance(cpu_usage, (int, float)) or cpu_usage < 0 or cpu_usage > 100:
                    issues.append(f"Invalid CPU usage: {cpu_usage}")
            
            if 'core_count' in cpu_data:
                cores = cpu_data['core_count']
                if not isinstance(cores, int) or cores <= 0:
                    issues.append(f"Invalid core count: {cores}")
        
        # Memory validation
        if 'memory' in metrics:
            memory_data = metrics['memory']
            if 'percent' in memory_data:
                mem_percent = memory_data['percent']
                if not isinstance(mem_percent, (int, float)) or mem_percent < 0 or mem_percent > 100:
                    issues.append(f"Invalid memory percent: {mem_percent}")
            
            # Check memory consistency
            if all(field in memory_data for field in ['total_mb', 'used_mb', 'available_mb']):
                total = memory_data['total_mb']
                used = memory_data['used_mb']
                available = memory_data['available_mb']
                
                if abs((used + available) - total) > total * 0.05:  # 5% tolerance
                    issues.append(f"Memory values inconsistent: {used}+{available} != {total}")
        
        # Disk validation
        if 'disk' in metrics:
            disk_data = metrics['disk']
            if 'percent' in disk_data:
                disk_percent = disk_data['percent']
                if not isinstance(disk_percent, (int, float)) or disk_percent < 0 or disk_percent > 100:
                    issues.append(f"Invalid disk percent: {disk_percent}")
        
        # Add to history if valid
        if not issues:
            self.system_history.append(metrics)
        
        return len(issues) == 0, issues
    
    def _detect_performance_anomalies(self, current_metrics: Dict[str, Any]) -> List[str]:
        """
        Detect anomalies in performance metrics based on historical data.
        
        Args:
            current_metrics: Current metrics to check
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if len(self.performance_history) < 10:  # Need sufficient history
            return anomalies
        
        # Extract historical values for comparison
        historical_cpu = [m.get('cpu_usage', 0) for m in self.performance_history]
        historical_memory = [m.get('memory_usage_mb', 0) for m in self.performance_history]
        historical_network = [m.get('network_kb_s', 0) for m in self.performance_history]
        
        # CPU anomaly detection
        if 'cpu_usage' in current_metrics:
            cpu_current = current_metrics['cpu_usage']
            cpu_mean = statistics.mean(historical_cpu)
            cpu_stdev = statistics.stdev(historical_cpu) if len(historical_cpu) > 1 else 0
            
            if cpu_stdev > 0 and abs(cpu_current - cpu_mean) > 3 * cpu_stdev:
                anomalies.append(f"CPU usage anomaly: {cpu_current}% (mean: {cpu_mean:.1f}±{cpu_stdev:.1f})")
        
        # Memory anomaly detection
        if 'memory_usage_mb' in current_metrics:
            memory_current = current_metrics['memory_usage_mb']
            memory_mean = statistics.mean(historical_memory)
            memory_stdev = statistics.stdev(historical_memory) if len(historical_memory) > 1 else 0
            
            if memory_stdev > 0 and abs(memory_current - memory_mean) > 3 * memory_stdev:
                anomalies.append(f"Memory usage anomaly: {memory_current}MB (mean: {memory_mean:.1f}±{memory_stdev:.1f})")
        
        # Network spike detection
        if 'network_kb_s' in current_metrics:
            network_current = current_metrics['network_kb_s']
            network_mean = statistics.mean(historical_network)
            
            # Check for network spikes (10x normal)
            if network_current > max(network_mean * 10, 1000):  # At least 1MB/s spike
                anomalies.append(f"Network spike detected: {network_current:.1f} KB/s (mean: {network_mean:.1f})")
        
        return anomalies
    
    def validate_comprehensive_analytics(self, analytics: Dict[str, Any]) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate a complete analytics payload.
        
        Args:
            analytics: Complete analytics data
            
        Returns:
            Tuple of (is_valid, dict_of_issues_by_category)
        """
        all_issues = {}
        
        # Validate system metrics
        if 'system_metrics' in analytics:
            is_valid, issues = self.validate_system_metrics(analytics['system_metrics'])
            if issues:
                all_issues['system_metrics'] = issues
        
        # Validate test analytics
        if 'test_analytics' in analytics:
            is_valid, issues = self.validate_test_metrics(analytics['test_analytics'])
            if issues:
                all_issues['test_analytics'] = issues
        
        # Validate performance trends
        if 'performance_trends' in analytics:
            trends = analytics['performance_trends']
            issues = []
            
            for trend_name, trend_value in trends.items():
                if trend_name.endswith('_trend') and trend_value not in ['increasing', 'decreasing', 'stable']:
                    issues.append(f"Invalid trend value: {trend_name}={trend_value}")
            
            if issues:
                all_issues['performance_trends'] = issues
        
        # Update validation statistics
        self.validation_stats['total_validations'] += 1
        if all_issues:
            self.validation_stats['failed_validations'] += 1
            total_issues = sum(len(issues) for issues in all_issues.values())
            if any('anomaly' in str(issues).lower() for issues in all_issues.values()):
                self.validation_stats['anomalies_detected'] += 1
        else:
            self.validation_stats['passed_validations'] += 1
        
        # Update data quality score
        if self.validation_stats['total_validations'] > 0:
            pass_rate = self.validation_stats['passed_validations'] / self.validation_stats['total_validations']
            self.validation_stats['data_quality_score'] = pass_rate * 100
        
        return len(all_issues) == 0, all_issues
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive data quality report.
        
        Returns:
            Data quality report with statistics and recommendations
        """
        report = {
            'validation_statistics': self.validation_stats.copy(),
            'data_coverage': {
                'performance_samples': len(self.performance_history),
                'test_samples': len(self.test_history),
                'system_samples': len(self.system_history)
            },
            'recommendations': []
        }
        
        # Generate recommendations
        if self.validation_stats['data_quality_score'] < 95:
            report['recommendations'].append({
                'type': 'quality',
                'message': f"Data quality score is {self.validation_stats['data_quality_score']:.1f}%. Review data sources.",
                'priority': 'medium'
            })
        
        if self.validation_stats['anomalies_detected'] > 0:
            report['recommendations'].append({
                'type': 'anomaly',
                'message': f"{self.validation_stats['anomalies_detected']} anomalies detected. Investigate system behavior.",
                'priority': 'high'
            })
        
        if len(self.performance_history) < self.history_window * 0.5:
            report['recommendations'].append({
                'type': 'coverage',
                'message': "Insufficient performance data for reliable anomaly detection.",
                'priority': 'low'
            })
        
        return report
    
    def reset_validation_stats(self):
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'anomalies_detected': 0,
            'data_quality_score': 100.0
        }
        logger.info("Validation statistics reset")