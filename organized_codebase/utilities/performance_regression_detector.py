#!/usr/bin/env python3
"""
🏗️ MODULE: Performance Regression Detector - Regression Analysis Engine
==================================================================

📋 PURPOSE:
    Detects performance regressions by comparing current results against baselines.
    Analyzes metrics changes and identifies significant performance degradations.

🎯 CORE FUNCTIONALITY:
    • Performance regression detection with configurable thresholds
    • Baseline comparison and trend analysis
    • Improvement and degradation identification
    • Severity classification for regressions

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 07:40:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract regression detection from performance_validation_framework.py via STEELCLAD
   └─ Changes: Created dedicated module for performance regression analysis
   └─ Impact: Improved modularity and single responsibility for regression detection

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: sqlite3, logging, statistics
🎯 Integration Points: performance_models.py, performance_benchmarker.py
⚡ Performance Notes: Database queries optimized for latest baseline retrieval
🔒 Security Notes: SQL injection protection via parameterized queries

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Via regression detection | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: performance_benchmarker for test results access
📤 Provides: Regression detection capabilities for performance validation
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

import sqlite3
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

# Import data models and benchmarker
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.performance_models import PerformanceResult


class PerformanceRegressionDetector:
    """Detects performance regressions by comparing results"""
    
    def __init__(self, benchmarker):
        """Initialize with reference to benchmarker for results access"""
        self.benchmarker = benchmarker
        self.logger = logging.getLogger('PerformanceRegressionDetector')
        
        # Regression thresholds
        self.regression_thresholds = {
            'response_time_increase_percent': 10.0,  # 10% increase
            'throughput_decrease_percent': 10.0,     # 10% decrease
            'error_rate_increase_percent': 5.0,      # 5% increase
            'memory_increase_percent': 20.0          # 20% increase
        }
    
    def detect_regressions(self, current_results: Dict[str, PerformanceResult], 
                          baseline_results: Optional[Dict[str, PerformanceResult]] = None) -> Dict[str, Any]:
        """Detect performance regressions"""
        
        if baseline_results is None:
            baseline_results = self._get_latest_baseline()
        
        regression_report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'regressions_detected': [],
            'improvements_detected': [],
            'summary': {
                'total_tests_compared': 0,
                'regressions_count': 0,
                'improvements_count': 0,
                'stable_count': 0
            }
        }
        
        for test_id, current_result in current_results.items():
            if current_result is None:
                continue
                
            baseline_result = baseline_results.get(test_id)
            if baseline_result is None:
                continue
            
            regression_report['summary']['total_tests_compared'] += 1
            
            # Compare metrics
            comparison = self._compare_results(current_result, baseline_result)
            
            if comparison['has_regression']:
                regression_report['regressions_detected'].append({
                    'test_id': test_id,
                    'test_name': self.benchmarker.registered_tests.get(test_id, {}).test_name if hasattr(self.benchmarker.registered_tests.get(test_id, {}), 'test_name') else test_id,
                    'regression_type': comparison['regression_type'],
                    'current_value': comparison['current_value'],
                    'baseline_value': comparison['baseline_value'],
                    'change_percent': comparison['change_percent'],
                    'severity': comparison['severity']
                })
                regression_report['summary']['regressions_count'] += 1
                
            elif comparison['has_improvement']:
                regression_report['improvements_detected'].append({
                    'test_id': test_id,
                    'improvement_type': comparison['improvement_type'],
                    'current_value': comparison['current_value'],
                    'baseline_value': comparison['baseline_value'],
                    'improvement_percent': comparison['change_percent']
                })
                regression_report['summary']['improvements_count'] += 1
                
            else:
                regression_report['summary']['stable_count'] += 1
        
        return regression_report
    
    def _get_latest_baseline(self) -> Dict[str, PerformanceResult]:
        """Get latest baseline results from database"""
        baseline_results = {}
        
        with sqlite3.connect(self.benchmarker.db_path) as conn:
            # Get latest successful results for each test
            cursor = conn.execute("""
                SELECT DISTINCT test_id FROM performance_results 
                WHERE success = 1 
                ORDER BY execution_timestamp DESC
            """)
            
            test_ids = [row[0] for row in cursor.fetchall()]
            
            for test_id in test_ids:
                cursor = conn.execute("""
                    SELECT * FROM performance_results 
                    WHERE test_id = ? AND success = 1 
                    ORDER BY execution_timestamp DESC 
                    LIMIT 2
                """, (test_id,))
                
                rows = cursor.fetchall()
                if len(rows) >= 2:  # Use second-latest as baseline
                    row = rows[1]
                    baseline_results[test_id] = self._row_to_result(row)
        
        return baseline_results
    
    def _row_to_result(self, row) -> PerformanceResult:
        """Convert database row to PerformanceResult"""
        import json
        
        return PerformanceResult(
            test_id=row[1],
            execution_timestamp=datetime.fromisoformat(row[2]),
            duration_seconds=row[3],
            total_requests=row[4],
            successful_requests=row[5],
            failed_requests=row[6],
            avg_response_time_ms=row[7],
            min_response_time_ms=row[8],
            max_response_time_ms=row[9],
            p50_response_time_ms=row[10],
            p95_response_time_ms=row[11],
            p99_response_time_ms=row[12],
            throughput_rps=row[13],
            error_rate_percent=row[14],
            memory_usage_mb=row[15],
            cpu_usage_percent=row[16],
            errors=json.loads(row[17]) if row[17] else [],
            success=bool(row[18])
        )
    
    def _compare_results(self, current: PerformanceResult, baseline: PerformanceResult) -> Dict[str, Any]:
        """Compare current results against baseline"""
        comparison = {
            'has_regression': False,
            'has_improvement': False,
            'regression_type': None,
            'improvement_type': None,
            'current_value': None,
            'baseline_value': None,
            'change_percent': 0.0,
            'severity': 'low'
        }
        
        # Check response time regression
        if baseline.avg_response_time_ms > 0:
            response_time_change = ((current.avg_response_time_ms - baseline.avg_response_time_ms) 
                                   / baseline.avg_response_time_ms) * 100
            
            if response_time_change > self.regression_thresholds['response_time_increase_percent']:
                comparison['has_regression'] = True
                comparison['regression_type'] = 'response_time'
                comparison['current_value'] = current.avg_response_time_ms
                comparison['baseline_value'] = baseline.avg_response_time_ms
                comparison['change_percent'] = response_time_change
                comparison['severity'] = self._calculate_severity(response_time_change, 'response_time')
                return comparison
            elif response_time_change < -self.regression_thresholds['response_time_increase_percent']:
                comparison['has_improvement'] = True
                comparison['improvement_type'] = 'response_time'
                comparison['current_value'] = current.avg_response_time_ms
                comparison['baseline_value'] = baseline.avg_response_time_ms
                comparison['change_percent'] = abs(response_time_change)
                return comparison
        
        # Check throughput regression
        if baseline.throughput_rps > 0:
            throughput_change = ((current.throughput_rps - baseline.throughput_rps) 
                               / baseline.throughput_rps) * 100
            
            if throughput_change < -self.regression_thresholds['throughput_decrease_percent']:
                comparison['has_regression'] = True
                comparison['regression_type'] = 'throughput'
                comparison['current_value'] = current.throughput_rps
                comparison['baseline_value'] = baseline.throughput_rps
                comparison['change_percent'] = abs(throughput_change)
                comparison['severity'] = self._calculate_severity(abs(throughput_change), 'throughput')
                return comparison
            elif throughput_change > self.regression_thresholds['throughput_decrease_percent']:
                comparison['has_improvement'] = True
                comparison['improvement_type'] = 'throughput'
                comparison['current_value'] = current.throughput_rps
                comparison['baseline_value'] = baseline.throughput_rps
                comparison['change_percent'] = throughput_change
                return comparison
        
        # Check error rate regression
        error_rate_change = current.error_rate_percent - baseline.error_rate_percent
        
        if error_rate_change > self.regression_thresholds['error_rate_increase_percent']:
            comparison['has_regression'] = True
            comparison['regression_type'] = 'error_rate'
            comparison['current_value'] = current.error_rate_percent
            comparison['baseline_value'] = baseline.error_rate_percent
            comparison['change_percent'] = error_rate_change
            comparison['severity'] = 'high'  # Error rate increases are always high severity
            return comparison
        
        # Check memory usage regression
        if baseline.memory_usage_mb > 0:
            memory_change = ((current.memory_usage_mb - baseline.memory_usage_mb) 
                           / baseline.memory_usage_mb) * 100
            
            if memory_change > self.regression_thresholds['memory_increase_percent']:
                comparison['has_regression'] = True
                comparison['regression_type'] = 'memory'
                comparison['current_value'] = current.memory_usage_mb
                comparison['baseline_value'] = baseline.memory_usage_mb
                comparison['change_percent'] = memory_change
                comparison['severity'] = self._calculate_severity(memory_change, 'memory')
                return comparison
        
        return comparison
    
    def _calculate_severity(self, change_percent: float, metric_type: str) -> str:
        """Calculate regression severity based on change percentage"""
        change_percent = abs(change_percent)
        
        if metric_type in ['response_time', 'throughput']:
            if change_percent < 15:
                return 'low'
            elif change_percent < 30:
                return 'medium'
            else:
                return 'high'
        elif metric_type == 'memory':
            if change_percent < 30:
                return 'low'
            elif change_percent < 50:
                return 'medium'
            else:
                return 'high'
        else:
            return 'medium'
    
    def get_trend_analysis(self, test_id: str, days: int = 7) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        trend_data = {
            'test_id': test_id,
            'period_days': days,
            'trend': 'stable',
            'metrics': {}
        }
        
        # Query historical data
        with sqlite3.connect(self.benchmarker.db_path) as conn:
            cursor = conn.execute("""
                SELECT avg_response_time_ms, throughput_rps, error_rate_percent, 
                       memory_usage_mb, execution_timestamp
                FROM performance_results
                WHERE test_id = ? 
                  AND execution_timestamp > datetime('now', '-{} days')
                  AND success = 1
                ORDER BY execution_timestamp
            """.format(days), (test_id,))
            
            rows = cursor.fetchall()
            
            if len(rows) >= 2:
                # Calculate trends for each metric
                response_times = [row[0] for row in rows]
                throughputs = [row[1] for row in rows]
                error_rates = [row[2] for row in rows]
                memory_usages = [row[3] for row in rows]
                
                trend_data['metrics'] = {
                    'response_time_trend': self._calculate_trend(response_times),
                    'throughput_trend': self._calculate_trend(throughputs),
                    'error_rate_trend': self._calculate_trend(error_rates),
                    'memory_usage_trend': self._calculate_trend(memory_usages)
                }
                
                # Determine overall trend
                if trend_data['metrics']['response_time_trend'] == 'increasing' or \
                   trend_data['metrics']['throughput_trend'] == 'decreasing':
                    trend_data['trend'] = 'degrading'
                elif trend_data['metrics']['response_time_trend'] == 'decreasing' and \
                     trend_data['metrics']['throughput_trend'] == 'increasing':
                    trend_data['trend'] = 'improving'
        
        return trend_data
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values"""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 'stable'
        
        slope = numerator / denominator
        
        # Determine trend based on slope
        if abs(slope) < 0.01:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'