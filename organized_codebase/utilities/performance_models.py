#!/usr/bin/env python3
"""
🏗️ MODULE: Performance Models - Data Structures and Type Definitions
==================================================================

📋 PURPOSE:
    Data classes and type definitions for performance validation framework.
    Contains all performance test configurations, results, and scenarios.

🎯 CORE FUNCTIONALITY:
    • PerformanceTest configuration and criteria management
    • PerformanceResult data structure for test execution results  
    • LoadTestScenario definition for complex load testing

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 07:10:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract data models from performance_validation_framework.py via STEELCLAD
   └─ Changes: Created dedicated module for performance data structures
   └─ Impact: Improved modularity and testability of performance system components

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: typing, datetime, dataclasses
🎯 Integration Points: performance_benchmarker.py, load_test_executor.py
⚡ Performance Notes: Lightweight data structures for performance testing
🔒 Security Notes: No security-sensitive operations

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: N/A | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: N/A | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: Standard library only
📤 Provides: Data structures for performance validation system
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass


@dataclass
class PerformanceTest:
    """Individual performance test definition"""
    test_id: str
    test_name: str
    test_type: str  # 'load', 'stress', 'volume', 'spike', 'endurance'
    target_function: Optional[Callable] = None
    target_url: Optional[str] = None
    expected_response_time_ms: float = 100.0
    max_response_time_ms: float = 1000.0
    concurrent_users: int = 1
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    success_criteria: Dict[str, float] = None
    
    def __post_init__(self):
        if self.success_criteria is None:
            self.success_criteria = {
                'avg_response_time_ms': self.expected_response_time_ms,
                'max_response_time_ms': self.max_response_time_ms,
                'success_rate_percent': 95.0,
                'throughput_requests_per_second': 10.0
            }


@dataclass
class PerformanceResult:
    """Performance test execution result"""
    test_id: str
    execution_timestamp: datetime
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    throughput_rps: float
    error_rate_percent: float
    memory_usage_mb: float
    cpu_usage_percent: float
    errors: List[str] = None
    success: bool = True
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class LoadTestScenario:
    """Load test scenario configuration"""
    name: str
    base_url: str
    endpoints: List[Dict[str, Any]]
    user_scenarios: List[Dict[str, Any]]
    ramp_up_pattern: str  # 'linear', 'exponential', 'step'
    max_concurrent_users: int
    test_duration_minutes: int
    think_time_seconds: Tuple[float, float] = (0.5, 2.0)  # min, max