#!/usr/bin/env python3
"""
🏗️ MODULE: Performance Enhancement Classes - Dashboard Support
==================================================================

📋 PURPOSE:
    Performance monitoring, API tracking, and data integration classes
    extracted from unified_gamma_dashboard_enhanced.py via STEELCLAD protocol.

🎯 CORE FUNCTIONALITY:
    • APIUsageTracker - Enhanced API usage tracking with Agent E integration
    • DataIntegrator - Enhanced data integration with personal analytics support  
    • PerformanceMonitor - Performance monitoring for enhanced dashboard

🔄 EXTRACTION HISTORY:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 STEELCLAD EXTRACTION  
   └─ Goal: Extract performance classes from unified_gamma_dashboard_enhanced.py
   └─ Source: Lines 442-555 (113 lines)
   └─ Purpose: Modularize support classes while preserving 100% functionality

📞 DEPENDENCIES:
==================================================================
🤝 Imports: Standard library (datetime, collections)
📤 Provides: APIUsageTracker, DataIntegrator, PerformanceMonitor classes
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque


class APIUsageTracker:
    """Enhanced API usage tracking with Agent E integration support."""
    
    def __init__(self):
        self.api_calls = defaultdict(int)
        self.daily_budget = 50.0
        self.daily_spending = 0.0
        self.last_reset = datetime.now().date()
        self.personal_analytics_calls = 0
    
    def track_api_call(self, endpoint: str, purpose: str = "dashboard"):
        """Track API calls including personal analytics."""
        current_time = datetime.now()
        
        # Reset daily spending if new day
        if current_time.date() > self.last_reset:
            self.daily_spending = 0.0
            self.personal_analytics_calls = 0
            self.last_reset = current_time.date()
        
        self.api_calls[endpoint] += 1
        
        # Track personal analytics calls separately
        if 'personal' in endpoint.lower():
            self.personal_analytics_calls += 1
        
        return {
            'endpoint': endpoint,
            'purpose': purpose,
            'timestamp': current_time.isoformat(),
            'daily_total': sum(self.api_calls.values()),
            'personal_analytics_calls': self.personal_analytics_calls
        }
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get API usage summary."""
        return {
            'total_calls': sum(self.api_calls.values()),
            'personal_analytics_calls': self.personal_analytics_calls,
            'daily_budget': self.daily_budget,
            'daily_spending': self.daily_spending,
            'endpoints': dict(self.api_calls)
        }


class DataIntegrator:
    """Enhanced data integration with personal analytics support."""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 60  # seconds
    
    def integrate_data(self, sources: List[Dict]) -> Dict[str, Any]:
        """Integrate data from multiple sources including personal analytics."""
        integrated = {
            'timestamp': datetime.now().isoformat(),
            'sources': len(sources),
            'data': {}
        }
        
        for source in sources:
            source_name = source.get('name', 'unknown')
            integrated['data'][source_name] = source.get('data', {})
        
        # Special handling for personal analytics data
        if 'personal_analytics' in integrated['data']:
            integrated['has_personal_insights'] = True
            integrated['quality_score'] = integrated['data']['personal_analytics'].get(
                'quality_metrics', {}
            ).get('overall_score', 0)
        
        return integrated
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data if still valid."""
        if key in self.cache:
            cached_time, data = self.cache[key]
            if (datetime.now() - cached_time).seconds < self.cache_timeout:
                return data
        return None
    
    def set_cached_data(self, key: str, data: Any):
        """Cache data with timestamp."""
        self.cache[key] = (datetime.now(), data)


class PerformanceMonitor:
    """Performance monitoring for enhanced dashboard."""
    
    def __init__(self):
        self.metrics = deque(maxlen=100)
        self.target_response_time = 100  # ms
        self.target_fps = 60
    
    def record_metric(self, metric_type: str, value: float):
        """Record a performance metric."""
        self.metrics.append({
            'type': metric_type,
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'meets_target': self._check_target(metric_type, value)
        })
    
    def _check_target(self, metric_type: str, value: float) -> bool:
        """Check if metric meets performance target."""
        if metric_type == 'response_time':
            return value <= self.target_response_time
        elif metric_type == 'fps':
            return value >= self.target_fps
        return True
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        recent_metrics = list(self.metrics)[-10:] if self.metrics else []
        
        return {
            'average_response_time': self._calculate_average('response_time'),
            'average_fps': self._calculate_average('fps'),
            'target_compliance': self._calculate_compliance(),
            'recent_metrics': recent_metrics
        }
    
    def _calculate_average(self, metric_type: str) -> float:
        """Calculate average for a metric type."""
        values = [m['value'] for m in self.metrics if m['type'] == metric_type]
        return sum(values) / len(values) if values else 0
    
    def _calculate_compliance(self) -> float:
        """Calculate target compliance percentage."""
        if not self.metrics:
            return 100.0
        
        compliant = sum(1 for m in self.metrics if m['meets_target'])
        return (compliant / len(self.metrics)) * 100
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics for dashboard display."""
        return {
            'response_times': [m for m in self.metrics if m['type'] == 'response_time'],
            'fps_metrics': [m for m in self.metrics if m['type'] == 'fps'],
            'compliance_rate': self._calculate_compliance(),
            'total_metrics': len(self.metrics)
        }