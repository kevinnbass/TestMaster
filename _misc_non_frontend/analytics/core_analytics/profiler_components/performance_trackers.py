#!/usr/bin/env python3
"""
🏗️ MODULE: Performance Trackers - Analysis & Dashboard Data Generation
==================================================================

📋 PURPOSE:
    Performance analysis and dashboard data generation functionality
    extracted from performance_profiler.py via STEELCLAD protocol.

🎯 CORE FUNCTIONALITY:
    • Dashboard performance data formatting
    • Health status calculation and recommendations
    • Performance chart generation for visualization
    • Alert checking and threshold monitoring

🔄 EXTRACTION HISTORY:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 STEELCLAD EXTRACTION
   └─ Goal: Extract performance trackers from performance_profiler.py
   └─ Source: Lines 249-453 (204 lines)
   └─ Purpose: Separate analysis logic into focused module

📞 DEPENDENCIES:
==================================================================
🤝 Imports: datetime, typing, PerformanceMetric
📤 Provides: Performance analysis and dashboard data generation
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List
from .profiling_metrics import PerformanceMetric


class PerformanceTracker:
    """Handles performance analysis and dashboard data generation."""
    
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
    
    def get_dashboard_performance_data(self, metrics_history: List[PerformanceMetric], 
                                     current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get performance data formatted for dashboard display.
        
        Returns:
            Formatted performance data for Gamma dashboard integration
        """
        now = datetime.now()
        last_5_min = now - timedelta(minutes=5)
        
        # Filter recent metrics
        recent_metrics = [
            m for m in metrics_history 
            if m.timestamp >= last_5_min
        ]
        
        # Calculate performance statistics
        api_times = [m.value for m in recent_metrics if m.metric_type == 'api_response_time_ms']
        cpu_values = [m.value for m in recent_metrics if m.metric_type == 'system_cpu_percent']
        memory_values = [m.value for m in recent_metrics if m.metric_type == 'system_memory_percent']
        
        # Performance summary
        performance_summary = {
            'api_performance': {
                'avg_response_ms': round(sum(api_times) / len(api_times), 2) if api_times else 0,
                'max_response_ms': round(max(api_times), 2) if api_times else 0,
                'min_response_ms': round(min(api_times), 2) if api_times else 0,
                'total_requests': len(api_times),
                'fast_requests': len([t for t in api_times if t < 200]),
                'slow_requests': len([t for t in api_times if t > 600])
            },
            'system_performance': {
                'avg_cpu_percent': round(sum(cpu_values) / len(cpu_values), 1) if cpu_values else 0,
                'max_cpu_percent': round(max(cpu_values), 1) if cpu_values else 0,
                'avg_memory_percent': round(sum(memory_values) / len(memory_values), 1) if memory_values else 0,
                'max_memory_percent': round(max(memory_values), 1) if memory_values else 0
            },
            'health_status': self.calculate_health_status(current_metrics),
            'recommendations': self.generate_performance_recommendations(current_metrics, metrics_history)
        }
        
        return {
            'summary': performance_summary,
            'current_metrics': current_metrics.copy(),
            'charts': self.generate_performance_charts(recent_metrics),
            'alerts': self.check_performance_alerts(current_metrics),
            'timestamp': now.isoformat()
        }
    
    def calculate_health_status(self, current_metrics: Dict[str, Any]) -> str:
        """Calculate overall system health status."""
        # Check critical metrics
        cpu = current_metrics.get('system_cpu_percent', 0)
        memory = current_metrics.get('system_memory_percent', 0)
        
        if cpu > 90 or memory > 90:
            return 'critical'
        elif cpu > 70 or memory > 70:
            return 'warning'
        else:
            return 'healthy'
    
    def generate_performance_recommendations(self, current_metrics: Dict[str, Any], 
                                           metrics_history: List[PerformanceMetric]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # CPU recommendations
        cpu = current_metrics.get('system_cpu_percent', 0)
        if cpu > 80:
            recommendations.append("High CPU usage detected - consider optimizing background processes")
        
        # Memory recommendations
        memory = current_metrics.get('system_memory_percent', 0)
        if memory > 80:
            recommendations.append("High memory usage - consider clearing caches or restarting services")
        
        # API performance recommendations
        api_times = [m.value for m in metrics_history[-50:] if m.metric_type == 'api_response_time_ms']
        if api_times and sum(api_times) / len(api_times) > 200:
            recommendations.append("API response times above target - consider cache optimization")
        
        return recommendations[:3]  # Top 3 recommendations
    
    def generate_performance_charts(self, recent_metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Generate chart data for performance visualization."""
        
        # Time series data for the last 5 minutes
        timestamps = []
        cpu_data = []
        memory_data = []
        api_times = []
        
        # Group metrics by minute for cleaner visualization
        minute_groups = {}
        for metric in recent_metrics:
            minute_key = metric.timestamp.replace(second=0, microsecond=0)
            if minute_key not in minute_groups:
                minute_groups[minute_key] = {'cpu': [], 'memory': [], 'api': []}
            
            if metric.metric_type == 'system_cpu_percent':
                minute_groups[minute_key]['cpu'].append(metric.value)
            elif metric.metric_type == 'system_memory_percent':
                minute_groups[minute_key]['memory'].append(metric.value)
            elif metric.metric_type == 'api_response_time_ms':
                minute_groups[minute_key]['api'].append(metric.value)
        
        # Prepare chart data
        sorted_minutes = sorted(minute_groups.keys())
        for minute in sorted_minutes:
            timestamps.append(minute.strftime('%H:%M'))
            
            # Average values for each minute
            group = minute_groups[minute]
            cpu_data.append(round(sum(group['cpu']) / len(group['cpu']), 1) if group['cpu'] else 0)
            memory_data.append(round(sum(group['memory']) / len(group['memory']), 1) if group['memory'] else 0)
            api_times.append(round(sum(group['api']) / len(group['api']), 1) if group['api'] else 0)
        
        return {
            'system_performance_chart': {
                'type': 'line',
                'data': {
                    'labels': timestamps,
                    'datasets': [
                        {
                            'label': 'CPU %',
                            'data': cpu_data,
                            'borderColor': '#ff6b6b',
                            'backgroundColor': 'rgba(255, 107, 107, 0.1)'
                        },
                        {
                            'label': 'Memory %',
                            'data': memory_data,
                            'borderColor': '#4ecdc4',
                            'backgroundColor': 'rgba(78, 205, 196, 0.1)'
                        }
                    ]
                },
                'options': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'scales': {
                        'y': {
                            'beginAtZero': True,
                            'max': 100
                        }
                    }
                }
            },
            'api_performance_chart': {
                'type': 'line',
                'data': {
                    'labels': timestamps,
                    'datasets': [{
                        'label': 'Response Time (ms)',
                        'data': api_times,
                        'borderColor': '#45b7d1',
                        'backgroundColor': 'rgba(69, 183, 209, 0.1)',
                        'borderDash': []
                    }]
                },
                'options': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'plugins': {
                        'legend': {
                            'display': True
                        }
                    }
                }
            }
        }
    
    def check_performance_alerts(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for performance alerts based on thresholds."""
        alerts = []
        
        # Check each threshold
        for metric_key, threshold in self.thresholds.items():
            current_value = current_metrics.get(metric_key.replace('_ms', '').replace('_percent', ''), 0)
            
            if current_value > threshold:
                severity = 'critical' if current_value > threshold * 1.2 else 'warning'
                alerts.append({
                    'metric': metric_key,
                    'current_value': current_value,
                    'threshold': threshold,
                    'severity': severity,
                    'message': f"{metric_key.replace('_', ' ').title()} is {current_value} (threshold: {threshold})"
                })
        
        return alerts
    
    def get_performance_summary(self, metrics_history: List[PerformanceMetric], 
                              current_metrics: Dict[str, Any], 
                              monitoring_active: bool) -> Dict[str, Any]:
        """Get a concise performance summary for logging/debugging."""
        return {
            'metrics_collected': len(metrics_history),
            'monitoring_active': monitoring_active,
            'current_health': self.calculate_health_status(current_metrics),
            'last_update': current_metrics.get('timestamp', 'never')
        }