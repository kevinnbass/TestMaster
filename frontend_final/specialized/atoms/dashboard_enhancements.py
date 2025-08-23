#!/usr/bin/env python3
"""
✨ ATOM: Dashboard Enhancements Component
=========================================
Enhancement features for dashboard functionality.
Part of STEELCLAD atomization - Agent T coordination specialist.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

@dataclass
class UserPreferences:
    """User dashboard preferences"""
    refresh_interval: int = 30
    alert_threshold_cpu: float = 80.0
    alert_threshold_memory_gb: float = 28.0
    dashboard_theme: str = "light"
    show_detailed_metrics: bool = True
    auto_backup_alerts: bool = True
    growth_prediction_days: int = 30
    notification_enabled: bool = True
    export_format: str = "json"
    timezone: str = "UTC"

@dataclass
class PerformanceBaseline:
    """Performance baseline metrics"""
    metric_name: str
    baseline_value: float
    threshold_warning: float
    threshold_critical: float
    measurement_window: timedelta
    last_updated: datetime

class DashboardEnhancements:
    """Dashboard enhancement features"""
    
    def __init__(self):
        self.user_preferences = UserPreferences()
        self.performance_baselines: Dict[str, PerformanceBaseline] = {}
        self.custom_widgets = {}
        self.saved_layouts = {}
        self.enhancement_config = self._initialize_config()
    
    def _initialize_config(self) -> Dict[str, Any]:
        """Initialize enhancement configuration"""
        return {
            'features': {
                'predictive_analytics': True,
                'anomaly_detection': True,
                'custom_dashboards': True,
                'export_automation': True,
                'collaborative_mode': False
            },
            'performance': {
                'cache_enabled': True,
                'cache_ttl': 300,  # 5 minutes
                'lazy_loading': True,
                'virtualization': True
            },
            'ui_enhancements': {
                'dark_mode': True,
                'responsive_design': True,
                'accessibility': True,
                'keyboard_shortcuts': True
            }
        }
    
    def apply_user_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Apply user preferences to dashboard"""
        # Update preferences
        for key, value in preferences.items():
            if hasattr(self.user_preferences, key):
                setattr(self.user_preferences, key, value)
        
        return {
            'status': 'applied',
            'preferences': {
                'theme': self.user_preferences.dashboard_theme,
                'refresh_rate': self.user_preferences.refresh_interval * 1000,
                'alerts': {
                    'cpu_threshold': self.user_preferences.alert_threshold_cpu,
                    'memory_threshold': self.user_preferences.alert_threshold_memory_gb,
                    'enabled': self.user_preferences.notification_enabled
                },
                'display': {
                    'detailed_metrics': self.user_preferences.show_detailed_metrics,
                    'timezone': self.user_preferences.timezone,
                    'export_format': self.user_preferences.export_format
                }
            }
        }
    
    def calculate_performance_baselines(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, PerformanceBaseline]:
        """Calculate performance baselines from historical data"""
        baselines = {}
        
        if not metrics_history:
            return baselines
        
        # Group metrics by type
        metric_groups = {}
        for entry in metrics_history:
            for metric_name, value in entry.items():
                if isinstance(value, (int, float)):
                    if metric_name not in metric_groups:
                        metric_groups[metric_name] = []
                    metric_groups[metric_name].append(value)
        
        # Calculate baselines
        for metric_name, values in metric_groups.items():
            if values:
                avg_value = sum(values) / len(values)
                std_dev = self._calculate_std_dev(values, avg_value)
                
                baselines[metric_name] = PerformanceBaseline(
                    metric_name=metric_name,
                    baseline_value=avg_value,
                    threshold_warning=avg_value + std_dev,
                    threshold_critical=avg_value + (2 * std_dev),
                    measurement_window=timedelta(hours=24),
                    last_updated=datetime.utcnow()
                )
        
        self.performance_baselines = baselines
        return baselines
    
    def detect_anomalies(self, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect anomalies in current metrics"""
        anomalies = []
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in self.performance_baselines:
                baseline = self.performance_baselines[metric_name]
                
                deviation = abs(current_value - baseline.baseline_value)
                severity = self._calculate_anomaly_severity(
                    current_value,
                    baseline
                )
                
                if severity != 'normal':
                    anomalies.append({
                        'metric': metric_name,
                        'current_value': current_value,
                        'baseline_value': baseline.baseline_value,
                        'deviation': deviation,
                        'severity': severity,
                        'timestamp': datetime.utcnow().isoformat()
                    })
        
        return anomalies
    
    def generate_predictive_insights(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate predictive insights from historical data"""
        if len(metrics_history) < 10:
            return {'status': 'insufficient_data'}
        
        predictions = {}
        trends = {}
        
        # Analyze trends for each metric
        metric_names = set()
        for entry in metrics_history:
            metric_names.update(entry.keys())
        
        for metric_name in metric_names:
            values = [
                entry.get(metric_name, 0) 
                for entry in metrics_history 
                if isinstance(entry.get(metric_name), (int, float))
            ]
            
            if len(values) >= 5:
                trend = self._calculate_trend(values)
                prediction = self._predict_future_value(values, self.user_preferences.growth_prediction_days)
                
                trends[metric_name] = trend
                predictions[metric_name] = prediction
        
        return {
            'status': 'success',
            'trends': trends,
            'predictions': predictions,
            'confidence': self._calculate_prediction_confidence(metrics_history),
            'insights': self._generate_insight_messages(trends, predictions)
        }
    
    def create_custom_widget(self, widget_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom dashboard widget"""
        widget_id = f"widget_{datetime.utcnow().timestamp()}"
        
        widget = {
            'id': widget_id,
            'type': widget_config.get('type', 'custom'),
            'title': widget_config.get('title', 'Custom Widget'),
            'data_source': widget_config.get('data_source'),
            'visualization': widget_config.get('visualization', 'text'),
            'refresh_interval': widget_config.get('refresh_interval', 30000),
            'position': widget_config.get('position', {'x': 0, 'y': 0}),
            'size': widget_config.get('size', {'width': 300, 'height': 200}),
            'config': widget_config.get('config', {})
        }
        
        self.custom_widgets[widget_id] = widget
        return widget
    
    def save_dashboard_layout(self, layout_name: str, layout_config: Dict[str, Any]):
        """Save dashboard layout configuration"""
        self.saved_layouts[layout_name] = {
            'name': layout_name,
            'config': layout_config,
            'widgets': list(self.custom_widgets.keys()),
            'created': datetime.utcnow().isoformat(),
            'preferences': self.user_preferences.__dict__.copy()
        }
    
    def load_dashboard_layout(self, layout_name: str) -> Optional[Dict[str, Any]]:
        """Load saved dashboard layout"""
        if layout_name in self.saved_layouts:
            return self.saved_layouts[layout_name]
        return None
    
    def export_dashboard_data(self, data: Dict[str, Any], format: str = None) -> Any:
        """Export dashboard data in specified format"""
        format = format or self.user_preferences.export_format
        
        if format == 'json':
            return json.dumps(data, indent=2, default=str)
        elif format == 'csv':
            return self._convert_to_csv(data)
        elif format == 'html':
            return self._generate_html_report(data)
        else:
            return data
    
    def enable_keyboard_shortcuts(self) -> Dict[str, str]:
        """Get keyboard shortcut configuration"""
        return {
            'refresh': 'Ctrl+R',
            'toggle_theme': 'Ctrl+T',
            'export': 'Ctrl+E',
            'search': 'Ctrl+F',
            'settings': 'Ctrl+,',
            'fullscreen': 'F11',
            'help': 'F1',
            'navigate_next': 'Tab',
            'navigate_prev': 'Shift+Tab',
            'zoom_in': 'Ctrl++',
            'zoom_out': 'Ctrl+-'
        }
    
    def _calculate_std_dev(self, values: List[float], mean: float) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_anomaly_severity(self, value: float, baseline: PerformanceBaseline) -> str:
        """Calculate anomaly severity"""
        if value >= baseline.threshold_critical:
            return 'critical'
        elif value >= baseline.threshold_warning:
            return 'warning'
        else:
            return 'normal'
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from values"""
        if len(values) < 2:
            return 'stable'
        
        recent_avg = sum(values[-5:]) / len(values[-5:])
        older_avg = sum(values[:-5]) / max(1, len(values[:-5]))
        
        change_percent = ((recent_avg - older_avg) / max(older_avg, 1)) * 100
        
        if change_percent > 10:
            return 'increasing'
        elif change_percent < -10:
            return 'decreasing'
        else:
            return 'stable'
    
    def _predict_future_value(self, values: List[float], days_ahead: int) -> float:
        """Simple linear prediction"""
        if len(values) < 2:
            return values[-1] if values else 0
        
        # Simple moving average prediction
        recent_values = values[-min(7, len(values)):]
        avg_change = (recent_values[-1] - recent_values[0]) / len(recent_values)
        predicted = recent_values[-1] + (avg_change * days_ahead)
        
        return max(0, predicted)  # Ensure non-negative
    
    def _calculate_prediction_confidence(self, history: List[Dict[str, Any]]) -> float:
        """Calculate prediction confidence based on data quality"""
        if len(history) < 10:
            return 0.3
        elif len(history) < 30:
            return 0.6
        elif len(history) < 100:
            return 0.8
        else:
            return 0.9
    
    def _generate_insight_messages(self, trends: Dict[str, str], predictions: Dict[str, float]) -> List[str]:
        """Generate human-readable insight messages"""
        insights = []
        
        for metric, trend in trends.items():
            if trend == 'increasing':
                insights.append(f"{metric} is trending upward")
            elif trend == 'decreasing':
                insights.append(f"{metric} is trending downward")
        
        # Add prediction insights
        for metric, value in predictions.items():
            if metric in self.performance_baselines:
                baseline = self.performance_baselines[metric]
                if value > baseline.threshold_warning:
                    insights.append(f"{metric} may exceed warning threshold in {self.user_preferences.growth_prediction_days} days")
        
        return insights
    
    def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """Convert data to CSV format"""
        # Simplified CSV conversion
        lines = []
        for key, value in data.items():
            lines.append(f"{key},{value}")
        return '\n'.join(lines)
    
    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML report"""
        # Simplified HTML generation
        html = "<html><body><h1>Dashboard Report</h1><ul>"
        for key, value in data.items():
            html += f"<li><b>{key}:</b> {value}</li>"
        html += "</ul></body></html>"
        return html