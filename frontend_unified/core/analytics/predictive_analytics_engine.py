#!/usr/bin/env python3
"""
🏗️ MODULE: Predictive Analytics Engine - Extracted from Advanced Gamma Dashboard
============================================================================

📋 PURPOSE:
    Advanced predictive analytics and insights engine providing trend analysis,
    anomaly detection, performance forecasting, and usage optimization.

🎯 CORE FUNCTIONALITY:
    • Comprehensive analytics data collection and processing
    • Trend analysis with direction, magnitude, and confidence metrics
    • Predictive modeling for performance forecasting
    • Anomaly detection with configurable thresholds
    • Intelligent recommendations generation

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 [2025-08-23] | Agent Z | 🔧 STEELCLAD
   └─ Goal: Extract predictive analytics from advanced_gamma_dashboard.py
   └─ Changes: Modularized analytics engine with 80 lines of focused functionality
   └─ Impact: Reduces main dashboard size while maintaining full analytics capability

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent Z (STEELCLAD extraction)
🔧 Language: Python
📦 Dependencies: psutil, numpy, collections
🎯 Integration Points: AdvancedDashboardEngine class
⚡ Performance Notes: Optimized for real-time analytics with deque-based history
🔒 Security Notes: Safe system metrics collection with error handling
"""

import os
import psutil
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List, Any, Optional

class TrendPredictor:
    """Trend prediction model for metrics analysis."""
    
    def predict_trend(self, values: List[float], periods: int = 5) -> Dict[str, Any]:
        """Predict future trend based on historical values."""
        if len(values) < 3:
            return {"predictions": [], "confidence": 0.0}
        
        # Simple linear extrapolation
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        
        future_x = np.arange(len(values), len(values) + periods)
        predictions = np.polyval(coeffs, future_x)
        
        return {
            "predictions": predictions.tolist(),
            "confidence": min(1.0, abs(coeffs[0]) * 10),  # Normalize confidence
            "trend_coefficient": coeffs[0]
        }

class AnomalyDetector:
    """Anomaly detection for system metrics."""
    
    def __init__(self, z_threshold: float = 2.5):
        self.z_threshold = z_threshold
    
    def detect_anomalies(self, values: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies using z-score method."""
        if len(values) < 5:
            return []
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return []
        
        anomalies = []
        for i, value in enumerate(values):
            z_score = abs((value - mean_val) / std_val)
            if z_score > self.z_threshold:
                anomalies.append({
                    "index": i,
                    "value": value,
                    "z_score": z_score,
                    "severity": "high" if z_score > 3.0 else "medium"
                })
        
        return anomalies

class PerformanceForecaster:
    """Performance forecasting model."""
    
    def forecast_performance(self, metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Forecast system performance based on current metrics."""
        forecasts = {}
        
        for metric_name, values in metrics.items():
            if len(values) >= 5:
                # Simple moving average forecast
                recent_avg = np.mean(values[-5:])
                trend_slope = np.polyfit(range(len(values)), values, 1)[0]
                
                forecast = recent_avg + (trend_slope * 5)  # 5 periods ahead
                
                forecasts[metric_name] = {
                    "current_avg": recent_avg,
                    "forecast": forecast,
                    "trend_slope": trend_slope,
                    "confidence": min(1.0, 1 / (1 + abs(trend_slope)))
                }
        
        return forecasts

class UsageOptimizer:
    """Usage optimization recommendations."""
    
    def generate_optimization_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        
        if metrics.get('cpu_usage', 0) > 80:
            recommendations.append("High CPU usage detected. Consider closing unnecessary applications.")
        
        if metrics.get('memory_usage', 0) > 85:
            recommendations.append("High memory usage detected. Consider restarting memory-intensive applications.")
        
        if metrics.get('disk_usage', 0) > 90:
            recommendations.append("Disk space critically low. Consider cleaning temporary files.")
        
        if len(recommendations) == 0:
            recommendations.append("System performance is optimal.")
        
        return recommendations

class PredictiveAnalyticsEngine:
    """Advanced predictive analytics and insights engine."""
    
    def __init__(self, anomaly_threshold: float = 2.5, max_history: int = 1000):
        self.models = self.initialize_models()
        self.metrics_history = deque(maxlen=max_history)
        self.anomaly_threshold = anomaly_threshold
        
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize predictive models."""
        return {
            "trend_predictor": TrendPredictor(),
            "anomaly_detector": AnomalyDetector(self.anomaly_threshold),
            "performance_forecaster": PerformanceForecaster(),
            "usage_optimizer": UsageOptimizer()
        }
    
    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics data."""
        current_metrics = self.collect_current_metrics()
        self.metrics_history.append(current_metrics)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": current_metrics,
            "trends": self.analyze_trends(),
            "predictions": self.generate_predictions(),
            "anomalies": self.detect_anomalies(),
            "recommendations": self.generate_recommendations()
        }
    
    def collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            return {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
                "network_io": psutil.net_io_counters()._asdict(),
                "process_count": len(psutil.pids()),
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            return {
                "cpu_usage": 0,
                "memory_usage": 0,
                "disk_usage": 0,
                "network_io": {},
                "process_count": 0,
                "load_average": [0, 0, 0]
            }
    
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze current trends in metrics."""
        if len(self.metrics_history) < 10:
            return {"status": "insufficient_data"}
        
        recent_metrics = list(self.metrics_history)[-10:]
        trends = {}
        
        for metric in ['cpu_usage', 'memory_usage', 'disk_usage']:
            values = [m[metric] for m in recent_metrics]
            trend = self.calculate_trend(values)
            trends[metric] = {
                "direction": trend["direction"],
                "magnitude": trend["magnitude"],
                "confidence": trend["confidence"]
            }
        
        return trends
    
    def calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend direction and magnitude."""
        if len(values) < 2:
            return {"direction": "stable", "magnitude": 0, "confidence": 0}
        
        # Simple linear regression slope
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        direction = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
        magnitude = abs(slope)
        confidence = min(1.0, magnitude / 5.0)  # Normalize confidence
        
        return {
            "direction": direction,
            "magnitude": magnitude,
            "confidence": confidence
        }
    
    def generate_predictions(self) -> Dict[str, Any]:
        """Generate predictions using all available models."""
        if len(self.metrics_history) < 5:
            return {"status": "insufficient_data"}
        
        # Prepare metrics data for forecasting
        metrics_data = {}
        for metric in ['cpu_usage', 'memory_usage', 'disk_usage']:
            metrics_data[metric] = [m[metric] for m in self.metrics_history]
        
        return self.models["performance_forecaster"].forecast_performance(metrics_data)
    
    def detect_anomalies(self) -> Dict[str, List[Dict[str, Any]]]:
        """Detect anomalies in current metrics."""
        if len(self.metrics_history) < 10:
            return {"status": "insufficient_data"}
        
        anomalies = {}
        for metric in ['cpu_usage', 'memory_usage', 'disk_usage']:
            values = [m[metric] for m in self.metrics_history]
            anomalies[metric] = self.models["anomaly_detector"].detect_anomalies(values)
        
        return anomalies
    
    def generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        if not self.metrics_history:
            return ["No data available for recommendations"]
        
        current_metrics = self.metrics_history[-1]
        return self.models["usage_optimizer"].generate_optimization_recommendations(current_metrics)
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get a summary of analytics status."""
        return {
            "total_metrics_collected": len(self.metrics_history),
            "models_available": list(self.models.keys()),
            "anomaly_threshold": self.anomaly_threshold,
            "history_capacity": self.metrics_history.maxlen,
            "analytics_engine_status": "operational"
        }

def create_analytics_engine(anomaly_threshold: float = 2.5, max_history: int = 1000) -> PredictiveAnalyticsEngine:
    """Factory function to create a configured analytics engine."""
    return PredictiveAnalyticsEngine(anomaly_threshold=anomaly_threshold, max_history=max_history)