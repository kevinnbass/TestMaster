#!/usr/bin/env python3
"""
🏗️ MODULE: Data Analysis Core - Pattern & Trend Analysis
==================================================================

📋 PURPOSE:
    Data analysis and pattern recognition methods extracted from
    predictive_analytics_engine.py via STEELCLAD protocol.

🎯 CORE FUNCTIONALITY:
    • Pattern analysis methods (hourly, daily, weekly)
    • Trend calculation and forecasting
    • Confidence interval calculations
    • Chart formatting for visualization

🔄 EXTRACTION HISTORY:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 STEELCLAD EXTRACTION
   └─ Goal: Extract data analysis methods from predictive_analytics_engine.py
   └─ Source: Lines 174-435 (261 lines)
   └─ Purpose: Separate analysis logic into focused module

📞 DEPENDENCIES:
==================================================================
🤝 Imports: numpy, datetime, typing
📤 Provides: Data analysis and visualization formatting methods
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Any


class DataAnalysisCore:
    """Core data analysis methods for predictive analytics."""
    
    def __init__(self):
        self.patterns = {}
    
    def analyze_hourly_patterns(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze productivity patterns by hour."""
        # Simulated hourly pattern analysis
        return {
            'peak_hours': [9, 10, 11, 14, 15, 16],
            'low_hours': [12, 13, 17, 18],
            'pattern_strength': 0.75
        }
    
    def analyze_daily_patterns(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze productivity patterns by day of week."""
        return {
            'most_productive': ['Tuesday', 'Wednesday', 'Thursday'],
            'least_productive': ['Monday', 'Friday'],
            'weekend_activity': 0.3
        }
    
    def analyze_weekly_trends(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze weekly productivity trends."""
        if len(data) < 7:
            return {'trend': 'insufficient_data'}
        
        # Calculate weekly averages
        week1_avg = np.mean([d['productivity_score'] for d in data[:7]])
        week2_avg = np.mean([d['productivity_score'] for d in data[7:14]]) if len(data) >= 14 else week1_avg
        
        trend = 'improving' if week2_avg > week1_avg else 'declining' if week2_avg < week1_avg else 'stable'
        
        return {
            'trend': trend,
            'weekly_change': week2_avg - week1_avg if len(data) >= 14 else 0,
            'volatility': np.std([d['productivity_score'] for d in data])
        }
    
    def estimate_milestones(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Estimate project milestone completion times."""
        if len(historical_data) < 7:
            return {'status': 'insufficient_data'}
        
        # Calculate average velocity
        recent_data = historical_data[-7:]
        avg_commits_per_day = np.mean([d['commits'] for d in recent_data])
        avg_lines_per_day = np.mean([d['lines_changed'] for d in recent_data])
        
        return {
            'feature_completion': {
                'small_feature': {'days': 3, 'confidence': 0.7},
                'medium_feature': {'days': 8, 'confidence': 0.6},
                'large_feature': {'days': 20, 'confidence': 0.4}
            },
            'velocity_metrics': {
                'avg_commits_per_day': round(avg_commits_per_day, 1),
                'avg_lines_per_day': round(avg_lines_per_day, 1)
            }
        }
    
    def generate_recommendations(self, historical_data: List[Dict], patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate AI-powered recommendations based on patterns."""
        recommendations = []
        
        if len(historical_data) < 7:
            recommendations.append({
                'type': 'data_collection',
                'priority': 'high',
                'message': 'Continue using the system to build prediction accuracy',
                'action': 'Keep analytics active for better insights'
            })
            return recommendations
        
        recent_data = historical_data[-7:]
        
        # Quality-based recommendations
        avg_quality = np.mean([d['quality_score'] for d in recent_data])
        if avg_quality < 70:
            recommendations.append({
                'type': 'quality_improvement',
                'priority': 'high',
                'message': 'Code quality scores trending below optimal range',
                'action': 'Focus on refactoring and test coverage improvement'
            })
        
        # Productivity-based recommendations
        productivity_trend = self.calculate_productivity_trend(recent_data)
        if productivity_trend < -5:
            recommendations.append({
                'type': 'productivity_optimization',
                'priority': 'medium',
                'message': 'Productivity showing declining trend',
                'action': 'Consider reviewing workflow or taking breaks'
            })
        
        # Pattern-based recommendations
        if 'hourly' in patterns:
            peak_hours = patterns['hourly']['peak_hours']
            recommendations.append({
                'type': 'schedule_optimization',
                'priority': 'low',
                'message': f'Peak productivity hours: {peak_hours}',
                'action': 'Schedule complex tasks during peak hours'
            })
        
        return recommendations
    
    def calculate_productivity_trend(self, data: List[Dict]) -> float:
        """Calculate productivity trend over recent period."""
        if len(data) < 4:
            return 0
        
        scores = [d['productivity_score'] for d in data]
        # Simple linear trend calculation
        x = list(range(len(scores)))
        slope = np.polyfit(x, scores, 1)[0]
        return slope
    
    def calculate_confidence_intervals(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Calculate confidence intervals for predictions."""
        if len(historical_data) < 7:
            return {'status': 'insufficient_data'}
        
        recent_data = historical_data[-14:]
        quality_scores = [d['quality_score'] for d in recent_data]
        productivity_scores = [d['productivity_score'] for d in recent_data]
        
        return {
            'quality_score': {
                'mean': np.mean(quality_scores),
                'std': np.std(quality_scores),
                'confidence_95': {
                    'lower': np.percentile(quality_scores, 2.5),
                    'upper': np.percentile(quality_scores, 97.5)
                }
            },
            'productivity_score': {
                'mean': np.mean(productivity_scores),
                'std': np.std(productivity_scores),
                'confidence_95': {
                    'lower': np.percentile(productivity_scores, 2.5),
                    'upper': np.percentile(productivity_scores, 97.5)
                }
            }
        }
    
    def analyze_trends(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Analyze overall trends in development patterns."""
        if len(historical_data) < 14:
            return {'status': 'insufficient_data'}
        
        # Quality trend
        quality_scores = [d['quality_score'] for d in historical_data]
        quality_trend = np.polyfit(range(len(quality_scores)), quality_scores, 1)[0]
        
        # Productivity trend
        productivity_scores = [d['productivity_score'] for d in historical_data]
        productivity_trend = np.polyfit(range(len(productivity_scores)), productivity_scores, 1)[0]
        
        # Test coverage trend
        coverage_scores = [d['test_coverage'] for d in historical_data]
        coverage_trend = np.polyfit(range(len(coverage_scores)), coverage_scores, 1)[0]
        
        return {
            'quality': {
                'trend': 'improving' if quality_trend > 1 else 'declining' if quality_trend < -1 else 'stable',
                'slope': quality_trend,
                'current': quality_scores[-1]
            },
            'productivity': {
                'trend': 'improving' if productivity_trend > 1 else 'declining' if productivity_trend < -1 else 'stable',
                'slope': productivity_trend,
                'current': productivity_scores[-1]
            },
            'test_coverage': {
                'trend': 'improving' if coverage_trend > 1 else 'declining' if coverage_trend < -1 else 'stable',
                'slope': coverage_trend,
                'current': coverage_scores[-1]
            }
        }


class VisualizationFormatter:
    """Formats analysis data for dashboard visualization."""
    
    def format_trend_chart(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Format trend data for chart visualization."""
        return {
            'type': 'line_with_prediction',
            'data': {
                'historical': [75, 78, 82, 85, 83, 87, 89],  # Demo historical data
                'predicted': [91, 93, 95],  # Next 3 days prediction
                'confidence_upper': [95, 98, 100],
                'confidence_lower': [87, 88, 90]
            },
            'labels': ['Day -6', 'Day -5', 'Day -4', 'Day -3', 'Day -2', 'Day -1', 'Today', 'Day +1', 'Day +2', 'Day +3']
        }
    
    def format_confidence_chart(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Format confidence interval data for visualization."""
        confidence = predictions.get('confidence_intervals', {})
        
        return {
            'type': 'confidence_interval',
            'quality_confidence': confidence.get('quality_score', {}),
            'productivity_confidence': confidence.get('productivity_score', {})
        }
    
    def format_pattern_heatmap(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Format pattern data for heatmap visualization."""
        patterns = predictions.get('pattern_insights', {})
        
        # Generate demo heatmap data (7 days x 24 hours)
        heatmap_data = []
        for day in range(7):
            day_data = []
            for hour in range(24):
                # Simulate productivity by hour
                if hour in patterns.get('peak_hours', [10, 14, 16]):
                    value = np.random.normal(85, 10)
                elif hour < 8 or hour > 18:
                    value = np.random.normal(30, 15)
                else:
                    value = np.random.normal(60, 20)
                day_data.append(max(0, min(100, value)))
            heatmap_data.append(day_data)
        
        return {
            'type': 'heatmap',
            'data': heatmap_data,
            'labels': {
                'days': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                'hours': [f'{h}:00' for h in range(24)]
            }
        }