#!/usr/bin/env python3
"""
🏗️ MODULE: Prediction Algorithms - ML-Powered Prediction Classes
==================================================================

📋 PURPOSE:
    Machine learning prediction algorithms extracted from
    predictive_analytics_engine.py via STEELCLAD protocol.

🎯 CORE FUNCTIONALITY:
    • VelocityPredictor - Development velocity prediction
    • QualityForecaster - Code quality forecasting
    • PatternAnalyzer - Productivity pattern recognition
    • RiskAssessor - Development risk assessment

🔄 EXTRACTION HISTORY:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 STEELCLAD EXTRACTION
   └─ Goal: Extract prediction algorithms from predictive_analytics_engine.py
   └─ Source: Lines 437-565 (128 lines)
   └─ Purpose: Separate ML algorithms into focused module

📞 DEPENDENCIES:
==================================================================
🤝 Imports: numpy, typing
📤 Provides: Specialized prediction algorithm classes
"""

import numpy as np
from typing import Dict, List, Any


class VelocityPredictor:
    """Handles velocity prediction using time series analysis."""
    
    def predict(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Predict development velocity."""
        commits = [d['commits'] for d in historical_data[-7:]]  # Last week
        
        if len(commits) < 3:
            return {'status': 'insufficient_data'}
        
        # Simple moving average prediction
        avg_commits = np.mean(commits)
        trend = np.polyfit(range(len(commits)), commits, 1)[0]
        
        return {
            'next_day': {
                'commits': max(0, round(avg_commits + trend)),
                'confidence': min(0.8, len(commits) / 7)
            },
            'next_week': {
                'commits': max(0, round(avg_commits * 7 + trend * 7)),
                'confidence': min(0.7, len(commits) / 7)
            },
            'trend': 'improving' if trend > 0.5 else 'declining' if trend < -0.5 else 'stable'
        }


class QualityForecaster:
    """Handles code quality forecasting."""
    
    def predict(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Predict code quality trends."""
        quality_scores = [d['quality_score'] for d in historical_data[-14:]]  # Last 2 weeks
        
        if len(quality_scores) < 5:
            return {'status': 'insufficient_data'}
        
        current_score = quality_scores[-1]
        trend = np.polyfit(range(len(quality_scores)), quality_scores, 1)[0]
        
        predicted_score = max(0, min(100, current_score + trend * 7))
        
        return {
            'next_week_score': round(predicted_score, 1),
            'confidence': min(0.8, len(quality_scores) / 14),
            'trend': 'improving' if trend > 1 else 'declining' if trend < -1 else 'stable',
            'volatility': np.std(quality_scores[-7:]) if len(quality_scores) >= 7 else 0
        }


class PatternAnalyzer:
    """Analyzes productivity and development patterns."""
    
    def analyze(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in development data."""
        if len(historical_data) < 7:
            return {'status': 'insufficient_data'}
        
        productivity_scores = [d['productivity_score'] for d in historical_data]
        
        return {
            'peak_hours': [9, 10, 11, 14, 15, 16],  # Simulated
            'productive_days': ['Tuesday', 'Wednesday', 'Thursday'],
            'average_productivity': round(np.mean(productivity_scores), 1),
            'productivity_consistency': round(1 / (1 + np.std(productivity_scores) / 10), 2),
            'pattern_strength': 0.75,
            'confidence': min(0.9, len(historical_data) / 30)
        }


class RiskAssessor:
    """Assesses various development risks."""
    
    def assess(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Assess development-related risks."""
        if len(historical_data) < 7:
            return {
                'technical_debt_risk': 'unknown',
                'burnout_risk': 'unknown',
                'quality_degradation_risk': 'unknown'
            }
        
        recent_data = historical_data[-7:]
        
        # Technical debt risk (based on quality trend)
        quality_scores = [d['quality_score'] for d in recent_data]
        quality_trend = np.polyfit(range(len(quality_scores)), quality_scores, 1)[0]
        
        # Burnout risk (based on productivity volatility)
        productivity_scores = [d['productivity_score'] for d in recent_data]
        productivity_volatility = np.std(productivity_scores)
        
        # Quality degradation (based on test coverage)
        test_coverage = [d['test_coverage'] for d in recent_data]
        coverage_trend = np.polyfit(range(len(test_coverage)), test_coverage, 1)[0]
        
        return {
            'technical_debt_risk': 'high' if quality_trend < -2 else 'medium' if quality_trend < 0 else 'low',
            'burnout_risk': 'high' if productivity_volatility > 20 else 'medium' if productivity_volatility > 10 else 'low',
            'quality_degradation_risk': 'high' if coverage_trend < -2 else 'medium' if coverage_trend < 0 else 'low',
            'overall_risk_score': self._calculate_overall_risk(quality_trend, productivity_volatility, coverage_trend)
        }
    
    def _calculate_overall_risk(self, quality_trend: float, productivity_volatility: float, coverage_trend: float) -> str:
        """Calculate overall risk assessment."""
        risk_score = 0
        
        if quality_trend < -2:
            risk_score += 3
        elif quality_trend < 0:
            risk_score += 1
        
        if productivity_volatility > 20:
            risk_score += 3
        elif productivity_volatility > 10:
            risk_score += 1
        
        if coverage_trend < -2:
            risk_score += 3
        elif coverage_trend < 0:
            risk_score += 1
        
        if risk_score >= 6:
            return 'high'
        elif risk_score >= 3:
            return 'medium'
        else:
            return 'low'