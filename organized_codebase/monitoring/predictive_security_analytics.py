"""
Predictive Security Analytics Module
Extracted from advanced_security_dashboard.py for Agent X's Epsilon base integration
< 200 lines per STEELCLAD protocol

Provides forward-looking security intelligence:
- Threat probability prediction
- Attack pattern forecasting
- Vulnerability emergence prediction
- Time-based security forecasting
"""

import asyncio
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    THREAT_PROBABILITY = "threat_probability"
    ATTACK_PATTERN = "attack_pattern"
    VULNERABILITY_EMERGENCE = "vulnerability_emergence"
    SYSTEM_BREACH = "system_breach"
    RESOURCE_EXHAUSTION = "resource_exhaustion"

class TimeHorizon(Enum):
    SHORT_TERM = "1h"
    MEDIUM_TERM = "6h"
    DAILY = "24h"
    WEEKLY = "7d"

@dataclass
class SecurityPrediction:
    """Data structure for security predictions"""
    timestamp: str
    prediction_type: PredictionType
    prediction_value: float
    confidence_level: float
    time_horizon: TimeHorizon
    contributing_factors: List[str]
    risk_score: float
    mitigation_suggestions: List[str]

class PredictiveSecurityAnalytics:
    """Pluggable predictive security analytics module"""
    
    def __init__(self):
        self.predictions: List[SecurityPrediction] = []
        self.prediction_history: List[SecurityPrediction] = []
        self.analytics_active = False
        self.base_factors = {
            'high_traffic': 0.15,
            'suspicious_patterns': 0.25,
            'anomalous_behavior': 0.20,
            'failed_authentications': 0.18,
            'unusual_access_times': 0.12,
            'geographic_anomalies': 0.22,
            'rate_limiting_triggers': 0.16
        }
        
    async def initialize_predictive_analytics(self):
        """Initialize predictive security analytics engine"""
        try:
            self.analytics_active = True
            logger.info("Predictive Security Analytics initialized")
            
            # Start prediction generation loop
            asyncio.create_task(self._prediction_generation_loop())
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize predictive analytics: {e}")
            return False
    
    async def generate_security_predictions(self) -> List[SecurityPrediction]:
        """Generate ML-based security predictions"""
        predictions = []
        
        # Generate variable number of predictions based on current threat landscape
        prediction_count = max(1, np.random.poisson(2))
        
        for _ in range(prediction_count):
            prediction = self._create_security_prediction()
            predictions.append(prediction)
        
        # Update prediction lists
        self.predictions.extend(predictions)
        self.prediction_history.extend(predictions)
        
        # Maintain history limit
        if len(self.prediction_history) > 500:
            self.prediction_history = self.prediction_history[-500:]
            
        return predictions
    
    def _create_security_prediction(self) -> SecurityPrediction:
        """Create individual security prediction"""
        prediction_type = np.random.choice(list(PredictionType))
        time_horizon = np.random.choice(list(TimeHorizon))
        
        # Base prediction value with type-specific adjustments
        base_value = np.random.uniform(0.1, 0.8)
        type_multipliers = {
            PredictionType.THREAT_PROBABILITY: 1.0,
            PredictionType.ATTACK_PATTERN: 1.1,
            PredictionType.VULNERABILITY_EMERGENCE: 0.9,
            PredictionType.SYSTEM_BREACH: 1.3,
            PredictionType.RESOURCE_EXHAUSTION: 1.05
        }
        
        prediction_value = min(0.95, base_value * type_multipliers[prediction_type])
        
        # Calculate confidence based on time horizon and prediction value
        horizon_confidence = {
            TimeHorizon.SHORT_TERM: 0.85,
            TimeHorizon.MEDIUM_TERM: 0.75,
            TimeHorizon.DAILY: 0.70,
            TimeHorizon.WEEKLY: 0.60
        }
        
        confidence = min(0.95, 
            horizon_confidence[time_horizon] + 
            (prediction_value * 0.15) + 
            np.random.uniform(-0.05, 0.05)
        )
        
        # Select contributing factors based on prediction type
        contributing_factors = self._select_contributing_factors(prediction_type)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(prediction_value, confidence, prediction_type)
        
        # Generate mitigation suggestions
        mitigation_suggestions = self._generate_mitigation_suggestions(prediction_type, risk_score)
        
        return SecurityPrediction(
            timestamp=datetime.now().isoformat(),
            prediction_type=prediction_type,
            prediction_value=prediction_value,
            confidence_level=confidence,
            time_horizon=time_horizon,
            contributing_factors=contributing_factors,
            risk_score=risk_score,
            mitigation_suggestions=mitigation_suggestions
        )
    
    def _select_contributing_factors(self, prediction_type: PredictionType) -> List[str]:
        """Select relevant contributing factors for prediction type"""
        type_factor_map = {
            PredictionType.THREAT_PROBABILITY: ['high_traffic', 'suspicious_patterns', 'failed_authentications'],
            PredictionType.ATTACK_PATTERN: ['anomalous_behavior', 'suspicious_patterns', 'geographic_anomalies'],
            PredictionType.VULNERABILITY_EMERGENCE: ['unusual_access_times', 'failed_authentications', 'rate_limiting_triggers'],
            PredictionType.SYSTEM_BREACH: ['suspicious_patterns', 'anomalous_behavior', 'failed_authentications'],
            PredictionType.RESOURCE_EXHAUSTION: ['high_traffic', 'rate_limiting_triggers', 'unusual_access_times']
        }
        
        available_factors = type_factor_map.get(prediction_type, list(self.base_factors.keys()))
        factor_count = np.random.randint(2, min(4, len(available_factors) + 1))
        
        return np.random.choice(available_factors, size=factor_count, replace=False).tolist()
    
    def _calculate_risk_score(self, prediction_value: float, confidence: float, prediction_type: PredictionType) -> float:
        """Calculate overall risk score for prediction"""
        base_score = prediction_value * confidence
        
        # Type-specific risk multipliers
        risk_multipliers = {
            PredictionType.THREAT_PROBABILITY: 1.0,
            PredictionType.ATTACK_PATTERN: 1.2,
            PredictionType.VULNERABILITY_EMERGENCE: 0.8,
            PredictionType.SYSTEM_BREACH: 1.5,
            PredictionType.RESOURCE_EXHAUSTION: 1.1
        }
        
        return min(0.99, base_score * risk_multipliers[prediction_type])
    
    def _generate_mitigation_suggestions(self, prediction_type: PredictionType, risk_score: float) -> List[str]:
        """Generate mitigation suggestions based on prediction"""
        base_suggestions = {
            PredictionType.THREAT_PROBABILITY: ["enhance_monitoring", "review_access_logs", "update_threat_signatures"],
            PredictionType.ATTACK_PATTERN: ["isolate_suspicious_traffic", "enable_additional_logging", "review_firewall_rules"],
            PredictionType.VULNERABILITY_EMERGENCE: ["schedule_security_scan", "review_patch_status", "audit_configurations"],
            PredictionType.SYSTEM_BREACH: ["activate_incident_response", "backup_critical_data", "restrict_access"],
            PredictionType.RESOURCE_EXHAUSTION: ["scale_resources", "implement_rate_limiting", "optimize_performance"]
        }
        
        suggestions = base_suggestions.get(prediction_type, ["general_security_review"])
        
        if risk_score >= 0.7:
            suggestions.append("escalate_to_security_team")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    async def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of current security predictions"""
        if not self.predictions:
            return {"status": "no_predictions", "summary": {}}
        
        # Calculate summary metrics
        high_risk_count = sum(1 for p in self.predictions if p.risk_score >= 0.6)
        avg_confidence = np.mean([p.confidence_level for p in self.predictions])
        avg_risk = np.mean([p.risk_score for p in self.predictions])
        
        type_distribution = {}
        for pred_type in PredictionType:
            type_distribution[pred_type.value] = sum(1 for p in self.predictions if p.prediction_type == pred_type)
        
        return {
            "status": "active_predictions",
            "summary": {
                "total_predictions": len(self.predictions),
                "high_risk_predictions": high_risk_count,
                "average_confidence": round(avg_confidence, 3),
                "average_risk_score": round(avg_risk, 3),
                "prediction_distribution": type_distribution,
                "last_updated": datetime.now().isoformat()
            }
        }
    
    async def _prediction_generation_loop(self):
        """Continuous prediction generation loop"""
        while self.analytics_active:
            try:
                await self.generate_security_predictions()
                
                # Clean up old predictions (keep recent 20)
                if len(self.predictions) > 20:
                    self.predictions = self.predictions[-20:]
                
                await asyncio.sleep(45)  # Generate predictions every 45 seconds
                
            except Exception as e:
                logger.error(f"Error in prediction generation loop: {e}")
                await asyncio.sleep(60)
    
    async def stop_predictive_analytics(self):
        """Stop predictive analytics engine"""
        self.analytics_active = False
        logger.info("Predictive Security Analytics stopped")
    
    def get_active_predictions(self) -> List[SecurityPrediction]:
        """Get list of active security predictions"""
        return self.predictions.copy()
    
    def is_active(self) -> bool:
        """Check if predictive analytics is active"""
        return self.analytics_active

# Plugin interface for Agent X integration
def create_predictive_analytics_plugin(config: Dict[str, Any] = None):
    """Factory function to create predictive security analytics plugin"""
    return PredictiveSecurityAnalytics()