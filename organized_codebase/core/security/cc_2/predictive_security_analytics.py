#!/usr/bin/env python3
"""
Predictive Security Analytics Engine
Agent D Hour 5 - Proactive Security Management with Machine Learning

Implements predictive analytics for proactive threat detection and security management
following STEELCLAD Anti-Regression Modularization Protocol.
"""

import asyncio
import datetime
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sqlite3
from enum import Enum

# Import security modules
from .monitoring_modules.security_events import SecurityEvent, ThreatLevel, ResponseAction

class PredictionType(Enum):
    """Types of security predictions"""
    THREAT_PROBABILITY = "THREAT_PROBABILITY"
    ATTACK_PATTERN = "ATTACK_PATTERN"
    VULNERABILITY_EMERGENCE = "VULNERABILITY_EMERGENCE"
    SYSTEM_BREACH = "SYSTEM_BREACH"
    PERFORMANCE_DEGRADATION = "PERFORMANCE_DEGRADATION"

class PredictionConfidence(Enum):
    """Confidence levels for predictions"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

@dataclass
class SecurityPrediction:
    """Security prediction data structure"""
    prediction_id: str
    prediction_type: PredictionType
    confidence: PredictionConfidence
    probability_score: float
    predicted_event: str
    time_horizon_hours: int
    evidence_factors: List[str]
    recommended_actions: List[str]
    created_at: str
    expires_at: str
    
    # Additional metadata
    model_version: str = "1.0"
    training_data_size: int = 0
    feature_importance: Dict[str, float] = None

@dataclass
class ThreatPattern:
    """Identified threat pattern for prediction"""
    pattern_id: str
    pattern_name: str
    frequency: int
    threat_level: ThreatLevel
    time_pattern: List[int]  # Hours when pattern typically occurs
    source_pattern: List[str]  # Common source files/locations
    escalation_probability: float
    
class PredictiveAnalyticsEngine:
    """Core predictive analytics engine for security management"""
    
    def __init__(self, db_path: str = None):
        """Initialize predictive analytics engine"""
        if db_path is None:
            db_path = Path(__file__).parent / "predictive_analytics.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Analytics state
        self.historical_events = deque(maxlen=10000)
        self.threat_patterns = {}
        self.prediction_models = {}
        self.feature_extractors = {}
        
        # Prediction cache
        self.active_predictions = {}
        self.prediction_accuracy_history = deque(maxlen=1000)
        
        # Configuration
        self.config = {
            'min_pattern_frequency': 3,
            'prediction_horizon_hours': [1, 4, 12, 24, 72],
            'confidence_thresholds': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8,
                'very_high': 0.9
            },
            'max_predictions_per_type': 10,
            'prediction_retention_hours': 168  # 7 days
        }
        
        # Initialize database and models
        self._init_database()
        self._init_prediction_models()
        
        self.logger.info("Predictive Analytics Engine initialized")
    
    def _init_database(self):
        """Initialize predictive analytics database"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT UNIQUE NOT NULL,
                    prediction_type TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    probability_score REAL NOT NULL,
                    predicted_event TEXT NOT NULL,
                    time_horizon_hours INTEGER NOT NULL,
                    evidence_factors TEXT NOT NULL,
                    recommended_actions TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    training_data_size INTEGER NOT NULL,
                    feature_importance TEXT,
                    validated BOOLEAN DEFAULT NULL,
                    actual_outcome TEXT,
                    accuracy_score REAL
                )
            ''')
            
            # Threat patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS threat_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id TEXT UNIQUE NOT NULL,
                    pattern_name TEXT NOT NULL,
                    frequency INTEGER NOT NULL,
                    threat_level TEXT NOT NULL,
                    time_pattern TEXT NOT NULL,
                    source_pattern TEXT NOT NULL,
                    escalation_probability REAL NOT NULL,
                    discovered_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            ''')
            
            # Model performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    accuracy_score REAL NOT NULL,
                    precision_score REAL NOT NULL,
                    recall_score REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    evaluation_date TEXT NOT NULL,
                    training_samples INTEGER NOT NULL
                )
            ''')
            
            # Feature importance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feature_importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    recorded_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
    
    def _init_prediction_models(self):
        """Initialize prediction models for different threat types"""
        # Threat probability model
        self.prediction_models[PredictionType.THREAT_PROBABILITY] = {
            'type': 'frequency_based',
            'parameters': {
                'time_window_hours': 24,
                'frequency_threshold': 0.1,
                'trend_weight': 0.7,
                'pattern_weight': 0.3
            }
        }
        
        # Attack pattern model
        self.prediction_models[PredictionType.ATTACK_PATTERN] = {
            'type': 'pattern_matching',
            'parameters': {
                'similarity_threshold': 0.8,
                'sequence_length': 5,
                'time_decay_factor': 0.9
            }
        }
        
        # Vulnerability emergence model
        self.prediction_models[PredictionType.VULNERABILITY_EMERGENCE] = {
            'type': 'trend_analysis',
            'parameters': {
                'trend_window_hours': 72,
                'acceleration_threshold': 1.5,
                'severity_weight': 0.8
            }
        }
        
        # System breach model
        self.prediction_models[PredictionType.SYSTEM_BREACH] = {
            'type': 'risk_scoring',
            'parameters': {
                'critical_event_weight': 0.9,
                'escalation_pattern_weight': 0.8,
                'time_proximity_weight': 0.7
            }
        }
        
        # Performance degradation model
        self.prediction_models[PredictionType.PERFORMANCE_DEGRADATION] = {
            'type': 'anomaly_detection',
            'parameters': {
                'anomaly_threshold': 2.0,  # Standard deviations
                'trend_sensitivity': 0.8,
                'seasonal_adjustment': True
            }
        }
    
    async def process_security_event(self, event: SecurityEvent):
        """Process new security event for predictive analysis"""
        try:
            # Add to historical data
            self.historical_events.append(event)
            
            # Update threat patterns
            await self._update_threat_patterns(event)
            
            # Generate predictions if patterns warrant it
            await self._generate_predictions(event)
            
            # Validate previous predictions
            await self._validate_predictions(event)
            
        except Exception as e:
            self.logger.error(f"Error processing security event for prediction: {e}")
    
    async def _update_threat_patterns(self, event: SecurityEvent):
        """Update threat patterns based on new event"""
        try:
            pattern_key = f"{event.event_type}_{event.threat_level.value}"
            
            current_time = datetime.datetime.fromisoformat(event.timestamp)
            current_hour = current_time.hour
            
            if pattern_key in self.threat_patterns:
                pattern = self.threat_patterns[pattern_key]
                pattern.frequency += 1
                
                # Update time pattern
                if current_hour not in pattern.time_pattern:
                    pattern.time_pattern.append(current_hour)
                
                # Update source pattern
                if event.source_file not in pattern.source_pattern:
                    pattern.source_pattern.append(event.source_file)
                    # Keep only top 10 sources
                    if len(pattern.source_pattern) > 10:
                        pattern.source_pattern = pattern.source_pattern[-10:]
                
            else:
                # Create new pattern
                pattern = ThreatPattern(
                    pattern_id=pattern_key,
                    pattern_name=f"{event.event_type} ({event.threat_level.value})",
                    frequency=1,
                    threat_level=event.threat_level,
                    time_pattern=[current_hour],
                    source_pattern=[event.source_file],
                    escalation_probability=0.1  # Initial estimate
                )
                self.threat_patterns[pattern_key] = pattern
            
            # Store pattern in database
            await self._store_threat_pattern(pattern)
            
        except Exception as e:
            self.logger.error(f"Error updating threat patterns: {e}")
    
    async def _generate_predictions(self, trigger_event: SecurityEvent):
        """Generate predictions based on current threat landscape"""
        try:
            predictions = []
            
            # Generate different types of predictions
            for prediction_type in PredictionType:
                model = self.prediction_models.get(prediction_type)
                if not model:
                    continue
                
                prediction = await self._generate_prediction(
                    prediction_type, trigger_event, model
                )
                
                if prediction and prediction.confidence != PredictionConfidence.LOW:
                    predictions.append(prediction)
            
            # Store predictions
            for prediction in predictions:
                await self._store_prediction(prediction)
                self.active_predictions[prediction.prediction_id] = prediction
                
            if predictions:
                self.logger.info(f"Generated {len(predictions)} security predictions")
            
        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
    
    async def _generate_prediction(self, prediction_type: PredictionType, 
                                 trigger_event: SecurityEvent, 
                                 model: Dict[str, Any]) -> Optional[SecurityPrediction]:
        """Generate specific type of prediction"""
        try:
            if prediction_type == PredictionType.THREAT_PROBABILITY:
                return await self._predict_threat_probability(trigger_event, model)
            elif prediction_type == PredictionType.ATTACK_PATTERN:
                return await self._predict_attack_pattern(trigger_event, model)
            elif prediction_type == PredictionType.VULNERABILITY_EMERGENCE:
                return await self._predict_vulnerability_emergence(trigger_event, model)
            elif prediction_type == PredictionType.SYSTEM_BREACH:
                return await self._predict_system_breach(trigger_event, model)
            elif prediction_type == PredictionType.PERFORMANCE_DEGRADATION:
                return await self._predict_performance_degradation(trigger_event, model)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating {prediction_type.value} prediction: {e}")
            return None
    
    async def _predict_threat_probability(self, trigger_event: SecurityEvent, 
                                        model: Dict[str, Any]) -> Optional[SecurityPrediction]:
        """Predict probability of similar threats occurring"""
        try:
            # Analyze historical patterns
            similar_events = [
                event for event in self.historical_events
                if event.event_type == trigger_event.event_type
            ]
            
            if len(similar_events) < self.config['min_pattern_frequency']:
                return None
            
            # Calculate frequency-based probability
            recent_events = [
                event for event in similar_events
                if (datetime.datetime.now() - datetime.datetime.fromisoformat(event.timestamp)).days <= 7
            ]
            
            frequency_score = len(recent_events) / 7.0  # Events per day
            
            # Calculate trend
            time_points = [
                datetime.datetime.fromisoformat(event.timestamp) 
                for event in recent_events[-10:]
            ]
            
            if len(time_points) >= 2:
                # Simple trend calculation
                time_diffs = [
                    (time_points[i] - time_points[i-1]).total_seconds() / 3600
                    for i in range(1, len(time_points))
                ]
                avg_interval = sum(time_diffs) / len(time_diffs)
                trend_score = max(0, 1 - (avg_interval / 24))  # Higher score for shorter intervals
            else:
                trend_score = 0.1
            
            # Combine scores
            probability_score = (frequency_score * 0.7 + trend_score * 0.3)
            probability_score = min(probability_score, 1.0)
            
            # Determine confidence
            confidence = self._calculate_confidence(probability_score, len(similar_events))
            
            if confidence == PredictionConfidence.LOW:
                return None
            
            # Generate prediction
            prediction_id = f"threat_prob_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            time_horizon = 24  # Predict for next 24 hours
            
            return SecurityPrediction(
                prediction_id=prediction_id,
                prediction_type=PredictionType.THREAT_PROBABILITY,
                confidence=confidence,
                probability_score=probability_score,
                predicted_event=f"Similar {trigger_event.event_type} threat occurrence",
                time_horizon_hours=time_horizon,
                evidence_factors=[
                    f"Historical frequency: {frequency_score:.2f} events/day",
                    f"Recent trend acceleration: {trend_score:.2f}",
                    f"Pattern matches from {len(similar_events)} historical events"
                ],
                recommended_actions=[
                    "Increase monitoring frequency for similar threat patterns",
                    "Review and strengthen defenses for this threat type",
                    "Prepare automated response procedures",
                    "Consider proactive mitigation measures"
                ],
                created_at=datetime.datetime.now().isoformat(),
                expires_at=(datetime.datetime.now() + datetime.timedelta(hours=time_horizon)).isoformat(),
                training_data_size=len(similar_events)
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting threat probability: {e}")
            return None
    
    async def _predict_attack_pattern(self, trigger_event: SecurityEvent, 
                                    model: Dict[str, Any]) -> Optional[SecurityPrediction]:
        """Predict coordinated attack patterns"""
        try:
            # Look for sequences of related events
            recent_events = list(self.historical_events)[-50:]  # Last 50 events
            
            # Group events by source and time proximity
            event_groups = defaultdict(list)
            for event in recent_events:
                time_key = datetime.datetime.fromisoformat(event.timestamp).strftime('%Y%m%d_%H')
                source_key = event.source_file
                group_key = f"{source_key}_{time_key}"
                event_groups[group_key].append(event)
            
            # Find groups with multiple events (potential attack patterns)
            attack_candidates = [
                group for group in event_groups.values() 
                if len(group) >= 2 and any(
                    event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]
                    for event in group
                )
            ]
            
            if not attack_candidates:
                return None
            
            # Calculate attack pattern probability
            max_group_size = max(len(group) for group in attack_candidates)
            escalation_patterns = sum(
                1 for group in attack_candidates
                if len(group) > 1 and 
                group[-1].threat_level.severity_score > group[0].threat_level.severity_score
            )
            
            pattern_strength = (max_group_size - 1) / 10.0  # Normalize
            escalation_factor = escalation_patterns / max(len(attack_candidates), 1)
            
            probability_score = min(pattern_strength * 0.6 + escalation_factor * 0.4, 1.0)
            
            confidence = self._calculate_confidence(probability_score, len(attack_candidates))
            
            if confidence == PredictionConfidence.LOW:
                return None
            
            prediction_id = f"attack_pattern_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return SecurityPrediction(
                prediction_id=prediction_id,
                prediction_type=PredictionType.ATTACK_PATTERN,
                confidence=confidence,
                probability_score=probability_score,
                predicted_event="Coordinated attack pattern development",
                time_horizon_hours=4,
                evidence_factors=[
                    f"Detected {len(attack_candidates)} potential attack sequences",
                    f"Maximum events in sequence: {max_group_size}",
                    f"Escalation patterns observed: {escalation_patterns}",
                    f"Pattern strength score: {pattern_strength:.2f}"
                ],
                recommended_actions=[
                    "Implement immediate additional monitoring",
                    "Activate incident response procedures",
                    "Review access controls for affected systems",
                    "Consider temporary security hardening measures"
                ],
                created_at=datetime.datetime.now().isoformat(),
                expires_at=(datetime.datetime.now() + datetime.timedelta(hours=4)).isoformat(),
                training_data_size=len(recent_events)
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting attack pattern: {e}")
            return None
    
    async def _predict_vulnerability_emergence(self, trigger_event: SecurityEvent, 
                                             model: Dict[str, Any]) -> Optional[SecurityPrediction]:
        """Predict emergence of new vulnerabilities"""
        try:
            # Analyze trend in vulnerability types and sources
            vulnerability_events = [
                event for event in self.historical_events
                if 'vulnerability' in event.event_type.lower() or 
                   'injection' in event.event_type.lower() or
                   'security' in event.event_type.lower()
            ]
            
            if len(vulnerability_events) < 5:
                return None
            
            # Group by source file
            source_vulnerability_counts = defaultdict(int)
            for event in vulnerability_events[-30:]:  # Last 30 vulnerability events
                source_vulnerability_counts[event.source_file] += 1
            
            # Find files with increasing vulnerability trends
            high_risk_sources = [
                source for source, count in source_vulnerability_counts.items()
                if count >= 3
            ]
            
            if not high_risk_sources:
                return None
            
            # Calculate emergence probability
            trend_strength = len(high_risk_sources) / max(len(source_vulnerability_counts), 1)
            recent_acceleration = len([
                event for event in vulnerability_events[-10:]
                if event.source_file in high_risk_sources
            ]) / 10.0
            
            probability_score = min(trend_strength * 0.5 + recent_acceleration * 0.5, 1.0)
            
            confidence = self._calculate_confidence(probability_score, len(vulnerability_events))
            
            if confidence == PredictionConfidence.LOW:
                return None
            
            prediction_id = f"vuln_emerge_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return SecurityPrediction(
                prediction_id=prediction_id,
                prediction_type=PredictionType.VULNERABILITY_EMERGENCE,
                confidence=confidence,
                probability_score=probability_score,
                predicted_event=f"New vulnerabilities likely in {len(high_risk_sources)} source files",
                time_horizon_hours=72,
                evidence_factors=[
                    f"High-risk sources identified: {len(high_risk_sources)}",
                    f"Vulnerability trend strength: {trend_strength:.2f}",
                    f"Recent acceleration factor: {recent_acceleration:.2f}",
                    f"Based on {len(vulnerability_events)} historical vulnerability events"
                ],
                recommended_actions=[
                    "Conduct targeted security code review",
                    "Implement additional input validation",
                    "Increase testing coverage for identified files",
                    "Consider security-focused refactoring"
                ],
                created_at=datetime.datetime.now().isoformat(),
                expires_at=(datetime.datetime.now() + datetime.timedelta(hours=72)).isoformat(),
                training_data_size=len(vulnerability_events),
                feature_importance={
                    'trend_strength': trend_strength,
                    'acceleration': recent_acceleration,
                    'high_risk_sources': len(high_risk_sources)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting vulnerability emergence: {e}")
            return None
    
    async def _predict_system_breach(self, trigger_event: SecurityEvent, 
                                   model: Dict[str, Any]) -> Optional[SecurityPrediction]:
        """Predict potential system breach based on escalating threats"""
        try:
            # Analyze escalation patterns
            critical_events = [
                event for event in self.historical_events
                if event.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]
            ]
            
            # Recent critical events (last 24 hours)
            recent_critical = [
                event for event in critical_events
                if (datetime.datetime.now() - datetime.datetime.fromisoformat(event.timestamp)).hours <= 24
            ]
            
            if len(recent_critical) < 2:
                return None
            
            # Analyze unresolved events
            unresolved_critical = [event for event in recent_critical if not event.resolved]
            
            # Calculate breach probability
            critical_density = len(recent_critical) / 24.0  # Critical events per hour
            unresolved_factor = len(unresolved_critical) / max(len(recent_critical), 1)
            escalation_factor = 1.0 if trigger_event.threat_level == ThreatLevel.EMERGENCY else 0.7
            
            probability_score = min(
                critical_density * 0.4 + unresolved_factor * 0.4 + escalation_factor * 0.2,
                1.0
            )
            
            confidence = self._calculate_confidence(probability_score, len(critical_events))
            
            if confidence == PredictionConfidence.LOW or probability_score < 0.3:
                return None
            
            prediction_id = f"system_breach_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return SecurityPrediction(
                prediction_id=prediction_id,
                prediction_type=PredictionType.SYSTEM_BREACH,
                confidence=confidence,
                probability_score=probability_score,
                predicted_event="Potential system breach or security compromise",
                time_horizon_hours=12,
                evidence_factors=[
                    f"Critical event density: {critical_density:.2f} events/hour",
                    f"Unresolved critical events: {len(unresolved_critical)}",
                    f"Recent escalation trigger: {trigger_event.threat_level.value}",
                    f"Historical critical events: {len(critical_events)}"
                ],
                recommended_actions=[
                    "IMMEDIATE: Activate incident response team",
                    "URGENT: Review all unresolved critical events",
                    "Implement emergency security measures",
                    "Consider system isolation if breach confirmed",
                    "Prepare forensic investigation procedures"
                ],
                created_at=datetime.datetime.now().isoformat(),
                expires_at=(datetime.datetime.now() + datetime.timedelta(hours=12)).isoformat(),
                training_data_size=len(critical_events)
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting system breach: {e}")
            return None
    
    async def _predict_performance_degradation(self, trigger_event: SecurityEvent, 
                                             model: Dict[str, Any]) -> Optional[SecurityPrediction]:
        """Predict performance degradation based on security event patterns"""
        try:
            # Analyze security events that might impact performance
            resource_intensive_events = [
                event for event in self.historical_events
                if any(keyword in event.event_type.lower() 
                      for keyword in ['scan', 'correlation', 'analysis', 'processing'])
            ]
            
            # Recent resource-intensive events
            recent_intensive = [
                event for event in resource_intensive_events
                if (datetime.datetime.now() - datetime.datetime.fromisoformat(event.timestamp)).hours <= 6
            ]
            
            if len(recent_intensive) < 3:
                return None
            
            # Calculate load factor
            event_frequency = len(recent_intensive) / 6.0  # Events per hour
            
            # Check for event clustering (multiple events in short time)
            time_clustering = self._calculate_time_clustering(recent_intensive)
            
            # Consider current event impact
            current_impact = 0.8 if 'scan' in trigger_event.event_type.lower() else 0.3
            
            probability_score = min(
                event_frequency * 0.4 + time_clustering * 0.4 + current_impact * 0.2,
                1.0
            )
            
            confidence = self._calculate_confidence(probability_score, len(resource_intensive_events))
            
            if confidence == PredictionConfidence.LOW:
                return None
            
            prediction_id = f"perf_degrad_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return SecurityPrediction(
                prediction_id=prediction_id,
                prediction_type=PredictionType.PERFORMANCE_DEGRADATION,
                confidence=confidence,
                probability_score=probability_score,
                predicted_event="Performance degradation due to security operations",
                time_horizon_hours=6,
                evidence_factors=[
                    f"Resource-intensive events frequency: {event_frequency:.2f}/hour",
                    f"Time clustering factor: {time_clustering:.2f}",
                    f"Current event impact: {current_impact:.2f}",
                    f"Recent intensive events: {len(recent_intensive)}"
                ],
                recommended_actions=[
                    "Monitor system resource utilization closely",
                    "Consider throttling security operations",
                    "Prepare performance optimization measures",
                    "Schedule non-critical operations for off-peak hours"
                ],
                created_at=datetime.datetime.now().isoformat(),
                expires_at=(datetime.datetime.now() + datetime.timedelta(hours=6)).isoformat(),
                training_data_size=len(resource_intensive_events)
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting performance degradation: {e}")
            return None
    
    def _calculate_time_clustering(self, events: List[SecurityEvent]) -> float:
        """Calculate how clustered events are in time"""
        if len(events) < 2:
            return 0.0
        
        timestamps = [datetime.datetime.fromisoformat(event.timestamp) for event in events]
        timestamps.sort()
        
        # Calculate intervals between events
        intervals = [
            (timestamps[i] - timestamps[i-1]).total_seconds() / 60  # Minutes
            for i in range(1, len(timestamps))
        ]
        
        # Lower average interval = higher clustering
        avg_interval = sum(intervals) / len(intervals)
        clustering_score = max(0, 1 - (avg_interval / 60))  # Normalize to 0-1
        
        return clustering_score
    
    def _calculate_confidence(self, probability_score: float, sample_size: int) -> PredictionConfidence:
        """Calculate confidence level based on probability and sample size"""
        # Adjust confidence based on sample size
        sample_confidence = min(sample_size / 20.0, 1.0)  # Full confidence at 20+ samples
        
        # Combined confidence
        combined_confidence = probability_score * 0.7 + sample_confidence * 0.3
        
        if combined_confidence >= self.config['confidence_thresholds']['very_high']:
            return PredictionConfidence.VERY_HIGH
        elif combined_confidence >= self.config['confidence_thresholds']['high']:
            return PredictionConfidence.HIGH
        elif combined_confidence >= self.config['confidence_thresholds']['medium']:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW
    
    async def _validate_predictions(self, event: SecurityEvent):
        """Validate previous predictions against actual outcomes"""
        try:
            # Check if this event validates any previous predictions
            for prediction in list(self.active_predictions.values()):
                if self._prediction_matches_event(prediction, event):
                    # Mark prediction as validated
                    prediction_accuracy = self._calculate_prediction_accuracy(prediction, event)
                    
                    # Store validation result
                    await self._store_prediction_validation(
                        prediction.prediction_id, True, prediction_accuracy
                    )
                    
                    # Update accuracy history
                    self.prediction_accuracy_history.append({
                        'prediction_type': prediction.prediction_type.value,
                        'accuracy': prediction_accuracy,
                        'confidence': prediction.confidence.value,
                        'validated_at': datetime.datetime.now().isoformat()
                    })
                    
                    self.logger.info(f"Prediction {prediction.prediction_id} validated with accuracy {prediction_accuracy:.2f}")
            
            # Clean up expired predictions
            await self._cleanup_expired_predictions()
            
        except Exception as e:
            self.logger.error(f"Error validating predictions: {e}")
    
    def _prediction_matches_event(self, prediction: SecurityPrediction, event: SecurityEvent) -> bool:
        """Check if an event matches a prediction"""
        # Simple matching logic - can be enhanced
        if prediction.prediction_type == PredictionType.THREAT_PROBABILITY:
            return any(keyword in event.event_type.lower() 
                      for keyword in prediction.predicted_event.lower().split())
        elif prediction.prediction_type == PredictionType.ATTACK_PATTERN:
            return event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]
        elif prediction.prediction_type == PredictionType.VULNERABILITY_EMERGENCE:
            return 'vulnerability' in event.event_type.lower()
        elif prediction.prediction_type == PredictionType.SYSTEM_BREACH:
            return event.threat_level == ThreatLevel.EMERGENCY
        elif prediction.prediction_type == PredictionType.PERFORMANCE_DEGRADATION:
            return 'performance' in event.event_type.lower()
        
        return False
    
    def _calculate_prediction_accuracy(self, prediction: SecurityPrediction, event: SecurityEvent) -> float:
        """Calculate accuracy of a validated prediction"""
        # Time accuracy (how close to predicted timeframe)
        predicted_time = datetime.datetime.fromisoformat(prediction.expires_at)
        actual_time = datetime.datetime.fromisoformat(event.timestamp)
        time_diff_hours = abs((predicted_time - actual_time).total_seconds()) / 3600
        
        time_accuracy = max(0, 1 - (time_diff_hours / prediction.time_horizon_hours))
        
        # Severity accuracy (if applicable)
        severity_accuracy = 1.0  # Default
        if hasattr(event, 'threat_level'):
            predicted_severity = prediction.confidence.value
            actual_severity = event.threat_level.value
            
            # Simple severity matching
            severity_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'VERY_HIGH': 4}
            threat_map = {'INFO': 1, 'LOW': 2, 'MEDIUM': 3, 'HIGH': 4, 'CRITICAL': 5, 'EMERGENCY': 6}
            
            pred_score = severity_map.get(predicted_severity, 2)
            actual_score = threat_map.get(actual_severity, 3)
            
            severity_accuracy = 1.0 - abs(pred_score - actual_score) / 6.0
        
        # Combined accuracy
        return (time_accuracy * 0.6 + severity_accuracy * 0.4)
    
    async def _store_prediction(self, prediction: SecurityPrediction):
        """Store prediction in database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO predictions (
                        prediction_id, prediction_type, confidence, probability_score,
                        predicted_event, time_horizon_hours, evidence_factors,
                        recommended_actions, created_at, expires_at, model_version,
                        training_data_size, feature_importance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction.prediction_id,
                    prediction.prediction_type.value,
                    prediction.confidence.value,
                    prediction.probability_score,
                    prediction.predicted_event,
                    prediction.time_horizon_hours,
                    json.dumps(prediction.evidence_factors),
                    json.dumps(prediction.recommended_actions),
                    prediction.created_at,
                    prediction.expires_at,
                    prediction.model_version,
                    prediction.training_data_size,
                    json.dumps(prediction.feature_importance) if prediction.feature_importance else None
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing prediction: {e}")
    
    async def _store_threat_pattern(self, pattern: ThreatPattern):
        """Store threat pattern in database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO threat_patterns (
                        pattern_id, pattern_name, frequency, threat_level,
                        time_pattern, source_pattern, escalation_probability,
                        discovered_at, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern.pattern_id,
                    pattern.pattern_name,
                    pattern.frequency,
                    pattern.threat_level.value,
                    json.dumps(pattern.time_pattern),
                    json.dumps(pattern.source_pattern),
                    pattern.escalation_probability,
                    datetime.datetime.now().isoformat(),
                    datetime.datetime.now().isoformat()
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing threat pattern: {e}")
    
    async def _store_prediction_validation(self, prediction_id: str, validated: bool, accuracy: float):
        """Store prediction validation result"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE predictions 
                    SET validated = ?, accuracy_score = ?
                    WHERE prediction_id = ?
                ''', (validated, accuracy, prediction_id))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing prediction validation: {e}")
    
    async def _cleanup_expired_predictions(self):
        """Clean up expired predictions"""
        try:
            current_time = datetime.datetime.now()
            expired_predictions = []
            
            for prediction_id, prediction in list(self.active_predictions.items()):
                expires_at = datetime.datetime.fromisoformat(prediction.expires_at)
                if current_time > expires_at:
                    expired_predictions.append(prediction_id)
            
            for prediction_id in expired_predictions:
                del self.active_predictions[prediction_id]
            
            if expired_predictions:
                self.logger.info(f"Cleaned up {len(expired_predictions)} expired predictions")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired predictions: {e}")
    
    async def get_active_predictions(self, prediction_type: PredictionType = None) -> List[SecurityPrediction]:
        """Get currently active predictions"""
        predictions = list(self.active_predictions.values())
        
        if prediction_type:
            predictions = [p for p in predictions if p.prediction_type == prediction_type]
        
        return predictions
    
    async def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics and performance metrics"""
        try:
            total_predictions = len(self.active_predictions)
            accuracy_history = list(self.prediction_accuracy_history)
            
            if accuracy_history:
                avg_accuracy = sum(entry['accuracy'] for entry in accuracy_history) / len(accuracy_history)
                accuracy_by_type = defaultdict(list)
                
                for entry in accuracy_history:
                    accuracy_by_type[entry['prediction_type']].append(entry['accuracy'])
                
                type_accuracy = {
                    pred_type: sum(accuracies) / len(accuracies)
                    for pred_type, accuracies in accuracy_by_type.items()
                }
            else:
                avg_accuracy = 0.0
                type_accuracy = {}
            
            return {
                'total_active_predictions': total_predictions,
                'predictions_by_type': {
                    pred_type.value: len([p for p in self.active_predictions.values() 
                                        if p.prediction_type == pred_type])
                    for pred_type in PredictionType
                },
                'prediction_accuracy': {
                    'overall_average': avg_accuracy,
                    'by_type': type_accuracy,
                    'total_validated': len(accuracy_history)
                },
                'threat_patterns_discovered': len(self.threat_patterns),
                'historical_events_analyzed': len(self.historical_events),
                'generated_at': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting prediction statistics: {e}")
            return {}

# Factory function
def create_predictive_analytics_engine(db_path: str = None) -> PredictiveAnalyticsEngine:
    """Factory function to create a predictive analytics engine"""
    return PredictiveAnalyticsEngine(db_path)