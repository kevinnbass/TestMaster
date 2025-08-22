"""
Advanced Pattern Recognition System
====================================

Advanced pattern recognition system that enhances existing semantic analysis
capabilities with machine learning-powered pattern discovery and cross-system
pattern learning mechanisms.

Author: Agent A Phase 2 - Advanced Pattern Recognition
"""

import ast
import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import hashlib
from pathlib import Path
import statistics

from .semantic_pattern_detector import SemanticPatternDetector
from .semantic_relationship_analyzer import SemanticRelationshipAnalyzer
from .semantic_intent_analyzer import SemanticIntentAnalyzer


@dataclass
class AdvancedPattern:
    """Advanced pattern with ML-enhanced detection"""
    pattern_id: str
    pattern_name: str
    pattern_type: str  # "architectural", "behavioral", "semantic", "cross_system"
    confidence: float
    complexity_score: float
    impact_score: float
    locations: List[str]
    characteristics: Dict[str, Any]
    related_patterns: List[str]
    evolution_stage: str  # "emerging", "established", "declining"
    discovered_at: datetime = field(default_factory=datetime.now)


@dataclass
class PatternEvolution:
    """Pattern evolution tracking"""
    pattern_id: str
    evolution_history: List[Dict[str, Any]]
    trend: str  # "growing", "stable", "declining"
    prediction: Dict[str, Any]
    adaptation_recommendations: List[str]


@dataclass
class CrossSystemPattern:
    """Pattern that spans multiple intelligence systems"""
    pattern_id: str
    pattern_name: str
    affected_systems: List[str]  # "analytics", "ml", "api"
    system_interactions: Dict[str, List[str]]
    coordination_requirements: List[str]
    optimization_opportunities: List[str]
    impact_assessment: Dict[str, float]


class AdvancedPatternRecognizer:
    """
    Advanced pattern recognition system that enhances existing semantic analysis
    with machine learning and cross-system pattern discovery.
    """
    
    def __init__(self, analytics_hub=None, ml_orchestrator=None, api_gateway=None):
        self.analytics_hub = analytics_hub
        self.ml_orchestrator = ml_orchestrator
        self.api_gateway = api_gateway
        self.logger = logging.getLogger(__name__)
        
        # Pattern storage and tracking
        self.discovered_patterns: Dict[str, AdvancedPattern] = {}
        self.pattern_evolution_history: Dict[str, PatternEvolution] = {}
        self.cross_system_patterns: Dict[str, CrossSystemPattern] = {}
        self.pattern_correlation_matrix: Dict[str, Dict[str, float]] = {}
        
        # Learning and analysis data
        self.pattern_occurrence_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.pattern_effectiveness_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.pattern_adaptation_insights: List[Dict[str, Any]] = []
        
        # Configuration
        self.config = {
            "analysis_interval": 900,  # 15 minutes
            "min_occurrences_for_pattern": 5,
            "confidence_threshold": 0.7,
            "pattern_evolution_window_days": 7,
            "cross_system_analysis_enabled": True,
            "ml_pattern_detection_enabled": True,
            "adaptive_threshold_tuning": True
        }
        
        # Statistics
        self.recognition_stats = {
            "total_patterns_discovered": 0,
            "advanced_patterns_discovered": 0,
            "cross_system_patterns_discovered": 0,
            "pattern_evolution_predictions": 0,
            "successful_adaptations": 0,
            "start_time": datetime.now()
        }
        
        # Processing state
        self.is_recognizing = False
        self.recognition_task = None
        
        self.logger.info("Advanced Pattern Recognizer initialized")
    
    async def start_recognition(self):
        """Start advanced pattern recognition"""
        if self.is_recognizing:
            self.logger.warning("Pattern recognition already running")
            return
        
        self.is_recognizing = True
        self.recognition_task = asyncio.create_task(self._recognition_loop())
        self.logger.info("Started advanced pattern recognition")
    
    async def stop_recognition(self):
        """Stop advanced pattern recognition"""
        self.is_recognizing = False
        if self.recognition_task:
            self.recognition_task.cancel()
            try:
                await self.recognition_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped advanced pattern recognition")
    
    async def _recognition_loop(self):
        """Main pattern recognition loop"""
        while self.is_recognizing:
            try:
                await asyncio.sleep(self.config["analysis_interval"])
                
                # Discover new patterns using ML techniques
                if self.config["ml_pattern_detection_enabled"]:
                    await self._ml_pattern_discovery()
                
                # Analyze cross-system patterns
                if self.config["cross_system_analysis_enabled"]:
                    await self._cross_system_pattern_analysis()
                
                # Track pattern evolution
                await self._track_pattern_evolution()
                
                # Generate pattern correlation matrix
                await self._update_pattern_correlations()
                
                # Adapt recognition thresholds
                if self.config["adaptive_threshold_tuning"]:
                    await self._adaptive_threshold_tuning()
                
                # Generate pattern insights
                await self._generate_pattern_insights()
                
            except Exception as e:
                self.logger.error(f"Pattern recognition loop error: {e}")
                await asyncio.sleep(300)
    
    async def _ml_pattern_discovery(self):
        """Discover patterns using machine learning techniques"""
        try:
            # Collect pattern data from all available systems
            pattern_data = await self._collect_pattern_data()
            
            if not pattern_data:
                return
            
            # Apply ML algorithms for pattern discovery
            ml_patterns = await self._apply_ml_pattern_detection(pattern_data)
            
            # Validate and store discovered patterns
            for pattern in ml_patterns:
                if pattern.confidence >= self.config["confidence_threshold"]:
                    await self._store_discovered_pattern(pattern)
                    
        except Exception as e:
            self.logger.error(f"ML pattern discovery failed: {e}")
    
    async def _collect_pattern_data(self) -> Dict[str, Any]:
        """Collect pattern data from all intelligence systems"""
        pattern_data = {
            "analytics_patterns": [],
            "ml_patterns": [],
            "api_patterns": [],
            "semantic_patterns": [],
            "temporal_patterns": []
        }
        
        try:
            # Collect from analytics hub
            if self.analytics_hub:
                analytics_status = self.analytics_hub.get_hub_status()
                insights = self.analytics_hub.get_recent_insights(limit=50)
                
                pattern_data["analytics_patterns"] = [
                    {
                        "type": "analytics_insight",
                        "category": insight.category,
                        "confidence": insight.confidence,
                        "priority": insight.priority,
                        "timestamp": insight.timestamp.isoformat() if hasattr(insight, 'timestamp') else datetime.now().isoformat()
                    }
                    for insight in insights
                ]
            
            # Collect from ML orchestrator
            if self.ml_orchestrator:
                ml_insights = self.ml_orchestrator.get_integration_insights()
                
                pattern_data["ml_patterns"] = [
                    {
                        "type": "ml_optimization",
                        "optimization_type": opp.get("type", "unknown"),
                        "confidence": 0.8,  # Default confidence
                        "impact": opp.get("potential_improvement", 0.0),
                        "timestamp": datetime.now().isoformat()
                    }
                    for opp in ml_insights.get("optimization_opportunities", [])
                ]
            
            # Collect from API gateway
            if self.api_gateway:
                gateway_stats = self.api_gateway.get_gateway_statistics()
                
                # Convert gateway patterns to pattern data
                pattern_data["api_patterns"] = [
                    {
                        "type": "api_usage",
                        "pattern": "high_traffic" if gateway_stats["gateway_metrics"]["total_requests"] > 1000 else "normal_traffic",
                        "confidence": 0.9,
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            
            return pattern_data
            
        except Exception as e:
            self.logger.error(f"Failed to collect pattern data: {e}")
            return {}
    
    async def _apply_ml_pattern_detection(self, pattern_data: Dict[str, Any]) -> List[AdvancedPattern]:
        """Apply ML algorithms to detect advanced patterns"""
        discovered_patterns = []
        
        try:
            # Pattern 1: Temporal clustering of events
            temporal_patterns = await self._detect_temporal_patterns(pattern_data)
            discovered_patterns.extend(temporal_patterns)
            
            # Pattern 2: Cross-system correlation patterns
            correlation_patterns = await self._detect_correlation_patterns(pattern_data)
            discovered_patterns.extend(correlation_patterns)
            
            # Pattern 3: Anomaly-based pattern discovery
            anomaly_patterns = await self._detect_anomaly_patterns(pattern_data)
            discovered_patterns.extend(anomaly_patterns)
            
            # Pattern 4: Frequency-based pattern clustering
            frequency_patterns = await self._detect_frequency_patterns(pattern_data)
            discovered_patterns.extend(frequency_patterns)
            
            return discovered_patterns
            
        except Exception as e:
            self.logger.error(f"ML pattern detection failed: {e}")
            return []
    
    async def _detect_temporal_patterns(self, pattern_data: Dict[str, Any]) -> List[AdvancedPattern]:
        """Detect temporal patterns in the data"""
        patterns = []
        
        try:
            # Collect all timestamped events
            all_events = []
            for system, events in pattern_data.items():
                for event in events:
                    if "timestamp" in event:
                        all_events.append({
                            "system": system,
                            "timestamp": datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")),
                            "type": event.get("type", "unknown"),
                            "data": event
                        })
            
            if len(all_events) < 10:
                return patterns
            
            # Sort by timestamp
            all_events.sort(key=lambda x: x["timestamp"])
            
            # Detect time-based clustering
            time_clusters = self._find_temporal_clusters(all_events)
            
            for cluster_id, cluster_events in time_clusters.items():
                if len(cluster_events) >= self.config["min_occurrences_for_pattern"]:
                    # Calculate pattern characteristics
                    systems_involved = list(set(event["system"] for event in cluster_events))
                    event_types = list(set(event["type"] for event in cluster_events))
                    
                    patterns.append(AdvancedPattern(
                        pattern_id=f"temporal_{cluster_id}_{int(datetime.now().timestamp())}",
                        pattern_name=f"Temporal Cluster {cluster_id}",
                        pattern_type="temporal",
                        confidence=min(0.9, len(cluster_events) / 20.0),
                        complexity_score=len(systems_involved) * 0.3 + len(event_types) * 0.2,
                        impact_score=min(1.0, len(cluster_events) / 10.0),
                        locations=systems_involved,
                        characteristics={
                            "cluster_id": cluster_id,
                            "event_count": len(cluster_events),
                            "systems_involved": systems_involved,
                            "event_types": event_types,
                            "time_span_minutes": (cluster_events[-1]["timestamp"] - cluster_events[0]["timestamp"]).total_seconds() / 60
                        },
                        related_patterns=[],
                        evolution_stage="emerging"
                    ))
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Temporal pattern detection failed: {e}")
            return []
    
    def _find_temporal_clusters(self, events: List[Dict[str, Any]], time_window_minutes: int = 30) -> Dict[int, List[Dict[str, Any]]]:
        """Find temporal clusters in events"""
        clusters = {}
        cluster_id = 0
        
        try:
            current_cluster = []
            cluster_start_time = None
            
            for event in events:
                event_time = event["timestamp"]
                
                if not current_cluster or not cluster_start_time:
                    # Start new cluster
                    current_cluster = [event]
                    cluster_start_time = event_time
                    continue
                
                # Check if event belongs to current cluster
                time_diff = (event_time - cluster_start_time).total_seconds() / 60
                
                if time_diff <= time_window_minutes:
                    current_cluster.append(event)
                else:
                    # Save current cluster and start new one
                    if len(current_cluster) >= 3:  # Minimum cluster size
                        clusters[cluster_id] = current_cluster
                        cluster_id += 1
                    
                    current_cluster = [event]
                    cluster_start_time = event_time
            
            # Don't forget the last cluster
            if len(current_cluster) >= 3:
                clusters[cluster_id] = current_cluster
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Temporal clustering failed: {e}")
            return {}
    
    async def _detect_correlation_patterns(self, pattern_data: Dict[str, Any]) -> List[AdvancedPattern]:
        """Detect correlation patterns between different systems"""
        patterns = []
        
        try:
            # Extract numeric features from each system
            system_features = {}
            
            for system, events in pattern_data.items():
                features = []
                for event in events:
                    # Extract numeric features
                    feature_vector = []
                    
                    if "confidence" in event:
                        feature_vector.append(event["confidence"])
                    if "priority" in event:
                        feature_vector.append(event.get("priority", 0))
                    if "impact" in event:
                        feature_vector.append(event["impact"])
                    
                    if feature_vector:
                        features.append(feature_vector)
                
                if features and len(features) >= 5:  # Minimum for correlation analysis
                    # Calculate average feature vector
                    avg_features = np.mean(features, axis=0)
                    system_features[system] = avg_features
            
            # Calculate correlations between systems
            system_names = list(system_features.keys())
            for i, system_a in enumerate(system_names):
                for system_b in system_names[i+1:]:
                    features_a = system_features[system_a]
                    features_b = system_features[system_b]
                    
                    if len(features_a) == len(features_b) and len(features_a) > 0:
                        correlation = np.corrcoef(features_a, features_b)[0, 1]
                        
                        if abs(correlation) > 0.6:  # Strong correlation
                            patterns.append(AdvancedPattern(
                                pattern_id=f"correlation_{system_a}_{system_b}_{int(datetime.now().timestamp())}",
                                pattern_name=f"System Correlation: {system_a} - {system_b}",
                                pattern_type="correlation",
                                confidence=abs(correlation),
                                complexity_score=0.7,
                                impact_score=abs(correlation),
                                locations=[system_a, system_b],
                                characteristics={
                                    "correlation_coefficient": correlation,
                                    "correlation_type": "positive" if correlation > 0 else "negative",
                                    "systems": [system_a, system_b]
                                },
                                related_patterns=[],
                                evolution_stage="established"
                            ))
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Correlation pattern detection failed: {e}")
            return []
    
    async def _detect_anomaly_patterns(self, pattern_data: Dict[str, Any]) -> List[AdvancedPattern]:
        """Detect patterns through anomaly analysis"""
        patterns = []
        
        try:
            # Look for unusual concentrations of events
            for system, events in pattern_data.items():
                if len(events) < 5:
                    continue
                
                # Group events by type
                event_type_counts = defaultdict(int)
                for event in events:
                    event_type_counts[event.get("type", "unknown")] += 1
                
                # Find anomalous event types (unusually high frequency)
                total_events = len(events)
                for event_type, count in event_type_counts.items():
                    frequency = count / total_events
                    
                    if frequency > 0.7:  # Anomalously high frequency
                        patterns.append(AdvancedPattern(
                            pattern_id=f"anomaly_{system}_{event_type}_{int(datetime.now().timestamp())}",
                            pattern_name=f"Anomalous Pattern: {event_type} in {system}",
                            pattern_type="anomaly",
                            confidence=frequency,
                            complexity_score=0.5,
                            impact_score=frequency,
                            locations=[system],
                            characteristics={
                                "event_type": event_type,
                                "frequency": frequency,
                                "count": count,
                                "total_events": total_events
                            },
                            related_patterns=[],
                            evolution_stage="emerging"
                        ))
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Anomaly pattern detection failed: {e}")
            return []
    
    async def _detect_frequency_patterns(self, pattern_data: Dict[str, Any]) -> List[AdvancedPattern]:
        """Detect patterns based on event frequency analysis"""
        patterns = []
        
        try:
            # Analyze frequency patterns across all systems
            all_event_types = defaultdict(int)
            system_event_distributions = {}
            
            for system, events in pattern_data.items():
                event_types = defaultdict(int)
                for event in events:
                    event_type = event.get("type", "unknown")
                    event_types[event_type] += 1
                    all_event_types[event_type] += 1
                
                system_event_distributions[system] = dict(event_types)
            
            # Find dominant patterns
            total_events = sum(all_event_types.values())
            for event_type, count in all_event_types.items():
                frequency = count / max(total_events, 1)
                
                if frequency > 0.3 and count >= self.config["min_occurrences_for_pattern"]:
                    # Find which systems contribute to this pattern
                    contributing_systems = [
                        system for system, distribution in system_event_distributions.items()
                        if event_type in distribution and distribution[event_type] > 0
                    ]
                    
                    patterns.append(AdvancedPattern(
                        pattern_id=f"frequency_{event_type}_{int(datetime.now().timestamp())}",
                        pattern_name=f"Dominant Pattern: {event_type}",
                        pattern_type="frequency",
                        confidence=min(0.95, frequency * 2),
                        complexity_score=len(contributing_systems) * 0.2,
                        impact_score=frequency,
                        locations=contributing_systems,
                        characteristics={
                            "event_type": event_type,
                            "global_frequency": frequency,
                            "total_occurrences": count,
                            "contributing_systems": contributing_systems
                        },
                        related_patterns=[],
                        evolution_stage="established"
                    ))
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Frequency pattern detection failed: {e}")
            return []
    
    async def _store_discovered_pattern(self, pattern: AdvancedPattern):
        """Store a newly discovered pattern"""
        try:
            # Check if similar pattern already exists
            existing_pattern = self._find_similar_pattern(pattern)
            
            if existing_pattern:
                # Update existing pattern
                await self._update_existing_pattern(existing_pattern, pattern)
            else:
                # Store new pattern
                self.discovered_patterns[pattern.pattern_id] = pattern
                self.recognition_stats["advanced_patterns_discovered"] += 1
                
                # Track occurrence
                self.pattern_occurrence_history[pattern.pattern_id].append({
                    "timestamp": datetime.now(),
                    "confidence": pattern.confidence,
                    "impact_score": pattern.impact_score
                })
                
                self.logger.info(f"Discovered new advanced pattern: {pattern.pattern_name} (confidence: {pattern.confidence:.2f})")
                
        except Exception as e:
            self.logger.error(f"Failed to store discovered pattern: {e}")
    
    def _find_similar_pattern(self, new_pattern: AdvancedPattern) -> Optional[AdvancedPattern]:
        """Find if a similar pattern already exists"""
        try:
            for existing_pattern in self.discovered_patterns.values():
                # Check similarity based on type, locations, and characteristics
                if (existing_pattern.pattern_type == new_pattern.pattern_type and
                    set(existing_pattern.locations) == set(new_pattern.locations)):
                    
                    # Check characteristic similarity
                    similarity_score = self._calculate_pattern_similarity(existing_pattern, new_pattern)
                    if similarity_score > 0.8:
                        return existing_pattern
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find similar pattern: {e}")
            return None
    
    def _calculate_pattern_similarity(self, pattern_a: AdvancedPattern, pattern_b: AdvancedPattern) -> float:
        """Calculate similarity between two patterns"""
        try:
            similarity_factors = []
            
            # Type similarity
            if pattern_a.pattern_type == pattern_b.pattern_type:
                similarity_factors.append(1.0)
            else:
                similarity_factors.append(0.0)
            
            # Location similarity (Jaccard similarity)
            set_a = set(pattern_a.locations)
            set_b = set(pattern_b.locations)
            location_similarity = len(set_a.intersection(set_b)) / max(len(set_a.union(set_b)), 1)
            similarity_factors.append(location_similarity)
            
            # Confidence similarity
            conf_diff = abs(pattern_a.confidence - pattern_b.confidence)
            conf_similarity = 1.0 - min(conf_diff, 1.0)
            similarity_factors.append(conf_similarity)
            
            return statistics.mean(similarity_factors)
            
        except Exception:
            return 0.0
    
    async def _update_existing_pattern(self, existing: AdvancedPattern, new: AdvancedPattern):
        """Update an existing pattern with new information"""
        try:
            # Update confidence with weighted average
            weight = 0.3  # Weight for new observation
            existing.confidence = (existing.confidence * (1 - weight) + new.confidence * weight)
            
            # Update impact score
            existing.impact_score = max(existing.impact_score, new.impact_score)
            
            # Track occurrence
            self.pattern_occurrence_history[existing.pattern_id].append({
                "timestamp": datetime.now(),
                "confidence": new.confidence,
                "impact_score": new.impact_score
            })
            
        except Exception as e:
            self.logger.error(f"Failed to update existing pattern: {e}")
    
    async def _cross_system_pattern_analysis(self):
        """Analyze patterns that span multiple intelligence systems"""
        try:
            # Find patterns that involve multiple systems
            multi_system_patterns = [
                pattern for pattern in self.discovered_patterns.values()
                if len(pattern.locations) > 1
            ]
            
            for pattern in multi_system_patterns:
                if pattern.pattern_id not in self.cross_system_patterns:
                    cross_pattern = await self._analyze_cross_system_implications(pattern)
                    if cross_pattern:
                        self.cross_system_patterns[pattern.pattern_id] = cross_pattern
                        self.recognition_stats["cross_system_patterns_discovered"] += 1
                        
        except Exception as e:
            self.logger.error(f"Cross-system pattern analysis failed: {e}")
    
    async def _analyze_cross_system_implications(self, pattern: AdvancedPattern) -> Optional[CrossSystemPattern]:
        """Analyze cross-system implications of a pattern"""
        try:
            if len(pattern.locations) < 2:
                return None
            
            # Determine system interactions
            system_interactions = {}
            for system in pattern.locations:
                interactions = []
                
                # Analytics system interactions
                if system == "analytics_patterns" and self.analytics_hub:
                    interactions.extend(["predictive_analytics", "anomaly_detection", "correlation_analysis"])
                
                # ML system interactions
                elif system == "ml_patterns" and self.ml_orchestrator:
                    interactions.extend(["model_orchestration", "resource_optimization", "flow_management"])
                
                # API system interactions
                elif system == "api_patterns" and self.api_gateway:
                    interactions.extend(["request_routing", "security_validation", "performance_optimization"])
                
                system_interactions[system] = interactions
            
            # Identify coordination requirements
            coordination_requirements = []
            if len(pattern.locations) >= 2:
                coordination_requirements.extend([
                    "synchronized_processing",
                    "shared_state_management",
                    "cross_system_event_propagation"
                ])
            
            # Identify optimization opportunities
            optimization_opportunities = []
            if pattern.impact_score > 0.7:
                optimization_opportunities.extend([
                    "unified_pattern_handling",
                    "cross_system_optimization",
                    "pattern_based_automation"
                ])
            
            # Calculate impact assessment
            impact_assessment = {}
            for system in pattern.locations:
                # Base impact on pattern characteristics
                impact_assessment[system] = pattern.impact_score * pattern.confidence
            
            return CrossSystemPattern(
                pattern_id=pattern.pattern_id,
                pattern_name=f"Cross-System: {pattern.pattern_name}",
                affected_systems=pattern.locations,
                system_interactions=system_interactions,
                coordination_requirements=coordination_requirements,
                optimization_opportunities=optimization_opportunities,
                impact_assessment=impact_assessment
            )
            
        except Exception as e:
            self.logger.error(f"Cross-system analysis failed for pattern {pattern.pattern_id}: {e}")
            return None
    
    async def _track_pattern_evolution(self):
        """Track how patterns evolve over time"""
        try:
            for pattern_id, pattern in self.discovered_patterns.items():
                history = list(self.pattern_occurrence_history[pattern_id])
                
                if len(history) >= 5:  # Enough data for trend analysis
                    evolution = await self._analyze_pattern_evolution(pattern_id, history)
                    if evolution:
                        self.pattern_evolution_history[pattern_id] = evolution
                        self.recognition_stats["pattern_evolution_predictions"] += 1
                        
        except Exception as e:
            self.logger.error(f"Pattern evolution tracking failed: {e}")
    
    async def _analyze_pattern_evolution(self, pattern_id: str, history: List[Dict[str, Any]]) -> Optional[PatternEvolution]:
        """Analyze evolution of a specific pattern"""
        try:
            if len(history) < 5:
                return None
            
            # Extract time series data
            timestamps = [h["timestamp"] for h in history]
            confidences = [h["confidence"] for h in history]
            impacts = [h["impact_score"] for h in history]
            
            # Calculate trends
            confidence_trend = self._calculate_trend(confidences)
            impact_trend = self._calculate_trend(impacts)
            
            # Determine overall trend
            overall_trend = "stable"
            if confidence_trend > 0.1 and impact_trend > 0.1:
                overall_trend = "growing"
            elif confidence_trend < -0.1 and impact_trend < -0.1:
                overall_trend = "declining"
            
            # Generate prediction
            prediction = {
                "confidence_prediction": confidences[-1] + confidence_trend * 2,  # 2 steps ahead
                "impact_prediction": impacts[-1] + impact_trend * 2,
                "trend_strength": abs(confidence_trend) + abs(impact_trend)
            }
            
            # Generate adaptation recommendations
            recommendations = []
            if overall_trend == "declining":
                recommendations.extend([
                    "Review pattern detection parameters",
                    "Investigate environmental changes",
                    "Consider pattern archival"
                ])
            elif overall_trend == "growing":
                recommendations.extend([
                    "Enhance pattern-based automation",
                    "Increase monitoring frequency",
                    "Explore optimization opportunities"
                ])
            
            return PatternEvolution(
                pattern_id=pattern_id,
                evolution_history=history,
                trend=overall_trend,
                prediction=prediction,
                adaptation_recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Pattern evolution analysis failed for {pattern_id}: {e}")
            return None
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a series of values"""
        try:
            if len(values) < 2:
                return 0.0
            
            # Simple linear trend calculation
            x = list(range(len(values)))
            slope = np.polyfit(x, values, 1)[0]
            return slope
            
        except Exception:
            return 0.0
    
    async def _update_pattern_correlations(self):
        """Update pattern correlation matrix"""
        try:
            pattern_ids = list(self.discovered_patterns.keys())
            
            for i, pattern_a in enumerate(pattern_ids):
                for pattern_b in pattern_ids[i+1:]:
                    correlation = await self._calculate_pattern_correlation(pattern_a, pattern_b)
                    
                    if pattern_a not in self.pattern_correlation_matrix:
                        self.pattern_correlation_matrix[pattern_a] = {}
                    
                    self.pattern_correlation_matrix[pattern_a][pattern_b] = correlation
                    
        except Exception as e:
            self.logger.error(f"Pattern correlation update failed: {e}")
    
    async def _calculate_pattern_correlation(self, pattern_a_id: str, pattern_b_id: str) -> float:
        """Calculate correlation between two patterns"""
        try:
            history_a = list(self.pattern_occurrence_history[pattern_a_id])
            history_b = list(self.pattern_occurrence_history[pattern_b_id])
            
            if len(history_a) < 3 or len(history_b) < 3:
                return 0.0
            
            # Align timeseries (simplified approach)
            min_len = min(len(history_a), len(history_b))
            confidences_a = [h["confidence"] for h in history_a[-min_len:]]
            confidences_b = [h["confidence"] for h in history_b[-min_len:]]
            
            if min_len > 1:
                correlation = np.corrcoef(confidences_a, confidences_b)[0, 1]
                return correlation if not np.isnan(correlation) else 0.0
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _adaptive_threshold_tuning(self):
        """Adaptively tune recognition thresholds based on performance"""
        try:
            # Analyze recognition performance
            recent_patterns = [
                pattern for pattern in self.discovered_patterns.values()
                if (datetime.now() - pattern.discovered_at).days <= 1
            ]
            
            if len(recent_patterns) >= 5:
                # Calculate effectiveness metrics
                avg_confidence = statistics.mean(p.confidence for p in recent_patterns)
                avg_impact = statistics.mean(p.impact_score for p in recent_patterns)
                
                # Adjust thresholds based on performance
                if avg_confidence > 0.9 and avg_impact > 0.8:
                    # Performance is high, can afford to be more selective
                    self.config["confidence_threshold"] = min(0.8, self.config["confidence_threshold"] + 0.05)
                elif avg_confidence < 0.6 or avg_impact < 0.4:
                    # Performance is low, need to be less selective
                    self.config["confidence_threshold"] = max(0.5, self.config["confidence_threshold"] - 0.05)
                
                self.logger.debug(f"Adapted confidence threshold to {self.config['confidence_threshold']:.2f}")
                
        except Exception as e:
            self.logger.error(f"Adaptive threshold tuning failed: {e}")
    
    async def _generate_pattern_insights(self):
        """Generate high-level insights about discovered patterns"""
        try:
            insights = []
            
            # Overall pattern statistics
            if len(self.discovered_patterns) >= 5:
                pattern_types = [p.pattern_type for p in self.discovered_patterns.values()]
                type_distribution = defaultdict(int)
                for pt in pattern_types:
                    type_distribution[pt] += 1
                
                most_common_type = max(type_distribution.items(), key=lambda x: x[1])
                
                insights.append({
                    "type": "pattern_distribution",
                    "insight": f"Most common pattern type: {most_common_type[0]} ({most_common_type[1]} patterns)",
                    "significance": "high" if most_common_type[1] > len(self.discovered_patterns) * 0.4 else "medium"
                })
            
            # Cross-system pattern insights
            if len(self.cross_system_patterns) > 0:
                insights.append({
                    "type": "cross_system_patterns",
                    "insight": f"Discovered {len(self.cross_system_patterns)} cross-system patterns",
                    "significance": "high" if len(self.cross_system_patterns) > 3 else "medium"
                })
            
            # Pattern evolution insights
            evolving_patterns = sum(1 for evo in self.pattern_evolution_history.values() if evo.trend == "growing")
            if evolving_patterns > 0:
                insights.append({
                    "type": "pattern_evolution",
                    "insight": f"{evolving_patterns} patterns showing growth trend",
                    "significance": "high" if evolving_patterns > 2 else "medium"
                })
            
            # Store insights for analysis
            self.pattern_adaptation_insights.extend(insights)
            
            # Log significant insights
            for insight in insights:
                if insight["significance"] == "high":
                    self.logger.info(f"Pattern insight: {insight['insight']}")
                    
        except Exception as e:
            self.logger.error(f"Pattern insight generation failed: {e}")
    
    def get_recognition_status(self) -> Dict[str, Any]:
        """Get comprehensive pattern recognition status"""
        return {
            "is_recognizing": self.is_recognizing,
            "recognition_stats": self.recognition_stats.copy(),
            "discovered_patterns_count": len(self.discovered_patterns),
            "cross_system_patterns_count": len(self.cross_system_patterns),
            "pattern_evolution_tracked": len(self.pattern_evolution_history),
            "recent_patterns": [
                {
                    "name": pattern.pattern_name,
                    "type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "impact_score": pattern.impact_score,
                    "locations": pattern.locations,
                    "evolution_stage": pattern.evolution_stage,
                    "discovered_at": pattern.discovered_at.isoformat()
                }
                for pattern in sorted(self.discovered_patterns.values(), 
                                   key=lambda x: x.discovered_at, reverse=True)[:10]
            ],
            "pattern_insights": self.pattern_adaptation_insights[-5:],  # Recent insights
            "configuration": self.config.copy()
        }
    
    def get_pattern_correlations(self) -> Dict[str, Any]:
        """Get pattern correlation analysis"""
        return {
            "correlation_matrix": self.pattern_correlation_matrix,
            "strong_correlations": [
                {
                    "pattern_a": pa,
                    "pattern_b": pb,
                    "correlation": corr
                }
                for pa, correlations in self.pattern_correlation_matrix.items()
                for pb, corr in correlations.items()
                if abs(corr) > 0.7
            ],
            "cross_system_patterns": [
                {
                    "pattern_name": pattern.pattern_name,
                    "affected_systems": pattern.affected_systems,
                    "coordination_requirements": pattern.coordination_requirements,
                    "optimization_opportunities": pattern.optimization_opportunities,
                    "impact_assessment": pattern.impact_assessment
                }
                for pattern in self.cross_system_patterns.values()
            ]
        }


# Export
__all__ = ['AdvancedPatternRecognizer', 'AdvancedPattern', 'PatternEvolution', 'CrossSystemPattern']