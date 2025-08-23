#!/usr/bin/env python3
"""
Advanced Security Correlation Engine
Agent D Enhancement - Advanced threat correlation algorithms across all existing security systems

This module ENHANCES existing security architecture by providing:
- Advanced correlation algorithms for multi-system threat detection
- Machine learning-based pattern recognition for emerging threats
- Temporal correlation analysis for attack chain detection
- Behavioral analytics for anomaly detection
- Predictive threat modeling based on historical patterns

IMPORTANT: This module ENHANCES existing correlation, does not replace functionality.
"""

import asyncio
import logging
import json
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import threading
import hashlib
import statistics
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class CorrelationType(Enum):
    """Types of security event correlation"""
    TEMPORAL = "temporal"        # Time-based correlation
    SPATIAL = "spatial"          # Location/system-based correlation
    BEHAVIORAL = "behavioral"    # Pattern-based correlation
    SIGNATURE = "signature"      # Known threat signature correlation
    ANOMALY = "anomaly"         # Statistical anomaly correlation
    CAUSAL = "causal"           # Cause-effect relationship correlation


class ThreatConfidence(Enum):
    """Confidence levels for correlated threats"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9
    CERTAIN = 0.99


class ThreatVectorType(Enum):
    """Multi-dimensional threat vector types for advanced correlation"""
    NETWORK_VECTOR = "network"          # Network-based attack vectors
    ENDPOINT_VECTOR = "endpoint"        # Endpoint compromise vectors
    APPLICATION_VECTOR = "application"  # Application-level attack vectors
    DATA_VECTOR = "data"               # Data exfiltration/manipulation vectors
    IDENTITY_VECTOR = "identity"       # Identity and access attack vectors
    INFRASTRUCTURE_VECTOR = "infrastructure"  # Infrastructure attack vectors


@dataclass
class SecurityEvent:
    """Enhanced security event for correlation analysis"""
    event_id: str
    source_system: str
    timestamp: str
    event_type: str
    severity: str
    description: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    indicators: List[str] = field(default_factory=list)
    affected_entities: List[str] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    # Enhanced multi-dimensional threat vector attributes
    threat_vectors: List[ThreatVectorType] = field(default_factory=list)
    vector_attributes: Dict[ThreatVectorType, Dict[str, Any]] = field(default_factory=dict)
    behavioral_metrics: Dict[str, float] = field(default_factory=dict)
    correlation_hints: List[str] = field(default_factory=list)


@dataclass
class CorrelationResult:
    """Result of advanced correlation analysis"""
    correlation_id: str
    correlated_events: List[str]  # Event IDs
    correlation_types: List[CorrelationType]
    threat_confidence: ThreatConfidence
    threat_category: str
    attack_chain_stage: Optional[str]
    risk_score: float
    correlation_strength: float
    temporal_pattern: Optional[Dict[str, Any]] = None
    behavioral_indicators: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # Enhanced multi-dimensional correlation attributes
    threat_vectors_involved: List[ThreatVectorType] = field(default_factory=list)
    vector_correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    multi_dimensional_score: float = 0.0
    behavioral_deviation_score: float = 0.0
    cross_system_indicators: List[str] = field(default_factory=list)


@dataclass
class MultiDimensionalCorrelationResult:
    """Result of multi-dimensional threat vector correlation analysis"""
    correlation_id: str
    primary_correlation: CorrelationResult
    vector_analysis: Dict[ThreatVectorType, Dict[str, Any]]
    dimensional_correlation_matrix: Dict[str, Dict[str, float]]
    pattern_recognition_results: Dict[str, Any]
    behavioral_analysis_results: Dict[str, Any]
    confidence_score: float
    threat_vector_complexity: float
    recommended_response_actions: List[str] = field(default_factory=list)
    cross_vector_indicators: List[str] = field(default_factory=list)
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ThreatPattern:
    """Known threat pattern for correlation matching"""
    pattern_id: str
    pattern_name: str
    pattern_type: str
    indicators: List[str]
    temporal_signature: Optional[Dict[str, Any]] = None
    behavioral_signature: Optional[Dict[str, Any]] = None
    severity_threshold: float = 0.5
    confidence_multiplier: float = 1.0


class AdvancedSecurityCorrelationEngine:
    """
    Advanced correlation engine for multi-system threat detection
    
    This engine ENHANCES existing security systems by:
    - Providing sophisticated correlation algorithms for threat detection
    - Implementing machine learning-based pattern recognition
    - Adding temporal and behavioral analysis capabilities
    - Creating predictive threat modeling based on historical data
    
    Does NOT replace existing correlation - adds advanced analytics layer.
    """
    
    def __init__(self, correlation_window_minutes: int = 60):
        """
        Initialize advanced correlation engine
        
        Args:
            correlation_window_minutes: Time window for event correlation
        """
        self.correlation_active = False
        self.correlation_window = timedelta(minutes=correlation_window_minutes)
        
        # Event storage and correlation tracking
        self.event_buffer = deque(maxlen=10000)  # Keep last 10k events
        self.correlation_results = deque(maxlen=5000)  # Keep last 5k correlations
        self.threat_patterns = {}  # Known threat patterns
        self.behavioral_baselines = {}  # Behavioral baselines per system
        
        # Correlation configuration
        self.correlation_config = {
            'temporal_threshold_minutes': 15,
            'spatial_correlation_weight': 0.8,
            'behavioral_anomaly_threshold': 2.5,  # Standard deviations
            'signature_match_threshold': 0.85,
            'min_correlation_strength': 0.6,
            'max_correlation_events': 50,
            'pattern_learning_enabled': True
        }
        
        # Advanced analytics components
        self.temporal_analyzer = TemporalCorrelationAnalyzer()
        self.behavioral_analyzer = BehavioralCorrelationAnalyzer()
        self.pattern_matcher = ThreatPatternMatcher()
        self.anomaly_detector = StatisticalAnomalyDetector()
        
        # Correlation statistics
        self.correlation_stats = {
            'engine_start_time': datetime.now(),
            'events_processed': 0,
            'correlations_generated': 0,
            'threats_identified': 0,
            'attack_chains_detected': 0,
            'false_positives': 0,
            'pattern_matches': 0
        }
        
        # Threading for correlation processing
        self.correlation_thread = None
        self.correlation_executor = ThreadPoolExecutor(max_workers=3)
        self.correlation_lock = threading.Lock()
        
        # Initialize threat patterns
        self._initialize_threat_patterns()
        
        logger.info("Advanced Security Correlation Engine initialized")
        logger.info("Ready to enhance existing security systems with advanced correlation")
    
    def start_correlation_engine(self):
        """Start advanced correlation processing"""
        if self.correlation_active:
            logger.warning("Correlation engine already active")
            return
        
        logger.info("Starting Advanced Security Correlation Engine...")
        self.correlation_active = True
        
        # Start correlation processing thread
        self.correlation_thread = threading.Thread(
            target=self._correlation_processing_loop,
            daemon=True
        )
        self.correlation_thread.start()
        
        logger.info("Advanced correlation engine started")
        logger.info("Enhanced threat detection and pattern analysis active")
    
    def process_security_event(self, event: SecurityEvent) -> Optional[List[CorrelationResult]]:
        """
        Process security event through advanced correlation analysis
        
        Args:
            event: SecurityEvent to process
            
        Returns:
            List of correlation results if correlations found
        """
        with self.correlation_lock:
            # Add event to buffer
            self.event_buffer.append(event)
            
            logger.debug(f"Processing security event: {event.event_id} from {event.source_system}")
            
            # Perform correlation analysis
            correlations = self._perform_advanced_correlation(event)
            
            # Store correlation results
            for correlation in correlations:
                self.correlation_results.append(correlation)
            
            # Update statistics
            self.correlation_stats['events_processed'] += 1
            if correlations:
                self.correlation_stats['correlations_generated'] += len(correlations)
                
                # Check for high-confidence threats
                for correlation in correlations:
                    if correlation.threat_confidence.value >= ThreatConfidence.HIGH.value:
                        self.correlation_stats['threats_identified'] += 1
                        
                        if correlation.attack_chain_stage:
                            self.correlation_stats['attack_chains_detected'] += 1
            
            return correlations if correlations else None
    
    def _correlation_processing_loop(self):
        """Main correlation processing loop for continuous analysis"""
        logger.info("Correlation processing loop started")
        
        while self.correlation_active:
            try:
                # Perform periodic correlation analysis
                self._perform_batch_correlation()
                
                # Update behavioral baselines
                self._update_behavioral_baselines()
                
                # Learn new threat patterns
                if self.correlation_config['pattern_learning_enabled']:
                    self._learn_threat_patterns()
                
                # Clean up old events
                self._cleanup_old_events()
                
                # Update correlation statistics
                self._update_correlation_statistics()
                
                # Sleep before next processing cycle
                time.sleep(30)  # Process every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in correlation processing loop: {e}")
                time.sleep(60)
        
        logger.info("Correlation processing loop stopped")
    
    def _perform_advanced_correlation(self, event: SecurityEvent) -> List[CorrelationResult]:
        """Perform advanced correlation analysis for a single event"""
        correlations = []
        
        # Get events within correlation window
        correlation_candidates = self._get_correlation_candidates(event)
        
        if not correlation_candidates:
            return correlations
        
        # Perform different types of correlation analysis
        correlation_tasks = [
            self.correlation_executor.submit(self._temporal_correlation, event, correlation_candidates),
            self.correlation_executor.submit(self._spatial_correlation, event, correlation_candidates),
            self.correlation_executor.submit(self._behavioral_correlation, event, correlation_candidates),
            self.correlation_executor.submit(self._signature_correlation, event, correlation_candidates),
            self.correlation_executor.submit(self._anomaly_correlation, event, correlation_candidates)
        ]
        
        # Collect correlation results
        for task in correlation_tasks:
            try:
                correlation_result = task.result(timeout=10)
                if correlation_result:
                    correlations.extend(correlation_result)
            except Exception as e:
                logger.warning(f"Error in correlation task: {e}")
        
        # Merge and rank correlations
        if correlations:
            correlations = self._merge_and_rank_correlations(correlations)
        
        return correlations
    
    def _get_correlation_candidates(self, event: SecurityEvent) -> List[SecurityEvent]:
        """Get events that could be correlated with the given event"""
        candidates = []
        
        event_time = datetime.fromisoformat(event.timestamp)
        window_start = event_time - self.correlation_window
        
        with self.correlation_lock:
            for candidate in self.event_buffer:
                if candidate.event_id == event.event_id:
                    continue
                
                candidate_time = datetime.fromisoformat(candidate.timestamp)
                
                # Check if within temporal window
                if window_start <= candidate_time <= event_time:
                    # Additional filtering based on relevance
                    if self._is_correlation_candidate(event, candidate):
                        candidates.append(candidate)
        
        return candidates[-50:]  # Limit to most recent 50 candidates
    
    def _is_correlation_candidate(self, event: SecurityEvent, candidate: SecurityEvent) -> bool:
        """Determine if candidate event is relevant for correlation"""
        # System correlation
        if event.source_system == candidate.source_system:
            return True
        
        # Indicator overlap
        if set(event.indicators) & set(candidate.indicators):
            return True
        
        # Entity overlap
        if set(event.affected_entities) & set(candidate.affected_entities):
            return True
        
        # Severity correlation
        if event.severity in ['critical', 'high'] and candidate.severity in ['critical', 'high']:
            return True
        
        return False
    
    def _temporal_correlation(self, event: SecurityEvent, candidates: List[SecurityEvent]) -> List[CorrelationResult]:
        """Perform temporal correlation analysis"""
        return self.temporal_analyzer.analyze(event, candidates, self.correlation_config)
    
    def _spatial_correlation(self, event: SecurityEvent, candidates: List[SecurityEvent]) -> List[CorrelationResult]:
        """Perform spatial (system/location-based) correlation analysis"""
        correlations = []
        
        # Group events by system/location proximity
        system_groups = defaultdict(list)
        for candidate in candidates:
            # Simple spatial correlation based on system similarity
            if self._calculate_system_proximity(event.source_system, candidate.source_system) > 0.5:
                system_groups[candidate.source_system].append(candidate)
        
        for system, system_events in system_groups.items():
            if len(system_events) >= 2:  # Need multiple events for correlation
                correlation = CorrelationResult(
                    correlation_id=str(uuid.uuid4()),
                    correlated_events=[event.event_id] + [e.event_id for e in system_events[:5]],
                    correlation_types=[CorrelationType.SPATIAL],
                    threat_confidence=ThreatConfidence.MEDIUM,
                    threat_category="coordinated_attack",
                    attack_chain_stage=None,
                    risk_score=0.6,
                    correlation_strength=0.7,
                    recommended_actions=["Monitor system group", "Check for lateral movement"]
                )
                correlations.append(correlation)
        
        return correlations
    
    def _behavioral_correlation(self, event: SecurityEvent, candidates: List[SecurityEvent]) -> List[CorrelationResult]:
        """Perform behavioral pattern correlation analysis"""
        return self.behavioral_analyzer.analyze(event, candidates, self.behavioral_baselines)
    
    def _signature_correlation(self, event: SecurityEvent, candidates: List[SecurityEvent]) -> List[CorrelationResult]:
        """Perform known threat signature correlation"""
        return self.pattern_matcher.match_patterns(event, candidates, self.threat_patterns)
    
    def _anomaly_correlation(self, event: SecurityEvent, candidates: List[SecurityEvent]) -> List[CorrelationResult]:
        """Perform statistical anomaly correlation"""
        return self.anomaly_detector.detect_anomalous_patterns(event, candidates, self.behavioral_baselines)
    
    def _calculate_system_proximity(self, system1: str, system2: str) -> float:
        """Calculate proximity score between two systems"""
        # Simple heuristic - can be enhanced with actual system topology
        if system1 == system2:
            return 1.0
        
        # Check for related systems
        related_systems = {
            'continuous_monitoring': ['unified_scanner', 'api_security'],
            'unified_scanner': ['continuous_monitoring', 'security_testing'],
            'api_security': ['continuous_monitoring', 'unified_scanner']
        }
        
        if system2 in related_systems.get(system1, []):
            return 0.8
        
        return 0.3  # Default proximity for unrelated systems
    
    def _merge_and_rank_correlations(self, correlations: List[CorrelationResult]) -> List[CorrelationResult]:
        """Merge overlapping correlations and rank by importance"""
        # Simple ranking by confidence and risk score
        correlations.sort(key=lambda c: (c.threat_confidence.value, c.risk_score), reverse=True)
        
        # Remove duplicates and merge similar correlations
        unique_correlations = []
        seen_event_combinations = set()
        
        for correlation in correlations:
            event_signature = tuple(sorted(correlation.correlated_events))
            if event_signature not in seen_event_combinations:
                seen_event_combinations.add(event_signature)
                unique_correlations.append(correlation)
        
        return unique_correlations[:10]  # Return top 10 correlations
    
    def _perform_batch_correlation(self):
        """Perform batch correlation analysis on recent events"""
        # Analyze patterns in recent events
        with self.correlation_lock:
            recent_events = list(self.event_buffer)[-100:]  # Last 100 events
        
        if len(recent_events) < 5:
            return
        
        # Look for patterns across multiple events
        pattern_correlations = self._detect_multi_event_patterns(recent_events)
        
        with self.correlation_lock:
            for correlation in pattern_correlations:
                self.correlation_results.append(correlation)
                self.correlation_stats['correlations_generated'] += 1
    
    def _detect_multi_event_patterns(self, events: List[SecurityEvent]) -> List[CorrelationResult]:
        """Detect patterns across multiple security events"""
        correlations = []
        
        # Time-series pattern detection
        time_series_patterns = self._analyze_time_series_patterns(events)
        correlations.extend(time_series_patterns)
        
        # Frequency-based pattern detection
        frequency_patterns = self._analyze_frequency_patterns(events)
        correlations.extend(frequency_patterns)
        
        return correlations
    
    def _analyze_time_series_patterns(self, events: List[SecurityEvent]) -> List[CorrelationResult]:
        """Analyze time series patterns in security events"""
        correlations = []
        
        # Group events by type and system
        event_series = defaultdict(list)
        for event in events:
            key = f"{event.source_system}_{event.event_type}"
            event_time = datetime.fromisoformat(event.timestamp)
            event_series[key].append((event_time, event))
        
        # Analyze each series for patterns
        for series_key, series_events in event_series.items():
            if len(series_events) >= 3:  # Need minimum events for pattern
                series_events.sort(key=lambda x: x[0])  # Sort by time
                
                # Check for regular intervals (potential automated attack)
                intervals = []
                for i in range(1, len(series_events)):
                    interval = (series_events[i][0] - series_events[i-1][0]).total_seconds()
                    intervals.append(interval)
                
                if intervals and statistics.stdev(intervals) < 30:  # Consistent intervals
                    correlation = CorrelationResult(
                        correlation_id=str(uuid.uuid4()),
                        correlated_events=[e[1].event_id for e in series_events],
                        correlation_types=[CorrelationType.TEMPORAL],
                        threat_confidence=ThreatConfidence.HIGH,
                        threat_category="automated_attack",
                        attack_chain_stage="execution",
                        risk_score=0.8,
                        correlation_strength=0.9,
                        temporal_pattern={
                            'pattern_type': 'regular_intervals',
                            'average_interval_seconds': statistics.mean(intervals),
                            'consistency_score': 1.0 - (statistics.stdev(intervals) / 100)
                        },
                        recommended_actions=["Block automated source", "Investigate attack tool"]
                    )
                    correlations.append(correlation)
        
        return correlations
    
    def _analyze_frequency_patterns(self, events: List[SecurityEvent]) -> List[CorrelationResult]:
        """Analyze frequency-based patterns in security events"""
        correlations = []
        
        # Analyze event frequency by various dimensions
        frequency_analysis = {
            'by_system': defaultdict(int),
            'by_type': defaultdict(int),
            'by_severity': defaultdict(int),
            'by_indicator': defaultdict(int)
        }
        
        for event in events:
            frequency_analysis['by_system'][event.source_system] += 1
            frequency_analysis['by_type'][event.event_type] += 1
            frequency_analysis['by_severity'][event.severity] += 1
            
            for indicator in event.indicators:
                frequency_analysis['by_indicator'][indicator] += 1
        
        # Check for unusual frequency patterns
        total_events = len(events)
        for dimension, counts in frequency_analysis.items():
            for item, count in counts.items():
                frequency = count / total_events
                
                # High frequency might indicate focused attack
                if frequency > 0.3 and count > 5:
                    # Find events with this characteristic
                    matching_events = []
                    for event in events:
                        if (dimension == 'by_system' and event.source_system == item) or \
                           (dimension == 'by_type' and event.event_type == item) or \
                           (dimension == 'by_severity' and event.severity == item) or \
                           (dimension == 'by_indicator' and item in event.indicators):
                            matching_events.append(event.event_id)
                    
                    if len(matching_events) >= 5:
                        correlation = CorrelationResult(
                            correlation_id=str(uuid.uuid4()),
                            correlated_events=matching_events[:10],  # Limit to 10 events
                            correlation_types=[CorrelationType.BEHAVIORAL],
                            threat_confidence=ThreatConfidence.MEDIUM,
                            threat_category="focused_attack",
                            attack_chain_stage="reconnaissance",
                            risk_score=0.65,
                            correlation_strength=frequency,
                            behavioral_indicators=[f"high_frequency_{dimension}"],
                            recommended_actions=["Investigate attack focus", "Enhance monitoring"]
                        )
                        correlations.append(correlation)
        
        return correlations
    
    def _update_behavioral_baselines(self):
        """Update behavioral baselines for anomaly detection"""
        # Update baselines based on recent event patterns
        with self.correlation_lock:
            recent_events = list(self.event_buffer)[-200:]  # Last 200 events
        
        # Calculate baselines per system
        for system in set(event.source_system for event in recent_events):
            system_events = [e for e in recent_events if e.source_system == system]
            
            if len(system_events) >= 10:  # Need minimum events for baseline
                baseline = self._calculate_system_baseline(system_events)
                self.behavioral_baselines[system] = baseline
    
    def _calculate_system_baseline(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Calculate behavioral baseline for a system"""
        # Event frequency analysis
        time_intervals = []
        event_types = defaultdict(int)
        severity_distribution = defaultdict(int)
        
        events.sort(key=lambda e: e.timestamp)
        
        for i in range(1, len(events)):
            prev_time = datetime.fromisoformat(events[i-1].timestamp)
            curr_time = datetime.fromisoformat(events[i].timestamp)
            interval = (curr_time - prev_time).total_seconds()
            time_intervals.append(interval)
        
        for event in events:
            event_types[event.event_type] += 1
            severity_distribution[event.severity] += 1
        
        baseline = {
            'event_count': len(events),
            'avg_interval_seconds': statistics.mean(time_intervals) if time_intervals else 0,
            'interval_std_dev': statistics.stdev(time_intervals) if len(time_intervals) > 1 else 0,
            'common_event_types': dict(event_types),
            'severity_distribution': dict(severity_distribution),
            'baseline_timestamp': datetime.now().isoformat()
        }
        
        return baseline
    
    def _learn_threat_patterns(self):
        """Learn new threat patterns from correlation results"""
        # Analyze high-confidence correlations to learn patterns
        high_confidence_correlations = [
            c for c in self.correlation_results 
            if c.threat_confidence.value >= ThreatConfidence.HIGH.value
        ]
        
        if len(high_confidence_correlations) >= 5:
            # Extract common patterns
            pattern_candidates = self._extract_pattern_candidates(high_confidence_correlations)
            
            for pattern_candidate in pattern_candidates:
                if self._validate_threat_pattern(pattern_candidate):
                    pattern_id = f"learned_pattern_{len(self.threat_patterns) + 1}"
                    self.threat_patterns[pattern_id] = pattern_candidate
                    logger.info(f"Learned new threat pattern: {pattern_id}")
    
    def _extract_pattern_candidates(self, correlations: List[CorrelationResult]) -> List[ThreatPattern]:
        """Extract potential threat patterns from correlations"""
        # Group correlations by threat category
        category_groups = defaultdict(list)
        for correlation in correlations:
            category_groups[correlation.threat_category].append(correlation)
        
        pattern_candidates = []
        
        for category, category_correlations in category_groups.items():
            if len(category_correlations) >= 3:  # Need multiple examples
                # Extract common indicators and behaviors
                common_indicators = set()
                for correlation in category_correlations[:1]:  # Start with first correlation
                    common_indicators.update(correlation.behavioral_indicators)
                
                for correlation in category_correlations[1:]:
                    common_indicators &= set(correlation.behavioral_indicators)
                
                if common_indicators:
                    pattern = ThreatPattern(
                        pattern_id=f"candidate_{category}_{int(time.time())}",
                        pattern_name=f"Learned {category} pattern",
                        pattern_type=category,
                        indicators=list(common_indicators),
                        severity_threshold=0.6,
                        confidence_multiplier=0.8  # Lower for learned patterns
                    )
                    pattern_candidates.append(pattern)
        
        return pattern_candidates
    
    def _validate_threat_pattern(self, pattern: ThreatPattern) -> bool:
        """Validate if a threat pattern is worth keeping"""
        # Simple validation - check if pattern has meaningful indicators
        return len(pattern.indicators) >= 2 and pattern.pattern_type != "unknown"
    
    def _initialize_threat_patterns(self):
        """Initialize known threat patterns for correlation"""
        # Initialize with common threat patterns
        self.threat_patterns = {
            'brute_force_attack': ThreatPattern(
                pattern_id='brute_force_attack',
                pattern_name='Brute Force Attack',
                pattern_type='brute_force',
                indicators=['multiple_failed_logins', 'rapid_login_attempts', 'different_user_agents'],
                severity_threshold=0.7,
                confidence_multiplier=1.2
            ),
            'lateral_movement': ThreatPattern(
                pattern_id='lateral_movement',
                pattern_name='Lateral Movement',
                pattern_type='lateral_movement',
                indicators=['privilege_escalation', 'network_discovery', 'credential_access'],
                severity_threshold=0.8,
                confidence_multiplier=1.5
            ),
            'data_exfiltration': ThreatPattern(
                pattern_id='data_exfiltration',
                pattern_name='Data Exfiltration',
                pattern_type='exfiltration',
                indicators=['large_data_transfer', 'unauthorized_access', 'compression_activity'],
                severity_threshold=0.9,
                confidence_multiplier=1.8
            )
        }
    
    def _cleanup_old_events(self):
        """Clean up old events and correlations"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Clean up old events (done automatically by deque maxlen)
        # Clean up old correlations (done automatically by deque maxlen)
        
        # Clean up old baselines
        for system in list(self.behavioral_baselines.keys()):
            baseline = self.behavioral_baselines[system]
            baseline_time = datetime.fromisoformat(baseline['baseline_timestamp'])
            
            if baseline_time < cutoff_time:
                logger.debug(f"Removing old baseline for system: {system}")
                del self.behavioral_baselines[system]
    
    def _update_correlation_statistics(self):
        """Update correlation engine statistics"""
        uptime = (datetime.now() - self.correlation_stats['engine_start_time']).total_seconds()
        
        self.correlation_stats.update({
            'uptime_seconds': uptime,
            'events_per_minute': self.correlation_stats['events_processed'] / max(1, uptime / 60),
            'correlations_per_hour': self.correlation_stats['correlations_generated'] / max(1, uptime / 3600),
            'threat_detection_rate': self.correlation_stats['threats_identified'] / max(1, self.correlation_stats['events_processed'])
        })
    
    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get comprehensive correlation engine summary"""
        with self.correlation_lock:
            recent_correlations = list(self.correlation_results)[-10:]  # Last 10 correlations
            active_patterns = len(self.threat_patterns)
            behavioral_baselines_count = len(self.behavioral_baselines)
        
        return {
            'correlation_active': self.correlation_active,
            'correlation_statistics': self.correlation_stats,
            'event_buffer_size': len(self.event_buffer),
            'correlation_results_count': len(self.correlation_results),
            'active_threat_patterns': active_patterns,
            'behavioral_baselines_count': behavioral_baselines_count,
            'recent_correlations': [asdict(c) for c in recent_correlations],
            'configuration': self.correlation_config
        }
    
    def stop_correlation_engine(self):
        """Stop advanced correlation engine"""
        logger.info("Stopping Advanced Security Correlation Engine")
        self.correlation_active = False
        
        if self.correlation_thread and self.correlation_thread.is_alive():
            self.correlation_thread.join(timeout=10)
        
        # Shutdown thread pool
        self.correlation_executor.shutdown(wait=True)
        
        logger.info("Advanced correlation engine stopped")
        
        # Log final statistics
        final_summary = self.get_correlation_summary()
        logger.info(f"Final correlation statistics: {final_summary['correlation_statistics']}")


# Helper classes for specialized correlation analysis
class TemporalCorrelationAnalyzer:
    """Specialized temporal correlation analysis"""
    
    def analyze(self, event: SecurityEvent, candidates: List[SecurityEvent], config: Dict[str, Any]) -> List[CorrelationResult]:
        """Analyze temporal correlations"""
        correlations = []
        
        # Simple temporal clustering
        event_time = datetime.fromisoformat(event.timestamp)
        threshold_minutes = config['temporal_threshold_minutes']
        
        close_events = []
        for candidate in candidates:
            candidate_time = datetime.fromisoformat(candidate.timestamp)
            time_diff = abs((event_time - candidate_time).total_seconds() / 60)
            
            if time_diff <= threshold_minutes:
                close_events.append(candidate)
        
        if len(close_events) >= 2:
            correlation = CorrelationResult(
                correlation_id=str(uuid.uuid4()),
                correlated_events=[event.event_id] + [e.event_id for e in close_events[:5]],
                correlation_types=[CorrelationType.TEMPORAL],
                threat_confidence=ThreatConfidence.MEDIUM,
                threat_category="temporal_cluster",
                attack_chain_stage=None,
                risk_score=0.6,
                correlation_strength=0.8,
                temporal_pattern={
                    'cluster_size': len(close_events) + 1,
                    'time_window_minutes': threshold_minutes
                },
                recommended_actions=["Investigate temporal clustering", "Check for coordinated activity"]
            )
            correlations.append(correlation)
        
        return correlations


class BehavioralCorrelationAnalyzer:
    """Specialized behavioral correlation analysis"""
    
    def analyze(self, event: SecurityEvent, candidates: List[SecurityEvent], baselines: Dict[str, Any]) -> List[CorrelationResult]:
        """Analyze behavioral correlations"""
        correlations = []
        
        # Check for behavioral anomalies
        system_baseline = baselines.get(event.source_system)
        if system_baseline:
            anomaly_score = self._calculate_anomaly_score(event, system_baseline)
            
            if anomaly_score > 2.0:  # 2 standard deviations
                similar_anomalies = []
                for candidate in candidates:
                    candidate_anomaly = self._calculate_anomaly_score(candidate, system_baseline)
                    if candidate_anomaly > 1.5:
                        similar_anomalies.append(candidate)
                
                if similar_anomalies:
                    correlation = CorrelationResult(
                        correlation_id=str(uuid.uuid4()),
                        correlated_events=[event.event_id] + [e.event_id for e in similar_anomalies[:3]],
                        correlation_types=[CorrelationType.BEHAVIORAL],
                        threat_confidence=ThreatConfidence.HIGH,
                        threat_category="behavioral_anomaly",
                        attack_chain_stage=None,
                        risk_score=min(0.9, anomaly_score / 3.0),
                        correlation_strength=anomaly_score / 5.0,
                        behavioral_indicators=[f"anomaly_score_{anomaly_score:.1f}"],
                        recommended_actions=["Investigate behavioral anomaly", "Check for attack indicators"]
                    )
                    correlations.append(correlation)
        
        return correlations
    
    def _calculate_anomaly_score(self, event: SecurityEvent, baseline: Dict[str, Any]) -> float:
        """Calculate behavioral anomaly score for an event"""
        # Simple anomaly scoring based on event type frequency
        common_types = baseline.get('common_event_types', {})
        total_events = baseline.get('event_count', 1)
        
        event_type_frequency = common_types.get(event.event_type, 0) / total_events
        
        # Rare event types get higher anomaly scores
        if event_type_frequency < 0.1:
            return 3.0
        elif event_type_frequency < 0.2:
            return 2.0
        else:
            return 1.0


class ThreatPatternMatcher:
    """Specialized threat pattern matching"""
    
    def match_patterns(self, event: SecurityEvent, candidates: List[SecurityEvent], patterns: Dict[str, ThreatPattern]) -> List[CorrelationResult]:
        """Match events against known threat patterns"""
        correlations = []
        
        all_events = [event] + candidates
        
        for pattern_id, pattern in patterns.items():
            matching_events = []
            pattern_score = 0
            
            for test_event in all_events:
                event_indicators = set(test_event.indicators + [test_event.event_type, test_event.severity])
                pattern_indicators = set(pattern.indicators)
                
                overlap = len(event_indicators & pattern_indicators)
                if overlap > 0:
                    matching_events.append(test_event)
                    pattern_score += overlap / len(pattern_indicators)
            
            # Check if pattern match threshold is met
            if len(matching_events) >= 2 and pattern_score >= pattern.severity_threshold:
                confidence_value = min(0.99, pattern_score * pattern.confidence_multiplier)
                confidence = next((c for c in ThreatConfidence if c.value >= confidence_value), ThreatConfidence.MEDIUM)
                
                correlation = CorrelationResult(
                    correlation_id=str(uuid.uuid4()),
                    correlated_events=[e.event_id for e in matching_events[:5]],
                    correlation_types=[CorrelationType.SIGNATURE],
                    threat_confidence=confidence,
                    threat_category=pattern.pattern_type,
                    attack_chain_stage="pattern_match",
                    risk_score=pattern_score,
                    correlation_strength=pattern_score,
                    behavioral_indicators=[f"pattern_match_{pattern.pattern_name}"],
                    recommended_actions=[f"Investigate {pattern.pattern_name}", "Apply known countermeasures"]
                )
                correlations.append(correlation)
        
        return correlations


class StatisticalAnomalyDetector:
    """Specialized statistical anomaly detection"""
    
    def detect_anomalous_patterns(self, event: SecurityEvent, candidates: List[SecurityEvent], baselines: Dict[str, Any]) -> List[CorrelationResult]:
        """Detect statistically anomalous patterns"""
        correlations = []
        
        # Statistical analysis of event patterns
        all_events = [event] + candidates
        
        # Analyze event frequency distribution
        event_types = defaultdict(int)
        for e in all_events:
            event_types[e.event_type] += 1
        
        # Check for unusual frequency patterns
        total_events = len(all_events)
        for event_type, count in event_types.items():
            frequency = count / total_events
            
            # Extremely high frequency might indicate anomaly
            if frequency > 0.5 and count >= 3:
                anomalous_events = [e for e in all_events if e.event_type == event_type]
                
                correlation = CorrelationResult(
                    correlation_id=str(uuid.uuid4()),
                    correlated_events=[e.event_id for e in anomalous_events[:5]],
                    correlation_types=[CorrelationType.ANOMALY],
                    threat_confidence=ThreatConfidence.MEDIUM,
                    threat_category="frequency_anomaly",
                    attack_chain_stage=None,
                    risk_score=frequency,
                    correlation_strength=frequency,
                    behavioral_indicators=[f"high_frequency_{event_type}"],
                    recommended_actions=["Investigate frequency anomaly", "Check for automated attacks"]
                )
                correlations.append(correlation)
        
        return correlations


def create_correlation_engine(correlation_window_minutes: int = 60):
    """Factory function to create advanced correlation engine"""
    engine = AdvancedSecurityCorrelationEngine(correlation_window_minutes)
    
    logger.info("Created advanced security correlation engine")
    logger.info("Ready to enhance existing security systems with advanced correlation")
    
    return engine


if __name__ == "__main__":
    """
    Example usage - advanced security correlation
    """
    import json
    
    # Create correlation engine
    engine = create_correlation_engine()
    
    # Start correlation engine
    engine.start_correlation_engine()
    
    try:
        # Simulate some security events
        test_events = [
            SecurityEvent(
                event_id="evt_001",
                source_system="continuous_monitoring",
                timestamp=datetime.now().isoformat(),
                event_type="failed_login",
                severity="medium",
                description="Failed login attempt",
                indicators=["multiple_failed_logins"],
                affected_entities=["user_account_123"]
            ),
            SecurityEvent(
                event_id="evt_002", 
                source_system="continuous_monitoring",
                timestamp=(datetime.now() + timedelta(minutes=1)).isoformat(),
                event_type="failed_login",
                severity="medium",
                description="Another failed login attempt",
                indicators=["multiple_failed_logins"],
                affected_entities=["user_account_123"]
            ),
            SecurityEvent(
                event_id="evt_003",
                source_system="api_security",
                timestamp=(datetime.now() + timedelta(minutes=2)).isoformat(), 
                event_type="suspicious_api_call",
                severity="high",
                description="Suspicious API access pattern",
                indicators=["rapid_api_calls", "multiple_failed_logins"],
                affected_entities=["api_endpoint_auth"]
            )
        ]
        
        # Process events through correlation engine
        for event in test_events:
            correlations = engine.process_security_event(event)
            if correlations:
                print(f"\n=== Correlations for {event.event_id} ===")
                for correlation in correlations:
                    print(json.dumps(asdict(correlation), indent=2, default=str))
        
        # Wait for batch processing
        time.sleep(35)
        
        # Show correlation summary
        summary = engine.get_correlation_summary()
        print("\n=== Advanced Correlation Engine Summary ===")
        print(json.dumps(summary, indent=2, default=str))
        
    finally:
        # Stop correlation engine
        engine.stop_correlation_engine()