"""
Advanced ML Integrity Guardian System
====================================
"""Core Module - Split from integrity_ml_guardian.py"""


import logging
import hashlib
import json
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import asyncio

# ML imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN, KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score


logger = logging.getLogger(__name__)

class MLIntegrityStatus(Enum):
    """ML-enhanced integrity status"""
    VERIFIED = "verified"
    CORRUPTED = "corrupted"
    TAMPERED = "tampered"
    MISSING = "missing"
    RECOVERED = "recovered"
    ML_PREDICTED_RISK = "ml_predicted_risk"
    ANOMALY_DETECTED = "anomaly_detected"

class IntegrityRiskLevel(Enum):
    """ML-assessed integrity risk levels"""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MLAlgorithm(Enum):
    """ML algorithms for integrity analysis"""
    ANOMALY_DETECTION = "anomaly_detection"
    PATTERN_CLASSIFICATION = "pattern_classification"
    ENSEMBLE_VERIFICATION = "ensemble_verification"
    PREDICTIVE_CORRUPTION = "predictive_corruption"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"

@dataclass
class MLIntegrityRecord:
    """ML-enhanced integrity record"""
    analytics_id: str
    original_checksum: str
    verification_checksum: str
    status: MLIntegrityStatus
    created_at: datetime
    
    # ML analysis features
    ml_features: List[float] = field(default_factory=list)
    anomaly_score: float = 0.0
    tamper_probability: float = 0.0
    corruption_risk: IntegrityRiskLevel = IntegrityRiskLevel.MINIMAL
    ml_confidence: float = 0.0
    
    # Predictive insights
    predicted_failure_time: Optional[datetime] = None
    integrity_trend: str = "stable"
    verification_frequency_score: float = 0.0
    
    # Advanced tracking
    verified_at: Optional[datetime] = None
    error_message: Optional[str] = None
    recovery_attempted: bool = False
    ml_recovery_strategy: Optional[str] = None

@dataclass
class IntegrityThreatProfile:
    """ML-driven threat profile for data integrity"""
    analytics_id: str
    threat_level: IntegrityRiskLevel
    threat_vectors: List[str]
    attack_patterns: List[str]
    ml_indicators: Dict[str, float]
    behavioral_anomalies: List[str]
    recommendation: str
    confidence_score: float

class AdvancedMLIntegrityGuardian:
    """
    Advanced ML-driven integrity guardian with predictive threat detection,
    intelligent verification scheduling, and adaptive recovery strategies.
    """
    
    def __init__(self, ml_enabled: bool = True, verification_interval: float = 30.0):
        """
        Initialize ML integrity guardian.
        
        Args:
            ml_enabled: Enable ML-enhanced features
            verification_interval: Base verification interval in seconds
        """
        self.ml_enabled = ml_enabled
        self.verification_interval = verification_interval
        
        # ML models for integrity protection
        self.anomaly_detector: Optional[IsolationForest] = None
        self.tamper_classifier: Optional[RandomForestClassifier] = None
        self.corruption_predictor: Optional[LogisticRegression] = None
        self.pattern_analyzer: Optional[DBSCAN] = None
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Integrity tracking
        self.integrity_records: Dict[str, MLIntegrityRecord] = {}
        self.analytics_checksums: Dict[str, Dict[str, Any]] = {}
        self.verified_analytics: Set[str] = set()
        
        # ML-enhanced backup and recovery
        self.ml_backup_storage: Dict[str, Dict[str, Any]] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.threat_profiles: Dict[str, IntegrityThreatProfile] = {}
        
        # ML feature tracking
        self.integrity_features_history: deque = deque(maxlen=2000)
        self.verification_patterns: defaultdict = defaultdict(list)
        self.tamper_patterns: defaultdict = defaultdict(list)
        
        # Adaptive verification scheduling
        self.adaptive_intervals: Dict[str, float] = {}
        self.priority_verification_queue: deque = deque()
        self.verification_predictions: Dict[str, float] = {}
        
        # ML configuration
        self.ml_config = {
            "anomaly_threshold": 0.8,
            "tamper_confidence_threshold": 0.7,
            "corruption_risk_threshold": 0.6,
            "adaptive_scheduling": True,
            "min_training_samples": 30,
            "model_retrain_hours": 12,
            "feature_extraction_depth": 15,
            "pattern_analysis_window": 100,
            "predictive_horizon_hours": 24
        }
        
        # Performance statistics
        self.guardian_stats = {
            'total_verifications': 0,
            'ml_verifications': 0,
            'anomalies_detected': 0,
            'tamper_attempts_blocked': 0,
            'corruption_predictions': 0,
            'successful_recoveries': 0,
            'ml_accuracy': 0.0,
            'false_positives': 0,
            'false_negatives': 0,
            'adaptive_optimizations': 0,
            'threat_profiles_generated': 0,
            'start_time': datetime.now()
        }
        
        # Background ML processing
        self.ml_guardian_active = False
        self.ml_verification_worker: Optional[threading.Thread] = None
        self.ml_analysis_worker: Optional[threading.Thread] = None
        self.ml_training_worker: Optional[threading.Thread] = None
        
        # Thread safety
        self.guardian_lock = threading.RLock()
        
        self._initialize_ml_models()
        self._setup_recovery_strategies()
        
        logger.info("Advanced ML Integrity Guardian initialized")
    
    def start_ml_guardian(self):
        """Start ML-enhanced integrity guardian"""
        if self.ml_guardian_active:
            return
        
        self.ml_guardian_active = True
        
        # Start ML workers
        self.ml_verification_worker = threading.Thread(
            target=self._ml_verification_loop, daemon=True)
        self.ml_analysis_worker = threading.Thread(
            target=self._ml_analysis_loop, daemon=True)
        self.ml_training_worker = threading.Thread(
            target=self._ml_training_loop, daemon=True)
        
        self.ml_verification_worker.start()
        self.ml_analysis_worker.start()
        self.ml_training_worker.start()
        
        logger.info("ML Integrity Guardian started")
    
    def stop_ml_guardian(self):
        """Stop ML integrity guardian"""
        self.ml_guardian_active = False
        
        # Wait for workers to finish
        for worker in [self.ml_verification_worker, self.ml_analysis_worker, self.ml_training_worker]:
            if worker and worker.is_alive():
                worker.join(timeout=5)
        
        logger.info("ML Integrity Guardian stopped")
    
    def register_analytics_ml(self, analytics_id: str, analytics_data: Dict[str, Any],
                             priority: IntegrityRiskLevel = IntegrityRiskLevel.MEDIUM) -> str:
        """Register analytics with ML-enhanced integrity protection"""
        with self.guardian_lock:
            # Extract ML features for analysis
            ml_features = self._extract_integrity_features(analytics_data)
            
            # Generate enhanced checksum
            checksum = self._generate_ml_enhanced_checksum(analytics_data)
            
            # Analyze initial risk profile
            threat_profile = self._analyze_threat_profile(analytics_id, analytics_data, ml_features)
            
            # Predict optimal verification frequency
            verification_frequency = self._predict_verification_frequency(
                analytics_id, ml_features, threat_profile.threat_level
            )
            
            # Create ML integrity record
            record = MLIntegrityRecord(
                analytics_id=analytics_id,
                original_checksum=checksum,
                verification_checksum="",
                status=MLIntegrityStatus.VERIFIED,
                created_at=datetime.now(),
                ml_features=ml_features,
                corruption_risk=threat_profile.threat_level,
                verification_frequency_score=verification_frequency,
                integrity_trend="stable"
            )
            
            # Store records and analysis
            self.integrity_records[analytics_id] = record
            self.analytics_checksums[analytics_id] = {
                'checksum': checksum,
                'data': analytics_data,
                'ml_features': ml_features,
                'registration_time': datetime.now()
            }
            self.threat_profiles[analytics_id] = threat_profile
            
            # Schedule adaptive verification
            if self.ml_config["adaptive_scheduling"]:
                self.adaptive_intervals[analytics_id] = verification_frequency
            
            # Create ML-enhanced backup
            self._create_ml_backup(analytics_id, analytics_data, ml_features)
            
            self.guardian_stats['total_verifications'] += 1
            self.guardian_stats['threat_profiles_generated'] += 1
            
            logger.debug(f"Registered analytics with ML protection: {analytics_id}")
            return checksum
    
    def verify_analytics_ml(self, analytics_id: str, current_data: Dict[str, Any]) -> MLIntegrityRecord:
        """Perform ML-enhanced integrity verification"""
        with self.guardian_lock:
            self.guardian_stats['total_verifications'] += 1
            
            if analytics_id not in self.integrity_records:
                # Register new analytics
                return self.register_analytics_ml(analytics_id, current_data)
            
            record = self.integrity_records[analytics_id]
            
            # Extract current ML features
            current_features = self._extract_integrity_features(current_data)
            
            # Perform multi-layer ML verification
            verification_results = self._perform_ml_verification(
                analytics_id, current_data, current_features, record
            )
            
            # Update record with ML analysis
            record.verification_checksum = verification_results['checksum']
            record.verified_at = datetime.now()
            record.ml_features = current_features
            record.anomaly_score = verification_results['anomaly_score']
            record.tamper_probability = verification_results['tamper_probability']
            record.ml_confidence = verification_results['ml_confidence']
            
            # Determine ML-enhanced status
            if verification_results['integrity_valid']:
                record.status = MLIntegrityStatus.VERIFIED
                record.error_message = None
                self.verified_analytics.add(analytics_id)
                
                # Update threat profile based on verification
                self._update_threat_profile(analytics_id, verification_results)
                
            else:
                # Integrity violation detected
                self._handle_integrity_violation(record, verification_results, current_data)
            
            # Update verification patterns for ML learning
            self._record_verification_pattern(analytics_id, verification_results)
            
            # Adaptive verification scheduling
            if self.ml_config["adaptive_scheduling"]:
                self._update_verification_schedule(analytics_id, verification_results)
            
            self.guardian_stats['ml_verifications'] += 1
            
            return record
    
    def _extract_integrity_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract ML features for integrity analysis"""
        try:
            features = []
            
            # Data structure features
            features.append(float(len(data)))  # Data size
            features.append(float(self._calculate_data_depth(data)))  # Nesting depth
            features.append(float(len(str(data))))  # Serialized size
            
            # Content diversity features
            type_counts = self._count_data_types(data)
            features.extend([
                type_counts.get('dict', 0),
                type_counts.get('list', 0),
                type_counts.get('str', 0),
                type_counts.get('int', 0),
                type_counts.get('float', 0)
            ])
            
            # Temporal features
            current_time = datetime.now()
            features.append(float(current_time.hour))
            features.append(float(current_time.minute))
            features.append(float(current_time.weekday()))
            
            # Data complexity features
            complexity_score = self._calculate_data_complexity(data)
            features.append(complexity_score)
            
            # Hash-based features for pattern detection
            data_hash = hash(str(sorted(data.keys()))) % 10000
            features.append(float(data_hash))
            
            # Content entropy (randomness measure)
            entropy = self._calculate_content_entropy(data)
            features.append(entropy)
            
            # Ensure consistent feature count
            while len(features) < self.ml_config["feature_extraction_depth"]:
                features.append(0.0)
            
            return features[:self.ml_config["feature_extraction_depth"]]
            
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return [0.0] * self.ml_config["feature_extraction_depth"]
    
    def _calculate_data_depth(self, obj, current_depth=0) -> int:
        """Calculate maximum nesting depth"""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_data_depth(value, current_depth + 1) 
                      for value in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._calculate_data_depth(item, current_depth + 1) 
                      for item in obj)
        else:
            return current_depth
    
    def _count_data_types(self, obj, type_counts=None) -> Dict[str, int]:
        """Count occurrences of different data types"""
        if type_counts is None:
            type_counts = defaultdict(int)
        
        type_name = type(obj).__name__
        type_counts[type_name] += 1
        
        if isinstance(obj, dict):
            for value in obj.values():
                self._count_data_types(value, type_counts)
        elif isinstance(obj, list):
            for item in obj:
                self._count_data_types(item, type_counts)
        
        return dict(type_counts)
    
    def _calculate_data_complexity(self, data: Dict[str, Any]) -> float:
        """Calculate data structure complexity score"""
        try:
            complexity = 0.0
            
            def analyze_complexity(obj, depth=0):
                nonlocal complexity
                complexity += depth * 0.1
                
                if isinstance(obj, dict):
                    complexity += len(obj) * 0.1
                    for value in obj.values():
                        if isinstance(value, (dict, list)):
                            analyze_complexity(value, depth + 1)
                elif isinstance(obj, list):
                    complexity += len(obj) * 0.05
                    for item in obj:
                        if isinstance(item, (dict, list)):
                            analyze_complexity(item, depth + 1)
            
            analyze_complexity(data)
            return min(complexity, 10.0)  # Cap at 10.0
            
        except Exception:
            return 1.0
    
    def _calculate_content_entropy(self, data: Dict[str, Any]) -> float:
        """Calculate content entropy for randomness detection"""
        try:
            content_str = json.dumps(data, sort_keys=True)
            byte_counts = defaultdict(int)
            
            for byte in content_str.encode():
                byte_counts[byte] += 1
            
            total_bytes = len(content_str.encode())
            entropy = 0.0
            
            for count in byte_counts.values():
                probability = count / total_bytes
                if probability > 0:
                    entropy -= probability * np.log2(probability)
            
            return min(entropy / 8.0, 1.0)  # Normalize to 0-1
            
        except Exception:
            return 0.5
    
    def _generate_ml_enhanced_checksum(self, data: Dict[str, Any]) -> str:
        """Generate ML-enhanced checksum with multiple algorithms"""
        try:
            # Normalize data for consistent checksums
            normalized_data = self._normalize_data_for_checksum(data)
            data_bytes = json.dumps(normalized_data, sort_keys=True).encode('utf-8')
            
            # Generate multiple checksums for enhanced security
            md5_hash = hashlib.md5(data_bytes).hexdigest()
            sha256_hash = hashlib.sha256(data_bytes).hexdigest()
            
            # Combine with ML features hash
            ml_features = self._extract_integrity_features(data)
            features_hash = hashlib.sha256(str(ml_features).encode()).hexdigest()[:16]
            
            # Create composite checksum
            composite = f"ml:{md5_hash[:8]}|{sha256_hash[:16]}|{features_hash}"
            return composite
            
        except Exception as e:
            logger.error(f"ML checksum generation failed: {e}")
            return "error_generating_ml_checksum"
    
    def _normalize_data_for_checksum(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data for consistent checksum generation"""
        try:
            normalized = {}
            
            for key, value in data.items():
                if isinstance(value, dict):
                    normalized[key] = self._normalize_data_for_checksum(value)
                elif isinstance(value, list):
                    # Sort lists if possible for consistency
                    try:
                        normalized[key] = sorted(value) if all(isinstance(x, (str, int, float)) for x in value) else value
                    except TypeError:
                        normalized[key] = value
                elif isinstance(value, float):
                    # Round floats to avoid precision issues
                    normalized[key] = round(value, 6)
                elif isinstance(value, datetime):
                    # Convert datetime to ISO string
                    normalized[key] = value.isoformat()
                else:
                    normalized[key] = value
            
            return normalized
            
        except Exception as e:
            logger.error(f"Data normalization failed: {e}")