"""
Advanced ML Integrity Guardian System
====================================
"""ML Algorithms Module - Split from integrity_ml_guardian.py"""


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


            return data
    
    def _analyze_threat_profile(self, analytics_id: str, data: Dict[str, Any], 
                               features: List[float]) -> IntegrityThreatProfile:
        """Analyze and generate threat profile using ML"""
        try:
            # Calculate threat indicators
            threat_indicators = {}
            
            # Data complexity threat (complex data more likely to be tampered)
            complexity = features[13] if len(features) > 13 else 0.5
            threat_indicators['complexity_risk'] = min(complexity / 5.0, 1.0)
            
            # Size anomaly threat (unusually large/small data)
            data_size = features[0] if features else 0
            if data_size > 1000:
                threat_indicators['size_anomaly'] = min((data_size - 1000) / 5000, 1.0)
            else:
                threat_indicators['size_anomaly'] = 0.0
            
            # Temporal threat (activity at unusual times)
            hour = features[8] if len(features) > 8 else 12
            if hour < 6 or hour > 22:  # Night hours
                threat_indicators['temporal_anomaly'] = 0.3
            else:
                threat_indicators['temporal_anomaly'] = 0.0
            
            # Content entropy threat (highly random content)
            entropy = features[14] if len(features) > 14 else 0.5
            if entropy > 0.8:
                threat_indicators['entropy_anomaly'] = (entropy - 0.8) / 0.2
            else:
                threat_indicators['entropy_anomaly'] = 0.0
            
            # Calculate overall threat level
            avg_threat = np.mean(list(threat_indicators.values()))
            
            if avg_threat >= 0.8:
                threat_level = IntegrityRiskLevel.CRITICAL
            elif avg_threat >= 0.6:
                threat_level = IntegrityRiskLevel.HIGH
            elif avg_threat >= 0.4:
                threat_level = IntegrityRiskLevel.MEDIUM
            elif avg_threat >= 0.2:
                threat_level = IntegrityRiskLevel.LOW
            else:
                threat_level = IntegrityRiskLevel.MINIMAL
            
            # Generate threat vectors and patterns
            threat_vectors = []
            attack_patterns = []
            
            if threat_indicators['complexity_risk'] > 0.5:
                threat_vectors.append("complex_data_manipulation")
                attack_patterns.append("structure_modification")
            
            if threat_indicators['size_anomaly'] > 0.3:
                threat_vectors.append("data_injection_attack")
                attack_patterns.append("volume_anomaly")
            
            if threat_indicators['temporal_anomaly'] > 0.2:
                threat_vectors.append("off_hours_tampering")
                attack_patterns.append("temporal_attack")
            
            if threat_indicators['entropy_anomaly'] > 0.3:
                threat_vectors.append("content_scrambling")
                attack_patterns.append("randomization_attack")
            
            # Generate behavioral anomalies
            behavioral_anomalies = []
            if avg_threat > 0.5:
                behavioral_anomalies.append("unusual_data_patterns")
            if len(threat_vectors) > 2:
                behavioral_anomalies.append("multiple_threat_vectors")
            
            # Generate recommendation
            if threat_level in [IntegrityRiskLevel.CRITICAL, IntegrityRiskLevel.HIGH]:
                recommendation = "Increase verification frequency and enable enhanced monitoring"
            elif threat_level == IntegrityRiskLevel.MEDIUM:
                recommendation = "Apply standard verification with periodic deep scans"
            else:
                recommendation = "Standard verification sufficient"
            
            # Calculate confidence score
            confidence_score = 1.0 - (np.std(list(threat_indicators.values())) / 2.0)
            confidence_score = max(0.5, min(1.0, confidence_score))
            
            return IntegrityThreatProfile(
                analytics_id=analytics_id,
                threat_level=threat_level,
                threat_vectors=threat_vectors,
                attack_patterns=attack_patterns,
                ml_indicators=threat_indicators,
                behavioral_anomalies=behavioral_anomalies,
                recommendation=recommendation,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Threat profile analysis failed: {e}")
            return IntegrityThreatProfile(
                analytics_id=analytics_id,
                threat_level=IntegrityRiskLevel.MEDIUM,
                threat_vectors=["unknown"],
                attack_patterns=["unknown"],
                ml_indicators={},
                behavioral_anomalies=[],
                recommendation="Apply standard verification",
                confidence_score=0.5
            )
    
    def _predict_verification_frequency(self, analytics_id: str, features: List[float], 
                                      threat_level: IntegrityRiskLevel) -> float:
        """Predict optimal verification frequency using ML"""
        try:
            # Base frequency from threat level
            base_frequencies = {
                IntegrityRiskLevel.CRITICAL: 10.0,    # Every 10 seconds
                IntegrityRiskLevel.HIGH: 30.0,        # Every 30 seconds  
                IntegrityRiskLevel.MEDIUM: 60.0,      # Every minute
                IntegrityRiskLevel.LOW: 300.0,        # Every 5 minutes
                IntegrityRiskLevel.MINIMAL: 600.0     # Every 10 minutes
            }
            
            base_frequency = base_frequencies.get(threat_level, 60.0)
            
            # Adjust based on ML features
            if len(features) >= 10:
                # Complexity adjustment
                complexity = features[13] if len(features) > 13 else 1.0
                if complexity > 3.0:
                    base_frequency *= 0.7  # More frequent for complex data
                
                # Size adjustment
                data_size = features[0]
                if data_size > 500:
                    base_frequency *= 0.8  # More frequent for large data
                
                # Activity pattern adjustment
                hour = features[8]
                if hour < 6 or hour > 22:
                    base_frequency *= 0.5  # More frequent during off-hours
            
            return max(5.0, min(3600.0, base_frequency))  # Clamp between 5s and 1h
            
        except Exception:
            return 60.0  # Default 1 minute
    
    def _perform_ml_verification(self, analytics_id: str, current_data: Dict[str, Any],
                                current_features: List[float], record: MLIntegrityRecord) -> Dict[str, Any]:
        """Perform comprehensive ML-enhanced verification"""
        try:
            results = {
                'integrity_valid': True,
                'checksum': '',
                'anomaly_score': 0.0,
                'tamper_probability': 0.0,
                'ml_confidence': 0.0,
                'verification_method': 'ml_enhanced',
                'threat_indicators': {}
            }
            
            # Generate verification checksum
            verification_checksum = self._generate_ml_enhanced_checksum(current_data)
            results['checksum'] = verification_checksum
            
            # Compare checksums
            checksum_match = verification_checksum == record.original_checksum
            results['integrity_valid'] = checksum_match
            
            # ML anomaly detection
            if self.ml_enabled and self.anomaly_detector:
                anomaly_score = self._detect_anomalies_ml(current_features)
                results['anomaly_score'] = anomaly_score
                
                if anomaly_score > self.ml_config["anomaly_threshold"]:
                    results['integrity_valid'] = False
                    results['threat_indicators']['anomaly_detected'] = anomaly_score
            
            # ML tamper detection
            if self.ml_enabled and self.tamper_classifier:
                tamper_probability = self._detect_tampering_ml(
                    analytics_id, current_data, current_features
                )
                results['tamper_probability'] = tamper_probability
                
                if tamper_probability > self.ml_config["tamper_confidence_threshold"]:
                    results['integrity_valid'] = False
                    results['threat_indicators']['tampering_detected'] = tamper_probability
            
            # Pattern analysis
            if self.ml_enabled and self.pattern_analyzer:
                pattern_anomaly = self._analyze_patterns_ml(analytics_id, current_features)
                if pattern_anomaly > 0.6:
                    results['integrity_valid'] = False
                    results['threat_indicators']['pattern_anomaly'] = pattern_anomaly
            
            # Calculate ML confidence
            confidence_factors = []
            if checksum_match:
                confidence_factors.append(0.9)
            if results['anomaly_score'] < 0.3:
                confidence_factors.append(0.8)
            if results['tamper_probability'] < 0.3:
                confidence_factors.append(0.8)
            
            results['ml_confidence'] = np.mean(confidence_factors) if confidence_factors else 0.5
            
            return results
            
        except Exception as e:
            logger.error(f"ML verification failed: {e}")
            return {
                'integrity_valid': False,
                'checksum': '',
                'anomaly_score': 1.0,
                'tamper_probability': 1.0,
                'ml_confidence': 0.0,
                'verification_method': 'fallback',
                'error': str(e)
            }
    
    def _detect_anomalies_ml(self, features: List[float]) -> float:
        """Detect anomalies using ML models"""
        try:
            if not self.anomaly_detector or len(features) < 10:
                return 0.0
            
            # Scale features
            if 'anomaly_detection' in self.scalers:
                features_scaled = self.scalers['anomaly_detection'].transform([features])
            else:
                features_scaled = [features]
            
            # Get anomaly score
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            
            # Convert to 0-1 range (negative scores indicate anomalies)
            normalized_score = max(0, 1 - (anomaly_score + 1) / 2)
            
            return normalized_score
            
        except Exception as e:
            logger.debug(f"ML anomaly detection error: {e}")
            return 0.0
    
    def _detect_tampering_ml(self, analytics_id: str, current_data: Dict[str, Any], 
                            features: List[float]) -> float:
        """Detect tampering using ML classification"""
        try:
            if not self.tamper_classifier or analytics_id not in self.analytics_checksums:
                return 0.0
            
            original_data = self.analytics_checksums[analytics_id]['data']
            original_features = self.analytics_checksums[analytics_id]['ml_features']
            
            # Calculate feature differences
            feature_diffs = []
            for i in range(min(len(features), len(original_features))):
                diff = abs(features[i] - original_features[i])
                feature_diffs.append(diff)
            
            # Pad or truncate to consistent length
            while len(feature_diffs) < 10:
                feature_diffs.append(0.0)
            feature_diffs = feature_diffs[:10]
            
            # Additional tampering indicators
            structural_changes = self._detect_structural_changes(original_data, current_data)
            feature_diffs.extend(structural_changes)
            
            # Scale features
            if 'tamper_detection' in self.scalers:
                features_scaled = self.scalers['tamper_detection'].transform([feature_diffs])
            else:
                features_scaled = [feature_diffs]
            
            # Predict tampering probability
            if hasattr(self.tamper_classifier, 'predict_proba'):
                tamper_prob = self.tamper_classifier.predict_proba(features_scaled)[0][1]
            else:
                tamper_prob = self.tamper_classifier.predict(features_scaled)[0]
            
            return max(0.0, min(1.0, tamper_prob))
            
        except Exception as e:
            logger.debug(f"ML tamper detection error: {e}")
            return 0.0
    
    def _detect_structural_changes(self, original_data: Dict[str, Any], 
                                  current_data: Dict[str, Any]) -> List[float]:
        """Detect structural changes between data versions"""
        try:
            changes = []
            
            # Key set changes
            orig_keys = set(original_data.keys())
            curr_keys = set(current_data.keys())
            
            added_keys = len(curr_keys - orig_keys)
            removed_keys = len(orig_keys - curr_keys)
            changes.extend([float(added_keys), float(removed_keys)])
            
            # Value type changes
            type_changes = 0
            for key in orig_keys & curr_keys:
                if type(original_data[key]) != type(current_data[key]):
                    type_changes += 1
            changes.append(float(type_changes))
            
            # Size changes
            orig_size = len(str(original_data))
            curr_size = len(str(current_data))
            size_change_ratio = abs(curr_size - orig_size) / max(orig_size, 1)
            changes.append(size_change_ratio)
            
            # Content diversity changes
            orig_types = self._count_data_types(original_data)
            curr_types = self._count_data_types(current_data)
            
            type_diversity_change = abs(len(orig_types) - len(curr_types))
            changes.append(float(type_diversity_change))
            
            return changes[:5]  # Return first 5 structural indicators
            
        except Exception:
            return [0.0] * 5
    
    def _analyze_patterns_ml(self, analytics_id: str, features: List[float]) -> float:
        """Analyze patterns using ML clustering"""
        try:
            if not self.pattern_analyzer or len(features) < 10:
                return 0.0
            
            # Get recent feature history for the analytics
            recent_features = [
                pattern['features'] for pattern in self.verification_patterns[analytics_id][-20:]
                if 'features' in pattern
            ]
            
            if len(recent_features) < 5:
                return 0.0
            
            # Add current features
            all_features = recent_features + [features]
            features_array = np.array(all_features)
            
            # Perform clustering to detect outliers
            clusters = self.pattern_analyzer.fit_predict(features_array)
            
            # Check if current features are outliers
            current_cluster = clusters[-1]
            if current_cluster == -1:  # DBSCAN outlier
                return 0.8
            
            # Calculate distance from cluster center
            cluster_points = features_array[clusters == current_cluster]
            if len(cluster_points) > 1:
                cluster_center = np.mean(cluster_points, axis=0)
                distance = np.linalg.norm(np.array(features) - cluster_center)
                
                # Normalize distance to 0-1 range
                max_distance = np.linalg.norm(np.std(features_array, axis=0))
                normalized_distance = min(distance / max_distance, 1.0) if max_distance > 0 else 0.0
                
                return normalized_distance
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Pattern analysis error: {e}")
            return 0.0
    
    def _handle_integrity_violation(self, record: MLIntegrityRecord, 
                                   verification_results: Dict[str, Any], 
                                   current_data: Dict[str, Any]):
        """Handle integrity violation with ML-enhanced response"""
        try:
            # Determine violation type and severity
            threat_indicators = verification_results.get('threat_indicators', {})
            
            if 'tampering_detected' in threat_indicators:
                record.status = MLIntegrityStatus.TAMPERED
                self.guardian_stats['tamper_attempts_blocked'] += 1
                
            elif 'anomaly_detected' in threat_indicators:
                record.status = MLIntegrityStatus.ANOMALY_DETECTED
                self.guardian_stats['anomalies_detected'] += 1
                
            else:
                record.status = MLIntegrityStatus.CORRUPTED
            
            # Set error message
            error_details = []
            for indicator, score in threat_indicators.items():
                error_details.append(f"{indicator}: {score:.3f}")
            
            record.error_message = f"ML Integrity violation - {', '.join(error_details)}"
            
            # Determine ML recovery strategy
            recovery_strategy = self._select_ml_recovery_strategy(record, verification_results)
            record.ml_recovery_strategy = recovery_strategy
            
            # Attempt ML-guided recovery
            if recovery_strategy and not record.recovery_attempted:
                recovery_success = self._attempt_ml_recovery(record.analytics_id, recovery_strategy)
                record.recovery_attempted = True
                
                if recovery_success:
                    record.status = MLIntegrityStatus.RECOVERED
                    self.guardian_stats['successful_recoveries'] += 1
                    logger.info(f"ML recovery successful: {record.analytics_id}")
            
            # Update threat profile
            if record.analytics_id in self.threat_profiles:
                threat_profile = self.threat_profiles[record.analytics_id]
                threat_profile.threat_level = IntegrityRiskLevel.HIGH
                threat_profile.threat_vectors.append("integrity_violation")
            
            logger.warning(f"ML integrity violation: {record.analytics_id} - {record.error_message}")
            
        except Exception as e:
            logger.error(f"Error handling integrity violation: {e}")
    
    def _select_ml_recovery_strategy(self, record: MLIntegrityRecord, 
                                    verification_results: Dict[str, Any]) -> Optional[str]:
        """Select optimal recovery strategy using ML analysis"""
        try:
            threat_indicators = verification_results.get('threat_indicators', {})
            
            # Strategy selection based on threat type
            if 'tampering_detected' in threat_indicators:
                return "backup_restoration"
            elif 'anomaly_detected' in threat_indicators:
                return "pattern_correction"
            elif 'pattern_anomaly' in threat_indicators:
                return "structural_repair"
            else:
                return "checksum_regeneration"
                
        except Exception:
            return "backup_restoration"  # Default strategy
    
    def _attempt_ml_recovery(self, analytics_id: str, strategy: str) -> bool:
        """Attempt ML-guided recovery"""
        try:
            if strategy not in self.recovery_strategies:
                logger.warning(f"Unknown recovery strategy: {strategy}")
                return False
            
            recovery_function = self.recovery_strategies[strategy]
            return recovery_function(analytics_id)
            
        except Exception as e:
            logger.error(f"ML recovery attempt failed: {e}")
            return False
    
    def _backup_restoration_recovery(self, analytics_id: str) -> bool:
        """Backup restoration recovery strategy"""
        try:
            if analytics_id not in self.ml_backup_storage:
                return False
            
            backup_data = self.ml_backup_storage[analytics_id]
            original_data = backup_data['data']
            
            # Verify backup integrity
            backup_checksum = self._generate_ml_enhanced_checksum(original_data)
            expected_checksum = backup_data['checksum']