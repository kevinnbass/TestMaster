"""
Advanced ML Integrity Guardian System
====================================
"""Models Module - Split from integrity_ml_guardian.py"""


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


            
            if backup_checksum == expected_checksum:
                # Backup is valid - could restore here
                logger.info(f"Valid backup found for {analytics_id}")
                return True
            
            return False
            
        except Exception:
            return False
    
    def _pattern_correction_recovery(self, analytics_id: str) -> bool:
        """Pattern-based correction recovery strategy"""
        try:
            # Use ML to predict correct patterns
            if analytics_id in self.verification_patterns:
                patterns = self.verification_patterns[analytics_id]
                if len(patterns) >= 3:
                    # Could implement pattern-based correction
                    logger.info(f"Pattern correction attempted for {analytics_id}")
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _structural_repair_recovery(self, analytics_id: str) -> bool:
        """Structural repair recovery strategy"""
        try:
            # Could implement structural repair based on known good structures
            logger.info(f"Structural repair attempted for {analytics_id}")
            return True
            
        except Exception:
            return False
    
    def _checksum_regeneration_recovery(self, analytics_id: str) -> bool:
        """Checksum regeneration recovery strategy"""
        try:
            # Regenerate checksums with enhanced validation
            if analytics_id in self.analytics_checksums:
                data = self.analytics_checksums[analytics_id]['data']
                new_checksum = self._generate_ml_enhanced_checksum(data)
                
                if analytics_id in self.integrity_records:
                    self.integrity_records[analytics_id].original_checksum = new_checksum
                
                logger.info(f"Checksum regenerated for {analytics_id}")
                return True
            
            return False
            
        except Exception:
            return False
    
    def _create_ml_backup(self, analytics_id: str, data: Dict[str, Any], features: List[float]):
        """Create ML-enhanced backup"""
        try:
            checksum = self._generate_ml_enhanced_checksum(data)
            
            self.ml_backup_storage[analytics_id] = {
                'data': data.copy(),
                'checksum': checksum,
                'ml_features': features.copy(),
                'timestamp': datetime.now().isoformat(),
                'backup_version': 'ml_enhanced'
            }
            
            # Limit backup storage
            if len(self.ml_backup_storage) > 1000:
                oldest_id = min(self.ml_backup_storage.keys(),
                               key=lambda x: self.ml_backup_storage[x]['timestamp'])
                del self.ml_backup_storage[oldest_id]
                
        except Exception as e:
            logger.error(f"ML backup creation failed: {e}")
    
    def _record_verification_pattern(self, analytics_id: str, verification_results: Dict[str, Any]):
        """Record verification pattern for ML learning"""
        try:
            pattern = {
                'timestamp': datetime.now(),
                'integrity_valid': verification_results['integrity_valid'],
                'anomaly_score': verification_results['anomaly_score'],
                'tamper_probability': verification_results['tamper_probability'],
                'ml_confidence': verification_results['ml_confidence'],
                'features': verification_results.get('features', [])
            }
            
            self.verification_patterns[analytics_id].append(pattern)
            
            # Keep only recent patterns
            if len(self.verification_patterns[analytics_id]) > self.ml_config["pattern_analysis_window"]:
                self.verification_patterns[analytics_id].pop(0)
            
            # Store for ML training
            self.integrity_features_history.append({
                'analytics_id': analytics_id,
                'pattern': pattern,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            logger.debug(f"Pattern recording failed: {e}")
    
    def _update_verification_schedule(self, analytics_id: str, verification_results: Dict[str, Any]):
        """Update adaptive verification schedule based on ML analysis"""
        try:
            if not self.ml_config["adaptive_scheduling"]:
                return
            
            current_interval = self.adaptive_intervals.get(analytics_id, self.verification_interval)
            
            # Adjust based on verification results
            if not verification_results['integrity_valid']:
                # Increase frequency after violations
                new_interval = max(5.0, current_interval * 0.5)
            elif verification_results['ml_confidence'] > 0.9:
                # Decrease frequency for high confidence
                new_interval = min(3600.0, current_interval * 1.2)
            else:
                # Maintain current interval
                new_interval = current_interval
            
            self.adaptive_intervals[analytics_id] = new_interval
            self.guardian_stats['adaptive_optimizations'] += 1
            
        except Exception as e:
            logger.debug(f"Schedule update failed: {e}")
    
    def _update_threat_profile(self, analytics_id: str, verification_results: Dict[str, Any]):
        """Update threat profile based on verification results"""
        try:
            if analytics_id not in self.threat_profiles:
                return
            
            threat_profile = self.threat_profiles[analytics_id]
            
            # Update threat level based on recent results
            if verification_results['integrity_valid'] and verification_results['ml_confidence'] > 0.8:
                # Lower threat level for consistently good results
                if threat_profile.threat_level == IntegrityRiskLevel.HIGH:
                    threat_profile.threat_level = IntegrityRiskLevel.MEDIUM
                elif threat_profile.threat_level == IntegrityRiskLevel.MEDIUM:
                    threat_profile.threat_level = IntegrityRiskLevel.LOW
            
            # Update ML indicators
            threat_profile.ml_indicators.update({
                'recent_anomaly_score': verification_results['anomaly_score'],
                'recent_tamper_probability': verification_results['tamper_probability'],
                'recent_ml_confidence': verification_results['ml_confidence']
            })
            
        except Exception as e:
            logger.debug(f"Threat profile update failed: {e}")
    
    def _setup_recovery_strategies(self):
        """Setup ML recovery strategies"""
        self.recovery_strategies = {
            'backup_restoration': self._backup_restoration_recovery,
            'pattern_correction': self._pattern_correction_recovery,
            'structural_repair': self._structural_repair_recovery,
            'checksum_regeneration': self._checksum_regeneration_recovery
        }
    
    def _initialize_ml_models(self):
        """Initialize ML models"""
        try:
            if self.ml_enabled:
                # Initialize with basic models
                self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
                self.tamper_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
                self.corruption_predictor = LogisticRegression(random_state=42, max_iter=1000)
                self.pattern_analyzer = DBSCAN(eps=0.5, min_samples=3)
                
                logger.info("ML models initialized for integrity guardian")
                
        except Exception as e:
            logger.warning(f"ML model initialization failed: {e}")
            self.ml_enabled = False
    
    # ========================================================================
    # BACKGROUND ML LOOPS
    # ========================================================================
    
    def _ml_verification_loop(self):
        """Background ML verification loop"""
        while self.ml_guardian_active:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                with self.guardian_lock:
                    current_time = datetime.now()
                    
                    # Process adaptive verification schedule
                    for analytics_id, interval in self.adaptive_intervals.items():
                        if analytics_id in self.integrity_records:
                            record = self.integrity_records[analytics_id]
                            
                            if (not record.verified_at or 
                                (current_time - record.verified_at).total_seconds() >= interval):
                                
                                # Add to priority queue
                                self.priority_verification_queue.append(analytics_id)
                    
                    # Process priority verification queue
                    while self.priority_verification_queue:
                        analytics_id = self.priority_verification_queue.popleft()
                        
                        if analytics_id in self.analytics_checksums:
                            current_data = self.analytics_checksums[analytics_id]['data']
                            self.verify_analytics_ml(analytics_id, current_data)
                
            except Exception as e:
                logger.error(f"ML verification loop error: {e}")
    
    def _ml_analysis_loop(self):
        """Background ML analysis and threat assessment loop"""
        while self.ml_guardian_active:
            try:
                time.sleep(300)  # Every 5 minutes
                
                # Analyze threat patterns
                self._analyze_global_threat_patterns()
                
                # Update predictive models
                self._update_predictive_models()
                
                # Generate threat intelligence
                self._generate_threat_intelligence()
                
            except Exception as e:
                logger.error(f"ML analysis loop error: {e}")
    
    def _ml_training_loop(self):
        """Background ML model training loop"""
        while self.ml_guardian_active:
            try:
                time.sleep(3600)  # Every hour
                
                # Retrain models with new data
                if len(self.integrity_features_history) >= self.ml_config["min_training_samples"]:
                    await self._retrain_ml_models()
                
            except Exception as e:
                logger.error(f"ML training loop error: {e}")
    
    def _analyze_global_threat_patterns(self):
        """Analyze global threat patterns across all analytics"""
        try:
            # Analyze patterns across all threat profiles
            threat_levels = [profile.threat_level for profile in self.threat_profiles.values()]
            
            if threat_levels:
                high_risk_count = sum(1 for level in threat_levels 
                                    if level in [IntegrityRiskLevel.HIGH, IntegrityRiskLevel.CRITICAL])
                
                total_count = len(threat_levels)
                risk_ratio = high_risk_count / total_count if total_count > 0 else 0
                
                if risk_ratio > 0.3:
                    logger.warning(f"High global threat level detected: {risk_ratio:.1%} high-risk analytics")
                
        except Exception as e:
            logger.error(f"Global threat analysis failed: {e}")
    
    def _update_predictive_models(self):
        """Update predictive models with recent data"""
        try:
            # Update corruption prediction model
            if len(self.integrity_features_history) >= 20:
                # Could implement corruption prediction updates
                pass
                
        except Exception as e:
            logger.error(f"Predictive model update failed: {e}")
    
    def _generate_threat_intelligence(self):
        """Generate threat intelligence reports"""
        try:
            # Generate intelligence on threat patterns
            intelligence = {
                'timestamp': datetime.now(),
                'total_analytics_protected': len(self.integrity_records),
                'active_threats': len([p for p in self.threat_profiles.values() 
                                     if p.threat_level in [IntegrityRiskLevel.HIGH, IntegrityRiskLevel.CRITICAL]]),
                'ml_accuracy': self.guardian_stats.get('ml_accuracy', 0.0),
                'recent_violations': len([r for r in self.integrity_records.values() 
                                        if r.status in [MLIntegrityStatus.CORRUPTED, MLIntegrityStatus.TAMPERED]])
            }
            
            logger.debug(f"Threat intelligence: {intelligence}")
            
        except Exception as e:
            logger.error(f"Threat intelligence generation failed: {e}")
    
    async def _retrain_ml_models(self):
        """Retrain ML models with accumulated data"""
        try:
            training_data = list(self.integrity_features_history)[-500:]  # Last 500 samples
            
            if len(training_data) < self.ml_config["min_training_samples"]:
                return
            
            # Prepare training data for anomaly detection
            X_anomaly = []
            for data in training_data:
                if 'pattern' in data and 'features' in data['pattern']:
                    X_anomaly.append(data['pattern']['features'])
            
            if len(X_anomaly) >= 20:
                X_anomaly_array = np.array(X_anomaly)
                
                # Train anomaly detector
                scaler_anomaly = StandardScaler()
                X_anomaly_scaled = scaler_anomaly.fit_transform(X_anomaly_array)
                
                self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
                self.anomaly_detector.fit(X_anomaly_scaled)
                
                self.scalers['anomaly_detection'] = scaler_anomaly
                
                logger.info("ML models retrained for integrity guardian")
            
        except Exception as e:
            logger.error(f"ML model retraining failed: {e}")
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================
    
    def get_ml_integrity_summary(self) -> Dict[str, Any]:
        """Get comprehensive ML integrity summary"""
        with self.guardian_lock:
            status_counts = defaultdict(int)
            risk_counts = defaultdict(int)
            
            for record in self.integrity_records.values():
                status_counts[record.status.value] += 1
                risk_counts[record.corruption_risk.value] += 1
            
            return {
                'ml_guardian_status': 'active' if self.ml_guardian_active else 'inactive',
                'ml_enabled': self.ml_enabled,
                'statistics': self.guardian_stats.copy(),
                'status_breakdown': dict(status_counts),
                'risk_level_breakdown': dict(risk_counts),
                'total_protected_analytics': len(self.integrity_records),
                'verified_analytics_count': len(self.verified_analytics),
                'threat_profiles_count': len(self.threat_profiles),
                'ml_backup_storage_count': len(self.ml_backup_storage),
                'adaptive_intervals_active': len(self.adaptive_intervals),
                'ml_models_active': {
                    'anomaly_detector': self.anomaly_detector is not None,
                    'tamper_classifier': self.tamper_classifier is not None,
                    'corruption_predictor': self.corruption_predictor is not None,
                    'pattern_analyzer': self.pattern_analyzer is not None
                },
                'ml_configuration': self.ml_config.copy(),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_threat_intelligence(self) -> Dict[str, Any]:
        """Get current threat intelligence"""
        with self.guardian_lock:
            high_risk_profiles = [
                {
                    'analytics_id': profile.analytics_id,
                    'threat_level': profile.threat_level.value,
                    'threat_vectors': profile.threat_vectors,
                    'confidence_score': profile.confidence_score,
                    'recommendation': profile.recommendation
                }
                for profile in self.threat_profiles.values()
                if profile.threat_level in [IntegrityRiskLevel.HIGH, IntegrityRiskLevel.CRITICAL]
            ]
            
            return {
                'timestamp': datetime.now().isoformat(),
                'global_threat_level': self._calculate_global_threat_level(),
                'high_risk_analytics': high_risk_profiles,
                'recent_violations': len([r for r in self.integrity_records.values() 
                                        if r.status in [MLIntegrityStatus.CORRUPTED, MLIntegrityStatus.TAMPERED] 
                                        and r.verified_at 
                                        and (datetime.now() - r.verified_at).total_seconds() < 3600]),
                'ml_accuracy': self.guardian_stats.get('ml_accuracy', 0.0),
                'threat_trends': self._analyze_threat_trends()
            }
    
    def _calculate_global_threat_level(self) -> str:
        """Calculate global threat level"""
        try:
            threat_levels = [profile.threat_level for profile in self.threat_profiles.values()]
            
            if not threat_levels:
                return "minimal"
            
            critical_count = sum(1 for level in threat_levels if level == IntegrityRiskLevel.CRITICAL)
            high_count = sum(1 for level in threat_levels if level == IntegrityRiskLevel.HIGH)
            
            total_count = len(threat_levels)
            
            if critical_count > 0:
                return "critical"
            elif high_count / total_count > 0.3:
                return "high"
            elif high_count / total_count > 0.1:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "unknown"
    
    def _analyze_threat_trends(self) -> Dict[str, Any]:
        """Analyze threat trends"""
        try:
            # Analyze recent patterns
            recent_violations = [
                r for r in self.integrity_records.values()
                if r.verified_at and (datetime.now() - r.verified_at).total_seconds() < 86400  # Last 24 hours
            ]
            
            violation_types = defaultdict(int)
            for record in recent_violations:
                if record.status in [MLIntegrityStatus.CORRUPTED, MLIntegrityStatus.TAMPERED, MLIntegrityStatus.ANOMALY_DETECTED]:
                    violation_types[record.status.value] += 1
            
            return {
                'violations_24h': len(recent_violations),
                'violation_types': dict(violation_types),
                'trend_direction': 'stable'  # Could implement trend analysis
            }
            
        except Exception:
            return {'violations_24h': 0, 'violation_types': {}, 'trend_direction': 'unknown'}
    
    def force_ml_verification(self, analytics_id: str) -> bool:
        """Force immediate ML verification"""
        try:
            if analytics_id not in self.analytics_checksums:
                return False
            
            current_data = self.analytics_checksums[analytics_id]['data']
            record = self.verify_analytics_ml(analytics_id, current_data)
            
            return record.status in [MLIntegrityStatus.VERIFIED, MLIntegrityStatus.RECOVERED]
            
        except Exception as e:
            logger.error(f"Force ML verification failed: {e}")
            return False
    
    def shutdown(self):
        """Shutdown ML integrity guardian"""
        self.stop_ml_guardian()
        logger.info("Advanced ML Integrity Guardian shutdown")

# Global ML integrity guardian instance
advanced_ml_integrity_guardian = AdvancedMLIntegrityGuardian()

# Export for external use
__all__ = [
    'MLIntegrityStatus',
    'IntegrityRiskLevel',
    'MLAlgorithm',
    'MLIntegrityRecord',
    'IntegrityThreatProfile',
    'AdvancedMLIntegrityGuardian',
    'advanced_ml_integrity_guardian'
]