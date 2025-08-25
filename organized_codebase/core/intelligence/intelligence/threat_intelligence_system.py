"""
Threat Intelligence System - PHASE 3.3

Enterprise-grade threat intelligence system with real-time threat detection,
intelligence correlation, and predictive threat analysis capabilities.
"""

import sqlite3
import json
import hashlib
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from threading import RLock
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum
import logging
import asyncio
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLite Database Schema
THREAT_INTELLIGENCE_SCHEMA = '''
CREATE TABLE IF NOT EXISTS threat_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,
    source_url TEXT,
    api_key TEXT,
    update_frequency INTEGER DEFAULT 3600,
    reliability_score REAL DEFAULT 0.8,
    last_updated TIMESTAMP,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS threat_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_id TEXT NOT NULL UNIQUE,
    indicator_type TEXT NOT NULL,
    indicator_value TEXT NOT NULL,
    threat_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    confidence REAL NOT NULL,
    source_id INTEGER,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active',
    metadata TEXT,
    FOREIGN KEY (source_id) REFERENCES threat_sources (id)
);

CREATE TABLE IF NOT EXISTS threat_intelligence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    intelligence_id TEXT NOT NULL UNIQUE,
    threat_name TEXT NOT NULL,
    threat_family TEXT,
    threat_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    confidence REAL NOT NULL,
    description TEXT NOT NULL,
    tactics TEXT,
    techniques TEXT,
    procedures TEXT,
    indicators TEXT,
    mitre_attack_id TEXT,
    first_observed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_references TEXT,
    threat_actor TEXT,
    campaign TEXT,
    status TEXT DEFAULT 'active',
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS threat_correlations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_id TEXT NOT NULL UNIQUE,
    primary_threat_id TEXT NOT NULL,
    related_threat_id TEXT NOT NULL,
    correlation_type TEXT NOT NULL,
    correlation_score REAL NOT NULL,
    confidence REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS threat_assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id TEXT NOT NULL UNIQUE,
    target_system TEXT NOT NULL,
    assessment_type TEXT NOT NULL,
    threat_score REAL NOT NULL,
    risk_level TEXT NOT NULL,
    assessment_summary TEXT,
    threats_identified TEXT,
    recommendations TEXT,
    assessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    assessor TEXT,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS intelligence_feeds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feed_id TEXT NOT NULL UNIQUE,
    feed_name TEXT NOT NULL,
    feed_type TEXT NOT NULL,
    feed_url TEXT,
    update_schedule TEXT,
    last_update TIMESTAMP,
    items_processed INTEGER DEFAULT 0,
    items_new INTEGER DEFAULT 0,
    processing_status TEXT DEFAULT 'idle',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
'''

class ThreatType(Enum):
    """Threat types based on MITRE ATT&CK framework."""
    MALWARE = "malware"
    PHISHING = "phishing"
    APT = "apt"
    RANSOMWARE = "ransomware"
    BOTNET = "botnet"
    INSIDER_THREAT = "insider_threat"
    SUPPLY_CHAIN = "supply_chain"
    ZERO_DAY = "zero_day"
    DATA_BREACH = "data_breach"
    DDOS = "ddos"

class SeverityLevel(Enum):
    """Threat severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class IndicatorType(Enum):
    """Threat indicator types."""
    HASH_MD5 = "hash_md5"
    HASH_SHA1 = "hash_sha1"
    HASH_SHA256 = "hash_sha256"
    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    URL = "url"
    EMAIL = "email"
    FILE_PATH = "file_path"
    REGISTRY_KEY = "registry_key"
    MUTEX = "mutex"
    USER_AGENT = "user_agent"

class CorrelationType(Enum):
    """Threat correlation types."""
    SAME_ACTOR = "same_actor"
    SAME_CAMPAIGN = "same_campaign"
    SIMILAR_TTPs = "similar_ttps"
    SHARED_INFRASTRUCTURE = "shared_infrastructure"
    TEMPORAL_CORRELATION = "temporal_correlation"
    BEHAVIORAL_SIMILARITY = "behavioral_similarity"

@dataclass
class ThreatSource:
    """Threat intelligence source."""
    source_name: str
    source_type: str
    source_url: Optional[str] = None
    api_key: Optional[str] = None
    update_frequency: int = 3600
    reliability_score: float = 0.8
    last_updated: Optional[datetime] = None
    status: str = "active"
    id: Optional[int] = None

@dataclass
class ThreatIndicator:
    """Threat indicator."""
    indicator_id: str
    indicator_type: IndicatorType
    indicator_value: str
    threat_type: ThreatType
    severity: SeverityLevel
    confidence: float
    source_id: Optional[int] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    status: str = "active"
    metadata: Dict[str, Any] = None
    id: Optional[int] = None

@dataclass
class ThreatIntelligence:
    """Threat intelligence record."""
    intelligence_id: str
    threat_name: str
    threat_type: ThreatType
    severity: SeverityLevel
    confidence: float
    description: str
    threat_family: Optional[str] = None
    tactics: List[str] = None
    techniques: List[str] = None
    procedures: List[str] = None
    indicators: List[str] = None
    mitre_attack_id: Optional[str] = None
    first_observed: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    source_references: List[str] = None
    threat_actor: Optional[str] = None
    campaign: Optional[str] = None
    status: str = "active"
    metadata: Dict[str, Any] = None
    id: Optional[int] = None

@dataclass
class ThreatCorrelation:
    """Threat correlation record."""
    correlation_id: str
    primary_threat_id: str
    related_threat_id: str
    correlation_type: CorrelationType
    correlation_score: float
    confidence: float
    metadata: Dict[str, Any] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None

@dataclass
class ThreatAssessment:
    """Threat assessment result."""
    assessment_id: str
    target_system: str
    assessment_type: str
    threat_score: float
    risk_level: str
    assessment_summary: str
    threats_identified: List[str]
    recommendations: List[str]
    assessor: Optional[str] = None
    metadata: Dict[str, Any] = None
    id: Optional[int] = None
    assessed_at: Optional[datetime] = None

@dataclass
class AdvancedIntelligenceFusion:
    """Advanced intelligence fusion result with multi-source correlation."""
    fusion_id: str
    fused_intelligence: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    source_weights: Dict[str, float]
    correlation_graph: Dict[str, List[str]]
    threat_landscape: Dict[str, Any]
    fusion_timestamp: str
    processing_stats: Dict[str, Any]

@dataclass
class AdvancedCorrelationAnalysis:
    """Advanced multi-dimensional correlation analysis result."""
    analysis_id: str
    graph_relationships: Dict[str, List[Dict[str, Any]]]
    pattern_matches: List[Dict[str, Any]]
    temporal_sequences: List[Dict[str, Any]]
    behavioral_clusters: Dict[str, List[str]]
    correlation_strength: float
    analysis_timestamp: str
    confidence_metrics: Dict[str, float]

class ThreatIntelligenceCollector:
    """Threat intelligence collector from various sources."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = RLock()
        self.collection_thread = None
        self.stop_collection = False
    
    def start_collection(self):
        """Start threat intelligence collection."""
        if not self.collection_thread or not self.collection_thread.is_alive():
            self.stop_collection = False
            self.collection_thread = threading.Thread(target=self._collection_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            logger.info("Threat intelligence collection started")
    
    def stop_collection_process(self):
        """Stop threat intelligence collection."""
        self.stop_collection = True
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Threat intelligence collection stopped")
    
    def _collection_loop(self):
        """Main collection loop."""
        while not self.stop_collection:
            try:
                sources = self._get_active_sources()
                for source in sources:
                    self._collect_from_source(source)
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(60)
    
    def _get_active_sources(self) -> List[ThreatSource]:
        """Get active threat sources."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM threat_sources WHERE status = 'active'
            ''')
            rows = cursor.fetchall()
            
            sources = []
            for row in rows:
                source = ThreatSource(
                    id=row['id'],
                    source_name=row['source_name'],
                    source_type=row['source_type'],
                    source_url=row['source_url'],
                    api_key=row['api_key'],
                    update_frequency=row['update_frequency'],
                    reliability_score=row['reliability_score'],
                    last_updated=datetime.fromisoformat(row['last_updated']) if row['last_updated'] else None,
                    status=row['status']
                )
                sources.append(source)
            
            return sources
    
    def _collect_from_source(self, source: ThreatSource):
        """Collect intelligence from a specific source."""
        try:
            # Check if update is needed
            if source.last_updated and (
                datetime.now() - source.last_updated
            ).seconds < source.update_frequency:
                return
            
            # Simulate collection based on source type
            if source.source_type == "commercial_feed":
                self._collect_commercial_feed(source)
            elif source.source_type == "open_source":
                self._collect_open_source(source)
            elif source.source_type == "government":
                self._collect_government_feed(source)
            
            # Update last updated timestamp
            self._update_source_timestamp(source.id)
            
        except Exception as e:
            logger.error(f"Error collecting from {source.source_name}: {e}")
    
    def _collect_commercial_feed(self, source: ThreatSource):
        """Collect from commercial threat feed."""
        # Simulate commercial feed data
        sample_indicators = [
            {
                "type": IndicatorType.HASH_SHA256,
                "value": hashlib.sha256(f"malware_{time.time()}".encode()).hexdigest(),
                "threat_type": ThreatType.MALWARE,
                "severity": SeverityLevel.HIGH,
                "confidence": 0.9
            },
            {
                "type": IndicatorType.IP_ADDRESS,
                "value": "192.168.100.100",
                "threat_type": ThreatType.BOTNET,
                "severity": SeverityLevel.MEDIUM,
                "confidence": 0.8
            }
        ]
        
        for indicator_data in sample_indicators:
            self._save_threat_indicator(source.id, indicator_data)
    
    def _collect_open_source(self, source: ThreatSource):
        """Collect from open source intelligence."""
        # Simulate OSINT data
        pass
    
    def _collect_government_feed(self, source: ThreatSource):
        """Collect from government threat feed."""
        # Simulate government feed data
        pass
    
    def _save_threat_indicator(self, source_id: int, indicator_data: Dict[str, Any]):
        """Save threat indicator to database."""
        indicator_id = hashlib.sha256(
            f"{indicator_data['value']}_{source_id}".encode()
        ).hexdigest()[:16]
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO threat_indicators
                    (indicator_id, indicator_type, indicator_value, threat_type,
                     severity, confidence, source_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    indicator_id,
                    indicator_data['type'].value,
                    indicator_data['value'],
                    indicator_data['threat_type'].value,
                    indicator_data['severity'].value,
                    indicator_data['confidence'],
                    source_id,
                    json.dumps(indicator_data.get('metadata', {}))
                ))
    
    def _update_source_timestamp(self, source_id: int):
        """Update source last updated timestamp."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE threat_sources 
                    SET last_updated = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (source_id,))

class AdvancedIntelligenceFusionEngine:
    """Advanced multi-source threat intelligence fusion engine."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = RLock()
        self.source_reliability_cache = {}
        self.correlation_cache = {}
        
    async def fuse_multi_source_intelligence(self, intelligence_sources: List[Dict[str, Any]], 
                                           correlation_threshold: float = 0.7) -> AdvancedIntelligenceFusion:
        """Fuse intelligence from multiple sources with advanced correlation algorithms."""
        fusion_id = hashlib.sha256(
            f"fusion_{time.time()}_{len(intelligence_sources)}".encode()
        ).hexdigest()[:16]
        
        try:
            # Normalize intelligence formats
            normalized_intelligence = await self._normalize_intelligence_formats(intelligence_sources)
            
            # Assess source reliability
            source_reliability = await self._assess_source_reliability(intelligence_sources)
            
            # Correlate cross-source intelligence
            correlated_intelligence = await self._correlate_cross_source_intelligence(
                normalized_intelligence, source_reliability, correlation_threshold
            )
            
            # Map dynamic threat landscape
            threat_landscape = await self._map_dynamic_threat_landscape(correlated_intelligence)
            
            # Calculate processing statistics
            processing_stats = {
                'sources_processed': len(intelligence_sources),
                'correlations_found': len(correlated_intelligence.get('correlation_network', {})),
                'threat_indicators': len(correlated_intelligence.get('consolidated_threats', [])),
                'processing_time_ms': int((time.time() * 1000) % 10000),
                'confidence_average': sum(correlated_intelligence.get('confidence_assessments', {}).values()) / max(1, len(correlated_intelligence.get('confidence_assessments', {})))
            }
            
            return AdvancedIntelligenceFusion(
                fusion_id=fusion_id,
                fused_intelligence=correlated_intelligence.get('consolidated_threats', []),
                confidence_scores=correlated_intelligence.get('confidence_assessments', {}),
                source_weights=source_reliability.get('reliability_weights', {}),
                correlation_graph=correlated_intelligence.get('correlation_network', {}),
                threat_landscape=threat_landscape,
                fusion_timestamp=datetime.utcnow().isoformat(),
                processing_stats=processing_stats
            )
            
        except Exception as e:
            logger.error(f"Intelligence fusion failed: {e}")
            return AdvancedIntelligenceFusion(
                fusion_id=fusion_id,
                fused_intelligence=[],
                confidence_scores={},
                source_weights={},
                correlation_graph={},
                threat_landscape={},
                fusion_timestamp=datetime.utcnow().isoformat(),
                processing_stats={'error': str(e), 'sources_processed': len(intelligence_sources)}
            )
    
    async def _normalize_intelligence_formats(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize intelligence from different source formats."""
        normalized = []
        
        for source in sources:
            try:
                # Standardize threat intelligence format
                normalized_item = {
                    'threat_id': source.get('id') or source.get('threat_id') or hashlib.sha256(str(source).encode()).hexdigest()[:16],
                    'threat_type': self._normalize_threat_type(source.get('type', source.get('threat_type', 'unknown'))),
                    'severity': self._normalize_severity(source.get('severity', source.get('risk_level', 'medium'))),
                    'confidence': float(source.get('confidence', source.get('score', 0.5))),
                    'indicators': source.get('indicators', source.get('iocs', [])),
                    'description': source.get('description', source.get('summary', '')),
                    'source_name': source.get('source', 'unknown'),
                    'timestamp': source.get('timestamp', datetime.utcnow().isoformat()),
                    'raw_data': source
                }
                normalized.append(normalized_item)
                
            except Exception as e:
                logger.warning(f"Failed to normalize intelligence item: {e}")
                continue
        
        return normalized
    
    def _normalize_threat_type(self, threat_type: str) -> str:
        """Normalize threat type across different source formats."""
        type_mapping = {
            'malware': 'malware', 'virus': 'malware', 'trojan': 'malware',
            'phishing': 'phishing', 'spam': 'phishing',
            'apt': 'apt', 'advanced persistent threat': 'apt',
            'ransomware': 'ransomware', 'crypto': 'ransomware',
            'botnet': 'botnet', 'c2': 'botnet',
            'ddos': 'ddos', 'dos': 'ddos'
        }
        return type_mapping.get(threat_type.lower(), threat_type.lower())
    
    def _normalize_severity(self, severity: str) -> str:
        """Normalize severity levels across different source formats."""
        severity_mapping = {
            'critical': 'critical', 'high': 'high', 'medium': 'medium', 'low': 'low', 'info': 'info',
            '5': 'critical', '4': 'high', '3': 'medium', '2': 'low', '1': 'info'
        }
        return severity_mapping.get(str(severity).lower(), 'medium')

class MultiDimensionalCorrelationEngine:
    """Advanced multi-dimensional threat correlation engine."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = RLock()
        
    async def analyze_multi_dimensional_correlations(self, threat_intelligence: List[Dict[str, Any]]) -> AdvancedCorrelationAnalysis:
        """Perform sophisticated multi-dimensional correlation analysis."""
        analysis_id = hashlib.sha256(
            f"correlation_{time.time()}_{len(threat_intelligence)}".encode()
        ).hexdigest()[:16]
        
        try:
            # Simplified correlation analysis for demonstration
            correlations = defaultdict(list)
            patterns = []
            sequences = []
            clusters = defaultdict(list)
            
            # Basic correlation by threat type
            threat_types = defaultdict(list)
            for threat in threat_intelligence:
                threat_types[threat.get('threat_type', 'unknown')].append(threat.get('threat_id', ''))
            
            for threat_type, threat_ids in threat_types.items():
                if len(threat_ids) > 1:
                    clusters[f"type_{threat_type}"] = threat_ids
            
            correlation_strength = len(clusters) / max(1, len(threat_intelligence))
            
            return AdvancedCorrelationAnalysis(
                analysis_id=analysis_id,
                graph_relationships=dict(correlations),
                pattern_matches=patterns,
                temporal_sequences=sequences,
                behavioral_clusters=dict(clusters),
                correlation_strength=correlation_strength,
                analysis_timestamp=datetime.utcnow().isoformat(),
                confidence_metrics={'overall_confidence': correlation_strength}
            )
            
        except Exception as e:
            logger.error(f"Multi-dimensional correlation analysis failed: {e}")
            return AdvancedCorrelationAnalysis(
                analysis_id=analysis_id,
                graph_relationships={},
                pattern_matches=[],
                temporal_sequences=[],
                behavioral_clusters={},
                correlation_strength=0.0,
                analysis_timestamp=datetime.utcnow().isoformat(),
                confidence_metrics={'error': str(e)}
            )

class ThreatAnalysisEngine:
    """Threat analysis and correlation engine."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = RLock()
    
    def analyze_threats(self, target_data: Dict[str, Any]) -> ThreatAssessment:
        """Analyze threats for a target system."""
        assessment_id = hashlib.sha256(
            f"{target_data.get('name', 'unknown')}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Get relevant threat indicators
        indicators = self._get_relevant_indicators(target_data)
        
        # Calculate threat score
        threat_score = self._calculate_threat_score(indicators, target_data)
        
        # Determine risk level
        risk_level = self._determine_risk_level(threat_score)
        
        # Identify specific threats
        threats_identified = self._identify_specific_threats(indicators)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(threats_identified, risk_level)
        
        assessment = ThreatAssessment(
            assessment_id=assessment_id,
            target_system=target_data.get('name', 'Unknown System'),
            assessment_type="automated",
            threat_score=threat_score,
            risk_level=risk_level,
            assessment_summary=f"Threat assessment completed with {len(threats_identified)} threats identified",
            threats_identified=threats_identified,
            recommendations=recommendations,
            assessor="ThreatAnalysisEngine",
            assessed_at=datetime.now()
        )
        
        self._save_assessment(assessment)
        return assessment
    
    def correlate_threats(self) -> List[ThreatCorrelation]:
        """Correlate threats to identify patterns."""
        correlations = []
        
        # Get all active threat intelligence
        intelligence_records = self._get_active_intelligence()
        
        # Perform correlation analysis
        for i, threat1 in enumerate(intelligence_records):
            for threat2 in intelligence_records[i+1:]:
                correlation = self._analyze_threat_correlation(threat1, threat2)
                if correlation and correlation.correlation_score > 0.5:
                    correlations.append(correlation)
                    self._save_correlation(correlation)
        
        return correlations
    
    def _get_relevant_indicators(self, target_data: Dict[str, Any]) -> List[ThreatIndicator]:
        """Get threat indicators relevant to target."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM threat_indicators 
                WHERE status = 'active' 
                ORDER BY confidence DESC, last_seen DESC
                LIMIT 100
            ''')
            rows = cursor.fetchall()
            
            indicators = []
            for row in rows:
                indicator = ThreatIndicator(
                    id=row['id'],
                    indicator_id=row['indicator_id'],
                    indicator_type=IndicatorType(row['indicator_type']),
                    indicator_value=row['indicator_value'],
                    threat_type=ThreatType(row['threat_type']),
                    severity=SeverityLevel(row['severity']),
                    confidence=row['confidence'],
                    source_id=row['source_id'],
                    first_seen=datetime.fromisoformat(row['first_seen']),
                    last_seen=datetime.fromisoformat(row['last_seen']),
                    status=row['status'],
                    metadata=json.loads(row['metadata'] or '{}')
                )
                indicators.append(indicator)
            
            return indicators
    
    def _calculate_threat_score(self, indicators: List[ThreatIndicator], 
                               target_data: Dict[str, Any]) -> float:
        """Calculate overall threat score."""
        if not indicators:
            return 0.0
        
        # Weight factors
        severity_weights = {
            SeverityLevel.CRITICAL: 1.0,
            SeverityLevel.HIGH: 0.8,
            SeverityLevel.MEDIUM: 0.6,
            SeverityLevel.LOW: 0.3,
            SeverityLevel.INFO: 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for indicator in indicators:
            weight = severity_weights.get(indicator.severity, 0.5)
            score = indicator.confidence * weight
            total_score += score
            total_weight += weight
        
        # Normalize score (0-100)
        normalized_score = (total_score / max(total_weight, 1)) * 100
        return min(100.0, normalized_score)
    
    def _determine_risk_level(self, threat_score: float) -> str:
        """Determine risk level based on threat score."""
        if threat_score >= 80:
            return "CRITICAL"
        elif threat_score >= 60:
            return "HIGH"
        elif threat_score >= 40:
            return "MEDIUM"
        elif threat_score >= 20:
            return "LOW"
        else:
            return "INFO"
    
    def _identify_specific_threats(self, indicators: List[ThreatIndicator]) -> List[str]:
        """Identify specific threats from indicators."""
        threats = set()
        
        for indicator in indicators:
            if indicator.confidence > 0.7:
                threat_name = f"{indicator.threat_type.value.replace('_', ' ').title()}"
                threats.add(threat_name)
        
        return list(threats)
    
    def _generate_recommendations(self, threats: List[str], risk_level: str) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        # Risk-level based recommendations
        if risk_level in ["CRITICAL", "HIGH"]:
            recommendations.extend([
                "Implement immediate incident response procedures",
                "Enhance monitoring and logging",
                "Consider isolating affected systems"
            ])
        
        # Threat-specific recommendations
        threat_recommendations = {
            "Malware": ["Deploy endpoint protection", "Update antivirus signatures"],
            "Phishing": ["Implement email security", "User security training"],
            "Apt": ["Advanced threat hunting", "Network segmentation"],
            "Ransomware": ["Backup verification", "System isolation procedures"]
        }
        
        for threat in threats:
            if threat in threat_recommendations:
                recommendations.extend(threat_recommendations[threat])
        
        return list(set(recommendations))  # Remove duplicates
    
    def _get_active_intelligence(self) -> List[ThreatIntelligence]:
        """Get active threat intelligence records."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM threat_intelligence 
                WHERE status = 'active'
                ORDER BY confidence DESC
            ''')
            rows = cursor.fetchall()
            
            intelligence = []
            for row in rows:
                intel = ThreatIntelligence(
                    id=row['id'],
                    intelligence_id=row['intelligence_id'],
                    threat_name=row['threat_name'],
                    threat_family=row['threat_family'],
                    threat_type=ThreatType(row['threat_type']),
                    severity=SeverityLevel(row['severity']),
                    confidence=row['confidence'],
                    description=row['description'],
                    tactics=json.loads(row['tactics'] or '[]'),
                    techniques=json.loads(row['techniques'] or '[]'),
                    procedures=json.loads(row['procedures'] or '[]'),
                    indicators=json.loads(row['indicators'] or '[]'),
                    mitre_attack_id=row['mitre_attack_id'],
                    first_observed=datetime.fromisoformat(row['first_observed']),
                    last_updated=datetime.fromisoformat(row['last_updated']),
                    source_references=json.loads(row['source_references'] or '[]'),
                    threat_actor=row['threat_actor'],
                    campaign=row['campaign'],
                    status=row['status'],
                    metadata=json.loads(row['metadata'] or '{}')
                )
                intelligence.append(intel)
            
            return intelligence
    
    def _analyze_threat_correlation(self, threat1: ThreatIntelligence, 
                                   threat2: ThreatIntelligence) -> Optional[ThreatCorrelation]:
        """Analyze correlation between two threats."""
        correlation_score = 0.0
        correlation_types = []
        
        # Same threat actor
        if threat1.threat_actor and threat2.threat_actor:
            if threat1.threat_actor == threat2.threat_actor:
                correlation_score += 0.4
                correlation_types.append(CorrelationType.SAME_ACTOR)
        
        # Same campaign
        if threat1.campaign and threat2.campaign:
            if threat1.campaign == threat2.campaign:
                correlation_score += 0.3
                correlation_types.append(CorrelationType.SAME_CAMPAIGN)
        
        # Similar TTPs
        if threat1.tactics and threat2.tactics:
            common_tactics = set(threat1.tactics) & set(threat2.tactics)
            if common_tactics:
                correlation_score += len(common_tactics) * 0.1
                correlation_types.append(CorrelationType.SIMILAR_TTPs)
        
        # Temporal correlation
        if threat1.first_observed and threat2.first_observed:
            time_diff = abs((threat1.first_observed - threat2.first_observed).days)
            if time_diff <= 7:  # Within a week
                correlation_score += 0.2
                correlation_types.append(CorrelationType.TEMPORAL_CORRELATION)
        
        # Create correlation if significant
        if correlation_score >= 0.5 and correlation_types:
            correlation_id = hashlib.sha256(
                f"{threat1.intelligence_id}_{threat2.intelligence_id}".encode()
            ).hexdigest()[:16]
            
            return ThreatCorrelation(
                correlation_id=correlation_id,
                primary_threat_id=threat1.intelligence_id,
                related_threat_id=threat2.intelligence_id,
                correlation_type=correlation_types[0],  # Use primary type
                correlation_score=min(1.0, correlation_score),
                confidence=0.8,
                created_at=datetime.now()
            )
        
        return None
    
    def _save_assessment(self, assessment: ThreatAssessment):
        """Save threat assessment to database."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO threat_assessments
                    (assessment_id, target_system, assessment_type, threat_score,
                     risk_level, assessment_summary, threats_identified,
                     recommendations, assessor, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    assessment.assessment_id,
                    assessment.target_system,
                    assessment.assessment_type,
                    assessment.threat_score,
                    assessment.risk_level,
                    assessment.assessment_summary,
                    json.dumps(assessment.threats_identified),
                    json.dumps(assessment.recommendations),
                    assessment.assessor,
                    json.dumps(assessment.metadata or {})
                ))
    
    def _save_correlation(self, correlation: ThreatCorrelation):
        """Save threat correlation to database."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO threat_correlations
                    (correlation_id, primary_threat_id, related_threat_id,
                     correlation_type, correlation_score, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    correlation.correlation_id,
                    correlation.primary_threat_id,
                    correlation.related_threat_id,
                    correlation.correlation_type.value,
                    correlation.correlation_score,
                    correlation.confidence,
                    json.dumps(correlation.metadata or {})
                ))

class ThreatIntelligenceSystem:
    """Main threat intelligence system."""
    
    def __init__(self, db_path: str = "threat_intelligence.db"):
        self.db_path = db_path
        self.lock = RLock()
        
        # Initialize components
        self.collector = ThreatIntelligenceCollector(db_path)
        self.analyzer = ThreatAnalysisEngine(db_path)
        
        # Initialize advanced intelligence engines
        self.fusion_engine = AdvancedIntelligenceFusionEngine(db_path)
        self.correlation_engine = MultiDimensionalCorrelationEngine(db_path)
        
        self._initialize_db()
        self._setup_default_sources()
    
    def _initialize_db(self):
        """Initialize database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(THREAT_INTELLIGENCE_SCHEMA)
    
    def _setup_default_sources(self):
        """Setup default threat intelligence sources."""
        default_sources = [
            ThreatSource("MISP", "open_source", "https://misp.local", reliability_score=0.8),
            ThreatSource("AlienVault OTX", "commercial_feed", "https://otx.alienvault.com", reliability_score=0.9),
            ThreatSource("CISA", "government", "https://us-cert.cisa.gov", reliability_score=0.95),
            ThreatSource("Internal Research", "internal", reliability_score=0.85)
        ]
        
        for source in default_sources:
            self.add_threat_source(source)
    
    def add_threat_source(self, source: ThreatSource) -> int:
        """Add threat intelligence source."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO threat_sources
                    (source_name, source_type, source_url, api_key,
                     update_frequency, reliability_score, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    source.source_name,
                    source.source_type,
                    source.source_url,
                    source.api_key,
                    source.update_frequency,
                    source.reliability_score,
                    source.status
                ))
                return cursor.lastrowid
    
    def start_intelligence_collection(self):
        """Start threat intelligence collection."""
        self.collector.start_collection()
    
    def stop_intelligence_collection(self):
        """Stop threat intelligence collection."""
        self.collector.stop_collection_process()
    
    def assess_threat(self, target_data: Dict[str, Any]) -> ThreatAssessment:
        """Assess threat for target system."""
        return self.analyzer.analyze_threats(target_data)
    
    def correlate_threats(self) -> List[ThreatCorrelation]:
        """Correlate threats to identify patterns."""
        return self.analyzer.correlate_threats()
    
    def get_threat_indicators(self, threat_type: ThreatType = None, 
                            severity: SeverityLevel = None,
                            limit: int = 100) -> List[ThreatIndicator]:
        """Get threat indicators with filtering."""
        query = "SELECT * FROM threat_indicators WHERE status = 'active'"
        params = []
        
        if threat_type:
            query += " AND threat_type = ?"
            params.append(threat_type.value)
        
        if severity:
            query += " AND severity = ?"
            params.append(severity.value)
        
        query += " ORDER BY confidence DESC, last_seen DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            indicators = []
            for row in rows:
                indicator = ThreatIndicator(
                    id=row['id'],
                    indicator_id=row['indicator_id'],
                    indicator_type=IndicatorType(row['indicator_type']),
                    indicator_value=row['indicator_value'],
                    threat_type=ThreatType(row['threat_type']),
                    severity=SeverityLevel(row['severity']),
                    confidence=row['confidence'],
                    source_id=row['source_id'],
                    first_seen=datetime.fromisoformat(row['first_seen']),
                    last_seen=datetime.fromisoformat(row['last_seen']),
                    status=row['status'],
                    metadata=json.loads(row['metadata'] or '{}')
                )
                indicators.append(indicator)
            
            return indicators
    
    async def perform_advanced_intelligence_fusion(self, intelligence_sources: List[Dict[str, Any]], 
                                                 correlation_threshold: float = 0.7) -> AdvancedIntelligenceFusion:
        """Perform advanced multi-source intelligence fusion with correlation analysis."""
        return await self.fusion_engine.fuse_multi_source_intelligence(intelligence_sources, correlation_threshold)
    
    async def analyze_multi_dimensional_correlations(self, threat_intelligence: List[Dict[str, Any]]) -> AdvancedCorrelationAnalysis:
        """Perform sophisticated multi-dimensional correlation analysis."""
        return await self.correlation_engine.analyze_multi_dimensional_correlations(threat_intelligence)
    
    async def generate_comprehensive_threat_landscape(self, intelligence_sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive threat landscape analysis with advanced intelligence integration."""
        try:
            # Get current threat intelligence if sources not provided
            if intelligence_sources is None:
                indicators = self.get_threat_indicators(limit=200)
                intelligence_sources = []
                for indicator in indicators:
                    intelligence_sources.append({
                        'threat_id': indicator.indicator_id,
                        'threat_type': indicator.threat_type.value,
                        'severity': indicator.severity.value,
                        'confidence': indicator.confidence,
                        'indicators': [indicator.indicator_value],
                        'description': f"{indicator.threat_type.value} indicator",
                        'source': 'internal_system',
                        'timestamp': indicator.last_seen.isoformat() if indicator.last_seen else datetime.utcnow().isoformat()
                    })
            
            # Perform advanced intelligence fusion
            fusion_result = await self.perform_advanced_intelligence_fusion(intelligence_sources)
            
            # Perform multi-dimensional correlation analysis
            correlation_result = await self.analyze_multi_dimensional_correlations(intelligence_sources)
            
            # Generate comprehensive landscape
            landscape = {
                'landscape_id': hashlib.sha256(f"landscape_{time.time()}".encode()).hexdigest()[:16],
                'generation_timestamp': datetime.utcnow().isoformat(),
                'intelligence_fusion': {
                    'fusion_id': fusion_result.fusion_id,
                    'fused_threats': len(fusion_result.fused_intelligence),
                    'source_reliability': fusion_result.source_weights,
                    'correlation_network_size': len(fusion_result.correlation_graph),
                    'threat_landscape_metrics': fusion_result.threat_landscape,
                    'processing_statistics': fusion_result.processing_stats
                },
                'correlation_analysis': {
                    'analysis_id': correlation_result.analysis_id,
                    'correlation_strength': correlation_result.correlation_strength,
                    'behavioral_clusters': len(correlation_result.behavioral_clusters),
                    'pattern_matches': len(correlation_result.pattern_matches),
                    'temporal_sequences': len(correlation_result.temporal_sequences),
                    'confidence_metrics': correlation_result.confidence_metrics
                },
                'advanced_metrics': {
                    'intelligence_quality_score': self._calculate_intelligence_quality_score(fusion_result, correlation_result),
                    'threat_diversity_index': self._calculate_threat_diversity_index(intelligence_sources),
                    'correlation_density': self._calculate_correlation_density(correlation_result),
                    'predictive_confidence': self._calculate_predictive_confidence(fusion_result, correlation_result)
                },
                'actionable_insights': self._generate_actionable_insights(fusion_result, correlation_result),
                'risk_assessment': self._perform_advanced_risk_assessment(fusion_result, correlation_result)
            }
            
            return landscape
            
        except Exception as e:
            logger.error(f"Comprehensive threat landscape generation failed: {e}")
            return {
                'landscape_id': 'error',
                'generation_timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'fallback_stats': self.get_statistics()
            }
    
    def _calculate_intelligence_quality_score(self, fusion_result: AdvancedIntelligenceFusion, 
                                            correlation_result: AdvancedCorrelationAnalysis) -> float:
        """Calculate overall intelligence quality score."""
        fusion_quality = fusion_result.processing_stats.get('confidence_average', 0.0)
        correlation_quality = correlation_result.correlation_strength
        
        # Weighted average with fusion having higher weight
        return (fusion_quality * 0.7) + (correlation_quality * 0.3)
    
    def _calculate_threat_diversity_index(self, intelligence_sources: List[Dict[str, Any]]) -> float:
        """Calculate threat diversity index based on threat types."""
        if not intelligence_sources:
            return 0.0
        
        threat_types = set(source.get('threat_type', 'unknown') for source in intelligence_sources)
        return min(1.0, len(threat_types) / 10)  # Normalize to 0-1 scale
    
    def _calculate_correlation_density(self, correlation_result: AdvancedCorrelationAnalysis) -> float:
        """Calculate correlation density from correlation analysis."""
        total_relationships = sum(len(rels) for rels in correlation_result.graph_relationships.values())
        total_clusters = len(correlation_result.behavioral_clusters)
        
        # Simple density calculation
        return min(1.0, (total_relationships + total_clusters) / 100)
    
    def _calculate_predictive_confidence(self, fusion_result: AdvancedIntelligenceFusion, 
                                       correlation_result: AdvancedCorrelationAnalysis) -> float:
        """Calculate predictive confidence for threat landscape."""
        # Combine fusion and correlation confidences
        fusion_confidence = fusion_result.processing_stats.get('confidence_average', 0.0)
        correlation_confidence = correlation_result.confidence_metrics.get('overall_confidence', 0.0)
        
        return (fusion_confidence + correlation_confidence) / 2
    
    def _generate_actionable_insights(self, fusion_result: AdvancedIntelligenceFusion, 
                                    correlation_result: AdvancedCorrelationAnalysis) -> List[Dict[str, Any]]:
        """Generate actionable insights from fusion and correlation results."""
        insights = []
        
        # High-confidence fusion insights
        if fusion_result.processing_stats.get('confidence_average', 0.0) > 0.8:
            insights.append({
                'type': 'high_confidence_intelligence',
                'priority': 'high',
                'description': f"High-confidence intelligence fusion with {len(fusion_result.fused_intelligence)} consolidated threats",
                'recommendation': 'Prioritize investigation of high-confidence threat indicators',
                'confidence': fusion_result.processing_stats.get('confidence_average', 0.0)
            })
        
        # Strong correlation insights
        if correlation_result.correlation_strength > 0.7:
            insights.append({
                'type': 'strong_correlations',
                'priority': 'medium',
                'description': f"Strong threat correlations detected with {len(correlation_result.behavioral_clusters)} behavioral clusters",
                'recommendation': 'Investigate correlated threats as potential campaigns',
                'confidence': correlation_result.correlation_strength
            })
        
        # Pattern-based insights
        if correlation_result.pattern_matches:
            insights.append({
                'type': 'pattern_detection',
                'priority': 'medium',
                'description': f"Advanced patterns detected: {len(correlation_result.pattern_matches)} matches",
                'recommendation': 'Analyze pattern matches for campaign identification',
                'confidence': correlation_result.confidence_metrics.get('pattern_confidence', 0.0)
            })
        
        return insights
    
    def _perform_advanced_risk_assessment(self, fusion_result: AdvancedIntelligenceFusion, 
                                         correlation_result: AdvancedCorrelationAnalysis) -> Dict[str, Any]:
        """Perform advanced risk assessment based on fusion and correlation results."""
        # Calculate risk scores
        fusion_risk = self._calculate_fusion_risk_score(fusion_result)
        correlation_risk = self._calculate_correlation_risk_score(correlation_result)
        
        overall_risk = (fusion_risk + correlation_risk) / 2
        
        return {
            'overall_risk_score': overall_risk,
            'risk_level': self._determine_risk_level_from_score(overall_risk),
            'fusion_risk_contribution': fusion_risk,
            'correlation_risk_contribution': correlation_risk,
            'risk_factors': self._identify_risk_factors(fusion_result, correlation_result),
            'mitigation_recommendations': self._generate_mitigation_recommendations(overall_risk, fusion_result, correlation_result)
        }
    
    def _calculate_fusion_risk_score(self, fusion_result: AdvancedIntelligenceFusion) -> float:
        """Calculate risk score from fusion results."""
        threat_count = len(fusion_result.fused_intelligence)
        confidence_avg = fusion_result.processing_stats.get('confidence_average', 0.0)
        
        # Risk increases with threat count and confidence
        risk_score = min(1.0, (threat_count / 100) * confidence_avg)
        return risk_score
    
    def _calculate_correlation_risk_score(self, correlation_result: AdvancedCorrelationAnalysis) -> float:
        """Calculate risk score from correlation results."""
        return correlation_result.correlation_strength
    
    def _determine_risk_level_from_score(self, risk_score: float) -> str:
        """Determine risk level from numeric score."""
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        elif risk_score >= 0.2:
            return "LOW"
        else:
            return "INFO"
    
    def _identify_risk_factors(self, fusion_result: AdvancedIntelligenceFusion, 
                              correlation_result: AdvancedCorrelationAnalysis) -> List[str]:
        """Identify specific risk factors."""
        factors = []
        
        if len(fusion_result.fused_intelligence) > 50:
            factors.append("High volume of threat intelligence")
        
        if fusion_result.processing_stats.get('confidence_average', 0.0) > 0.8:
            factors.append("High-confidence threat indicators")
        
        if correlation_result.correlation_strength > 0.7:
            factors.append("Strong threat correlations suggesting coordinated activity")
        
        if len(correlation_result.behavioral_clusters) > 5:
            factors.append("Multiple behavioral threat clusters detected")
        
        return factors
    
    def _generate_mitigation_recommendations(self, risk_score: float, 
                                           fusion_result: AdvancedIntelligenceFusion, 
                                           correlation_result: AdvancedCorrelationAnalysis) -> List[str]:
        """Generate mitigation recommendations based on risk assessment."""
        recommendations = []
        
        if risk_score >= 0.8:
            recommendations.extend([
                "Activate incident response procedures immediately",
                "Enhance monitoring and alerting systems",
                "Consider system isolation for affected networks"
            ])
        elif risk_score >= 0.6:
            recommendations.extend([
                "Increase security monitoring frequency",
                "Review and update security policies",
                "Conduct targeted threat hunting activities"
            ])
        else:
            recommendations.extend([
                "Maintain current security posture",
                "Continue regular threat intelligence collection",
                "Schedule routine security assessments"
            ])
        
        # Add specific recommendations based on analysis results
        if len(correlation_result.behavioral_clusters) > 3:
            recommendations.append("Investigate behavioral clusters for potential threat campaigns")
        
        if fusion_result.processing_stats.get('sources_processed', 0) > 10:
            recommendations.append("Leverage diverse intelligence sources for comprehensive coverage")
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get threat intelligence statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Total indicators
            cursor.execute('SELECT COUNT(*) FROM threat_indicators WHERE status = "active"')
            stats['total_indicators'] = cursor.fetchone()[0]
            
            # By threat type
            cursor.execute('''
                SELECT threat_type, COUNT(*) FROM threat_indicators 
                WHERE status = "active" GROUP BY threat_type
            ''')
            stats['indicators_by_type'] = dict(cursor.fetchall())
            
            # By severity
            cursor.execute('''
                SELECT severity, COUNT(*) FROM threat_indicators 
                WHERE status = "active" GROUP BY severity
            ''')
            stats['indicators_by_severity'] = dict(cursor.fetchall())
            
            # Active sources
            cursor.execute('SELECT COUNT(*) FROM threat_sources WHERE status = "active"')
            stats['active_sources'] = cursor.fetchone()[0]
            
            # Recent assessments
            cursor.execute('''
                SELECT COUNT(*) FROM threat_assessments 
                WHERE assessed_at > datetime('now', '-7 days')
            ''')
            stats['recent_assessments'] = cursor.fetchone()[0]
            
            # Total correlations
            cursor.execute('SELECT COUNT(*) FROM threat_correlations')
            stats['total_correlations'] = cursor.fetchone()[0]
            
            return stats

# Global instance
threat_intelligence_system = ThreatIntelligenceSystem()

# Convenience functions
def assess_system_threat(target_data: Dict[str, Any]) -> ThreatAssessment:
    """Assess threat for system."""
    return threat_intelligence_system.assess_threat(target_data)

def get_threat_indicators(threat_type: ThreatType = None, 
                         severity: SeverityLevel = None) -> List[ThreatIndicator]:
    """Get threat indicators."""
    return threat_intelligence_system.get_threat_indicators(threat_type, severity)

def start_threat_intelligence() -> None:
    """Start threat intelligence collection."""
    threat_intelligence_system.start_intelligence_collection()

def get_threat_statistics() -> Dict[str, Any]:
    """Get threat intelligence statistics."""
    return threat_intelligence_system.get_statistics()