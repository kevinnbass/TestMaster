#!/usr/bin/env python3
"""
Threat Detection Module
Agent D Hour 5 - Modularized Threat Pattern Recognition

Handles threat pattern detection, vulnerability scanning, and security analysis
following STEELCLAD Anti-Regression Modularization Protocol.
"""

import re
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import datetime

from .security_events import ThreatLevel, SecurityEvent, ResponseAction

@dataclass
class ThreatPattern:
    """Threat pattern definition with metadata"""
    name: str
    patterns: List[str]
    threat_level: ThreatLevel
    description: str
    response_action: ResponseAction
    confidence_threshold: float = 0.8
    pattern_type: str = "regex"
    enabled: bool = True

@dataclass
class ThreatDetectionResult:
    """Result of threat detection analysis"""
    threat_found: bool
    pattern_name: str
    threat_level: ThreatLevel
    confidence_score: float
    matched_patterns: List[str]
    evidence: Dict[str, Any]
    suggested_action: ResponseAction
    detection_timestamp: str

class ThreatPatternLibrary:
    """Centralized threat pattern management"""
    
    def __init__(self):
        """Initialize threat patterns following security best practices"""
        self.patterns = self._initialize_default_patterns()
        self.custom_patterns = {}
        self.pattern_stats = {}
    
    def _initialize_default_patterns(self) -> Dict[str, ThreatPattern]:
        """Initialize default threat detection patterns"""
        patterns = {}
        
        # Code injection patterns
        patterns['code_injection'] = ThreatPattern(
            name="code_injection",
            patterns=[
                r'eval\s*\(',
                r'exec\s*\(',
                r'__import__\s*\(',
                r'compile\s*\(',
                r'subprocess\..*shell\s*=\s*True',
                r'os\.system\s*\(',
                r'os\.popen\s*\(',
                r'commands\..*\('
            ],
            threat_level=ThreatLevel.CRITICAL,
            description="Potential code injection vulnerability detected",
            response_action=ResponseAction.QUARANTINE,
            confidence_threshold=0.9
        )
        
        # Suspicious imports
        patterns['suspicious_imports'] = ThreatPattern(
            name="suspicious_imports",
            patterns=[
                r'import\s+os\s*;.*system',
                r'from\s+os\s+import.*system',
                r'import\s+subprocess\s*;.*shell',
                r'import\s+pickle\s*;.*loads',
                r'import\s+marshal',
                r'from\s+ctypes\s+import'
            ],
            threat_level=ThreatLevel.HIGH,
            description="Suspicious import patterns detected",
            response_action=ResponseAction.ALERT,
            confidence_threshold=0.7
        )
        
        # SQL injection patterns
        patterns['sql_injection'] = ThreatPattern(
            name="sql_injection",
            patterns=[
                r'SELECT\s+.*\s+FROM\s+.*\s+WHERE\s+.*=.*%s',
                r'INSERT\s+INTO\s+.*\s+VALUES\s*\(.*%s',
                r'UPDATE\s+.*\s+SET\s+.*=.*%s',
                r'DELETE\s+FROM\s+.*\s+WHERE\s+.*=.*%s',
                r'UNION\s+SELECT',
                r'OR\s+1\s*=\s*1',
                r'DROP\s+TABLE',
                r';\s*--'
            ],
            threat_level=ThreatLevel.CRITICAL,
            description="SQL injection vulnerability pattern detected",
            response_action=ResponseAction.QUARANTINE,
            confidence_threshold=0.85
        )
        
        # Hardcoded secrets
        patterns['hardcoded_secrets'] = ThreatPattern(
            name="hardcoded_secrets",
            patterns=[
                r'password\s*=\s*["\'][^"\']{8,}["\']',
                r'api_key\s*=\s*["\'][^"\']{16,}["\']',
                r'secret\s*=\s*["\'][^"\']{12,}["\']',
                r'token\s*=\s*["\'][^"\']{20,}["\']',
                r'private_key\s*=\s*["\']',
                r'-----BEGIN\s+PRIVATE\s+KEY-----'
            ],
            threat_level=ThreatLevel.HIGH,
            description="Hardcoded secrets or credentials detected",
            response_action=ResponseAction.ALERT,
            confidence_threshold=0.8
        )
        
        # Unsafe deserialization
        patterns['unsafe_deserialization'] = ThreatPattern(
            name="unsafe_deserialization",
            patterns=[
                r'pickle\.loads?\s*\(',
                r'cPickle\.loads?\s*\(',
                r'yaml\.load\s*\(',
                r'marshal\.loads?\s*\(',
                r'shelve\.open\s*\(',
                r'json\.loads?.*object_hook'
            ],
            threat_level=ThreatLevel.HIGH,
            description="Unsafe deserialization pattern detected",
            response_action=ResponseAction.ALERT,
            confidence_threshold=0.85
        )
        
        # Path traversal
        patterns['path_traversal'] = ThreatPattern(
            name="path_traversal",
            patterns=[
                r'\.\./+',
                r'\.\.\\\\+',
                r'%2e%2e%2f',
                r'%2e%2e\\',
                r'/etc/passwd',
                r'/etc/shadow',
                r'C:\\\\Windows\\\\System32'
            ],
            threat_level=ThreatLevel.MEDIUM,
            description="Path traversal attack pattern detected",
            response_action=ResponseAction.ALERT,
            confidence_threshold=0.7
        )
        
        return patterns
    
    def add_custom_pattern(self, pattern: ThreatPattern) -> bool:
        """Add custom threat pattern"""
        if pattern.name in self.patterns or pattern.name in self.custom_patterns:
            return False
        
        self.custom_patterns[pattern.name] = pattern
        self.pattern_stats[pattern.name] = {
            'detections': 0,
            'false_positives': 0,
            'true_positives': 0,
            'last_detection': None
        }
        return True
    
    def get_pattern(self, name: str) -> Optional[ThreatPattern]:
        """Get threat pattern by name"""
        if name in self.patterns:
            return self.patterns[name]
        return self.custom_patterns.get(name)
    
    def get_all_patterns(self) -> List[ThreatPattern]:
        """Get all enabled threat patterns"""
        all_patterns = list(self.patterns.values()) + list(self.custom_patterns.values())
        return [p for p in all_patterns if p.enabled]

class ThreatDetectionEngine:
    """Core threat detection and analysis engine"""
    
    def __init__(self):
        """Initialize threat detection engine"""
        self.pattern_library = ThreatPatternLibrary()
        self.detection_cache = {}
        self.whitelist_patterns = []
        self.detection_stats = {
            'scans_performed': 0,
            'threats_detected': 0,
            'false_positives': 0,
            'scan_duration_total': 0.0
        }
    
    def scan_file(self, file_path: str) -> List[ThreatDetectionResult]:
        """Scan single file for security threats"""
        start_time = datetime.datetime.now()
        
        if not os.path.exists(file_path):
            return []
        
        # Check cache first
        file_hash = self._get_file_hash(file_path)
        if file_hash in self.detection_cache:
            return self.detection_cache[file_hash]
        
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Scan against all threat patterns
            for pattern in self.pattern_library.get_all_patterns():
                detection_result = self._scan_content_with_pattern(
                    content, pattern, file_path
                )
                if detection_result and detection_result.threat_found:
                    results.append(detection_result)
                    
        except Exception as e:
            # Log error but continue with other files
            pass
        
        # Cache results
        self.detection_cache[file_hash] = results
        
        # Update statistics
        scan_duration = (datetime.datetime.now() - start_time).total_seconds()
        self.detection_stats['scans_performed'] += 1
        self.detection_stats['scan_duration_total'] += scan_duration
        if results:
            self.detection_stats['threats_detected'] += len(results)
        
        return results
    
    def scan_directory(self, directory_path: str, 
                      file_extensions: List[str] = None,
                      max_files: int = 1000) -> List[ThreatDetectionResult]:
        """Scan directory for security threats"""
        if file_extensions is None:
            file_extensions = ['.py', '.js', '.sql', '.json', '.xml', '.yaml', '.yml']
        
        all_results = []
        files_scanned = 0
        
        directory = Path(directory_path)
        
        for file_path in directory.rglob('*'):
            if files_scanned >= max_files:
                break
                
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                file_results = self.scan_file(str(file_path))
                all_results.extend(file_results)
                files_scanned += 1
        
        return all_results
    
    def _scan_content_with_pattern(self, content: str, pattern: ThreatPattern, 
                                  file_path: str) -> Optional[ThreatDetectionResult]:
        """Scan content against specific threat pattern"""
        matched_patterns = []
        confidence_scores = []
        
        for regex_pattern in pattern.patterns:
            try:
                matches = re.findall(regex_pattern, content, re.IGNORECASE | re.MULTILINE)
                if matches:
                    matched_patterns.extend(matches if isinstance(matches[0], str) else [str(m) for m in matches])
                    
                    # Calculate confidence based on match strength
                    match_count = len(matches)
                    confidence = min(0.5 + (match_count * 0.1), 1.0)
                    confidence_scores.append(confidence)
                    
            except re.error:
                # Skip invalid regex patterns
                continue
        
        if not matched_patterns:
            return None
        
        # Calculate overall confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Check against whitelist
        if self._is_whitelisted(file_path, pattern.name):
            avg_confidence *= 0.3  # Reduce confidence for whitelisted items
        
        if avg_confidence < pattern.confidence_threshold:
            return None
        
        # Generate evidence
        evidence = {
            'file_path': file_path,
            'pattern_name': pattern.name,
            'matched_content': matched_patterns[:5],  # Limit to first 5 matches
            'match_count': len(matched_patterns),
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'scan_timestamp': datetime.datetime.now().isoformat()
        }
        
        return ThreatDetectionResult(
            threat_found=True,
            pattern_name=pattern.name,
            threat_level=pattern.threat_level,
            confidence_score=avg_confidence,
            matched_patterns=matched_patterns,
            evidence=evidence,
            suggested_action=pattern.response_action,
            detection_timestamp=datetime.datetime.now().isoformat()
        )
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for file caching"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()
        except:
            return hashlib.md5(file_path.encode()).hexdigest()
    
    def _is_whitelisted(self, file_path: str, pattern_name: str) -> bool:
        """Check if file/pattern combination is whitelisted"""
        for whitelist_pattern in self.whitelist_patterns:
            if re.match(whitelist_pattern, file_path):
                return True
        return False
    
    def add_to_whitelist(self, file_pattern: str):
        """Add file pattern to whitelist"""
        self.whitelist_patterns.append(file_pattern)
    
    def clear_cache(self):
        """Clear detection cache"""
        self.detection_cache.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection engine statistics"""
        stats = self.detection_stats.copy()
        if stats['scans_performed'] > 0:
            stats['average_scan_duration'] = stats['scan_duration_total'] / stats['scans_performed']
            stats['threats_per_scan'] = stats['threats_detected'] / stats['scans_performed']
        else:
            stats['average_scan_duration'] = 0.0
            stats['threats_per_scan'] = 0.0
        
        return stats

class VulnerabilityAssessment:
    """Advanced vulnerability assessment and scoring"""
    
    CVSS_BASE_SCORES = {
        ThreatLevel.INFO: 0.0,
        ThreatLevel.LOW: 3.9,
        ThreatLevel.MEDIUM: 6.9,
        ThreatLevel.HIGH: 8.9,
        ThreatLevel.CRITICAL: 10.0,
        ThreatLevel.EMERGENCY: 10.0
    }
    
    @staticmethod
    def calculate_cvss_score(detection_result: ThreatDetectionResult,
                           exploitability: float = 0.8,
                           impact: float = 0.8) -> float:
        """Calculate CVSS-like vulnerability score"""
        base_score = VulnerabilityAssessment.CVSS_BASE_SCORES.get(
            detection_result.threat_level, 5.0
        )
        
        # Adjust based on confidence and context
        confidence_modifier = detection_result.confidence_score
        exploitability_modifier = exploitability
        impact_modifier = impact
        
        # Calculate temporal score
        temporal_score = base_score * confidence_modifier
        
        # Calculate environmental score
        environmental_score = temporal_score * (exploitability_modifier * impact_modifier)
        
        return min(environmental_score, 10.0)
    
    @staticmethod
    def assess_risk_level(cvss_score: float) -> str:
        """Assess risk level based on CVSS score"""
        if cvss_score >= 9.0:
            return "CRITICAL"
        elif cvss_score >= 7.0:
            return "HIGH"
        elif cvss_score >= 4.0:
            return "MEDIUM"
        elif cvss_score > 0.0:
            return "LOW"
        else:
            return "NONE"
    
    @staticmethod
    def generate_remediation_advice(detection_result: ThreatDetectionResult) -> List[str]:
        """Generate remediation advice based on threat pattern"""
        advice = []
        
        pattern_remediation = {
            'code_injection': [
                "Use parameterized queries and prepared statements",
                "Implement input validation and sanitization",
                "Avoid dynamic code execution (eval, exec)",
                "Use safe alternatives like ast.literal_eval for data parsing"
            ],
            'sql_injection': [
                "Use parameterized queries or prepared statements",
                "Implement proper input validation",
                "Use ORM frameworks with built-in protection",
                "Apply principle of least privilege to database accounts"
            ],
            'hardcoded_secrets': [
                "Move secrets to environment variables",
                "Use secure secret management systems",
                "Implement proper key rotation policies",
                "Never commit secrets to version control"
            ],
            'unsafe_deserialization': [
                "Validate input before deserialization",
                "Use safe serialization formats like JSON",
                "Implement integrity checks and signatures",
                "Avoid deserializing untrusted data"
            ],
            'path_traversal': [
                "Validate and sanitize file paths",
                "Use whitelist approach for allowed paths",
                "Implement proper access controls",
                "Use safe file operation APIs"
            ]
        }
        
        return pattern_remediation.get(detection_result.pattern_name, [
            "Review code for security vulnerabilities",
            "Follow secure coding best practices",
            "Implement proper input validation",
            "Conduct security code review"
        ])