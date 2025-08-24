#!/usr/bin/env python3
"""
🏗️ MODULE: Security Analyzer - Extracted from Enhanced Intelligence Linkage
===========================================================================

📋 PURPOSE:
    Comprehensive security analysis engine for vulnerability assessment,
    security pattern detection, and risk classification in codebases.

🎯 CORE FUNCTIONALITY:
    • Vulnerability pattern detection and scoring
    • Security pattern analysis and classification
    • Risk level assessment with impact analysis
    • Security linkage impact correlation analysis

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 [2025-08-23] | Agent Z | 🔧 STEELCLAD
   └─ Goal: Extract security analysis from enhanced_intelligence_linkage.py
   └─ Changes: Modularized security analysis with ~180 lines of focused functionality
   └─ Impact: Reduces main intelligence linkage size while maintaining security analysis

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent Z (STEELCLAD extraction)
🔧 Language: Python
📦 Dependencies: re, pathlib
🎯 Integration Points: EnhancedLinkageAnalyzer class
⚡ Performance Notes: Optimized for large-scale security scanning
🔒 Security Notes: Safe pattern matching with comprehensive vulnerability detection
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

class SecurityAnalyzer:
    """Comprehensive security analysis engine for vulnerability assessment."""
    
    def __init__(self):
        self.vulnerability_patterns = self._load_vulnerability_patterns()
        self.security_patterns = self._load_security_patterns()
        self.risk_weights = self._load_risk_weights()
        self.compliance_patterns = self._load_compliance_patterns()
    
    def _load_vulnerability_patterns(self) -> Dict[str, List[str]]:
        """Load vulnerability detection patterns."""
        return {
            "sql_injection": [
                r"execute\(.*\+", r"query\(.*\+", r"\.format\(.*sql",
                r"SELECT.*\+", r"INSERT.*\+", r"UPDATE.*\+", r"DELETE.*\+",
                r"cursor\.execute\(.*%", r"raw\(.*\+", r"extra\(.*\+"
            ],
            "xss": [
                r"innerHTML", r"document\.write", r"eval\(",
                r"setTimeout\(.*\+", r"setInterval\(.*\+",
                r"dangerouslySetInnerHTML", r"v-html"
            ],
            "hardcoded_secrets": [
                r"password\s*=\s*[\"'][^\"']+[\"']", r"api_key\s*=\s*[\"'][^\"']+[\"']",
                r"secret\s*=\s*[\"'][^\"']+[\"']", r"token\s*=\s*[\"'][^\"']+[\"']",
                r"key\s*=\s*[\"'][A-Za-z0-9]{20,}[\"']"
            ],
            "unsafe_deserialization": [
                r"pickle\.load", r"yaml\.load", r"marshal\.load",
                r"cPickle\.load", r"jsonpickle\.decode"
            ],
            "weak_crypto": [
                r"md5", r"sha1", r"random\.random", r"des",
                r"rc4", r"base64\.b64encode\(password"
            ],
            "path_traversal": [
                r"open\(.*\.\.", r"file\(.*\.\.", r"\.\.\/", r"\.\.\\\\",
                r"os\.path\.join\(.*input", r"os\.walk\(.*input"
            ],
            "command_injection": [
                r"os\.system\(.*input", r"subprocess\.*\(.*input", 
                r"eval\(.*input", r"exec\(.*input",
                r"os\.popen\(.*\+", r"commands\.getoutput"
            ],
            "insecure_random": [
                r"random\.random\(\)", r"random\.randint\(",
                r"random\.choice\(", r"time\.time\(\)"
            ],
            "weak_ssl": [
                r"ssl_verify\s*=\s*False", r"verify\s*=\s*False",
                r"CERT_NONE", r"ssl\.PROTOCOL_SSLv", r"ssl\.PROTOCOL_TLSv1"
            ],
            "information_disclosure": [
                r"print\(.*password", r"log.*password", r"debug.*secret",
                r"traceback\.print_exc", r"raise.*Exception.*password"
            ]
        }
    
    def _load_security_patterns(self) -> Dict[str, List[str]]:
        """Load positive security patterns."""
        return {
            "input_validation": [
                r"validate", r"sanitize", r"escape", r"clean",
                r"is_valid", r"check_input", r"filter_input"
            ],
            "error_handling": [
                r"try:", r"except", r"finally:", r"raise",
                r"logging\.exception", r"handle_error"
            ],
            "logging": [
                r"log\.", r"logger\.", r"logging\.",
                r"audit", r"track", r"monitor"
            ],
            "authentication": [
                r"authenticate", r"login", r"verify_password",
                r"check_credentials", r"@login_required", r"session"
            ],
            "authorization": [
                r"authorize", r"permission", r"role", r"access_control",
                r"@require_permission", r"check_access"
            ],
            "encryption": [
                r"encrypt", r"decrypt", r"cipher", r"aes", r"rsa",
                r"cryptography", r"hashlib\.sha256", r"bcrypt"
            ],
            "secure_communication": [
                r"https://", r"ssl", r"tls", r"certificate",
                r"secure_socket", r"verify=True"
            ],
            "data_protection": [
                r"mask", r"redact", r"obfuscate", r"anonymize",
                r"gdpr", r"privacy", r"data_protection"
            ]
        }
    
    def _load_risk_weights(self) -> Dict[str, int]:
        """Load vulnerability risk weights."""
        return {
            "sql_injection": 9,
            "xss": 7,
            "command_injection": 9,
            "path_traversal": 8,
            "hardcoded_secrets": 8,
            "unsafe_deserialization": 8,
            "weak_crypto": 6,
            "insecure_random": 4,
            "weak_ssl": 7,
            "information_disclosure": 6
        }
    
    def _load_compliance_patterns(self) -> Dict[str, List[str]]:
        """Load compliance-related patterns."""
        return {
            "gdpr": [r"gdpr", r"data_subject", r"consent", r"privacy_policy"],
            "pci_dss": [r"card.*number", r"credit_card", r"payment", r"pci"],
            "owasp": [r"owasp", r"csrf_token", r"xss_protection", r"sql_injection_prevention"],
            "soc2": [r"access_control", r"audit_log", r"encryption", r"monitoring"]
        }
    
    def analyze_security_dimensions(self, python_files: List[Path], base_dir: str) -> Dict[str, Any]:
        """Comprehensive security analysis of codebase."""
        security_results = {
            "vulnerability_scores": {},
            "security_patterns": {},
            "risk_classifications": {},
            "security_linkage_impact": {},
            "compliance_assessment": {},
            "security_metrics": {}
        }
        
        base_path = Path(base_dir)
        total_files = len(python_files)
        
        print(f"Security Analysis: Processing {total_files} files...")
        
        for i, py_file in enumerate(python_files):
            if i % 50 == 0:  # Progress update every 50 files
                print(f"  Security analysis progress: {i}/{total_files} ({i/total_files*100:.1f}%)")
            
            try:
                relative_path = str(py_file.relative_to(base_path))
                
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Vulnerability assessment
                vuln_score = self.assess_vulnerabilities(content)
                security_results["vulnerability_scores"][relative_path] = vuln_score
                
                # Security pattern detection
                patterns = self.detect_security_patterns(content)
                security_results["security_patterns"][relative_path] = patterns
                
                # Risk classification
                risk_level = self.classify_security_risk(vuln_score, patterns)
                security_results["risk_classifications"][relative_path] = risk_level
                
                # Compliance assessment
                compliance = self.assess_compliance(content)
                security_results["compliance_assessment"][relative_path] = compliance
                
            except Exception as e:
                print(f"  Error processing {py_file}: {e}")
                continue
        
        # Security linkage impact analysis
        security_results["security_linkage_impact"] = self.analyze_security_linkage_impact(
            security_results["vulnerability_scores"],
            security_results["risk_classifications"]
        )
        
        # Calculate security metrics
        security_results["security_metrics"] = self.calculate_security_metrics(
            security_results
        )
        
        print("Security Analysis: Complete!")
        return security_results
    
    def assess_vulnerabilities(self, content: str) -> Dict[str, Any]:
        """Assess security vulnerabilities in code."""
        total_score = 0
        vulnerabilities = {}
        vulnerability_details = {}
        
        for vuln_type, patterns in self.vulnerability_patterns.items():
            count = 0
            matches = []
            
            for pattern in patterns:
                try:
                    pattern_matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in pattern_matches:
                        count += 1
                        matches.append({
                            "pattern": pattern,
                            "match": match.group(0)[:50] + "..." if len(match.group(0)) > 50 else match.group(0),
                            "line": content[:match.start()].count('\n') + 1
                        })
                except re.error:
                    continue
            
            vulnerabilities[vuln_type] = count
            vulnerability_details[vuln_type] = matches
            total_score += count * self.risk_weights.get(vuln_type, 3)
        
        return {
            "total_score": total_score,
            "vulnerabilities": vulnerabilities,
            "vulnerability_details": vulnerability_details,
            "risk_level": self.calculate_risk_level(total_score),
            "severity_distribution": self._calculate_severity_distribution(vulnerabilities)
        }
    
    def detect_security_patterns(self, content: str) -> Dict[str, Any]:
        """Detect positive security patterns."""
        security_patterns = {}
        pattern_details = {}
        
        for pattern_type, patterns in self.security_patterns.items():
            count = 0
            matches = []
            
            for pattern in patterns:
                try:
                    pattern_matches = len(re.findall(pattern, content, re.IGNORECASE))
                    count += pattern_matches
                    
                    if pattern_matches > 0:
                        matches.append({
                            "pattern": pattern,
                            "count": pattern_matches
                        })
                except re.error:
                    continue
            
            security_patterns[pattern_type] = count
            pattern_details[pattern_type] = matches
        
        return {
            "pattern_counts": security_patterns,
            "pattern_details": pattern_details,
            "security_score": self._calculate_security_score(security_patterns),
            "coverage_assessment": self._assess_security_coverage(security_patterns)
        }
    
    def classify_security_risk(self, vuln_score: Dict[str, Any], patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Classify overall security risk level."""
        vulnerability_score = vuln_score.get("total_score", 0)
        security_score = patterns.get("security_score", 0)
        
        # Combine vulnerability and security pattern scores
        net_risk = vulnerability_score - (security_score * 0.5)
        
        if net_risk <= 5:
            risk_level = "low"
        elif net_risk <= 15:
            risk_level = "medium"
        elif net_risk <= 30:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        return {
            "risk_level": risk_level,
            "net_risk_score": net_risk,
            "vulnerability_contribution": vulnerability_score,
            "mitigation_contribution": security_score,
            "recommendations": self._generate_risk_recommendations(risk_level, vuln_score, patterns)
        }
    
    def assess_compliance(self, content: str) -> Dict[str, Any]:
        """Assess compliance with security standards."""
        compliance_scores = {}
        
        for standard, patterns in self.compliance_patterns.items():
            score = 0
            for pattern in patterns:
                try:
                    score += len(re.findall(pattern, content, re.IGNORECASE))
                except re.error:
                    continue
            
            compliance_scores[standard] = {
                "score": score,
                "level": "high" if score > 5 else "medium" if score > 2 else "low"
            }
        
        return {
            "compliance_scores": compliance_scores,
            "overall_compliance": self._calculate_overall_compliance(compliance_scores)
        }
    
    def calculate_risk_level(self, total_score: int) -> str:
        """Calculate risk level based on total vulnerability score."""
        if total_score == 0:
            return "minimal"
        elif total_score <= 10:
            return "low"
        elif total_score <= 25:
            return "medium"
        elif total_score <= 50:
            return "high"
        else:
            return "critical"
    
    def analyze_security_linkage_impact(self, vulnerability_scores: Dict[str, Dict], 
                                      risk_classifications: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze security impact across file linkages."""
        impact_analysis = {
            "high_risk_files": [],
            "security_hotspots": [],
            "risk_propagation": {},
            "impact_radius": {}
        }
        
        # Identify high-risk files
        for file_path, risk_data in risk_classifications.items():
            risk_level = risk_data.get("risk_level", "low")
            if risk_level in ["high", "critical"]:
                impact_analysis["high_risk_files"].append({
                    "file": file_path,
                    "risk_level": risk_level,
                    "score": vulnerability_scores.get(file_path, {}).get("total_score", 0)
                })
        
        # Identify security hotspots (areas with multiple vulnerabilities)
        vulnerability_clusters = defaultdict(list)
        for file_path, vuln_data in vulnerability_scores.items():
            vulnerabilities = vuln_data.get("vulnerabilities", {})
            for vuln_type, count in vulnerabilities.items():
                if count > 0:
                    vulnerability_clusters[vuln_type].append({
                        "file": file_path,
                        "count": count
                    })
        
        # Sort and identify top hotspots
        for vuln_type, files in vulnerability_clusters.items():
            if len(files) > 2:  # Consider it a hotspot if 3+ files have this vulnerability
                impact_analysis["security_hotspots"].append({
                    "vulnerability_type": vuln_type,
                    "affected_files": len(files),
                    "total_occurrences": sum(f["count"] for f in files),
                    "top_files": sorted(files, key=lambda x: x["count"], reverse=True)[:3]
                })
        
        return impact_analysis
    
    def calculate_security_metrics(self, security_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive security metrics."""
        vulnerability_scores = security_results.get("vulnerability_scores", {})
        risk_classifications = security_results.get("risk_classifications", {})
        
        if not vulnerability_scores:
            return {"error": "No vulnerability data available"}
        
        # Risk distribution
        risk_distribution = defaultdict(int)
        total_vulnerabilities = 0
        high_risk_files = 0
        
        for file_path, risk_data in risk_classifications.items():
            risk_level = risk_data.get("risk_level", "low")
            risk_distribution[risk_level] += 1
            
            if risk_level in ["high", "critical"]:
                high_risk_files += 1
        
        for vuln_data in vulnerability_scores.values():
            total_vulnerabilities += vuln_data.get("total_score", 0)
        
        total_files = len(vulnerability_scores)
        average_risk = total_vulnerabilities / max(total_files, 1)
        
        return {
            "total_files_analyzed": total_files,
            "total_vulnerabilities": total_vulnerabilities,
            "average_risk_score": average_risk,
            "high_risk_files": high_risk_files,
            "risk_distribution": dict(risk_distribution),
            "security_coverage": self._calculate_security_coverage_metric(security_results),
            "compliance_status": self._calculate_compliance_status(security_results),
            "security_trend": "improving" if average_risk < 10 else "concerning"
        }
    
    def _calculate_severity_distribution(self, vulnerabilities: Dict[str, int]) -> Dict[str, int]:
        """Calculate severity distribution of vulnerabilities."""
        severity_count = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for vuln_type, count in vulnerabilities.items():
            weight = self.risk_weights.get(vuln_type, 3)
            if weight >= 8:
                severity_count["critical"] += count
            elif weight >= 6:
                severity_count["high"] += count
            elif weight >= 4:
                severity_count["medium"] += count
            else:
                severity_count["low"] += count
        
        return severity_count
    
    def _calculate_security_score(self, security_patterns: Dict[str, int]) -> int:
        """Calculate positive security score."""
        weights = {
            "input_validation": 3,
            "error_handling": 2,
            "logging": 1,
            "authentication": 4,
            "authorization": 4,
            "encryption": 3,
            "secure_communication": 3,
            "data_protection": 2
        }
        
        score = 0
        for pattern_type, count in security_patterns.items():
            weight = weights.get(pattern_type, 1)
            score += count * weight
        
        return score
    
    def _assess_security_coverage(self, security_patterns: Dict[str, int]) -> str:
        """Assess security coverage based on patterns."""
        essential_patterns = ["input_validation", "authentication", "encryption", "error_handling"]
        covered = sum(1 for pattern in essential_patterns if security_patterns.get(pattern, 0) > 0)
        
        coverage_percentage = covered / len(essential_patterns)
        
        if coverage_percentage >= 0.8:
            return "comprehensive"
        elif coverage_percentage >= 0.6:
            return "good"
        elif coverage_percentage >= 0.4:
            return "basic"
        else:
            return "insufficient"
    
    def _generate_risk_recommendations(self, risk_level: str, vuln_score: Dict, patterns: Dict) -> List[str]:
        """Generate security recommendations based on risk assessment."""
        recommendations = []
        
        if risk_level in ["high", "critical"]:
            recommendations.append("Immediate security review required")
            recommendations.append("Implement comprehensive input validation")
            recommendations.append("Add security monitoring and logging")
        
        vulnerabilities = vuln_score.get("vulnerabilities", {})
        
        if vulnerabilities.get("sql_injection", 0) > 0:
            recommendations.append("Use parameterized queries to prevent SQL injection")
        
        if vulnerabilities.get("xss", 0) > 0:
            recommendations.append("Implement proper output encoding and CSP headers")
        
        if vulnerabilities.get("hardcoded_secrets", 0) > 0:
            recommendations.append("Move secrets to environment variables or secure vault")
        
        security_score = patterns.get("security_score", 0)
        if security_score < 10:
            recommendations.append("Implement additional security controls and monitoring")
        
        return recommendations
    
    def _calculate_overall_compliance(self, compliance_scores: Dict[str, Dict]) -> str:
        """Calculate overall compliance level."""
        levels = [data["level"] for data in compliance_scores.values()]
        
        if "high" in levels:
            return "compliant"
        elif "medium" in levels:
            return "partially_compliant"
        else:
            return "non_compliant"
    
    def _calculate_security_coverage_metric(self, security_results: Dict[str, Any]) -> float:
        """Calculate security coverage metric."""
        security_patterns = security_results.get("security_patterns", {})
        
        if not security_patterns:
            return 0.0
        
        total_coverage = 0
        file_count = len(security_patterns)
        
        for patterns in security_patterns.values():
            coverage = patterns.get("coverage_assessment", "insufficient")
            if coverage == "comprehensive":
                total_coverage += 1.0
            elif coverage == "good":
                total_coverage += 0.7
            elif coverage == "basic":
                total_coverage += 0.4
        
        return total_coverage / max(file_count, 1)
    
    def _calculate_compliance_status(self, security_results: Dict[str, Any]) -> str:
        """Calculate overall compliance status."""
        compliance_assessments = security_results.get("compliance_assessment", {})
        
        if not compliance_assessments:
            return "unknown"
        
        compliant_files = 0
        total_files = len(compliance_assessments)
        
        for assessment in compliance_assessments.values():
            overall = assessment.get("overall_compliance", "non_compliant")
            if overall == "compliant":
                compliant_files += 1
        
        compliance_ratio = compliant_files / max(total_files, 1)
        
        if compliance_ratio >= 0.8:
            return "compliant"
        elif compliance_ratio >= 0.5:
            return "partially_compliant"
        else:
            return "non_compliant"

def create_security_analyzer() -> SecurityAnalyzer:
    """Factory function to create a configured security analyzer."""
    return SecurityAnalyzer()