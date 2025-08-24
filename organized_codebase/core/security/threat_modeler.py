"""
Threat Modeler

Performs threat modeling and risk assessment for applications.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ThreatCategory(Enum):
    """STRIDE threat categories."""
    SPOOFING = "Spoofing"
    TAMPERING = "Tampering"
    REPUDIATION = "Repudiation"
    INFORMATION_DISCLOSURE = "Information Disclosure"
    DENIAL_OF_SERVICE = "Denial of Service"
    ELEVATION_OF_PRIVILEGE = "Elevation of Privilege"
    

class RiskLevel(Enum):
    """Risk severity levels."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    INFO = 1
    

@dataclass
class Threat:
    """Represents a security threat."""
    id: str
    name: str
    category: ThreatCategory
    description: str
    impact: str
    likelihood: int  # 1-5
    risk_score: int  # likelihood * impact
    affected_components: List[str]
    mitigations: List[str]
    

@dataclass
class Asset:
    """Represents a system asset."""
    name: str
    type: str  # data, service, component
    value: str  # critical, high, medium, low
    threats: List[Threat]
    

@dataclass
class ThreatModel:
    """Complete threat model."""
    application: str
    version: str
    assets: List[Asset]
    threats: List[Threat]
    attack_surface: Dict[str, Any]
    risk_matrix: Dict[str, int]
    timestamp: str
    

class ThreatModeler:
    """
    Performs comprehensive threat modeling using STRIDE methodology.
    Identifies threats, assesses risks, and provides mitigation strategies.
    """
    
    def __init__(self):
        """Initialize the threat modeler."""
        self.threats = []
        self.assets = []
        self.attack_vectors = []
        logger.info("Threat Modeler initialized")
        
    def analyze_application(self, directory: str) -> ThreatModel:
        """
        Perform threat modeling for an application.
        
        Args:
            directory: Application directory
            
        Returns:
            Complete threat model
        """
        # Identify assets
        self.assets = self._identify_assets(directory)
        
        # Analyze attack surface
        attack_surface = self._analyze_attack_surface(directory)
        
        # Identify threats using STRIDE
        self.threats = self._identify_stride_threats(self.assets, attack_surface)
        
        # Calculate risk scores
        risk_matrix = self._calculate_risk_matrix()
        
        return ThreatModel(
            application=Path(directory).name,
            version="1.0",
            assets=self.assets,
            threats=self.threats,
            attack_surface=attack_surface,
            risk_matrix=risk_matrix,
            timestamp=datetime.now().isoformat()
        )
        
    def generate_threat_report(self, model: ThreatModel) -> str:
        """
        Generate threat modeling report.
        
        Args:
            model: Threat model
            
        Returns:
            Markdown formatted report
        """
        report = [
            f"# Threat Model Report: {model.application}",
            f"Generated: {model.timestamp}",
            "",
            "## Executive Summary",
            f"- Total Threats Identified: {len(model.threats)}",
            f"- Critical Risks: {sum(1 for t in model.threats if t.risk_score >= 20)}",
            f"- High Risks: {sum(1 for t in model.threats if 15 <= t.risk_score < 20)}",
            "",
            "## Assets",
            ""
        ]
        
        for asset in model.assets:
            report.extend([
                f"### {asset.name}",
                f"- Type: {asset.type}",
                f"- Value: {asset.value}",
                f"- Threats: {len(asset.threats)}",
                ""
            ])
            
        report.extend([
            "## Threat Analysis (STRIDE)",
            ""
        ])
        
        for category in ThreatCategory:
            threats = [t for t in model.threats if t.category == category]
            if threats:
                report.extend([
                    f"### {category.value}",
                    ""
                ])
                for threat in threats:
                    report.extend([
                        f"#### {threat.name}",
                        f"- Risk Score: {threat.risk_score}",
                        f"- Impact: {threat.impact}",
                        f"- Likelihood: {threat.likelihood}/5",
                        f"- Description: {threat.description}",
                        f"- Mitigations:",
                        ""
                    ])
                    for mitigation in threat.mitigations:
                        report.append(f"  - {mitigation}")
                    report.append("")
                    
        report.extend([
            "## Attack Surface Analysis",
            f"- Entry Points: {len(model.attack_surface.get('entry_points', []))}",
            f"- External Dependencies: {len(model.attack_surface.get('dependencies', []))}",
            f"- Network Exposure: {model.attack_surface.get('network_exposure', 'Unknown')}",
            ""
        ])
        
        return "\n".join(report)
        
    def prioritize_threats(self, threats: List[Threat]) -> List[Threat]:
        """
        Prioritize threats by risk score.
        
        Args:
            threats: List of threats
            
        Returns:
            Sorted list of threats
        """
        return sorted(threats, key=lambda t: t.risk_score, reverse=True)
        
    def generate_mitigations(self, threat: Threat) -> List[str]:
        """
        Generate mitigation strategies for a threat.
        
        Args:
            threat: Threat to mitigate
            
        Returns:
            List of mitigation strategies
        """
        mitigations = []
        
        if threat.category == ThreatCategory.SPOOFING:
            mitigations.extend([
                "Implement strong authentication",
                "Use multi-factor authentication",
                "Validate all inputs"
            ])
        elif threat.category == ThreatCategory.TAMPERING:
            mitigations.extend([
                "Use integrity checks",
                "Implement audit logging",
                "Apply principle of least privilege"
            ])
        elif threat.category == ThreatCategory.REPUDIATION:
            mitigations.extend([
                "Implement comprehensive logging",
                "Use digital signatures",
                "Maintain audit trails"
            ])
        elif threat.category == ThreatCategory.INFORMATION_DISCLOSURE:
            mitigations.extend([
                "Encrypt sensitive data",
                "Implement access controls",
                "Use secure communication channels"
            ])
        elif threat.category == ThreatCategory.DENIAL_OF_SERVICE:
            mitigations.extend([
                "Implement rate limiting",
                "Use resource quotas",
                "Deploy DDoS protection"
            ])
        elif threat.category == ThreatCategory.ELEVATION_OF_PRIVILEGE:
            mitigations.extend([
                "Apply least privilege principle",
                "Validate authorization",
                "Use secure defaults"
            ])
            
        return mitigations
        
    # Analysis methods
    def _identify_assets(self, directory: str) -> List[Asset]:
        """Identify valuable assets in the application."""
        assets = []
        
        # Check for database connections (data assets)
        for file_path in Path(directory).rglob("*.py"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if 'database' in content.lower() or 'db' in content.lower():
                assets.append(Asset(
                    name="Database",
                    type="data",
                    value="critical",
                    threats=[]
                ))
                break
                
        # Check for API endpoints (service assets)
        if any(Path(directory).rglob("*api*.py")):
            assets.append(Asset(
                name="API Services",
                type="service",
                value="high",
                threats=[]
            ))
            
        # Check for authentication (component assets)
        if any(Path(directory).rglob("*auth*.py")):
            assets.append(Asset(
                name="Authentication System",
                type="component",
                value="critical",
                threats=[]
            ))
            
        return assets
        
    def _analyze_attack_surface(self, directory: str) -> Dict[str, Any]:
        """Analyze the application's attack surface."""
        surface = {
            'entry_points': [],
            'dependencies': [],
            'network_exposure': 'low',
            'authentication_points': [],
            'data_flows': []
        }
        
        # Find entry points
        for file_path in Path(directory).rglob("*.py"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if '@app.route' in content or '@router' in content:
                surface['entry_points'].append(str(file_path))
                
            if 'import' in content:
                lines = content.split('\n')
                for line in lines:
                    if line.startswith('import ') or line.startswith('from '):
                        surface['dependencies'].append(line.strip())
                        
        # Determine network exposure
        if surface['entry_points']:
            surface['network_exposure'] = 'high' if len(surface['entry_points']) > 10 else 'medium'
            
        return surface
        
    def _identify_stride_threats(self, assets: List[Asset], attack_surface: Dict) -> List[Threat]:
        """Identify threats using STRIDE methodology."""
        threats = []
        threat_id = 1
        
        for asset in assets:
            # Spoofing threats
            if asset.type == "component" and "Authentication" in asset.name:
                threat = Threat(
                    id=f"T{threat_id:03d}",
                    name="Authentication Bypass",
                    category=ThreatCategory.SPOOFING,
                    description="Attacker could bypass authentication",
                    impact="Critical",
                    likelihood=3,
                    risk_score=15,
                    affected_components=[asset.name],
                    mitigations=self.generate_mitigations(Threat(
                        id="", name="", category=ThreatCategory.SPOOFING,
                        description="", impact="", likelihood=0, risk_score=0,
                        affected_components=[], mitigations=[]
                    ))
                )
                threats.append(threat)
                asset.threats.append(threat)
                threat_id += 1
                
            # Information Disclosure threats
            if asset.type == "data":
                threat = Threat(
                    id=f"T{threat_id:03d}",
                    name="Data Exposure",
                    category=ThreatCategory.INFORMATION_DISCLOSURE,
                    description="Sensitive data could be exposed",
                    impact="High",
                    likelihood=4,
                    risk_score=16,
                    affected_components=[asset.name],
                    mitigations=self.generate_mitigations(Threat(
                        id="", name="", category=ThreatCategory.INFORMATION_DISCLOSURE,
                        description="", impact="", likelihood=0, risk_score=0,
                        affected_components=[], mitigations=[]
                    ))
                )
                threats.append(threat)
                asset.threats.append(threat)
                threat_id += 1
                
        return threats
        
    def _calculate_risk_matrix(self) -> Dict[str, int]:
        """Calculate risk matrix from threats."""
        matrix = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }
        
        for threat in self.threats:
            if threat.risk_score >= 20:
                matrix['critical'] += 1
            elif threat.risk_score >= 15:
                matrix['high'] += 1
            elif threat.risk_score >= 10:
                matrix['medium'] += 1
            else:
                matrix['low'] += 1
                
        return matrix
        
    def export_model(self, model: ThreatModel, output_path: str) -> None:
        """
        Export threat model to file.
        
        Args:
            model: Threat model
            output_path: Output file path
        """
        model_dict = {
            'application': model.application,
            'version': model.version,
            'timestamp': model.timestamp,
            'assets': [
                {
                    'name': a.name,
                    'type': a.type,
                    'value': a.value,
                    'threat_count': len(a.threats)
                } for a in model.assets
            ],
            'threats': [
                {
                    'id': t.id,
                    'name': t.name,
                    'category': t.category.value,
                    'risk_score': t.risk_score,
                    'mitigations': t.mitigations
                } for t in model.threats
            ],
            'attack_surface': model.attack_surface,
            'risk_matrix': model.risk_matrix
        }
        
        with open(output_path, 'w') as f:
            json.dump(model_dict, f, indent=2)
            
        logger.info(f"Exported threat model to {output_path}")