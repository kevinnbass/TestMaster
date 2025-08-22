"""
Certification Types and Data Structures
=======================================

Core type definitions and data structures for the Intelligence Certification Engine.
Provides enterprise-grade type safety for certification, compliance validation,
safety assessment, and trust verification with comprehensive standards compliance.

This module contains all Enum definitions and dataclass structures used throughout
the intelligence certification system, implementing industry standards and protocols.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: certification_types.py (140 lines)
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class CertificationLevel(Enum):
    """Levels of intelligence certification with progression hierarchy"""
    UNCERTIFIED = "uncertified"
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"
    AGI_CANDIDATE = "agi_candidate"
    AGI_CERTIFIED = "agi_certified"
    SUPERINTELLIGENCE = "superintelligence"


class ComplianceStandard(Enum):
    """International compliance standards for AI systems"""
    ISO_IEC_23053 = "iso_iec_23053"  # AI trustworthiness
    ISO_IEC_23894 = "iso_iec_23894"  # AI risk management
    IEEE_P2802 = "ieee_p2802"  # Data privacy
    GDPR = "gdpr"  # General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    AI_ACT = "ai_act"  # EU AI Act
    ASILOMAR = "asilomar"  # Asilomar AI Principles
    MONTREAL = "montreal"  # Montreal Declaration
    PARTNERSHIP_AI = "partnership_ai"  # Partnership on AI
    OECD_AI = "oecd_ai"  # OECD AI Principles


class SafetyLevel(Enum):
    """Safety classification levels for risk assessment"""
    MINIMAL_RISK = "minimal_risk"
    LOW_RISK = "low_risk"
    MODERATE_RISK = "moderate_risk"
    HIGH_RISK = "high_risk"
    CRITICAL_RISK = "critical_risk"
    UNACCEPTABLE_RISK = "unacceptable_risk"


class TrustDimension(Enum):
    """Dimensions of AI trustworthiness measurement"""
    RELIABILITY = "reliability"
    SAFETY = "safety"
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    PRIVACY = "privacy"
    ROBUSTNESS = "robustness"
    EXPLAINABILITY = "explainability"
    INTERPRETABILITY = "interpretability"
    AUDITABILITY = "auditability"


@dataclass
class CertificationCriteria:
    """Comprehensive criteria for certification evaluation"""
    criteria_id: str
    name: str
    category: str
    description: str
    minimum_score: float
    weight: float
    mandatory: bool
    test_methods: List[str]
    evidence_required: List[str]
    validation_frequency: timedelta = field(default_factory=lambda: timedelta(days=90))
    exemptions: List[str] = field(default_factory=list)


@dataclass
class ComplianceRequirement:
    """Detailed compliance requirement specification"""
    requirement_id: str
    standard: ComplianceStandard
    description: str
    verification_method: str
    documentation_required: List[str]
    automated_check: bool = True
    severity: str = "mandatory"
    grace_period: Optional[timedelta] = None


@dataclass
class SafetyAssessment:
    """Comprehensive safety assessment results"""
    assessment_id: str
    safety_level: SafetyLevel
    risk_score: float
    hazards_identified: List[str]
    mitigations_required: List[str]
    alignment_score: float
    human_oversight_required: bool
    deployment_restrictions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    reassessment_triggers: List[str] = field(default_factory=list)


@dataclass
class TrustScore:
    """Trust score for specific dimension"""
    dimension: TrustDimension
    score: float
    confidence: float
    evidence: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    evaluator_id: Optional[str] = None


@dataclass
class Certificate:
    """Official certification certificate with digital signature"""
    certificate_id: str
    system_id: str
    system_name: str
    certification_level: CertificationLevel
    issue_date: datetime
    expiry_date: datetime
    compliance_standards: List[ComplianceStandard]
    safety_level: SafetyLevel
    capabilities_certified: List[str]
    restrictions: List[str]
    trust_scores: Dict[TrustDimension, float]
    digital_signature: str
    issuer: str = "Intelligence Certification Authority"
    blockchain_hash: Optional[str] = None
    renewal_required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CertificationResult:
    """Complete certification result package"""
    certificate: Certificate
    test_results: Dict[str, Any]
    compliance_results: Dict[str, Any]
    safety_assessment: SafetyAssessment
    trust_evaluation: Dict[TrustDimension, TrustScore]
    recommendations: List[str]
    certification_report: Dict[str, Any]
    next_review_date: datetime
    continuous_monitoring_enabled: bool = True


# Export all certification types
__all__ = [
    'CertificationLevel', 'ComplianceStandard', 'SafetyLevel', 'TrustDimension',
    'CertificationCriteria', 'ComplianceRequirement', 'SafetyAssessment',
    'TrustScore', 'Certificate', 'CertificationResult'
]