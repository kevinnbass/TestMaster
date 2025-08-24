#!/usr/bin/env python3
"""
LLM Intelligence Data Classes
=============================

Data structures for LLM intelligence system.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"
    MOCK = "mock"


class Classification(Enum):
    """Standard classification categories"""
    SECURITY = "security"
    INTELLIGENCE = "intelligence"
    FRONTEND_DASHBOARD = "frontend_dashboard"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    UTILITY = "utility"
    API = "api"
    DATABASE = "database"
    DATA_PROCESSING = "data_processing"
    ORCHESTRATION = "orchestration"
    AUTOMATION = "automation"
    MONITORING = "monitoring"
    ANALYTICS = "analytics"
    DEVOPS = "devops"
    UNCATEGORIZED = "uncategorized"


@dataclass
class LLMIntelligenceEntry:
    """Single entry in the LLM intelligence map"""
    full_path: str
    relative_path: str
    file_hash: str
    analysis_timestamp: str
    module_summary: str = ""
    functionality_details: str = ""
    dependencies_analysis: str = ""
    security_implications: str = ""
    testing_requirements: str = ""
    architectural_role: str = ""
    primary_classification: str = "uncategorized"
    secondary_classifications: List[str] = field(default_factory=list)
    reorganization_recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.5
    key_features: List[str] = field(default_factory=list)
    integration_points: List[str] = field(default_factory=list)
    complexity_assessment: str = "unknown"
    maintainability_notes: str = ""
    file_size: int = 0
    line_count: int = 0
    class_count: int = 0
    function_count: int = 0
    analysis_errors: List[str] = field(default_factory=list)


@dataclass
class StaticAnalysisResult:
    """Results from static analysis tools"""
    semantic: Dict[str, Any] = field(default_factory=dict)
    relationship: Dict[str, Any] = field(default_factory=dict)
    pattern: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class IntegratedIntelligence:
    """Integrated intelligence from multiple sources"""
    file_path: str
    relative_path: str
    static_analysis: StaticAnalysisResult = field(default_factory=StaticAnalysisResult)
    llm_analysis: LLMIntelligenceEntry = field(default_factory=LLMIntelligenceEntry)
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    integrated_classification: str = "uncategorized"
    reorganization_priority: int = 5
    integration_confidence: float = 0.5
    final_recommendations: List[str] = field(default_factory=list)
    synthesis_reasoning: str = ""
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LLMIntelligenceMap:
    """Complete LLM intelligence map"""
    scan_timestamp: str
    scan_id: str
    total_files_scanned: int = 0
    total_lines_analyzed: int = 0
    directory_structure: Dict[str, Any] = field(default_factory=dict)
    intelligence_entries: List[LLMIntelligenceEntry] = field(default_factory=list)
    scan_metadata: Dict[str, Any] = field(default_factory=dict)
    scan_statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReorganizationPhase:
    """Single phase in the reorganization plan"""
    phase_number: int
    phase_name: str
    description: str
    modules: List[Dict[str, Any]] = field(default_factory=list)
    estimated_time_minutes: int = 0
    risk_level: str = "medium"
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class ReorganizationPlan:
    """Complete reorganization plan"""
    plan_timestamp: str
    total_modules: int = 0
    reorganization_phases: List[ReorganizationPhase] = field(default_factory=list)
    estimated_total_time_hours: float = 0.0
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    success_metrics: Dict[str, Any] = field(default_factory=dict)
    implementation_guidelines: List[str] = field(default_factory=list)


@dataclass
class LLMConfig:
    """Configuration for LLM intelligence system"""
    provider: str = "mock"
    model: str = "default"
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30
    cache_enabled: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    confidence_threshold: float = 0.7


@dataclass
class ScanResult:
    """Result of a complete intelligence scan"""
    scan_id: str
    scan_timestamp: str
    intelligence_map: LLMIntelligenceMap
    integrated_intelligence: List[IntegratedIntelligence]
    reorganization_plan: ReorganizationPlan
    scan_duration_seconds: float
    files_processed: int
    errors_encountered: List[str]
    warnings: List[str]

