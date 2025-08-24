#!/usr/bin/env python3
"""
LLM Intelligence Scanner Data Models
====================================

Data structures and models for the LLM intelligence scanner system.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"  # Local LLM
    MOCK = "mock"  # For testing without API


@dataclass
class LLMIntelligenceEntry:
    """Single entry in the LLM intelligence map"""
    full_path: str
    relative_path: str
    file_hash: str
    analysis_timestamp: str
    module_summary: str
    functionality_details: str
    dependencies_analysis: str
    security_implications: str
    testing_requirements: str
    architectural_role: str
    primary_classification: str
    secondary_classifications: List[str]
    reorganization_recommendations: List[str]
    confidence_score: float
    key_features: List[str]
    integration_points: List[str]
    complexity_assessment: str
    maintainability_notes: str
    file_size: int
    line_count: int
    class_count: int
    function_count: int


@dataclass
class LLMIntelligenceMap:
    """Complete LLM intelligence map"""
    scan_timestamp: str
    total_files_scanned: int
    total_lines_analyzed: int
    directory_structure: Dict[str, Any]
    intelligence_entries: List[LLMIntelligenceEntry]
    classification_summary: Dict[str, int]
    reorganization_insights: Dict[str, Any]
    scan_metadata: Dict[str, Any]
