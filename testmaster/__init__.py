#!/usr/bin/env python3
"""
TestMaster - Intelligent Test Generation & Maintenance System

A comprehensive, AI-powered test generation and maintenance platform that
automatically creates, verifies, and maintains test suites with self-healing
capabilities.

Main Components:
- Core: Orchestration and configuration
- Generators: Test generation engines  
- Converters: Test conversion and migration
- Verification: Quality assurance and self-healing
- Analysis: Coverage and failure pattern analysis
- Execution: Optimized test execution
- Monitoring: Real-time test monitoring
"""

__version__ = "2.0.0"
__author__ = "TestMaster Development Team"

# Core imports
from .core.orchestrator import PipelineOrchestrator
from .core.config import TestMasterConfig

# Main functionality
from .generators import (
    IntelligentTestGenerator,
    IntelligentTestBuilder,  # Legacy alias
    IntelligentTestBuilderV2,  # Legacy alias
    OfflineIntelligentTestBuilder,  # Legacy alias
    BaseGenerator,
    ModuleAnalysis,
    GenerationConfig
)

try:
    from .verification.self_healing import SelfHealingVerifier
except ImportError:
    SelfHealingVerifier = None

try:
    from .execution.runner import TestRunner
except ImportError:
    TestRunner = None

# Configuration instance
config = TestMasterConfig()

__all__ = [
    "PipelineOrchestrator",
    "TestMasterConfig", 
    "IntelligentTestGenerator",
    "IntelligentTestBuilder",
    "IntelligentTestBuilderV2", 
    "OfflineIntelligentTestBuilder",
    "BaseGenerator",
    "ModuleAnalysis",
    "GenerationConfig",
    "SelfHealingVerifier",
    "TestRunner",
    "config"
]