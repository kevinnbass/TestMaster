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
from .generators.intelligent import IntelligentTestGenerator
from .verification.self_healing import SelfHealingVerifier
from .execution.runner import TestRunner

# Configuration instance
config = TestMasterConfig()

__all__ = [
    "PipelineOrchestrator",
    "TestMasterConfig", 
    "IntelligentTestGenerator",
    "SelfHealingVerifier",
    "TestRunner",
    "config"
]