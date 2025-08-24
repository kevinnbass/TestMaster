#!/usr/bin/env python3
"""
🏗️ MODULE: Agent Initialization - Intelligence Agent Setup
==================================================================

📋 PURPOSE:
    Intelligence agent initialization and coordination setup
    extracted from unified_intelligence_api.py via STEELCLAD protocol.

🎯 CORE FUNCTIONALITY:
    • Technical Debt Agent configuration
    • ML Analysis Agent setup
    • Coverage Agent initialization
    • Agent registration and coordination startup
    • Capability and task type definitions

🔄 EXTRACTION HISTORY:
==================================================================
📝 [2025-08-23] | Agent T | 🔧 STEELCLAD EXTRACTION
   └─ Goal: Extract agent initialization from unified_intelligence_api.py
   └─ Source: Lines 83-140 (57 lines)
   └─ Purpose: Separate agent setup into focused module

📞 DEPENDENCIES:
==================================================================
🤝 Imports: Agent coordination classes and logging
📤 Provides: Intelligence agent setup and configuration
"""

import logging

logger = logging.getLogger(__name__)


class AgentInitializer:
    """Handles initialization and configuration of intelligence agents."""
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
    
    def initialize_agents(self):
        """Initialize intelligence agents"""
        # Import here to avoid circular dependencies
        try:
            from ...orchestration.agent_coordinator import (
                IntelligenceAgent, AgentCapability
            )
            
            # Technical Debt Agent
            debt_agent = IntelligenceAgent(
                agent_id="debt_agent",
                name="Technical Debt Analyzer",
                capabilities=[
                    AgentCapability(
                        name="debt_analysis",
                        description="Analyze technical debt in codebases",
                        task_types=["analyze_debt", "quantify_debt", "remediation_plan"],
                        max_concurrent_tasks=2,
                        estimated_task_time=30.0,
                        specialized_domains=["code_quality", "maintainability"]
                    )
                ]
            )
            
            # ML Analysis Agent
            ml_agent = IntelligenceAgent(
                agent_id="ml_agent", 
                name="ML Code Analyzer",
                capabilities=[
                    AgentCapability(
                        name="ml_analysis",
                        description="Analyze ML/AI code for issues",
                        task_types=["analyze_ml", "check_tensor_shapes", "model_architecture"],
                        max_concurrent_tasks=1,
                        estimated_task_time=45.0,
                        specialized_domains=["machine_learning", "deep_learning", "data_science"]
                    )
                ]
            )
            
            # Coverage Agent
            coverage_agent = IntelligenceAgent(
                agent_id="coverage_agent",
                name="Coverage Analyzer", 
                capabilities=[
                    AgentCapability(
                        name="coverage_analysis",
                        description="Analyze code coverage and quality",
                        task_types=["analyze_coverage", "generate_report", "identify_gaps"],
                        max_concurrent_tasks=3,
                        estimated_task_time=20.0,
                        specialized_domains=["testing", "quality_assurance"]
                    )
                ]
            )
            
            # Register agents
            self.coordinator.register_agent(debt_agent)
            self.coordinator.register_agent(ml_agent)
            self.coordinator.register_agent(coverage_agent)
            
            # Start coordination
            self.coordinator.start_coordination()
            
            logger.info("Intelligence agents initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to initialize agents - missing dependencies: {e}")
            # Create fallback agent configuration
            self._initialize_fallback_agents()
    
    def _initialize_fallback_agents(self):
        """Initialize fallback agents if imports fail"""
        logger.warning("Using fallback agent initialization")
        # Simplified agent setup without full coordination
        pass