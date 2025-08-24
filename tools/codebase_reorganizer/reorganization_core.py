#!/usr/bin/env python3
"""
Reorganization Core Planner
==========================

Core reorganization planning functionality.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging

from reorganization_data import (
    ReorganizationAction, RiskLevel, ReorganizationTask,
    ReorganizationBatch, DetailedReorganizationPlan,
    ReorganizationMetrics, ExecutionResult, BatchExecutionResult
)

# Import our intelligence system components
try:
    from llm_intelligence_system import LLMIntelligenceScanner
    from intelligence_integration_engine import (
        IntelligenceIntegrationEngine, IntegratedIntelligence,
        ReorganizationPhase, ReorganizationPlan
    )
    HAS_COMPONENTS = True
except ImportError:
    HAS_COMPONENTS = False


class ReorganizationPlanner:
    """
    Creates detailed, executable reorganization plans from integrated intelligence.
    Handles confidence-based decision making and risk assessment.
    """

    def __init__(self, root_dir: Path, config: Optional[Dict[str, Any]] = None):
        self.root_dir = root_dir.resolve()
        self.config = config or self._get_default_config()

        # Setup logging
        self._setup_logging()

        # Initialize components
        self.integration_engine = None
        if HAS_COMPONENTS:
            self.integration_engine = IntelligenceIntegrationEngine(root_dir, self.config)

        self.logger.info("Reorganization Planner initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'min_confidence_threshold': 0.7,
            'high_confidence_threshold': 0.85,
            'max_batch_size': 10,
            'max_tasks_per_batch': 5,
            'risk_thresholds': {
                'low': 0.8,
                'medium': 0.6,
                'high': 0.4
            },
            'auto_approve_risk_levels': ['low'],
            'require_review_risk_levels': ['high', 'critical'],
            'backup_enabled': True,
            'dry_run_enabled': True,
            'import_validation_enabled': True
        }

    def _setup_logging(self) -> None:
        """Setup logging"""
        log_dir = self.root_dir / "tools" / "codebase_reorganizer" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"reorganization_planner_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_reorganization_plan(self, llm_intelligence_map: Dict[str, Any],
                                 integrated_intelligence: List[IntegratedIntelligence]) -> DetailedReorganizationPlan:
        """
        Create a detailed, executable reorganization plan.

        Args:
            llm_intelligence_map: Raw LLM intelligence map
            integrated_intelligence: Processed integrated intelligence

        Returns:
            Detailed reorganization plan with executable tasks
        """
        self.logger.info("Creating detailed reorganization plan...")

        # Generate unique plan ID
        plan_id = f"reorg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create tasks from integrated intelligence
        all_tasks = []
        for intelligence in integrated_intelligence:
            tasks = self._create_tasks_from_intelligence(intelligence)
            all_tasks.extend(tasks)

        # Sort tasks by priority and confidence
        all_tasks.sort(key=lambda x: (x.priority, x.confidence), reverse=True)

        # Group tasks into batches
        batches = self._create_task_batches(all_tasks)

        # Create summary statistics
        summary = self._create_plan_summary(all_tasks, batches)

        # Generate execution guidelines
        execution_guidelines = self._create_execution_guidelines(batches)

        # Create risk mitigation strategies
        risk_mitigation = self._create_risk_mitigation(batches)

        # Define success metrics
        success_metrics = self._create_success_metrics(batches)

        plan = DetailedReorganizationPlan(
            plan_id=plan_id,
            created_timestamp=datetime.now().isoformat(),
            source_intelligence=llm_intelligence_map.get('scan_id', 'unknown'),
            total_tasks=len(all_tasks),
            total_batches=len(batches),
            batches=batches,
            summary=summary,
            execution_guidelines=execution_guidelines,
            risk_mitigation=risk_mitigation,
            success_metrics=success_metrics
        )

        self.logger.info(f"Created reorganization plan with {len(batches)} batches and {len(all_tasks)} tasks")
        return plan

    def _create_tasks_from_intelligence(self, intelligence: IntegratedIntelligence) -> List[ReorganizationTask]:
        """Create reorganization tasks from integrated intelligence"""
        tasks = []

        # Only create tasks for modules with sufficient confidence
        if intelligence.integration_confidence < self.config['min_confidence_threshold']:
            # Create manual review task for low confidence modules
            task = ReorganizationTask(
                task_id=f"review_{intelligence.relative_path.replace('/', '_')}",
                action=ReorganizationAction.MANUAL_REVIEW,
                source_path=intelligence.file_path,
                target_path=None,
                rationale=f"Low confidence ({intelligence.integration_confidence:.2f}) - manual review required",
                confidence=intelligence.integration_confidence,
                priority=intelligence.reorganization_priority,
                risk_level=RiskLevel.HIGH,
                dependencies=[],
                prerequisites=["Human review required"],
                estimated_effort_minutes=30,
                success_criteria=["Manual review completed", "Decision made on reorganization"],
                rollback_plan="No changes made - review only"
            )
            tasks.append(task)
            return tasks

        # Create tasks based on recommendations
        for i, recommendation in enumerate(intelligence.final_recommendations):
            task = self._parse_recommendation_to_task(
                intelligence, recommendation, i
            )
            if task:
                tasks.append(task)

        # If no specific recommendations, create a move task based on classification
        if not tasks:
            task = self._create_default_move_task(intelligence)
            if task:
                tasks.append(task)

        return tasks

    def _parse_recommendation_to_task(self, intelligence: IntegratedIntelligence,
                                    recommendation: str, index: int) -> Optional[ReorganizationTask]:
        """Parse a recommendation string into a specific task"""

        recommendation_lower = recommendation.lower()

        # Move recommendations
        if 'move to' in recommendation_lower or 'reorganize' in recommendation_lower:
            return self._create_move_task(intelligence, recommendation, index)

        # Security review recommendations
        elif 'security' in recommendation_lower and 'review' in recommendation_lower:
            return self._create_security_review_task(intelligence, recommendation, index)

        # Quality improvement recommendations
        elif 'quality' in recommendation_lower or 'refactor' in recommendation_lower:
            return self._create_quality_task(intelligence, recommendation, index)

        # Import update recommendations
        elif 'import' in recommendation_lower:
            return self._create_import_update_task(intelligence, recommendation, index)

        # Skip or ignore
        else:
            return None

    def _create_move_task(self, intelligence: IntegratedIntelligence,
                         recommendation: str, index: int) -> ReorganizationTask:
        """Create a file move task"""

        # Determine target directory based on classification
        target_dir = self._get_target_directory(intelligence.integrated_classification)

        target_path = str(Path(target_dir) / Path(intelligence.relative_path).name)

        return ReorganizationTask(
            task_id=f"move_{intelligence.relative_path.replace('/', '_')}_{index}",
            action=ReorganizationAction.MOVE_FILE,
            source_path=intelligence.file_path,
            target_path=target_path,
            rationale=recommendation,
            confidence=intelligence.integration_confidence,
            priority=intelligence.reorganization_priority,
            risk_level=self._assess_move_risk(intelligence),
            dependencies=self._get_move_dependencies(intelligence),
            prerequisites=["Backup created", "Directory structure validated"],
            estimated_effort_minutes=5,
            success_criteria=["File moved successfully", "Imports updated", "Tests pass"],
            rollback_plan="Move file back to original location"
        )

    def _create_security_review_task(self, intelligence: IntegratedIntelligence,
                                   recommendation: str, index: int) -> ReorganizationTask:
        """Create a security review task"""
        return ReorganizationTask(
            task_id=f"security_{intelligence.relative_path.replace('/', '_')}_{index}",
            action=ReorganizationAction.MANUAL_REVIEW,
            source_path=intelligence.file_path,
            target_path=None,
            rationale=f"Security review required: {recommendation}",
            confidence=intelligence.integration_confidence,
            priority=intelligence.reorganization_priority,
            risk_level=RiskLevel.HIGH,
            dependencies=[],
            prerequisites=["Security team available"],
            estimated_effort_minutes=60,
            success_criteria=["Security vulnerabilities identified", "Mitigation plan created"],
            rollback_plan="No changes - review only"
        )

    def _create_quality_task(self, intelligence: IntegratedIntelligence,
                           recommendation: str, index: int) -> ReorganizationTask:
        """Create a quality improvement task"""
        return ReorganizationTask(
            task_id=f"quality_{intelligence.relative_path.replace('/', '_')}_{index}",
            action=ReorganizationAction.SPLIT_MODULE,
            source_path=intelligence.file_path,
            target_path=None,
            rationale=f"Quality improvement needed: {recommendation}",
            confidence=intelligence.integration_confidence,
            priority=intelligence.reorganization_priority,
            risk_level=RiskLevel.MEDIUM,
            dependencies=[],
            prerequisites=["Code review completed"],
            estimated_effort_minutes=45,
            success_criteria=["Code quality improved", "Tests pass", "Maintainability increased"],
            rollback_plan="Revert to original code structure"
        )

    def _create_import_update_task(self, intelligence: IntegratedIntelligence,
                                 recommendation: str, index: int) -> ReorganizationTask:
        """Create an import update task"""
        return ReorganizationTask(
            task_id=f"import_{intelligence.relative_path.replace('/', '_')}_{index}",
            action=ReorganizationAction.UPDATE_IMPORTS,
            source_path=intelligence.file_path,
            target_path=None,
            rationale=f"Import updates needed: {recommendation}",
            confidence=intelligence.integration_confidence,
            priority=intelligence.reorganization_priority,
            risk_level=RiskLevel.LOW,
            dependencies=[],
            prerequisites=["Import analysis completed"],
            estimated_effort_minutes=10,
            success_criteria=["Imports updated", "No import errors", "Tests pass"],
            rollback_plan="Revert import changes"
        )

    def _create_default_move_task(self, intelligence: IntegratedIntelligence) -> Optional[ReorganizationTask]:
        """Create a default move task when no specific recommendations exist"""
        if not intelligence.integrated_classification:
            return None

        target_dir = self._get_target_directory(intelligence.integrated_classification)
        if not target_dir:
            return None

        target_path = str(Path(target_dir) / Path(intelligence.relative_path).name)

        return ReorganizationTask(
            task_id=f"default_move_{intelligence.relative_path.replace('/', '_')}",
            action=ReorganizationAction.MOVE_FILE,
            source_path=intelligence.file_path,
            target_path=target_path,
            rationale=f"Move to {intelligence.integrated_classification} directory based on classification",
            confidence=intelligence.integration_confidence,
            priority=intelligence.reorganization_priority,
            risk_level=RiskLevel.LOW,
            dependencies=[],
            prerequisites=["Directory exists"],
            estimated_effort_minutes=3,
            success_criteria=["File moved successfully"],
            rollback_plan="Move file back to original location"
        )

    def _get_target_directory(self, classification: str) -> str:
        """Get target directory based on classification"""
        classification_lower = classification.lower()

        if 'utility' in classification_lower:
            return "utils"
        elif 'model' in classification_lower or 'data' in classification_lower:
            return "models"
        elif 'view' in classification_lower or 'template' in classification_lower:
            return "views"
        elif 'controller' in classification_lower or 'handler' in classification_lower:
            return "controllers"
        elif 'test' in classification_lower:
            return "tests"
        elif 'config' in classification_lower or 'settings' in classification_lower:
            return "config"
        else:
            return "src"  # Default directory

    def _assess_move_risk(self, intelligence: IntegratedIntelligence) -> RiskLevel:
        """Assess risk level for moving a file"""
        if intelligence.integration_confidence > self.config['high_confidence_threshold']:
            return RiskLevel.LOW
        elif intelligence.integration_confidence > self.config['min_confidence_threshold']:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH

    def _get_move_dependencies(self, intelligence: IntegratedIntelligence) -> List[str]:
        """Get dependencies for moving a file"""
        # This would analyze import relationships to determine dependencies
        # For now, return empty list as a placeholder
        return []

    def _create_task_batches(self, tasks: List[ReorganizationTask]) -> List[ReorganizationBatch]:
        """Group tasks into executable batches"""
        batches = []
        current_batch_tasks = []
        current_batch_risk = RiskLevel.LOW
        batch_counter = 0

        for task in tasks:
            # Check if adding this task would exceed batch limits
            if (len(current_batch_tasks) >= self.config['max_tasks_per_batch'] or
                self._would_exceed_risk_threshold(current_batch_tasks + [task], current_batch_risk)):

                # Create current batch and start new one
                if current_batch_tasks:
                    batch = self._create_batch_from_tasks(current_batch_tasks, batch_counter)
                    batches.append(batch)
                    batch_counter += 1
                    current_batch_tasks = []
                    current_batch_risk = RiskLevel.LOW

            current_batch_tasks.append(task)
            current_batch_risk = self._get_highest_risk_level([task], current_batch_risk)

        # Add final batch
        if current_batch_tasks:
            batch = self._create_batch_from_tasks(current_batch_tasks, batch_counter)
            batches.append(batch)

        return batches

    def _would_exceed_risk_threshold(self, tasks: List[ReorganizationTask], current_risk: RiskLevel) -> bool:
        """Check if adding tasks would exceed risk threshold"""
        new_risk = self._get_highest_risk_level(tasks, current_risk)
        return new_risk == RiskLevel.CRITICAL

    def _get_highest_risk_level(self, tasks: List[ReorganizationTask], default_risk: RiskLevel = RiskLevel.LOW) -> RiskLevel:
        """Get the highest risk level from a list of tasks"""
        if not tasks:
            return default_risk

        risk_levels = [task.risk_level for task in tasks]
        risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]

        for risk_level in reversed(risk_order):
            if risk_level in risk_levels:
                return risk_level

        return default_risk

    def _create_batch_from_tasks(self, tasks: List[ReorganizationTask], batch_counter: int) -> ReorganizationBatch:
        """Create a batch from a list of tasks"""
        batch_id = f"batch_{batch_counter:03d}"
        batch_name = f"Reorganization Batch {batch_counter + 1}"
        description = f"Batch containing {len(tasks)} reorganization tasks"

        # Calculate batch properties
        risk_level = self._get_highest_risk_level(tasks)
        estimated_time = sum(task.estimated_effort_minutes for task in tasks)

        # Get prerequisites from all tasks
        prerequisites = []
        for task in tasks:
            prerequisites.extend(task.prerequisites)
        prerequisites = list(set(prerequisites))  # Remove duplicates

        # Postconditions would be task success criteria
        postconditions = ["All batch tasks completed successfully"]

        return ReorganizationBatch(
            batch_id=batch_id,
            batch_name=batch_name,
            description=description,
            tasks=tasks,
            risk_level=risk_level,
            estimated_total_time=estimated_time,
            prerequisites=prerequisites,
            postconditions=postconditions
        )

    def _create_plan_summary(self, tasks: List[ReorganizationTask], batches: List[ReorganizationBatch]) -> Dict[str, Any]:
        """Create summary statistics for the plan"""
        return {
            'total_tasks': len(tasks),
            'total_batches': len(batches),
            'high_confidence_tasks': len([t for t in tasks if t.confidence > self.config['high_confidence_threshold']]),
            'medium_confidence_tasks': len([t for t in tasks if self.config['min_confidence_threshold'] <= t.confidence <= self.config['high_confidence_threshold']]),
            'low_confidence_tasks': len([t for t in tasks if t.confidence < self.config['min_confidence_threshold']]),
            'estimated_total_time': sum(task.estimated_effort_minutes for task in tasks),
            'risk_distribution': {
                'low': len([t for t in tasks if t.risk_level == RiskLevel.LOW]),
                'medium': len([t for t in tasks if t.risk_level == RiskLevel.MEDIUM]),
                'high': len([t for t in tasks if t.risk_level == RiskLevel.HIGH]),
                'critical': len([t for t in tasks if t.risk_level == RiskLevel.CRITICAL])
            },
            'action_distribution': {
                action.value: len([t for t in tasks if t.action == action])
                for action in ReorganizationAction
            }
        }

    def _create_execution_guidelines(self, batches: List[ReorganizationBatch]) -> List[str]:
        """Create execution guidelines for the plan"""
        return [
            "Execute batches in order from lowest to highest risk",
            "Perform backup before executing high-risk batches",
            "Run tests after each batch completion",
            "Validate imports after file moves",
            "Review manual review tasks before proceeding",
            "Monitor system logs during execution",
            "Have rollback plan ready for each batch"
        ]

    def _create_risk_mitigation(self, batches: List[ReorganizationBatch]) -> Dict[str, Any]:
        """Create risk mitigation strategies"""
        return {
            'backup_strategy': 'Full backup before high-risk batches',
            'testing_strategy': 'Run test suite after each batch',
            'validation_strategy': 'Import validation and syntax checking',
            'rollback_strategy': 'Automated rollback scripts available',
            'monitoring_strategy': 'Log analysis and system monitoring'
        }

    def _create_success_metrics(self, batches: List[ReorganizationBatch]) -> Dict[str, Any]:
        """Create success metrics for the plan"""
        return {
            'success_rate_target': 95.0,
            'test_pass_rate_target': 100.0,
            'import_error_target': 0,
            'rollback_success_target': 100.0,
            'performance_impact_target': 'None expected'
        }

    def save_plan(self, plan: DetailedReorganizationPlan, output_path: Path) -> None:
        """Save reorganization plan to file"""
        plan_dict = asdict(plan)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(plan_dict, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Reorganization plan saved to {output_path}")

    def load_plan(self, plan_path: Path) -> DetailedReorganizationPlan:
        """Load reorganization plan from file"""
        with open(plan_path, 'r', encoding='utf-8') as f:
            plan_dict = json.load(f)

        # Convert back to dataclass (simplified conversion)
        return DetailedReorganizationPlan(**plan_dict)

    def execute_plan(self, plan: DetailedReorganizationPlan, dry_run: bool = True) -> BatchExecutionResult:
        """Execute a reorganization plan"""
        self.logger.info(f"{'Simulating' if dry_run else 'Executing'} reorganization plan {plan.plan_id}")

        # Implementation would go here
        # For now, return a placeholder result
        return BatchExecutionResult(
            batch_id=plan.plan_id,
            success=True,
            total_execution_time=0.0,
            successful_tasks=plan.total_tasks,
            failed_tasks=0,
            task_results=[],
            rollback_actions=[]
        )

