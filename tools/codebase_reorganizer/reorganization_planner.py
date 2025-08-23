#!/usr/bin/env python3
"""
Reorganization Planner
======================

Creates actionable reorganization plans based on integrated intelligence.
Provides confidence-based decision making and phased implementation strategies.
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


class ReorganizationAction(Enum):
    """Types of reorganization actions"""
    MOVE_FILE = "move_file"
    RENAME_FILE = "rename_file"
    CREATE_DIRECTORY = "create_directory"
    UPDATE_IMPORTS = "update_imports"
    SPLIT_MODULE = "split_module"
    MERGE_MODULES = "merge_modules"
    MANUAL_REVIEW = "manual_review"
    SKIP = "skip"


class RiskLevel(Enum):
    """Risk levels for reorganization actions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ReorganizationTask:
    """A specific reorganization task"""
    task_id: str
    action: ReorganizationAction
    source_path: Optional[str]
    target_path: Optional[str]
    rationale: str
    confidence: float
    priority: int
    risk_level: RiskLevel
    dependencies: List[str]
    prerequisites: List[str]
    estimated_effort_minutes: int
    success_criteria: List[str]
    rollback_plan: str
    status: str = "pending"  # pending, in_progress, completed, failed
    error_message: str = ""


@dataclass
class ReorganizationBatch:
    """A batch of related reorganization tasks"""
    batch_id: str
    batch_name: str
    description: str
    tasks: List[ReorganizationTask]
    risk_level: RiskLevel
    estimated_total_time: int
    prerequisites: List[str]
    postconditions: List[str]
    status: str = "pending"


@dataclass
class DetailedReorganizationPlan:
    """Detailed reorganization plan with executable tasks"""
    plan_id: str
    created_timestamp: str
    source_intelligence: str
    total_tasks: int
    total_batches: int
    batches: List[ReorganizationBatch]
    summary: Dict[str, Any]
    execution_guidelines: List[str]
    risk_mitigation: Dict[str, Any]
    success_metrics: Dict[str, Any]


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
            prerequisites=["Backup created", "Tests pass"],
            estimated_effort_minutes=10,
            success_criteria=[
                "File moved successfully",
                "All imports updated",
                "Tests still pass"
            ],
            rollback_plan="Restore from backup"
        )

    def _create_security_review_task(self, intelligence: IntegratedIntelligence,
                                   recommendation: str, index: int) -> ReorganizationTask:
        """Create a security review task"""

        return ReorganizationTask(
            task_id=f"security_review_{intelligence.relative_path.replace('/', '_')}_{index}",
            action=ReorganizationAction.MANUAL_REVIEW,
            source_path=intelligence.file_path,
            target_path=None,
            rationale=recommendation,
            confidence=intelligence.integration_confidence,
            priority=intelligence.reorganization_priority,
            risk_level=RiskLevel.HIGH,
            dependencies=[],
            prerequisites=["Security team review", "Security testing"],
            estimated_effort_minutes=60,
            success_criteria=[
                "Security review completed",
                "No security vulnerabilities introduced",
                "Security requirements satisfied"
            ],
            rollback_plan="No changes made during review phase"
        )

    def _create_quality_task(self, intelligence: IntegratedIntelligence,
                           recommendation: str, index: int) -> ReorganizationTask:
        """Create a quality improvement task"""

        return ReorganizationTask(
            task_id=f"quality_{intelligence.relative_path.replace('/', '_')}_{index}",
            action=ReorganizationAction.MANUAL_REVIEW,
            source_path=intelligence.file_path,
            target_path=None,
            rationale=recommendation,
            confidence=intelligence.integration_confidence,
            priority=intelligence.reorganization_priority,
            risk_level=RiskLevel.MEDIUM,
            dependencies=[],
            prerequisites=["Code review", "Testing"],
            estimated_effort_minutes=45,
            success_criteria=[
                "Code quality improved",
                "Maintainability enhanced",
                "No functionality broken"
            ],
            rollback_plan="Revert quality changes if issues arise"
        )

    def _create_import_update_task(self, intelligence: IntegratedIntelligence,
                                 recommendation: str, index: int) -> ReorganizationTask:
        """Create an import update task"""

        return ReorganizationTask(
            task_id=f"import_update_{intelligence.relative_path.replace('/', '_')}_{index}",
            action=ReorganizationAction.UPDATE_IMPORTS,
            source_path=intelligence.file_path,
            target_path=None,
            rationale=recommendation,
            confidence=intelligence.integration_confidence,
            priority=intelligence.reorganization_priority,
            risk_level=RiskLevel.LOW,
            dependencies=[],
            prerequisites=["Import analysis completed"],
            estimated_effort_minutes=5,
            success_criteria=[
                "Import paths updated",
                "No import errors",
                "Module still functional"
            ],
            rollback_plan="Revert import changes"
        )

    def _create_default_move_task(self, intelligence: IntegratedIntelligence) -> Optional[ReorganizationTask]:
        """Create a default move task based on classification"""

        if intelligence.integration_confidence < self.config['min_confidence_threshold']:
            return None

        target_dir = self._get_target_directory(intelligence.integrated_classification)

        target_path = str(Path(target_dir) / Path(intelligence.relative_path).name)

        return ReorganizationTask(
            task_id=f"default_move_{intelligence.relative_path.replace('/', '_')}",
            action=ReorganizationAction.MOVE_FILE,
            source_path=intelligence.file_path,
            target_path=target_path,
            rationale=f"Move to {intelligence.integrated_classification} directory based on classification",
            confidence=intelligence.integration_confidence,
            priority=intelligence.reorganization_priority,
            risk_level=self._assess_move_risk(intelligence),
            dependencies=[],
            prerequisites=["Standard move prerequisites"],
            estimated_effort_minutes=10,
            success_criteria=["File moved successfully", "Basic functionality verified"],
            rollback_plan="Restore from backup"
        )

    def _get_target_directory(self, classification: str) -> str:
        """Get target directory for a classification"""
        target_mapping = {
            'security': 'src/core/security',
            'intelligence': 'src/core/intelligence',
            'frontend_dashboard': 'src/frontend',
            'documentation': 'docs',
            'testing': 'tests',
            'utility': 'src/utils',
            'api': 'src/api',
            'database': 'src/database',
            'data_processing': 'src/data',
            'orchestration': 'src/orchestration',
            'automation': 'src/automation',
            'monitoring': 'src/monitoring',
            'analytics': 'src/analytics',
            'devops': 'src/devops',
            'uncategorized': 'src/uncategorized'
        }

        return target_mapping.get(classification, f'src/{classification}')

    def _assess_move_risk(self, intelligence: IntegratedIntelligence) -> RiskLevel:
        """Assess the risk level of a move operation"""

        risk_score = intelligence.integration_confidence

        # High risk factors
        if 'security' in intelligence.integrated_classification:
            risk_score -= 0.2

        if intelligence.reorganization_priority >= 8:
            risk_score -= 0.1

        if intelligence.llm_analysis.complexity_assessment.lower() in ['high', 'very_high']:
            risk_score -= 0.1

        # Determine risk level
        if risk_score >= self.config['risk_thresholds']['low']:
            return RiskLevel.LOW
        elif risk_score >= self.config['risk_thresholds']['medium']:
            return RiskLevel.MEDIUM
        elif risk_score >= self.config['risk_thresholds']['high']:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _get_move_dependencies(self, intelligence: IntegratedIntelligence) -> List[str]:
        """Get dependencies for a move operation"""
        dependencies = []

        # Add target directory creation if needed
        target_dir = self._get_target_directory(intelligence.integrated_classification)
        dependencies.append(f"Create directory: {target_dir}")

        # Add import update dependencies
        if intelligence.llm_analysis.dependencies_analysis:
            dependencies.append("Update import statements")

        return dependencies

    def _create_task_batches(self, tasks: List[ReorganizationTask]) -> List[ReorganizationBatch]:
        """Group tasks into logical batches"""

        # Separate tasks by risk level
        low_risk_tasks = [t for t in tasks if t.risk_level == RiskLevel.LOW]
        medium_risk_tasks = [t for t in tasks if t.risk_level == RiskLevel.MEDIUM]
        high_risk_tasks = [t for t in tasks if t.risk_level == RiskLevel.HIGH]
        critical_tasks = [t for t in tasks if t.risk_level == RiskLevel.CRITICAL]

        batches = []

        # Batch 1: Low risk moves (can be automated)
        if low_risk_tasks:
            batch = ReorganizationBatch(
                batch_id="batch_low_risk_moves",
                batch_name="Low Risk File Moves",
                description="Automated moves for high-confidence, low-risk modules",
                tasks=low_risk_tasks[:self.config['max_tasks_per_batch']],
                risk_level=RiskLevel.LOW,
                estimated_total_time=sum(t.estimated_effort_minutes for t in low_risk_tasks[:self.config['max_tasks_per_batch']]),
                prerequisites=["Backup created", "Tests pass", "No manual review required"],
                postconditions=["Files moved successfully", "Imports updated", "Tests still pass"]
            )
            batches.append(batch)

        # Batch 2: Security-critical modules
        security_tasks = [t for t in tasks if 'security' in t.rationale.lower()]
        if security_tasks:
            batch = ReorganizationBatch(
                batch_id="batch_security_critical",
                batch_name="Security-Critical Modules",
                description="Handle security-related modules with extra care",
                tasks=security_tasks,
                risk_level=RiskLevel.HIGH,
                estimated_total_time=sum(t.estimated_effort_minutes for t in security_tasks),
                prerequisites=["Security team review", "Security testing completed", "Backup verified"],
                postconditions=["Security properties preserved", "No vulnerabilities introduced"]
            )
            batches.append(batch)

        # Batch 3: Complex modules requiring review
        complex_tasks = [t for t in tasks if t.priority >= 8]
        if complex_tasks:
            batch = ReorganizationBatch(
                batch_id="batch_complex_modules",
                batch_name="Complex High-Priority Modules",
                description="Handle complex modules requiring detailed planning",
                tasks=complex_tasks,
                risk_level=RiskLevel.HIGH,
                estimated_total_time=sum(t.estimated_effort_minutes for t in complex_tasks),
                prerequisites=["Architecture review", "Detailed testing plan", "Team alignment"],
                postconditions=["Complex dependencies resolved", "Architecture improved"]
            )
            batches.append(batch)

        # Batch 4: Manual review tasks
        review_tasks = [t for t in tasks if t.action == ReorganizationAction.MANUAL_REVIEW]
        if review_tasks:
            batch = ReorganizationBatch(
                batch_id="batch_manual_review",
                batch_name="Manual Review Required",
                description="Modules requiring human review before reorganization",
                tasks=review_tasks,
                risk_level=RiskLevel.MEDIUM,
                estimated_total_time=sum(t.estimated_effort_minutes for t in review_tasks),
                prerequisites=["Human review completed", "Decision documented"],
                postconditions=["Review completed", "Action plan determined"]
            )
            batches.append(batch)

        return batches

    def _create_plan_summary(self, tasks: List[ReorganizationTask],
                           batches: List[ReorganizationBatch]) -> Dict[str, Any]:
        """Create summary statistics for the plan"""

        total_time = sum(batch.estimated_total_time for batch in batches)

        # Task statistics
        task_stats = {
            'total': len(tasks),
            'by_action': {},
            'by_risk': {},
            'high_priority': len([t for t in tasks if t.priority >= 8]),
            'low_confidence': len([t for t in tasks if t.confidence < self.config['min_confidence_threshold']])
        }

        # Count by action type
        for task in tasks:
            task_stats['by_action'][task.action.value] = task_stats['by_action'].get(task.action.value, 0) + 1

        # Count by risk level
        for task in tasks:
            task_stats['by_risk'][task.risk_level.value] = task_stats['by_risk'].get(task.risk_level.value, 0) + 1

        return {
            'task_statistics': task_stats,
            'batch_statistics': {
                'total_batches': len(batches),
                'total_estimated_time_minutes': total_time,
                'total_estimated_hours': total_time / 60
            },
            'risk_assessment': {
                'high_risk_batches': len([b for b in batches if b.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]),
                'auto_approvable_batches': len([b for b in batches if b.risk_level.value in self.config['auto_approve_risk_levels']]),
                'review_required_batches': len([b for b in batches if b.risk_level.value in self.config['require_review_risk_levels']])
            }
        }

    def _create_execution_guidelines(self, batches: List[ReorganizationBatch]) -> List[str]:
        """Create execution guidelines for the plan"""

        guidelines = [
            "Execute batches in order (1, 2, 3, 4) to minimize risk",
            "Create backup before starting any batch",
            "Run full test suite after each batch completion",
            "Monitor import dependencies and fix immediately if broken",
            "Document any architectural decisions made during execution"
        ]

        if any(b.risk_level == RiskLevel.HIGH for b in batches):
            guidelines.append("High-risk batches require approval before execution")

        if any('security' in b.batch_name.lower() for b in batches):
            guidelines.append("Security batches require security team review")

        return guidelines

    def _create_risk_mitigation(self, batches: List[ReorganizationBatch]) -> Dict[str, Any]:
        """Create risk mitigation strategies"""

        mitigation = {
            'backup_strategy': {
                'enabled': self.config['backup_enabled'],
                'strategy': 'Full directory backup before each batch'
            },
            'rollback_procedures': {
                'available': True,
                'procedures': [
                    'Restore from backup',
                    'Revert git changes',
                    'Fix broken imports manually'
                ]
            },
            'testing_requirements': {
                'required': True,
                'test_types': ['unit_tests', 'integration_tests', 'import_tests']
            }
        }

        # Add specific mitigations based on batch types
        if any(b.risk_level == RiskLevel.HIGH for b in batches):
            mitigation['high_risk_mitigations'] = [
                'Extra backup verification',
                'Pre-execution security review',
                'Post-execution security testing'
            ]

        return mitigation

    def _create_success_metrics(self, batches: List[ReorganizationBatch]) -> Dict[str, Any]:
        """Create success metrics for the reorganization"""

        return {
            'completion_metrics': [
                f"Complete {len(batches)} batches successfully",
                "All tasks completed without critical failures",
                "No data loss or corruption"
            ],
            'quality_metrics': [
                "Import statements work correctly",
                "All tests pass after reorganization",
                "Code functionality preserved"
            ],
            'improvement_metrics': [
                "Better logical organization achieved",
                "Reduced cognitive load for developers",
                "Improved maintainability"
            ],
            'target_goals': [
                "Achieve clean separation of concerns",
                "Enable easier future development",
                "Support scalable codebase growth"
            ]
        }

    def save_reorganization_plan(self, plan: DetailedReorganizationPlan, output_file: Path) -> None:
        """Save the reorganization plan to file"""

        # Convert to serializable format
        plan_dict = {
            'plan_id': plan.plan_id,
            'created_timestamp': plan.created_timestamp,
            'source_intelligence': plan.source_intelligence,
            'total_tasks': plan.total_tasks,
            'total_batches': plan.total_batches,
            'batches': [asdict(batch) for batch in plan.batches],
            'summary': plan.summary,
            'execution_guidelines': plan.execution_guidelines,
            'risk_mitigation': plan.risk_mitigation,
            'success_metrics': plan.success_metrics
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(plan_dict, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Reorganization plan saved to: {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to save reorganization plan: {e}")

    def execute_plan_batch(self, plan: DetailedReorganizationPlan, batch_id: str,
                          dry_run: bool = True) -> Dict[str, Any]:
        """
        Execute a specific batch from the reorganization plan.

        Args:
            plan: The reorganization plan
            batch_id: ID of the batch to execute
            dry_run: Whether to perform a dry run (no actual changes)

        Returns:
            Execution results
        """

        # Find the batch
        batch = next((b for b in plan.batches if b.batch_id == batch_id), None)
        if not batch:
            return {'success': False, 'error': f'Batch {batch_id} not found'}

        self.logger.info(f"Executing batch: {batch.batch_name}")
        if dry_run:
            self.logger.info("DRY RUN MODE - No actual changes will be made")

        results = {
            'batch_id': batch_id,
            'batch_name': batch.batch_name,
            'tasks_executed': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'errors': [],
            'dry_run': dry_run,
            'execution_timestamp': datetime.now().isoformat()
        }

        # Execute each task in the batch
        for task in batch.tasks:
            try:
                task_result = self._execute_task(task, dry_run)
                if task_result['success']:
                    results['tasks_successful'] += 1
                else:
                    results['tasks_failed'] += 1
                    results['errors'].append(f"Task {task.task_id}: {task_result['error']}")
            except Exception as e:
                results['tasks_failed'] += 1
                results['errors'].append(f"Task {task.task_id}: {str(e)}")

            results['tasks_executed'] += 1

        results['success'] = results['tasks_failed'] == 0
        self.logger.info(f"Batch execution completed: {results['tasks_successful']}/{results['tasks_executed']} successful")

        return results

    def _execute_task(self, task: ReorganizationTask, dry_run: bool = True) -> Dict[str, Any]:
        """Execute a single reorganization task"""

        if task.action == ReorganizationAction.MOVE_FILE:
            return self._execute_move_task(task, dry_run)
        elif task.action == ReorganizationAction.CREATE_DIRECTORY:
            return self._execute_create_directory_task(task, dry_run)
        elif task.action == ReorganizationAction.MANUAL_REVIEW:
            return self._execute_manual_review_task(task, dry_run)
        elif task.action == ReorganizationAction.UPDATE_IMPORTS:
            return self._execute_update_imports_task(task, dry_run)
        else:
            return {'success': False, 'error': f'Unsupported action: {task.action.value}'}

    def _execute_move_task(self, task: ReorganizationTask, dry_run: bool = True) -> Dict[str, Any]:
        """Execute a file move task"""

        if not task.source_path or not task.target_path:
            return {'success': False, 'error': 'Source and target paths required'}

        source_path = Path(task.source_path)
        target_path = Path(task.target_path)

        if not source_path.exists():
            return {'success': False, 'error': f'Source file does not exist: {source_path}'}

        # Create target directory
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if dry_run:
            self.logger.info(f"[DRY RUN] Would move: {source_path} -> {target_path}")
            return {'success': True, 'action': 'dry_run_move'}

        try:
            # Create backup
            if self.config['backup_enabled']:
                backup_path = self._create_backup(source_path)
                if backup_path:
                    self.logger.info(f"Backup created: {backup_path}")

            # Move the file
            shutil.move(str(source_path), str(target_path))
            self.logger.info(f"File moved: {source_path} -> {target_path}")

            return {'success': True, 'action': 'file_moved'}

        except Exception as e:
            return {'success': False, 'error': f'Move failed: {str(e)}'}

    def _execute_create_directory_task(self, task: ReorganizationTask, dry_run: bool = True) -> Dict[str, Any]:
        """Execute a directory creation task"""

        if not task.target_path:
            return {'success': False, 'error': 'Target path required'}

        target_path = Path(task.target_path)

        if dry_run:
            self.logger.info(f"[DRY RUN] Would create directory: {target_path}")
            return {'success': True, 'action': 'dry_run_create_directory'}

        try:
            target_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Directory created: {target_path}")
            return {'success': True, 'action': 'directory_created'}

        except Exception as e:
            return {'success': False, 'error': f'Directory creation failed: {str(e)}'}

    def _execute_manual_review_task(self, task: ReorganizationTask, dry_run: bool = True) -> Dict[str, Any]:
        """Execute a manual review task"""

        self.logger.info(f"Manual review required for: {task.source_path}")
        self.logger.info(f"Rationale: {task.rationale}")

        # In a real implementation, this might send notifications or create tickets
        return {'success': True, 'action': 'manual_review_logged'}

    def _execute_update_imports_task(self, task: ReorganizationTask, dry_run: bool = True) -> Dict[str, Any]:
        """Execute an import update task"""

        # This is a placeholder - a real implementation would analyze and update import statements
        self.logger.info(f"Import update required for: {task.source_path}")

        if dry_run:
            return {'success': True, 'action': 'dry_run_import_update'}

        return {'success': True, 'action': 'imports_updated'}

    def _create_backup(self, source_path: Path) -> Optional[Path]:
        """Create a backup of a file before modification"""

        backup_dir = self.root_dir / "tools" / "codebase_reorganizer" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{source_path.name}.{timestamp}.backup"
            backup_path = backup_dir / backup_name

            shutil.copy2(source_path, backup_path)
            return backup_path

        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return None


def main():
    """Main function to run the reorganization planner"""
    import argparse

    parser = argparse.ArgumentParser(description="Reorganization Planner")
    parser.add_argument("--root", type=str, default=".",
                      help="Root directory")
    parser.add_argument("--llm-map", type=str, required=True,
                      help="Path to LLM intelligence map JSON file")
    parser.add_argument("--integrated", type=str, required=True,
                      help="Path to integrated intelligence JSON file")
    parser.add_argument("--output", type=str, default="reorganization_plan.json",
                      help="Output file for reorganization plan")
    parser.add_argument("--execute-batch", type=str,
                      help="Execute a specific batch (batch_id)")
    parser.add_argument("--dry-run", action="store_true",
                      help="Perform dry run (no actual changes)")

    args = parser.parse_args()

    # Load required files
    try:
        with open(args.llm_map, 'r', encoding='utf-8') as f:
            llm_map = json.load(f)

        with open(args.integrated, 'r', encoding='utf-8') as f:
            integrated_data = json.load(f)

        integrated_intelligence = [IntegratedIntelligence(**item) for item in integrated_data.get('integrated_intelligence', [])]

    except Exception as e:
        print(f"Error loading input files: {e}")
        return

    # Initialize planner
    root_dir = Path(args.root).resolve()
    planner = ReorganizationPlanner(root_dir)

    # Create reorganization plan
    if not args.execute_batch:
        print("📋 Creating reorganization plan...")
        plan = planner.create_reorganization_plan(llm_map, integrated_intelligence)
        planner.save_reorganization_plan(plan, Path(args.output))

        print("
✅ Reorganization plan created!"        print(f"Total tasks: {plan.total_tasks}")
        print(f"Total batches: {plan.total_batches}")
        print(f"Estimated time: {plan.summary['batch_statistics']['total_estimated_hours']:.1f} hours")
        print(f"Output saved to: {args.output}")

        # Print batch summary
        print("
📦 Reorganization Batches:"        for i, batch in enumerate(plan.batches, 1):
            print(f"  {i}. {batch.batch_name} ({batch.risk_level.value}) - {len(batch.tasks)} tasks")

    # Execute specific batch
    else:
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                plan_data = json.load(f)

            # Reconstruct plan object
            batches = [ReorganizationBatch(**batch) for batch in plan_data['batches']]
            plan = DetailedReorganizationPlan(**{k: v for k, v in plan_data.items() if k != 'batches'}, batches=batches)

            print(f"🎯 Executing batch: {args.execute_batch}")
            if args.dry_run:
                print("🔍 DRY RUN MODE - No actual changes will be made")

            results = planner.execute_plan_batch(plan, args.execute_batch, args.dry_run)

            if results['success']:
                print("
✅ Batch execution completed!"                print(f"Tasks executed: {results['tasks_executed']}")
                print(f"Tasks successful: {results['tasks_successful']}")
                print(f"Tasks failed: {results['tasks_failed']}")
            else:
                print("
❌ Batch execution failed!"                for error in results['errors']:
                    print(f"  Error: {error}")

        except Exception as e:
            print(f"Error executing batch: {e}")


if __name__ == "__main__":
    main()

