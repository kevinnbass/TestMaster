#!/usr/bin/env python3
"""
Reorganization Data Classes
==========================

Data structures for reorganization planning and execution.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


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


@dataclass
class ReorganizationMetrics:
    """Metrics for reorganization planning and execution"""
    total_files_analyzed: int
    total_directories_analyzed: int
    high_confidence_actions: int
    medium_confidence_actions: int
    low_confidence_actions: int
    estimated_total_time: int
    risk_distribution: Dict[str, int]
    action_distribution: Dict[str, int]


@dataclass
class ExecutionResult:
    """Result of executing a reorganization task"""
    task_id: str
    success: bool
    execution_time: float
    error_message: str
    output_files: List[str]
    verification_results: Dict[str, bool]


@dataclass
class BatchExecutionResult:
    """Result of executing a reorganization batch"""
    batch_id: str
    success: bool
    total_execution_time: float
    successful_tasks: int
    failed_tasks: int
    task_results: List[ExecutionResult]
    rollback_actions: List[str]

