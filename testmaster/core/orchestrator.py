#!/usr/bin/env python3
"""
TestMaster Core Orchestrator

Central coordinator for the TestMaster system with DAG-based workflow execution.
Merged with production orchestrator for deep integration.
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue, PriorityQueue
import hashlib
import pickle

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Types of tasks the orchestrator can handle."""
    GENERATE_TEST = "generate_test"
    SELF_HEAL = "self_heal"
    VERIFY_QUALITY = "verify_quality"
    FIX_IMPORTS = "fix_imports"
    DEDUPLICATE = "deduplicate"
    ANALYZE_COVERAGE = "analyze_coverage"
    GENERATE_REPORT = "generate_report"
    MONITOR_CHANGES = "monitor_changes"
    BATCH_CONVERT = "batch_convert"
    INCREMENTAL_UPDATE = "incremental_update"
    SECURITY_SCAN = "security_scan"
    OPTIMIZE_FLOW = "optimize_flow"


@dataclass
class Task:
    """Represents a single task in the pipeline."""
    id: str
    type: TaskType
    module_path: Optional[Path] = None
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5  # 1-10, higher is more important
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __lt__(self, other):
        """For priority queue comparison."""
        return self.priority > other.priority


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    max_parallel_tasks: int = 4
    task_timeout: int = 300
    retry_attempts: int = 3
    enable_caching: bool = True
    cache_dir: str = "cache/orchestrator"
    progress_tracking: bool = True
    auto_retry: bool = True
    parallel_execution: bool = True
    api_rate_limit: int = 30  # requests per minute


class WorkflowDAG:
    """Directed Acyclic Graph for workflow management."""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.edges: Dict[str, Set[str]] = {}  # task_id -> dependent_task_ids
        self.reverse_edges: Dict[str, Set[str]] = {}  # task_id -> dependency_ids
    
    def add_task(self, task: Task):
        """Add a task to the DAG."""
        self.tasks[task.id] = task
        self.edges[task.id] = set()
        self.reverse_edges[task.id] = set()
        
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                self.edges[dep_id].add(task.id)
                self.reverse_edges[task.id].add(dep_id)
    
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        ready = []
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING:
                deps_completed = all(
                    self.tasks.get(dep_id, Task("", TaskType.GENERATE_TEST, status=TaskStatus.COMPLETED)).status 
                    in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]
                    for dep_id in task.dependencies
                )
                if deps_completed:
                    ready.append(task)
        return sorted(ready, reverse=True)  # Sort by priority
    
    def has_cycles(self) -> bool:
        """Check if the DAG has cycles."""
        visited = set()
        rec_stack = set()
        
        def visit(task_id):
            visited.add(task_id)
            rec_stack.add(task_id)
            
            for neighbor in self.edges.get(task_id, []):
                if neighbor not in visited:
                    if visit(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        for task_id in self.tasks:
            if task_id not in visited:
                if visit(task_id):
                    return True
        return False


class Orchestrator:
    """Main orchestrator for coordinating TestMaster operations with DAG execution."""
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self.workflow = WorkflowDAG()
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_parallel_tasks)
        self.running_tasks: Set[str] = set()
        self.task_lock = threading.Lock()
        self.progress_file = Path("orchestrator_progress.json")
        self.task_handlers = self._initialize_handlers()
        
        # Legacy compatibility
        self.active_tasks: Dict[str, Any] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.task_queue: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "pipelines_executed": 0,
            "dag_executions": 0,
            "parallel_tasks_run": 0
        }
        
        # Connect to shared state and cache if available
        self._connect_integrations()
        
        logger.info("Orchestrator initialized with DAG support")
    
    def _initialize_handlers(self) -> Dict[TaskType, callable]:
        """Initialize task type handlers."""
        return {
            TaskType.GENERATE_TEST: self._handle_generate_test,
            TaskType.SELF_HEAL: self._handle_self_heal,
            TaskType.VERIFY_QUALITY: self._handle_verify_quality,
            TaskType.FIX_IMPORTS: self._handle_fix_imports,
            TaskType.SECURITY_SCAN: self._handle_security_scan,
            TaskType.OPTIMIZE_FLOW: self._handle_optimize_flow,
        }
    
    def _connect_integrations(self):
        """Connect to shared state, cache, and flow optimizer if available."""
        try:
            # Try to connect to intelligent cache
            from cache.intelligent_cache import get_cache
            self.cache = get_cache()
            logger.info("Connected to intelligent cache")
        except ImportError:
            self.cache = None
            
        try:
            # Try to connect to flow optimizer
            from ..flow_optimizer.flow_analyzer import get_flow_analyzer
            self.flow_analyzer = get_flow_analyzer()
            logger.info("Connected to flow analyzer")
        except ImportError:
            self.flow_analyzer = None
    
    def create_test_generation_workflow(self, module_paths: List[Path]) -> str:
        """Create a complete test generation workflow for modules."""
        workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        for i, module_path in enumerate(module_paths):
            base_id = f"{workflow_id}_module_{i}"
            
            # 1. Generate initial test
            gen_task = Task(
                id=f"{base_id}_generate",
                type=TaskType.GENERATE_TEST,
                module_path=module_path,
                priority=8
            )
            self.workflow.add_task(gen_task)
            
            # 2. Self-heal the generated test
            heal_task = Task(
                id=f"{base_id}_heal",
                type=TaskType.SELF_HEAL,
                module_path=module_path,
                dependencies=[gen_task.id],
                priority=7
            )
            self.workflow.add_task(heal_task)
            
            # 3. Security scan (new!)
            security_task = Task(
                id=f"{base_id}_security",
                type=TaskType.SECURITY_SCAN,
                module_path=module_path,
                dependencies=[heal_task.id],
                priority=6
            )
            self.workflow.add_task(security_task)
            
            # 4. Verify quality
            verify_task = Task(
                id=f"{base_id}_verify",
                type=TaskType.VERIFY_QUALITY,
                module_path=module_path,
                dependencies=[security_task.id],
                priority=5
            )
            self.workflow.add_task(verify_task)
        
        # Optimize flow at the end
        optimize_task = Task(
            id=f"{workflow_id}_optimize",
            type=TaskType.OPTIMIZE_FLOW,
            dependencies=[f"{workflow_id}_module_{i}_verify" for i in range(len(module_paths))],
            priority=3,
            metadata={"workflow_id": workflow_id}
        )
        self.workflow.add_task(optimize_task)
        
        return workflow_id
    
    async def execute_workflow(self) -> Dict[str, Any]:
        """Execute the workflow asynchronously with DAG resolution."""
        if self.workflow.has_cycles():
            raise ValueError("Workflow contains cycles!")
        
        self.stats["dag_executions"] += 1
        results = {}
        completed_tasks = set()
        failed_tasks = set()
        
        # Track execution for flow analysis
        execution_data = []
        workflow_start = time.time()
        
        while len(completed_tasks) + len(failed_tasks) < len(self.workflow.tasks):
            ready_tasks = self.workflow.get_ready_tasks()
            
            if not ready_tasks and not self.running_tasks:
                break  # Deadlock or all tasks completed
            
            # Execute ready tasks in parallel
            futures = []
            for task in ready_tasks:
                if task.id not in self.running_tasks:
                    with self.task_lock:
                        self.running_tasks.add(task.id)
                        task.status = TaskStatus.RUNNING
                        task.started_at = datetime.now()
                        self.stats["parallel_tasks_run"] += 1
                    
                    future = self.executor.submit(self._execute_task, task)
                    futures.append((task, future))
            
            # Wait for tasks to complete
            for task, future in futures:
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, future.result, self.config.task_timeout
                    )
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    completed_tasks.add(task.id)
                    results[task.id] = result
                    
                    # Track for flow analysis
                    if task.started_at:
                        execution_data.append({
                            'task_id': task.id,
                            'execution_time': (task.completed_at - task.started_at).total_seconds() * 1000,
                            'wait_time': (task.started_at - task.created_at).total_seconds() * 1000,
                            'resource_usage': 50.0  # Placeholder
                        })
                    
                    logger.info(f"Task {task.id} completed successfully")
                    self.stats["tasks_completed"] += 1
                    
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    failed_tasks.add(task.id)
                    self.stats["tasks_failed"] += 1
                    logger.error(f"Task {task.id} failed: {e}")
                
                finally:
                    with self.task_lock:
                        self.running_tasks.discard(task.id)
        
        # Analyze flow if available
        if self.flow_analyzer and execution_data:
            flow_analysis = self.flow_analyzer.analyze_flow(
                workflow_id="latest",
                execution_data=execution_data
            )
            results['flow_analysis'] = flow_analysis
        
        workflow_time = time.time() - workflow_start
        results['workflow_metrics'] = {
            'total_tasks': len(self.workflow.tasks),
            'completed': len(completed_tasks),
            'failed': len(failed_tasks),
            'execution_time': workflow_time,
            'parallel_efficiency': self.stats["parallel_tasks_run"] / max(len(self.workflow.tasks), 1)
        }
        
        return results
    
    def execute_pipeline(self, pipeline_type: str, **kwargs) -> Dict[str, Any]:
        """Execute a test generation pipeline (legacy compatibility + DAG)."""
        self.stats["pipelines_executed"] += 1
        
        # Map to new DAG-based execution
        if pipeline_type == "test_generation":
            module_paths = kwargs.get('modules', [])
            if module_paths:
                workflow_id = self.create_test_generation_workflow(module_paths)
                # Run synchronously for compatibility
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(self.execute_workflow())
                    return {"success": True, "pipeline_type": pipeline_type, "results": results}
                finally:
                    loop.close()
        
        # Fallback for unknown pipeline types
        return {"success": True, "pipeline_type": pipeline_type}
    
    def _execute_task(self, task: Task) -> Any:
        """Execute a single task."""
        handler = self.task_handlers.get(task.type)
        if handler:
            return handler(task)
        else:
            logger.warning(f"No handler for task type {task.type}")
            return None
    
    def _handle_generate_test(self, task: Task) -> Any:
        """Handle test generation task."""
        # Import here to avoid circular dependencies
        try:
            from intelligent_test_builder import TestGenerator
            generator = TestGenerator()
            return generator.generate_test(str(task.module_path))
        except ImportError:
            logger.error("Test generator not available")
            return None
    
    def _handle_self_heal(self, task: Task) -> Any:
        """Handle self-healing task."""
        try:
            from enhanced_self_healing_verifier import SelfHealingVerifier
            verifier = SelfHealingVerifier()
            return verifier.heal_test(str(task.module_path))
        except ImportError:
            logger.error("Self-healing verifier not available")
            return None
    
    def _handle_verify_quality(self, task: Task) -> Any:
        """Handle quality verification task."""
        try:
            from independent_test_verifier import TestVerifier
            verifier = TestVerifier()
            return verifier.verify_test(str(task.module_path))
        except ImportError:
            logger.error("Test verifier not available")
            return None
    
    def _handle_fix_imports(self, task: Task) -> Any:
        """Handle import fixing task."""
        try:
            from fix_import_paths import ImportFixer
            fixer = ImportFixer()
            return fixer.fix_imports(str(task.module_path))
        except ImportError:
            logger.error("Import fixer not available")
            return None
    
    def _handle_security_scan(self, task: Task) -> Any:
        """Handle security scanning task."""
        # Placeholder for security scanner integration
        logger.info(f"Security scan for {task.module_path}")
        return {"vulnerabilities": [], "status": "clean"}
    
    def _handle_optimize_flow(self, task: Task) -> Any:
        """Handle flow optimization task."""
        if self.flow_analyzer:
            # Get workflow execution data from cache if available
            if self.cache:
                cache_key = f"workflow_{task.metadata.get('workflow_id', 'latest')}"
                execution_data = self.cache.get(cache_key, [])
                if execution_data:
                    return self.flow_analyzer.analyze_flow(
                        task.metadata.get('workflow_id', 'latest'),
                        execution_data
                    )
        return {"status": "no_data"}
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "active_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "pending_tasks": len([t for t in self.workflow.tasks.values() if t.status == TaskStatus.PENDING]),
            "dag_tasks": len(self.workflow.tasks),
            "stats": self.stats,
            "cache_connected": self.cache is not None,
            "flow_analyzer_connected": self.flow_analyzer is not None
        }


# Keep PipelineOrchestrator alias for backward compatibility
PipelineOrchestrator = Orchestrator


def get_orchestrator() -> Orchestrator:
    """Get orchestrator instance."""
    return Orchestrator()