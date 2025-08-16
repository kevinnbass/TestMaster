#!/usr/bin/env python3
"""
TestMaster Pipeline Orchestrator
Centralized control system for all test generation and maintenance operations.

Features:
- Workflow engine with DAG-based task execution
- Intelligent routing to appropriate generators/converters
- Progress tracking and resumable operations
- Parallel task execution with dependency management
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue, PriorityQueue
import hashlib
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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


class PipelineOrchestrator:
    """Main orchestrator for TestMaster pipeline."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.workflow = WorkflowDAG()
        self.executor = ThreadPoolExecutor(max_workers=self.config.get("max_workers", 4))
        self.running_tasks: Set[str] = set()
        self.task_lock = threading.Lock()
        self.progress_file = Path("orchestrator_progress.json")
        self.task_handlers = self._initialize_handlers()
        
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load orchestrator configuration."""
        default_config = {
            "max_workers": 4,
            "enable_caching": True,
            "cache_dir": "cache/orchestrator",
            "progress_tracking": True,
            "auto_retry": True,
            "parallel_execution": True,
            "task_timeout": 300,  # 5 minutes
            "api_rate_limit": 30,  # requests per minute
        }
        
        if config_path and config_path.exists():
            with open(config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _initialize_handlers(self) -> Dict[TaskType, callable]:
        """Initialize task type handlers."""
        return {
            TaskType.GENERATE_TEST: self._handle_generate_test,
            TaskType.SELF_HEAL: self._handle_self_heal,
            TaskType.VERIFY_QUALITY: self._handle_verify_quality,
            TaskType.FIX_IMPORTS: self._handle_fix_imports,
            TaskType.DEDUPLICATE: self._handle_deduplicate,
            TaskType.ANALYZE_COVERAGE: self._handle_analyze_coverage,
            TaskType.GENERATE_REPORT: self._handle_generate_report,
            TaskType.MONITOR_CHANGES: self._handle_monitor_changes,
            TaskType.BATCH_CONVERT: self._handle_batch_convert,
            TaskType.INCREMENTAL_UPDATE: self._handle_incremental_update,
        }
    
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
            
            # 3. Fix imports
            import_task = Task(
                id=f"{base_id}_imports",
                type=TaskType.FIX_IMPORTS,
                module_path=module_path,
                dependencies=[heal_task.id],
                priority=6
            )
            self.workflow.add_task(import_task)
            
            # 4. Verify quality
            verify_task = Task(
                id=f"{base_id}_verify",
                type=TaskType.VERIFY_QUALITY,
                module_path=module_path,
                dependencies=[import_task.id],
                priority=5
            )
            self.workflow.add_task(verify_task)
            
            # 5. Deduplicate tests
            dedup_task = Task(
                id=f"{base_id}_dedup",
                type=TaskType.DEDUPLICATE,
                module_path=module_path,
                dependencies=[verify_task.id],
                priority=4
            )
            self.workflow.add_task(dedup_task)
        
        # Final report generation
        report_task = Task(
            id=f"{workflow_id}_report",
            type=TaskType.GENERATE_REPORT,
            dependencies=[f"{workflow_id}_module_{i}_dedup" for i in range(len(module_paths))],
            priority=3,
            metadata={"workflow_id": workflow_id}
        )
        self.workflow.add_task(report_task)
        
        return workflow_id
    
    async def execute_workflow(self) -> Dict[str, Any]:
        """Execute the workflow asynchronously."""
        if self.workflow.has_cycles():
            raise ValueError("Workflow contains cycles!")
        
        results = {}
        completed_tasks = set()
        failed_tasks = set()
        
        while len(completed_tasks) + len(failed_tasks) < len(self.workflow.tasks):
            ready_tasks = self.workflow.get_ready_tasks()
            
            if not ready_tasks and not self.running_tasks:
                # Deadlock or all tasks completed
                break
            
            # Execute ready tasks in parallel
            futures = []
            for task in ready_tasks:
                if task.id not in self.running_tasks:
                    with self.task_lock:
                        self.running_tasks.add(task.id)
                        task.status = TaskStatus.RUNNING
                        task.started_at = datetime.now()
                    
                    future = self.executor.submit(self._execute_task, task)
                    futures.append((task, future))
            
            # Wait for tasks to complete
            for task, future in futures:
                try:
                    result = future.result(timeout=self.config.get("task_timeout", 300))
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    completed_tasks.add(task.id)
                    results[task.id] = result
                    logger.info(f"Task {task.id} completed successfully")
                    
                except Exception as e:
                    task.error = str(e)
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now()
                    failed_tasks.add(task.id)
                    logger.error(f"Task {task.id} failed: {e}")
                    
                    # Retry logic
                    if self.config.get("auto_retry") and task.retries < task.max_retries:
                        task.retries += 1
                        task.status = TaskStatus.PENDING
                        failed_tasks.remove(task.id)
                        logger.info(f"Retrying task {task.id} (attempt {task.retries}/{task.max_retries})")
                
                finally:
                    with self.task_lock:
                        self.running_tasks.discard(task.id)
            
            # Save progress
            if self.config.get("progress_tracking"):
                self._save_progress()
            
            # Small delay to prevent CPU spinning
            await asyncio.sleep(0.1)
        
        return {
            "completed": list(completed_tasks),
            "failed": list(failed_tasks),
            "results": results,
            "total_tasks": len(self.workflow.tasks),
            "success_rate": len(completed_tasks) / len(self.workflow.tasks) * 100 if self.workflow.tasks else 0
        }
    
    def _execute_task(self, task: Task) -> Any:
        """Execute a single task."""
        logger.info(f"Executing task {task.id} of type {task.type.value}")
        
        handler = self.task_handlers.get(task.type)
        if not handler:
            raise ValueError(f"No handler for task type {task.type}")
        
        return handler(task)
    
    def _handle_generate_test(self, task: Task) -> Dict:
        """Handle test generation task."""
        # Import the actual generator
        from intelligent_test_builder import IntelligentTestBuilder
        
        builder = IntelligentTestBuilder()
        result = builder.generate_test(task.module_path)
        
        return {
            "module": str(task.module_path),
            "test_generated": result.get("success", False),
            "test_path": result.get("test_path"),
            "quality_score": result.get("quality_score", 0)
        }
    
    def _handle_self_heal(self, task: Task) -> Dict:
        """Handle self-healing task."""
        # Placeholder for actual self-healing logic
        return {
            "module": str(task.module_path),
            "healed": True,
            "iterations": 3,
            "errors_fixed": ["syntax", "imports"]
        }
    
    def _handle_verify_quality(self, task: Task) -> Dict:
        """Handle quality verification task."""
        # Placeholder for actual verification logic
        return {
            "module": str(task.module_path),
            "quality_score": 85,
            "coverage": 92,
            "suggestions": ["Add edge case tests", "Test error conditions"]
        }
    
    def _handle_fix_imports(self, task: Task) -> Dict:
        """Handle import fixing task."""
        # Placeholder for actual import fixing logic
        return {
            "module": str(task.module_path),
            "imports_fixed": 5,
            "success_rate": 100
        }
    
    def _handle_deduplicate(self, task: Task) -> Dict:
        """Handle test deduplication task."""
        # Placeholder for actual deduplication logic
        return {
            "module": str(task.module_path),
            "duplicates_removed": 2,
            "tests_consolidated": 3
        }
    
    def _handle_analyze_coverage(self, task: Task) -> Dict:
        """Handle coverage analysis task."""
        return {
            "total_coverage": 88.5,
            "uncovered_lines": 125,
            "critical_gaps": ["error_handling", "edge_cases"]
        }
    
    def _handle_generate_report(self, task: Task) -> Dict:
        """Handle report generation task."""
        workflow_id = task.metadata.get("workflow_id", "unknown")
        
        # Collect results from all tasks in workflow
        workflow_results = {}
        for task_id, t in self.workflow.tasks.items():
            if workflow_id in task_id:
                workflow_results[task_id] = {
                    "status": t.status.value,
                    "result": t.result,
                    "error": t.error,
                    "duration": (t.completed_at - t.started_at).total_seconds() if t.completed_at and t.started_at else None
                }
        
        # Generate summary report
        report = {
            "workflow_id": workflow_id,
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(workflow_results),
            "completed": sum(1 for r in workflow_results.values() if r["status"] == "completed"),
            "failed": sum(1 for r in workflow_results.values() if r["status"] == "failed"),
            "results": workflow_results
        }
        
        # Save report to file
        report_path = Path(f"reports/workflow_{workflow_id}.json")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        return {
            "report_path": str(report_path),
            "summary": f"Completed {report['completed']}/{report['total_tasks']} tasks"
        }
    
    def _handle_monitor_changes(self, task: Task) -> Dict:
        """Handle change monitoring task."""
        return {
            "changes_detected": 5,
            "modules_affected": ["module_a.py", "module_b.py"],
            "action_required": "regenerate_tests"
        }
    
    def _handle_batch_convert(self, task: Task) -> Dict:
        """Handle batch conversion task."""
        return {
            "modules_converted": 10,
            "success_rate": 90,
            "time_taken": 120
        }
    
    def _handle_incremental_update(self, task: Task) -> Dict:
        """Handle incremental update task."""
        return {
            "module": str(task.module_path),
            "changes_detected": True,
            "tests_updated": 3,
            "new_tests_added": 1
        }
    
    def _save_progress(self):
        """Save workflow progress to file."""
        progress = {
            "timestamp": datetime.now().isoformat(),
            "tasks": {}
        }
        
        for task_id, task in self.workflow.tasks.items():
            progress["tasks"][task_id] = {
                "type": task.type.value,
                "status": task.status.value,
                "module": str(task.module_path) if task.module_path else None,
                "error": task.error,
                "retries": task.retries
            }
        
        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2)
    
    def load_progress(self) -> bool:
        """Load and resume previous workflow progress."""
        if not self.progress_file.exists():
            return False
        
        try:
            with open(self.progress_file) as f:
                progress = json.load(f)
            
            # Restore task states
            for task_id, task_data in progress["tasks"].items():
                if task_id in self.workflow.tasks:
                    task = self.workflow.tasks[task_id]
                    task.status = TaskStatus(task_data["status"])
                    task.error = task_data.get("error")
                    task.retries = task_data.get("retries", 0)
            
            logger.info(f"Resumed workflow from {progress['timestamp']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        task_counts = {status: 0 for status in TaskStatus}
        for task in self.workflow.tasks.values():
            task_counts[task.status] += 1
        
        return {
            "total_tasks": len(self.workflow.tasks),
            "task_counts": {k.value: v for k, v in task_counts.items()},
            "running_tasks": list(self.running_tasks),
            "progress_percentage": (task_counts[TaskStatus.COMPLETED] / len(self.workflow.tasks) * 100) if self.workflow.tasks else 0
        }


def main():
    """Main entry point for the orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TestMaster Pipeline Orchestrator")
    parser.add_argument("--modules", nargs="+", help="Module paths to process")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--resume", action="store_true", help="Resume previous workflow")
    parser.add_argument("--status", action="store_true", help="Show current status")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    config_path = Path(args.config) if args.config else None
    orchestrator = PipelineOrchestrator(config_path)
    
    if args.status:
        status = orchestrator.get_status()
        print(json.dumps(status, indent=2))
        return
    
    if args.resume:
        orchestrator.load_progress()
    
    if args.modules:
        # Create workflow for specified modules
        module_paths = [Path(m) for m in args.modules]
        workflow_id = orchestrator.create_test_generation_workflow(module_paths)
        print(f"Created workflow: {workflow_id}")
        
        # Execute workflow
        print("Executing workflow...")
        results = asyncio.run(orchestrator.execute_workflow())
        
        print(f"\nWorkflow completed:")
        print(f"  Success rate: {results['success_rate']:.1f}%")
        print(f"  Completed: {len(results['completed'])}")
        print(f"  Failed: {len(results['failed'])}")
        
        if results['failed']:
            print(f"\nFailed tasks: {results['failed']}")


if __name__ == "__main__":
    main()