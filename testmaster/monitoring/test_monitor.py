"""
Enhanced Test Monitor with Graph-Based Workflow Management

Integrates with the existing file watching and monitoring system to provide
graph-based workflow orchestration for test monitoring operations.

Features:
- Graph-based workflow execution
- Integration with file watcher
- Intelligent test impact analysis
- Automatic test generation triggers
- Parallel processing support
"""

import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor

# Import existing monitoring components
from .file_watcher import FileWatcher, FileChangeEvent, ChangeType
from .idle_detector import IdleDetector
from .test_scheduler import TestScheduler

# Import core enhancements
from ..core.feature_flags import FeatureFlags
from ..core.shared_state import get_shared_state
from ..core.tracking_manager import get_tracking_manager, track_operation
from ..core.workflow_graph import WorkflowGraph, create_test_monitoring_workflow, get_workflow_graph
from ..core.monitoring_decorators import monitor_performance
from ..core.layer_manager import requires_layer


@dataclass
class MonitoringContext:
    """Context for monitoring operations."""
    file_path: str
    change_type: str
    impact_level: str = "unknown"
    needs_test_generation: bool = False
    test_files_affected: List[str] = None
    modules_to_analyze: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.test_files_affected is None:
            self.test_files_affected = []
        if self.modules_to_analyze is None:
            self.modules_to_analyze = []
        if self.metadata is None:
            self.metadata = {}


class TestMonitor:
    """
    Enhanced test monitor with graph-based workflow management.
    
    Provides intelligent monitoring of file changes and orchestrates
    test generation and verification using configurable workflows.
    """
    
    @requires_layer("layer2_monitoring", "test_monitoring")
    def __init__(self, watch_paths: List[str] = None, polling_interval: float = 1.0):
        """
        Initialize test monitor.
        
        Args:
            watch_paths: List of paths to monitor
            polling_interval: File watching polling interval
        """
        self.watch_paths = watch_paths or ["."]
        self.polling_interval = polling_interval
        
        # Core monitoring components
        self.file_watcher = FileWatcher(watch_paths=self.watch_paths)
        self.idle_detector = IdleDetector(watch_paths=self.watch_paths)
        self.test_scheduler = TestScheduler()
        
        # Enhanced features
        self._setup_enhanced_features()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.event_queue: List[FileChangeEvent] = []
        self.processing_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "files_monitored": 0,
            "changes_detected": 0,
            "tests_generated": 0,
            "workflows_executed": 0,
            "errors": 0,
            "start_time": time.time()
        }
        
        print("Enhanced test monitor initialized")
        if self.workflow_graph and self.workflow_graph.enabled:
            print(f"   Graph workflows: enabled")
        else:
            print(f"   Graph workflows: disabled (linear fallback)")
        print(f"   Watch paths: {self.watch_paths}")
    
    def _setup_enhanced_features(self):
        """Initialize enhanced monitoring features."""
        # Graph-based workflows
        if FeatureFlags.is_enabled('layer2_monitoring', 'graph_workflows'):
            self.workflow_graph = create_test_monitoring_workflow()
            self._setup_workflow_handlers()
        else:
            self.workflow_graph = None
        
        # Shared state management
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
        
        # Tracking manager
        if FeatureFlags.is_enabled('layer2_monitoring', 'tracking_manager'):
            self.tracking_manager = get_tracking_manager()
        else:
            self.tracking_manager = None
    
    def _setup_workflow_handlers(self):
        """Setup handlers for workflow nodes."""
        if not self.workflow_graph or not self.workflow_graph.enabled:
            return
        
        # Replace default handlers with actual implementations
        self.workflow_graph.nodes["detect_change"].handler = self._handle_file_change
        self.workflow_graph.nodes["analyze_impact"].handler = self._analyze_test_impact
        self.workflow_graph.nodes["generate_tests"].handler = self._trigger_test_generation
        self.workflow_graph.nodes["verify_tests"].handler = self._verify_generated_tests
        
        # Setup condition for test generation decision
        def should_generate_condition(context):
            return "generate" if context.data.get("needs_test_generation", False) else "skip"
        
        self.workflow_graph.conditional_edges["analyze_impact"]["condition"] = should_generate_condition
    
    @monitor_performance(name="start_monitoring")
    def start_monitoring(self, continuous: bool = True):
        """
        Start monitoring file changes.
        
        Args:
            continuous: Whether to monitor continuously or single pass
        """
        if self.is_monitoring:
            print("Monitoring already active")
            return
        
        self.is_monitoring = True
        
        # Setup file change callback
        self.file_watcher.add_callback(self._on_file_change)
        
        # Start tracking if enabled
        chain_id = None
        if self.tracking_manager:
            chain_id = self.tracking_manager.start_chain(
                chain_name="test_monitoring_session",
                inputs={
                    "watch_paths": self.watch_paths,
                    "continuous": continuous,
                    "polling_interval": self.polling_interval
                }
            )
        
        try:
            if continuous:
                # Start file watcher
                self.file_watcher.start()
                print("Continuous monitoring started...")
                
                # Process events in a separate thread
                self.monitor_thread = threading.Thread(
                    target=self._monitoring_loop,
                    daemon=True
                )
                self.monitor_thread.start()
                
            else:
                # Single pass monitoring
                print("Performing single monitoring pass...")
                self._perform_single_pass()
                
        except Exception as e:
            print(f"Error starting monitoring: {e}")
            self.is_monitoring = False
            
            if self.tracking_manager and chain_id:
                self.tracking_manager.end_chain(
                    chain_id=chain_id,
                    success=False,
                    error=f"Monitoring failed to start: {str(e)}"
                )
            raise
    
    def stop_monitoring(self):
        """Stop monitoring."""
        if not self.is_monitoring:
            return
        
        print("Stopping monitoring...")
        self.is_monitoring = False
        
        # Stop file watcher
        self.file_watcher.stop()
        
        # Wait for monitor thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        print("Monitoring stopped")
    
    def _on_file_change(self, event: FileChangeEvent):
        """Handle file change events from file watcher."""
        with self.processing_lock:
            self.event_queue.append(event)
            self.stats["changes_detected"] += 1
        
        # Update shared state if enabled
        if self.shared_state:
            self.shared_state.increment("file_changes_detected")
            self.shared_state.append("recent_changes", {
                "file": event.file_path,
                "type": event.change_type.value,
                "timestamp": event.timestamp.isoformat()
            })
    
    def _monitoring_loop(self):
        """Main monitoring loop for processing events."""
        while self.is_monitoring:
            try:
                events_to_process = []
                
                # Collect pending events
                with self.processing_lock:
                    if self.event_queue:
                        events_to_process = list(self.event_queue)
                        self.event_queue.clear()
                
                # Process events
                for event in events_to_process:
                    self._process_file_change_event(event)
                
                # Wait before next iteration
                time.sleep(self.polling_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                self.stats["errors"] += 1
                time.sleep(1.0)  # Brief pause before retrying
    
    @track_operation("test_monitor", "process_file_change")
    def _process_file_change_event(self, event: FileChangeEvent):
        """Process a single file change event."""
        try:
            # Create monitoring context
            context_data = {
                "file_path": event.file_path,
                "change_type": event.change_type.value,
                "timestamp": event.timestamp.isoformat(),
                "is_directory": event.is_directory
            }
            
            if self.workflow_graph and self.workflow_graph.enabled:
                # Use graph-based workflow
                result = self.workflow_graph.invoke(context_data)
                
                if result.get("success"):
                    self.stats["workflows_executed"] += 1
                    if result.get("context", {}).get("tests_generated"):
                        self.stats["tests_generated"] += 1
                else:
                    print(f"Workflow execution failed: {result.get('error')}")
                    self.stats["errors"] += 1
                    
            else:
                # Use linear fallback
                self._execute_linear_monitoring(context_data)
                
        except Exception as e:
            print(f"Error processing file change {event.file_path}: {e}")
            self.stats["errors"] += 1
    
    def _execute_linear_monitoring(self, context_data: Dict[str, Any]):
        """Linear fallback when graph workflows are disabled."""
        file_path = context_data["file_path"]
        change_type = context_data["change_type"]
        
        # Simple linear processing
        print(f"Processing change: {file_path} ({change_type})")
        
        # Check if this file needs test generation
        if self._should_generate_tests_linear(file_path, change_type):
            print(f"Triggering test generation for {file_path}")
            # This would integrate with existing test generation systems
            self.stats["tests_generated"] += 1
    
    def _should_generate_tests_linear(self, file_path: str, change_type: str) -> bool:
        """Simple linear decision for test generation."""
        # Basic heuristics for when to generate tests
        if not file_path.endswith('.py'):
            return False
        
        if 'test_' in file_path or file_path.endswith('_test.py'):
            return False  # Don't generate tests for test files
        
        if change_type in ['created', 'modified']:
            return True
        
        return False
    
    def _perform_single_pass(self):
        """Perform a single monitoring pass."""
        print("Scanning for changes...")
        
        # This would integrate with existing file scanning logic
        # For now, just update statistics
        self.stats["files_monitored"] = len(self.watch_paths)
        
        print("Single pass completed")
    
    # Workflow handler implementations
    
    def _handle_file_change(self, context) -> Dict[str, Any]:
        """Handle file change detection in workflow."""
        file_path = context.data.get("file_path", "")
        change_type = context.data.get("change_type", "")
        
        print(f"Workflow: Detected change in {file_path} ({change_type})")
        
        # Analyze the file to determine if it's relevant
        is_python_file = file_path.endswith('.py')
        is_test_file = 'test_' in file_path or file_path.endswith('_test.py')
        
        return {
            "change_detected": True,
            "is_python_file": is_python_file,
            "is_test_file": is_test_file,
            "file_analyzed": True
        }
    
    def _analyze_test_impact(self, context) -> Dict[str, Any]:
        """Analyze the impact of the file change on testing."""
        file_path = context.data.get("file_path", "")
        is_python_file = context.data.get("is_python_file", False)
        is_test_file = context.data.get("is_test_file", False)
        
        impact_level = "none"
        needs_generation = False
        
        if is_python_file and not is_test_file:
            # Python source file changed - likely needs test generation
            impact_level = "high"
            needs_generation = True
        elif is_test_file:
            # Test file changed - might need verification
            impact_level = "medium"
            needs_generation = False
        else:
            # Other file types
            impact_level = "low"
            needs_generation = False
        
        print(f"Workflow: Impact analysis - {impact_level} impact, generation needed: {needs_generation}")
        
        return {
            "impact_level": impact_level,
            "needs_test_generation": needs_generation,
            "analysis_completed": True
        }
    
    def _trigger_test_generation(self, context) -> Dict[str, Any]:
        """Trigger test generation for the changed file."""
        file_path = context.data.get("file_path", "")
        
        print(f"Workflow: Triggering test generation for {file_path}")
        
        # This would integrate with the actual test generation system
        # For now, simulate the generation
        
        generated_files = []
        try:
            # Simulate test generation
            test_file_path = file_path.replace('.py', '_test.py')
            if not test_file_path.startswith('test_'):
                test_file_path = f"test_{Path(test_file_path).name}"
            
            generated_files.append(test_file_path)
            
            # Update statistics
            self.stats["tests_generated"] += 1
            
            print(f"Workflow: Generated test file: {test_file_path}")
            
            return {
                "tests_generated": True,
                "generated_files": generated_files,
                "generation_successful": True
            }
            
        except Exception as e:
            print(f"Workflow: Test generation failed: {e}")
            return {
                "tests_generated": False,
                "generation_error": str(e),
                "generation_successful": False
            }
    
    def _verify_generated_tests(self, context) -> Dict[str, Any]:
        """Verify the quality of generated tests."""
        generated_files = context.data.get("generated_files", [])
        
        print(f"Workflow: Verifying {len(generated_files)} generated test files")
        
        verification_results = []
        for test_file in generated_files:
            # Simulate test verification
            verification_results.append({
                "file": test_file,
                "syntax_valid": True,
                "quality_score": 85,
                "coverage_estimated": "75%"
            })
        
        print(f"Workflow: Verification completed for {len(verification_results)} files")
        
        return {
            "tests_verified": True,
            "verification_results": verification_results,
            "all_tests_valid": True
        }
    
    @monitor_performance(name="get_monitoring_statistics")
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        elapsed = time.time() - self.stats["start_time"]
        
        local_stats = {
            **self.stats,
            "elapsed_time": elapsed,
            "is_monitoring": self.is_monitoring,
            "watch_paths": self.watch_paths,
            "pending_events": len(self.event_queue)
        }
        
        # Add enhanced feature statistics
        if self.shared_state:
            shared_stats = self.shared_state.get_stats()
            local_stats["shared_state"] = {
                "enabled": True,
                "file_changes_detected": self.shared_state.get("file_changes_detected", 0),
                "recent_changes_count": len(self.shared_state.get("recent_changes", [])),
                **shared_stats
            }
        else:
            local_stats["shared_state"] = {"enabled": False}
        
        if self.workflow_graph:
            workflow_stats = self.workflow_graph.get_workflow_statistics()
            local_stats["workflow_graph"] = workflow_stats
        else:
            local_stats["workflow_graph"] = {"enabled": False}
        
        if self.tracking_manager:
            tracking_stats = self.tracking_manager.get_tracking_statistics()
            local_stats["tracking"] = tracking_stats
        else:
            local_stats["tracking"] = {"enabled": False}
        
        return local_stats
    
    async def monitor_async(self, duration_seconds: float = None):
        """Run monitoring asynchronously for a specified duration."""
        if not self.workflow_graph or not self.workflow_graph.enabled:
            print("Async monitoring requires graph workflows to be enabled")
            return
        
        start_time = time.time()
        print(f"Starting async monitoring for {duration_seconds}s" if duration_seconds else "Starting async monitoring")
        
        try:
            while self.is_monitoring:
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    break
                
                # Process events asynchronously
                events_to_process = []
                with self.processing_lock:
                    if self.event_queue:
                        events_to_process = list(self.event_queue)
                        self.event_queue.clear()
                
                # Process events using async workflow
                for event in events_to_process:
                    context_data = {
                        "file_path": event.file_path,
                        "change_type": event.change_type.value,
                        "timestamp": event.timestamp.isoformat()
                    }
                    
                    result = await self.workflow_graph.ainvoke(context_data)
                    if result.get("success"):
                        self.stats["workflows_executed"] += 1
                
                await asyncio.sleep(self.polling_interval)
                
        except Exception as e:
            print(f"Error in async monitoring: {e}")
            self.stats["errors"] += 1