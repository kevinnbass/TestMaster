"""
Execution Flow Analyzer for TestMaster

Advanced execution flow analysis and visualization system
inspired by PraisonAI's flow analysis capabilities.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
import json
import uuid

from core.feature_flags import FeatureFlags
from core.shared_state import get_shared_state
from .telemetry_collector import get_telemetry_collector

@dataclass
class FlowNode:
    """Represents a node in the execution flow."""
    node_id: str
    component: str
    operation: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    thread_id: str = ""
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    depth: int = 0

@dataclass
class FlowPath:
    """Represents a complete execution path."""
    path_id: str
    nodes: List[FlowNode]
    total_duration_ms: float
    start_time: datetime
    end_time: datetime
    success: bool
    critical_path: bool = False
    bottleneck_nodes: List[str] = field(default_factory=list)

@dataclass
class FlowAnalysis:
    """Results of flow analysis."""
    analysis_id: str
    total_flows: int
    successful_flows: int
    failed_flows: int
    avg_duration_ms: float
    critical_paths: List[FlowPath]
    bottleneck_components: List[str]
    parallelism_analysis: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime

class ExecutionFlowAnalyzer:
    """
    Advanced execution flow analyzer for TestMaster.
    
    Provides:
    - Real-time flow tracking and visualization
    - Critical path analysis
    - Bottleneck detection
    - Parallelism analysis
    - Performance optimization recommendations
    """
    
    def __init__(self, max_flows: int = 10000, analysis_window: int = 1000):
        """
        Initialize execution flow analyzer.
        
        Args:
            max_flows: Maximum flow nodes to keep in memory
            analysis_window: Window size for analysis
        """
        self.enabled = FeatureFlags.is_enabled('layer3_orchestration', 'telemetry_system')
        
        if not self.enabled:
            return
        
        self.max_flows = max_flows
        self.analysis_window = analysis_window
        
        # Data storage
        self.flow_nodes: deque = deque(maxlen=max_flows)
        self.active_flows: Dict[str, FlowNode] = {}  # thread_id -> current node
        self.flow_stack: Dict[str, List[str]] = defaultdict(list)  # thread_id -> node_id stack
        
        # Threading
        self.lock = threading.RLock()
        self.analyzer_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Analysis results
        self.flows_analyzed = 0
        self.last_analysis: Optional[FlowAnalysis] = None
        self.bottlenecks_cache: List[str] = []
        
        # Integrations
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
        
        self.telemetry = get_telemetry_collector()
        
        # Start background analysis
        self._start_analyzer_thread()
        
        print("Execution flow analyzer initialized")
        print(f"   Flow tracking: {self.max_flows} nodes")
        print(f"   Analysis window: {self.analysis_window}")
    
    def start_flow(self, component: str, operation: str, 
                   metadata: Dict[str, Any] = None) -> str:
        """
        Start tracking a new flow node.
        
        Args:
            component: Component name
            operation: Operation name
            metadata: Additional metadata
            
        Returns:
            Node ID for the started flow
        """
        if not self.enabled:
            return ""
        
        thread_id = str(threading.get_ident())
        node_id = str(uuid.uuid4())
        
        # Determine parent and depth
        parent_id = None
        depth = 0
        
        with self.lock:
            if thread_id in self.flow_stack and self.flow_stack[thread_id]:
                parent_id = self.flow_stack[thread_id][-1]
                # Find parent node to get depth
                for node in reversed(self.flow_nodes):
                    if node.node_id == parent_id:
                        depth = node.depth + 1
                        break
            
            # Create flow node
            node = FlowNode(
                node_id=node_id,
                component=component,
                operation=operation,
                start_time=datetime.now(),
                thread_id=thread_id,
                parent_id=parent_id,
                metadata=metadata or {},
                depth=depth
            )
            
            # Add to tracking structures
            self.active_flows[node_id] = node
            self.flow_stack[thread_id].append(node_id)
            
            # Update parent's children
            if parent_id:
                for existing_node in reversed(self.flow_nodes):
                    if existing_node.node_id == parent_id:
                        existing_node.children_ids.append(node_id)
                        break
        
        return node_id
    
    def end_flow(self, node_id: str, success: bool = True, error_message: str = None):
        """
        End tracking for a flow node.
        
        Args:
            node_id: Node ID returned from start_flow
            success: Whether the operation succeeded
            error_message: Error message if failed
        """
        if not self.enabled or not node_id:
            return
        
        thread_id = str(threading.get_ident())
        
        with self.lock:
            # Find and update the node
            if node_id in self.active_flows:
                node = self.active_flows[node_id]
                node.end_time = datetime.now()
                node.duration_ms = (node.end_time - node.start_time).total_seconds() * 1000
                node.success = success
                node.error_message = error_message
                
                # Move to completed flows
                self.flow_nodes.append(node)
                del self.active_flows[node_id]
                
                # Remove from flow stack
                if thread_id in self.flow_stack and node_id in self.flow_stack[thread_id]:
                    self.flow_stack[thread_id].remove(node_id)
                
                # Send to telemetry
                self.telemetry.record_event(
                    event_type="execution_flow",
                    component=node.component,
                    operation=node.operation,
                    metadata={
                        **node.metadata,
                        "node_id": node_id,
                        "depth": node.depth,
                        "parent_id": node.parent_id,
                        "children_count": len(node.children_ids)
                    },
                    duration_ms=node.duration_ms,
                    success=success,
                    error_message=error_message
                )
                
                # Update shared state
                if self.shared_state:
                    self.shared_state.increment("flow_nodes_completed")
                    if not success:
                        self.shared_state.increment("flow_nodes_failed")
    
    def analyze_flows(self, timeframe_hours: int = 1) -> FlowAnalysis:
        """
        Analyze execution flows and identify patterns.
        
        Args:
            timeframe_hours: Analysis timeframe in hours
            
        Returns:
            Flow analysis results
        """
        if not self.enabled:
            return FlowAnalysis(
                analysis_id="disabled",
                total_flows=0,
                successful_flows=0,
                failed_flows=0,
                avg_duration_ms=0,
                critical_paths=[],
                bottleneck_components=[],
                parallelism_analysis={},
                recommendations=[],
                generated_at=datetime.now()
            )
        
        cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
        
        with self.lock:
            # Filter nodes by timeframe
            relevant_nodes = [
                node for node in self.flow_nodes
                if node.end_time and node.end_time >= cutoff_time
            ]
        
        if not relevant_nodes:
            return FlowAnalysis(
                analysis_id=str(uuid.uuid4()),
                total_flows=0,
                successful_flows=0,
                failed_flows=0,
                avg_duration_ms=0,
                critical_paths=[],
                bottleneck_components=[],
                parallelism_analysis={},
                recommendations=["No flows to analyze in the specified timeframe"],
                generated_at=datetime.now()
            )
        
        # Build flow paths
        flow_paths = self._build_flow_paths(relevant_nodes)
        
        # Analyze patterns
        total_flows = len(flow_paths)
        successful_flows = sum(1 for path in flow_paths if path.success)
        failed_flows = total_flows - successful_flows
        avg_duration = sum(path.total_duration_ms for path in flow_paths) / total_flows if total_flows > 0 else 0
        
        # Identify critical paths
        critical_paths = self._identify_critical_paths(flow_paths)
        
        # Analyze bottlenecks
        bottleneck_components = self._analyze_bottlenecks(relevant_nodes)
        
        # Analyze parallelism
        parallelism_analysis = self._analyze_parallelism(relevant_nodes)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            flow_paths, bottleneck_components, parallelism_analysis
        )
        
        analysis = FlowAnalysis(
            analysis_id=str(uuid.uuid4()),
            total_flows=total_flows,
            successful_flows=successful_flows,
            failed_flows=failed_flows,
            avg_duration_ms=round(avg_duration, 2),
            critical_paths=critical_paths,
            bottleneck_components=bottleneck_components,
            parallelism_analysis=parallelism_analysis,
            recommendations=recommendations,
            generated_at=datetime.now()
        )
        
        self.last_analysis = analysis
        self.flows_analyzed += 1
        self.bottlenecks_cache = bottleneck_components
        
        return analysis
    
    def _build_flow_paths(self, nodes: List[FlowNode]) -> List[FlowPath]:
        """Build complete flow paths from nodes."""
        # Group nodes by thread and build hierarchical paths
        thread_nodes = defaultdict(list)
        for node in nodes:
            thread_nodes[node.thread_id].append(node)
        
        flow_paths = []
        
        for thread_id, thread_node_list in thread_nodes.items():
            # Sort by start time
            thread_node_list.sort(key=lambda x: x.start_time)
            
            # Find root nodes (no parent or parent not in same thread)
            root_nodes = [
                node for node in thread_node_list
                if not node.parent_id or not any(
                    n.node_id == node.parent_id for n in thread_node_list
                )
            ]
            
            # Build paths from each root
            for root in root_nodes:
                path_nodes = self._collect_path_nodes(root, thread_node_list)
                if path_nodes:
                    path = self._create_flow_path(path_nodes)
                    flow_paths.append(path)
        
        return flow_paths
    
    def _collect_path_nodes(self, root: FlowNode, all_nodes: List[FlowNode]) -> List[FlowNode]:
        """Collect all nodes in a path starting from root."""
        path_nodes = [root]
        
        # Recursively collect children
        def collect_children(node: FlowNode):
            for child_id in node.children_ids:
                child = next((n for n in all_nodes if n.node_id == child_id), None)
                if child:
                    path_nodes.append(child)
                    collect_children(child)
        
        collect_children(root)
        return path_nodes
    
    def _create_flow_path(self, nodes: List[FlowNode]) -> FlowPath:
        """Create a FlowPath from a list of nodes."""
        if not nodes:
            return None
        
        # Sort by start time
        nodes.sort(key=lambda x: x.start_time)
        
        total_duration = sum(node.duration_ms or 0 for node in nodes)
        start_time = min(node.start_time for node in nodes)
        end_time = max(node.end_time for node in nodes if node.end_time)
        success = all(node.success for node in nodes)
        
        # Identify bottleneck nodes (top 10% slowest)
        sorted_by_duration = sorted(
            [node for node in nodes if node.duration_ms],
            key=lambda x: x.duration_ms,
            reverse=True
        )
        bottleneck_count = max(1, len(sorted_by_duration) // 10)
        bottleneck_nodes = [node.node_id for node in sorted_by_duration[:bottleneck_count]]
        
        return FlowPath(
            path_id=str(uuid.uuid4()),
            nodes=nodes,
            total_duration_ms=total_duration,
            start_time=start_time,
            end_time=end_time,
            success=success,
            bottleneck_nodes=bottleneck_nodes
        )
    
    def _identify_critical_paths(self, flow_paths: List[FlowPath]) -> List[FlowPath]:
        """Identify critical paths (longest duration)."""
        if not flow_paths:
            return []
        
        # Sort by duration and take top 10%
        sorted_paths = sorted(flow_paths, key=lambda x: x.total_duration_ms, reverse=True)
        critical_count = max(1, len(sorted_paths) // 10)
        
        critical_paths = sorted_paths[:critical_count]
        for path in critical_paths:
            path.critical_path = True
        
        return critical_paths
    
    def _analyze_bottlenecks(self, nodes: List[FlowNode]) -> List[str]:
        """Analyze and identify bottleneck components."""
        component_stats = defaultdict(lambda: {"total_duration": 0, "count": 0})
        
        for node in nodes:
            if node.duration_ms:
                component_stats[node.component]["total_duration"] += node.duration_ms
                component_stats[node.component]["count"] += 1
        
        # Calculate average duration per component
        component_avg_duration = {}
        for component, stats in component_stats.items():
            if stats["count"] > 0:
                component_avg_duration[component] = stats["total_duration"] / stats["count"]
        
        # Sort by average duration and identify top bottlenecks
        sorted_components = sorted(
            component_avg_duration.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top 5 bottleneck components
        return [comp for comp, _ in sorted_components[:5]]
    
    def _analyze_parallelism(self, nodes: List[FlowNode]) -> Dict[str, Any]:
        """Analyze parallelism patterns in execution."""
        if not nodes:
            return {}
        
        # Group by time windows to analyze concurrent execution
        time_windows = defaultdict(set)
        
        for node in nodes:
            if node.end_time and node.duration_ms:
                # Create 1-second time windows
                start_window = int(node.start_time.timestamp())
                end_window = int(node.end_time.timestamp())
                
                for window in range(start_window, end_window + 1):
                    time_windows[window].add(node.thread_id)
        
        # Calculate parallelism metrics
        max_concurrent_threads = max(len(threads) for threads in time_windows.values()) if time_windows else 0
        avg_concurrent_threads = (
            sum(len(threads) for threads in time_windows.values()) / len(time_windows)
            if time_windows else 0
        )
        
        # Thread utilization
        thread_ids = set(node.thread_id for node in nodes)
        total_threads = len(thread_ids)
        
        return {
            "max_concurrent_threads": max_concurrent_threads,
            "avg_concurrent_threads": round(avg_concurrent_threads, 2),
            "total_threads_used": total_threads,
            "parallelism_efficiency": round(avg_concurrent_threads / max(total_threads, 1), 2)
        }
    
    def _generate_recommendations(self, flow_paths: List[FlowPath],
                                bottleneck_components: List[str],
                                parallelism_analysis: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Bottleneck recommendations
        if bottleneck_components:
            recommendations.append(
                f"Optimize bottleneck components: {', '.join(bottleneck_components[:3])}"
            )
        
        # Parallelism recommendations
        parallelism_efficiency = parallelism_analysis.get("parallelism_efficiency", 0)
        if parallelism_efficiency < 0.5:
            recommendations.append(
                "Consider increasing parallelism to improve throughput"
            )
        
        # Flow path recommendations
        if flow_paths:
            avg_duration = sum(path.total_duration_ms for path in flow_paths) / len(flow_paths)
            if avg_duration > 5000:  # 5 seconds
                recommendations.append(
                    "Consider breaking down long-running operations into smaller chunks"
                )
        
        # Error rate recommendations
        failed_paths = sum(1 for path in flow_paths if not path.success)
        if failed_paths > 0:
            error_rate = (failed_paths / len(flow_paths)) * 100
            if error_rate > 5:
                recommendations.append(
                    f"High error rate ({error_rate:.1f}%) - investigate failure patterns"
                )
        
        if not recommendations:
            recommendations.append("No significant performance issues detected")
        
        return recommendations
    
    def visualize_flow(self, analysis: FlowAnalysis = None) -> str:
        """Generate a text-based flow visualization."""
        if not self.enabled:
            return "Flow visualization disabled"
        
        if not analysis:
            analysis = self.last_analysis
        
        if not analysis or not analysis.critical_paths:
            return "No flow data available for visualization"
        
        visualization = []
        visualization.append("EXECUTION FLOW VISUALIZATION")
        visualization.append("=" * 50)
        
        for i, path in enumerate(analysis.critical_paths[:3]):  # Show top 3 critical paths
            visualization.append(f"\nCritical Path {i+1} (Duration: {path.total_duration_ms:.1f}ms)")
            visualization.append("-" * 30)
            
            for node in path.nodes:
                indent = "  " * node.depth
                status = "[OK]" if node.success else "[FAIL]"
                duration = f"({node.duration_ms:.1f}ms)" if node.duration_ms else ""
                
                visualization.append(f"{indent}{status} {node.component}.{node.operation} {duration}")
        
        return "\n".join(visualization)
    
    def get_flow_statistics(self) -> Dict[str, Any]:
        """Get flow analysis statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        with self.lock:
            active_count = len(self.active_flows)
            completed_count = len(self.flow_nodes)
            
            stats = {
                "enabled": True,
                "flows_analyzed": self.flows_analyzed,
                "active_flows": active_count,
                "completed_flows": completed_count,
                "bottlenecks_detected": len(self.bottlenecks_cache),
                "last_analysis": self.last_analysis.generated_at.isoformat() if self.last_analysis else None
            }
            
            if self.last_analysis:
                stats["last_analysis_summary"] = {
                    "total_flows": self.last_analysis.total_flows,
                    "success_rate": (self.last_analysis.successful_flows / max(self.last_analysis.total_flows, 1)) * 100,
                    "avg_duration_ms": self.last_analysis.avg_duration_ms
                }
        
        return stats
    
    def clear_data(self):
        """Clear all flow data."""
        if not self.enabled:
            return
        
        with self.lock:
            self.flow_nodes.clear()
            self.active_flows.clear()
            self.flow_stack.clear()
            self.flows_analyzed = 0
            self.last_analysis = None
            self.bottlenecks_cache.clear()
    
    def _start_analyzer_thread(self):
        """Start background analyzer thread."""
        if not self.enabled:
            return
        
        def analyzer_worker():
            while not self.shutdown_event.is_set():
                try:
                    if self.shutdown_event.wait(timeout=300):  # Analyze every 5 minutes
                        break
                    
                    # Perform periodic analysis
                    self.analyze_flows(timeframe_hours=1)
                    
                except Exception:
                    # Handle errors silently
                    pass
        
        self.analyzer_thread = threading.Thread(target=analyzer_worker, daemon=True)
        self.analyzer_thread.start()
    
    def shutdown(self):
        """Shutdown flow analyzer."""
        if not self.enabled:
            return
        
        self.shutdown_event.set()
        
        if self.analyzer_thread and self.analyzer_thread.is_alive():
            self.analyzer_thread.join(timeout=1.0)
        
        print(f"Flow analyzer shutdown - analyzed {self.flows_analyzed} flows")

# Global instance
_flow_analyzer: Optional[ExecutionFlowAnalyzer] = None

def get_flow_analyzer() -> ExecutionFlowAnalyzer:
    """Get the global flow analyzer instance."""
    global _flow_analyzer
    if _flow_analyzer is None:
        _flow_analyzer = ExecutionFlowAnalyzer()
    return _flow_analyzer

# Convenience functions
def analyze_execution_flow(timeframe_hours: int = 1) -> FlowAnalysis:
    """Analyze execution flow patterns."""
    analyzer = get_flow_analyzer()
    return analyzer.analyze_flows(timeframe_hours)

def visualize_flow(analysis: FlowAnalysis = None) -> str:
    """Generate flow visualization."""
    analyzer = get_flow_analyzer()
    return analyzer.visualize_flow(analysis)

def detect_bottlenecks(timeframe_hours: int = 1) -> List[str]:
    """Detect bottleneck components."""
    analysis = analyze_execution_flow(timeframe_hours)
    return analysis.bottleneck_components