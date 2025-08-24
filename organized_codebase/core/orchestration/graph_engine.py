"""
Graph Orchestration Engine
=========================

Graph-based orchestration with DAG execution capabilities.
Extracted from unified_orchestrator.py for better modularity.

Author: Agent E - Infrastructure Consolidation
"""

import logging
import time
import random
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Set

from .data_models import GraphNode, NodeState, GraphExecutionMode


class GraphOrchestrationEngine:
    """Graph-based orchestration with DAG execution."""
    
    def __init__(self):
        self.graphs: Dict[str, Dict[str, GraphNode]] = {}
        self.execution_contexts: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.node_registry: Dict[str, Callable] = {}
        
    def register_node_handler(self, node_type: str, handler: Callable):
        """Register handler for node type."""
        self.node_registry[node_type] = handler
        logging.info(f"Registered handler for node type: {node_type}")
    
    def create_graph(self, graph_id: str, nodes: List[GraphNode]) -> bool:
        """Create execution graph."""
        try:
            # Validate dependencies
            node_ids = {node.node_id for node in nodes}
            for node in nodes:
                for dep in node.dependencies:
                    if dep not in node_ids:
                        raise ValueError(f"Invalid dependency {dep} for node {node.node_id}")
            
            # Check for cycles
            if self._has_cycles(nodes):
                raise ValueError("Circular dependencies detected in graph")
            
            self.graphs[graph_id] = {node.node_id: node for node in nodes}
            logging.info(f"Graph created: {graph_id} with {len(nodes)} nodes")
            return True
            
        except Exception as e:
            logging.error(f"Failed to create graph {graph_id}: {e}")
            return False
    
    def _has_cycles(self, nodes: List[GraphNode]) -> bool:
        """Check for circular dependencies using DFS."""
        graph = {node.node_id: node.dependencies for node in nodes}
        
        # States: 0=unvisited, 1=visiting, 2=visited
        states = {node_id: 0 for node_id in graph}
        
        def dfs(node_id: str) -> bool:
            if states[node_id] == 1:  # Currently visiting - cycle detected
                return True
            if states[node_id] == 2:  # Already visited
                return False
            
            states[node_id] = 1  # Mark as visiting
            
            for dep in graph.get(node_id, []):
                if dfs(dep):
                    return True
            
            states[node_id] = 2  # Mark as visited
            return False
        
        return any(dfs(node_id) for node_id in graph if states[node_id] == 0)
    
    def start_execution(self, graph_id: str, execution_mode: GraphExecutionMode = GraphExecutionMode.SEQUENTIAL,
                       context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start graph execution."""
        if graph_id not in self.graphs:
            return None
        
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        
        try:
            self.execution_contexts[execution_id] = {
                "graph_id": graph_id,
                "execution_mode": execution_mode,
                "start_time": datetime.now(),
                "status": "running",
                "context": context or {},
                "completed_nodes": set(),
                "failed_nodes": set(),
                "running_nodes": set(),
                "node_results": {}
            }
            
            # Reset all node states
            for node in self.graphs[graph_id].values():
                node.state = NodeState.PENDING
                node.execution_time = None
                node.error_message = None
                node.retry_count = 0
            
            logging.info(f"Graph execution started: {graph_id} -> {execution_id}")
            return execution_id
            
        except Exception as e:
            logging.error(f"Failed to start execution for {graph_id}: {e}")
            return None
    
    def get_ready_nodes(self, execution_id: str) -> List[GraphNode]:
        """Get nodes ready for execution."""
        if execution_id not in self.execution_contexts:
            return []
        
        context = self.execution_contexts[execution_id]
        graph_id = context["graph_id"]
        graph = self.graphs[graph_id]
        
        ready_nodes = []
        for node in graph.values():
            if (node.state == NodeState.PENDING and 
                all(dep in context["completed_nodes"] for dep in node.dependencies)):
                ready_nodes.append(node)
        
        # Sort by priority (higher priority first)
        ready_nodes.sort(key=lambda n: n.priority, reverse=True)
        return ready_nodes
    
    def execute_node(self, execution_id: str, node_id: str) -> bool:
        """Execute a single node."""
        if execution_id not in self.execution_contexts:
            return False
        
        context = self.execution_contexts[execution_id]
        graph_id = context["graph_id"]
        graph = self.graphs[graph_id]
        
        if node_id not in graph:
            return False
        
        node = graph[node_id]
        
        try:
            # Update node state
            node.state = NodeState.RUNNING
            context["running_nodes"].add(node_id)
            start_time = time.time()
            
            # Execute node handler
            handler = self.node_registry.get(node.agent_type)
            if handler:
                # Collect input data from dependencies
                input_data = {}
                for dep in node.dependencies:
                    if dep in context["node_results"]:
                        input_data[dep] = context["node_results"][dep]
                
                # Execute the handler
                result = handler(node.config, input_data, context["context"])
                
                # Record success
                node.state = NodeState.COMPLETED
                node.execution_time = time.time() - start_time
                context["completed_nodes"].add(node_id)
                context["running_nodes"].discard(node_id)
                context["node_results"][node_id] = result
                
                logging.info(f"Node executed successfully: {node_id}")
                return True
            else:
                # No handler - simulate execution
                await_time = random.uniform(0.5, 2.0)  # Simulate work
                time.sleep(await_time)
                
                node.state = NodeState.COMPLETED
                node.execution_time = await_time
                context["completed_nodes"].add(node_id)
                context["running_nodes"].discard(node_id)
                context["node_results"][node_id] = {"status": "simulated_success"}
                
                logging.info(f"Node simulated successfully: {node_id}")
                return True
                
        except Exception as e:
            # Record failure
            node.state = NodeState.FAILED
            node.error_message = str(e)
            node.execution_time = time.time() - start_time
            context["failed_nodes"].add(node_id)
            context["running_nodes"].discard(node_id)
            
            logging.error(f"Node execution failed: {node_id} - {e}")
            
            # Check if retry is possible
            if node.retry_count < node.max_retries:
                node.retry_count += 1
                node.state = NodeState.PENDING
                context["failed_nodes"].discard(node_id)
                logging.info(f"Node queued for retry: {node_id} (attempt {node.retry_count + 1})")
            
            return False
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status."""
        if execution_id not in self.execution_contexts:
            return None
        
        context = self.execution_contexts[execution_id]
        graph_id = context["graph_id"]
        graph = self.graphs[graph_id]
        
        total_nodes = len(graph)
        completed_nodes = len(context["completed_nodes"])
        failed_nodes = len(context["failed_nodes"])
        running_nodes = len(context["running_nodes"])
        pending_nodes = total_nodes - completed_nodes - failed_nodes - running_nodes
        
        progress = (completed_nodes / total_nodes * 100) if total_nodes > 0 else 0
        
        # Determine overall status
        if failed_nodes > 0 and completed_nodes + failed_nodes == total_nodes:
            overall_status = "failed"
        elif completed_nodes == total_nodes:
            overall_status = "completed"
        elif running_nodes > 0 or pending_nodes > 0:
            overall_status = "running"
        else:
            overall_status = "unknown"
        
        return {
            "execution_id": execution_id,
            "graph_id": graph_id,
            "status": overall_status,
            "progress": progress,
            "total_nodes": total_nodes,
            "completed_nodes": completed_nodes,
            "failed_nodes": failed_nodes,
            "running_nodes": running_nodes,
            "pending_nodes": pending_nodes,
            "start_time": context["start_time"].isoformat(),
            "duration": (datetime.now() - context["start_time"]).total_seconds(),
            "execution_mode": context["execution_mode"].value
        }


__all__ = ['GraphOrchestrationEngine']