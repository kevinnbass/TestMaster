"""
Distributed Test Coordinator - Cross-repository test execution and coordination

This coordinator provides:
- Distributed test execution across multiple repositories
- Cross-repository test synchronization and dependencies
- Load balancing and resource optimization
- Real-time coordination and communication
- Fault tolerance and recovery mechanisms
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import threading
import uuid
import hashlib
from pathlib import Path
import concurrent.futures
import websockets
import time
import socket

# Mock Framework Imports for Testing
import pytest
from unittest.mock import Mock, patch, MagicMock
import unittest

class NodeType(Enum):
    COORDINATOR = "coordinator"  # Central coordinator node
    WORKER = "worker"           # Test execution worker
    MONITOR = "monitor"         # Monitoring and metrics node
    GATEWAY = "gateway"         # API gateway node

class NodeStatus(Enum):
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    OFFLINE = "offline"

class TestJobStatus(Enum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class Priority(Enum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

@dataclass
class NodeInfo:
    """Information about a distributed test node"""
    node_id: str
    node_type: NodeType
    host: str
    port: int
    status: NodeStatus
    capabilities: List[str]
    max_concurrent_jobs: int
    current_jobs: int = 0
    last_heartbeat: Optional[datetime] = None
    load_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestJob:
    """Distributed test job definition"""
    job_id: str
    repository: str
    test_suite: str
    test_command: str
    priority: Priority
    timeout: int = 3600  # 1 hour default
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    resources_required: Dict[str, Any] = field(default_factory=dict)
    assigned_node: Optional[str] = None
    status: TestJobStatus = TestJobStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

@dataclass
class ExecutionPlan:
    """Execution plan for distributed testing"""
    plan_id: str
    repositories: List[str]
    total_jobs: int
    dependency_graph: Dict[str, List[str]]
    estimated_duration: int
    resource_requirements: Dict[str, Any]
    execution_strategy: str = "parallel"  # parallel, sequential, hybrid
    created_at: datetime = field(default_factory=datetime.now)

class LoadBalancer:
    """Intelligent load balancing for distributed test execution"""
    
    def __init__(self):
        self.nodes: Dict[str, NodeInfo] = {}
        self.job_history: List[Dict[str, Any]] = []
        
    def register_node(self, node_info: NodeInfo) -> bool:
        """Register a new node"""
        self.nodes[node_info.node_id] = node_info
        return True
    
    def unregister_node(self, node_id: str) -> bool:
        """Unregister a node"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            return True
        return False
    
    def update_node_status(self, node_id: str, status: NodeStatus, 
                          load_metrics: Optional[Dict[str, float]] = None) -> None:
        """Update node status and metrics"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.status = status
            node.last_heartbeat = datetime.now()
            
            if load_metrics:
                node.load_metrics.update(load_metrics)
    
    def find_best_node(self, job: TestJob) -> Optional[str]:
        """Find the best node for job execution"""
        available_nodes = [
            node for node in self.nodes.values()
            if node.status == NodeStatus.READY and
            node.current_jobs < node.max_concurrent_jobs and
            self._node_supports_job(node, job)
        ]
        
        if not available_nodes:
            return None
        
        # Score nodes based on multiple factors
        node_scores = []
        for node in available_nodes:
            score = self._calculate_node_score(node, job)
            node_scores.append((node.node_id, score))
        
        # Return node with highest score
        node_scores.sort(key=lambda x: x[1], reverse=True)
        return node_scores[0][0]
    
    def _node_supports_job(self, node: NodeInfo, job: TestJob) -> bool:
        """Check if node supports job requirements"""
        # Check capabilities
        required_capabilities = job.resources_required.get("capabilities", [])
        if not all(cap in node.capabilities for cap in required_capabilities):
            return False
        
        # Check resource requirements
        required_memory = job.resources_required.get("memory_mb", 0)
        available_memory = node.load_metrics.get("available_memory_mb", 0)
        if required_memory > available_memory:
            return False
        
        return True
    
    def _calculate_node_score(self, node: NodeInfo, job: TestJob) -> float:
        """Calculate node suitability score for job"""
        score = 100.0  # Base score
        
        # Load factor (lower load = higher score)
        cpu_load = node.load_metrics.get("cpu_usage", 0.5)
        memory_load = node.load_metrics.get("memory_usage", 0.5)
        load_penalty = (cpu_load + memory_load) * 25
        score -= load_penalty
        
        # Concurrency factor
        concurrency_ratio = node.current_jobs / max(1, node.max_concurrent_jobs)
        score -= concurrency_ratio * 20
        
        # Priority bonus for critical jobs
        if job.priority == Priority.CRITICAL:
            score += 10
        
        # Historical performance bonus
        node_performance = self._get_node_performance(node.node_id)
        score += node_performance * 15
        
        return max(0, score)
    
    def _get_node_performance(self, node_id: str) -> float:
        """Get historical performance score for node"""
        # Analyze job history for this node
        node_jobs = [
            job for job in self.job_history[-100:]  # Last 100 jobs
            if job.get("assigned_node") == node_id
        ]
        
        if not node_jobs:
            return 0.5  # Neutral score for new nodes
        
        success_rate = sum(1 for job in node_jobs if job.get("status") == "completed") / len(node_jobs)
        avg_duration_ratio = sum(
            job.get("actual_duration", 0) / max(1, job.get("estimated_duration", 1))
            for job in node_jobs
        ) / len(node_jobs)
        
        # Performance score (higher is better)
        performance = success_rate * (2.0 - min(avg_duration_ratio, 2.0))
        return min(1.0, max(0.0, performance))

class DependencyResolver:
    """Resolves job dependencies for execution planning"""
    
    def __init__(self):
        self.dependency_cache: Dict[str, List[str]] = {}
        
    def resolve_execution_order(self, jobs: List[TestJob]) -> List[List[str]]:
        """Resolve job execution order respecting dependencies"""
        # Build dependency graph
        job_map = {job.job_id: job for job in jobs}
        dependency_graph = {}
        
        for job in jobs:
            dependency_graph[job.job_id] = job.dependencies.copy()
        
        # Topological sort to determine execution levels
        execution_levels = []
        remaining_jobs = set(job_map.keys())
        
        while remaining_jobs:
            # Find jobs with no pending dependencies
            ready_jobs = []
            for job_id in remaining_jobs:
                if not dependency_graph[job_id]:
                    ready_jobs.append(job_id)
            
            if not ready_jobs:
                # Circular dependency detected
                logging.error("Circular dependency detected in job graph")
                break
            
            execution_levels.append(ready_jobs)
            
            # Remove ready jobs and their dependencies
            for job_id in ready_jobs:
                remaining_jobs.remove(job_id)
                
                # Remove this job from other jobs' dependencies
                for other_job_id in remaining_jobs:
                    if job_id in dependency_graph[other_job_id]:
                        dependency_graph[other_job_id].remove(job_id)
        
        return execution_levels
    
    def validate_dependencies(self, jobs: List[TestJob]) -> List[str]:
        """Validate job dependencies and return any issues"""
        issues = []
        job_ids = {job.job_id for job in jobs}
        
        for job in jobs:
            # Check for missing dependencies
            for dep_id in job.dependencies:
                if dep_id not in job_ids:
                    issues.append(f"Job {job.job_id} depends on non-existent job {dep_id}")
            
            # Check for self-dependencies
            if job.job_id in job.dependencies:
                issues.append(f"Job {job.job_id} has self-dependency")
        
        # Check for circular dependencies
        if self._has_circular_dependencies(jobs):
            issues.append("Circular dependencies detected in job graph")
        
        return issues
    
    def _has_circular_dependencies(self, jobs: List[TestJob]) -> bool:
        """Check for circular dependencies using DFS"""
        job_map = {job.job_id: job for job in jobs}
        visited = set()
        rec_stack = set()
        
        def dfs(job_id: str) -> bool:
            if job_id in rec_stack:
                return True  # Circular dependency found
            
            if job_id in visited:
                return False
            
            visited.add(job_id)
            rec_stack.add(job_id)
            
            if job_id in job_map:
                for dep_id in job_map[job_id].dependencies:
                    if dfs(dep_id):
                        return True
            
            rec_stack.remove(job_id)
            return False
        
        for job in jobs:
            if job.job_id not in visited:
                if dfs(job.job_id):
                    return True
        
        return False

class CommunicationManager:
    """Manages communication between distributed nodes"""
    
    def __init__(self, coordinator_port: int = 8765):
        self.coordinator_port = coordinator_port
        self.connections: Dict[str, Any] = {}  # WebSocket connections
        self.message_queue: List[Dict[str, Any]] = []
        self.is_running = False
        self.server_thread: Optional[threading.Thread] = None
        
    async def start_coordinator_server(self) -> None:
        """Start WebSocket server for coordinator"""
        self.is_running = True
        
        async def handle_client(websocket, path):
            node_id = None
            try:
                async for message in websocket:
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    if message_type == "register":
                        node_id = data.get("node_id")
                        self.connections[node_id] = websocket
                        await self._send_response(websocket, {
                            "type": "registration_ack",
                            "node_id": node_id,
                            "status": "registered"
                        })
                    
                    elif message_type == "heartbeat":
                        node_id = data.get("node_id")
                        await self._handle_heartbeat(node_id, data)
                    
                    elif message_type == "job_result":
                        await self._handle_job_result(data)
                    
                    elif message_type == "status_update":
                        await self._handle_status_update(data)
                        
            except Exception as e:
                logging.error(f"WebSocket error for node {node_id}: {e}")
            finally:
                if node_id and node_id in self.connections:
                    del self.connections[node_id]
        
        # Mock WebSocket server - in real implementation would use websockets library
        logging.info(f"Coordinator server started on port {self.coordinator_port}")
        
        # Keep server running
        while self.is_running:
            await asyncio.sleep(1)
    
    async def _send_response(self, websocket: Any, data: Dict[str, Any]) -> None:
        """Send response to WebSocket client"""
        try:
            await websocket.send(json.dumps(data))
        except Exception as e:
            logging.error(f"Failed to send WebSocket response: {e}")
    
    async def _handle_heartbeat(self, node_id: str, data: Dict[str, Any]) -> None:
        """Handle heartbeat from node"""
        # Update node status and metrics
        logging.debug(f"Heartbeat received from {node_id}")
        
        # Store heartbeat data
        self.message_queue.append({
            "type": "heartbeat",
            "node_id": node_id,
            "timestamp": datetime.now(),
            "data": data
        })
    
    async def _handle_job_result(self, data: Dict[str, Any]) -> None:
        """Handle job result from worker node"""
        job_id = data.get("job_id")
        result = data.get("result")
        
        logging.info(f"Job result received for {job_id}: {result.get('status')}")
        
        self.message_queue.append({
            "type": "job_result",
            "job_id": job_id,
            "timestamp": datetime.now(),
            "data": data
        })
    
    async def _handle_status_update(self, data: Dict[str, Any]) -> None:
        """Handle status update from node"""
        node_id = data.get("node_id")
        status = data.get("status")
        
        logging.info(f"Status update from {node_id}: {status}")
        
        self.message_queue.append({
            "type": "status_update",
            "node_id": node_id,
            "timestamp": datetime.now(),
            "data": data
        })
    
    async def broadcast_message(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all connected nodes"""
        successful_sends = 0
        
        for node_id, websocket in self.connections.items():
            try:
                await self._send_response(websocket, message)
                successful_sends += 1
            except Exception as e:
                logging.error(f"Failed to send broadcast to {node_id}: {e}")
        
        return successful_sends
    
    async def send_to_node(self, node_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific node"""
        if node_id not in self.connections:
            return False
        
        try:
            await self._send_response(self.connections[node_id], message)
            return True
        except Exception as e:
            logging.error(f"Failed to send message to {node_id}: {e}")
            return False
    
    def stop_server(self) -> None:
        """Stop the coordinator server"""
        self.is_running = False

class FaultTolerance:
    """Fault tolerance and recovery mechanisms"""
    
    def __init__(self):
        self.failed_jobs: List[TestJob] = []
        self.node_failures: Dict[str, List[datetime]] = {}
        self.recovery_strategies = {
            "retry": self._retry_job,
            "reschedule": self._reschedule_job,
            "skip": self._skip_job
        }
        
    def handle_job_failure(self, job: TestJob, error: str) -> str:
        """Handle job failure and determine recovery strategy"""
        job.error_message = error
        job.status = TestJobStatus.FAILED
        self.failed_jobs.append(job)
        
        # Determine recovery strategy
        if job.retry_count < job.max_retries:
            return "retry"
        elif job.priority in [Priority.CRITICAL, Priority.HIGH]:
            return "reschedule"
        else:
            return "skip"
    
    def handle_node_failure(self, node_id: str) -> List[TestJob]:
        """Handle node failure and recover running jobs"""
        # Record node failure
        if node_id not in self.node_failures:
            self.node_failures[node_id] = []
        self.node_failures[node_id].append(datetime.now())
        
        # Find jobs assigned to failed node
        affected_jobs = []
        for job in self.failed_jobs:
            if job.assigned_node == node_id and job.status == TestJobStatus.RUNNING:
                affected_jobs.append(job)
                job.assigned_node = None
                job.status = TestJobStatus.QUEUED
        
        return affected_jobs
    
    def _retry_job(self, job: TestJob) -> TestJob:
        """Retry failed job"""
        job.retry_count += 1
        job.status = TestJobStatus.RETRYING
        job.assigned_node = None
        job.error_message = None
        return job
    
    def _reschedule_job(self, job: TestJob) -> TestJob:
        """Reschedule job to different node"""
        job.assigned_node = None
        job.status = TestJobStatus.QUEUED
        job.error_message = None
        return job
    
    def _skip_job(self, job: TestJob) -> TestJob:
        """Skip failed job"""
        job.status = TestJobStatus.CANCELLED
        return job
    
    def is_node_healthy(self, node_id: str) -> bool:
        """Check if node is considered healthy"""
        if node_id not in self.node_failures:
            return True
        
        # Check failure rate in last hour
        recent_failures = [
            failure for failure in self.node_failures[node_id]
            if datetime.now() - failure < timedelta(hours=1)
        ]
        
        return len(recent_failures) < 3  # Less than 3 failures per hour

class DistributedTestCoordinator:
    """Main coordinator for distributed test execution"""
    
    def __init__(self, coordinator_port: int = 8765):
        self.coordinator_id = str(uuid.uuid4())
        self.load_balancer = LoadBalancer()
        self.dependency_resolver = DependencyResolver()
        self.communication_manager = CommunicationManager(coordinator_port)
        self.fault_tolerance = FaultTolerance()
        
        self.job_queue: List[TestJob] = []
        self.active_jobs: Dict[str, TestJob] = {}
        self.completed_jobs: List[TestJob] = []
        self.execution_plans: Dict[str, ExecutionPlan] = {}
        
        self.is_running = False
        self._coordination_thread: Optional[threading.Thread] = None
        
    async def start_coordination(self) -> None:
        """Start the distributed coordination system"""
        self.is_running = True
        
        # Start communication server
        server_task = asyncio.create_task(
            self.communication_manager.start_coordinator_server()
        )
        
        # Start coordination loop
        coordination_task = asyncio.create_task(self._coordination_loop())
        
        # Wait for both tasks
        await asyncio.gather(server_task, coordination_task)
    
    async def _coordination_loop(self) -> None:
        """Main coordination loop"""
        while self.is_running:
            try:
                # Process message queue
                await self._process_messages()
                
                # Schedule pending jobs
                await self._schedule_jobs()
                
                # Monitor active jobs
                await self._monitor_jobs()
                
                # Update node statuses
                await self._update_node_statuses()
                
                # Sleep briefly to prevent busy waiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logging.error(f"Coordination loop error: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _process_messages(self) -> None:
        """Process incoming messages from nodes"""
        messages_to_process = self.communication_manager.message_queue.copy()
        self.communication_manager.message_queue.clear()
        
        for message in messages_to_process:
            message_type = message.get("type")
            
            if message_type == "heartbeat":
                node_id = message.get("node_id")
                data = message.get("data", {})
                self.load_balancer.update_node_status(
                    node_id, NodeStatus.READY, data.get("load_metrics")
                )
                
            elif message_type == "job_result":
                job_id = message.get("job_id")
                if job_id in self.active_jobs:
                    await self._handle_job_completion(job_id, message.get("data"))
                    
            elif message_type == "status_update":
                node_id = message.get("node_id")
                status_data = message.get("data", {})
                await self._handle_node_status_update(node_id, status_data)
    
    async def _schedule_jobs(self) -> None:
        """Schedule queued jobs to available nodes"""
        jobs_to_schedule = [job for job in self.job_queue if job.status == TestJobStatus.QUEUED]
        
        for job in jobs_to_schedule:
            # Check dependencies
            if not self._dependencies_satisfied(job):
                continue
            
            # Find best node for job
            best_node = self.load_balancer.find_best_node(job)
            if not best_node:
                continue
            
            # Assign job to node
            job.assigned_node = best_node
            job.status = TestJobStatus.ASSIGNED
            job.started_at = datetime.now()
            
            # Send job to node
            await self._send_job_to_node(job, best_node)
            
            # Move to active jobs
            self.active_jobs[job.job_id] = job
            self.job_queue.remove(job)
            
            # Update node job count
            if best_node in self.load_balancer.nodes:
                self.load_balancer.nodes[best_node].current_jobs += 1
    
    def _dependencies_satisfied(self, job: TestJob) -> bool:
        """Check if job dependencies are satisfied"""
        for dep_id in job.dependencies:
            # Check if dependency is completed
            completed_job = next(
                (j for j in self.completed_jobs if j.job_id == dep_id),
                None
            )
            
            if not completed_job or completed_job.status != TestJobStatus.COMPLETED:
                return False
        
        return True
    
    async def _send_job_to_node(self, job: TestJob, node_id: str) -> None:
        """Send job to worker node for execution"""
        message = {
            "type": "execute_job",
            "job": {
                "job_id": job.job_id,
                "repository": job.repository,
                "test_suite": job.test_suite,
                "test_command": job.test_command,
                "timeout": job.timeout,
                "environment": job.environment
            }
        }
        
        success = await self.communication_manager.send_to_node(node_id, message)
        if success:
            job.status = TestJobStatus.RUNNING
            logging.info(f"Job {job.job_id} sent to node {node_id}")
        else:
            job.status = TestJobStatus.QUEUED
            job.assigned_node = None
            logging.error(f"Failed to send job {job.job_id} to node {node_id}")
    
    async def _monitor_jobs(self) -> None:
        """Monitor active jobs for timeouts and failures"""
        current_time = datetime.now()
        
        timed_out_jobs = []
        for job_id, job in self.active_jobs.items():
            if job.started_at and current_time - job.started_at > timedelta(seconds=job.timeout):
                timed_out_jobs.append(job)
        
        # Handle timed out jobs
        for job in timed_out_jobs:
            await self._handle_job_timeout(job)
    
    async def _handle_job_completion(self, job_id: str, result_data: Dict[str, Any]) -> None:
        """Handle job completion from worker node"""
        if job_id not in self.active_jobs:
            return
        
        job = self.active_jobs[job_id]
        job.completed_at = datetime.now()
        job.result = result_data
        
        if result_data.get("status") == "success":
            job.status = TestJobStatus.COMPLETED
        else:
            job.status = TestJobStatus.FAILED
            job.error_message = result_data.get("error", "Unknown error")
        
        # Move to completed jobs
        self.completed_jobs.append(job)
        del self.active_jobs[job_id]
        
        # Update node job count
        if job.assigned_node and job.assigned_node in self.load_balancer.nodes:
            self.load_balancer.nodes[job.assigned_node].current_jobs -= 1
        
        logging.info(f"Job {job_id} completed with status {job.status.value}")
    
    async def _handle_job_timeout(self, job: TestJob) -> None:
        """Handle job timeout"""
        logging.warning(f"Job {job.job_id} timed out on node {job.assigned_node}")
        
        # Apply fault tolerance strategy
        strategy = self.fault_tolerance.handle_job_failure(job, "Timeout")
        
        if strategy == "retry":
            retry_job = self.fault_tolerance._retry_job(job)
            self.job_queue.append(retry_job)
        
        # Remove from active jobs
        del self.active_jobs[job.job_id]
        
        # Update node job count
        if job.assigned_node and job.assigned_node in self.load_balancer.nodes:
            self.load_balancer.nodes[job.assigned_node].current_jobs -= 1
    
    async def _update_node_statuses(self) -> None:
        """Update node statuses based on heartbeats"""
        current_time = datetime.now()
        heartbeat_timeout = timedelta(minutes=2)
        
        for node_id, node in self.load_balancer.nodes.items():
            if node.last_heartbeat and current_time - node.last_heartbeat > heartbeat_timeout:
                if node.status != NodeStatus.OFFLINE:
                    logging.warning(f"Node {node_id} appears to be offline")
                    node.status = NodeStatus.OFFLINE
                    
                    # Handle node failure
                    affected_jobs = self.fault_tolerance.handle_node_failure(node_id)
                    for job in affected_jobs:
                        self.job_queue.append(job)
                        if job.job_id in self.active_jobs:
                            del self.active_jobs[job.job_id]
    
    async def _handle_node_status_update(self, node_id: str, status_data: Dict[str, Any]) -> None:
        """Handle node status update"""
        if node_id in self.load_balancer.nodes:
            node = self.load_balancer.nodes[node_id]
            new_status = NodeStatus(status_data.get("status", "ready"))
            node.status = new_status
            
            if "current_jobs" in status_data:
                node.current_jobs = status_data["current_jobs"]
    
    def submit_job(self, job: TestJob) -> bool:
        """Submit job for distributed execution"""
        # Validate job
        if not job.job_id or not job.repository:
            return False
        
        # Add to queue
        self.job_queue.append(job)
        logging.info(f"Job {job.job_id} submitted for execution")
        return True
    
    def submit_execution_plan(self, plan: ExecutionPlan, jobs: List[TestJob]) -> bool:
        """Submit execution plan with multiple jobs"""
        # Validate dependencies
        issues = self.dependency_resolver.validate_dependencies(jobs)
        if issues:
            logging.error(f"Execution plan validation failed: {issues}")
            return False
        
        # Store plan
        self.execution_plans[plan.plan_id] = plan
        
        # Submit all jobs
        for job in jobs:
            self.submit_job(job)
        
        return True
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status"""
        return {
            "coordinator_id": self.coordinator_id,
            "is_running": self.is_running,
            "registered_nodes": len(self.load_balancer.nodes),
            "queued_jobs": len(self.job_queue),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "execution_plans": len(self.execution_plans),
            "node_statuses": {
                node_id: {
                    "status": node.status.value,
                    "current_jobs": node.current_jobs,
                    "max_jobs": node.max_concurrent_jobs
                }
                for node_id, node in self.load_balancer.nodes.items()
            }
        }
    
    def stop_coordination(self) -> None:
        """Stop the coordination system"""
        self.is_running = False
        self.communication_manager.stop_server()


# Comprehensive Test Suite
class TestDistributedTestCoordinator(unittest.TestCase):
    
    def setUp(self):
        self.coordinator = DistributedTestCoordinator(8766)  # Use different port for testing
        
    def test_coordinator_initialization(self):
        """Test coordinator initialization"""
        self.assertIsNotNone(self.coordinator.load_balancer)
        self.assertIsNotNone(self.coordinator.dependency_resolver)
        self.assertIsNotNone(self.coordinator.communication_manager)
        self.assertIsNotNone(self.coordinator.fault_tolerance)
        
    def test_node_registration(self):
        """Test node registration"""
        node_info = NodeInfo(
            node_id="worker_001",
            node_type=NodeType.WORKER,
            host="localhost",
            port=8080,
            status=NodeStatus.READY,
            capabilities=["python", "javascript"],
            max_concurrent_jobs=4
        )
        
        success = self.coordinator.load_balancer.register_node(node_info)
        self.assertTrue(success)
        self.assertIn("worker_001", self.coordinator.load_balancer.nodes)
        
    def test_job_submission(self):
        """Test job submission"""
        job = TestJob(
            job_id="test_job_001",
            repository="test_repo",
            test_suite="unit_tests",
            test_command="pytest tests/",
            priority=Priority.NORMAL
        )
        
        success = self.coordinator.submit_job(job)
        self.assertTrue(success)
        self.assertEqual(len(self.coordinator.job_queue), 1)
        
    def test_dependency_resolution(self):
        """Test job dependency resolution"""
        job1 = TestJob(job_id="job_1", repository="repo", test_suite="suite", test_command="cmd", priority=Priority.NORMAL)
        job2 = TestJob(job_id="job_2", repository="repo", test_suite="suite", test_command="cmd", priority=Priority.NORMAL, dependencies=["job_1"])
        job3 = TestJob(job_id="job_3", repository="repo", test_suite="suite", test_command="cmd", priority=Priority.NORMAL, dependencies=["job_2"])
        
        jobs = [job1, job2, job3]
        execution_order = self.coordinator.dependency_resolver.resolve_execution_order(jobs)
        
        self.assertEqual(len(execution_order), 3)  # 3 levels
        self.assertIn("job_1", execution_order[0])
        self.assertIn("job_2", execution_order[1])
        self.assertIn("job_3", execution_order[2])
        
    def test_load_balancing(self):
        """Test load balancing functionality"""
        # Register multiple nodes
        for i in range(3):
            node = NodeInfo(
                node_id=f"worker_{i:03d}",
                node_type=NodeType.WORKER,
                host="localhost",
                port=8080 + i,
                status=NodeStatus.READY,
                capabilities=["python"],
                max_concurrent_jobs=2
            )
            self.coordinator.load_balancer.register_node(node)
        
        # Create test job
        job = TestJob(
            job_id="load_test_job",
            repository="test_repo",
            test_suite="tests",
            test_command="pytest",
            priority=Priority.NORMAL
        )
        
        # Find best node
        best_node = self.coordinator.load_balancer.find_best_node(job)
        self.assertIsNotNone(best_node)
        self.assertIn(best_node, self.coordinator.load_balancer.nodes)
        
    def test_fault_tolerance(self):
        """Test fault tolerance mechanisms"""
        job = TestJob(
            job_id="fault_test_job",
            repository="test_repo",
            test_suite="tests",
            test_command="pytest",
            priority=Priority.HIGH,
            max_retries=2
        )
        
        # Simulate job failure
        strategy = self.coordinator.fault_tolerance.handle_job_failure(job, "Node crashed")
        self.assertEqual(strategy, "retry")
        self.assertEqual(job.status, TestJobStatus.FAILED)
        
        # Test retry
        retry_job = self.coordinator.fault_tolerance._retry_job(job)
        self.assertEqual(retry_job.status, TestJobStatus.RETRYING)
        self.assertEqual(retry_job.retry_count, 1)
        
    def test_execution_plan_validation(self):
        """Test execution plan validation"""
        job1 = TestJob(job_id="plan_job_1", repository="repo", test_suite="suite", test_command="cmd", priority=Priority.NORMAL)
        job2 = TestJob(job_id="plan_job_2", repository="repo", test_suite="suite", test_command="cmd", priority=Priority.NORMAL, dependencies=["plan_job_1"])
        
        jobs = [job1, job2]
        issues = self.coordinator.dependency_resolver.validate_dependencies(jobs)
        
        self.assertEqual(len(issues), 0)  # No validation issues
        
        # Test with invalid dependency
        job3 = TestJob(job_id="plan_job_3", repository="repo", test_suite="suite", test_command="cmd", priority=Priority.NORMAL, dependencies=["nonexistent"])
        jobs_with_invalid = [job1, job2, job3]
        issues_invalid = self.coordinator.dependency_resolver.validate_dependencies(jobs_with_invalid)
        
        self.assertGreater(len(issues_invalid), 0)  # Should have validation issues


if __name__ == "__main__":
    # Demo usage
    async def demo_coordinator():
        coordinator = DistributedTestCoordinator()
        
        # Register worker nodes
        for i in range(3):
            node = NodeInfo(
                node_id=f"worker_{i:03d}",
                node_type=NodeType.WORKER,
                host=f"worker-{i}.testmaster.local",
                port=8080,
                status=NodeStatus.READY,
                capabilities=["python", "javascript", "docker"],
                max_concurrent_jobs=4
            )
            coordinator.load_balancer.register_node(node)
        
        # Create test jobs
        jobs = []
        for i in range(10):
            job = TestJob(
                job_id=f"distributed_job_{i:03d}",
                repository=f"repo_{i % 3}",  # Distribute across 3 repos
                test_suite="comprehensive_tests",
                test_command="pytest tests/ --cov --junit-xml=results.xml",
                priority=Priority.NORMAL if i % 3 != 0 else Priority.HIGH,
                timeout=1800,  # 30 minutes
                dependencies=[f"distributed_job_{i-1:03d}"] if i > 0 and i % 3 == 0 else []
            )
            jobs.append(job)
        
        # Create execution plan
        plan = ExecutionPlan(
            plan_id="demo_execution_plan",
            repositories=["repo_0", "repo_1", "repo_2"],
            total_jobs=len(jobs),
            dependency_graph={job.job_id: job.dependencies for job in jobs},
            estimated_duration=3600  # 1 hour
        )
        
        # Submit execution plan
        success = coordinator.submit_execution_plan(plan, jobs)
        print(f"Execution plan submitted: {success}")
        
        # Get coordination status
        status = coordinator.get_coordination_status()
        print(f"Coordination Status: {json.dumps(status, indent=2)}")
        
        print("Distributed Test Coordinator Demo Complete")
    
    # Run demo
    import asyncio
    try:
        asyncio.run(demo_coordinator())
    except KeyboardInterrupt:
        print("Demo interrupted")
    
    # Run tests
    pytest.main([__file__, "-v"])