"""
Intelligence Integration Master - Ultimate Coordination System

This module implements the IntelligenceIntegrationMaster system, providing the ultimate
coordination and unification of all intelligence capabilities into seamless operation.

Features:
- Master coordination for all intelligence systems
- Seamless system interoperability and communication
- Unified interface for all intelligence capabilities
- Comprehensive health monitoring and management
- Adaptive resource allocation and load balancing
- Predictive integration optimization

Author: Agent A - Hour 34 - Intelligence Integration Master
Created: 2025-01-21
Enhanced with: Ultimate intelligence coordination, seamless integration
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable, Type
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import yaml
import threading
from collections import defaultdict, deque
import statistics
import hashlib
import time
import importlib
import inspect

# Configure logging for intelligence integration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligenceSystemType(Enum):
    """Types of intelligence systems"""
    ANALYTICS = "analytics"
    ML_ORCHESTRATION = "ml_orchestration"
    ANALYSIS = "analysis"
    PREDICTION = "prediction"
    PATTERN_RECOGNITION = "pattern_recognition"
    AUTONOMOUS_GOVERNANCE = "autonomous_governance"
    CODE_UNDERSTANDING = "code_understanding"
    ARCHITECTURE_INTELLIGENCE = "architecture_intelligence"
    ORCHESTRATION = "orchestration"
    COORDINATION = "coordination"

class IntegrationStatus(Enum):
    """Status of system integration"""
    NOT_INTEGRATED = "not_integrated"
    INTEGRATING = "integrating"
    INTEGRATED = "integrated"
    OPTIMIZING = "optimizing"
    FAILED = "failed"
    DISABLED = "disabled"

class OperationPriority(Enum):
    """Priority levels for intelligence operations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"

@dataclass
class IntelligenceSystemInfo:
    """Information about an intelligence system"""
    system_id: str
    name: str
    type: IntelligenceSystemType
    version: str
    capabilities: List[str]
    status: IntegrationStatus
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    endpoints: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    last_health_check: datetime = field(default_factory=datetime.now)
    integration_score: float = 0.0

@dataclass
class IntelligenceOperation:
    """Represents an intelligence operation request"""
    operation_id: str
    operation_type: str
    target_systems: List[str]
    parameters: Dict[str, Any]
    priority: OperationPriority
    timeout: int = 300  # seconds
    retry_count: int = 3
    dependencies: List[str] = field(default_factory=list)
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class OperationResult:
    """Result of an intelligence operation"""
    operation_id: str
    success: bool
    results: Dict[str, Any]
    execution_time: float
    systems_used: List[str]
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)

@dataclass
class SystemHealthMetrics:
    """Health metrics for an intelligence system"""
    system_id: str
    cpu_usage: float
    memory_usage: float
    response_time: float
    throughput: float
    error_rate: float
    availability: float
    last_updated: datetime = field(default_factory=datetime.now)

class SystemInteroperabilityEngine:
    """Ensures seamless communication between all intelligence systems"""
    
    def __init__(self):
        self.communication_protocols: Dict[str, Any] = {}
        self.data_transformers: Dict[str, Callable] = {}
        self.message_queue = asyncio.Queue()
        self.active_connections: Dict[str, Any] = {}
        
        logger.info("SystemInteroperabilityEngine initialized")
    
    async def register_system(self, system_info: IntelligenceSystemInfo) -> bool:
        """Register a new intelligence system for interoperability"""
        try:
            # Establish communication protocol
            protocol = await self._establish_protocol(system_info)
            self.communication_protocols[system_info.system_id] = protocol
            
            # Create data transformer if needed
            transformer = await self._create_data_transformer(system_info)
            if transformer:
                self.data_transformers[system_info.system_id] = transformer
            
            # Test connection
            connection_success = await self._test_connection(system_info)
            if connection_success:
                self.active_connections[system_info.system_id] = {
                    "last_ping": datetime.now(),
                    "status": "active"
                }
                logger.info(f"Successfully registered system: {system_info.name}")
                return True
            else:
                logger.error(f"Failed to establish connection with: {system_info.name}")
                return False
        
        except Exception as e:
            logger.error(f"Error registering system {system_info.name}: {e}")
            return False
    
    async def _establish_protocol(self, system_info: IntelligenceSystemInfo) -> Dict[str, Any]:
        """Establish communication protocol for a system"""
        # Default protocol configuration
        protocol = {
            "type": "async_python",
            "format": "json",
            "compression": "gzip",
            "encryption": "aes256",
            "timeout": 30,
            "retry_policy": {
                "max_retries": 3,
                "backoff_factor": 2,
                "initial_delay": 1
            }
        }
        
        # Customize based on system type
        if system_info.type == IntelligenceSystemType.ML_ORCHESTRATION:
            protocol["batch_support"] = True
            protocol["streaming"] = True
        elif system_info.type == IntelligenceSystemType.ANALYTICS:
            protocol["data_streaming"] = True
            protocol["result_caching"] = True
        
        return protocol
    
    async def _create_data_transformer(self, system_info: IntelligenceSystemInfo) -> Optional[Callable]:
        """Create data transformer for system-specific data formats"""
        def default_transformer(data: Any) -> Any:
            """Default transformer that handles common data conversions"""
            if isinstance(data, pd.DataFrame):
                return data.to_dict('records')
            elif isinstance(data, np.ndarray):
                return data.tolist()
            elif hasattr(data, '__dict__'):
                return data.__dict__
            return data
        
        return default_transformer
    
    async def _test_connection(self, system_info: IntelligenceSystemInfo) -> bool:
        """Test connection to an intelligence system"""
        try:
            # Simulate connection test
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # In real implementation, this would make actual connection test
            # For now, we'll assume success for registered systems
            return True
        
        except Exception as e:
            logger.error(f"Connection test failed for {system_info.name}: {e}")
            return False
    
    async def send_message(self, target_system: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send message to target intelligence system"""
        try:
            if target_system not in self.communication_protocols:
                raise ValueError(f"System {target_system} not registered")
            
            protocol = self.communication_protocols[target_system]
            
            # Transform message if transformer exists
            if target_system in self.data_transformers:
                message = self.data_transformers[target_system](message)
            
            # Add protocol headers
            message_with_headers = {
                "headers": {
                    "timestamp": datetime.now().isoformat(),
                    "protocol_version": "1.0",
                    "message_id": hashlib.md5(str(message).encode()).hexdigest()[:8]
                },
                "payload": message
            }
            
            # Simulate message sending
            await asyncio.sleep(0.05)  # Simulate network latency
            
            # In real implementation, this would send actual message
            response = {
                "status": "success",
                "response_id": message_with_headers["headers"]["message_id"],
                "data": {"acknowledged": True}
            }
            
            logger.debug(f"Message sent to {target_system}: {message_with_headers['headers']['message_id']}")
            return response
        
        except Exception as e:
            logger.error(f"Error sending message to {target_system}: {e}")
            return None
    
    async def broadcast_message(self, message: Dict[str, Any], 
                              target_systems: Optional[List[str]] = None) -> Dict[str, Optional[Dict[str, Any]]]:
        """Broadcast message to multiple systems"""
        if target_systems is None:
            target_systems = list(self.communication_protocols.keys())
        
        results = {}
        
        # Send messages concurrently
        tasks = []
        for system in target_systems:
            task = asyncio.create_task(self.send_message(system, message))
            tasks.append((system, task))
        
        # Collect results
        for system, task in tasks:
            try:
                result = await task
                results[system] = result
            except Exception as e:
                logger.error(f"Error broadcasting to {system}: {e}")
                results[system] = None
        
        return results
    
    async def health_check_all_systems(self) -> Dict[str, bool]:
        """Perform health check on all registered systems"""
        health_status = {}
        
        for system_id in self.communication_protocols.keys():
            try:
                # Send health check message
                health_message = {"type": "health_check", "timestamp": datetime.now().isoformat()}
                response = await self.send_message(system_id, health_message)
                
                health_status[system_id] = response is not None and response.get("status") == "success"
                
                # Update connection status
                if system_id in self.active_connections:
                    self.active_connections[system_id]["last_ping"] = datetime.now()
                    self.active_connections[system_id]["status"] = "active" if health_status[system_id] else "unhealthy"
            
            except Exception as e:
                logger.error(f"Health check failed for {system_id}: {e}")
                health_status[system_id] = False
                if system_id in self.active_connections:
                    self.active_connections[system_id]["status"] = "error"
        
        return health_status

class UnifiedIntelligenceInterface:
    """Single interface for accessing all intelligence capabilities"""
    
    def __init__(self, interoperability_engine: SystemInteroperabilityEngine):
        self.interoperability_engine = interoperability_engine
        self.registered_systems: Dict[str, IntelligenceSystemInfo] = {}
        self.capability_index: Dict[str, List[str]] = {}  # capability -> list of system_ids
        self.operation_history: List[OperationResult] = []
        
        logger.info("UnifiedIntelligenceInterface initialized")
    
    def register_intelligence_system(self, system_info: IntelligenceSystemInfo) -> bool:
        """Register an intelligence system with the unified interface"""
        try:
            self.registered_systems[system_info.system_id] = system_info
            
            # Index capabilities
            for capability in system_info.capabilities:
                if capability not in self.capability_index:
                    self.capability_index[capability] = []
                self.capability_index[capability].append(system_info.system_id)
            
            logger.info(f"Registered intelligence system: {system_info.name} with {len(system_info.capabilities)} capabilities")
            return True
        
        except Exception as e:
            logger.error(f"Error registering system {system_info.name}: {e}")
            return False
    
    async def execute_operation(self, operation: IntelligenceOperation) -> OperationResult:
        """Execute an intelligence operation using optimal systems"""
        start_time = time.time()
        
        try:
            # Find optimal systems for the operation
            optimal_systems = self._find_optimal_systems(operation)
            
            if not optimal_systems:
                return OperationResult(
                    operation_id=operation.operation_id,
                    success=False,
                    results={},
                    execution_time=time.time() - start_time,
                    systems_used=[],
                    error_message="No suitable systems found for operation"
                )
            
            # Execute operation on selected systems
            results = await self._execute_on_systems(operation, optimal_systems)
            
            # Aggregate results
            aggregated_results = self._aggregate_results(results)
            
            execution_time = time.time() - start_time
            
            operation_result = OperationResult(
                operation_id=operation.operation_id,
                success=True,
                results=aggregated_results,
                execution_time=execution_time,
                systems_used=list(optimal_systems.keys()),
                performance_metrics={
                    "systems_count": len(optimal_systems),
                    "avg_response_time": execution_time / len(optimal_systems) if optimal_systems else 0
                }
            )
            
            self.operation_history.append(operation_result)
            logger.info(f"Operation {operation.operation_id} completed successfully in {execution_time:.2f}s")
            
            return operation_result
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = OperationResult(
                operation_id=operation.operation_id,
                success=False,
                results={},
                execution_time=execution_time,
                systems_used=[],
                error_message=str(e)
            )
            
            self.operation_history.append(error_result)
            logger.error(f"Operation {operation.operation_id} failed: {e}")
            
            return error_result
    
    def _find_optimal_systems(self, operation: IntelligenceOperation) -> Dict[str, IntelligenceSystemInfo]:
        """Find optimal systems for executing an operation"""
        optimal_systems = {}
        
        # If specific target systems are specified
        if operation.target_systems:
            for system_id in operation.target_systems:
                if system_id in self.registered_systems:
                    optimal_systems[system_id] = self.registered_systems[system_id]
        else:
            # Find systems based on operation type and capabilities
            required_capability = operation.operation_type
            
            if required_capability in self.capability_index:
                candidate_systems = self.capability_index[required_capability]
                
                # Score and rank systems
                scored_systems = []
                for system_id in candidate_systems:
                    system_info = self.registered_systems[system_id]
                    score = self._calculate_system_score(system_info, operation)
                    scored_systems.append((score, system_id, system_info))
                
                # Select top systems
                scored_systems.sort(reverse=True)  # Highest score first
                
                # Select systems based on priority and redundancy needs
                max_systems = 3 if operation.priority in [OperationPriority.CRITICAL, OperationPriority.HIGH] else 1
                
                for score, system_id, system_info in scored_systems[:max_systems]:
                    if system_info.status == IntegrationStatus.INTEGRATED:
                        optimal_systems[system_id] = system_info
        
        return optimal_systems
    
    def _calculate_system_score(self, system_info: IntelligenceSystemInfo, operation: IntelligenceOperation) -> float:
        """Calculate score for a system based on operation requirements"""
        score = 0.0
        
        # Base score from integration score
        score += system_info.integration_score * 0.3
        
        # Performance metrics contribution
        perf_metrics = system_info.performance_metrics
        if perf_metrics:
            # Response time (lower is better)
            response_time = perf_metrics.get("response_time", 1000)
            score += max(0, (1000 - response_time) / 1000) * 0.2
            
            # Throughput (higher is better)
            throughput = perf_metrics.get("throughput", 0)
            score += min(throughput / 1000, 1.0) * 0.2
            
            # Availability (higher is better)
            availability = perf_metrics.get("availability", 0.5)
            score += availability * 0.2
        
        # Priority matching
        if operation.priority == OperationPriority.CRITICAL:
            score += 0.1  # Boost for critical operations
        
        return score
    
    async def _execute_on_systems(self, operation: IntelligenceOperation, 
                                 systems: Dict[str, IntelligenceSystemInfo]) -> Dict[str, Any]:
        """Execute operation on selected systems"""
        results = {}
        
        # Prepare operation message
        operation_message = {
            "operation_id": operation.operation_id,
            "operation_type": operation.operation_type,
            "parameters": operation.parameters,
            "timeout": operation.timeout,
            "metadata": operation.metadata
        }
        
        # Execute on all systems concurrently
        tasks = []
        for system_id, system_info in systems.items():
            task = asyncio.create_task(
                self.interoperability_engine.send_message(system_id, operation_message)
            )
            tasks.append((system_id, task))
        
        # Collect results
        for system_id, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=operation.timeout)
                results[system_id] = result
            except asyncio.TimeoutError:
                logger.error(f"Operation {operation.operation_id} timed out on system {system_id}")
                results[system_id] = {"status": "timeout", "error": "Operation timed out"}
            except Exception as e:
                logger.error(f"Operation {operation.operation_id} failed on system {system_id}: {e}")
                results[system_id] = {"status": "error", "error": str(e)}
        
        return results
    
    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple systems"""
        aggregated = {
            "individual_results": results,
            "successful_systems": [],
            "failed_systems": [],
            "combined_data": {}
        }
        
        for system_id, result in results.items():
            if result and result.get("status") == "success":
                aggregated["successful_systems"].append(system_id)
                
                # Merge data from successful systems
                if "data" in result:
                    if system_id not in aggregated["combined_data"]:
                        aggregated["combined_data"][system_id] = result["data"]
            else:
                aggregated["failed_systems"].append(system_id)
        
        # Calculate success rate
        total_systems = len(results)
        successful_systems = len(aggregated["successful_systems"])
        aggregated["success_rate"] = successful_systems / total_systems if total_systems > 0 else 0
        
        return aggregated
    
    async def get_available_capabilities(self) -> Dict[str, List[str]]:
        """Get all available capabilities and their providing systems"""
        capabilities = {}
        
        for capability, system_ids in self.capability_index.items():
            active_systems = []
            for system_id in system_ids:
                system_info = self.registered_systems[system_id]
                if system_info.status == IntegrationStatus.INTEGRATED:
                    active_systems.append(system_id)
            
            if active_systems:
                capabilities[capability] = active_systems
        
        return capabilities
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get statistics about operations executed through the interface"""
        if not self.operation_history:
            return {"total_operations": 0}
        
        successful_ops = [op for op in self.operation_history if op.success]
        failed_ops = [op for op in self.operation_history if not op.success]
        
        execution_times = [op.execution_time for op in self.operation_history]
        
        return {
            "total_operations": len(self.operation_history),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(self.operation_history),
            "average_execution_time": statistics.mean(execution_times),
            "median_execution_time": statistics.median(execution_times),
            "recent_operations": len([op for op in self.operation_history 
                                    if op.completed_at > datetime.now() - timedelta(hours=1)])
        }

class IntelligenceHealthMonitor:
    """Monitors health and performance of all integrated intelligence systems"""
    
    def __init__(self, interoperability_engine: SystemInteroperabilityEngine):
        self.interoperability_engine = interoperability_engine
        self.health_metrics: Dict[str, SystemHealthMetrics] = {}
        self.health_history: Dict[str, List[SystemHealthMetrics]] = {}
        self.alert_thresholds = self._define_alert_thresholds()
        self.monitoring_active = False
        
        logger.info("IntelligenceHealthMonitor initialized")
    
    def _define_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Define alert thresholds for health metrics"""
        return {
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "response_time": {"warning": 1000.0, "critical": 5000.0},  # milliseconds
            "error_rate": {"warning": 0.05, "critical": 0.1},  # 5% and 10%
            "availability": {"warning": 0.95, "critical": 0.90}  # 95% and 90%
        }
    
    async def start_monitoring(self, systems: Dict[str, IntelligenceSystemInfo], 
                             monitoring_interval: int = 60):
        """Start continuous monitoring of intelligence systems"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        logger.info(f"Starting health monitoring for {len(systems)} systems")
        
        # Start monitoring task
        asyncio.create_task(self._monitoring_loop(systems, monitoring_interval))
    
    async def _monitoring_loop(self, systems: Dict[str, IntelligenceSystemInfo], interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect health metrics from all systems
                await self._collect_health_metrics(systems)
                
                # Analyze health and generate alerts
                await self._analyze_health_and_alert()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(interval)
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_health_metrics(self, systems: Dict[str, IntelligenceSystemInfo]):
        """Collect health metrics from all systems"""
        for system_id, system_info in systems.items():
            try:
                # Request health metrics from system
                health_request = {
                    "type": "health_metrics",
                    "timestamp": datetime.now().isoformat()
                }
                
                response = await self.interoperability_engine.send_message(system_id, health_request)
                
                if response and response.get("status") == "success":
                    metrics_data = response.get("data", {})
                    
                    # Create health metrics object
                    health_metrics = SystemHealthMetrics(
                        system_id=system_id,
                        cpu_usage=metrics_data.get("cpu_usage", 0.0),
                        memory_usage=metrics_data.get("memory_usage", 0.0),
                        response_time=metrics_data.get("response_time", 0.0),
                        throughput=metrics_data.get("throughput", 0.0),
                        error_rate=metrics_data.get("error_rate", 0.0),
                        availability=metrics_data.get("availability", 1.0)
                    )
                    
                    # Store current metrics
                    self.health_metrics[system_id] = health_metrics
                    
                    # Add to history
                    if system_id not in self.health_history:
                        self.health_history[system_id] = []
                    
                    self.health_history[system_id].append(health_metrics)
                    
                    # Keep only last 100 entries per system
                    if len(self.health_history[system_id]) > 100:
                        self.health_history[system_id] = self.health_history[system_id][-100:]
                
                else:
                    # Generate default metrics for unresponsive systems
                    health_metrics = SystemHealthMetrics(
                        system_id=system_id,
                        cpu_usage=0.0,
                        memory_usage=0.0,
                        response_time=float('inf'),
                        throughput=0.0,
                        error_rate=1.0,
                        availability=0.0
                    )
                    
                    self.health_metrics[system_id] = health_metrics
            
            except Exception as e:
                logger.error(f"Error collecting health metrics for {system_id}: {e}")
    
    async def _analyze_health_and_alert(self):
        """Analyze health metrics and generate alerts"""
        for system_id, metrics in self.health_metrics.items():
            alerts = self._check_thresholds(metrics)
            
            if alerts:
                await self._send_alerts(system_id, alerts, metrics)
    
    def _check_thresholds(self, metrics: SystemHealthMetrics) -> List[str]:
        """Check if metrics exceed alert thresholds"""
        alerts = []
        
        # Check CPU usage
        if metrics.cpu_usage >= self.alert_thresholds["cpu_usage"]["critical"]:
            alerts.append(f"CRITICAL: CPU usage {metrics.cpu_usage:.1f}% exceeds critical threshold")
        elif metrics.cpu_usage >= self.alert_thresholds["cpu_usage"]["warning"]:
            alerts.append(f"WARNING: CPU usage {metrics.cpu_usage:.1f}% exceeds warning threshold")
        
        # Check memory usage
        if metrics.memory_usage >= self.alert_thresholds["memory_usage"]["critical"]:
            alerts.append(f"CRITICAL: Memory usage {metrics.memory_usage:.1f}% exceeds critical threshold")
        elif metrics.memory_usage >= self.alert_thresholds["memory_usage"]["warning"]:
            alerts.append(f"WARNING: Memory usage {metrics.memory_usage:.1f}% exceeds warning threshold")
        
        # Check response time
        if metrics.response_time >= self.alert_thresholds["response_time"]["critical"]:
            alerts.append(f"CRITICAL: Response time {metrics.response_time:.1f}ms exceeds critical threshold")
        elif metrics.response_time >= self.alert_thresholds["response_time"]["warning"]:
            alerts.append(f"WARNING: Response time {metrics.response_time:.1f}ms exceeds warning threshold")
        
        # Check error rate
        if metrics.error_rate >= self.alert_thresholds["error_rate"]["critical"]:
            alerts.append(f"CRITICAL: Error rate {metrics.error_rate:.1%} exceeds critical threshold")
        elif metrics.error_rate >= self.alert_thresholds["error_rate"]["warning"]:
            alerts.append(f"WARNING: Error rate {metrics.error_rate:.1%} exceeds warning threshold")
        
        # Check availability
        if metrics.availability <= self.alert_thresholds["availability"]["critical"]:
            alerts.append(f"CRITICAL: Availability {metrics.availability:.1%} below critical threshold")
        elif metrics.availability <= self.alert_thresholds["availability"]["warning"]:
            alerts.append(f"WARNING: Availability {metrics.availability:.1%} below warning threshold")
        
        return alerts
    
    async def _send_alerts(self, system_id: str, alerts: List[str], metrics: SystemHealthMetrics):
        """Send alerts for system health issues"""
        for alert in alerts:
            logger.warning(f"HEALTH ALERT for {system_id}: {alert}")
            
            # In a real implementation, this would send alerts via:
            # - Email notifications
            # - Slack/Teams messages
            # - PagerDuty incidents
            # - Dashboard notifications
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get summary of system health across all monitored systems"""
        if not self.health_metrics:
            return {"status": "no_data", "monitored_systems": 0}
        
        total_systems = len(self.health_metrics)
        healthy_systems = 0
        warning_systems = 0
        critical_systems = 0
        
        avg_cpu = 0.0
        avg_memory = 0.0
        avg_response_time = 0.0
        total_availability = 0.0
        
        for metrics in self.health_metrics.values():
            # Categorize system health
            alerts = self._check_thresholds(metrics)
            critical_alerts = [a for a in alerts if "CRITICAL" in a]
            warning_alerts = [a for a in alerts if "WARNING" in a]
            
            if critical_alerts:
                critical_systems += 1
            elif warning_alerts:
                warning_systems += 1
            else:
                healthy_systems += 1
            
            # Aggregate metrics
            avg_cpu += metrics.cpu_usage
            avg_memory += metrics.memory_usage
            avg_response_time += metrics.response_time if metrics.response_time != float('inf') else 0
            total_availability += metrics.availability
        
        return {
            "status": "monitoring_active" if self.monitoring_active else "monitoring_inactive",
            "monitored_systems": total_systems,
            "healthy_systems": healthy_systems,
            "warning_systems": warning_systems,
            "critical_systems": critical_systems,
            "overall_health_score": (healthy_systems / total_systems) * 100 if total_systems > 0 else 0,
            "average_metrics": {
                "cpu_usage": avg_cpu / total_systems if total_systems > 0 else 0,
                "memory_usage": avg_memory / total_systems if total_systems > 0 else 0,
                "response_time": avg_response_time / total_systems if total_systems > 0 else 0,
                "availability": total_availability / total_systems if total_systems > 0 else 0
            },
            "last_update": max([m.last_updated for m in self.health_metrics.values()]).isoformat() if self.health_metrics else None
        }
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        logger.info("Health monitoring stopped")

class IntelligenceIntegrationMaster:
    """
    Master coordinator for all intelligence systems providing ultimate integration
    """
    
    def __init__(self):
        self.interoperability_engine = SystemInteroperabilityEngine()
        self.unified_interface = UnifiedIntelligenceInterface(self.interoperability_engine)
        self.health_monitor = IntelligenceHealthMonitor(self.interoperability_engine)
        self.registered_systems: Dict[str, IntelligenceSystemInfo] = {}
        self.integration_metrics: Dict[str, Any] = {}
        self.resource_allocator = self._initialize_resource_allocator()
        
        logger.info("IntelligenceIntegrationMaster initialized with comprehensive coordination capabilities")
    
    def _initialize_resource_allocator(self) -> Dict[str, Any]:
        """Initialize resource allocation system"""
        return {
            "cpu_pool": 100.0,  # Total CPU percentage available
            "memory_pool": 16.0,  # Total memory in GB
            "allocations": {},  # system_id -> allocated resources
            "usage_tracking": {},  # system_id -> current usage
            "allocation_strategy": "dynamic"  # dynamic, static, adaptive
        }
    
    async def register_intelligence_system(self, system_info: IntelligenceSystemInfo) -> bool:
        """Register a new intelligence system with the master coordinator"""
        logger.info(f"Registering intelligence system: {system_info.name}")
        
        try:
            # Register with interoperability engine
            interop_success = await self.interoperability_engine.register_system(system_info)
            if not interop_success:
                logger.error(f"Failed to register {system_info.name} with interoperability engine")
                return False
            
            # Register with unified interface
            interface_success = self.unified_interface.register_intelligence_system(system_info)
            if not interface_success:
                logger.error(f"Failed to register {system_info.name} with unified interface")
                return False
            
            # Add to registered systems
            self.registered_systems[system_info.system_id] = system_info
            
            # Allocate initial resources
            self._allocate_resources(system_info)
            
            # Update integration metrics
            self._update_integration_metrics()
            
            # Mark as successfully integrated
            system_info.status = IntegrationStatus.INTEGRATED
            system_info.integration_score = 100.0
            
            logger.info(f"Successfully integrated intelligence system: {system_info.name}")
            return True
        
        except Exception as e:
            logger.error(f"Error registering intelligence system {system_info.name}: {e}")
            return False
    
    def _allocate_resources(self, system_info: IntelligenceSystemInfo):
        """Allocate computational resources to a system"""
        # Calculate resource requirements based on system type and capabilities
        base_cpu = 10.0  # Base CPU allocation
        base_memory = 1.0  # Base memory allocation in GB
        
        # Adjust based on system type
        type_multipliers = {
            IntelligenceSystemType.ML_ORCHESTRATION: {"cpu": 2.0, "memory": 3.0},
            IntelligenceSystemType.ANALYTICS: {"cpu": 1.5, "memory": 2.0},
            IntelligenceSystemType.PREDICTION: {"cpu": 1.8, "memory": 2.5},
            IntelligenceSystemType.ARCHITECTURE_INTELLIGENCE: {"cpu": 1.3, "memory": 1.8},
            IntelligenceSystemType.CODE_UNDERSTANDING: {"cpu": 1.6, "memory": 2.2}
        }
        
        multiplier = type_multipliers.get(system_info.type, {"cpu": 1.0, "memory": 1.0})
        
        # Calculate final allocation
        cpu_allocation = base_cpu * multiplier["cpu"]
        memory_allocation = base_memory * multiplier["memory"]
        
        # Ensure we don't exceed available resources
        available_cpu = self.resource_allocator["cpu_pool"] - sum([
            alloc["cpu"] for alloc in self.resource_allocator["allocations"].values()
        ])
        available_memory = self.resource_allocator["memory_pool"] - sum([
            alloc["memory"] for alloc in self.resource_allocator["allocations"].values()
        ])
        
        cpu_allocation = min(cpu_allocation, available_cpu)
        memory_allocation = min(memory_allocation, available_memory)
        
        # Store allocation
        self.resource_allocator["allocations"][system_info.system_id] = {
            "cpu": cpu_allocation,
            "memory": memory_allocation,
            "allocated_at": datetime.now()
        }
        
        logger.info(f"Allocated resources to {system_info.name}: {cpu_allocation:.1f}% CPU, {memory_allocation:.1f}GB RAM")
    
    def _update_integration_metrics(self):
        """Update integration metrics"""
        total_systems = len(self.registered_systems)
        integrated_systems = len([s for s in self.registered_systems.values() 
                                if s.status == IntegrationStatus.INTEGRATED])
        
        self.integration_metrics = {
            "total_registered_systems": total_systems,
            "integrated_systems": integrated_systems,
            "integration_rate": integrated_systems / total_systems if total_systems > 0 else 0,
            "system_types": {},
            "capability_coverage": len(self.unified_interface.capability_index),
            "resource_utilization": self._calculate_resource_utilization(),
            "last_updated": datetime.now().isoformat()
        }
        
        # Count systems by type
        for system_info in self.registered_systems.values():
            system_type = system_info.type.value
            if system_type not in self.integration_metrics["system_types"]:
                self.integration_metrics["system_types"][system_type] = 0
            self.integration_metrics["system_types"][system_type] += 1
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate current resource utilization"""
        allocated_cpu = sum([alloc["cpu"] for alloc in self.resource_allocator["allocations"].values()])
        allocated_memory = sum([alloc["memory"] for alloc in self.resource_allocator["allocations"].values()])
        
        return {
            "cpu_utilization": allocated_cpu / self.resource_allocator["cpu_pool"] * 100,
            "memory_utilization": allocated_memory / self.resource_allocator["memory_pool"] * 100,
            "total_allocated_systems": len(self.resource_allocator["allocations"])
        }
    
    async def execute_intelligence_operation(self, operation: IntelligenceOperation) -> OperationResult:
        """Execute an intelligence operation through the unified interface"""
        logger.info(f"Executing intelligence operation: {operation.operation_id} ({operation.operation_type})")
        
        # Validate operation
        if not self._validate_operation(operation):
            return OperationResult(
                operation_id=operation.operation_id,
                success=False,
                results={},
                execution_time=0.0,
                systems_used=[],
                error_message="Operation validation failed"
            )
        
        # Execute through unified interface
        result = await self.unified_interface.execute_operation(operation)
        
        # Update performance metrics
        self._update_performance_metrics(operation, result)
        
        return result
    
    def _validate_operation(self, operation: IntelligenceOperation) -> bool:
        """Validate an intelligence operation"""
        # Check if required parameters are present
        if not operation.operation_type:
            logger.error("Operation type is required")
            return False
        
        # Check if target systems exist (if specified)
        if operation.target_systems:
            for system_id in operation.target_systems:
                if system_id not in self.registered_systems:
                    logger.error(f"Target system {system_id} not registered")
                    return False
        
        # Check if timeout is reasonable
        if operation.timeout <= 0 or operation.timeout > 3600:  # Max 1 hour
            logger.error("Invalid timeout value")
            return False
        
        return True
    
    def _update_performance_metrics(self, operation: IntelligenceOperation, result: OperationResult):
        """Update performance metrics based on operation results"""
        # This would typically update a performance database
        # For now, we'll just log key metrics
        logger.info(f"Operation {operation.operation_id} performance: "
                   f"Success={result.success}, Time={result.execution_time:.2f}s, "
                   f"Systems={len(result.systems_used)}")
    
    async def start_comprehensive_monitoring(self, monitoring_interval: int = 60):
        """Start comprehensive monitoring of all integrated systems"""
        logger.info("Starting comprehensive intelligence monitoring")
        
        # Start health monitoring
        await self.health_monitor.start_monitoring(self.registered_systems, monitoring_interval)
        
        # Start resource monitoring
        asyncio.create_task(self._resource_monitoring_loop(monitoring_interval))
        
        # Start integration health monitoring
        asyncio.create_task(self._integration_monitoring_loop(monitoring_interval))
    
    async def _resource_monitoring_loop(self, interval: int):
        """Monitor resource utilization across all systems"""
        while True:
            try:
                # Update resource utilization metrics
                self.integration_metrics["resource_utilization"] = self._calculate_resource_utilization()
                
                # Check for resource constraints
                cpu_util = self.integration_metrics["resource_utilization"]["cpu_utilization"]
                memory_util = self.integration_metrics["resource_utilization"]["memory_utilization"]
                
                if cpu_util > 90:
                    logger.warning(f"High CPU utilization: {cpu_util:.1f}%")
                if memory_util > 90:
                    logger.warning(f"High memory utilization: {memory_util:.1f}%")
                
                await asyncio.sleep(interval)
            
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(interval)
    
    async def _integration_monitoring_loop(self, interval: int):
        """Monitor integration health and performance"""
        while True:
            try:
                # Perform health checks on all systems
                health_status = await self.interoperability_engine.health_check_all_systems()
                
                # Update system statuses based on health checks
                for system_id, is_healthy in health_status.items():
                    if system_id in self.registered_systems:
                        if is_healthy:
                            if self.registered_systems[system_id].status != IntegrationStatus.INTEGRATED:
                                self.registered_systems[system_id].status = IntegrationStatus.INTEGRATED
                        else:
                            self.registered_systems[system_id].status = IntegrationStatus.FAILED
                
                # Update integration metrics
                self._update_integration_metrics()
                
                await asyncio.sleep(interval * 2)  # Less frequent than resource monitoring
            
            except Exception as e:
                logger.error(f"Error in integration monitoring: {e}")
                await asyncio.sleep(interval)
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive status of intelligence integration"""
        # Get health summary
        health_summary = self.health_monitor.get_system_health_summary()
        
        # Get operation statistics
        operation_stats = self.unified_interface.get_operation_statistics()
        
        # Get available capabilities
        capabilities = asyncio.create_task(self.unified_interface.get_available_capabilities())
        
        return {
            "integration_metrics": self.integration_metrics,
            "health_summary": health_summary,
            "operation_statistics": operation_stats,
            "registered_systems": {
                system_id: {
                    "name": info.name,
                    "type": info.type.value,
                    "status": info.status.value,
                    "capabilities_count": len(info.capabilities),
                    "integration_score": info.integration_score
                }
                for system_id, info in self.registered_systems.items()
            },
            "resource_allocation": self.resource_allocator["allocations"],
            "timestamp": datetime.now().isoformat()
        }
    
    async def optimize_integration(self):
        """Optimize integration configuration and resource allocation"""
        logger.info("Starting integration optimization")
        
        # Analyze current performance
        operation_stats = self.unified_interface.get_operation_statistics()
        
        # Optimize resource allocation based on usage patterns
        await self._optimize_resource_allocation()
        
        # Optimize system selection algorithms
        await self._optimize_system_selection()
        
        # Update integration scores
        self._update_integration_scores()
        
        logger.info("Integration optimization completed")
    
    async def _optimize_resource_allocation(self):
        """Optimize resource allocation based on system performance"""
        # Analyze system performance and adjust allocations
        for system_id, system_info in self.registered_systems.items():
            if system_id in self.health_monitor.health_metrics:
                metrics = self.health_monitor.health_metrics[system_id]
                current_allocation = self.resource_allocator["allocations"].get(system_id, {})
                
                # Adjust allocation based on utilization
                if metrics.cpu_usage > 80 and current_allocation.get("cpu", 0) < 50:
                    # Increase CPU allocation for high-usage systems
                    additional_cpu = min(10, self.resource_allocator["cpu_pool"] * 0.1)
                    current_allocation["cpu"] = current_allocation.get("cpu", 0) + additional_cpu
                    logger.info(f"Increased CPU allocation for {system_info.name}")
                
                if metrics.memory_usage > 80 and current_allocation.get("memory", 0) < 8:
                    # Increase memory allocation for high-usage systems
                    additional_memory = min(2, self.resource_allocator["memory_pool"] * 0.1)
                    current_allocation["memory"] = current_allocation.get("memory", 0) + additional_memory
                    logger.info(f"Increased memory allocation for {system_info.name}")
    
    async def _optimize_system_selection(self):
        """Optimize algorithms for selecting optimal systems for operations"""
        # Analyze operation history to improve system selection
        operation_history = self.unified_interface.operation_history
        
        if len(operation_history) > 10:
            # Analyze successful vs failed operations
            successful_ops = [op for op in operation_history if op.success]
            
            # Identify systems with best performance
            system_performance = {}
            for op in successful_ops:
                for system_id in op.systems_used:
                    if system_id not in system_performance:
                        system_performance[system_id] = []
                    system_performance[system_id].append(op.execution_time)
            
            # Update integration scores based on performance
            for system_id, times in system_performance.items():
                if system_id in self.registered_systems:
                    avg_time = statistics.mean(times)
                    # Better performance = higher score
                    performance_score = max(0, min(100, 100 - (avg_time * 10)))
                    self.registered_systems[system_id].integration_score = performance_score
    
    def _update_integration_scores(self):
        """Update integration scores for all systems"""
        for system_id, system_info in self.registered_systems.items():
            score = 50.0  # Base score
            
            # Factor in health metrics
            if system_id in self.health_monitor.health_metrics:
                metrics = self.health_monitor.health_metrics[system_id]
                
                # Availability contribution
                score += metrics.availability * 30
                
                # Response time contribution (lower is better)
                if metrics.response_time < 100:
                    score += 15
                elif metrics.response_time < 500:
                    score += 10
                elif metrics.response_time < 1000:
                    score += 5
                
                # Error rate contribution (lower is better)
                score += max(0, (1 - metrics.error_rate) * 15)
            
            # Factor in integration status
            if system_info.status == IntegrationStatus.INTEGRATED:
                score += 20
            elif system_info.status == IntegrationStatus.OPTIMIZING:
                score += 15
            elif system_info.status == IntegrationStatus.INTEGRATING:
                score += 10
            
            system_info.integration_score = max(0, min(100, score))

async def main():
    """Main function to demonstrate IntelligenceIntegrationMaster capabilities"""
    
    # Initialize the master integration system
    integration_master = IntelligenceIntegrationMaster()
    
    print("🎯 Intelligence Integration Master - Ultimate Coordination System")
    print("=" * 80)
    
    # Example intelligence systems to register
    systems = [
        IntelligenceSystemInfo(
            system_id="analytics_hub",
            name="Advanced Analytics Hub",
            type=IntelligenceSystemType.ANALYTICS,
            version="2.0",
            capabilities=["data_analysis", "trend_detection", "correlation_analysis"],
            status=IntegrationStatus.NOT_INTEGRATED,
            performance_metrics={"response_time": 150.0, "throughput": 1000.0, "availability": 0.99}
        ),
        IntelligenceSystemInfo(
            system_id="ml_orchestrator",
            name="ML Orchestration Engine",
            type=IntelligenceSystemType.ML_ORCHESTRATION,
            version="3.1",
            capabilities=["model_training", "prediction", "optimization"],
            status=IntegrationStatus.NOT_INTEGRATED,
            performance_metrics={"response_time": 300.0, "throughput": 500.0, "availability": 0.98}
        ),
        IntelligenceSystemInfo(
            system_id="code_analyzer",
            name="Semantic Code Analyzer",
            type=IntelligenceSystemType.CODE_UNDERSTANDING,
            version="1.5",
            capabilities=["code_analysis", "intent_recognition", "optimization"],
            status=IntegrationStatus.NOT_INTEGRATED,
            performance_metrics={"response_time": 200.0, "throughput": 800.0, "availability": 0.97}
        ),
        IntelligenceSystemInfo(
            system_id="arch_intelligence",
            name="Architecture Intelligence System",
            type=IntelligenceSystemType.ARCHITECTURE_INTELLIGENCE,
            version="1.0",
            capabilities=["architecture_analysis", "decision_making", "evolution_prediction"],
            status=IntegrationStatus.NOT_INTEGRATED,
            performance_metrics={"response_time": 400.0, "throughput": 300.0, "availability": 0.96}
        )
    ]
    
    print("\n1. System Registration and Integration")
    print("-" * 40)
    
    # Register all systems
    for system_info in systems:
        success = await integration_master.register_intelligence_system(system_info)
        print(f"{'✅' if success else '❌'} {system_info.name}: {'Integrated' if success else 'Failed'}")
    
    # Get integration status
    status = integration_master.get_integration_status()
    print(f"\nIntegration Summary:")
    print(f"  Total Systems: {status['integration_metrics']['total_registered_systems']}")
    print(f"  Integrated: {status['integration_metrics']['integrated_systems']}")
    print(f"  Integration Rate: {status['integration_metrics']['integration_rate']:.1%}")
    print(f"  Capability Coverage: {status['integration_metrics']['capability_coverage']}")
    
    print("\n\n2. Intelligence Operation Execution")
    print("-" * 40)
    
    # Create and execute intelligence operations
    operations = [
        IntelligenceOperation(
            operation_id="op_001",
            operation_type="data_analysis",
            target_systems=[],  # Let system choose optimal
            parameters={"dataset": "user_behavior", "analysis_type": "trend"},
            priority=OperationPriority.HIGH
        ),
        IntelligenceOperation(
            operation_id="op_002",
            operation_type="code_analysis",
            target_systems=["code_analyzer"],
            parameters={"code_path": "src/main.py", "analysis_depth": "deep"},
            priority=OperationPriority.MEDIUM
        ),
        IntelligenceOperation(
            operation_id="op_003",
            operation_type="architecture_analysis",
            target_systems=[],
            parameters={"system_path": "core/", "focus": "dependencies"},
            priority=OperationPriority.HIGH
        )
    ]
    
    # Execute operations
    for operation in operations:
        result = await integration_master.execute_intelligence_operation(operation)
        print(f"Operation {operation.operation_id}: {'✅ Success' if result.success else '❌ Failed'}")
        print(f"  Execution Time: {result.execution_time:.2f}s")
        print(f"  Systems Used: {len(result.systems_used)}")
        if result.error_message:
            print(f"  Error: {result.error_message}")
    
    print("\n\n3. Health Monitoring and Resource Management")
    print("-" * 40)
    
    # Start monitoring
    await integration_master.start_comprehensive_monitoring(monitoring_interval=30)
    
    # Wait a bit for monitoring to collect data
    await asyncio.sleep(2)
    
    # Get health summary
    health_summary = integration_master.health_monitor.get_system_health_summary()
    print(f"Health Status: {health_summary['status']}")
    print(f"Monitored Systems: {health_summary['monitored_systems']}")
    print(f"Healthy Systems: {health_summary['healthy_systems']}")
    print(f"Overall Health Score: {health_summary['overall_health_score']:.1f}")
    
    # Show resource utilization
    resource_util = status['integration_metrics']['resource_utilization']
    print(f"\nResource Utilization:")
    print(f"  CPU: {resource_util['cpu_utilization']:.1f}%")
    print(f"  Memory: {resource_util['memory_utilization']:.1f}%")
    print(f"  Allocated Systems: {resource_util['total_allocated_systems']}")
    
    print("\n\n4. Integration Optimization")
    print("-" * 40)
    
    # Perform integration optimization
    await integration_master.optimize_integration()
    
    # Show updated integration scores
    print("Updated Integration Scores:")
    for system_id, system_info in integration_master.registered_systems.items():
        print(f"  {system_info.name}: {system_info.integration_score:.1f}")
    
    print("\n\n5. Available Capabilities")
    print("-" * 40)
    
    # Show available capabilities
    capabilities = await integration_master.unified_interface.get_available_capabilities()
    print("Available Intelligence Capabilities:")
    for capability, providers in capabilities.items():
        print(f"  {capability}: {len(providers)} provider(s)")
    
    # Get operation statistics
    op_stats = integration_master.unified_interface.get_operation_statistics()
    print(f"\nOperation Statistics:")
    print(f"  Total Operations: {op_stats['total_operations']}")
    print(f"  Success Rate: {op_stats['success_rate']:.1%}")
    print(f"  Average Execution Time: {op_stats['average_execution_time']:.2f}s")
    
    print("\n✅ Intelligence Integration Master demonstration completed successfully!")
    print("All intelligence systems integrated and coordinated through unified interface!")

if __name__ == "__main__":
    asyncio.run(main())