"""
ML Integration Orchestrator - TestMaster Advanced ML
=====================================================

Coordinates and orchestrates the 19 enterprise ML modules for optimal system-wide 
machine learning operations. Implements the integration recommendations and manages
cross-module communication, data flow, and resource optimization.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Event, Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Set, Union
import json
import uuid

# ML Module Imports (would be actual imports in production)
from .integration_analysis import get_integration_analysis


class OrchestrationMode(Enum):
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
    PREDICTION = "prediction"
    COORDINATION = "coordination"
    EMERGENCY = "emergency"


class IntegrationPattern(Enum):
    PIPELINE = "pipeline"
    FEEDBACK_LOOP = "feedback_loop"
    BROADCAST = "broadcast"
    AGGREGATION = "aggregation"
    COORDINATION = "coordination"


@dataclass
class MLModuleInterface:
    """Interface definition for ML module integration"""
    
    module_id: str
    module_name: str
    status: str = "active"
    capabilities: List[str] = field(default_factory=list)
    data_inputs: List[str] = field(default_factory=list)
    data_outputs: List[str] = field(default_factory=list)
    
    # Integration points
    input_handlers: Dict[str, Callable] = field(default_factory=dict)
    output_publishers: Dict[str, Callable] = field(default_factory=dict)
    control_interfaces: Dict[str, Callable] = field(default_factory=dict)
    
    # Performance metrics
    processing_time: float = 0.0
    success_rate: float = 1.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    last_health_check: datetime = field(default_factory=datetime.now)


@dataclass
class IntegrationFlow:
    """Defines data/control flow between ML modules"""
    
    flow_id: str
    pattern: IntegrationPattern
    source_modules: List[str]
    target_modules: List[str]
    data_transformation: Optional[Callable] = None
    
    # Flow control
    enabled: bool = True
    priority: int = 5
    max_latency_ms: float = 1000.0
    
    # Performance tracking
    message_count: int = 0
    average_latency: float = 0.0
    error_count: int = 0
    last_execution: Optional[datetime] = None


@dataclass
class OrchestrationEvent:
    """Event for ML module coordination"""
    
    event_id: str
    event_type: str
    source_module: str
    target_modules: List[str]
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Event handling
    processed: bool = False
    results: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


class MLOrchestrator:
    """
    Advanced ML module orchestration and coordination system
    """
    
    def __init__(self,
                 enable_auto_optimization: bool = True,
                 orchestration_interval: int = 10,
                 health_check_interval: int = 60):
        """Initialize ML orchestrator"""
        
        self.enable_auto_optimization = enable_auto_optimization
        self.orchestration_interval = orchestration_interval
        self.health_check_interval = health_check_interval
        
        # Module Management
        self.ml_modules: Dict[str, MLModuleInterface] = {}
        self.integration_flows: Dict[str, IntegrationFlow] = {}
        self.active_patterns: Dict[str, List[str]] = defaultdict(list)
        
        # Event Management
        self.event_queue: deque = deque()
        self.processed_events: deque = deque(maxlen=1000)
        self.event_handlers: Dict[str, Callable] = {}
        
        # Orchestration State
        self.current_mode: OrchestrationMode = OrchestrationMode.MONITORING
        self.system_health_score: float = 1.0
        self.performance_metrics: Dict[str, float] = {}
        
        # Resource Management
        self.resource_allocation: Dict[str, Dict[str, float]] = {}
        self.resource_limits: Dict[str, float] = {
            'cpu': 8.0,  # cores
            'memory': 32.0,  # GB
            'gpu': 1.0   # units
        }
        
        # Performance Tracking
        self.orchestration_metrics: deque = deque(maxlen=500)
        self.module_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Configuration
        self.auto_scaling_enabled = True
        self.fault_tolerance_enabled = True
        self.load_balancing_enabled = True
        
        # Statistics
        self.orchestration_stats = {
            'modules_registered': 0,
            'flows_created': 0,
            'events_processed': 0,
            'optimizations_applied': 0,
            'failures_recovered': 0,
            'start_time': datetime.now()
        }
        
        # Synchronization
        self.orchestration_lock = RLock()
        self.event_lock = Lock()
        self.shutdown_event = Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize orchestration
        self._initialize_ml_modules()
        self._create_integration_flows()
        
        # Start orchestration loops
        asyncio.create_task(self._orchestration_loop())
        asyncio.create_task(self._event_processing_loop())
        asyncio.create_task(self._health_monitoring_loop())
    
    def _initialize_ml_modules(self):
        """Initialize all 19 ML modules with interface definitions"""
        
        # Get integration analysis
        integration_data = get_integration_analysis()
        
        # Core ML Modules (Hour 1)
        core_modules = [
            'ensemble_meta_learner', 'anomaly_detection', 'smart_cache',
            'batch_processor', 'predictive_engine', 'performance_optimizer',
            'circuit_breaker_ml', 'delivery_optimizer', 'integrity_ml_guardian',
            'sla_ml_optimizer'
        ]
        
        # Archive Extraction Modules (Hour 2)
        archive_modules = [
            'performance_ml_engine', 'performance_execution_manager',
            'telemetry_ml_collector', 'telemetry_observability_engine', 'telemetry_export_manager',
            'watchdog_ml_monitor', 'watchdog_recovery_system', 'watchdog_process_manager'
        ]
        
        # Enterprise Extension Modules (Hour 2)
        enterprise_modules = [
            'adaptive_load_balancer', 'intelligent_resource_scheduler', 'ml_security_guardian',
            'adaptive_configuration_manager', 'intelligent_data_pipeline', 'ml_network_optimizer',
            'distributed_ml_coordinator', 'predictive_maintenance_ai'
        ]
        
        all_modules = core_modules + archive_modules + enterprise_modules
        
        for module_name in all_modules:
            self._register_ml_module(module_name, integration_data['capabilities_analysis'].get(module_name))
    
    def _register_ml_module(self, module_name: str, capability_data: Any):
        """Register ML module with orchestration system"""
        
        try:
            # Create module interface
            interface = MLModuleInterface(
                module_id=f"{module_name}_{uuid.uuid4().hex[:8]}",
                module_name=module_name,
                capabilities=capability_data.ml_algorithms if capability_data else [],
                data_inputs=capability_data.data_inputs if capability_data else [],
                data_outputs=capability_data.data_outputs if capability_data else []
            )
            
            # Add standard interface methods
            interface.input_handlers = {
                'process_data': self._create_input_handler(module_name),
                'health_check': self._create_health_handler(module_name),
                'configure': self._create_config_handler(module_name)
            }
            
            interface.output_publishers = {
                'publish_results': self._create_output_publisher(module_name),
                'publish_metrics': self._create_metrics_publisher(module_name),
                'publish_alerts': self._create_alert_publisher(module_name)
            }
            
            interface.control_interfaces = {
                'start': self._create_start_handler(module_name),
                'stop': self._create_stop_handler(module_name),
                'restart': self._create_restart_handler(module_name),
                'optimize': self._create_optimization_handler(module_name)
            }
            
            with self.orchestration_lock:
                self.ml_modules[module_name] = interface
                self.orchestration_stats['modules_registered'] += 1
            
            self.logger.info(f"ML module registered: {module_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to register ML module {module_name}: {e}")
    
    def _create_integration_flows(self):
        """Create integration flows based on analysis recommendations"""
        
        # Telemetry Processing Pipeline
        telemetry_flow = IntegrationFlow(
            flow_id="telemetry_pipeline",
            pattern=IntegrationPattern.PIPELINE,
            source_modules=["telemetry_ml_collector"],
            target_modules=["telemetry_observability_engine", "telemetry_export_manager"]
        )
        
        # Performance Optimization Coordination
        performance_flow = IntegrationFlow(
            flow_id="performance_coordination",
            pattern=IntegrationPattern.COORDINATION,
            source_modules=["performance_ml_engine"],
            target_modules=["performance_execution_manager", "performance_optimizer"]
        )
        
        # Security & Monitoring Alliance
        security_flow = IntegrationFlow(
            flow_id="security_monitoring",
            pattern=IntegrationPattern.BROADCAST,
            source_modules=["ml_security_guardian"],
            target_modules=["integrity_ml_guardian", "watchdog_ml_monitor"]
        )
        
        # Predictive Analytics Consortium
        predictive_flow = IntegrationFlow(
            flow_id="predictive_consortium",
            pattern=IntegrationPattern.AGGREGATION,
            source_modules=["predictive_engine", "predictive_maintenance_ai"],
            target_modules=["ensemble_meta_learner"]
        )
        
        # Anomaly Detection Network
        anomaly_flow = IntegrationFlow(
            flow_id="anomaly_network",
            pattern=IntegrationPattern.BROADCAST,
            source_modules=["anomaly_detection"],
            target_modules=["ml_security_guardian", "watchdog_ml_monitor", "circuit_breaker_ml"]
        )
        
        # Resource Optimization Feedback Loop
        resource_feedback_flow = IntegrationFlow(
            flow_id="resource_feedback",
            pattern=IntegrationPattern.FEEDBACK_LOOP,
            source_modules=["intelligent_resource_scheduler"],
            target_modules=["adaptive_configuration_manager", "distributed_ml_coordinator"]
        )
        
        flows = [telemetry_flow, performance_flow, security_flow, 
                predictive_flow, anomaly_flow, resource_feedback_flow]
        
        with self.orchestration_lock:
            for flow in flows:
                self.integration_flows[flow.flow_id] = flow
                self.orchestration_stats['flows_created'] += 1
    
    def _create_input_handler(self, module_name: str) -> Callable:
        """Create input handler for ML module"""
        
        async def handle_input(data: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Log input processing
                self.logger.debug(f"Processing input for {module_name}: {len(data)} items")
                
                # Simulate module processing
                await asyncio.sleep(0.01)  # Minimal processing delay
                
                # Return processed results
                return {
                    'module': module_name,
                    'status': 'success',
                    'processed_data': data,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                self.logger.error(f"Input processing failed for {module_name}: {e}")
                return {'module': module_name, 'status': 'error', 'error': str(e)}
        
        return handle_input
    
    def _create_output_publisher(self, module_name: str) -> Callable:
        """Create output publisher for ML module"""
        
        async def publish_output(results: Dict[str, Any]):
            try:
                # Create orchestration event
                event = OrchestrationEvent(
                    event_id=str(uuid.uuid4()),
                    event_type='module_output',
                    source_module=module_name,
                    target_modules=[],  # Will be determined by flows
                    data=results
                )
                
                # Queue event for processing
                with self.event_lock:
                    self.event_queue.append(event)
                
                self.logger.debug(f"Output published from {module_name}")
                
            except Exception as e:
                self.logger.error(f"Output publishing failed for {module_name}: {e}")
        
        return publish_output
    
    def _create_health_handler(self, module_name: str) -> Callable:
        """Create health check handler for ML module"""
        
        async def check_health() -> Dict[str, Any]:
            try:
                # Simulate health check
                interface = self.ml_modules.get(module_name)
                if not interface:
                    return {'status': 'error', 'message': 'Module not found'}
                
                # Update last health check
                interface.last_health_check = datetime.now()
                
                # Calculate health metrics
                health_score = min(1.0, interface.success_rate * 0.8 + 
                                 (1.0 - interface.resource_usage.get('cpu', 0.5)) * 0.2)
                
                return {
                    'status': 'healthy' if health_score > 0.7 else 'degraded',
                    'health_score': health_score,
                    'success_rate': interface.success_rate,
                    'resource_usage': interface.resource_usage,
                    'last_check': interface.last_health_check.isoformat()
                }
                
            except Exception as e:
                return {'status': 'error', 'message': str(e)}
        
        return check_health
    
    async def _orchestration_loop(self):
        """Main orchestration coordination loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.orchestration_interval)
                
                # Update system health
                await self._update_system_health()
                
                # Optimize resource allocation
                if self.enable_auto_optimization:
                    await self._optimize_resource_allocation()
                
                # Execute integration flows
                await self._execute_integration_flows()
                
                # Update performance metrics
                await self._update_orchestration_metrics()
                
            except Exception as e:
                self.logger.error(f"Orchestration loop error: {e}")
                await asyncio.sleep(5)
    
    async def _event_processing_loop(self):
        """Process orchestration events"""
        
        while not self.shutdown_event.is_set():
            try:
                if not self.event_queue:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get next event
                with self.event_lock:
                    if self.event_queue:
                        event = self.event_queue.popleft()
                    else:
                        continue
                
                # Process event
                await self._process_orchestration_event(event)
                
                # Track processed event
                with self.event_lock:
                    self.processed_events.append(event)
                    self.orchestration_stats['events_processed'] += 1
                
            except Exception as e:
                self.logger.error(f"Event processing error: {e}")
                await asyncio.sleep(1)
    
    async def _process_orchestration_event(self, event: OrchestrationEvent):
        """Process individual orchestration event"""
        
        try:
            start_time = time.time()
            
            # Determine target modules based on integration flows
            target_modules = self._determine_event_targets(event)
            event.target_modules = target_modules
            
            # Execute event on target modules
            for module_name in target_modules:
                if module_name in self.ml_modules:
                    interface = self.ml_modules[module_name]
                    
                    # Process input through module
                    if 'process_data' in interface.input_handlers:
                        result = await interface.input_handlers['process_data'](event.data)
                        event.results[module_name] = result
            
            # Calculate processing time
            event.processing_time = time.time() - start_time
            event.processed = True
            
            self.logger.debug(f"Event processed: {event.event_id} -> {len(target_modules)} modules")
            
        except Exception as e:
            self.logger.error(f"Event processing failed: {e}")
            event.results['error'] = str(e)
    
    def _determine_event_targets(self, event: OrchestrationEvent) -> List[str]:
        """Determine target modules for orchestration event"""
        
        target_modules = []
        
        # Check integration flows
        for flow in self.integration_flows.values():
            if not flow.enabled:
                continue
            
            if event.source_module in flow.source_modules:
                target_modules.extend(flow.target_modules)
        
        # Remove duplicates and source module
        target_modules = list(set(target_modules))
        if event.source_module in target_modules:
            target_modules.remove(event.source_module)
        
        return target_modules
    
    async def _health_monitoring_loop(self):
        """Monitor health of all ML modules"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Check health of all modules
                for module_name, interface in self.ml_modules.items():
                    if 'health_check' in interface.control_interfaces:
                        health_result = await interface.control_interfaces['health_check']()
                        
                        # Update interface status
                        if health_result.get('status') == 'healthy':
                            interface.status = 'active'
                        elif health_result.get('status') == 'degraded':
                            interface.status = 'degraded'
                        else:
                            interface.status = 'failed'
                        
                        # Store health metrics
                        health_score = health_result.get('health_score', 0.5)
                        self.module_performance_history[module_name].append({
                            'timestamp': datetime.now(),
                            'health_score': health_score,
                            'status': interface.status
                        })
                
                # Update overall system health
                await self._calculate_system_health()
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(10)
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration status"""
        
        # Module status summary
        module_status = {}
        for module_name, interface in self.ml_modules.items():
            module_status[module_name] = {
                'status': interface.status,
                'success_rate': interface.success_rate,
                'processing_time': interface.processing_time,
                'resource_usage': interface.resource_usage,
                'last_health_check': interface.last_health_check.isoformat()
            }
        
        # Integration flow status
        flow_status = {}
        for flow_id, flow in self.integration_flows.items():
            flow_status[flow_id] = {
                'pattern': flow.pattern.value,
                'enabled': flow.enabled,
                'source_modules': flow.source_modules,
                'target_modules': flow.target_modules,
                'message_count': flow.message_count,
                'average_latency': flow.average_latency,
                'error_count': flow.error_count
            }
        
        # System metrics
        active_modules = len([m for m in self.ml_modules.values() if m.status == 'active'])
        
        return {
            'orchestration_overview': {
                'total_modules': len(self.ml_modules),
                'active_modules': active_modules,
                'integration_flows': len(self.integration_flows),
                'system_health_score': self.system_health_score,
                'current_mode': self.current_mode.value,
                'events_in_queue': len(self.event_queue)
            },
            'module_status': module_status,
            'integration_flows': flow_status,
            'performance_metrics': self.performance_metrics,
            'statistics': self.orchestration_stats.copy()
        }
    
    def get_integration_insights(self) -> Dict[str, Any]:
        """Get ML integration insights and recommendations"""
        
        # Analyze module interactions
        interaction_matrix = defaultdict(dict)
        for flow in self.integration_flows.values():
            for source in flow.source_modules:
                for target in flow.target_modules:
                    interaction_matrix[source][target] = {
                        'flow_id': flow.flow_id,
                        'pattern': flow.pattern.value,
                        'message_count': flow.message_count,
                        'avg_latency': flow.average_latency
                    }
        
        # Calculate module importance scores
        importance_scores = {}
        for module_name in self.ml_modules.keys():
            # Count incoming and outgoing connections
            incoming = sum(1 for flow in self.integration_flows.values() 
                          if module_name in flow.target_modules)
            outgoing = sum(1 for flow in self.integration_flows.values() 
                          if module_name in flow.source_modules)
            
            importance_scores[module_name] = incoming + outgoing * 1.5  # Outgoing weighted higher
        
        # Identify bottlenecks
        bottlenecks = []
        for module_name, interface in self.ml_modules.items():
            if interface.processing_time > 100:  # ms
                bottlenecks.append({
                    'module': module_name,
                    'processing_time': interface.processing_time,
                    'status': interface.status
                })
        
        return {
            'interaction_matrix': dict(interaction_matrix),
            'module_importance_scores': importance_scores,
            'bottlenecks': bottlenecks,
            'optimization_opportunities': self._identify_optimization_opportunities(),
            'integration_health': self._calculate_integration_health(),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify optimization opportunities in the ML ecosystem"""
        
        opportunities = []
        
        # Underutilized modules
        for module_name, interface in self.ml_modules.items():
            cpu_usage = interface.resource_usage.get('cpu', 0.0)
            if cpu_usage < 0.3 and interface.status == 'active':
                opportunities.append({
                    'type': 'underutilized',
                    'module': module_name,
                    'description': f'Module {module_name} is underutilized ({cpu_usage:.1%} CPU)',
                    'recommendation': 'Consider consolidating workloads or reducing resources',
                    'potential_savings': f'{(0.5 - cpu_usage) * 100:.1f}% resource reduction'
                })
        
        # High latency flows
        for flow_id, flow in self.integration_flows.values():
            if flow.average_latency > flow.max_latency_ms:
                opportunities.append({
                    'type': 'high_latency',
                    'flow': flow_id,
                    'description': f'Flow {flow_id} has high latency ({flow.average_latency:.1f}ms)',
                    'recommendation': 'Optimize data transformation or consider parallel processing',
                    'potential_improvement': f'{((flow.average_latency - flow.max_latency_ms) / flow.average_latency) * 100:.1f}% latency reduction'
                })
        
        return opportunities
    
    def _calculate_integration_health(self) -> float:
        """Calculate overall integration health score"""
        
        if not self.ml_modules:
            return 0.0
        
        # Module health contribution
        module_health = sum(1.0 if m.status == 'active' else 0.5 if m.status == 'degraded' else 0.0 
                           for m in self.ml_modules.values()) / len(self.ml_modules)
        
        # Flow health contribution
        flow_health = 1.0
        if self.integration_flows:
            flow_errors = sum(flow.error_count for flow in self.integration_flows.values())
            flow_total = sum(flow.message_count for flow in self.integration_flows.values())
            flow_health = max(0.0, 1.0 - (flow_errors / max(flow_total, 1)))
        
        # Overall health score
        return (module_health * 0.7 + flow_health * 0.3)
    
    async def shutdown(self):
        """Graceful shutdown of ML orchestrator"""
        
        self.logger.info("Shutting down ML orchestrator...")
        
        # Stop all modules
        for module_name, interface in self.ml_modules.items():
            if 'stop' in interface.control_interfaces:
                try:
                    await interface.control_interfaces['stop']()
                except Exception as e:
                    self.logger.error(f"Error stopping {module_name}: {e}")
        
        # Process remaining events (with timeout)
        timeout = 30
        while self.event_queue and timeout > 0:
            await asyncio.sleep(1)
            timeout -= 1
        
        self.shutdown_event.set()
        await asyncio.sleep(1)
        self.logger.info("ML orchestrator shutdown complete")


# Global orchestrator instance
ml_orchestrator = MLOrchestrator()

def get_ml_orchestration_status():
    """Get current ML orchestration status"""
    return ml_orchestrator.get_orchestration_status()

def get_ml_integration_insights():
    """Get ML integration insights"""
    return ml_orchestrator.get_integration_insights()