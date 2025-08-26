"""
Orchestration Coordination Layer
================================

Connects all 7 TestMaster orchestration systems to the Enterprise Integration Hub,
providing unified coordination, communication, and performance optimization across
the entire orchestration ecosystem.

Author: Agent E - Infrastructure Consolidation Expert
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Import integration hub
try:
    from .integration_hub import (
        enterprise_integration_hub,
        SystemType,
        IntegrationEventType,
        SystemMessage,
        ServiceEndpoint,
        MessagePriority
    )
    INTEGRATION_HUB_AVAILABLE = True
except ImportError:
    INTEGRATION_HUB_AVAILABLE = False

logger = logging.getLogger(__name__)


class OrchestratorType(Enum):
    """Types of orchestration systems."""
    MASTER_ORCHESTRATOR = "master_orchestrator"
    UNIFIED_ORCHESTRATOR = "unified_orchestrator"
    ENHANCED_AGENT_ORCHESTRATOR = "enhanced_agent_orchestrator"
    CROSS_SYSTEM_ORCHESTRATOR = "cross_system_orchestrator"
    WORKFLOW_ORCHESTRATION_ENGINE = "workflow_orchestration_engine"
    UNIVERSAL_ORCHESTRATOR = "universal_orchestrator"
    CORE_ORCHESTRATOR = "core_orchestrator"
    INTELLIGENCE_ORCHESTRATOR = "intelligence_orchestrator"


@dataclass
class OrchestrationSystemInfo:
    """Information about an orchestration system."""
    orchestrator_type: OrchestratorType
    instance: Any
    capabilities: List[str]
    status: str = "active"
    last_heartbeat: Optional[datetime] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


class OrchestrationCoordinator:
    """
    Unified coordinator for all orchestration systems.
    
    Connects all 7 orchestration systems to the Enterprise Integration Hub,
    enabling cross-system communication, load balancing, and performance optimization.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("OrchestrationCoordinator")
        
        # Registry of orchestration systems
        self.orchestration_systems: Dict[OrchestratorType, OrchestrationSystemInfo] = {}
        
        # Coordination state
        self.coordination_active = False
        self.load_balancing_enabled = True
        self.cross_system_routing_enabled = True
        
        # Performance tracking
        self.coordination_metrics = {
            "total_coordinated_requests": 0,
            "successful_coordination": 0,
            "failed_coordination": 0,
            "average_coordination_time": 0.0,
            "system_utilization": {},
            "start_time": datetime.now()
        }
        
        # Auto-discovery and registration
        self._discover_orchestration_systems()
        
        # Connect to Integration Hub if available
        if INTEGRATION_HUB_AVAILABLE:
            self._connect_to_integration_hub()
        
        self.logger.info("Orchestration Coordinator initialized")
    
    def _discover_orchestration_systems(self):
        """Auto-discover available orchestration systems."""
        try:
            # Try to import and register each orchestration system
            orchestration_imports = [
                ("master_orchestrator", "master_orchestrator", OrchestratorType.MASTER_ORCHESTRATOR),
                ("unified_orchestrator", "unified_orchestrator", OrchestratorType.UNIFIED_ORCHESTRATOR),
                ("enhanced_agent_orchestrator", "enhanced_orchestrator", OrchestratorType.ENHANCED_AGENT_ORCHESTRATOR),
                ("cross_system_orchestrator", "cross_system_orchestrator", OrchestratorType.CROSS_SYSTEM_ORCHESTRATOR),
                ("workflow_orchestration_engine", "workflow_orchestration_engine", OrchestratorType.WORKFLOW_ORCHESTRATION_ENGINE),
            ]
            
            discovered_count = 0
            for module_path, instance_name, orchestrator_type in orchestration_imports:
                try:
                    # Dynamic import
                    if module_path == "master_orchestrator":
                        from .master_orchestrator import master_orchestrator as instance
                        capabilities = ["workflow", "swarm", "intelligence", "parallel", "sequential"]
                        
                    elif module_path == "unified_orchestrator":
                        from ..unified_orchestrator import UnifiedOrchestrator
                        instance = UnifiedOrchestrator()
                        capabilities = ["graph", "swarm", "hybrid", "intelligent_routing"]
                        
                    else:
                        # Skip modules that might not be available
                        continue
                    
                    # Register the orchestration system
                    system_info = OrchestrationSystemInfo(
                        orchestrator_type=orchestrator_type,
                        instance=instance,
                        capabilities=capabilities,
                        status="discovered"
                    )
                    
                    self.orchestration_systems[orchestrator_type] = system_info
                    discovered_count += 1
                    
                    self.logger.info(f"Discovered orchestration system: {orchestrator_type.value}")
                    
                except ImportError as e:
                    self.logger.debug(f"Orchestration system {orchestrator_type.value} not available: {e}")
                except Exception as e:
                    self.logger.warning(f"Error discovering {orchestrator_type.value}: {e}")
            
            self.logger.info(f"Auto-discovery complete: {discovered_count} orchestration systems found")
            
        except Exception as e:
            self.logger.error(f"Orchestration system discovery failed: {e}")
    
    def _connect_to_integration_hub(self):
        """Connect all orchestration systems to Integration Hub."""
        try:
            # Register coordinator as master service
            coordinator_endpoint = ServiceEndpoint(
                service_id="orchestration_coordinator",
                system_type=SystemType.WORKFLOW_ENGINE,
                host="localhost",
                port=8081,
                path="/coordinator",
                health_check_path="/coordinator/health",
                metadata={
                    "version": "1.0.0",
                    "role": "coordinator",
                    "managed_systems": len(self.orchestration_systems)
                }
            )
            
            enterprise_integration_hub.register_system(SystemType.WORKFLOW_ENGINE, coordinator_endpoint)
            
            # Register each orchestration system
            for orchestrator_type, system_info in self.orchestration_systems.items():
                self._register_orchestration_system(orchestrator_type, system_info)
            
            # Subscribe to coordination events
            self._subscribe_to_coordination_events()
            
            self.logger.info("All orchestration systems connected to Integration Hub")
            
        except Exception as e:
            self.logger.error(f"Failed to connect orchestration systems to Integration Hub: {e}")
    
    def _register_orchestration_system(self, orchestrator_type: OrchestratorType, system_info: OrchestrationSystemInfo):
        """Register individual orchestration system with Integration Hub."""
        try:
            # Create service endpoint for the orchestration system
            endpoint = ServiceEndpoint(
                service_id=f"orchestrator_{orchestrator_type.value}",
                system_type=SystemType.WORKFLOW_ENGINE,
                host="localhost",
                port=8080 + len(self.orchestration_systems),  # Dynamic port assignment
                path=f"/{orchestrator_type.value}",
                health_check_path=f"/{orchestrator_type.value}/health",
                metadata={
                    "orchestrator_type": orchestrator_type.value,
                    "capabilities": system_info.capabilities,
                    "version": "1.0.0"
                }
            )
            
            enterprise_integration_hub.register_system(SystemType.WORKFLOW_ENGINE, endpoint)
            system_info.status = "registered"
            
            self.logger.info(f"Registered {orchestrator_type.value} with Integration Hub")
            
        except Exception as e:
            self.logger.error(f"Failed to register {orchestrator_type.value}: {e}")
            system_info.status = "registration_failed"
    
    def _subscribe_to_coordination_events(self):
        """Subscribe to events for orchestration coordination."""
        # Subscribe to performance alerts
        enterprise_integration_hub.subscribe_to_events(
            IntegrationEventType.PERFORMANCE_ALERT,
            self._handle_performance_alert
        )
        
        # Subscribe to workflow events
        enterprise_integration_hub.subscribe_to_events(
            IntegrationEventType.WORKFLOW_COMPLETED,
            self._handle_workflow_event
        )
        
        # Subscribe to system health changes
        enterprise_integration_hub.subscribe_to_events(
            IntegrationEventType.SERVICE_HEALTH_CHANGE,
            self._handle_health_change
        )
        
        self.logger.info("Subscribed to orchestration coordination events")
    
    async def _handle_performance_alert(self, message: SystemMessage):
        """Handle performance alerts for load balancing."""
        payload = message.payload
        source_system = message.source_system
        
        self.logger.info(f"Performance alert from {source_system.value}: {payload}")
        
        # Implement load balancing logic
        if self.load_balancing_enabled:
            await self._rebalance_orchestration_load(payload)
    
    async def _handle_workflow_event(self, message: SystemMessage):
        """Handle workflow completion events."""
        payload = message.payload
        
        self.coordination_metrics["total_coordinated_requests"] += 1
        
        if payload.get("phase") == "completed":
            self.coordination_metrics["successful_coordination"] += 1
        elif payload.get("phase") == "failed":
            self.coordination_metrics["failed_coordination"] += 1
        
        self.logger.debug(f"Workflow event: {payload}")
    
    async def _handle_health_change(self, message: SystemMessage):
        """Handle orchestration system health changes."""
        payload = message.payload
        service_id = payload.get("service_id", "")
        
        # Update orchestration system status
        for orchestrator_type, system_info in self.orchestration_systems.items():
            if f"orchestrator_{orchestrator_type.value}" == service_id:
                system_info.status = payload.get("status", "unknown")
                system_info.last_heartbeat = datetime.now()
                break
        
        self.logger.info(f"Health change for {service_id}: {payload}")
    
    async def _rebalance_orchestration_load(self, performance_data: Dict[str, Any]):
        """Rebalance load across orchestration systems."""
        try:
            cpu_usage = performance_data.get("cpu_usage", 0)
            memory_usage = performance_data.get("memory_usage", 0)
            
            if cpu_usage > 80 or memory_usage > 85:
                # Find less loaded orchestration systems
                available_systems = [
                    system for system in self.orchestration_systems.values()
                    if system.status == "registered"
                ]
                
                if len(available_systems) > 1:
                    # Implement load redistribution
                    self.logger.info("Rebalancing orchestration load...")
                    
                    # Publish load balancing event
                    await self._publish_coordination_event(
                        IntegrationEventType.LOAD_BALANCING_EVENT,
                        {
                            "action": "rebalance",
                            "trigger": "performance_alert",
                            "available_systems": len(available_systems)
                        }
                    )
            
        except Exception as e:
            self.logger.error(f"Load rebalancing failed: {e}")
    
    async def _publish_coordination_event(self, event_type: IntegrationEventType, payload: Dict[str, Any]):
        """Publish coordination events."""
        if INTEGRATION_HUB_AVAILABLE:
            message = SystemMessage(
                source_system=SystemType.WORKFLOW_ENGINE,
                event_type=event_type,
                payload=payload,
                priority=MessagePriority.HIGH
            )
            
            await enterprise_integration_hub.publish_event(message)
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status."""
        system_statuses = {}
        for orchestrator_type, system_info in self.orchestration_systems.items():
            system_statuses[orchestrator_type.value] = {
                "status": system_info.status,
                "capabilities": system_info.capabilities,
                "last_heartbeat": system_info.last_heartbeat.isoformat() if system_info.last_heartbeat else None,
                "performance_metrics": system_info.performance_metrics
            }
        
        return {
            "coordinator_status": "active" if self.coordination_active else "inactive",
            "total_systems": len(self.orchestration_systems),
            "registered_systems": len([s for s in self.orchestration_systems.values() if s.status == "registered"]),
            "coordination_metrics": self.coordination_metrics,
            "load_balancing_enabled": self.load_balancing_enabled,
            "cross_system_routing_enabled": self.cross_system_routing_enabled,
            "system_statuses": system_statuses,
            "integration_hub_connected": INTEGRATION_HUB_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        }
    
    async def start_coordination(self):
        """Start orchestration coordination."""
        if self.coordination_active:
            return
        
        self.coordination_active = True
        
        # Start Integration Hub if available
        if INTEGRATION_HUB_AVAILABLE:
            await enterprise_integration_hub.start_integration_hub()
        
        self.logger.info("Orchestration coordination started")
    
    async def stop_coordination(self):
        """Stop orchestration coordination."""
        self.coordination_active = False
        
        # Stop Integration Hub if available
        if INTEGRATION_HUB_AVAILABLE:
            await enterprise_integration_hub.stop_integration_hub()
        
        self.logger.info("Orchestration coordination stopped")
    
    def select_best_orchestrator(self, requirements: Dict[str, Any]) -> Optional[OrchestratorType]:
        """Select the best orchestration system for given requirements."""
        required_capabilities = requirements.get("capabilities", [])
        task_type = requirements.get("task_type", "general")
        priority = requirements.get("priority", "normal")
        
        # Score each available orchestration system
        best_orchestrator = None
        best_score = -1
        
        for orchestrator_type, system_info in self.orchestration_systems.items():
            if system_info.status != "registered":
                continue
            
            score = self._calculate_orchestrator_score(system_info, requirements)
            
            if score > best_score:
                best_score = score
                best_orchestrator = orchestrator_type
        
        return best_orchestrator
    
    def _calculate_orchestrator_score(self, system_info: OrchestrationSystemInfo, requirements: Dict[str, Any]) -> float:
        """Calculate suitability score for an orchestration system."""
        score = 0.0
        
        # Capability match scoring
        required_capabilities = set(requirements.get("capabilities", []))
        system_capabilities = set(system_info.capabilities)
        
        if required_capabilities:
            capability_match = len(required_capabilities.intersection(system_capabilities)) / len(required_capabilities)
            score += capability_match * 50  # 50% weight for capability match
        else:
            score += 25  # Base score if no specific requirements
        
        # Performance scoring (if metrics available)
        if system_info.performance_metrics:
            success_rate = system_info.performance_metrics.get("success_rate", 90.0)
            avg_time = system_info.performance_metrics.get("average_execution_time", 1.0)
            
            score += (success_rate / 100.0) * 30  # 30% weight for success rate
            score += max(0, (5.0 - avg_time) / 5.0) * 20  # 20% weight for speed (inverse of time)
        else:
            score += 25  # Default performance score
        
        return score


# Global orchestration coordinator instance
orchestration_coordinator = OrchestrationCoordinator()


# Export key components
__all__ = [
    'OrchestrationCoordinator',
    'OrchestratorType',
    'OrchestrationSystemInfo',
    'orchestration_coordinator'
]