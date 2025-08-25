"""
Orchestration Configuration
===========================

Hierarchical configuration for the orchestration layer providing
unified configuration management for all orchestration systems.

Author: Agent E - Infrastructure Consolidation
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from ...foundation.configuration.base.config_base import (
    LayeredConfiguration, 
    ConfigurationLayer,
    ConfigurationScope
)

# Import orchestration foundations for type integration
try:
    from ..foundations import (
        OrchestratorType,
        ExecutionStrategy,
        HybridMode,
        SwitchingStrategy,
        CompositionStrategy,
        IntelligenceType,
        LearningStrategy,
        SwarmBehavior,
        CommunicationPattern,
        CoordinationMode,
        IntegrationType,
        ServiceProtocol
    )
    FOUNDATIONS_AVAILABLE = True
except ImportError:
    FOUNDATIONS_AVAILABLE = False


class OrchestrationMode(Enum):
    """Orchestration execution modes."""
    WORKFLOW = "workflow"
    SWARM = "swarm"
    INTELLIGENCE = "intelligence"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LEAST_CONNECTIONS = "least_connections"
    PERFORMANCE_BASED = "performance_based"
    ADAPTIVE = "adaptive"


@dataclass
class OrchestrationDefaults:
    """Default orchestration configuration values."""
    
    # Execution settings
    max_parallel_tasks: int = 8
    task_timeout_seconds: int = 300
    retry_attempts: int = 3
    
    # Orchestration modes
    default_mode: OrchestrationMode = OrchestrationMode.WORKFLOW
    fallback_mode: OrchestrationMode = OrchestrationMode.ADAPTIVE
    
    # Load balancing
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.PERFORMANCE_BASED
    load_balancing_enabled: bool = True
    
    # Integration hub
    integration_hub_enabled: bool = True
    service_mesh_enabled: bool = True
    circuit_breaker_enabled: bool = True
    
    # Performance settings
    health_check_interval: int = 30
    performance_monitoring_enabled: bool = True
    metrics_collection_enabled: bool = True
    
    # Agent coordination
    agent_discovery_enabled: bool = True
    cross_agent_communication: bool = True
    agent_heartbeat_interval: int = 10
    
    # Workflow execution
    workflow_persistence_enabled: bool = True
    workflow_recovery_enabled: bool = True
    workflow_validation_enabled: bool = True


class OrchestrationConfiguration(LayeredConfiguration):
    """
    Hierarchical configuration for the orchestration layer.
    
    Provides unified configuration management for:
    - Master orchestrator
    - Workflow execution engines
    - Swarm orchestration
    - Intelligence orchestration
    - Integration hub
    - Cross-system coordination
    """
    
    def __init__(self, parent: Optional[LayeredConfiguration] = None):
        super().__init__(
            layer=ConfigurationLayer.ORCHESTRATION,
            domain="orchestration",
            parent=parent
        )
        
        self.defaults = OrchestrationDefaults()
        self._initialize_default_config()
        
        # Add orchestration-specific validators
        self._add_orchestration_validators()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default orchestration configuration."""
        config = {
            # Execution configuration
            'execution': {
                'max_parallel_tasks': self.defaults.max_parallel_tasks,
                'task_timeout_seconds': self.defaults.task_timeout_seconds,
                'retry_attempts': self.defaults.retry_attempts,
                'default_mode': self.defaults.default_mode.value,
                'fallback_mode': self.defaults.fallback_mode.value
            },
            
            # Load balancing configuration
            'load_balancing': {
                'enabled': self.defaults.load_balancing_enabled,
                'strategy': self.defaults.load_balancing_strategy.value,
                'health_check_interval': self.defaults.health_check_interval,
                'performance_weight': 0.7,
                'latency_weight': 0.3
            },
            
            # Integration hub configuration
            'integration_hub': {
                'enabled': self.defaults.integration_hub_enabled,
                'service_mesh_enabled': self.defaults.service_mesh_enabled,
                'circuit_breaker_enabled': self.defaults.circuit_breaker_enabled,
                'auto_discovery': True,
                'health_monitoring': True,
                'event_bus_enabled': True
            },
            
            # Performance monitoring configuration
            'monitoring': {
                'enabled': self.defaults.performance_monitoring_enabled,
                'metrics_collection': self.defaults.metrics_collection_enabled,
                'real_time_monitoring': True,
                'performance_alerts': True,
                'threshold_cpu_usage': 80.0,
                'threshold_memory_usage': 85.0,
                'threshold_response_time': 5000  # milliseconds
            },
            
            # Agent coordination configuration
            'agents': {
                'discovery_enabled': self.defaults.agent_discovery_enabled,
                'cross_communication': self.defaults.cross_agent_communication,
                'heartbeat_interval': self.defaults.agent_heartbeat_interval,
                'agent_timeout': 60,
                'max_agents': 100,
                'agent_pools': ['workflow', 'swarm', 'intelligence']
            },
            
            # Workflow configuration
            'workflow': {
                'persistence_enabled': self.defaults.workflow_persistence_enabled,
                'recovery_enabled': self.defaults.workflow_recovery_enabled,
                'validation_enabled': self.defaults.workflow_validation_enabled,
                'dag_validation': True,
                'cycle_detection': True,
                'dependency_resolution': True
            },
            
            # Swarm configuration
            'swarm': {
                'enabled': True,
                'swarm_size_min': 2,
                'swarm_size_max': 20,
                'swarm_coordination_mode': 'hierarchical',
                'swarm_communication_protocol': 'event_driven',
                'swarm_health_monitoring': True
            },
            
            # Intelligence configuration
            'intelligence': {
                'enabled': True,
                'ml_orchestration': True,
                'adaptive_learning': True,
                'predictive_scheduling': True,
                'intelligence_feedback_loop': True,
                'ml_model_updates': True
            }
        }
        
        # Add foundations-specific configuration if available
        if FOUNDATIONS_AVAILABLE:
            config.update(self._get_foundations_config())
        
        return config
    
    def _get_foundations_config(self) -> Dict[str, Any]:
        """Get configuration specific to orchestration foundations."""
        return {
            # Orchestrator types configuration
            'orchestrator_types': {
                'supported_types': [ot.value for ot in OrchestratorType],
                'default_type': OrchestratorType.HYBRID.value,
                'auto_type_selection': True
            },
            
            # Execution strategies
            'execution_strategies': {
                'supported_strategies': [es.value for es in ExecutionStrategy],
                'default_strategy': ExecutionStrategy.ADAPTIVE.value,
                'strategy_switching_enabled': True
            },
            
            # Hybrid orchestration
            'hybrid_orchestration': {
                'supported_modes': [hm.value for hm in HybridMode],
                'default_mode': HybridMode.ADAPTIVE_MULTI.value,
                'switching_strategies': [ss.value for ss in SwitchingStrategy],
                'composition_strategies': [cs.value for cs in CompositionStrategy],
                'pattern_switching_threshold': 0.2
            },
            
            # Intelligence configuration
            'intelligence_foundations': {
                'supported_types': [it.value for it in IntelligenceType],
                'learning_strategies': [ls.value for ls in LearningStrategy],
                'default_intelligence_type': IntelligenceType.ADAPTIVE.value,
                'learning_enabled': True,
                'adaptation_threshold': 0.1
            },
            
            # Swarm configuration
            'swarm_foundations': {
                'supported_behaviors': [sb.value for sb in SwarmBehavior],
                'default_behavior': SwarmBehavior.COLLABORATIVE.value,
                'emergent_behavior_enabled': True,
                'swarm_optimization': True
            },
            
            # Communication protocols
            'communication': {
                'supported_patterns': [cp.value for cp in CommunicationPattern],
                'default_pattern': CommunicationPattern.PUBLISH_SUBSCRIBE.value,
                'protocol_auto_selection': True,
                'reliability_level': 'high'
            },
            
            # Coordination protocols
            'coordination': {
                'supported_modes': [cm.value for cm in CoordinationMode],
                'default_mode': CoordinationMode.HIERARCHICAL.value,
                'consensus_enabled': True,
                'synchronization_enabled': True
            },
            
            # Integration protocols
            'integration_foundations': {
                'supported_types': [it.value for it in IntegrationType],
                'supported_protocols': [sp.value for sp in ServiceProtocol],
                'default_protocol': ServiceProtocol.HTTPS.value,
                'circuit_breaker_enabled': True,
                'retry_policies_enabled': True
            }
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """Get orchestration configuration schema."""
        return {
            'type': 'object',
            'required': ['execution', 'integration_hub'],
            'properties': {
                'execution': {
                    'type': 'object',
                    'required': ['max_parallel_tasks', 'task_timeout_seconds'],
                    'properties': {
                        'max_parallel_tasks': {
                            'type': 'integer',
                            'minimum': 1,
                            'maximum': 100
                        },
                        'task_timeout_seconds': {
                            'type': 'integer',
                            'minimum': 1,
                            'maximum': 3600
                        },
                        'retry_attempts': {
                            'type': 'integer',
                            'minimum': 0,
                            'maximum': 10
                        },
                        'default_mode': {
                            'type': 'string',
                            'enum': [mode.value for mode in OrchestrationMode]
                        }
                    }
                },
                'load_balancing': {
                    'type': 'object',
                    'properties': {
                        'enabled': {'type': 'boolean'},
                        'strategy': {
                            'type': 'string',
                            'enum': [strategy.value for strategy in LoadBalancingStrategy]
                        },
                        'health_check_interval': {
                            'type': 'integer',
                            'minimum': 5,
                            'maximum': 300
                        }
                    }
                },
                'integration_hub': {
                    'type': 'object',
                    'required': ['enabled'],
                    'properties': {
                        'enabled': {'type': 'boolean'},
                        'service_mesh_enabled': {'type': 'boolean'},
                        'circuit_breaker_enabled': {'type': 'boolean'}
                    }
                },
                'monitoring': {
                    'type': 'object',
                    'properties': {
                        'enabled': {'type': 'boolean'},
                        'threshold_cpu_usage': {
                            'type': 'number',
                            'minimum': 0,
                            'maximum': 100
                        },
                        'threshold_memory_usage': {
                            'type': 'number',
                            'minimum': 0,
                            'maximum': 100
                        }
                    }
                }
            },
            'dependencies': {
                'load_balancing': ['integration_hub'],
                'monitoring': ['integration_hub']
            }
        }
    
    def get_orchestration_mode_config(self, mode: OrchestrationMode) -> Dict[str, Any]:
        """Get configuration for specific orchestration mode."""
        base_config = self.to_dict()
        
        mode_specific = {
            OrchestrationMode.WORKFLOW: {
                'priority_scheduling': True,
                'dag_optimization': True,
                'parallel_execution': True
            },
            OrchestrationMode.SWARM: {
                'agent_coordination': True,
                'swarm_intelligence': True,
                'dynamic_scaling': True
            },
            OrchestrationMode.INTELLIGENCE: {
                'ml_optimization': True,
                'adaptive_learning': True,
                'predictive_execution': True
            },
            OrchestrationMode.HYBRID: {
                'multi_mode_coordination': True,
                'intelligent_mode_switching': True,
                'performance_optimization': True
            },
            OrchestrationMode.ADAPTIVE: {
                'dynamic_mode_selection': True,
                'performance_based_switching': True,
                'learning_enabled': True
            }
        }
        
        base_config['mode_specific'] = mode_specific.get(mode, {})
        return base_config
    
    def get_integration_hub_config(self) -> Dict[str, Any]:
        """Get integration hub specific configuration."""
        return self.get('integration_hub', {})
    
    def get_load_balancing_config(self) -> Dict[str, Any]:
        """Get load balancing specific configuration."""
        return self.get('load_balancing', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring specific configuration."""
        return self.get('monitoring', {})
    
    def get_foundations_config(self) -> Dict[str, Any]:
        """Get orchestration foundations configuration."""
        if not FOUNDATIONS_AVAILABLE:
            return {}
        return {
            'orchestrator_types': self.get('orchestrator_types', {}),
            'execution_strategies': self.get('execution_strategies', {}),
            'hybrid_orchestration': self.get('hybrid_orchestration', {}),
            'intelligence_foundations': self.get('intelligence_foundations', {}),
            'swarm_foundations': self.get('swarm_foundations', {}),
            'communication': self.get('communication', {}),
            'coordination': self.get('coordination', {}),
            'integration_foundations': self.get('integration_foundations', {})
        }
    
    def get_hybrid_config(self) -> Dict[str, Any]:
        """Get hybrid orchestration configuration."""
        return self.get('hybrid_orchestration', {})
    
    def get_intelligence_foundations_config(self) -> Dict[str, Any]:
        """Get intelligence foundations configuration."""
        return self.get('intelligence_foundations', {})
    
    def get_swarm_foundations_config(self) -> Dict[str, Any]:
        """Get swarm foundations configuration."""
        return self.get('swarm_foundations', {})
    
    def get_communication_config(self) -> Dict[str, Any]:
        """Get communication protocols configuration."""
        return self.get('communication', {})
    
    def get_coordination_config(self) -> Dict[str, Any]:
        """Get coordination protocols configuration."""
        return self.get('coordination', {})
    
    def get_integration_foundations_config(self) -> Dict[str, Any]:
        """Get integration foundations configuration."""
        return self.get('integration_foundations', {})
    
    def enable_high_performance_mode(self):
        """Enable high performance orchestration configuration."""
        self.update({
            'execution': {
                'max_parallel_tasks': 16,
                'task_timeout_seconds': 600,
                'retry_attempts': 5
            },
            'load_balancing': {
                'enabled': True,
                'strategy': LoadBalancingStrategy.PERFORMANCE_BASED.value,
                'health_check_interval': 15
            },
            'monitoring': {
                'enabled': True,
                'real_time_monitoring': True,
                'performance_alerts': True
            }
        })
        self.metadata.tags.append("high_performance")
    
    def enable_development_mode(self):
        """Enable development-friendly orchestration configuration."""
        self.update({
            'execution': {
                'max_parallel_tasks': 4,
                'task_timeout_seconds': 60,
                'retry_attempts': 1
            },
            'monitoring': {
                'enabled': True,
                'metrics_collection': True,
                'real_time_monitoring': False
            },
            'workflow': {
                'validation_enabled': True,
                'dag_validation': True,
                'cycle_detection': True
            }
        })
        self.metadata.tags.append("development")
    
    def _initialize_default_config(self):
        """Initialize with default configuration."""
        default_config = self.get_default_config()
        self.update(default_config)
    
    def _add_orchestration_validators(self):
        """Add orchestration-specific validators."""
        from ...foundation.configuration.base.validation import ConfigurationValidator
        
        validator = ConfigurationValidator()
        
        # Add custom validation rules
        def validate_orchestration_mode(config):
            """Validate orchestration mode configuration."""
            execution = config.get('execution', {})
            mode = execution.get('default_mode')
            
            if mode and mode not in [m.value for m in OrchestrationMode]:
                return [f"Invalid orchestration mode: {mode}"]
            return []
        
        def validate_load_balancing_dependencies(config):
            """Validate load balancing dependencies."""
            lb_config = config.get('load_balancing', {})
            hub_config = config.get('integration_hub', {})
            
            if lb_config.get('enabled') and not hub_config.get('enabled'):
                return ["Load balancing requires integration hub to be enabled"]
            return []
        
        validator.add_global_validator(validate_orchestration_mode)
        validator.add_global_validator(validate_load_balancing_dependencies)


# Export key classes
__all__ = [
    'OrchestrationMode',
    'LoadBalancingStrategy', 
    'OrchestrationDefaults',
    'OrchestrationConfiguration'
]