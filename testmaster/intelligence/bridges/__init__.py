"""
Bridge Components for TestMaster Deep Integration

These bridge components connect various systems within TestMaster to create
a unified, intelligent testing platform. Each bridge handles a specific
aspect of cross-system communication and coordination.

Phase 4: Bridge Implementation Agents
- Protocol Communication Bridge: Agent communication and message routing
- Event Monitoring Bridge: Unified event bus and component connectivity  
- Session Tracking Bridge: Session management and state persistence
- SOP Workflow Bridge: Standard Operating Procedure workflow patterns
- Context Variables Bridge: Context passing and inheritance across systems
"""

from .protocol_communication_bridge import (
    ProtocolCommunicationBridge,
    MessageProtocol,
    RoutingStrategy,
    CommunicationChannel,
    MessageBus,
    MessageRouter,
    AgentCommunicator,
    get_protocol_bridge
)

from .event_monitoring_bridge import (
    EventMonitoringBridge,
    EventType,
    EventSeverity,
    TestMasterEvent,
    EventBus,
    EventPersistence,
    EventCorrelationEngine,
    get_event_monitoring_bridge
)

from .session_tracking_bridge import (
    SessionTrackingBridge,
    SessionStatus,
    SessionType,
    StateScope,
    SessionManager,
    SessionMetadata,
    SessionState,
    SessionCheckpoint,
    get_session_tracking_bridge
)

from .sop_workflow_bridge import (
    SOPWorkflowBridge,
    SOPType,
    SOPComplexity,
    SOPStatus,
    SOPTemplate,
    SOPStep,
    SOPExecution,
    SOPPattern,
    get_sop_workflow_bridge
)

from .context_variables_bridge import (
    ContextVariablesBridge,
    ContextScope,
    ContextType,
    ContextAccess,
    ContextVariable,
    ContextNamespace,
    ContextManager,
    get_context_variables_bridge
)

__all__ = [
    'ProtocolCommunicationBridge',
    'MessageProtocol',
    'RoutingStrategy', 
    'CommunicationChannel',
    'MessageBus',
    'MessageRouter',
    'AgentCommunicator',
    'get_protocol_bridge',
    'EventMonitoringBridge',
    'EventType',
    'EventSeverity',
    'TestMasterEvent',
    'EventBus',
    'EventPersistence',
    'EventCorrelationEngine',
    'get_event_monitoring_bridge',
    'SessionTrackingBridge',
    'SessionStatus',
    'SessionType',
    'StateScope',
    'SessionManager',
    'SessionMetadata',
    'SessionState',
    'SessionCheckpoint',
    'get_session_tracking_bridge',
    'SOPWorkflowBridge',
    'SOPType',
    'SOPComplexity',
    'SOPStatus',
    'SOPTemplate',
    'SOPStep',
    'SOPExecution',
    'SOPPattern',
    'get_sop_workflow_bridge',
    'ContextVariablesBridge',
    'ContextScope',
    'ContextType',
    'ContextAccess',
    'ContextVariable',
    'ContextNamespace',
    'ContextManager',
    'get_context_variables_bridge'
]