"""
Base Testing Role
=================

Base class for all testing roles in the multi-agent testing framework.
Inspired by MetaGPT's role-based architecture with TestMaster specialization.

Author: TestMaster Team
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
from core.observability import global_observability

class TestActionType(Enum):
    """Types of test actions that roles can perform"""
    ANALYZE = "analyze"
    DESIGN = "design"
    IMPLEMENT = "implement"
    EXECUTE = "execute"
    REVIEW = "review"
    OPTIMIZE = "optimize"
    REPORT = "report"
    COORDINATE = "coordinate"

class MessageType(Enum):
    """Types of messages between roles"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    COLLABORATION_REQUEST = "collaboration_request"
    FEEDBACK = "feedback"
    COMPLETION_NOTICE = "completion_notice"

@dataclass
class TestMessage:
    """Message structure for inter-role communication"""
    id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    sender: str = ""
    recipient: str = ""
    message_type: MessageType = MessageType.TASK_REQUEST
    action_type: Optional[TestActionType] = None
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    requires_response: bool = False
    correlation_id: Optional[str] = None

@dataclass 
class TestAction:
    """Structure for test actions performed by roles"""
    id: str = field(default_factory=lambda: f"action_{uuid.uuid4().hex[:12]}")
    role: str = ""
    action_type: TestActionType = TestActionType.ANALYZE
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    error: Optional[str] = None

class RoleCapability(Enum):
    """Capabilities that roles can have"""
    TEST_ANALYSIS = "test_analysis"
    TEST_DESIGN = "test_design"
    TEST_IMPLEMENTATION = "test_implementation"
    TEST_EXECUTION = "test_execution"
    QUALITY_REVIEW = "quality_review"
    PERFORMANCE_MONITORING = "performance_monitoring"
    COORDINATION = "coordination"
    REPORTING = "reporting"
    OPTIMIZATION = "optimization"
    SECURITY_TESTING = "security_testing"
    INTEGRATION_TESTING = "integration_testing"
    UNIT_TESTING = "unit_testing"

class BaseTestRole(ABC):
    """
    Abstract base class for all testing roles.
    
    Inspired by MetaGPT's BaseRole but specialized for TestMaster's testing domain.
    Provides common functionality for message handling, action execution, and observability.
    """
    
    def __init__(
        self,
        name: str,
        profile: str,
        capabilities: List[RoleCapability],
        max_concurrent_actions: int = 3,
        **kwargs
    ):
        self.name = name
        self.profile = profile
        self.capabilities = capabilities
        self.max_concurrent_actions = max_concurrent_actions
        
        # State management
        self.role_id = f"role_{uuid.uuid4().hex[:12]}"
        self.active = False
        self.busy = False
        
        # Message and action handling
        self.message_queue: List[TestMessage] = []
        self.action_history: List[TestAction] = []
        self.active_actions: Dict[str, TestAction] = {}
        
        # Performance metrics
        self.performance_metrics = {
            "actions_completed": 0,
            "actions_failed": 0,
            "average_action_time": 0.0,
            "total_action_time": 0.0,
            "messages_processed": 0,
            "success_rate": 0.0
        }
        
        # Collaboration
        self.collaborators: Dict[str, 'BaseTestRole'] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        
        # Observability
        self.logger = logging.getLogger(f'TestRole.{self.name}')
        self.observability_session_id = None
        
        # Initialize default message handlers
        self._initialize_message_handlers()
        
        self.logger.info(f"Initialized role {self.name} with capabilities: {[c.value for c in capabilities]}")
    
    def _initialize_message_handlers(self):
        """Initialize default message handlers"""
        self.message_handlers = {
            MessageType.TASK_REQUEST: self._handle_task_request,
            MessageType.COLLABORATION_REQUEST: self._handle_collaboration_request,
            MessageType.FEEDBACK: self._handle_feedback,
            MessageType.STATUS_UPDATE: self._handle_status_update
        }
    
    @abstractmethod
    async def execute_action(self, action: TestAction) -> TestAction:
        """Execute a specific action. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def can_handle_action(self, action_type: TestActionType) -> bool:
        """Check if this role can handle a specific action type."""
        pass
    
    async def start_role(self, observability_session_id: Optional[str] = None):
        """Start the role and begin processing messages"""
        self.active = True
        self.observability_session_id = observability_session_id
        
        if self.observability_session_id:
            global_observability.track_agent_action(
                self.observability_session_id,
                self.name,
                "role_started",
                {"role_id": self.role_id, "capabilities": [c.value for c in self.capabilities]}
            )
        
        self.logger.info(f"Role {self.name} started and ready for action")
        
        # Start message processing loop
        asyncio.create_task(self._message_processing_loop())
    
    async def stop_role(self):
        """Stop the role and complete any pending actions"""
        self.active = False
        
        # Wait for active actions to complete
        while self.active_actions:
            await asyncio.sleep(0.1)
        
        if self.observability_session_id:
            global_observability.track_agent_action(
                self.observability_session_id,
                self.name,
                "role_stopped",
                {"final_metrics": self.performance_metrics}
            )
        
        self.logger.info(f"Role {self.name} stopped")
    
    async def send_message(self, recipient: str, message: TestMessage):
        """Send a message to another role"""
        message.sender = self.name
        message.recipient = recipient
        
        if recipient in self.collaborators:
            await self.collaborators[recipient].receive_message(message)
        else:
            self.logger.warning(f"Unknown recipient: {recipient}")
    
    async def receive_message(self, message: TestMessage):
        """Receive a message from another role"""
        self.message_queue.append(message)
        self.performance_metrics["messages_processed"] += 1
        
        self.logger.debug(f"Received message from {message.sender}: {message.message_type.value}")
    
    async def _message_processing_loop(self):
        """Main message processing loop"""
        while self.active:
            if self.message_queue and not self.busy:
                message = self.message_queue.pop(0)
                await self._process_message(message)
            
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
    
    async def _process_message(self, message: TestMessage):
        """Process a received message"""
        try:
            if message.message_type in self.message_handlers:
                await self.message_handlers[message.message_type](message)
            else:
                self.logger.warning(f"No handler for message type: {message.message_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing message {message.id}: {e}")
    
    async def _handle_task_request(self, message: TestMessage):
        """Handle a task request message"""
        if message.action_type and self.can_handle_action(message.action_type):
            action = TestAction(
                role=self.name,
                action_type=message.action_type,
                description=message.content,
                parameters=message.metadata
            )
            
            result = await self.perform_action(action)
            
            # Send response if required
            if message.requires_response:
                response = TestMessage(
                    recipient=message.sender,
                    message_type=MessageType.TASK_RESPONSE,
                    content=f"Action {action.action_type.value} completed",
                    metadata={"action_id": action.id, "result": result.result, "status": result.status},
                    correlation_id=message.id
                )
                await self.send_message(message.sender, response)
        else:
            self.logger.warning(f"Cannot handle action type: {message.action_type}")
    
    async def _handle_collaboration_request(self, message: TestMessage):
        """Handle a collaboration request"""
        # Default collaboration handling
        response = TestMessage(
            recipient=message.sender,
            message_type=MessageType.TASK_RESPONSE,
            content="Collaboration acknowledged",
            correlation_id=message.id
        )
        await self.send_message(message.sender, response)
    
    async def _handle_feedback(self, message: TestMessage):
        """Handle feedback from other roles"""
        self.logger.info(f"Received feedback from {message.sender}: {message.content}")
    
    async def _handle_status_update(self, message: TestMessage):
        """Handle status updates from other roles"""
        self.logger.debug(f"Status update from {message.sender}: {message.content}")
    
    async def perform_action(self, action: TestAction) -> TestAction:
        """Perform a test action with full lifecycle management"""
        if len(self.active_actions) >= self.max_concurrent_actions:
            action.status = "failed"
            action.error = "Maximum concurrent actions exceeded"
            return action
        
        action.start_time = datetime.now()
        action.status = "running"
        self.active_actions[action.id] = action
        
        try:
            self.busy = True
            
            # Track action start
            if self.observability_session_id:
                obs_action_id = global_observability.track_agent_action(
                    self.observability_session_id,
                    self.name,
                    action.action_type.value,
                    action.parameters
                )
                action.parameters["observability_action_id"] = obs_action_id
            
            # Execute the action
            completed_action = await self.execute_action(action)
            
            # Complete tracking
            if self.observability_session_id and "observability_action_id" in action.parameters:
                global_observability.complete_agent_action(
                    action.parameters["observability_action_id"],
                    result=completed_action.result
                )
            
            completed_action.status = "completed"
            self._update_performance_metrics(completed_action)
            
        except Exception as e:
            action.status = "failed"
            action.error = str(e)
            self.performance_metrics["actions_failed"] += 1
            self.logger.error(f"Action {action.id} failed: {e}")
            
        finally:
            action.end_time = datetime.now()
            if action.start_time:
                action.duration = (action.end_time - action.start_time).total_seconds()
            
            self.action_history.append(action)
            del self.active_actions[action.id]
            self.busy = False
        
        return action
    
    def _update_performance_metrics(self, action: TestAction):
        """Update role performance metrics"""
        self.performance_metrics["actions_completed"] += 1
        self.performance_metrics["total_action_time"] += action.duration
        
        total_actions = self.performance_metrics["actions_completed"] + self.performance_metrics["actions_failed"]
        if total_actions > 0:
            self.performance_metrics["success_rate"] = (
                self.performance_metrics["actions_completed"] / total_actions * 100
            )
            self.performance_metrics["average_action_time"] = (
                self.performance_metrics["total_action_time"] / self.performance_metrics["actions_completed"]
            )
    
    def add_collaborator(self, role: 'BaseTestRole'):
        """Add a collaborating role"""
        self.collaborators[role.name] = role
        self.logger.info(f"Added collaborator: {role.name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current role status"""
        return {
            "role_id": self.role_id,
            "name": self.name,
            "profile": self.profile,
            "capabilities": [c.value for c in self.capabilities],
            "active": self.active,
            "busy": self.busy,
            "queue_length": len(self.message_queue),
            "active_actions": len(self.active_actions),
            "collaborators": list(self.collaborators.keys()),
            "performance_metrics": self.performance_metrics
        }

# Export key components
__all__ = [
    'BaseTestRole',
    'TestMessage',
    'TestAction',
    'TestActionType',
    'MessageType',
    'RoleCapability'
]