"""
Message Handlers for Coordination Protocol

This module implements specialized handlers for different message types
including commands, queries, and events.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

from .message_types import (
    CoordinationMessage, MessageType, MessagePriority,
    CoordinationPattern, EventSubscription
)


class MessageHandler(ABC):
    """Abstract base class for message handlers"""
    
    @abstractmethod
    async def handle_message(self, message: CoordinationMessage) -> Optional[CoordinationMessage]:
        """Handle incoming message and optionally return response"""
        pass
    
    @abstractmethod
    def can_handle(self, message: CoordinationMessage) -> bool:
        """Check if handler can process this message"""
        pass


class CommandMessageHandler(MessageHandler):
    """Handler for command messages"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.command_registry: Dict[str, Callable] = {}
    
    def register_command(self, command_name: str, handler: Callable) -> None:
        """Register a command handler"""
        self.command_registry[command_name] = handler
        self.logger.info(f"Registered command handler: {command_name}")
    
    async def handle_message(self, message: CoordinationMessage) -> Optional[CoordinationMessage]:
        """Handle command message"""
        try:
            command_name = message.payload.get('command')
            if command_name not in self.command_registry:
                return self._create_error_response(message, f"Unknown command: {command_name}")
            
            handler = self.command_registry[command_name]
            result = await self._execute_command(handler, message.payload.get('parameters', {}))
            
            return self._create_success_response(message, result)
            
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return self._create_error_response(message, str(e))
    
    def can_handle(self, message: CoordinationMessage) -> bool:
        """Check if can handle command message"""
        return message.message_type == MessageType.COMMAND
    
    async def _execute_command(self, handler: Callable, parameters: Dict[str, Any]) -> Any:
        """Execute command handler"""
        if asyncio.iscoroutinefunction(handler):
            return await handler(**parameters)
        else:
            return handler(**parameters)
    
    def _create_success_response(self, request_message: CoordinationMessage, result: Any) -> CoordinationMessage:
        """Create success response message"""
        return CoordinationMessage(
            message_id=str(uuid.uuid4()),
            sender_id="command_handler",
            recipient_id=request_message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=request_message.priority,
            pattern=CoordinationPattern.REQUEST_RESPONSE,
            payload={'status': 'success', 'result': result},
            created_at=datetime.now(),
            correlation_id=request_message.message_id
        )
    
    def _create_error_response(self, request_message: CoordinationMessage, error: str) -> CoordinationMessage:
        """Create error response message"""
        return CoordinationMessage(
            message_id=str(uuid.uuid4()),
            sender_id="command_handler",
            recipient_id=request_message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=request_message.priority,
            pattern=CoordinationPattern.REQUEST_RESPONSE,
            payload={'status': 'error', 'error': error},
            created_at=datetime.now(),
            correlation_id=request_message.message_id
        )


class QueryMessageHandler(MessageHandler):
    """Handler for query messages"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.query_registry: Dict[str, Callable] = {}
    
    def register_query(self, query_name: str, handler: Callable) -> None:
        """Register a query handler"""
        self.query_registry[query_name] = handler
        self.logger.info(f"Registered query handler: {query_name}")
    
    async def handle_message(self, message: CoordinationMessage) -> Optional[CoordinationMessage]:
        """Handle query message"""
        try:
            query_name = message.payload.get('query')
            if query_name not in self.query_registry:
                return self._create_error_response(message, f"Unknown query: {query_name}")
            
            handler = self.query_registry[query_name]
            result = await self._execute_query(handler, message.payload.get('parameters', {}))
            
            return self._create_success_response(message, result)
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            return self._create_error_response(message, str(e))
    
    def can_handle(self, message: CoordinationMessage) -> bool:
        """Check if can handle query message"""
        return message.message_type == MessageType.QUERY
    
    async def _execute_query(self, handler: Callable, parameters: Dict[str, Any]) -> Any:
        """Execute query handler"""
        if asyncio.iscoroutinefunction(handler):
            return await handler(**parameters)
        else:
            return handler(**parameters)
    
    def _create_success_response(self, request_message: CoordinationMessage, result: Any) -> CoordinationMessage:
        """Create success response message"""
        return CoordinationMessage(
            message_id=str(uuid.uuid4()),
            sender_id="query_handler",
            recipient_id=request_message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=request_message.priority,
            pattern=CoordinationPattern.REQUEST_RESPONSE,
            payload={'status': 'success', 'data': result},
            created_at=datetime.now(),
            correlation_id=request_message.message_id
        )
    
    def _create_error_response(self, request_message: CoordinationMessage, error: str) -> CoordinationMessage:
        """Create error response message"""
        return CoordinationMessage(
            message_id=str(uuid.uuid4()),
            sender_id="query_handler",
            recipient_id=request_message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=request_message.priority,
            pattern=CoordinationPattern.REQUEST_RESPONSE,
            payload={'status': 'error', 'error': error},
            created_at=datetime.now(),
            correlation_id=request_message.message_id
        )


class EventMessageHandler(MessageHandler):
    """Handler for event messages"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.event_subscriptions: Dict[str, EventSubscription] = {}
    
    def subscribe_to_events(self, 
                          subscriber_id: str,
                          event_pattern: str,
                          callback: Optional[Callable] = None,
                          filter_criteria: Dict[str, Any] = None) -> str:
        """Subscribe to events matching pattern"""
        subscription_id = str(uuid.uuid4())
        
        subscription = EventSubscription(
            subscription_id=subscription_id,
            subscriber_id=subscriber_id,
            event_pattern=event_pattern,
            callback=callback,
            filter_criteria=filter_criteria or {}
        )
        
        self.event_subscriptions[subscription_id] = subscription
        self.logger.info(f"Created event subscription: {subscriber_id} -> {event_pattern}")
        
        return subscription_id
    
    def unsubscribe_from_events(self, subscription_id: str) -> bool:
        """Unsubscribe from events"""
        if subscription_id in self.event_subscriptions:
            del self.event_subscriptions[subscription_id]
            self.logger.info(f"Removed event subscription: {subscription_id}")
            return True
        return False
    
    async def handle_message(self, message: CoordinationMessage) -> Optional[CoordinationMessage]:
        """Handle event message by routing to subscribers"""
        try:
            event_type = message.payload.get('event_type', '')
            
            # Find matching subscriptions
            matching_subscriptions = self._find_matching_subscriptions(message, event_type)
            
            # Notify subscribers
            for subscription in matching_subscriptions:
                await self._notify_subscriber(subscription, message)
            
            # Events don't typically require responses
            return None
            
        except Exception as e:
            self.logger.error(f"Event handling failed: {e}")
            return None
    
    def can_handle(self, message: CoordinationMessage) -> bool:
        """Check if can handle event message"""
        return message.message_type == MessageType.EVENT
    
    def _find_matching_subscriptions(self, message: CoordinationMessage, event_type: str) -> List[EventSubscription]:
        """Find subscriptions matching the event"""
        matching = []
        
        for subscription in self.event_subscriptions.values():
            if not subscription.active:
                continue
            
            # Simple pattern matching (would be more sophisticated in production)
            if self._pattern_matches(subscription.event_pattern, event_type):
                if self._filter_matches(subscription.filter_criteria, message.payload):
                    matching.append(subscription)
        
        return matching
    
    def _pattern_matches(self, pattern: str, event_type: str) -> bool:
        """Check if pattern matches event type"""
        # Simple wildcard matching
        if pattern == '*':
            return True
        if pattern == event_type:
            return True
        if pattern.endswith('*'):
            prefix = pattern[:-1]
            return event_type.startswith(prefix)
        return False
    
    def _filter_matches(self, filter_criteria: Dict[str, Any], payload: Dict[str, Any]) -> bool:
        """Check if event payload matches filter criteria"""
        for key, expected_value in filter_criteria.items():
            if payload.get(key) != expected_value:
                return False
        return True
    
    async def _notify_subscriber(self, subscription: EventSubscription, message: CoordinationMessage) -> None:
        """Notify subscriber of matching event"""
        try:
            if subscription.callback:
                if asyncio.iscoroutinefunction(subscription.callback):
                    await subscription.callback(message)
                else:
                    subscription.callback(message)
            else:
                # Would route to subscriber through message routing in production
                self.logger.debug(f"Event notification: {subscription.subscriber_id} <- {message.payload.get('event_type')}")
        except Exception as e:
            self.logger.error(f"Subscriber notification failed: {e}")


# ========================================================================
# ENHANCED HOURS 30-40: ORCHESTRATION BASE INTEGRATION FOR MESSAGE HANDLING
# ========================================================================

# Enhanced orchestration imports
try:
    from ..foundations.abstractions.orchestrator_base import (
        OrchestratorBase, OrchestratorType, OrchestratorCapabilities, ExecutionStrategy
    )
    ORCHESTRATION_BASE_AVAILABLE = True
except ImportError:
    ORCHESTRATION_BASE_AVAILABLE = False
    logger.warning("OrchestratorBase not available for message handler integration")


class OrchestrationAwareMessageHandler(MessageHandler):
    """Enhanced message handler with orchestration base integration"""
    
    def __init__(self, enable_orchestration: bool = True):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.orchestration_base = None
        self.orchestration_enabled = False
        
        if enable_orchestration and ORCHESTRATION_BASE_AVAILABLE:
            self._initialize_orchestration_integration()
    
    def _initialize_orchestration_integration(self):
        """Initialize orchestration base integration for message handling"""
        try:
            # Create embedded orchestrator for message coordination
            class MessageOrchestrator(OrchestratorBase):
                """Message-specialized orchestrator"""
                
                def __init__(self, message_handler):
                    super().__init__(
                        orchestrator_type=OrchestratorType.SPECIALIZED,
                        name="MessageOrchestrator"
                    )
                    self.message_handler = message_handler
                    
                    # Set enhanced coordination capabilities
                    self.capabilities.supports_cross_system_coordination = True
                    self.capabilities.supports_intelligent_routing = True
                    self.capabilities.supports_adaptive_execution = True
                    self.capabilities.workflow_patterns.update({
                        'message_routing', 'event_coordination', 'protocol_management'
                    })
                
                async def execute_task(self, task: Any) -> Any:
                    """Execute message processing task"""
                    if isinstance(task, dict) and task.get('type') == 'message_processing':
                        message_data = task.get('message')
                        if message_data:
                            # Convert dict back to CoordinationMessage if needed
                            if isinstance(message_data, dict):
                                message = CoordinationMessage(**message_data)
                            else:
                                message = message_data
                            
                            result = await self.message_handler.handle_message(message)
                            return {"status": "completed", "response": result}
                    
                    return {"status": "completed", "result": task}
                
                async def coordinate_cross_system(self, systems: List[str], coordination_plan: Dict[str, Any]) -> Dict[str, Any]:
                    """Coordinate message handling across systems"""
                    results = {}
                    for system in systems:
                        results[system] = {"coordinated": True, "plan": coordination_plan}
                    return results
                
                def get_supported_capabilities(self) -> OrchestratorCapabilities:
                    """Get message orchestrator capabilities"""
                    return self.capabilities
            
            # Initialize the embedded orchestrator
            self.orchestration_base = MessageOrchestrator(self)
            self.orchestration_enabled = True
            
            self.logger.info("Message handler enhanced with orchestration base integration")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestration base integration: {e}")
            self.orchestration_enabled = False
    
    async def handle_message_with_orchestration(self, message: CoordinationMessage) -> Optional[CoordinationMessage]:
        """Handle message through orchestration base if available"""
        if not self.orchestration_enabled:
            # Fall back to direct message handling
            return await self.handle_message(message)
        
        # Process through orchestration base
        task = {
            "type": "message_processing",
            "message": message.__dict__ if hasattr(message, '__dict__') else message
        }
        
        result = await self.orchestration_base.execute_task(task)
        
        # Extract response from orchestration result
        if result.get("status") == "completed" and "response" in result:
            return result["response"]
        
        return None
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get orchestration integration status"""
        if not self.orchestration_enabled or not self.orchestration_base:
            return {"orchestration_enabled": False}
        
        return {
            "orchestration_enabled": True,
            "orchestrator_name": self.orchestration_base.name,
            "orchestrator_type": self.orchestration_base.orchestrator_type.value,
            "capabilities": {
                "cross_system_coordination": self.orchestration_base.capabilities.supports_cross_system_coordination,
                "intelligent_routing": self.orchestration_base.capabilities.supports_intelligent_routing,
                "adaptive_execution": self.orchestration_base.capabilities.supports_adaptive_execution
            },
            "coordination_patterns": list(self.orchestration_base.capabilities.workflow_patterns)
        }


class EnhancedCommandMessageHandler(OrchestrationAwareMessageHandler, CommandMessageHandler):
    """Command handler with orchestration integration"""
    
    def __init__(self, enable_orchestration: bool = True):
        CommandMessageHandler.__init__(self)
        OrchestrationAwareMessageHandler.__init__(self, enable_orchestration)
    
    async def handle_message(self, message: CoordinationMessage) -> Optional[CoordinationMessage]:
        """Handle command message with optional orchestration"""
        if self.orchestration_enabled:
            return await self.handle_message_with_orchestration(message)
        else:
            return await CommandMessageHandler.handle_message(self, message)


class EnhancedQueryMessageHandler(OrchestrationAwareMessageHandler, QueryMessageHandler):
    """Query handler with orchestration integration"""
    
    def __init__(self, enable_orchestration: bool = True):
        QueryMessageHandler.__init__(self)
        OrchestrationAwareMessageHandler.__init__(self, enable_orchestration)
    
    async def handle_message(self, message: CoordinationMessage) -> Optional[CoordinationMessage]:
        """Handle query message with optional orchestration"""
        if self.orchestration_enabled:
            return await self.handle_message_with_orchestration(message)
        else:
            return await QueryMessageHandler.handle_message(self, message)


class EnhancedEventMessageHandler(OrchestrationAwareMessageHandler, EventMessageHandler):
    """Event handler with orchestration integration"""
    
    def __init__(self, enable_orchestration: bool = True):
        EventMessageHandler.__init__(self)
        OrchestrationAwareMessageHandler.__init__(self, enable_orchestration)
    
    async def handle_message(self, message: CoordinationMessage) -> Optional[CoordinationMessage]:
        """Handle event message with optional orchestration"""
        if self.orchestration_enabled:
            return await self.handle_message_with_orchestration(message)
        else:
            return await EventMessageHandler.handle_message(self, message)