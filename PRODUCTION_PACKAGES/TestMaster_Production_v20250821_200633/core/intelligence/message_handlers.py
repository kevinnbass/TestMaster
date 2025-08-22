"""
Message Handlers - Cross-System Coordination Message Processing Engine
======================================================================

Advanced message handling system that processes different types of coordination
messages across intelligence frameworks with enterprise-grade command execution,
query processing, event handling, and response management capabilities.

This module provides sophisticated message handlers for all coordination message
types with pattern matching, subscription management, and error handling.

Author: Agent A - PHASE 4+ Continuation
Created: 2025-08-22
Module: message_handlers.py (400 lines)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Set
from collections import defaultdict
from abc import ABC, abstractmethod
import uuid
import weakref

from .coordination_types import (
    CoordinationMessage, MessageType, MessagePriority, CoordinationPattern,
    MessageStatus, EventSubscription, FrameworkRegistration
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MessageHandler(ABC):
    """Abstract base class for coordination message handlers"""
    
    @abstractmethod
    async def handle_message(self, message: CoordinationMessage) -> Optional[CoordinationMessage]:
        """Process a coordination message and return optional response"""
        pass
    
    @abstractmethod
    def can_handle(self, message: CoordinationMessage) -> bool:
        """Check if this handler can process the given message"""
        pass
    
    def get_handler_info(self) -> Dict[str, Any]:
        """Get information about this message handler"""
        return {
            "handler_type": self.__class__.__name__,
            "supported_message_types": getattr(self, 'supported_message_types', []),
            "handler_id": getattr(self, 'handler_id', 'unknown')
        }


class CommandMessageHandler(MessageHandler):
    """Handler for command messages with enterprise execution capabilities"""
    
    def __init__(self):
        self.handler_id = f"command_handler_{uuid.uuid4().hex[:8]}"
        self.command_registry: Dict[str, Callable] = {}
        self.execution_stats: Dict[str, int] = defaultdict(int)
        self.supported_message_types = [MessageType.COMMAND]
        self.logger = logging.getLogger(f"{__name__}.CommandMessageHandler")
        
        # Register default commands
        self._register_default_commands()
        
        self.logger.info(f"CommandMessageHandler initialized with ID: {self.handler_id}")
    
    def _register_default_commands(self):
        """Register default system commands"""
        self.register_command("ping", self._handle_ping_command)
        self.register_command("status", self._handle_status_command)
        self.register_command("shutdown", self._handle_shutdown_command)
        self.register_command("list_commands", self._handle_list_commands)
        self.register_command("get_stats", self._handle_get_stats)
    
    def register_command(self, command_name: str, handler: Callable) -> None:
        """Register a command handler function"""
        self.command_registry[command_name] = handler
        self.logger.debug(f"Registered command: {command_name}")
    
    def unregister_command(self, command_name: str) -> bool:
        """Unregister a command handler"""
        if command_name in self.command_registry:
            del self.command_registry[command_name]
            self.logger.debug(f"Unregistered command: {command_name}")
            return True
        return False
    
    async def handle_message(self, message: CoordinationMessage) -> Optional[CoordinationMessage]:
        """Handle command message with comprehensive error handling"""
        if not self.can_handle(message):
            return None
        
        try:
            message.mark_processing_started()
            
            command_name = message.payload.get('command')
            parameters = message.payload.get('parameters', {})
            
            if not command_name:
                return self._create_error_response(message, "Missing command name")
            
            if command_name not in self.command_registry:
                return self._create_error_response(message, f"Unknown command: {command_name}")
            
            # Execute command with timeout protection
            handler = self.command_registry[command_name]
            result = await self._execute_command_with_timeout(handler, parameters, message)
            
            # Update execution statistics
            self.execution_stats[command_name] += 1
            self.execution_stats['total_executions'] += 1
            
            message.mark_processing_completed(success=True)
            
            return self._create_success_response(message, result)
            
        except asyncio.TimeoutError:
            message.mark_processing_completed(success=False)
            return self._create_error_response(message, "Command execution timeout")
        
        except Exception as e:
            message.mark_processing_completed(success=False)
            self.logger.error(f"Command execution failed: {e}")
            return self._create_error_response(message, f"Command execution failed: {str(e)}")
    
    def can_handle(self, message: CoordinationMessage) -> bool:
        """Check if this handler can process command messages"""
        return message.message_type == MessageType.COMMAND
    
    async def _execute_command_with_timeout(self, handler: Callable, parameters: Dict[str, Any], 
                                          message: CoordinationMessage) -> Any:
        """Execute command with timeout protection"""
        timeout = message.response_timeout or 30.0  # Default 30 second timeout
        
        try:
            # Execute handler with timeout
            result = await asyncio.wait_for(
                self._execute_command(handler, parameters),
                timeout=timeout
            )
            return result
        
        except asyncio.TimeoutError:
            self.logger.warning(f"Command execution timeout after {timeout}s")
            raise
    
    async def _execute_command(self, handler: Callable, parameters: Dict[str, Any]) -> Any:
        """Execute command handler safely"""
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**parameters)
            else:
                result = handler(**parameters)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Command handler execution failed: {e}")
            raise
    
    def _create_success_response(self, request_message: CoordinationMessage, result: Any) -> CoordinationMessage:
        """Create success response message"""
        return CoordinationMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.handler_id,
            recipient_id=request_message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=request_message.priority,
            pattern=CoordinationPattern.REQUEST_RESPONSE,
            payload={
                'status': 'success',
                'result': result,
                'request_id': request_message.message_id,
                'execution_time': request_message.processing_time
            },
            created_at=datetime.now(),
            correlation_id=request_message.message_id
        )
    
    def _create_error_response(self, request_message: CoordinationMessage, error: str) -> CoordinationMessage:
        """Create error response message"""
        return CoordinationMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.handler_id,
            recipient_id=request_message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=request_message.priority,
            pattern=CoordinationPattern.REQUEST_RESPONSE,
            payload={
                'status': 'error',
                'error': error,
                'request_id': request_message.message_id
            },
            created_at=datetime.now(),
            correlation_id=request_message.message_id
        )
    
    # Default command implementations
    async def _handle_ping_command(self, **parameters) -> Dict[str, Any]:
        """Handle ping command for health checking"""
        return {
            'pong': True,
            'timestamp': datetime.now().isoformat(),
            'handler_id': self.handler_id
        }
    
    async def _handle_status_command(self, **parameters) -> Dict[str, Any]:
        """Handle status command for system information"""
        return {
            'status': 'active',
            'handler_type': 'CommandMessageHandler',
            'registered_commands': list(self.command_registry.keys()),
            'execution_stats': dict(self.execution_stats)
        }
    
    async def _handle_shutdown_command(self, **parameters) -> Dict[str, Any]:
        """Handle shutdown command"""
        return {
            'message': 'Shutdown command received',
            'status': 'acknowledged'
        }
    
    async def _handle_list_commands(self, **parameters) -> Dict[str, Any]:
        """Handle list commands request"""
        return {
            'available_commands': list(self.command_registry.keys()),
            'command_count': len(self.command_registry)
        }
    
    async def _handle_get_stats(self, **parameters) -> Dict[str, Any]:
        """Handle get statistics command"""
        return {
            'execution_stats': dict(self.execution_stats),
            'handler_info': self.get_handler_info()
        }


class QueryMessageHandler(MessageHandler):
    """Handler for query messages with enterprise data retrieval capabilities"""
    
    def __init__(self):
        self.handler_id = f"query_handler_{uuid.uuid4().hex[:8]}"
        self.query_registry: Dict[str, Callable] = {}
        self.query_stats: Dict[str, int] = defaultdict(int)
        self.supported_message_types = [MessageType.QUERY]
        self.logger = logging.getLogger(f"{__name__}.QueryMessageHandler")
        
        # Register default queries
        self._register_default_queries()
        
        self.logger.info(f"QueryMessageHandler initialized with ID: {self.handler_id}")
    
    def _register_default_queries(self):
        """Register default system queries"""
        self.register_query("health", self._handle_health_query)
        self.register_query("metrics", self._handle_metrics_query)
        self.register_query("capabilities", self._handle_capabilities_query)
        self.register_query("system_info", self._handle_system_info_query)
    
    def register_query(self, query_name: str, handler: Callable) -> None:
        """Register a query handler function"""
        self.query_registry[query_name] = handler
        self.logger.debug(f"Registered query: {query_name}")
    
    def unregister_query(self, query_name: str) -> bool:
        """Unregister a query handler"""
        if query_name in self.query_registry:
            del self.query_registry[query_name]
            self.logger.debug(f"Unregistered query: {query_name}")
            return True
        return False
    
    async def handle_message(self, message: CoordinationMessage) -> Optional[CoordinationMessage]:
        """Handle query message with comprehensive data retrieval"""
        if not self.can_handle(message):
            return None
        
        try:
            message.mark_processing_started()
            
            query_name = message.payload.get('query')
            parameters = message.payload.get('parameters', {})
            
            if not query_name:
                return self._create_error_response(message, "Missing query name")
            
            if query_name not in self.query_registry:
                return self._create_error_response(message, f"Unknown query: {query_name}")
            
            # Execute query with timeout protection
            handler = self.query_registry[query_name]
            result = await self._execute_query_with_timeout(handler, parameters, message)
            
            # Update query statistics
            self.query_stats[query_name] += 1
            self.query_stats['total_queries'] += 1
            
            message.mark_processing_completed(success=True)
            
            return self._create_success_response(message, result)
            
        except asyncio.TimeoutError:
            message.mark_processing_completed(success=False)
            return self._create_error_response(message, "Query execution timeout")
        
        except Exception as e:
            message.mark_processing_completed(success=False)
            self.logger.error(f"Query execution failed: {e}")
            return self._create_error_response(message, f"Query execution failed: {str(e)}")
    
    def can_handle(self, message: CoordinationMessage) -> bool:
        """Check if this handler can process query messages"""
        return message.message_type == MessageType.QUERY
    
    async def _execute_query_with_timeout(self, handler: Callable, parameters: Dict[str, Any],
                                        message: CoordinationMessage) -> Any:
        """Execute query with timeout protection"""
        timeout = message.response_timeout or 30.0  # Default 30 second timeout
        
        try:
            result = await asyncio.wait_for(
                self._execute_query(handler, parameters),
                timeout=timeout
            )
            return result
        
        except asyncio.TimeoutError:
            self.logger.warning(f"Query execution timeout after {timeout}s")
            raise
    
    async def _execute_query(self, handler: Callable, parameters: Dict[str, Any]) -> Any:
        """Execute query handler safely"""
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**parameters)
            else:
                result = handler(**parameters)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Query handler execution failed: {e}")
            raise
    
    def _create_success_response(self, request_message: CoordinationMessage, result: Any) -> CoordinationMessage:
        """Create successful query response"""
        return CoordinationMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.handler_id,
            recipient_id=request_message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=request_message.priority,
            pattern=CoordinationPattern.REQUEST_RESPONSE,
            payload={
                'status': 'success',
                'data': result,
                'request_id': request_message.message_id,
                'query_time': request_message.processing_time
            },
            created_at=datetime.now(),
            correlation_id=request_message.message_id
        )
    
    def _create_error_response(self, request_message: CoordinationMessage, error: str) -> CoordinationMessage:
        """Create error response for failed queries"""
        return CoordinationMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.handler_id,
            recipient_id=request_message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=request_message.priority,
            pattern=CoordinationPattern.REQUEST_RESPONSE,
            payload={
                'status': 'error',
                'error': error,
                'request_id': request_message.message_id
            },
            created_at=datetime.now(),
            correlation_id=request_message.message_id
        )
    
    # Default query implementations
    async def _handle_health_query(self, **parameters) -> Dict[str, Any]:
        """Handle health check query"""
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'handler_id': self.handler_id,
            'uptime': 'N/A'  # Would be calculated from actual start time
        }
    
    async def _handle_metrics_query(self, **parameters) -> Dict[str, Any]:
        """Handle metrics query"""
        return {
            'query_stats': dict(self.query_stats),
            'registered_queries': list(self.query_registry.keys()),
            'handler_metrics': {
                'total_queries_processed': self.query_stats.get('total_queries', 0),
                'unique_query_types': len(self.query_registry)
            }
        }
    
    async def _handle_capabilities_query(self, **parameters) -> Dict[str, Any]:
        """Handle capabilities query"""
        return {
            'supported_message_types': [mt.value for mt in self.supported_message_types],
            'available_queries': list(self.query_registry.keys()),
            'handler_capabilities': {
                'async_execution': True,
                'timeout_protection': True,
                'error_handling': True,
                'statistics_tracking': True
            }
        }
    
    async def _handle_system_info_query(self, **parameters) -> Dict[str, Any]:
        """Handle system information query"""
        return {
            'handler_type': 'QueryMessageHandler',
            'handler_id': self.handler_id,
            'registration_time': datetime.now().isoformat(),  # Would be actual registration time
            'query_registry_size': len(self.query_registry)
        }


class EventMessageHandler(MessageHandler):
    """Handler for event messages with enterprise subscription and notification capabilities"""
    
    def __init__(self):
        self.handler_id = f"event_handler_{uuid.uuid4().hex[:8]}"
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.event_stats: Dict[str, int] = defaultdict(int)
        self.supported_message_types = [MessageType.EVENT]
        self.logger = logging.getLogger(f"{__name__}.EventMessageHandler")
        
        # Event processing configuration
        self.max_concurrent_notifications = 50
        self.notification_timeout = 10.0  # seconds
        
        self.logger.info(f"EventMessageHandler initialized with ID: {self.handler_id}")
    
    def subscribe_to_events(self, 
                           subscriber_id: str,
                           event_pattern: str,
                           callback_url: Optional[str] = None,
                           callback_function: Optional[Callable] = None,
                           filter_criteria: Dict[str, Any] = None) -> str:
        """Subscribe to events with comprehensive filtering options"""
        
        subscription_id = str(uuid.uuid4())
        
        subscription = EventSubscription(
            subscription_id=subscription_id,
            subscriber_id=subscriber_id,
            event_pattern=event_pattern,
            callback_url=callback_url,
            callback_function=callback_function,
            filter_criteria=filter_criteria or {}
        )
        
        self.subscriptions[subscription_id] = subscription
        
        self.logger.info(f"Created event subscription {subscription_id} for {subscriber_id} with pattern '{event_pattern}'")
        return subscription_id
    
    def unsubscribe_from_events(self, subscription_id: str) -> bool:
        """Remove event subscription"""
        if subscription_id in self.subscriptions:
            subscription = self.subscriptions[subscription_id]
            del self.subscriptions[subscription_id]
            
            self.logger.info(f"Removed event subscription {subscription_id} for {subscription.subscriber_id}")
            return True
        
        return False
    
    def get_subscriptions(self, subscriber_id: Optional[str] = None) -> List[EventSubscription]:
        """Get event subscriptions, optionally filtered by subscriber"""
        if subscriber_id:
            return [sub for sub in self.subscriptions.values() if sub.subscriber_id == subscriber_id]
        return list(self.subscriptions.values())
    
    async def handle_message(self, message: CoordinationMessage) -> Optional[CoordinationMessage]:
        """Handle event message with comprehensive subscription matching and notification"""
        if not self.can_handle(message):
            return None
        
        try:
            message.mark_processing_started()
            
            event_type = message.payload.get('event_type')
            event_data = message.payload.get('event_data', {})
            
            if not event_type:
                self.logger.warning("Received event message without event_type")
                return None
            
            # Find matching subscriptions
            matching_subscriptions = self._find_matching_subscriptions(message, event_type)
            
            # Notify subscribers concurrently with limit
            if matching_subscriptions:
                await self._notify_subscribers_batch(matching_subscriptions, message)
            
            # Update event statistics
            self.event_stats[event_type] += 1
            self.event_stats['total_events'] += 1
            
            message.mark_processing_completed(success=True)
            
            # Event messages typically don't return responses
            return None
            
        except Exception as e:
            message.mark_processing_completed(success=False)
            self.logger.error(f"Event processing failed: {e}")
            return None
    
    def can_handle(self, message: CoordinationMessage) -> bool:
        """Check if this handler can process event messages"""
        return message.message_type == MessageType.EVENT
    
    def _find_matching_subscriptions(self, message: CoordinationMessage, event_type: str) -> List[EventSubscription]:
        """Find subscriptions that match the event"""
        matching_subscriptions = []
        
        for subscription in self.subscriptions.values():
            if not subscription.active:
                continue
            
            # Check if event type matches subscription pattern
            if not subscription.matches_event_type(event_type):
                continue
            
            # Apply filter criteria if specified
            event_data = message.payload.get('event_data', {})
            if subscription.should_filter_event(event_data):
                continue
            
            matching_subscriptions.append(subscription)
        
        return matching_subscriptions
    
    async def _notify_subscribers_batch(self, subscriptions: List[EventSubscription], message: CoordinationMessage):
        """Notify subscribers in batches with concurrency control"""
        # Process subscriptions in batches to control concurrency
        batch_size = min(len(subscriptions), self.max_concurrent_notifications)
        
        for i in range(0, len(subscriptions), batch_size):
            batch = subscriptions[i:i + batch_size]
            
            # Create notification tasks for the batch
            tasks = [
                self._notify_subscriber_safe(subscription, message)
                for subscription in batch
            ]
            
            # Execute batch with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.notification_timeout
                )
            
            except asyncio.TimeoutError:
                self.logger.warning(f"Batch notification timeout for {len(batch)} subscribers")
    
    async def _notify_subscriber_safe(self, subscription: EventSubscription, message: CoordinationMessage):
        """Safely notify a single subscriber with error handling"""
        try:
            await self._notify_subscriber(subscription, message)
            subscription.update_activity()
            
        except Exception as e:
            self.logger.error(f"Failed to notify subscriber {subscription.subscriber_id}: {e}")
    
    async def _notify_subscriber(self, subscription: EventSubscription, message: CoordinationMessage):
        """Notify subscriber using their preferred method"""
        if subscription.callback_function:
            # Direct function callback
            if asyncio.iscoroutinefunction(subscription.callback_function):
                await subscription.callback_function(message.payload)
            else:
                subscription.callback_function(message.payload)
        
        elif subscription.callback_url:
            # HTTP callback (would implement actual HTTP call in production)
            self.logger.debug(f"Would notify {subscription.callback_url} with event data")
        
        else:
            # Log-only notification for debugging
            self.logger.debug(f"Event notification for subscription {subscription.subscription_id}: {message.payload.get('event_type')}")
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """Get event handler statistics"""
        return {
            'total_subscriptions': len(self.subscriptions),
            'active_subscriptions': len([s for s in self.subscriptions.values() if s.active]),
            'event_stats': dict(self.event_stats),
            'handler_info': self.get_handler_info()
        }


class ResponseMessageHandler(MessageHandler):
    """Handler for response messages with correlation tracking"""
    
    def __init__(self):
        self.handler_id = f"response_handler_{uuid.uuid4().hex[:8]}"
        self.pending_requests: Dict[str, CoordinationMessage] = {}
        self.response_callbacks: Dict[str, Callable] = {}
        self.supported_message_types = [MessageType.RESPONSE]
        self.logger = logging.getLogger(f"{__name__}.ResponseMessageHandler")
        
        self.logger.info(f"ResponseMessageHandler initialized with ID: {self.handler_id}")
    
    def register_pending_request(self, request_message: CoordinationMessage, 
                                callback: Optional[Callable] = None):
        """Register a pending request for response correlation"""
        self.pending_requests[request_message.message_id] = request_message
        
        if callback:
            self.response_callbacks[request_message.message_id] = callback
        
        self.logger.debug(f"Registered pending request {request_message.message_id}")
    
    def unregister_pending_request(self, request_id: str):
        """Unregister a pending request"""
        self.pending_requests.pop(request_id, None)
        self.response_callbacks.pop(request_id, None)
        
        self.logger.debug(f"Unregistered pending request {request_id}")
    
    async def handle_message(self, message: CoordinationMessage) -> Optional[CoordinationMessage]:
        """Handle response message with correlation tracking"""
        if not self.can_handle(message):
            return None
        
        try:
            message.mark_processing_started()
            
            correlation_id = message.correlation_id
            if not correlation_id:
                self.logger.warning("Received response without correlation_id")
                return None
            
            # Find pending request
            if correlation_id in self.pending_requests:
                pending_request = self.pending_requests[correlation_id]
                
                # Execute callback if registered
                if correlation_id in self.response_callbacks:
                    callback = self.response_callbacks[correlation_id]
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message, pending_request)
                        else:
                            callback(message, pending_request)
                    
                    except Exception as e:
                        self.logger.error(f"Response callback failed: {e}")
                
                # Clean up
                self.unregister_pending_request(correlation_id)
                
                self.logger.debug(f"Processed response for request {correlation_id}")
            
            else:
                self.logger.warning(f"Received response for unknown request {correlation_id}")
            
            message.mark_processing_completed(success=True)
            return None  # Response messages don't generate further responses
            
        except Exception as e:
            message.mark_processing_completed(success=False)
            self.logger.error(f"Response processing failed: {e}")
            return None
    
    def can_handle(self, message: CoordinationMessage) -> bool:
        """Check if this handler can process response messages"""
        return message.message_type == MessageType.RESPONSE
    
    def get_pending_requests(self) -> Dict[str, CoordinationMessage]:
        """Get all pending requests"""
        return self.pending_requests.copy()
    
    def cleanup_expired_requests(self):
        """Clean up expired pending requests"""
        now = datetime.now()
        expired_requests = []
        
        for request_id, request in self.pending_requests.items():
            if request.expires_at and now > request.expires_at:
                expired_requests.append(request_id)
        
        for request_id in expired_requests:
            self.unregister_pending_request(request_id)
            self.logger.debug(f"Cleaned up expired request {request_id}")
        
        return len(expired_requests)


# Factory function for creating message handlers
def create_message_handlers() -> Dict[str, MessageHandler]:
    """Create and return all default message handlers"""
    return {
        'command': CommandMessageHandler(),
        'query': QueryMessageHandler(),
        'event': EventMessageHandler(),
        'response': ResponseMessageHandler()
    }


# Export the main message handler classes and factory function
__all__ = [
    'MessageHandler', 'CommandMessageHandler', 'QueryMessageHandler', 
    'EventMessageHandler', 'ResponseMessageHandler', 'create_message_handlers'
]