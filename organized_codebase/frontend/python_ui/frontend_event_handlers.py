#!/usr/bin/env python3
"""
Frontend Event Handlers - Atomic Component
Handles frontend-specific event processing
Agent Z - STEELCLAD Frontend Atomization
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict


class EventType(Enum):
    """Frontend event types"""
    CLICK = "click"
    SUBMIT = "submit"
    LOAD = "load"
    REFRESH = "refresh"
    FILTER = "filter"
    SORT = "sort"
    EXPORT = "export"
    TOGGLE = "toggle"
    RESIZE = "resize"
    ERROR = "error"


@dataclass
class FrontendEvent:
    """Frontend event data structure"""
    event_type: EventType
    component_id: str
    payload: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'type': self.event_type.value,
            'component_id': self.component_id,
            'payload': self.payload,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id
        }


class FrontendEventHandlers:
    """
    Frontend-specific event handling component
    Processes and responds to dashboard UI events
    """
    
    def __init__(self):
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.event_history = []
        self.max_history = 1000
        
        # Event processing metrics
        self.metrics = {
            'events_processed': 0,
            'events_failed': 0,
            'avg_processing_time': 0.0,
            'handlers_registered': 0
        }
        
        # Component state tracking
        self.component_states: Dict[str, Dict[str, Any]] = {}
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default event handlers"""
        self.register_handler(EventType.ERROR, self._handle_error_event)
        self.register_handler(EventType.LOAD, self._handle_load_event)
        self.register_handler(EventType.REFRESH, self._handle_refresh_event)
    
    def register_handler(self, event_type: EventType, handler: Callable):
        """Register event handler for specific event type"""
        self.event_handlers[event_type].append(handler)
        self.metrics['handlers_registered'] = sum(
            len(handlers) for handlers in self.event_handlers.values()
        )
    
    async def handle_frontend_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming frontend event
        Main interface for processing dashboard UI events
        """
        import time
        start_time = time.time()
        
        try:
            # Parse event
            event = self._parse_event(event_data)
            
            # Add to history
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history = self.event_history[-self.max_history:]
            
            # Process event
            response = await self._process_event(event)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics['avg_processing_time'] = (
                (self.metrics['avg_processing_time'] * 0.9) + (processing_time * 0.1)
            )
            self.metrics['events_processed'] += 1
            
            return {
                'success': True,
                'event_id': f"evt_{int(event.timestamp.timestamp())}",
                'response': response,
                'processing_time_ms': processing_time * 1000
            }
            
        except Exception as e:
            self.metrics['events_failed'] += 1
            return {
                'success': False,
                'error': str(e),
                'event_data': event_data
            }
    
    def _parse_event(self, event_data: Dict[str, Any]) -> FrontendEvent:
        """Parse raw event data into FrontendEvent"""
        return FrontendEvent(
            event_type=EventType(event_data.get('type', 'click')),
            component_id=event_data.get('component_id', 'unknown'),
            payload=event_data.get('payload', {}),
            timestamp=datetime.now(),
            user_id=event_data.get('user_id'),
            session_id=event_data.get('session_id')
        )
    
    async def _process_event(self, event: FrontendEvent) -> Dict[str, Any]:
        """Process frontend event through registered handlers"""
        responses = []
        
        # Get handlers for this event type
        handlers = self.event_handlers.get(event.event_type, [])
        
        # Execute handlers
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    response = await handler(event)
                else:
                    response = handler(event)
                
                if response:
                    responses.append(response)
                    
            except Exception:
                pass
        
        # Update component state if needed
        self._update_component_state(event)
        
        return {
            'event_type': event.event_type.value,
            'component_id': event.component_id,
            'responses': responses,
            'state_updated': event.component_id in self.component_states
        }
    
    def _update_component_state(self, event: FrontendEvent):
        """Update component state based on event"""
        component_id = event.component_id
        
        if component_id not in self.component_states:
            self.component_states[component_id] = {}
        
        # Update state based on event type
        if event.event_type == EventType.TOGGLE:
            current = self.component_states[component_id].get('toggled', False)
            self.component_states[component_id]['toggled'] = not current
            
        elif event.event_type == EventType.FILTER:
            self.component_states[component_id]['filter'] = event.payload.get('filter', {})
            
        elif event.event_type == EventType.SORT:
            self.component_states[component_id]['sort'] = event.payload.get('sort', {})
        
        # Update last interaction
        self.component_states[component_id]['last_interaction'] = datetime.now().isoformat()
    
    def _handle_error_event(self, event: FrontendEvent) -> Dict[str, Any]:
        """Handle error events from frontend"""
        return {
            'action': 'log_error',
            'error_details': event.payload,
            'timestamp': event.timestamp.isoformat()
        }
    
    def _handle_load_event(self, event: FrontendEvent) -> Dict[str, Any]:
        """Handle component load events"""
        # Track session if provided
        if event.session_id:
            self.active_sessions[event.session_id] = {
                'started': datetime.now().isoformat(),
                'components_loaded': [event.component_id]
            }
        
        return {
            'action': 'component_loaded',
            'component_id': event.component_id,
            'initial_state': self.component_states.get(event.component_id, {})
        }
    
    def _handle_refresh_event(self, event: FrontendEvent) -> Dict[str, Any]:
        """Handle refresh events"""
        return {
            'action': 'refresh_data',
            'component_id': event.component_id,
            'refresh_type': event.payload.get('refresh_type', 'full')
        }
    
    def handle_dashboard_interaction(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dashboard user interaction"""
        interaction_type = interaction.get('type', 'unknown')
        component = interaction.get('component', 'unknown')
        
        # Track interaction
        response = {
            'interaction_id': f"int_{int(datetime.now().timestamp())}",
            'type': interaction_type,
            'component': component,
            'processed': True
        }
        
        # Handle specific interactions
        if interaction_type == 'chart_zoom':
            response['data'] = self._handle_chart_zoom(interaction)
        elif interaction_type == 'data_export':
            response['data'] = self._handle_data_export(interaction)
        elif interaction_type == 'filter_apply':
            response['data'] = self._handle_filter_apply(interaction)
        
        return response
    
    def _handle_chart_zoom(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chart zoom interaction"""
        return {
            'zoom_level': interaction.get('zoom_level', 1.0),
            'viewport': interaction.get('viewport', {}),
            'data_range': interaction.get('data_range', {})
        }
    
    def _handle_data_export(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data export request"""
        return {
            'export_format': interaction.get('format', 'json'),
            'data_selection': interaction.get('selection', 'all'),
            'export_ready': True
        }
    
    def _handle_filter_apply(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Handle filter application"""
        filters = interaction.get('filters', {})
        component_id = interaction.get('component', 'unknown')
        
        # Store filter state
        if component_id not in self.component_states:
            self.component_states[component_id] = {}
        
        self.component_states[component_id]['active_filters'] = filters
        
        return {
            'filters_applied': True,
            'filter_count': len(filters),
            'component_updated': component_id
        }
    
    def get_component_state(self, component_id: str) -> Dict[str, Any]:
        """Get current state of a component"""
        return self.component_states.get(component_id, {})
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        return self.active_sessions.get(session_id, {})
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event handling metrics"""
        return {
            **self.metrics,
            'active_sessions': len(self.active_sessions),
            'tracked_components': len(self.component_states),
            'recent_events': len(self.event_history),
            'latency_target_met': self.metrics['avg_processing_time'] * 1000 < 50
        }