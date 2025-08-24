#!/usr/bin/env python3
"""
💬 ATOM: Agent Communication UI Component
=========================================
Handles inter-agent communication interface and messaging.
Part of STEELCLAD atomization - Agent T coordination specialist.
"""

from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid

class CoordinationType(Enum):
    """Types of coordination messages"""
    DATA_SYNC = "data_sync"
    TASK_ASSIGNMENT = "task_assignment"
    STATUS_UPDATE = "status_update"
    HEALTH_CHECK = "health_check"
    CONFIGURATION = "configuration"
    ALERT = "alert"
    RESPONSE = "response"

class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BACKGROUND = 1

@dataclass
class CoordinationMessage:
    """Coordination message structure"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_agent: str = ""
    target_agent: str = ""
    coordination_type: CoordinationType = CoordinationType.STATUS_UPDATE
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    retry_count: int = 0
    status: str = "pending"

class AgentCommunicationUI:
    """Agent communication interface component"""
    
    def __init__(self):
        self.message_queue: List[CoordinationMessage] = []
        self.message_history: List[CoordinationMessage] = []
        self.active_conversations: Dict[str, List[CoordinationMessage]] = {}
        self.message_handlers: Dict[CoordinationType, List[Callable]] = {}
        self.ui_config = self._initialize_ui_config()
    
    def _initialize_ui_config(self) -> Dict[str, Any]:
        """Initialize UI configuration"""
        return {
            'max_displayed_messages': 50,
            'enable_notifications': True,
            'group_by_conversation': True,
            'show_timestamps': True,
            'enable_filtering': True,
            'auto_scroll': True
        }
    
    def render_communication_interface(self, agents: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render agent communication interface
        
        Args:
            agents: Dictionary of connected agents
            
        Returns:
            Communication UI configuration
        """
        return {
            'interface_type': 'agent_communication',
            'layout': self._get_communication_layout(),
            'panels': {
                'message_list': self._render_message_list(),
                'conversation_view': self._render_conversation_view(),
                'compose_panel': self._render_compose_panel(agents),
                'status_bar': self._render_status_bar()
            },
            'controls': self._get_control_configuration()
        }
    
    def _get_communication_layout(self) -> Dict[str, Any]:
        """Get communication interface layout"""
        return {
            'type': 'split_view',
            'orientation': 'horizontal',
            'panels': [
                {'id': 'conversations', 'width': '30%', 'resizable': True},
                {'id': 'messages', 'width': '50%', 'resizable': True},
                {'id': 'details', 'width': '20%', 'resizable': True}
            ]
        }
    
    def _render_message_list(self) -> Dict[str, Any]:
        """Render message list panel"""
        recent_messages = self._get_recent_messages()
        
        return {
            'title': 'Messages',
            'count': len(recent_messages),
            'messages': [
                self._format_message_item(msg) 
                for msg in recent_messages
            ],
            'filters': {
                'types': [t.value for t in CoordinationType],
                'priorities': [p.value for p in MessagePriority],
                'agents': self._get_unique_agents()
            },
            'actions': ['reply', 'forward', 'archive', 'delete']
        }
    
    def _render_conversation_view(self) -> Dict[str, Any]:
        """Render conversation threads view"""
        conversations = []
        
        for conv_id, messages in self.active_conversations.items():
            conversations.append({
                'id': conv_id,
                'participants': self._get_conversation_participants(messages),
                'message_count': len(messages),
                'last_message': self._format_message_item(messages[-1]) if messages else None,
                'unread': sum(1 for m in messages if m.status == 'unread'),
                'priority': max((m.priority.value for m in messages), default=0)
            })
        
        return {
            'title': 'Conversations',
            'conversations': sorted(conversations, 
                                   key=lambda x: x['priority'], 
                                   reverse=True),
            'grouping': 'by_agent' if self.ui_config['group_by_conversation'] else 'chronological'
        }
    
    def _render_compose_panel(self, agents: Dict[str, Any]) -> Dict[str, Any]:
        """Render message composition panel"""
        return {
            'title': 'Compose Message',
            'fields': [
                {
                    'name': 'target_agent',
                    'type': 'select',
                    'label': 'To',
                    'options': [
                        {'value': 'all', 'label': 'All Agents'},
                        *[{'value': aid, 'label': a.get('agent_type', aid)} 
                          for aid, a in agents.items()]
                    ]
                },
                {
                    'name': 'coordination_type',
                    'type': 'select',
                    'label': 'Type',
                    'options': [
                        {'value': t.value, 'label': t.value.replace('_', ' ').title()}
                        for t in CoordinationType
                    ]
                },
                {
                    'name': 'priority',
                    'type': 'select',
                    'label': 'Priority',
                    'options': [
                        {'value': p.value, 'label': p.name}
                        for p in MessagePriority
                    ]
                },
                {
                    'name': 'payload',
                    'type': 'textarea',
                    'label': 'Message',
                    'placeholder': 'Enter message payload (JSON format)'
                }
            ],
            'actions': ['send', 'save_draft', 'cancel']
        }
    
    def _render_status_bar(self) -> Dict[str, Any]:
        """Render communication status bar"""
        queue_size = len(self.message_queue)
        pending_count = sum(1 for m in self.message_history if m.status == 'pending')
        failed_count = sum(1 for m in self.message_history if m.status == 'failed')
        
        return {
            'queue_size': queue_size,
            'pending': pending_count,
            'failed': failed_count,
            'success_rate': self._calculate_success_rate(),
            'avg_response_time': self._calculate_avg_response_time(),
            'status_text': self._get_status_text(queue_size, pending_count)
        }
    
    def _get_control_configuration(self) -> Dict[str, Any]:
        """Get control configuration"""
        return {
            'keyboard_shortcuts': {
                'send': 'Ctrl+Enter',
                'reply': 'R',
                'forward': 'F',
                'archive': 'A',
                'delete': 'Delete'
            },
            'context_menu': ['reply', 'forward', 'copy', 'archive'],
            'drag_drop': True,
            'multi_select': True
        }
    
    def _format_message_item(self, message: CoordinationMessage) -> Dict[str, Any]:
        """Format message for display"""
        return {
            'id': message.message_id,
            'from': message.source_agent,
            'to': message.target_agent,
            'type': message.coordination_type.value,
            'priority': {
                'level': message.priority.value,
                'name': message.priority.name,
                'color': self._get_priority_color(message.priority)
            },
            'timestamp': message.timestamp.isoformat(),
            'time_ago': self._format_time_ago(message.timestamp),
            'status': message.status,
            'preview': self._get_message_preview(message.payload),
            'retry_count': message.retry_count
        }
    
    def _get_recent_messages(self, limit: int = None) -> List[CoordinationMessage]:
        """Get recent messages"""
        limit = limit or self.ui_config['max_displayed_messages']
        all_messages = self.message_queue + self.message_history
        return sorted(all_messages, key=lambda m: m.timestamp, reverse=True)[:limit]
    
    def _get_unique_agents(self) -> List[str]:
        """Get list of unique agents from messages"""
        agents = set()
        for msg in self.message_history + self.message_queue:
            agents.add(msg.source_agent)
            agents.add(msg.target_agent)
        return sorted(list(agents - {'', 'all'}))
    
    def _get_conversation_participants(self, messages: List[CoordinationMessage]) -> List[str]:
        """Get unique participants in a conversation"""
        participants = set()
        for msg in messages:
            participants.add(msg.source_agent)
            participants.add(msg.target_agent)
        return sorted(list(participants - {'', 'all'}))
    
    def _get_priority_color(self, priority: MessagePriority) -> str:
        """Get color for priority level"""
        colors = {
            MessagePriority.CRITICAL: 'danger',
            MessagePriority.HIGH: 'warning',
            MessagePriority.NORMAL: 'info',
            MessagePriority.LOW: 'secondary',
            MessagePriority.BACKGROUND: 'light'
        }
        return colors.get(priority, 'secondary')
    
    def _format_time_ago(self, timestamp: datetime) -> str:
        """Format timestamp as time ago"""
        delta = datetime.utcnow() - timestamp
        
        if delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}h ago"
        elif delta.seconds > 60:
            return f"{delta.seconds // 60}m ago"
        else:
            return "just now"
    
    def _get_message_preview(self, payload: Dict[str, Any], max_length: int = 100) -> str:
        """Get message preview text"""
        if isinstance(payload, dict):
            text = str(payload.get('message', payload))
        else:
            text = str(payload)
        
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    def _calculate_success_rate(self) -> float:
        """Calculate message success rate"""
        if not self.message_history:
            return 100.0
        
        successful = sum(1 for m in self.message_history if m.status == 'delivered')
        return (successful / len(self.message_history)) * 100
    
    def _calculate_avg_response_time(self) -> str:
        """Calculate average response time"""
        # Placeholder implementation
        return "45ms"
    
    def _get_status_text(self, queue_size: int, pending: int) -> str:
        """Get status bar text"""
        if queue_size > 10:
            return f"High message volume: {queue_size} in queue"
        elif pending > 5:
            return f"{pending} messages pending delivery"
        else:
            return "All systems operational"
    
    def send_message(self, message: CoordinationMessage) -> bool:
        """Send a coordination message"""
        self.message_queue.append(message)
        return True
    
    def receive_message(self, message: CoordinationMessage):
        """Receive a coordination message"""
        message.status = 'received'
        self.message_history.append(message)
        
        # Group into conversation
        conv_key = f"{message.source_agent}_{message.target_agent}"
        if conv_key not in self.active_conversations:
            self.active_conversations[conv_key] = []
        self.active_conversations[conv_key].append(message)
        
        # Trigger handlers
        if message.coordination_type in self.message_handlers:
            for handler in self.message_handlers[message.coordination_type]:
                handler(message)
    
    def register_handler(self, message_type: CoordinationType, handler: Callable):
        """Register message handler"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    def clear_old_messages(self, days: int = 7):
        """Clear old messages from history"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        self.message_history = [
            m for m in self.message_history 
            if m.timestamp > cutoff
        ]