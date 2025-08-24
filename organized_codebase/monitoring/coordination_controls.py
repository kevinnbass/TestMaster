#!/usr/bin/env python3
"""
🎛️ ATOM: Coordination Controls Component
========================================
Control interface for agent coordination and synthesis.
Part of STEELCLAD atomization - Agent T coordination specialist.
"""

from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

class ControlAction(Enum):
    """Available control actions"""
    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    RESET = "reset"
    CONFIGURE = "configure"
    SYNC = "sync"
    BALANCE = "balance"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class ControlState:
    """Control system state"""
    is_running: bool = False
    is_paused: bool = False
    current_mode: str = "manual"
    active_processes: int = 0
    queued_actions: List[str] = field(default_factory=list)
    last_action: Optional[str] = None
    last_action_time: Optional[datetime] = None

class CoordinationControls:
    """Coordination control interface component"""
    
    def __init__(self):
        self.control_state = ControlState()
        self.action_handlers: Dict[ControlAction, Callable] = {}
        self.control_presets = self._initialize_presets()
        self.control_config = self._initialize_config()
        self.action_history = []
    
    def _initialize_presets(self) -> Dict[str, Dict[str, Any]]:
        """Initialize control presets"""
        return {
            'high_performance': {
                'name': 'High Performance',
                'description': 'Maximum throughput and speed',
                'settings': {
                    'parallel_processes': 10,
                    'timeout': 30,
                    'retry_limit': 3,
                    'confidence_threshold': 0.85
                }
            },
            'balanced': {
                'name': 'Balanced',
                'description': 'Balanced performance and accuracy',
                'settings': {
                    'parallel_processes': 5,
                    'timeout': 60,
                    'retry_limit': 5,
                    'confidence_threshold': 0.90
                }
            },
            'high_accuracy': {
                'name': 'High Accuracy',
                'description': 'Maximum accuracy and validation',
                'settings': {
                    'parallel_processes': 3,
                    'timeout': 120,
                    'retry_limit': 10,
                    'confidence_threshold': 0.95
                }
            },
            'safe_mode': {
                'name': 'Safe Mode',
                'description': 'Conservative settings for stability',
                'settings': {
                    'parallel_processes': 1,
                    'timeout': 180,
                    'retry_limit': 15,
                    'confidence_threshold': 0.98
                }
            }
        }
    
    def _initialize_config(self) -> Dict[str, Any]:
        """Initialize control configuration"""
        return {
            'enable_auto_mode': False,
            'enable_emergency_stop': True,
            'show_advanced_controls': False,
            'confirmation_required': ['emergency_stop', 'reset'],
            'keyboard_shortcuts': True,
            'theme': 'professional'
        }
    
    def render_control_interface(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render coordination control interface
        
        Args:
            system_state: Current system state information
            
        Returns:
            Control interface configuration
        """
        return {
            'interface_type': 'coordination_controls',
            'layout': self._get_control_layout(),
            'sections': {
                'main_controls': self._render_main_controls(),
                'preset_selector': self._render_preset_selector(),
                'parameter_controls': self._render_parameter_controls(),
                'status_indicators': self._render_status_indicators(system_state),
                'action_queue': self._render_action_queue()
            },
            'state': self._get_control_state(),
            'shortcuts': self._get_keyboard_shortcuts()
        }
    
    def _get_control_layout(self) -> Dict[str, Any]:
        """Get control interface layout"""
        return {
            'type': 'control_panel',
            'orientation': 'vertical',
            'sections': [
                {'id': 'status', 'height': '60px', 'fixed': True},
                {'id': 'main', 'height': 'auto', 'scrollable': False},
                {'id': 'parameters', 'height': '200px', 'collapsible': True},
                {'id': 'queue', 'height': '150px', 'scrollable': True}
            ]
        }
    
    def _render_main_controls(self) -> Dict[str, Any]:
        """Render main control buttons"""
        controls = []
        
        # Primary controls
        if not self.control_state.is_running:
            controls.append({
                'id': 'start',
                'label': 'Start Coordination',
                'icon': '▶️',
                'style': 'primary',
                'size': 'large',
                'action': ControlAction.START.value
            })
        else:
            if self.control_state.is_paused:
                controls.append({
                    'id': 'resume',
                    'label': 'Resume',
                    'icon': '▶️',
                    'style': 'success',
                    'size': 'large',
                    'action': ControlAction.RESUME.value
                })
            else:
                controls.append({
                    'id': 'pause',
                    'label': 'Pause',
                    'icon': '⏸️',
                    'style': 'warning',
                    'size': 'large',
                    'action': ControlAction.PAUSE.value
                })
            
            controls.append({
                'id': 'stop',
                'label': 'Stop',
                'icon': '⏹️',
                'style': 'secondary',
                'size': 'large',
                'action': ControlAction.STOP.value
            })
        
        # Secondary controls
        controls.extend([
            {
                'id': 'sync',
                'label': 'Sync Data',
                'icon': '🔄',
                'style': 'info',
                'size': 'medium',
                'action': ControlAction.SYNC.value,
                'disabled': not self.control_state.is_running
            },
            {
                'id': 'balance',
                'label': 'Balance Load',
                'icon': '⚖️',
                'style': 'info',
                'size': 'medium',
                'action': ControlAction.BALANCE.value,
                'disabled': not self.control_state.is_running
            },
            {
                'id': 'configure',
                'label': 'Configure',
                'icon': '⚙️',
                'style': 'secondary',
                'size': 'medium',
                'action': ControlAction.CONFIGURE.value
            }
        ])
        
        # Emergency control
        if self.control_config['enable_emergency_stop']:
            controls.append({
                'id': 'emergency',
                'label': 'Emergency Stop',
                'icon': '🚨',
                'style': 'danger',
                'size': 'medium',
                'action': ControlAction.EMERGENCY_STOP.value,
                'confirmation': True
            })
        
        return {
            'title': 'Coordination Controls',
            'controls': controls,
            'layout': 'grid',
            'columns': 3
        }
    
    def _render_preset_selector(self) -> Dict[str, Any]:
        """Render preset configuration selector"""
        presets = []
        
        for preset_id, preset in self.control_presets.items():
            presets.append({
                'id': preset_id,
                'name': preset['name'],
                'description': preset['description'],
                'settings_preview': self._format_settings_preview(preset['settings'])
            })
        
        return {
            'title': 'Configuration Presets',
            'type': 'radio_group',
            'options': presets,
            'current': 'balanced',
            'allow_custom': True,
            'on_change': 'apply_preset'
        }
    
    def _render_parameter_controls(self) -> Dict[str, Any]:
        """Render parameter control sliders"""
        return {
            'title': 'Parameters',
            'controls': [
                {
                    'name': 'parallel_processes',
                    'label': 'Parallel Processes',
                    'type': 'slider',
                    'min': 1,
                    'max': 20,
                    'value': 5,
                    'step': 1,
                    'unit': 'processes'
                },
                {
                    'name': 'confidence_threshold',
                    'label': 'Confidence Threshold',
                    'type': 'slider',
                    'min': 0.5,
                    'max': 1.0,
                    'value': 0.90,
                    'step': 0.05,
                    'unit': '%',
                    'display_multiplier': 100
                },
                {
                    'name': 'timeout',
                    'label': 'Process Timeout',
                    'type': 'slider',
                    'min': 10,
                    'max': 300,
                    'value': 60,
                    'step': 10,
                    'unit': 'seconds'
                },
                {
                    'name': 'retry_limit',
                    'label': 'Retry Limit',
                    'type': 'slider',
                    'min': 0,
                    'max': 20,
                    'value': 5,
                    'step': 1,
                    'unit': 'attempts'
                }
            ],
            'show_values': True,
            'live_update': False,
            'apply_button': True
        }
    
    def _render_status_indicators(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Render status indicators"""
        return {
            'title': 'System Status',
            'indicators': [
                {
                    'label': 'Coordination',
                    'status': 'active' if self.control_state.is_running else 'inactive',
                    'color': self._get_status_color(self.control_state.is_running)
                },
                {
                    'label': 'Mode',
                    'status': self.control_state.current_mode.capitalize(),
                    'color': 'info'
                },
                {
                    'label': 'Active Processes',
                    'status': str(self.control_state.active_processes),
                    'color': self._get_process_count_color(self.control_state.active_processes)
                },
                {
                    'label': 'Queue',
                    'status': f"{len(self.control_state.queued_actions)} pending",
                    'color': 'warning' if self.control_state.queued_actions else 'success'
                },
                {
                    'label': 'System Health',
                    'status': f"{system_state.get('health', 0) * 100:.0f}%",
                    'color': self._get_health_color(system_state.get('health', 0))
                }
            ],
            'layout': 'horizontal',
            'compact': True
        }
    
    def _render_action_queue(self) -> Dict[str, Any]:
        """Render action queue display"""
        queue_items = []
        
        for i, action in enumerate(self.control_state.queued_actions):
            queue_items.append({
                'position': i + 1,
                'action': action,
                'status': 'pending' if i > 0 else 'processing',
                'can_cancel': True
            })
        
        return {
            'title': f"Action Queue ({len(queue_items)} items)",
            'items': queue_items,
            'max_display': 10,
            'show_clear_button': len(queue_items) > 0,
            'auto_scroll': True
        }
    
    def _get_control_state(self) -> Dict[str, Any]:
        """Get current control state"""
        return {
            'is_running': self.control_state.is_running,
            'is_paused': self.control_state.is_paused,
            'mode': self.control_state.current_mode,
            'can_start': not self.control_state.is_running,
            'can_stop': self.control_state.is_running,
            'can_pause': self.control_state.is_running and not self.control_state.is_paused,
            'can_resume': self.control_state.is_running and self.control_state.is_paused
        }
    
    def _get_keyboard_shortcuts(self) -> Dict[str, str]:
        """Get keyboard shortcuts"""
        if not self.control_config['keyboard_shortcuts']:
            return {}
        
        return {
            'start': 'Ctrl+S',
            'stop': 'Ctrl+X',
            'pause': 'Ctrl+P',
            'resume': 'Ctrl+R',
            'sync': 'Ctrl+D',
            'balance': 'Ctrl+B',
            'emergency': 'Ctrl+Shift+X'
        }
    
    def _format_settings_preview(self, settings: Dict[str, Any]) -> str:
        """Format settings preview text"""
        parts = []
        if 'parallel_processes' in settings:
            parts.append(f"{settings['parallel_processes']} parallel")
        if 'confidence_threshold' in settings:
            parts.append(f"{settings['confidence_threshold']*100:.0f}% confidence")
        return " • ".join(parts)
    
    def _get_status_color(self, is_active: bool) -> str:
        """Get status indicator color"""
        return 'success' if is_active else 'secondary'
    
    def _get_process_count_color(self, count: int) -> str:
        """Get color based on process count"""
        if count == 0:
            return 'secondary'
        elif count < 5:
            return 'success'
        elif count < 10:
            return 'warning'
        else:
            return 'danger'
    
    def _get_health_color(self, health: float) -> str:
        """Get color based on health score"""
        if health >= 0.9:
            return 'success'
        elif health >= 0.7:
            return 'warning'
        else:
            return 'danger'
    
    def execute_action(self, action: ControlAction) -> bool:
        """Execute a control action"""
        # Record action
        self.action_history.append({
            'action': action.value,
            'timestamp': datetime.utcnow().isoformat(),
            'state_before': self.control_state.is_running
        })
        
        # Update state
        self.control_state.last_action = action.value
        self.control_state.last_action_time = datetime.utcnow()
        
        # Execute handler if registered
        if action in self.action_handlers:
            return self.action_handlers[action]()
        
        # Default state updates
        if action == ControlAction.START:
            self.control_state.is_running = True
            self.control_state.is_paused = False
        elif action == ControlAction.STOP:
            self.control_state.is_running = False
            self.control_state.is_paused = False
        elif action == ControlAction.PAUSE:
            self.control_state.is_paused = True
        elif action == ControlAction.RESUME:
            self.control_state.is_paused = False
        elif action == ControlAction.RESET:
            self.control_state = ControlState()
        
        return True
    
    def register_action_handler(self, action: ControlAction, handler: Callable):
        """Register handler for control action"""
        self.action_handlers[action] = handler
    
    def queue_action(self, action: str):
        """Queue an action for execution"""
        self.control_state.queued_actions.append(action)
    
    def clear_queue(self):
        """Clear action queue"""
        self.control_state.queued_actions.clear()
    
    def apply_preset(self, preset_id: str) -> bool:
        """Apply a configuration preset"""
        if preset_id in self.control_presets:
            # Apply preset settings
            # This would be implemented based on actual system integration
            return True
        return False