"""
AgentOps Derived Enum Security Module
Extracted from AgentOps enums.py patterns and secure enumeration handling
Enhanced for comprehensive enum validation and security state management
"""

import logging
from typing import Dict, List, Optional, Any, Set, Type, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum, EnumMeta
from .error_handler import SecurityError, ValidationError, security_error_handler


class SecurityStateLevel(Enum):
    """Security state levels based on AgentOps patterns"""
    SUCCESS = "success"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"
    UNSET = "unset"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def is_secure(self) -> bool:
        """Check if state represents a secure condition"""
        return self in [SecurityStateLevel.SUCCESS, SecurityStateLevel.UNSET]
    
    @property
    def requires_attention(self) -> bool:
        """Check if state requires security attention"""
        return self in [SecurityStateLevel.ERROR, SecurityStateLevel.CRITICAL]


class TraceSecurityState(Enum):
    """Trace security state enumeration based on AgentOps TraceState"""
    SECURE = "secure"
    SUSPICIOUS = "suspicious"
    BLOCKED = "blocked"
    QUARANTINED = "quarantined"
    UNKNOWN = "unknown"
    
    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def from_trace_state(cls, trace_state: str) -> 'TraceSecurityState':
        """Convert AgentOps TraceState to security state"""
        trace_lower = trace_state.lower()
        
        if trace_lower in ['success', 'ok', 'completed']:
            return cls.SECURE
        elif trace_lower in ['error', 'failed']:
            return cls.SUSPICIOUS
        elif trace_lower in ['unset', 'pending']:
            return cls.UNKNOWN
        else:
            return cls.UNKNOWN


class APISecurityState(Enum):
    """API security state enumeration"""
    AUTHENTICATED = "authenticated"
    UNAUTHENTICATED = "unauthenticated"
    TOKEN_EXPIRED = "token_expired"
    RATE_LIMITED = "rate_limited"
    FORBIDDEN = "forbidden"
    ERROR = "error"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def is_authorized(self) -> bool:
        """Check if API state allows access"""
        return self == APISecurityState.AUTHENTICATED


class ValidationSecurityState(Enum):
    """Validation security state enumeration"""
    VALID = "valid"
    INVALID = "invalid"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    TIMEOUT = "timeout"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def is_safe(self) -> bool:
        """Check if validation state is safe"""
        return self == ValidationSecurityState.VALID


@dataclass
class EnumSecurityEvent:
    """Security event for enum state changes"""
    enum_name: str
    old_value: Optional[str]
    new_value: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    security_impact: SecurityStateLevel = SecurityStateLevel.UNSET
    
    @property
    def is_security_relevant(self) -> bool:
        """Check if enum change is security relevant"""
        return self.security_impact.requires_attention


class SecureEnumValidator:
    """Secure enum validation and management system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.registered_enums: Dict[str, Type[Enum]] = {}
        self.enum_events: List[EnumSecurityEvent] = []
        self.security_transitions: Dict[str, Set[Tuple[str, str]]] = {}
        self.max_events = 10000
        
        # Register default security enums
        self._register_default_enums()
        
        # Define dangerous transitions
        self._define_security_transitions()
    
    def register_enum(self, enum_class: Type[Enum], security_relevant: bool = True):
        """Register an enum class for security monitoring"""
        try:
            enum_name = enum_class.__name__
            self.registered_enums[enum_name] = enum_class
            
            if security_relevant:
                self.logger.info(f"Registered security-relevant enum: {enum_name}")
            else:
                self.logger.debug(f"Registered enum: {enum_name}")
                
        except Exception as e:
            error = SecurityError(f"Failed to register enum: {str(e)}", "ENUM_REG_001")
            security_error_handler.handle_error(error)
    
    def validate_enum_value(self, enum_class: Type[Enum], value: Any, 
                           context: Dict[str, Any] = None) -> bool:
        """Validate enum value with security checks"""
        try:
            enum_name = enum_class.__name__
            
            # Check if value is valid for enum
            valid_values = [item.value for item in enum_class]
            
            if value not in valid_values:
                self.logger.warning(f"Invalid enum value for {enum_name}: {value}")
                self._record_security_event(
                    enum_name, None, str(value), 
                    SecurityStateLevel.ERROR, context
                )
                return False
            
            # Check for security-sensitive transitions
            if enum_name in self.security_transitions:
                dangerous_transitions = self.security_transitions[enum_name]
                current_context = context or {}
                old_value = current_context.get('old_value')
                
                if old_value and (str(old_value), str(value)) in dangerous_transitions:
                    self.logger.warning(
                        f"Dangerous enum transition in {enum_name}: {old_value} -> {value}"
                    )
                    self._record_security_event(
                        enum_name, str(old_value), str(value),
                        SecurityStateLevel.CRITICAL, context
                    )
            
            # Record successful validation
            self._record_security_event(
                enum_name, context.get('old_value') if context else None, 
                str(value), SecurityStateLevel.SUCCESS, context
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Enum validation error: {e}")
            self._record_security_event(
                enum_class.__name__, None, str(value),
                SecurityStateLevel.ERROR, {'error': str(e)}
            )
            return False
    
    def create_secure_enum_instance(self, enum_class: Type[Enum], value: Any,
                                   validate: bool = True) -> Optional[Enum]:
        """Create enum instance with security validation"""
        try:
            if validate and not self.validate_enum_value(enum_class, value):
                return None
            
            # Find matching enum item
            for item in enum_class:
                if item.value == value:
                    return item
            
            # If not found, try by name
            if hasattr(enum_class, str(value).upper()):
                return getattr(enum_class, str(value).upper())
            
            return None
            
        except Exception as e:
            error = SecurityError(f"Failed to create enum instance: {str(e)}", "ENUM_CREATE_001")
            security_error_handler.handle_error(error)
            return None
    
    def get_security_state_summary(self) -> Dict[str, Any]:
        """Get summary of enum security states"""
        try:
            # Count events by enum
            enum_event_counts = {}
            security_level_counts = {}
            
            # Recent events (last hour)
            recent_cutoff = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            recent_events = [e for e in self.enum_events if e.timestamp >= recent_cutoff]
            
            for event in recent_events:
                # Count by enum
                enum_event_counts[event.enum_name] = enum_event_counts.get(event.enum_name, 0) + 1
                
                # Count by security level
                level = event.security_impact.value
                security_level_counts[level] = security_level_counts.get(level, 0) + 1
            
            # Security-relevant events
            security_relevant_count = sum(1 for e in recent_events if e.is_security_relevant)
            
            return {
                'registered_enums': len(self.registered_enums),
                'total_events': len(self.enum_events),
                'recent_events_1h': len(recent_events),
                'security_relevant_events_1h': security_relevant_count,
                'enum_activity': enum_event_counts,
                'security_level_distribution': security_level_counts,
                'monitored_transitions': len(self.security_transitions)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating enum security summary: {e}")
            return {'error': str(e)}
    
    def _register_default_enums(self):
        """Register default security enums"""
        self.register_enum(SecurityStateLevel)
        self.register_enum(TraceSecurityState)
        self.register_enum(APISecurityState)
        self.register_enum(ValidationSecurityState)
    
    def _define_security_transitions(self):
        """Define dangerous enum state transitions"""
        # Security state transitions to monitor
        self.security_transitions['SecurityStateLevel'] = {
            ('SUCCESS', 'ERROR'),
            ('SUCCESS', 'CRITICAL'),
            ('WARNING', 'CRITICAL'),
        }
        
        self.security_transitions['TraceSecurityState'] = {
            ('SECURE', 'BLOCKED'),
            ('SECURE', 'QUARANTINED'),
            ('SUSPICIOUS', 'BLOCKED'),
        }
        
        self.security_transitions['APISecurityState'] = {
            ('AUTHENTICATED', 'FORBIDDEN'),
            ('AUTHENTICATED', 'RATE_LIMITED'),
            ('UNAUTHENTICATED', 'ERROR'),
        }
        
        self.security_transitions['ValidationSecurityState'] = {
            ('VALID', 'MALICIOUS'),
            ('VALID', 'SUSPICIOUS'),
            ('INVALID', 'MALICIOUS'),
        }
    
    def _record_security_event(self, enum_name: str, old_value: Optional[str], 
                              new_value: str, impact: SecurityStateLevel,
                              context: Dict[str, Any] = None):
        """Record enum security event"""
        event = EnumSecurityEvent(
            enum_name=enum_name,
            old_value=old_value,
            new_value=new_value,
            security_impact=impact,
            context=context or {}
        )
        
        self.enum_events.append(event)
        
        # Limit event history
        if len(self.enum_events) > self.max_events:
            self.enum_events = self.enum_events[-self.max_events // 2:]
        
        # Log security-relevant events
        if event.is_security_relevant:
            self.logger.warning(f"Security enum event: {enum_name} {old_value} -> {new_value}")


class EnumSecurityManager:
    """Central enum security management system"""
    
    def __init__(self):
        self.validator = SecureEnumValidator()
        self.state_machines: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_secure_state_machine(self, name: str, enum_class: Type[Enum],
                                   initial_state: Any, 
                                   allowed_transitions: Dict[str, List[str]] = None):
        """Create a secure state machine with enum-based states"""
        try:
            # Validate initial state
            if not self.validator.validate_enum_value(enum_class, initial_state):
                raise ValidationError(f"Invalid initial state: {initial_state}")
            
            initial_enum = self.validator.create_secure_enum_instance(enum_class, initial_state)
            if not initial_enum:
                raise ValidationError(f"Cannot create initial enum instance: {initial_state}")
            
            state_machine = {
                'name': name,
                'enum_class': enum_class,
                'current_state': initial_enum,
                'allowed_transitions': allowed_transitions or {},
                'transition_history': [],
                'created_at': datetime.utcnow(),
                'last_transition': datetime.utcnow()
            }
            
            self.state_machines[name] = state_machine
            self.logger.info(f"Created secure state machine: {name}")
            
            return state_machine
            
        except Exception as e:
            error = SecurityError(f"Failed to create state machine: {str(e)}", "STATE_MACHINE_001")
            security_error_handler.handle_error(error)
            raise error
    
    def transition_state(self, state_machine_name: str, new_state: Any,
                        context: Dict[str, Any] = None) -> bool:
        """Transition state machine to new state with security validation"""
        try:
            if state_machine_name not in self.state_machines:
                raise ValidationError(f"State machine not found: {state_machine_name}")
            
            state_machine = self.state_machines[state_machine_name]
            enum_class = state_machine['enum_class']
            current_state = state_machine['current_state']
            
            # Validate transition is allowed
            allowed_transitions = state_machine['allowed_transitions']
            if allowed_transitions and current_state.value in allowed_transitions:
                if new_state not in allowed_transitions[current_state.value]:
                    self.logger.warning(
                        f"Transition not allowed in {state_machine_name}: "
                        f"{current_state.value} -> {new_state}"
                    )
                    return False
            
            # Validate new state
            validation_context = {
                'old_value': current_state.value,
                'state_machine': state_machine_name,
                **(context or {})
            }
            
            if not self.validator.validate_enum_value(enum_class, new_state, validation_context):
                return False
            
            # Create new enum instance
            new_enum = self.validator.create_secure_enum_instance(enum_class, new_state)
            if not new_enum:
                return False
            
            # Record transition
            transition_record = {
                'from_state': current_state.value,
                'to_state': new_enum.value,
                'timestamp': datetime.utcnow(),
                'context': context or {}
            }
            
            state_machine['transition_history'].append(transition_record)
            state_machine['current_state'] = new_enum
            state_machine['last_transition'] = datetime.utcnow()
            
            # Limit history
            if len(state_machine['transition_history']) > 1000:
                state_machine['transition_history'] = state_machine['transition_history'][-500:]
            
            self.logger.info(
                f"State transition in {state_machine_name}: "
                f"{current_state.value} -> {new_enum.value}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"State transition error: {e}")
            return False
    
    def get_state_machine_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get state machine status"""
        if name not in self.state_machines:
            return None
        
        state_machine = self.state_machines[name]
        
        return {
            'name': name,
            'current_state': state_machine['current_state'].value,
            'enum_class': state_machine['enum_class'].__name__,
            'created_at': state_machine['created_at'].isoformat(),
            'last_transition': state_machine['last_transition'].isoformat(),
            'transition_count': len(state_machine['transition_history']),
            'recent_transitions': [
                {
                    'from': t['from_state'],
                    'to': t['to_state'], 
                    'timestamp': t['timestamp'].isoformat()
                }
                for t in state_machine['transition_history'][-5:]
            ]
        }
    
    def get_enum_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive enum security summary"""
        try:
            validator_summary = self.validator.get_security_state_summary()
            
            # Add state machine information
            state_machine_summary = {
                'total_state_machines': len(self.state_machines),
                'active_state_machines': len([
                    sm for sm in self.state_machines.values()
                    if (datetime.utcnow() - sm['last_transition']).total_seconds() < 3600
                ])
            }
            
            # Combine summaries
            return {
                **validator_summary,
                **state_machine_summary
            }
            
        except Exception as e:
            self.logger.error(f"Error generating enum security summary: {e}")
            return {'error': str(e)}


# Global enum security manager
enum_security_manager = EnumSecurityManager()


# Convenience functions
def validate_enum_security(enum_class: Type[Enum], value: Any) -> bool:
    """Convenience function to validate enum value"""
    return enum_security_manager.validator.validate_enum_value(enum_class, value)


def create_secure_enum(enum_class: Type[Enum], value: Any) -> Optional[Enum]:
    """Convenience function to create secure enum instance"""
    return enum_security_manager.validator.create_secure_enum_instance(enum_class, value)


def create_security_state_machine(name: str, enum_class: Type[Enum], 
                                 initial_state: Any) -> Dict[str, Any]:
    """Convenience function to create secure state machine"""
    return enum_security_manager.create_secure_state_machine(name, enum_class, initial_state)


# Security-focused enum factory
class SecureEnumFactory:
    """Factory for creating security-enhanced enums"""
    
    @staticmethod
    def create_security_enum(name: str, values: Dict[str, str], 
                           base_class: Type[Enum] = Enum) -> Type[Enum]:
        """Create a security-enhanced enum class"""
        
        class SecureEnumMeta(EnumMeta):
            def __call__(cls, value):
                # Add security validation to enum creation
                if not enum_security_manager.validator.validate_enum_value(cls, value):
                    raise ValidationError(f"Invalid enum value: {value}")
                return super().__call__(value)
        
        # Create the enum class with security meta
        SecureEnumClass = SecureEnumMeta(name, (base_class,), values)
        
        # Register with security manager
        enum_security_manager.validator.register_enum(SecureEnumClass)
        
        return SecureEnumClass