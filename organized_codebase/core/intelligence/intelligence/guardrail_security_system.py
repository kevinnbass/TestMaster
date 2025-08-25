"""
CrewAI Derived Guardrail Security System
Extracted from CrewAI guardrail patterns and safety check implementations
Enhanced for comprehensive validation and safety mechanisms
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable, Union, TypeVar
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from .error_handler import SecurityError, ValidationError, security_error_handler

T = TypeVar('T')


class GuardrailStatus(Enum):
    """Guardrail execution status based on CrewAI patterns"""
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    RETRY_NEEDED = "retry_needed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


class GuardrailSeverity(Enum):
    """Guardrail severity levels for safety classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GuardrailEvent:
    """Guardrail execution event tracking"""
    guardrail_name: str
    event_type: str  # start, complete, retry, timeout
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    status: Optional[GuardrailStatus] = None
    duration_ms: Optional[float] = None


@dataclass
class GuardrailResult:
    """Comprehensive guardrail execution result based on CrewAI patterns"""
    success: bool
    status: GuardrailStatus
    message: str
    guardrail_name: str
    execution_time_ms: float
    retry_count: int = 0
    severity: GuardrailSeverity = GuardrailSeverity.MEDIUM
    context: Dict[str, Any] = field(default_factory=dict)
    events: List[GuardrailEvent] = field(default_factory=list)
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def failed(self) -> bool:
        """Check if guardrail failed"""
        return not self.success
    
    @property
    def needs_retry(self) -> bool:
        """Check if guardrail needs retry"""
        return self.status == GuardrailStatus.RETRY_NEEDED
    
    @property
    def is_critical(self) -> bool:
        """Check if result is critical severity"""
        return self.severity == GuardrailSeverity.CRITICAL
    
    def add_event(self, event_type: str, context: Dict[str, Any] = None):
        """Add execution event"""
        event = GuardrailEvent(
            guardrail_name=self.guardrail_name,
            event_type=event_type,
            context=context or {},
            status=self.status
        )
        self.events.append(event)


class BaseGuardrail(ABC):
    """Base guardrail class following CrewAI patterns"""
    
    def __init__(self, name: str, severity: GuardrailSeverity = GuardrailSeverity.MEDIUM,
                 timeout_seconds: float = 30.0, max_retries: int = 3):
        self.name = name
        self.severity = severity
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def validate(self, data: Any, context: Dict[str, Any] = None) -> GuardrailResult:
        """Abstract method for guardrail validation"""
        pass
    
    def create_result(self, success: bool, status: GuardrailStatus, 
                     message: str, execution_time_ms: float,
                     retry_count: int = 0,
                     validation_details: Dict[str, Any] = None) -> GuardrailResult:
        """Create standardized guardrail result"""
        return GuardrailResult(
            success=success,
            status=status,
            message=message,
            guardrail_name=self.name,
            execution_time_ms=execution_time_ms,
            retry_count=retry_count,
            severity=self.severity,
            validation_details=validation_details or {}
        )


class HallucinationGuardrail(BaseGuardrail):
    """Hallucination detection guardrail based on CrewAI patterns"""
    
    def __init__(self, confidence_threshold: float = 0.8, **kwargs):
        super().__init__(name="hallucination_detection", **kwargs)
        self.confidence_threshold = confidence_threshold
    
    def validate(self, data: Any, context: Dict[str, Any] = None) -> GuardrailResult:
        """Validate for potential hallucinations in LLM output"""
        start_time = time.time()
        
        try:
            # Extract content for analysis
            content = str(data)
            if hasattr(data, 'raw') and data.raw:
                content = data.raw
            elif hasattr(data, 'content'):
                content = str(data.content)
            
            # Analyze for hallucination indicators
            validation_details = {
                'content_length': len(content),
                'confidence_threshold': self.confidence_threshold,
                'checks_performed': []
            }
            
            hallucination_score = 0.0
            checks = []
            
            # Check for repetitive patterns
            repetition_score = self._check_repetitive_patterns(content)
            checks.append(('repetitive_patterns', repetition_score))
            hallucination_score += repetition_score * 0.3
            
            # Check for inconsistencies
            inconsistency_score = self._check_inconsistencies(content, context)
            checks.append(('inconsistencies', inconsistency_score))
            hallucination_score += inconsistency_score * 0.4
            
            # Check for factual coherence
            coherence_score = self._check_factual_coherence(content)
            checks.append(('factual_coherence', coherence_score))
            hallucination_score += coherence_score * 0.3
            
            validation_details['checks_performed'] = checks
            validation_details['hallucination_score'] = hallucination_score
            
            execution_time = (time.time() - start_time) * 1000
            
            # Determine result based on confidence threshold
            if hallucination_score > (1.0 - self.confidence_threshold):
                return self.create_result(
                    False, GuardrailStatus.BLOCKED,
                    f"Potential hallucination detected (score: {hallucination_score:.3f})",
                    execution_time, validation_details=validation_details
                )
            
            return self.create_result(
                True, GuardrailStatus.SUCCESS,
                f"Content validated (hallucination score: {hallucination_score:.3f})",
                execution_time, validation_details=validation_details
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"Hallucination guardrail error: {e}")
            return self.create_result(
                False, GuardrailStatus.FAILURE,
                f"Validation error: {str(e)}",
                execution_time
            )
    
    def _check_repetitive_patterns(self, content: str) -> float:
        """Check for repetitive patterns indicating hallucination"""
        if len(content) < 50:
            return 0.0
        
        # Simple repetition detection
        words = content.lower().split()
        if len(words) < 10:
            return 0.0
        
        # Check for repeated sequences
        repetition_score = 0.0
        for i in range(len(words) - 3):
            sequence = ' '.join(words[i:i+3])
            remaining = ' '.join(words[i+3:])
            if sequence in remaining:
                repetition_score += 0.1
        
        return min(repetition_score, 1.0)
    
    def _check_inconsistencies(self, content: str, context: Dict[str, Any] = None) -> float:
        """Check for logical inconsistencies"""
        # Placeholder for more sophisticated inconsistency detection
        # In real implementation, this would use NLP models
        
        if not context:
            return 0.0
        
        # Basic keyword contradiction check
        inconsistency_score = 0.0
        content_lower = content.lower()
        
        # Look for contradictory statements
        positive_indicators = ['yes', 'true', 'correct', 'accurate']
        negative_indicators = ['no', 'false', 'incorrect', 'wrong']
        
        pos_count = sum(1 for word in positive_indicators if word in content_lower)
        neg_count = sum(1 for word in negative_indicators if word in content_lower)
        
        if pos_count > 0 and neg_count > 0:
            inconsistency_score = min((pos_count + neg_count) / len(content.split()) * 10, 0.5)
        
        return inconsistency_score
    
    def _check_factual_coherence(self, content: str) -> float:
        """Check for factual coherence and plausibility"""
        # Placeholder for factual coherence checking
        # In real implementation, this would use knowledge bases or fact-checking APIs
        
        coherence_score = 0.0
        
        # Basic implausibility detection
        implausible_phrases = [
            'impossible', '999%', 'always works', 'never fails',
            'guarantee', 'certainly will', 'definitely always'
        ]
        
        content_lower = content.lower()
        for phrase in implausible_phrases:
            if phrase in content_lower:
                coherence_score += 0.2
        
        return min(coherence_score, 1.0)


class LLMOutputGuardrail(BaseGuardrail):
    """LLM output validation guardrail based on CrewAI patterns"""
    
    def __init__(self, max_tokens: int = 4000, min_tokens: int = 10, **kwargs):
        super().__init__(name="llm_output_validation", **kwargs)
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
    
    def validate(self, data: Any, context: Dict[str, Any] = None) -> GuardrailResult:
        """Validate LLM output for safety and completeness"""
        start_time = time.time()
        
        try:
            # Type validation
            if not hasattr(data, '__str__') and not isinstance(data, (str, dict)):
                execution_time = (time.time() - start_time) * 1000
                return self.create_result(
                    False, GuardrailStatus.FAILURE,
                    f"Invalid data type: {type(data)}",
                    execution_time
                )
            
            content = str(data)
            tokens = content.split()
            
            validation_details = {
                'token_count': len(tokens),
                'character_count': len(content),
                'max_tokens': self.max_tokens,
                'min_tokens': self.min_tokens,
                'checks_performed': []
            }
            
            checks = []
            
            # Token count validation
            if len(tokens) > self.max_tokens:
                checks.append(('max_tokens', 'exceeded'))
                execution_time = (time.time() - start_time) * 1000
                return self.create_result(
                    False, GuardrailStatus.BLOCKED,
                    f"Output too long: {len(tokens)} tokens (max: {self.max_tokens})",
                    execution_time, validation_details=validation_details
                )
            
            if len(tokens) < self.min_tokens:
                checks.append(('min_tokens', 'insufficient'))
                execution_time = (time.time() - start_time) * 1000
                return self.create_result(
                    False, GuardrailStatus.RETRY_NEEDED,
                    f"Output too short: {len(tokens)} tokens (min: {self.min_tokens})",
                    execution_time, validation_details=validation_details
                )
            
            checks.append(('token_count', 'valid'))
            
            # Content safety validation
            safety_score = self._check_content_safety(content)
            checks.append(('content_safety', safety_score))
            
            if safety_score > 0.7:
                execution_time = (time.time() - start_time) * 1000
                return self.create_result(
                    False, GuardrailStatus.BLOCKED,
                    f"Unsafe content detected (safety score: {safety_score:.3f})",
                    execution_time, validation_details=validation_details
                )
            
            validation_details['checks_performed'] = checks
            validation_details['safety_score'] = safety_score
            
            execution_time = (time.time() - start_time) * 1000
            return self.create_result(
                True, GuardrailStatus.SUCCESS,
                f"Output validated successfully",
                execution_time, validation_details=validation_details
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"LLM output guardrail error: {e}")
            return self.create_result(
                False, GuardrailStatus.FAILURE,
                f"Validation error: {str(e)}",
                execution_time
            )
    
    def _check_content_safety(self, content: str) -> float:
        """Check content for safety issues"""
        unsafe_indicators = [
            'harm', 'dangerous', 'illegal', 'violent', 'threat',
            'attack', 'kill', 'destroy', 'hack', 'exploit',
            'personal information', 'private data', 'password',
            'social security', 'credit card'
        ]
        
        content_lower = content.lower()
        safety_score = 0.0
        
        for indicator in unsafe_indicators:
            if indicator in content_lower:
                safety_score += 0.15
        
        return min(safety_score, 1.0)


class GuardrailSecuritySystem:
    """Comprehensive guardrail security management system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.guardrails: Dict[str, BaseGuardrail] = {}
        self.execution_history: List[GuardrailResult] = []
        self.max_history = 10000
        
        # Register default guardrails
        self._register_default_guardrails()
    
    def register_guardrail(self, guardrail: BaseGuardrail):
        """Register a new guardrail"""
        try:
            self.guardrails[guardrail.name] = guardrail
            self.logger.info(f"Registered guardrail: {guardrail.name}")
            
        except Exception as e:
            error = SecurityError(f"Failed to register guardrail: {str(e)}", "GUARDRAIL_REG_001")
            security_error_handler.handle_error(error)
    
    def execute_guardrail(self, guardrail_name: str, data: Any,
                         context: Dict[str, Any] = None) -> GuardrailResult:
        """Execute specific guardrail with retry logic"""
        if guardrail_name not in self.guardrails:
            raise ValidationError(f"Guardrail not found: {guardrail_name}")
        
        guardrail = self.guardrails[guardrail_name]
        retry_count = 0
        
        while retry_count <= guardrail.max_retries:
            try:
                result = guardrail.validate(data, context)
                result.retry_count = retry_count
                
                # Add to execution history
                self._add_to_history(result)
                
                # Log execution
                self.logger.info(
                    f"Guardrail {guardrail_name} executed: {result.status.value} "
                    f"(attempt {retry_count + 1})"
                )
                
                # Return if successful or not retryable
                if result.success or not result.needs_retry:
                    return result
                
                retry_count += 1
                
            except Exception as e:
                self.logger.error(f"Guardrail execution error: {e}")
                error_result = guardrail.create_result(
                    False, GuardrailStatus.FAILURE,
                    f"Execution error: {str(e)}",
                    0.0, retry_count
                )
                self._add_to_history(error_result)
                return error_result
        
        # Max retries exceeded
        retry_result = guardrail.create_result(
            False, GuardrailStatus.TIMEOUT,
            f"Max retries exceeded ({guardrail.max_retries})",
            0.0, retry_count
        )
        self._add_to_history(retry_result)
        return retry_result
    
    def execute_all_guardrails(self, data: Any, 
                              context: Dict[str, Any] = None) -> Dict[str, GuardrailResult]:
        """Execute all registered guardrails"""
        results = {}
        
        for guardrail_name in self.guardrails:
            try:
                result = self.execute_guardrail(guardrail_name, data, context)
                results[guardrail_name] = result
                
            except Exception as e:
                self.logger.error(f"Error executing guardrail {guardrail_name}: {e}")
                results[guardrail_name] = GuardrailResult(
                    success=False,
                    status=GuardrailStatus.FAILURE,
                    message=f"Execution error: {str(e)}",
                    guardrail_name=guardrail_name,
                    execution_time_ms=0.0
                )
        
        return results
    
    def get_guardrail_statistics(self) -> Dict[str, Any]:
        """Get comprehensive guardrail execution statistics"""
        try:
            if not self.execution_history:
                return {'total_executions': 0}
            
            # Calculate statistics
            total_executions = len(self.execution_history)
            successful = sum(1 for r in self.execution_history if r.success)
            failed = total_executions - successful
            
            # Status distribution
            status_counts = {}
            for result in self.execution_history:
                status = result.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Severity distribution
            severity_counts = {}
            for result in self.execution_history:
                severity = result.severity.value
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Average execution times
            avg_exec_time = sum(r.execution_time_ms for r in self.execution_history) / total_executions
            
            return {
                'total_executions': total_executions,
                'successful_executions': successful,
                'failed_executions': failed,
                'success_rate_pct': (successful / total_executions) * 100,
                'average_execution_time_ms': avg_exec_time,
                'status_distribution': status_counts,
                'severity_distribution': severity_counts,
                'registered_guardrails': list(self.guardrails.keys()),
                'guardrail_count': len(self.guardrails)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating guardrail statistics: {e}")
            return {'error': str(e)}
    
    def _register_default_guardrails(self):
        """Register default CrewAI-based guardrails"""
        try:
            # Hallucination detection
            hallucination_guardrail = HallucinationGuardrail(
                confidence_threshold=0.8,
                severity=GuardrailSeverity.HIGH
            )
            self.register_guardrail(hallucination_guardrail)
            
            # LLM output validation
            llm_output_guardrail = LLMOutputGuardrail(
                max_tokens=4000,
                min_tokens=10,
                severity=GuardrailSeverity.MEDIUM
            )
            self.register_guardrail(llm_output_guardrail)
            
            self.logger.info("Default guardrails registered successfully")
            
        except Exception as e:
            self.logger.error(f"Error registering default guardrails: {e}")
    
    def _add_to_history(self, result: GuardrailResult):
        """Add result to execution history with limit"""
        self.execution_history.append(result)
        
        # Limit history size
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history // 2:]


# Global guardrail security system
guardrail_security_system = GuardrailSecuritySystem()


# Convenience functions
def execute_guardrail(name: str, data: Any, context: Dict[str, Any] = None) -> GuardrailResult:
    """Convenience function to execute guardrail"""
    return guardrail_security_system.execute_guardrail(name, data, context)


def validate_with_guardrails(data: Any, context: Dict[str, Any] = None) -> Dict[str, GuardrailResult]:
    """Convenience function to execute all guardrails"""
    return guardrail_security_system.execute_all_guardrails(data, context)