"""
Autonomous Decision Engine
=========================

Advanced autonomous decision-making system that operates without human intervention,
making intelligent decisions across all TestMaster intelligence frameworks.

Agent A - Hour 19-21: Predictive Intelligence Enhancement
Building upon perfect foundation with predictive intelligence capabilities.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import asyncio
import logging
import json
import hashlib
import uuid
from abc import ABC, abstractmethod

# Decision theory and optimization imports
try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.stats import norm, entropy
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    HAS_ADVANCED_OPTIMIZATION = True
except ImportError:
    HAS_ADVANCED_OPTIMIZATION = False
    logging.warning("Advanced optimization libraries not available. Using simplified methods.")

# Cognitive enhancement imports
try:
    from .pattern_recognition_engine import AdvancedPatternRecognitionEngine
    from ..intelligence import CognitiveEnhancementEngine
    HAS_COGNITIVE_ENHANCEMENT = True
except ImportError:
    HAS_COGNITIVE_ENHANCEMENT = False
    logging.warning("Cognitive enhancement modules not available. Using simplified methods.")


class DecisionUrgency(Enum):
    """Decision urgency levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class DecisionType(Enum):
    """Types of decisions the engine can make"""
    RESOURCE_ALLOCATION = "resource_allocation"
    SCALING_DECISION = "scaling_decision"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    ERROR_RESPONSE = "error_response"
    CAPACITY_MANAGEMENT = "capacity_management"
    SECURITY_ACTION = "security_action"
    MAINTENANCE_SCHEDULING = "maintenance_scheduling"
    CONFIGURATION_CHANGE = "configuration_change"
    PREDICTIVE_ACTION = "predictive_action"
    EMERGENCY_RESPONSE = "emergency_response"


class DecisionStatus(Enum):
    """Status of decision execution"""
    PENDING = "pending"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


@dataclass
class DecisionCriteria:
    """Criteria for making a decision"""
    name: str
    weight: float  # 0-1, importance weight
    value: float   # Current measured value
    target_value: float  # Target/optimal value
    tolerance: float  # Acceptable deviation
    unit: str = ""
    metadata: Dict[str, Any] = None
    
    def satisfaction_score(self) -> float:
        """Calculate how well this criterion is satisfied (0-1)"""
        if self.tolerance == 0:
            return 1.0 if abs(self.value - self.target_value) < 1e-6 else 0.0
        
        deviation = abs(self.value - self.target_value)
        if deviation <= self.tolerance:
            return 1.0
        else:
            # Exponential decay beyond tolerance
            return np.exp(-deviation / self.tolerance)


@dataclass
class DecisionOption:
    """A possible decision option"""
    option_id: str
    name: str
    description: str
    action_type: DecisionType
    parameters: Dict[str, Any]
    expected_outcomes: Dict[str, float]
    risk_score: float  # 0-1, higher is riskier
    cost_score: float  # 0-1, higher is more expensive
    effort_score: float  # 0-1, higher requires more effort
    reversibility: float  # 0-1, higher is more reversible
    confidence: float  # 0-1, confidence in outcomes
    prerequisites: List[str] = None
    side_effects: List[str] = None
    
    def utility_score(self, criteria_weights: Dict[str, float]) -> float:
        """Calculate utility score for this option"""
        utility = 0.0
        
        # Base utility from expected outcomes
        for outcome, value in self.expected_outcomes.items():
            weight = criteria_weights.get(outcome, 0.1)
            utility += weight * value
        
        # Penalties for risk, cost, and effort
        utility -= 0.1 * self.risk_score
        utility -= 0.05 * self.cost_score  
        utility -= 0.03 * self.effort_score
        
        # Bonus for reversibility and confidence
        utility += 0.05 * self.reversibility
        utility += 0.1 * self.confidence
        
        return max(0.0, min(1.0, utility))


@dataclass
class AutonomousDecision:
    """Represents an autonomous decision made by the engine"""
    decision_id: str
    timestamp: datetime
    decision_type: DecisionType
    urgency: DecisionUrgency
    context: Dict[str, Any]
    criteria_evaluated: List[DecisionCriteria]
    options_considered: List[DecisionOption]
    selected_option: DecisionOption
    decision_rationale: str
    confidence_score: float
    risk_assessment: Dict[str, Any]
    rollback_plan: Dict[str, Any]
    execution_timeline: timedelta
    status: DecisionStatus = DecisionStatus.PENDING
    execution_results: Dict[str, Any] = None
    validation_results: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'decision_id': self.decision_id,
            'timestamp': self.timestamp.isoformat(),
            'decision_type': self.decision_type.value,
            'urgency': self.urgency.value,
            'context': self.context,
            'criteria_evaluated': [asdict(c) for c in self.criteria_evaluated],
            'options_considered': [asdict(o) for o in self.options_considered],
            'selected_option': asdict(self.selected_option),
            'decision_rationale': self.decision_rationale,
            'confidence_score': self.confidence_score,
            'risk_assessment': self.risk_assessment,
            'rollback_plan': self.rollback_plan,
            'execution_timeline': str(self.execution_timeline),
            'status': self.status.value,
            'execution_results': self.execution_results,
            'validation_results': self.validation_results
        }


@dataclass
class RiskFactor:
    """Represents a risk factor for decision making"""
    factor_name: str
    probability: float  # 0-1
    impact: float      # 0-1  
    mitigation_cost: float  # 0-1
    mitigation_actions: List[str]
    
    def risk_score(self) -> float:
        """Calculate overall risk score"""
        return self.probability * self.impact


class DecisionValidator(ABC):
    """Abstract base class for decision validators"""
    
    @abstractmethod
    async def validate_decision(self, decision: AutonomousDecision) -> Tuple[bool, str]:
        """Validate a decision before execution"""
        pass
    
    @abstractmethod
    def get_validator_info(self) -> Dict[str, Any]:
        """Get information about the validator"""
        pass


class SafetyValidator(DecisionValidator):
    """Validates decisions for safety concerns"""
    
    def __init__(self, safety_thresholds: Dict[str, float] = None):
        self.safety_thresholds = safety_thresholds or {
            'max_risk_score': 0.7,
            'min_confidence': 0.6,
            'max_cost_score': 0.8,
            'min_reversibility': 0.3
        }
    
    async def validate_decision(self, decision: AutonomousDecision) -> Tuple[bool, str]:
        """Validate decision for safety"""
        try:
            option = decision.selected_option
            
            # Check risk score
            if option.risk_score > self.safety_thresholds['max_risk_score']:
                return False, f"Risk score too high: {option.risk_score}"
            
            # Check confidence
            if decision.confidence_score < self.safety_thresholds['min_confidence']:
                return False, f"Confidence too low: {decision.confidence_score}"
            
            # Check cost
            if option.cost_score > self.safety_thresholds['max_cost_score']:
                return False, f"Cost score too high: {option.cost_score}"
            
            # Check reversibility for high-risk decisions
            if decision.urgency in [DecisionUrgency.HIGH, DecisionUrgency.CRITICAL]:
                if option.reversibility < self.safety_thresholds['min_reversibility']:
                    return False, f"Insufficient reversibility for high urgency: {option.reversibility}"
            
            # Check for emergency conditions
            if decision.urgency == DecisionUrgency.EMERGENCY:
                # Emergency decisions have relaxed safety constraints
                if option.risk_score > 0.9:
                    return False, "Even emergency decisions cannot exceed maximum risk"
            
            return True, "Safety validation passed"
            
        except Exception as e:
            return False, f"Safety validation error: {e}"
    
    def get_validator_info(self) -> Dict[str, Any]:
        """Get validator information"""
        return {
            'validator_type': 'SafetyValidator',
            'safety_thresholds': self.safety_thresholds
        }


class BusinessRuleValidator(DecisionValidator):
    """Validates decisions against business rules"""
    
    def __init__(self, business_rules: Dict[str, Any] = None):
        self.business_rules = business_rules or {
            'max_daily_scaling_decisions': 10,
            'max_hourly_config_changes': 3,
            'maintenance_blackout_hours': [0, 1, 2, 3, 4, 5],  # 12am-6am
            'resource_allocation_limits': {
                'cpu': 0.95,
                'memory': 0.90,
                'storage': 0.85
            }
        }
        self.decision_counters = defaultdict(int)
        self.last_reset = datetime.now()
    
    async def validate_decision(self, decision: AutonomousDecision) -> Tuple[bool, str]:
        """Validate decision against business rules"""
        try:
            self._reset_counters_if_needed()
            
            # Check decision frequency limits
            if decision.decision_type == DecisionType.SCALING_DECISION:
                if self.decision_counters['daily_scaling'] >= self.business_rules['max_daily_scaling_decisions']:
                    return False, "Daily scaling decision limit exceeded"
            
            if decision.decision_type == DecisionType.CONFIGURATION_CHANGE:
                if self.decision_counters['hourly_config'] >= self.business_rules['max_hourly_config_changes']:
                    return False, "Hourly configuration change limit exceeded"
            
            # Check maintenance blackout windows
            if decision.decision_type == DecisionType.MAINTENANCE_SCHEDULING:
                current_hour = datetime.now().hour
                if current_hour in self.business_rules['maintenance_blackout_hours']:
                    return False, f"Maintenance not allowed during blackout hours"
            
            # Check resource allocation limits
            if decision.decision_type == DecisionType.RESOURCE_ALLOCATION:
                for resource, limit in self.business_rules['resource_allocation_limits'].items():
                    if resource in decision.selected_option.parameters:
                        requested = decision.selected_option.parameters[resource]
                        if requested > limit:
                            return False, f"Resource allocation exceeds limit for {resource}: {requested} > {limit}"
            
            # Update counters
            self._update_counters(decision)
            
            return True, "Business rule validation passed"
            
        except Exception as e:
            return False, f"Business rule validation error: {e}"
    
    def _reset_counters_if_needed(self):
        """Reset counters based on time windows"""
        now = datetime.now()
        
        # Reset daily counters
        if (now - self.last_reset).days >= 1:
            self.decision_counters['daily_scaling'] = 0
            self.last_reset = now
        
        # Reset hourly counters
        if (now - self.last_reset).seconds >= 3600:
            self.decision_counters['hourly_config'] = 0
    
    def _update_counters(self, decision: AutonomousDecision):
        """Update decision counters"""
        if decision.decision_type == DecisionType.SCALING_DECISION:
            self.decision_counters['daily_scaling'] += 1
        elif decision.decision_type == DecisionType.CONFIGURATION_CHANGE:
            self.decision_counters['hourly_config'] += 1
    
    def get_validator_info(self) -> Dict[str, Any]:
        """Get validator information"""
        return {
            'validator_type': 'BusinessRuleValidator',
            'business_rules': self.business_rules,
            'current_counters': dict(self.decision_counters)
        }


class EnhancedAutonomousDecisionEngine:
    """
    Advanced autonomous decision-making system with cognitive enhancement and pattern recognition,
    operating without human intervention and making intelligent decisions across all
    TestMaster intelligence frameworks.
    
    Enhanced Features (Hours 115-120):
    - Cognitive reasoning integration
    - Pattern-based decision optimization  
    - Advanced ML model ensemble
    - Real-time learning and adaptation
    - Cross-framework decision coordination
    - Predictive decision modeling
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the Autonomous Decision Engine"""
        self.config = config or self._get_default_config()
        
        # Core components
        self.decision_history: deque = deque(maxlen=1000)
        self.active_decisions: Dict[str, AutonomousDecision] = {}
        self.validators: List[DecisionValidator] = []
        self.decision_templates: Dict[DecisionType, Dict[str, Any]] = {}
        
        # Decision-making components
        self.criteria_weights: Dict[str, float] = {}
        self.learned_patterns: Dict[str, Any] = defaultdict(dict)
        self.risk_models: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'failed_decisions': 0,
            'rolled_back_decisions': 0,
            'average_confidence': 0.0,
            'average_execution_time': 0.0,
            'decision_type_counts': defaultdict(int)
        }
        
        # Initialize components
        self._initialize_validators()
        self._initialize_decision_templates()
        self._initialize_criteria_weights()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Enhanced ML components for learning
        if HAS_ADVANCED_OPTIMIZATION:
            # Ensemble of ML models for robust decision making
            self.decision_classifiers = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boost': GradientBoostingClassifier(n_estimators=50, random_state=42),
                'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
            }
            self.feature_scaler = StandardScaler()
            self.model_trained = False
            self.ensemble_weights = {'random_forest': 0.4, 'gradient_boost': 0.4, 'neural_network': 0.2}
        
        # Cognitive enhancement integration
        if HAS_COGNITIVE_ENHANCEMENT:
            self.cognitive_engine = CognitiveEnhancementEngine()
            self.pattern_engine = AdvancedPatternRecognitionEngine()
            self.cognitive_enabled = True
        else:
            self.cognitive_enabled = False
            
        # Advanced decision intelligence
        self.decision_intelligence = {
            'pattern_memory': deque(maxlen=500),
            'success_patterns': defaultdict(list),
            'failure_patterns': defaultdict(list),
            'optimization_cache': {},
            'adaptive_weights': dict(self.criteria_weights)
        }
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'auto_execution_enabled': True,
            'safety_validation_required': True,
            'business_rule_validation_required': True,
            'max_concurrent_decisions': 5,
            'decision_timeout': timedelta(minutes=30),
            'min_confidence_threshold': 0.7,
            'max_risk_threshold': 0.6,
            'learning_enabled': True,
            'rollback_on_failure': True,
            'emergency_override_enabled': False,
            'audit_logging_enabled': True
        }
    
    def _initialize_validators(self) -> None:
        """Initialize decision validators"""
        if self.config['safety_validation_required']:
            self.validators.append(SafetyValidator())
        
        if self.config['business_rule_validation_required']:
            self.validators.append(BusinessRuleValidator())
    
    def _initialize_decision_templates(self) -> None:
        """Initialize decision templates for common scenarios"""
        self.decision_templates = {
            DecisionType.SCALING_DECISION: {
                'criteria': ['cpu_utilization', 'memory_utilization', 'response_time', 'throughput'],
                'options_generator': self._generate_scaling_options,
                'default_urgency': DecisionUrgency.MEDIUM,
                'execution_timeout': timedelta(minutes=15)
            },
            DecisionType.PERFORMANCE_OPTIMIZATION: {
                'criteria': ['latency', 'throughput', 'error_rate', 'resource_efficiency'],
                'options_generator': self._generate_optimization_options,
                'default_urgency': DecisionUrgency.LOW,
                'execution_timeout': timedelta(minutes=20)
            },
            DecisionType.ERROR_RESPONSE: {
                'criteria': ['error_severity', 'error_frequency', 'impact_scope', 'recovery_feasibility'],
                'options_generator': self._generate_error_response_options,
                'default_urgency': DecisionUrgency.HIGH,
                'execution_timeout': timedelta(minutes=5)
            },
            DecisionType.CAPACITY_MANAGEMENT: {
                'criteria': ['utilization_trend', 'growth_rate', 'capacity_remaining', 'cost_efficiency'],
                'options_generator': self._generate_capacity_options,
                'default_urgency': DecisionUrgency.MEDIUM,
                'execution_timeout': timedelta(minutes=30)
            },
            DecisionType.EMERGENCY_RESPONSE: {
                'criteria': ['severity', 'scope', 'time_criticality', 'available_resources'],
                'options_generator': self._generate_emergency_options,
                'default_urgency': DecisionUrgency.EMERGENCY,
                'execution_timeout': timedelta(minutes=2)
            }
        }
    
    def _initialize_criteria_weights(self) -> None:
        """Initialize criteria weights for decision making"""
        self.criteria_weights = {
            # Performance criteria
            'cpu_utilization': 0.8,
            'memory_utilization': 0.75,
            'response_time': 0.9,
            'throughput': 0.85,
            'error_rate': 0.95,
            
            # Resource criteria  
            'resource_efficiency': 0.7,
            'cost_effectiveness': 0.6,
            'capacity_remaining': 0.8,
            
            # Quality criteria
            'reliability': 0.9,
            'availability': 0.95,
            'maintainability': 0.5,
            
            # Business criteria
            'user_satisfaction': 0.85,
            'business_impact': 0.8,
            'compliance': 0.9
        }
    
    def _setup_logging(self) -> None:
        """Setup logging for the engine"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    async def make_enhanced_decision(self, 
                           decision_type: DecisionType,
                           context: Dict[str, Any],
                           urgency: Optional[DecisionUrgency] = None,
                           override_validation: bool = False,
                           enable_cognitive_reasoning: bool = True) -> AutonomousDecision:
        """
        Make an enhanced autonomous decision with cognitive reasoning and pattern analysis
        
        Args:
            decision_type: Type of decision to make
            context: Context information for decision making
            urgency: Urgency level (optional, will be inferred if not provided)
            override_validation: Skip validation (use with extreme caution)
            enable_cognitive_reasoning: Enable cognitive enhancement for decision making
            
        Returns:
            AutonomousDecision object with enhanced intelligence and reasoning
        """
        try:
            decision_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Enhanced urgency inference with cognitive reasoning
            if urgency is None:
                urgency = await self._infer_urgency_enhanced(decision_type, context)
            
            # Apply cognitive reasoning if enabled
            if enable_cognitive_reasoning and self.cognitive_enabled:
                context = await self._enhance_context_with_cognition(context, decision_type)
                
            # Apply pattern recognition to optimize decision context
            if self.cognitive_enabled:
                context = await self._apply_pattern_recognition(context, decision_type)
            
            # Enhanced criteria evaluation with adaptive learning
            criteria = await self._evaluate_criteria_enhanced(decision_type, context)
            
            # Generate decision options
            options = await self._generate_options(decision_type, context, criteria)
            
            # Enhanced option selection with cognitive reasoning and ensemble ML
            selected_option = await self._select_best_option_enhanced(options, criteria, context)
            
            # Assess risks
            risk_assessment = await self._assess_risks(selected_option, context)
            
            # Create rollback plan
            rollback_plan = await self._create_rollback_plan(selected_option, context)
            
            # Generate decision rationale
            rationale = self._generate_rationale(selected_option, criteria, risk_assessment)
            
            # Enhanced confidence calculation with cognitive assessment
            confidence = await self._calculate_confidence_score_enhanced(selected_option, criteria, risk_assessment, context)
            
            # Create decision object
            decision = AutonomousDecision(
                decision_id=decision_id,
                timestamp=timestamp,
                decision_type=decision_type,
                urgency=urgency,
                context=context,
                criteria_evaluated=criteria,
                options_considered=options,
                selected_option=selected_option,
                decision_rationale=rationale,
                confidence_score=confidence,
                risk_assessment=risk_assessment,
                rollback_plan=rollback_plan,
                execution_timeline=self.decision_templates[decision_type]['execution_timeout']
            )
            
            # Validate decision
            if not override_validation:
                is_valid, validation_message = await self._validate_decision(decision)
                if not is_valid:
                    decision.status = DecisionStatus.CANCELLED
                    self.logger.warning(f"Decision {decision_id} cancelled: {validation_message}")
                    return decision
            
            # Store decision
            self.decision_history.append(decision)
            self.active_decisions[decision_id] = decision
            
            # Execute decision if auto-execution is enabled
            if self.config['auto_execution_enabled'] and decision.status == DecisionStatus.PENDING:
                await self._execute_decision(decision)
            
            # Update performance metrics
            self._update_performance_metrics(decision)
            
            # Learn from decision (if learning enabled)
            if self.config['learning_enabled']:
                await self._learn_from_decision(decision)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Decision making failed: {e}")
            # Return a safe fallback decision
            return self._create_fallback_decision(decision_type, context, str(e))
    
    # Legacy method for backward compatibility
    async def make_decision(self, *args, **kwargs) -> AutonomousDecision:
        """Legacy method that calls enhanced decision making"""
        return await self.make_enhanced_decision(*args, **kwargs)
    
    async def _infer_urgency_enhanced(self, decision_type: DecisionType, context: Dict[str, Any]) -> DecisionUrgency:
        """Enhanced urgency inference with cognitive reasoning"""
        try:
            # Apply cognitive reasoning for urgency assessment
            if self.cognitive_enabled:
                cognitive_assessment = await self.cognitive_engine.reason(
                    f"Assess urgency level for {decision_type.value} decision with context: {context}",
                    context_data=context
                )
                
                # Extract urgency indicators from cognitive assessment
                if 'emergency' in cognitive_assessment.get('reasoning', '').lower():
                    return DecisionUrgency.EMERGENCY
                elif 'critical' in cognitive_assessment.get('reasoning', '').lower():
                    return DecisionUrgency.CRITICAL
                elif 'high' in cognitive_assessment.get('reasoning', '').lower():
                    return DecisionUrgency.HIGH
            
            # Fallback to original logic
            return self._infer_urgency_original(decision_type, context)
            
        except Exception as e:
            self.logger.error(f"Enhanced urgency inference failed: {e}")
            return self._infer_urgency_original(decision_type, context)
    
    def _infer_urgency_original(self, decision_type: DecisionType, context: Dict[str, Any]) -> DecisionUrgency:
        """Infer urgency level based on decision type and context"""
        # Check for emergency indicators
        emergency_indicators = ['system_failure', 'security_breach', 'data_loss', 'critical_error']
        if any(indicator in str(context).lower() for indicator in emergency_indicators):
            return DecisionUrgency.EMERGENCY
        
        # Check for high urgency indicators
        high_urgency_indicators = ['performance_degradation', 'capacity_exceeded', 'error_spike']
        if any(indicator in str(context).lower() for indicator in high_urgency_indicators):
            return DecisionUrgency.HIGH
        
        # Use default urgency from template
        return self.decision_templates.get(decision_type, {}).get('default_urgency', DecisionUrgency.MEDIUM)
    
    async def _enhance_context_with_cognition(self, context: Dict[str, Any], decision_type: DecisionType) -> Dict[str, Any]:
        """Enhance context with cognitive insights"""
        try:
            if not self.cognitive_enabled:
                return context
                
            enhanced_context = dict(context)
            
            # Apply cognitive analysis
            cognitive_insights = await self.cognitive_engine.analyze(
                f"Analyze decision context for {decision_type.value}",
                context
            )
            
            # Add cognitive insights to context
            enhanced_context['cognitive_insights'] = cognitive_insights
            enhanced_context['cognitive_confidence'] = cognitive_insights.get('confidence', 0.5)
            enhanced_context['cognitive_recommendations'] = cognitive_insights.get('recommendations', [])
            
            return enhanced_context
            
        except Exception as e:
            self.logger.error(f"Cognitive context enhancement failed: {e}")
            return context
    
    async def _apply_pattern_recognition(self, context: Dict[str, Any], decision_type: DecisionType) -> Dict[str, Any]:
        """Apply pattern recognition to optimize decision context"""
        try:
            if not self.cognitive_enabled:
                return context
                
            enhanced_context = dict(context)
            
            # Extract patterns from historical data
            pattern_data = {
                'context': context,
                'decision_type': decision_type.value,
                'timestamp': datetime.now().isoformat()
            }
            
            # Analyze patterns
            patterns = await self.pattern_engine.analyze_comprehensive_patterns(
                data_source=pattern_data,
                pattern_types=['temporal', 'behavioral', 'performance', 'anomaly']
            )
            
            # Add pattern insights to context
            enhanced_context['pattern_insights'] = patterns
            enhanced_context['pattern_confidence'] = patterns.get('overall_confidence', 0.5)
            enhanced_context['pattern_predictions'] = patterns.get('predictions', {})
            
            # Store patterns for future learning
            self.decision_intelligence['pattern_memory'].append(pattern_data)
            
            return enhanced_context
            
        except Exception as e:
            self.logger.error(f"Pattern recognition failed: {e}")
            return context
    
    async def _evaluate_criteria_enhanced(self, decision_type: DecisionType, context: Dict[str, Any]) -> List[DecisionCriteria]:
        """Enhanced criteria evaluation with adaptive learning"""
        try:
            # Get base criteria
            base_criteria = await self._evaluate_criteria_original(decision_type, context)
            
            if not self.cognitive_enabled:
                return base_criteria
            
            # Apply adaptive weight learning
            await self._update_adaptive_weights(decision_type, context)
            
            # Enhance criteria with cognitive insights
            enhanced_criteria = []
            for criterion in base_criteria:
                # Apply adaptive weights
                adaptive_weight = self.decision_intelligence['adaptive_weights'].get(
                    criterion.name, criterion.weight
                )
                
                # Create enhanced criterion
                enhanced_criterion = DecisionCriteria(
                    name=criterion.name,
                    weight=adaptive_weight,
                    value=criterion.value,
                    target_value=criterion.target_value,
                    tolerance=criterion.tolerance,
                    unit=criterion.unit,
                    metadata={
                        **criterion.metadata,
                        'adaptive_weight': adaptive_weight,
                        'original_weight': criterion.weight,
                        'cognitive_enhanced': True
                    }
                )
                
                enhanced_criteria.append(enhanced_criterion)
            
            return enhanced_criteria
            
        except Exception as e:
            self.logger.error(f"Enhanced criteria evaluation failed: {e}")
            return await self._evaluate_criteria_original(decision_type, context)
    
    async def _evaluate_criteria_original(self, decision_type: DecisionType, context: Dict[str, Any]) -> List[DecisionCriteria]:
        """Evaluate decision criteria based on current context"""
        criteria = []
        
        try:
            # Get criteria names from template
            template = self.decision_templates.get(decision_type, {})
            criteria_names = template.get('criteria', [])
            
            for criterion_name in criteria_names:
                # Extract criterion value from context
                value = self._extract_criterion_value(criterion_name, context)
                target = self._get_target_value(criterion_name)
                tolerance = self._get_tolerance(criterion_name)
                weight = self.criteria_weights.get(criterion_name, 0.5)
                
                criterion = DecisionCriteria(
                    name=criterion_name,
                    weight=weight,
                    value=value,
                    target_value=target,
                    tolerance=tolerance,
                    metadata={'source': context.get('source', 'unknown')}
                )
                
                criteria.append(criterion)
            
        except Exception as e:
            self.logger.error(f"Criteria evaluation failed: {e}")
        
        return criteria
    
    async def _update_adaptive_weights(self, decision_type: DecisionType, context: Dict[str, Any]) -> None:
        """Update adaptive weights based on historical success patterns"""
        try:
            # Analyze historical success patterns
            decision_key = f"{decision_type.value}"
            
            if decision_key in self.decision_intelligence['success_patterns']:
                success_patterns = self.decision_intelligence['success_patterns'][decision_key]
                
                # Calculate weight adjustments based on successful decisions
                for criterion_name in self.criteria_weights:
                    successful_weights = [
                        pattern.get('weights', {}).get(criterion_name, 0.5)
                        for pattern in success_patterns[-20:]  # Last 20 patterns
                        if criterion_name in pattern.get('weights', {})
                    ]
                    
                    if successful_weights:
                        # Calculate weighted average with decay
                        avg_successful_weight = np.mean(successful_weights)
                        current_weight = self.decision_intelligence['adaptive_weights'].get(
                            criterion_name, self.criteria_weights[criterion_name]
                        )
                        
                        # Apply exponential smoothing
                        alpha = 0.1  # Learning rate
                        new_weight = alpha * avg_successful_weight + (1 - alpha) * current_weight
                        
                        self.decision_intelligence['adaptive_weights'][criterion_name] = new_weight
            
        except Exception as e:
            self.logger.error(f"Adaptive weight update failed: {e}")
    
    def _extract_criterion_value(self, criterion_name: str, context: Dict[str, Any]) -> float:
        """Extract criterion value from context"""
        # This is a simplified extraction - in practice would be more sophisticated
        value_map = {
            'cpu_utilization': context.get('cpu_usage', 0.5),
            'memory_utilization': context.get('memory_usage', 0.5),
            'response_time': context.get('avg_response_time', 100),
            'throughput': context.get('requests_per_second', 100),
            'error_rate': context.get('error_percentage', 0.01),
            'utilization_trend': context.get('growth_rate', 0.05),
            'capacity_remaining': context.get('available_capacity', 0.3),
            'error_severity': context.get('severity_score', 0.5),
            'error_frequency': context.get('error_frequency', 0.1)
        }
        
        return float(value_map.get(criterion_name, 0.5))
    
    def _get_target_value(self, criterion_name: str) -> float:
        """Get target value for a criterion"""
        targets = {
            'cpu_utilization': 0.7,
            'memory_utilization': 0.75,
            'response_time': 50,  # ms
            'throughput': 200,    # rps
            'error_rate': 0.001,  # 0.1%
            'utilization_trend': 0.0,  # stable
            'capacity_remaining': 0.3,  # 30%
            'error_severity': 0.0,
            'error_frequency': 0.0
        }
        
        return targets.get(criterion_name, 0.5)
    
    def _get_tolerance(self, criterion_name: str) -> float:
        """Get tolerance for a criterion"""
        tolerances = {
            'cpu_utilization': 0.1,
            'memory_utilization': 0.1,
            'response_time': 20,
            'throughput': 50,
            'error_rate': 0.005,
            'utilization_trend': 0.02,
            'capacity_remaining': 0.1,
            'error_severity': 0.2,
            'error_frequency': 0.05
        }
        
        return tolerances.get(criterion_name, 0.1)
    
    async def _generate_options(self, decision_type: DecisionType, context: Dict[str, Any], criteria: List[DecisionCriteria]) -> List[DecisionOption]:
        """Generate decision options"""
        try:
            template = self.decision_templates.get(decision_type, {})
            options_generator = template.get('options_generator')
            
            if options_generator:
                return await options_generator(context, criteria)
            else:
                # Fallback to generic options
                return await self._generate_generic_options(decision_type, context, criteria)
                
        except Exception as e:
            self.logger.error(f"Option generation failed: {e}")
            return [self._create_no_action_option()]
    
    async def _generate_scaling_options(self, context: Dict[str, Any], criteria: List[DecisionCriteria]) -> List[DecisionOption]:
        """Generate scaling decision options"""
        options = []
        
        # Scale up option
        options.append(DecisionOption(
            option_id="scale_up",
            name="Scale Up Resources",
            description="Increase system capacity by adding resources",
            action_type=DecisionType.SCALING_DECISION,
            parameters={'action': 'scale_up', 'factor': 1.5},
            expected_outcomes={'cpu_utilization': -0.2, 'response_time': -0.3, 'cost_score': 0.3},
            risk_score=0.2,
            cost_score=0.4,
            effort_score=0.3,
            reversibility=0.8,
            confidence=0.8
        ))
        
        # Scale down option
        options.append(DecisionOption(
            option_id="scale_down",
            name="Scale Down Resources", 
            description="Reduce system capacity to save costs",
            action_type=DecisionType.SCALING_DECISION,
            parameters={'action': 'scale_down', 'factor': 0.8},
            expected_outcomes={'cpu_utilization': 0.1, 'response_time': 0.1, 'cost_score': -0.3},
            risk_score=0.4,
            cost_score=0.1,
            effort_score=0.2,
            reversibility=0.9,
            confidence=0.7
        ))
        
        # Auto-scaling option
        options.append(DecisionOption(
            option_id="auto_scale",
            name="Enable Auto-Scaling",
            description="Configure automatic scaling based on metrics",
            action_type=DecisionType.SCALING_DECISION,
            parameters={'action': 'auto_scale', 'min_instances': 2, 'max_instances': 10},
            expected_outcomes={'cpu_utilization': -0.1, 'response_time': -0.2, 'reliability': 0.2},
            risk_score=0.1,
            cost_score=0.2,
            effort_score=0.5,
            reversibility=0.7,
            confidence=0.9
        ))
        
        return options
    
    async def _generate_optimization_options(self, context: Dict[str, Any], criteria: List[DecisionCriteria]) -> List[DecisionOption]:
        """Generate performance optimization options"""
        options = []
        
        # Cache optimization
        options.append(DecisionOption(
            option_id="optimize_cache",
            name="Optimize Caching",
            description="Improve caching strategy and configuration",
            action_type=DecisionType.PERFORMANCE_OPTIMIZATION,
            parameters={'action': 'optimize_cache', 'cache_size_multiplier': 1.5},
            expected_outcomes={'response_time': -0.4, 'throughput': 0.3},
            risk_score=0.1,
            cost_score=0.2,
            effort_score=0.4,
            reversibility=0.8,
            confidence=0.8
        ))
        
        # Database optimization
        options.append(DecisionOption(
            option_id="optimize_database",
            name="Optimize Database",
            description="Tune database queries and configuration",
            action_type=DecisionType.PERFORMANCE_OPTIMIZATION,
            parameters={'action': 'optimize_database', 'query_timeout': 30},
            expected_outcomes={'response_time': -0.3, 'resource_efficiency': 0.2},
            risk_score=0.3,
            cost_score=0.1,
            effort_score=0.6,
            reversibility=0.6,
            confidence=0.7
        ))
        
        return options
    
    async def _generate_error_response_options(self, context: Dict[str, Any], criteria: List[DecisionCriteria]) -> List[DecisionOption]:
        """Generate error response options"""
        options = []
        
        # Restart service
        options.append(DecisionOption(
            option_id="restart_service",
            name="Restart Service",
            description="Restart the affected service",
            action_type=DecisionType.ERROR_RESPONSE,
            parameters={'action': 'restart', 'service': context.get('service', 'unknown')},
            expected_outcomes={'error_rate': -0.8, 'availability': 0.2},
            risk_score=0.3,
            cost_score=0.1,
            effort_score=0.2,
            reversibility=1.0,
            confidence=0.8
        ))
        
        # Rollback deployment
        options.append(DecisionOption(
            option_id="rollback_deployment",
            name="Rollback Deployment",
            description="Rollback to previous stable version",
            action_type=DecisionType.ERROR_RESPONSE,
            parameters={'action': 'rollback', 'version': 'previous'},
            expected_outcomes={'error_rate': -0.9, 'reliability': 0.5},
            risk_score=0.2,
            cost_score=0.2,
            effort_score=0.4,
            reversibility=0.8,
            confidence=0.9
        ))
        
        return options
    
    async def _generate_capacity_options(self, context: Dict[str, Any], criteria: List[DecisionCriteria]) -> List[DecisionOption]:
        """Generate capacity management options"""
        options = []
        
        # Add capacity
        options.append(DecisionOption(
            option_id="add_capacity",
            name="Add Capacity",
            description="Provision additional system capacity",
            action_type=DecisionType.CAPACITY_MANAGEMENT,
            parameters={'action': 'add_capacity', 'amount': '25%'},
            expected_outcomes={'capacity_remaining': 0.25, 'performance': 0.1},
            risk_score=0.2,
            cost_score=0.5,
            effort_score=0.3,
            reversibility=0.7,
            confidence=0.8
        ))
        
        return options
    
    async def _generate_emergency_options(self, context: Dict[str, Any], criteria: List[DecisionCriteria]) -> List[DecisionOption]:
        """Generate emergency response options"""
        options = []
        
        # Emergency shutdown
        options.append(DecisionOption(
            option_id="emergency_shutdown",
            name="Emergency Shutdown",
            description="Shut down affected systems to prevent further damage",
            action_type=DecisionType.EMERGENCY_RESPONSE,
            parameters={'action': 'shutdown', 'scope': 'affected_systems'},
            expected_outcomes={'damage_prevention': 0.9, 'availability': -0.8},
            risk_score=0.8,
            cost_score=0.9,
            effort_score=0.1,
            reversibility=0.5,
            confidence=0.9
        ))
        
        # Failover
        options.append(DecisionOption(
            option_id="failover",
            name="Failover to Backup",
            description="Switch to backup systems",
            action_type=DecisionType.EMERGENCY_RESPONSE,
            parameters={'action': 'failover', 'target': 'backup_cluster'},
            expected_outcomes={'availability': 0.3, 'performance': -0.2},
            risk_score=0.4,
            cost_score=0.3,
            effort_score=0.2,
            reversibility=0.8,
            confidence=0.7
        ))
        
        return options
    
    async def _generate_generic_options(self, decision_type: DecisionType, context: Dict[str, Any], criteria: List[DecisionCriteria]) -> List[DecisionOption]:
        """Generate generic options for unknown decision types"""
        return [self._create_no_action_option()]
    
    def _create_no_action_option(self) -> DecisionOption:
        """Create a 'no action' option as fallback"""
        return DecisionOption(
            option_id="no_action",
            name="No Action",
            description="Take no action and continue monitoring",
            action_type=DecisionType.PERFORMANCE_OPTIMIZATION,
            parameters={'action': 'none'},
            expected_outcomes={},
            risk_score=0.1,
            cost_score=0.0,
            effort_score=0.0,
            reversibility=1.0,
            confidence=1.0
        )
    
    async def _select_best_option_enhanced(self, options: List[DecisionOption], criteria: List[DecisionCriteria], context: Dict[str, Any]) -> DecisionOption:
        """Enhanced option selection with cognitive reasoning and ensemble ML"""
        if not options:
            return self._create_no_action_option()
        
        try:
            # Apply cognitive reasoning if available
            if self.cognitive_enabled:
                options = await self._apply_cognitive_option_analysis(options, criteria, context)
            
            # Apply ML ensemble prediction if trained models available
            if HAS_ADVANCED_OPTIMIZATION and self.model_trained:
                options = await self._apply_ml_ensemble_scoring(options, criteria, context)
            
            # Use enhanced multi-criteria decision analysis
            return await self._select_best_option_original(options, criteria)
            
        except Exception as e:
            self.logger.error(f"Enhanced option selection failed: {e}")
            return await self._select_best_option_original(options, criteria)
    
    async def _select_best_option_original(self, options: List[DecisionOption], criteria: List[DecisionCriteria]) -> DecisionOption:
        """Select the best option using multi-criteria decision analysis"""
        if not options:
            return self._create_no_action_option()
        
        try:
            # Create criteria weights map
            criteria_weights = {c.name: c.weight for c in criteria}
            
            # Calculate utility scores for all options
            option_scores = []
            for option in options:
                utility = option.utility_score(criteria_weights)
                
                # Adjust for criteria satisfaction
                criteria_satisfaction = np.mean([c.satisfaction_score() for c in criteria])
                adjusted_utility = utility * (0.5 + 0.5 * criteria_satisfaction)
                
                option_scores.append((option, adjusted_utility))
            
            # Sort by utility score
            option_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return best option
            return option_scores[0][0]
            
        except Exception as e:
            self.logger.error(f"Option selection failed: {e}")
            return options[0]  # Return first option as fallback
    
    async def _apply_cognitive_option_analysis(self, options: List[DecisionOption], criteria: List[DecisionCriteria], context: Dict[str, Any]) -> List[DecisionOption]:
        """Apply cognitive analysis to enhance option scoring"""
        try:
            enhanced_options = []
            
            for option in options:
                # Get cognitive assessment of option
                cognitive_analysis = await self.cognitive_engine.evaluate(
                    f"Evaluate decision option: {option.name}",
                    {
                        'option': option.__dict__,
                        'criteria': [c.__dict__ for c in criteria],
                        'context': context
                    }
                )
                
                # Create enhanced option with cognitive insights
                enhanced_option = DecisionOption(
                    option_id=option.option_id,
                    name=option.name,
                    description=option.description,
                    action_type=option.action_type,
                    parameters=option.parameters,
                    expected_outcomes={**option.expected_outcomes, **cognitive_analysis.get('predicted_outcomes', {})},
                    risk_score=min(1.0, option.risk_score * cognitive_analysis.get('risk_multiplier', 1.0)),
                    cost_score=option.cost_score,
                    effort_score=option.effort_score,
                    reversibility=option.reversibility,
                    confidence=max(0.0, min(1.0, option.confidence * cognitive_analysis.get('confidence_multiplier', 1.0))),
                    prerequisites=option.prerequisites,
                    side_effects=option.side_effects
                )
                
                enhanced_options.append(enhanced_option)
            
            return enhanced_options
            
        except Exception as e:
            self.logger.error(f"Cognitive option analysis failed: {e}")
            return options
    
    async def _apply_ml_ensemble_scoring(self, options: List[DecisionOption], criteria: List[DecisionCriteria], context: Dict[str, Any]) -> List[DecisionOption]:
        """Apply ML ensemble to score and rank options"""
        try:
            if not self.model_trained:
                return options
            
            enhanced_options = []
            
            for option in options:
                # Extract features for ML prediction
                features = self._extract_option_features(option, criteria, context)
                features_scaled = self.feature_scaler.transform([features])
                
                # Get ensemble predictions
                ensemble_scores = {}
                for model_name, model in self.decision_classifiers.items():
                    try:
                        probability = model.predict_proba(features_scaled)[0][1]  # Probability of success
                        ensemble_scores[model_name] = probability
                    except:
                        ensemble_scores[model_name] = 0.5  # Neutral score if prediction fails
                
                # Calculate weighted ensemble score
                ensemble_score = sum(
                    ensemble_scores[model] * self.ensemble_weights[model]
                    for model in ensemble_scores
                )
                
                # Enhance option with ML insights
                enhanced_option = DecisionOption(
                    option_id=option.option_id,
                    name=option.name,
                    description=option.description,
                    action_type=option.action_type,
                    parameters=option.parameters,
                    expected_outcomes={**option.expected_outcomes, 'ml_success_probability': ensemble_score},
                    risk_score=option.risk_score * (2.0 - ensemble_score),  # Reduce risk for high probability options
                    cost_score=option.cost_score,
                    effort_score=option.effort_score,
                    reversibility=option.reversibility,
                    confidence=min(1.0, option.confidence * (0.5 + 0.5 * ensemble_score)),
                    prerequisites=option.prerequisites,
                    side_effects=option.side_effects
                )
                
                enhanced_options.append(enhanced_option)
            
            return enhanced_options
            
        except Exception as e:
            self.logger.error(f"ML ensemble scoring failed: {e}")
            return options
    
    def _extract_option_features(self, option: DecisionOption, criteria: List[DecisionCriteria], context: Dict[str, Any]) -> List[float]:
        """Extract features from option for ML prediction"""
        features = [
            option.risk_score,
            option.cost_score,
            option.effort_score,
            option.reversibility,
            option.confidence,
            len(option.expected_outcomes),
            len(criteria),
            context.get('cognitive_confidence', 0.5),
            context.get('pattern_confidence', 0.5),
            float(option.action_type.value == 'scaling_decision'),
            float(option.action_type.value == 'performance_optimization'),
            float(option.action_type.value == 'error_response')
        ]
        
        # Pad or truncate to fixed length
        while len(features) < 12:
            features.append(0.0)
        
        return features[:12]
    
    async def _assess_risks(self, option: DecisionOption, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks associated with the selected option"""
        risk_assessment = {
            'overall_risk': option.risk_score,
            'risk_factors': [],
            'mitigation_strategies': [],
            'contingency_plans': []
        }
        
        try:
            # Identify risk factors based on option characteristics
            if option.risk_score > 0.7:
                risk_assessment['risk_factors'].append('High inherent risk')
                risk_assessment['mitigation_strategies'].append('Implement additional monitoring')
            
            if option.reversibility < 0.5:
                risk_assessment['risk_factors'].append('Low reversibility')
                risk_assessment['mitigation_strategies'].append('Create detailed rollback plan')
            
            if option.confidence < 0.7:
                risk_assessment['risk_factors'].append('Low confidence in outcomes')
                risk_assessment['mitigation_strategies'].append('Implement gradual rollout')
            
            # Add context-specific risks
            if 'critical_system' in context:
                risk_assessment['risk_factors'].append('Critical system impact')
                risk_assessment['mitigation_strategies'].append('Schedule during maintenance window')
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
        
        return risk_assessment
    
    async def _create_rollback_plan(self, option: DecisionOption, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a rollback plan for the decision"""
        rollback_plan = {
            'rollback_feasible': option.reversibility > 0.5,
            'rollback_steps': [],
            'rollback_triggers': [],
            'rollback_timeline': timedelta(minutes=10)
        }
        
        try:
            # Define rollback steps based on action type
            action = option.parameters.get('action', 'unknown')
            
            if action == 'scale_up':
                rollback_plan['rollback_steps'] = ['Scale down to original size', 'Verify performance']
            elif action == 'scale_down':
                rollback_plan['rollback_steps'] = ['Scale up to original size', 'Monitor stability']
            elif action == 'restart':
                rollback_plan['rollback_steps'] = ['Service should auto-recover', 'Manual intervention if needed']
            elif action == 'rollback':
                rollback_plan['rollback_steps'] = ['Deploy previous version again', 'This is already a rollback']
            else:
                rollback_plan['rollback_steps'] = ['Reverse the action', 'Monitor system state']
            
            # Define rollback triggers
            rollback_plan['rollback_triggers'] = [
                'Performance degradation > 20%',
                'Error rate increase > 50%',
                'System instability detected',
                'Manual override requested'
            ]
            
        except Exception as e:
            self.logger.error(f"Rollback plan creation failed: {e}")
        
        return rollback_plan
    
    def _generate_rationale(self, option: DecisionOption, criteria: List[DecisionCriteria], risk_assessment: Dict[str, Any]) -> str:
        """Generate human-readable rationale for the decision"""
        try:
            rationale_parts = []
            
            # Main decision
            rationale_parts.append(f"Selected '{option.name}' based on multi-criteria analysis.")
            
            # Key criteria
            key_criteria = sorted(criteria, key=lambda c: c.weight * c.satisfaction_score(), reverse=True)[:3]
            if key_criteria:
                criteria_names = [c.name.replace('_', ' ') for c in key_criteria]
                rationale_parts.append(f"Primary factors: {', '.join(criteria_names)}.")
            
            # Confidence and risk
            rationale_parts.append(f"Decision confidence: {option.confidence:.1%}.")
            rationale_parts.append(f"Risk level: {option.risk_score:.1%}.")
            
            # Expected outcomes
            if option.expected_outcomes:
                positive_outcomes = [k for k, v in option.expected_outcomes.items() if v > 0]
                if positive_outcomes:
                    outcomes_str = ', '.join(positive_outcomes[:2])
                    rationale_parts.append(f"Expected improvements in: {outcomes_str}.")
            
            return ' '.join(rationale_parts)
            
        except Exception as e:
            self.logger.error(f"Rationale generation failed: {e}")
            return f"Selected {option.name} as the best available option."
    
    async def _calculate_confidence_score_enhanced(self, option: DecisionOption, criteria: List[DecisionCriteria], risk_assessment: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Enhanced confidence calculation with cognitive assessment"""
        try:
            # Base confidence calculation
            base_confidence = self._calculate_confidence_score_original(option, criteria, risk_assessment)
            
            if not self.cognitive_enabled:
                return base_confidence
            
            # Cognitive confidence enhancement
            cognitive_confidence = context.get('cognitive_confidence', 0.5)
            pattern_confidence = context.get('pattern_confidence', 0.5)
            
            # ML ensemble confidence if available
            ml_confidence = option.expected_outcomes.get('ml_success_probability', 0.5)
            
            # Weighted combination of confidence sources
            enhanced_confidence = (
                0.4 * base_confidence +
                0.2 * cognitive_confidence +
                0.2 * pattern_confidence +
                0.2 * ml_confidence
            )
            
            return max(0.0, min(1.0, enhanced_confidence))
            
        except Exception as e:
            self.logger.error(f"Enhanced confidence calculation failed: {e}")
            return self._calculate_confidence_score_original(option, criteria, risk_assessment)
    
    def _calculate_confidence_score_original(self, option: DecisionOption, criteria: List[DecisionCriteria], risk_assessment: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the decision"""
        try:
            # Base confidence from option
            confidence = option.confidence
            
            # Adjust for criteria satisfaction
            criteria_satisfaction = np.mean([c.satisfaction_score() for c in criteria])
            confidence *= (0.7 + 0.3 * criteria_satisfaction)
            
            # Adjust for risk
            risk_penalty = risk_assessment['overall_risk'] * 0.2
            confidence *= (1.0 - risk_penalty)
            
            # Adjust for data quality (simplified)
            if len(criteria) < 3:
                confidence *= 0.9  # Penalize for insufficient criteria
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    # Keep original method name for backward compatibility
    def _calculate_confidence_score(self, option: DecisionOption, criteria: List[DecisionCriteria], risk_assessment: Dict[str, Any]) -> float:
        """Legacy confidence calculation method"""
        return self._calculate_confidence_score_original(option, criteria, risk_assessment)
    
    async def _validate_decision(self, decision: AutonomousDecision) -> Tuple[bool, str]:
        """Validate decision using all configured validators"""
        try:
            for validator in self.validators:
                is_valid, message = await validator.validate_decision(decision)
                if not is_valid:
                    return False, f"{validator.__class__.__name__}: {message}"
            
            return True, "All validations passed"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    async def _execute_decision(self, decision: AutonomousDecision) -> None:
        """Execute the approved decision"""
        try:
            decision.status = DecisionStatus.EXECUTING
            execution_start = datetime.now()
            
            # Simulate decision execution
            # In a real system, this would call actual system APIs
            await self._simulate_execution(decision)
            
            execution_time = (datetime.now() - execution_start).total_seconds()
            
            # Update decision with execution results
            decision.execution_results = {
                'execution_time': execution_time,
                'status': 'completed',
                'timestamp': datetime.now().isoformat()
            }
            decision.status = DecisionStatus.COMPLETED
            
            self.logger.info(f"Decision {decision.decision_id} executed successfully in {execution_time:.2f}s")
            
        except Exception as e:
            decision.status = DecisionStatus.FAILED
            decision.execution_results = {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.error(f"Decision {decision.decision_id} execution failed: {e}")
            
            # Attempt rollback if configured
            if self.config['rollback_on_failure']:
                await self._rollback_decision(decision)
    
    async def _simulate_execution(self, decision: AutonomousDecision) -> None:
        """Simulate decision execution (placeholder for actual implementation)"""
        # This would be replaced with actual system calls in production
        await asyncio.sleep(0.1)  # Simulate execution time
        
        # Simulate occasional failures for testing
        if decision.selected_option.risk_score > 0.8:
            if np.random.random() < 0.1:  # 10% failure rate for high-risk decisions
                raise Exception("Simulated execution failure")
    
    async def _rollback_decision(self, decision: AutonomousDecision) -> None:
        """Rollback a failed decision"""
        try:
            if not decision.rollback_plan.get('rollback_feasible', False):
                self.logger.warning(f"Rollback not feasible for decision {decision.decision_id}")
                return
            
            self.logger.info(f"Rolling back decision {decision.decision_id}")
            
            # Execute rollback steps
            for step in decision.rollback_plan.get('rollback_steps', []):
                self.logger.info(f"Rollback step: {step}")
                await asyncio.sleep(0.05)  # Simulate rollback time
            
            decision.status = DecisionStatus.ROLLED_BACK
            self.performance_metrics['rolled_back_decisions'] += 1
            
        except Exception as e:
            self.logger.error(f"Rollback failed for decision {decision.decision_id}: {e}")
    
    def _create_fallback_decision(self, decision_type: DecisionType, context: Dict[str, Any], error: str) -> AutonomousDecision:
        """Create a safe fallback decision when normal processing fails"""
        fallback_option = self._create_no_action_option()
        
        return AutonomousDecision(
            decision_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            decision_type=decision_type,
            urgency=DecisionUrgency.LOW,
            context=context,
            criteria_evaluated=[],
            options_considered=[fallback_option],
            selected_option=fallback_option,
            decision_rationale=f"Fallback decision due to error: {error}",
            confidence_score=0.3,
            risk_assessment={'overall_risk': 0.1, 'fallback': True},
            rollback_plan={'rollback_feasible': True},
            execution_timeline=timedelta(minutes=1),
            status=DecisionStatus.COMPLETED
        )
    
    def _update_performance_metrics(self, decision: AutonomousDecision) -> None:
        """Update performance tracking metrics"""
        self.performance_metrics['total_decisions'] += 1
        self.performance_metrics['decision_type_counts'][decision.decision_type.value] += 1
        
        if decision.status == DecisionStatus.COMPLETED:
            self.performance_metrics['successful_decisions'] += 1
        elif decision.status == DecisionStatus.FAILED:
            self.performance_metrics['failed_decisions'] += 1
        
        # Update averages
        total = self.performance_metrics['total_decisions']
        if total > 0:
            success_rate = self.performance_metrics['successful_decisions'] / total
            self.performance_metrics['success_rate'] = success_rate
        
        # Update average confidence
        if decision.confidence_score > 0:
            current_avg = self.performance_metrics['average_confidence']
            self.performance_metrics['average_confidence'] = (
                (current_avg * (total - 1) + decision.confidence_score) / total
            )
    
    async def _learn_from_decision(self, decision: AutonomousDecision) -> None:
        """Learn from decision outcomes to improve future decisions"""
        try:
            if not HAS_ADVANCED_OPTIMIZATION or not self.config['learning_enabled']:
                return
            
            # Extract features for learning
            features = self._extract_decision_features(decision)
            outcome = 1.0 if decision.status == DecisionStatus.COMPLETED else 0.0
            
            # Update learned patterns
            decision_key = f"{decision.decision_type.value}_{decision.urgency.value}"
            if decision_key not in self.learned_patterns:
                self.learned_patterns[decision_key] = {
                    'successful_features': [],
                    'failed_features': [],
                    'success_rate': 0.0,
                    'sample_count': 0
                }
            
            pattern = self.learned_patterns[decision_key]
            pattern['sample_count'] += 1
            
            if outcome > 0.5:
                pattern['successful_features'].append(features)
                # Store success pattern for adaptive learning
                success_pattern = {
                    'decision_type': decision.decision_type.value,
                    'weights': {c.name: c.weight for c in decision.criteria_evaluated},
                    'features': features,
                    'confidence': decision.confidence_score,
                    'timestamp': decision.timestamp.isoformat()
                }
                self.decision_intelligence['success_patterns'][decision_key].append(success_pattern)
            else:
                pattern['failed_features'].append(features)
                # Store failure pattern for learning
                failure_pattern = {
                    'decision_type': decision.decision_type.value,
                    'weights': {c.name: c.weight for c in decision.criteria_evaluated},
                    'features': features,
                    'error': decision.execution_results.get('error', 'unknown'),
                    'timestamp': decision.timestamp.isoformat()
                }
                self.decision_intelligence['failure_patterns'][decision_key].append(failure_pattern)
            
            # Update success rate
            pattern['success_rate'] = (
                (pattern['success_rate'] * (pattern['sample_count'] - 1) + outcome) 
                / pattern['sample_count']
            )
            
            # Retrain model periodically
            if pattern['sample_count'] % 10 == 0:
                await self._retrain_decision_model()
            
        except Exception as e:
            self.logger.error(f"Learning from decision failed: {e}")
    
    def _extract_decision_features(self, decision: AutonomousDecision) -> List[float]:
        """Extract numerical features from a decision for machine learning"""
        features = []
        
        try:
            # Decision characteristics
            features.append(float(decision.urgency.value == 'high'))
            features.append(float(decision.urgency.value == 'critical'))
            features.append(decision.confidence_score)
            features.append(decision.selected_option.risk_score)
            features.append(decision.selected_option.cost_score)
            features.append(decision.selected_option.effort_score)
            features.append(decision.selected_option.reversibility)
            
            # Criteria satisfaction
            if decision.criteria_evaluated:
                satisfaction_scores = [c.satisfaction_score() for c in decision.criteria_evaluated]
                features.extend(satisfaction_scores[:5])  # Limit to first 5 criteria
                
                # Pad with zeros if fewer than 5 criteria
                while len(features) < 12:  # 7 base features + 5 criteria
                    features.append(0.0)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            features = [0.0] * 12  # Return zero features as fallback
        
        return features
    
    async def _retrain_decision_model(self) -> None:
        """Retrain the decision prediction model"""
        try:
            if not HAS_ADVANCED_OPTIMIZATION:
                return
            
            # Collect training data from decision history
            X = []
            y = []
            
            for decision in list(self.decision_history):
                if decision.status in [DecisionStatus.COMPLETED, DecisionStatus.FAILED]:
                    features = self._extract_decision_features(decision)
                    outcome = 1 if decision.status == DecisionStatus.COMPLETED else 0
                    X.append(features)
                    y.append(outcome)
            
            if len(X) < 20:  # Need minimum samples for training
                return
            
            # Train ensemble models
            X = np.array(X)
            y = np.array(y)
            
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train each model in ensemble
            model_scores = {}
            for model_name, model in self.decision_classifiers.items():
                try:
                    model.fit(X_scaled, y)
                    if len(X) > 5:
                        scores = cross_val_score(model, X_scaled, y, cv=min(5, len(X)))
                        model_scores[model_name] = np.mean(scores)
                    else:
                        model_scores[model_name] = 0.5
                except Exception as e:
                    self.logger.warning(f"Model {model_name} training failed: {e}")
                    model_scores[model_name] = 0.0
            
            # Update ensemble weights based on performance
            total_score = sum(model_scores.values())
            if total_score > 0:
                for model_name in self.ensemble_weights:
                    self.ensemble_weights[model_name] = model_scores.get(model_name, 0.1) / total_score
            
            self.model_trained = True
            avg_score = np.mean(list(model_scores.values()))
            self.logger.info(f"Decision ensemble retrained with average accuracy: {avg_score:.3f}")
            self.logger.info(f"Ensemble weights: {self.ensemble_weights}")
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
    
    async def get_decision_status(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific decision"""
        if decision_id in self.active_decisions:
            decision = self.active_decisions[decision_id]
            return decision.to_dict()
        
        # Search in history
        for decision in self.decision_history:
            if decision.decision_id == decision_id:
                return decision.to_dict()
        
        return None
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status"""
        return {
            'version': '1.0.0',
            'status': 'active',
            'performance_metrics': dict(self.performance_metrics),
            'active_decisions': len(self.active_decisions),
            'decision_history_size': len(self.decision_history),
            'validators_configured': len(self.validators),
            'learning_enabled': self.config['learning_enabled'],
            'model_trained': getattr(self, 'model_trained', False),
            'learned_patterns': len(self.learned_patterns),
            'config': self.config
        }
    
    async def get_decision_recommendations(self, decision_type: DecisionType, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations for a decision without actually making it"""
        try:
            # Evaluate criteria
            criteria = await self._evaluate_criteria(decision_type, context)
            
            # Generate options
            options = await self._generate_options(decision_type, context, criteria)
            
            # Rank options
            criteria_weights = {c.name: c.weight for c in criteria}
            option_rankings = []
            
            for option in options:
                utility = option.utility_score(criteria_weights)
                option_rankings.append({
                    'option': asdict(option),
                    'utility_score': utility,
                    'recommended': utility > 0.7
                })
            
            option_rankings.sort(key=lambda x: x['utility_score'], reverse=True)
            
            return {
                'decision_type': decision_type.value,
                'context_analyzed': True,
                'criteria_evaluated': [asdict(c) for c in criteria],
                'option_rankings': option_rankings,
                'top_recommendation': option_rankings[0] if option_rankings else None,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return {'error': str(e)}


# Factory functions for easy instantiation
def create_autonomous_decision_engine(config: Dict[str, Any] = None) -> EnhancedAutonomousDecisionEngine:
    """Create and return a configured Enhanced Autonomous Decision Engine"""
    return EnhancedAutonomousDecisionEngine(config)

def create_enhanced_autonomous_decision_engine(config: Dict[str, Any] = None) -> EnhancedAutonomousDecisionEngine:
    """Create and return a configured Enhanced Autonomous Decision Engine with cognitive capabilities"""
    enhanced_config = config or {}
    enhanced_config['cognitive_enhancement'] = True
    enhanced_config['pattern_recognition'] = True
    enhanced_config['ensemble_ml'] = True
    return EnhancedAutonomousDecisionEngine(enhanced_config)

# Backward compatibility alias
AutonomousDecisionEngine = EnhancedAutonomousDecisionEngine


# Export main classes
__all__ = [
    'EnhancedAutonomousDecisionEngine',
    'AutonomousDecisionEngine',  # Backward compatibility
    'AutonomousDecision',
    'DecisionOption', 
    'DecisionCriteria',
    'DecisionType',
    'DecisionUrgency',
    'DecisionStatus',
    'create_autonomous_decision_engine',
    'create_enhanced_autonomous_decision_engine'
]