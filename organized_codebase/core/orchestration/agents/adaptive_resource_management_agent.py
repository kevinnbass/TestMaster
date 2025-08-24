"""
Adaptive Resource Management Agent

Intelligent adaptive resource management system that dynamically optimizes
resource allocation based on workload patterns, predictive analytics, and
machine learning-driven resource scaling with consensus-based decision making.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
import statistics
import json
from collections import defaultdict, deque
import math

from ..hierarchical_planning import (
    HierarchicalTestPlanner, 
    PlanningNode, 
    TestPlanGenerator, 
    TestPlanEvaluator,
    EvaluationCriteria,
    get_best_planner
)
from ..consensus import AgentCoordinator, AgentVote
from ..consensus.agent_coordination import AgentRole
from ...core.shared_state import get_shared_state, cache_test_result, get_cached_test_result
from ...flow_optimizer.resource_optimizer import get_resource_optimizer, ResourceType, OptimizationPolicy, ResourcePool
from ...telemetry.performance_monitor import get_performance_monitor


class ScalingDirection(Enum):
    """Direction of resource scaling."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class ScalingTrigger(Enum):
    """Triggers for resource scaling."""
    UTILIZATION_THRESHOLD = "utilization_threshold"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    QUEUE_BUILDUP = "queue_buildup"
    PREDICTIVE_DEMAND = "predictive_demand"
    COST_OPTIMIZATION = "cost_optimization"
    MANUAL_REQUEST = "manual_request"


class ResourceStrategy(Enum):
    """Resource management strategies."""
    REACTIVE = "reactive"           # React to current conditions
    PREDICTIVE = "predictive"       # Predict future needs
    PROACTIVE = "proactive"         # Anticipate and prepare
    HYBRID = "hybrid"              # Combination of strategies


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    resource_type: ResourceType
    current_utilization: float     # 0-1 scale
    peak_utilization: float        # Peak in current period
    average_utilization: float     # Average over period
    trend: float                   # Utilization trend (-1 to 1)
    efficiency: float              # Resource efficiency score
    cost_per_hour: float
    performance_impact: float      # Impact on performance
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingDecision:
    """Resource scaling decision."""
    decision_id: str
    resource_type: ResourceType
    direction: ScalingDirection
    trigger: ScalingTrigger
    current_capacity: float
    target_capacity: float
    scaling_factor: float
    confidence: float
    reasoning: str
    estimated_cost_impact: float
    estimated_performance_impact: float
    execution_priority: float
    created_at: datetime = field(default_factory=datetime.now)
    executed: bool = False
    execution_results: Optional[Dict[str, Any]] = None


@dataclass
class ResourcePrediction:
    """Resource demand prediction."""
    resource_type: ResourceType
    predicted_demand: float
    prediction_horizon_minutes: int
    confidence: float
    factors: List[str]
    historical_accuracy: float
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AdaptiveConfiguration:
    """Configuration for adaptive resource management."""
    # Scaling thresholds
    scale_up_threshold: float = 0.8        # 80% utilization
    scale_down_threshold: float = 0.3       # 30% utilization
    
    # Prediction settings
    prediction_window_hours: int = 2
    prediction_accuracy_threshold: float = 0.7
    
    # Scaling parameters
    min_scaling_factor: float = 1.1         # Minimum 10% increase
    max_scaling_factor: float = 3.0         # Maximum 300% increase
    scaling_cooldown_minutes: int = 15      # Wait time between scalings
    
    # Cost optimization
    max_cost_increase: float = 0.5          # Maximum 50% cost increase
    cost_efficiency_weight: float = 0.3     # Weight in decision making
    
    # Strategy
    default_strategy: ResourceStrategy = ResourceStrategy.HYBRID
    enable_predictive_scaling: bool = True
    enable_cost_optimization: bool = True


class ResourcePredictor:
    """Predicts future resource needs based on historical patterns."""
    
    def __init__(self, config: AdaptiveConfiguration):
        self.config = config
        self.historical_data: Dict[ResourceType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.prediction_accuracy: Dict[ResourceType, deque] = defaultdict(lambda: deque(maxlen=100))
        
        print("Resource Predictor initialized")
        print(f"   Prediction window: {config.prediction_window_hours} hours")
        print(f"   Accuracy threshold: {config.prediction_accuracy_threshold}")
    
    def add_observation(self, resource_type: ResourceType, utilization: float, timestamp: datetime = None):
        """Add historical observation for prediction model."""
        timestamp = timestamp or datetime.now()
        
        observation = {
            'utilization': utilization,
            'timestamp': timestamp,
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'minute_of_hour': timestamp.minute
        }
        
        self.historical_data[resource_type].append(observation)
    
    def predict_demand(self, resource_type: ResourceType, horizon_minutes: int = None) -> ResourcePrediction:
        """Predict future resource demand."""
        
        horizon_minutes = horizon_minutes or (self.config.prediction_window_hours * 60)
        current_time = datetime.now()
        target_time = current_time + timedelta(minutes=horizon_minutes)
        
        # Get historical data for this resource type
        history = list(self.historical_data[resource_type])
        
        if len(history) < 10:
            # Not enough data for prediction
            return ResourcePrediction(
                resource_type=resource_type,
                predicted_demand=0.5,  # Default prediction
                prediction_horizon_minutes=horizon_minutes,
                confidence=0.1,
                factors=["insufficient_data"],
                historical_accuracy=0.0
            )
        
        # Simple time-series prediction based on patterns
        predicted_demand = self._calculate_time_based_prediction(history, target_time)
        
        # Calculate trend-based adjustment
        trend_adjustment = self._calculate_trend_adjustment(history)
        predicted_demand = max(0.0, min(1.0, predicted_demand + trend_adjustment))
        
        # Calculate confidence based on historical accuracy
        confidence = self._calculate_prediction_confidence(resource_type, history)
        
        # Identify prediction factors
        factors = self._identify_prediction_factors(history, target_time)
        
        # Get historical accuracy
        historical_accuracy = self._get_historical_accuracy(resource_type)
        
        return ResourcePrediction(
            resource_type=resource_type,
            predicted_demand=predicted_demand,
            prediction_horizon_minutes=horizon_minutes,
            confidence=confidence,
            factors=factors,
            historical_accuracy=historical_accuracy
        )
    
    def _calculate_time_based_prediction(self, history: List[Dict], target_time: datetime) -> float:
        """Calculate prediction based on time patterns."""
        
        target_hour = target_time.hour
        target_day = target_time.weekday()
        
        # Find similar time periods
        similar_periods = []
        for obs in history:
            if (obs['hour_of_day'] == target_hour and 
                obs['day_of_week'] == target_day):
                similar_periods.append(obs['utilization'])
        
        if similar_periods:
            return statistics.mean(similar_periods)
        
        # Fall back to hour-based prediction
        hour_based = [obs['utilization'] for obs in history if obs['hour_of_day'] == target_hour]
        if hour_based:
            return statistics.mean(hour_based)
        
        # Fall back to overall average
        return statistics.mean([obs['utilization'] for obs in history])
    
    def _calculate_trend_adjustment(self, history: List[Dict]) -> float:
        """Calculate trend adjustment based on recent data."""
        
        if len(history) < 20:
            return 0.0
        
        # Get recent trend (last 20 observations)
        recent_data = history[-20:]
        recent_utilizations = [obs['utilization'] for obs in recent_data]
        
        # Calculate linear trend
        n = len(recent_utilizations)
        x_values = list(range(n))
        
        # Simple linear regression
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(recent_utilizations)
        
        numerator = sum((x_values[i] - x_mean) * (recent_utilizations[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator != 0:
            slope = numerator / denominator
            # Scale trend to reasonable adjustment (-0.2 to 0.2)
            return max(-0.2, min(0.2, slope * 10))
        
        return 0.0
    
    def _calculate_prediction_confidence(self, resource_type: ResourceType, history: List[Dict]) -> float:
        """Calculate confidence in prediction."""
        
        # Base confidence on data quality
        data_quality = min(1.0, len(history) / 100.0)  # More data = higher confidence
        
        # Factor in historical accuracy if available
        accuracy_scores = list(self.prediction_accuracy[resource_type])
        if accuracy_scores:
            avg_accuracy = statistics.mean(accuracy_scores)
            accuracy_factor = avg_accuracy
        else:
            accuracy_factor = 0.5  # Neutral when no history
        
        # Factor in data variance (lower variance = higher confidence)
        utilizations = [obs['utilization'] for obs in history[-50:]]  # Last 50 observations
        if len(utilizations) > 1:
            variance = statistics.stdev(utilizations)
            variance_factor = max(0.0, 1.0 - variance)  # Lower variance = higher confidence
        else:
            variance_factor = 0.5
        
        # Combine factors
        confidence = (data_quality * 0.3 + accuracy_factor * 0.4 + variance_factor * 0.3)
        return max(0.1, min(0.95, confidence))
    
    def _identify_prediction_factors(self, history: List[Dict], target_time: datetime) -> List[str]:
        """Identify factors influencing the prediction."""
        
        factors = []
        
        # Time-based factors
        target_hour = target_time.hour
        if 9 <= target_hour <= 17:
            factors.append("business_hours")
        elif 18 <= target_hour <= 22:
            factors.append("evening_peak")
        else:
            factors.append("off_hours")
        
        # Day-based factors
        target_day = target_time.weekday()
        if target_day < 5:
            factors.append("weekday")
        else:
            factors.append("weekend")
        
        # Trend factors
        if len(history) >= 10:
            recent_trend = self._calculate_trend_adjustment(history)
            if recent_trend > 0.05:
                factors.append("increasing_demand")
            elif recent_trend < -0.05:
                factors.append("decreasing_demand")
            else:
                factors.append("stable_demand")
        
        return factors
    
    def _get_historical_accuracy(self, resource_type: ResourceType) -> float:
        """Get historical prediction accuracy."""
        accuracy_scores = list(self.prediction_accuracy[resource_type])
        if accuracy_scores:
            return statistics.mean(accuracy_scores)
        return 0.0
    
    def update_prediction_accuracy(self, resource_type: ResourceType, predicted: float, 
                                 actual: float, prediction_time: datetime):
        """Update prediction accuracy tracking."""
        
        # Calculate accuracy score (1 - normalized error)
        error = abs(predicted - actual)
        accuracy = max(0.0, 1.0 - error)
        
        self.prediction_accuracy[resource_type].append(accuracy)


class ResourceScaler:
    """Executes resource scaling decisions."""
    
    def __init__(self, config: AdaptiveConfiguration):
        self.config = config
        self.resource_optimizer = get_resource_optimizer()
        self.shared_state = get_shared_state()
        self.scaling_history: List[ScalingDecision] = []
        self.last_scaling: Dict[ResourceType, datetime] = {}
        
        print("Resource Scaler initialized")
        print(f"   Scaling factors: {config.min_scaling_factor} - {config.max_scaling_factor}")
        print(f"   Cooldown period: {config.scaling_cooldown_minutes} minutes")
    
    def can_scale(self, resource_type: ResourceType, direction: ScalingDirection) -> bool:
        """Check if scaling is allowed based on cooldown and constraints."""
        
        # Check cooldown period
        if resource_type in self.last_scaling:
            time_since_last = datetime.now() - self.last_scaling[resource_type]
            if time_since_last.total_seconds() < (self.config.scaling_cooldown_minutes * 60):
                return False
        
        # Check resource pool availability
        pools = [pool for pool in self.resource_optimizer.resource_pools.values() 
                if pool.resource_type == resource_type]
        
        if not pools:
            return False
        
        # Additional constraints can be added here
        return True
    
    def execute_scaling(self, decision: ScalingDecision) -> Dict[str, Any]:
        """Execute a scaling decision."""
        
        if not self.can_scale(decision.resource_type, decision.direction):
            return {
                'success': False,
                'error': 'Scaling not allowed due to cooldown or constraints',
                'decision_id': decision.decision_id
            }
        
        print(f"Executing scaling: {decision.direction.value} for {decision.resource_type.value}")
        print(f"   Target capacity: {decision.target_capacity}")
        print(f"   Reasoning: {decision.reasoning}")
        
        execution_start = datetime.now()
        
        # Execute scaling based on direction
        if decision.direction == ScalingDirection.SCALE_UP:
            result = self._scale_up_resource(decision)
        elif decision.direction == ScalingDirection.SCALE_DOWN:
            result = self._scale_down_resource(decision)
        else:  # MAINTAIN
            result = {'success': True, 'action': 'no_scaling_needed'}
        
        execution_time = (datetime.now() - execution_start).total_seconds()
        
        # Update decision with execution results
        decision.executed = True
        decision.execution_results = {
            **result,
            'execution_time_seconds': execution_time,
            'execution_timestamp': execution_start.isoformat()
        }
        
        # Update last scaling time
        self.last_scaling[decision.resource_type] = execution_start
        
        # Store in history
        self.scaling_history.append(decision)
        
        # Store in shared state
        self._store_scaling_result(decision)
        
        return result
    
    def _scale_up_resource(self, decision: ScalingDecision) -> Dict[str, Any]:
        """Scale up resource capacity."""
        
        # Find pools of the target resource type
        target_pools = [pool for pool in self.resource_optimizer.resource_pools.values() 
                       if pool.resource_type == decision.resource_type]
        
        if not target_pools:
            return {'success': False, 'error': 'No pools found for resource type'}
        
        # Scale up the pools
        total_capacity_added = 0.0
        pools_scaled = 0
        
        for pool in target_pools:
            # Calculate new capacity
            capacity_increase = pool.total_capacity * (decision.scaling_factor - 1.0)
            new_total_capacity = pool.total_capacity + capacity_increase
            new_available_capacity = pool.available_capacity + capacity_increase
            
            # Update pool
            pool.total_capacity = new_total_capacity
            pool.available_capacity = new_available_capacity
            
            total_capacity_added += capacity_increase
            pools_scaled += 1
        
        return {
            'success': True,
            'action': 'scale_up',
            'pools_scaled': pools_scaled,
            'total_capacity_added': total_capacity_added,
            'new_total_capacity': sum(pool.total_capacity for pool in target_pools)
        }
    
    def _scale_down_resource(self, decision: ScalingDecision) -> Dict[str, Any]:
        """Scale down resource capacity."""
        
        # Find pools of the target resource type
        target_pools = [pool for pool in self.resource_optimizer.resource_pools.values() 
                       if pool.resource_type == decision.resource_type]
        
        if not target_pools:
            return {'success': False, 'error': 'No pools found for resource type'}
        
        # Scale down the pools
        total_capacity_removed = 0.0
        pools_scaled = 0
        
        for pool in target_pools:
            # Calculate new capacity (ensure we don't go below current usage)
            current_usage = pool.total_capacity - pool.available_capacity
            capacity_decrease = pool.total_capacity * (1.0 - 1.0/decision.scaling_factor)
            
            # Ensure we don't reduce below current usage
            max_decrease = pool.total_capacity - current_usage - 10.0  # Keep 10 units buffer
            capacity_decrease = min(capacity_decrease, max(0, max_decrease))
            
            if capacity_decrease > 0:
                new_total_capacity = pool.total_capacity - capacity_decrease
                new_available_capacity = pool.available_capacity - capacity_decrease
                
                # Update pool
                pool.total_capacity = new_total_capacity
                pool.available_capacity = max(0, new_available_capacity)
                
                total_capacity_removed += capacity_decrease
                pools_scaled += 1
        
        return {
            'success': True,
            'action': 'scale_down',
            'pools_scaled': pools_scaled,
            'total_capacity_removed': total_capacity_removed,
            'new_total_capacity': sum(pool.total_capacity for pool in target_pools)
        }
    
    def _store_scaling_result(self, decision: ScalingDecision):
        """Store scaling result in shared state."""
        
        scaling_data = {
            'decision_id': decision.decision_id,
            'resource_type': decision.resource_type.value,
            'direction': decision.direction.value,
            'scaling_factor': decision.scaling_factor,
            'success': decision.execution_results.get('success', False),
            'timestamp': decision.created_at.isoformat()
        }
        
        # Store in shared state
        key = f"resource_scaling_{decision.resource_type.value}"
        self.shared_state.set(key, scaling_data)
        
        # Cache the result
        cache_test_result(key, scaling_data, decision.confidence * 100)


class AdaptiveResourceManagementAgent:
    """Main adaptive resource management agent."""
    
    def __init__(self, 
                 coordinator: AgentCoordinator = None,
                 config: AdaptiveConfiguration = None):
        
        self.coordinator = coordinator
        self.config = config or AdaptiveConfiguration()
        self.shared_state = get_shared_state()
        
        # Initialize components
        self.resource_optimizer = get_resource_optimizer()
        self.performance_monitor = get_performance_monitor()
        self.predictor = ResourcePredictor(self.config)
        self.scaler = ResourceScaler(self.config)
        
        # Agent state
        self.current_metrics: Dict[ResourceType, ResourceMetrics] = {}
        self.scaling_decisions: List[ScalingDecision] = []
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.decisions_made = 0
        self.successful_scalings = 0
        self.cost_savings = 0.0
        self.performance_improvements = 0.0
        
        # Register with coordinator if provided
        if self.coordinator:
            self.coordinator.register_agent(
                "adaptive_resource_management_agent",
                AgentRole.PERFORMANCE_OPTIMIZER,
                weight=1.3,
                specialization=["resource_management", "predictive_scaling", "cost_optimization"]
            )
        
        print("Adaptive Resource Management Agent initialized")
        print(f"   Strategy: {self.config.default_strategy.value}")
        print(f"   Predictive scaling: {self.config.enable_predictive_scaling}")
        print(f"   Cost optimization: {self.config.enable_cost_optimization}")
    
    def start_adaptive_management(self):
        """Start adaptive resource management."""
        
        if self.monitoring_active:
            print("Adaptive management already active")
            return
        
        self.monitoring_active = True
        self.shutdown_event.clear()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._management_loop, daemon=True)
        self.monitoring_thread.start()
        
        print("Adaptive resource management started")
    
    def stop_adaptive_management(self):
        """Stop adaptive resource management."""
        
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.shutdown_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        print("Adaptive resource management stopped")
    
    def _management_loop(self):
        """Main adaptive management loop."""
        
        while self.monitoring_active and not self.shutdown_event.is_set():
            try:
                # Collect current resource metrics
                self._collect_resource_metrics()
                
                # Update predictor with current observations
                self._update_predictor()
                
                # Analyze resource needs and make scaling decisions
                self._analyze_and_decide()
                
                # Execute approved scaling decisions
                self._execute_scaling_decisions()
                
                # Wait for next cycle (30 seconds)
                if self.shutdown_event.wait(timeout=30):
                    break
                    
            except Exception as e:
                print(f"Adaptive management loop error: {e}")
                time.sleep(10)  # Brief pause before retrying
    
    def _collect_resource_metrics(self):
        """Collect current resource utilization metrics."""
        
        # Get resource utilization from optimizer
        utilization_data = self.resource_optimizer.get_resource_utilization()
        
        # Get performance data
        try:
            perf_stats = self.performance_monitor.get_performance_stats()
        except Exception:
            perf_stats = {}
        
        current_time = datetime.now()
        
        # Process each resource type
        for resource_type in ResourceType:
            # Find pools of this type
            type_pools = [data for pool_id, data in utilization_data.items() 
                         if data.get('resource_type') == resource_type.value]
            
            if type_pools:
                # Aggregate metrics across pools
                current_util = statistics.mean([pool['utilization_ratio'] for pool in type_pools])
                
                # Calculate other metrics
                peak_util = max([pool['utilization_ratio'] for pool in type_pools])
                
                # Get historical data for trend calculation
                historical_utils = self._get_historical_utilization(resource_type)
                if len(historical_utils) > 1:
                    recent_utils = historical_utils[-10:]  # Last 10 observations
                    avg_util = statistics.mean(recent_utils)
                    
                    # Calculate trend
                    if len(recent_utils) >= 5:
                        older_avg = statistics.mean(recent_utils[:5])
                        newer_avg = statistics.mean(recent_utils[-5:])
                        trend = (newer_avg - older_avg) / max(older_avg, 0.1)
                    else:
                        trend = 0.0
                else:
                    avg_util = current_util
                    trend = 0.0
                
                # Calculate efficiency and cost metrics
                efficiency = self._calculate_resource_efficiency(resource_type, current_util)
                cost_per_hour = self._calculate_cost_per_hour(resource_type)
                performance_impact = self._calculate_performance_impact(resource_type, current_util)
                
                # Create metrics object
                metrics = ResourceMetrics(
                    resource_type=resource_type,
                    current_utilization=current_util,
                    peak_utilization=peak_util,
                    average_utilization=avg_util,
                    trend=trend,
                    efficiency=efficiency,
                    cost_per_hour=cost_per_hour,
                    performance_impact=performance_impact,
                    timestamp=current_time
                )
                
                self.current_metrics[resource_type] = metrics
    
    def _get_historical_utilization(self, resource_type: ResourceType) -> List[float]:
        """Get historical utilization data for trend analysis."""
        # Get from predictor's historical data
        history = list(self.predictor.historical_data[resource_type])
        return [obs['utilization'] for obs in history[-50:]]  # Last 50 observations
    
    def _calculate_resource_efficiency(self, resource_type: ResourceType, utilization: float) -> float:
        """Calculate resource efficiency score."""
        # Efficiency is optimal around 70-80% utilization
        optimal_range = (0.7, 0.8)
        
        if optimal_range[0] <= utilization <= optimal_range[1]:
            return 1.0  # Perfect efficiency
        elif utilization < optimal_range[0]:
            # Under-utilized
            return utilization / optimal_range[0]
        else:
            # Over-utilized (decreasing returns)
            excess = utilization - optimal_range[1]
            return max(0.1, 1.0 - (excess * 2))  # Rapidly decreasing efficiency
    
    def _calculate_cost_per_hour(self, resource_type: ResourceType) -> float:
        """Calculate cost per hour for resource type."""
        # Get pools of this type
        type_pools = [pool for pool in self.resource_optimizer.resource_pools.values() 
                     if pool.resource_type == resource_type]
        
        if not type_pools:
            return 0.0
        
        # Calculate weighted average cost
        total_cost = 0.0
        total_capacity = 0.0
        
        for pool in type_pools:
            pool_cost = pool.cost_per_unit * pool.total_capacity
            total_cost += pool_cost
            total_capacity += pool.total_capacity
        
        return total_cost / max(total_capacity, 1.0)
    
    def _calculate_performance_impact(self, resource_type: ResourceType, utilization: float) -> float:
        """Calculate performance impact of current utilization."""
        # Performance degrades as utilization increases beyond 80%
        if utilization <= 0.8:
            return 0.0  # No performance impact
        else:
            # Linear degradation from 0.8 to 1.0 utilization
            excess = utilization - 0.8
            return min(1.0, excess / 0.2)  # 0 to 1 scale
    
    def _update_predictor(self):
        """Update predictor with current observations."""
        current_time = datetime.now()
        
        for resource_type, metrics in self.current_metrics.items():
            self.predictor.add_observation(
                resource_type=resource_type,
                utilization=metrics.current_utilization,
                timestamp=current_time
            )
    
    def _analyze_and_decide(self):
        """Analyze resource needs and make scaling decisions."""
        
        for resource_type, metrics in self.current_metrics.items():
            # Skip if we can't scale this resource type
            if not self.scaler.can_scale(resource_type, ScalingDirection.SCALE_UP):
                continue
            
            # Analyze current situation
            scaling_decision = self._analyze_resource_scaling_need(resource_type, metrics)
            
            if scaling_decision and scaling_decision.direction != ScalingDirection.MAINTAIN:
                # Coordinate with other agents for consensus if coordinator available
                if self.coordinator:
                    consensus_result = self._coordinate_scaling_consensus(scaling_decision)
                    if consensus_result and consensus_result.get('approved', False):
                        self.scaling_decisions.append(scaling_decision)
                else:
                    # No coordinator, make decision independently
                    self.scaling_decisions.append(scaling_decision)
                
                self.decisions_made += 1
    
    def _analyze_resource_scaling_need(self, resource_type: ResourceType, 
                                     metrics: ResourceMetrics) -> Optional[ScalingDecision]:
        """Analyze if scaling is needed for a resource type."""
        
        current_util = metrics.current_utilization
        trend = metrics.trend
        efficiency = metrics.efficiency
        
        # Determine if scaling is needed
        scaling_direction = ScalingDirection.MAINTAIN
        trigger = None
        confidence = 0.5
        reasoning = "No scaling needed"
        
        # Check for scale-up conditions
        if current_util > self.config.scale_up_threshold:
            scaling_direction = ScalingDirection.SCALE_UP
            trigger = ScalingTrigger.UTILIZATION_THRESHOLD
            confidence = min(0.9, current_util)
            reasoning = f"High utilization: {current_util:.1%} > {self.config.scale_up_threshold:.1%}"
            
        elif trend > 0.2 and current_util > 0.6:
            # Predictive scaling up
            if self.config.enable_predictive_scaling:
                prediction = self.predictor.predict_demand(resource_type)
                if prediction.predicted_demand > self.config.scale_up_threshold:
                    scaling_direction = ScalingDirection.SCALE_UP
                    trigger = ScalingTrigger.PREDICTIVE_DEMAND
                    confidence = prediction.confidence * 0.8  # Slightly lower confidence for predictions
                    reasoning = f"Predicted demand: {prediction.predicted_demand:.1%} with trend: {trend:.2f}"
        
        # Check for scale-down conditions
        elif current_util < self.config.scale_down_threshold and trend < -0.1:
            scaling_direction = ScalingDirection.SCALE_DOWN
            trigger = ScalingTrigger.COST_OPTIMIZATION
            confidence = 1.0 - current_util
            reasoning = f"Low utilization: {current_util:.1%} < {self.config.scale_down_threshold:.1%}"
        
        # Create scaling decision if needed
        if scaling_direction != ScalingDirection.MAINTAIN:
            # Calculate scaling factor
            if scaling_direction == ScalingDirection.SCALE_UP:
                # Scale up based on how much over threshold we are
                excess = current_util - self.config.scale_up_threshold
                scaling_factor = max(self.config.min_scaling_factor, 
                                   min(self.config.max_scaling_factor, 1.0 + excess * 2))
            else:
                # Scale down conservatively
                under_usage = self.config.scale_down_threshold - current_util
                scaling_factor = max(1.0 / self.config.max_scaling_factor,
                                   1.0 - under_usage)
            
            # Get current capacity
            current_capacity = sum(pool.total_capacity for pool in self.resource_optimizer.resource_pools.values() 
                                 if pool.resource_type == resource_type)
            
            target_capacity = current_capacity * scaling_factor
            
            # Calculate impacts
            cost_impact = self._estimate_cost_impact(resource_type, scaling_factor)
            performance_impact = self._estimate_performance_impact(resource_type, scaling_factor)
            
            # Calculate execution priority
            priority = self._calculate_execution_priority(metrics, confidence, cost_impact, performance_impact)
            
            return ScalingDecision(
                decision_id=f"scaling_{resource_type.value}_{int(datetime.now().timestamp())}",
                resource_type=resource_type,
                direction=scaling_direction,
                trigger=trigger,
                current_capacity=current_capacity,
                target_capacity=target_capacity,
                scaling_factor=scaling_factor,
                confidence=confidence,
                reasoning=reasoning,
                estimated_cost_impact=cost_impact,
                estimated_performance_impact=performance_impact,
                execution_priority=priority
            )
        
        return None
    
    def _estimate_cost_impact(self, resource_type: ResourceType, scaling_factor: float) -> float:
        """Estimate cost impact of scaling."""
        current_cost = self._calculate_cost_per_hour(resource_type)
        new_cost = current_cost * scaling_factor
        return new_cost - current_cost
    
    def _estimate_performance_impact(self, resource_type: ResourceType, scaling_factor: float) -> float:
        """Estimate performance impact of scaling."""
        if scaling_factor > 1.0:
            # Scaling up should improve performance
            return (scaling_factor - 1.0) * 0.5  # 50% of scaling translates to performance improvement
        else:
            # Scaling down might hurt performance
            return -(1.0 - scaling_factor) * 0.3  # 30% of scaling reduction as performance impact
    
    def _calculate_execution_priority(self, metrics: ResourceMetrics, confidence: float,
                                    cost_impact: float, performance_impact: float) -> float:
        """Calculate execution priority for scaling decision."""
        
        # Factors: urgency (utilization), confidence, performance impact, cost efficiency
        urgency_factor = abs(metrics.current_utilization - 0.7)  # Distance from optimal
        confidence_factor = confidence
        performance_factor = max(0, performance_impact)  # Only positive impacts count
        cost_factor = max(0, -cost_impact / max(metrics.cost_per_hour, 1.0))  # Cost savings as positive
        
        priority = (urgency_factor * 0.3 + confidence_factor * 0.3 + 
                   performance_factor * 0.2 + cost_factor * 0.2)
        
        return round(priority, 3)
    
    def _coordinate_scaling_consensus(self, decision: ScalingDecision) -> Optional[Dict[str, Any]]:
        """Coordinate with other agents for scaling consensus."""
        
        try:
            # Create coordination task
            task_id = self.coordinator.create_coordination_task(
                description=f"Resource scaling: {decision.direction.value} {decision.resource_type.value}",
                required_roles={AgentRole.PERFORMANCE_OPTIMIZER, AgentRole.QUALITY_ASSESSOR},
                context={
                    'resource_type': decision.resource_type.value,
                    'direction': decision.direction.value,
                    'scaling_factor': decision.scaling_factor,
                    'cost_impact': decision.estimated_cost_impact,
                    'performance_impact': decision.estimated_performance_impact,
                    'confidence': decision.confidence
                }
            )
            
            # Submit scaling assessment vote
            self.coordinator.submit_vote(
                task_id=task_id,
                agent_id="adaptive_resource_management_agent",
                choice=decision.execution_priority,
                confidence=decision.confidence,
                reasoning=f"Resource scaling decision: {decision.reasoning}"
            )
            
            print(f"   Consensus requested for {decision.resource_type.value} scaling")
            
            # Wait for consensus (in real implementation, this would be event-driven)
            time.sleep(1)
            
            result = self.coordinator.get_coordination_result(task_id)
            
            if result and hasattr(result, 'decision'):
                # Interpret consensus result
                approval_threshold = 0.6
                approved = result.decision >= approval_threshold
                
                return {
                    'approved': approved,
                    'consensus_score': result.decision,
                    'confidence': result.confidence
                }
            
            return None
            
        except Exception as e:
            print(f"Scaling consensus coordination failed: {e}")
            return None
    
    def _execute_scaling_decisions(self):
        """Execute approved scaling decisions."""
        
        if not self.scaling_decisions:
            return
        
        # Sort by priority
        self.scaling_decisions.sort(key=lambda d: d.execution_priority, reverse=True)
        
        # Execute top priority decisions
        executed_count = 0
        for decision in self.scaling_decisions[:3]:  # Execute top 3 decisions
            if not decision.executed:
                result = self.scaler.execute_scaling(decision)
                
                if result.get('success', False):
                    self.successful_scalings += 1
                    
                    # Update statistics
                    if decision.estimated_cost_impact < 0:
                        self.cost_savings += abs(decision.estimated_cost_impact)
                    if decision.estimated_performance_impact > 0:
                        self.performance_improvements += decision.estimated_performance_impact
                
                executed_count += 1
        
        # Clear executed decisions
        self.scaling_decisions = [d for d in self.scaling_decisions if not d.executed]
        
        if executed_count > 0:
            print(f"Executed {executed_count} scaling decisions")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource management status."""
        
        # Calculate overall resource efficiency
        if self.current_metrics:
            avg_efficiency = statistics.mean([m.efficiency for m in self.current_metrics.values()])
            avg_utilization = statistics.mean([m.current_utilization for m in self.current_metrics.values()])
        else:
            avg_efficiency = 0.0
            avg_utilization = 0.0
        
        return {
            'management_active': self.monitoring_active,
            'decisions_made': self.decisions_made,
            'successful_scalings': self.successful_scalings,
            'success_rate': (self.successful_scalings / max(self.decisions_made, 1)) * 100,
            'cost_savings': self.cost_savings,
            'performance_improvements': self.performance_improvements,
            'avg_resource_efficiency': avg_efficiency,
            'avg_resource_utilization': avg_utilization,
            'pending_decisions': len(self.scaling_decisions),
            'resource_metrics': {
                rt.value: {
                    'utilization': metrics.current_utilization,
                    'efficiency': metrics.efficiency,
                    'trend': metrics.trend
                } for rt, metrics in self.current_metrics.items()
            }
        }
    
    def generate_resource_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive resource optimization report."""
        
        # Get current status
        status = self.get_resource_status()
        
        # Get predictions for each resource type
        predictions = {}
        for resource_type in ResourceType:
            prediction = self.predictor.predict_demand(resource_type)
            predictions[resource_type.value] = {
                'predicted_demand': prediction.predicted_demand,
                'confidence': prediction.confidence,
                'factors': prediction.factors,
                'horizon_minutes': prediction.prediction_horizon_minutes
            }
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations()
        
        return {
            'report_generated': datetime.now().isoformat(),
            'current_status': status,
            'resource_predictions': predictions,
            'scaling_history': [
                {
                    'resource_type': d.resource_type.value,
                    'direction': d.direction.value,
                    'scaling_factor': d.scaling_factor,
                    'success': d.executed and d.execution_results.get('success', False),
                    'timestamp': d.created_at.isoformat()
                } for d in self.scaler.scaling_history[-10:]  # Last 10 scaling operations
            ],
            'recommendations': recommendations,
            'configuration': {
                'strategy': self.config.default_strategy.value,
                'scale_up_threshold': self.config.scale_up_threshold,
                'scale_down_threshold': self.config.scale_down_threshold,
                'predictive_scaling': self.config.enable_predictive_scaling,
                'cost_optimization': self.config.enable_cost_optimization
            }
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        
        recommendations = []
        
        # Check resource efficiency
        if self.current_metrics:
            low_efficiency_resources = [
                rt.value for rt, metrics in self.current_metrics.items() 
                if metrics.efficiency < 0.6
            ]
            if low_efficiency_resources:
                recommendations.append(f"Optimize resource efficiency for: {', '.join(low_efficiency_resources)}")
        
        # Check utilization patterns
        if self.current_metrics:
            over_utilized = [
                rt.value for rt, metrics in self.current_metrics.items() 
                if metrics.current_utilization > 0.9
            ]
            under_utilized = [
                rt.value for rt, metrics in self.current_metrics.items() 
                if metrics.current_utilization < 0.3
            ]
            
            if over_utilized:
                recommendations.append(f"Consider scaling up: {', '.join(over_utilized)}")
            if under_utilized:
                recommendations.append(f"Consider scaling down: {', '.join(under_utilized)}")
        
        # Cost optimization recommendations
        if self.config.enable_cost_optimization and self.cost_savings > 0:
            recommendations.append(f"Continue cost optimization - saved ${self.cost_savings:.2f}")
        
        # Predictive scaling recommendations
        if self.config.enable_predictive_scaling:
            recommendations.append("Predictive scaling active - monitoring demand patterns")
        else:
            recommendations.append("Enable predictive scaling for proactive resource management")
        
        if not recommendations:
            recommendations.append("Resource management operating optimally")
        
        return recommendations


def test_adaptive_resource_management():
    """Test the adaptive resource management agent."""
    print("\n" + "="*60)
    print("Testing Adaptive Resource Management Agent")
    print("="*60)
    
    # Create test configuration
    config = AdaptiveConfiguration(
        scale_up_threshold=0.75,
        scale_down_threshold=0.25,
        prediction_window_hours=1,
        enable_predictive_scaling=True,
        enable_cost_optimization=True
    )
    
    # Create agent
    agent = AdaptiveResourceManagementAgent(config=config)
    
    # Test resource metrics collection
    print("\n1. Testing resource metrics collection...")
    agent._collect_resource_metrics()
    
    print(f"   Resource metrics collected: {len(agent.current_metrics)}")
    for rt, metrics in agent.current_metrics.items():
        print(f"     {rt.value}: {metrics.current_utilization:.1%} utilization, {metrics.efficiency:.2f} efficiency")
    
    # Test prediction
    print("\n2. Testing resource prediction...")
    for resource_type in [ResourceType.CPU, ResourceType.MEMORY]:
        # Add some sample observations
        for i in range(20):
            utilization = 0.5 + (i * 0.02)  # Increasing pattern
            agent.predictor.add_observation(resource_type, utilization)
        
        prediction = agent.predictor.predict_demand(resource_type)
        print(f"   {resource_type.value} prediction: {prediction.predicted_demand:.1%} (confidence: {prediction.confidence:.2f})")
    
    # Test scaling decision analysis
    print("\n3. Testing scaling decision analysis...")
    agent._analyze_and_decide()
    
    print(f"   Scaling decisions made: {len(agent.scaling_decisions)}")
    for decision in agent.scaling_decisions:
        print(f"     {decision.resource_type.value}: {decision.direction.value} by {decision.scaling_factor:.2f}x")
        print(f"       Reasoning: {decision.reasoning}")
    
    # Test scaling execution
    if agent.scaling_decisions:
        print("\n4. Testing scaling execution...")
        decision = agent.scaling_decisions[0]
        result = agent.scaler.execute_scaling(decision)
        
        print(f"   Scaling execution: {result.get('success', False)}")
        if result.get('success'):
            print(f"     Action: {result.get('action')}")
            print(f"     Pools scaled: {result.get('pools_scaled', 0)}")
    
    # Test status and reporting
    print("\n5. Testing status and reporting...")
    status = agent.get_resource_status()
    print(f"   Management active: {status['management_active']}")
    print(f"   Decisions made: {status['decisions_made']}")
    print(f"   Average efficiency: {status['avg_resource_efficiency']:.2f}")
    
    # Generate optimization report
    report = agent.generate_resource_optimization_report()
    print(f"\n6. Optimization report generated:")
    print(f"   Predictions available: {len(report['resource_predictions'])}")
    print(f"   Recommendations: {len(report['recommendations'])}")
    for i, rec in enumerate(report['recommendations'][:3], 1):
        print(f"     {i}. {rec}")
    
    print("\nAdaptive Resource Management Agent test completed successfully!")
    return True


if __name__ == "__main__":
    test_adaptive_resource_management()