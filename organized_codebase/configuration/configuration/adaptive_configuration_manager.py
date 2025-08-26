"""
Adaptive Configuration Manager - TestMaster Advanced ML
ML-driven configuration optimization and dynamic system tuning
Enterprise ML Module #4/8 for comprehensive system intelligence
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Event, Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import copy
import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


class ConfigScope(Enum):
    SYSTEM = "system"
    APPLICATION = "application"
    COMPONENT = "component"
    USER = "user"
    ENVIRONMENT = "environment"


class ConfigDataType(Enum):
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    LIST = "list"
    DICT = "dict"


class OptimizationObjective(Enum):
    PERFORMANCE = "performance"
    RESOURCE_USAGE = "resource_usage"
    RELIABILITY = "reliability"
    COST = "cost"
    ENERGY = "energy"
    USER_EXPERIENCE = "user_experience"


@dataclass
class ConfigParameter:
    """ML-enhanced configuration parameter"""
    
    param_id: str
    name: str
    scope: ConfigScope
    data_type: ConfigDataType
    current_value: Any
    default_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    allowed_values: Optional[List[Any]] = None
    
    # ML Enhancement Fields
    optimal_value: Any = None
    predicted_performance_impact: float = 0.0
    optimization_confidence: float = 0.0
    sensitivity_score: float = 0.0
    interaction_factors: Dict[str, float] = field(default_factory=dict)
    
    # Change tracking
    last_changed: datetime = field(default_factory=datetime.now)
    change_history: List[Tuple[datetime, Any, float]] = field(default_factory=list)
    adaptation_count: int = 0
    
    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ConfigurationProfile:
    """ML-optimized configuration profile"""
    
    profile_id: str
    profile_name: str
    description: str
    scope: ConfigScope
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    performance_score: float = 0.0
    resource_efficiency: float = 0.0
    reliability_score: float = 0.0
    
    # ML Enhancement
    ml_optimized: bool = False
    optimization_iteration: int = 0
    fitness_score: float = 0.0
    adaptation_rate: float = 0.1
    
    # Usage tracking
    activation_count: int = 0
    total_runtime: float = 0.0
    last_used: Optional[datetime] = None
    success_rate: float = 1.0


@dataclass
class OptimizationExperiment:
    """ML-driven configuration optimization experiment"""
    
    experiment_id: str
    objective: OptimizationObjective
    target_parameters: List[str]
    baseline_config: Dict[str, Any]
    
    # Experiment design
    parameter_space: Dict[str, Dict[str, Any]]
    max_iterations: int = 100
    convergence_threshold: float = 0.01
    
    # Results tracking
    current_iteration: int = 0
    best_config: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    iteration_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # ML insights
    parameter_importance: Dict[str, float] = field(default_factory=dict)
    interaction_effects: Dict[str, float] = field(default_factory=dict)
    convergence_rate: float = 0.0


class AdaptiveConfigurationManager:
    """
    ML-enhanced configuration management with dynamic optimization
    """
    
    def __init__(self,
                 enable_ml_optimization: bool = True,
                 optimization_interval: int = 300,
                 auto_apply_optimizations: bool = False,
                 learning_rate: float = 0.1):
        """Initialize adaptive configuration manager"""
        
        self.enable_ml_optimization = enable_ml_optimization
        self.optimization_interval = optimization_interval
        self.auto_apply_optimizations = auto_apply_optimizations
        self.learning_rate = learning_rate
        
        # ML Models for Configuration Intelligence
        self.performance_predictor: Optional[RandomForestRegressor] = None
        self.resource_optimizer: Optional[GradientBoostingRegressor] = None
        self.parameter_importance_model: Optional[Ridge] = None
        self.config_clusterer: Optional[KMeans] = None
        
        # ML Feature Processing
        self.feature_scaler = StandardScaler()
        self.performance_scaler = MinMaxScaler()
        self.config_feature_history: deque = deque(maxlen=2000)
        
        # Configuration Management
        self.config_parameters: Dict[str, ConfigParameter] = {}
        self.configuration_profiles: Dict[str, ConfigurationProfile] = {}
        self.active_experiments: Dict[str, OptimizationExperiment] = {}
        
        # Performance Tracking
        self.performance_metrics: deque = deque(maxlen=1000)
        self.configuration_snapshots: deque = deque(maxlen=500)
        self.optimization_results: List[Dict[str, Any]] = []
        
        # ML Insights
        self.ml_recommendations: Dict[str, Dict[str, Any]] = {}
        self.parameter_interactions: Dict[str, Dict[str, float]] = {}
        self.optimization_insights: List[Dict[str, Any]] = []
        
        # Change Management
        self.pending_changes: Dict[str, Dict[str, Any]] = {}
        self.change_approval_required: bool = True
        self.rollback_history: deque = deque(maxlen=100)
        
        # Configuration
        self.max_optimization_iterations = 200
        self.convergence_patience = 10
        self.safety_bounds_multiplier = 0.1
        self.performance_improvement_threshold = 0.05
        
        # Statistics
        self.config_stats = {
            'parameters_managed': 0,
            'optimizations_performed': 0,
            'configurations_applied': 0,
            'performance_improvements': 0,
            'ml_recommendations_generated': 0,
            'rollbacks_performed': 0,
            'start_time': datetime.now()
        }
        
        # Synchronization
        self.config_lock = RLock()
        self.ml_lock = Lock()
        self.shutdown_event = Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models and start optimization loop
        if enable_ml_optimization:
            self._initialize_ml_models()
            asyncio.create_task(self._ml_optimization_loop())
        
        asyncio.create_task(self._configuration_monitoring_loop())
    
    def _initialize_ml_models(self):
        """Initialize ML models for configuration intelligence"""
        
        try:
            # Performance prediction model
            self.performance_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=42,
                min_samples_split=5
            )
            
            # Resource optimization model
            self.resource_optimizer = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
            
            # Parameter importance analysis
            self.parameter_importance_model = Ridge(
                alpha=1.0,
                random_state=42
            )
            
            # Configuration clustering
            self.config_clusterer = KMeans(
                n_clusters=5,
                random_state=42,
                n_init=10
            )
            
            self.logger.info("Configuration ML models initialized")
            
        except Exception as e:
            self.logger.error(f"Configuration ML model initialization failed: {e}")
            self.enable_ml_optimization = False
    
    def register_parameter(self,
                          param_id: str,
                          name: str,
                          scope: ConfigScope,
                          data_type: ConfigDataType,
                          current_value: Any,
                          default_value: Any,
                          min_value: Any = None,
                          max_value: Any = None,
                          allowed_values: List[Any] = None,
                          description: str = "",
                          tags: List[str] = None) -> bool:
        """Register configuration parameter for ML optimization"""
        
        try:
            with self.config_lock:
                parameter = ConfigParameter(
                    param_id=param_id,
                    name=name,
                    scope=scope,
                    data_type=data_type,
                    current_value=current_value,
                    default_value=default_value,
                    min_value=min_value,
                    max_value=max_value,
                    allowed_values=allowed_values,
                    description=description,
                    tags=tags or []
                )
                
                self.config_parameters[param_id] = parameter
                self.config_stats['parameters_managed'] += 1
            
            self.logger.info(f"Configuration parameter registered: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter registration failed for {param_id}: {e}")
            return False
    
    def update_parameter(self,
                        param_id: str,
                        new_value: Any,
                        performance_impact: float = None) -> bool:
        """Update configuration parameter with performance tracking"""
        
        try:
            with self.config_lock:
                if param_id not in self.config_parameters:
                    self.logger.error(f"Parameter not found: {param_id}")
                    return False
                
                parameter = self.config_parameters[param_id]
                old_value = parameter.current_value
                
                # Validate new value
                if not self._validate_parameter_value(parameter, new_value):
                    self.logger.error(f"Invalid value for parameter {param_id}: {new_value}")
                    return False
                
                # Update parameter
                parameter.current_value = new_value
                parameter.last_changed = datetime.now()
                parameter.adaptation_count += 1
                
                # Record change history
                impact_score = performance_impact or 0.0
                parameter.change_history.append((datetime.now(), old_value, impact_score))
                
                # Keep history limited
                if len(parameter.change_history) > 100:
                    parameter.change_history = parameter.change_history[-50:]
                
                # ML enhancement if enabled
                if self.enable_ml_optimization:
                    asyncio.create_task(self._analyze_parameter_change(parameter, old_value, new_value))
            
            self.logger.info(f"Parameter updated: {param_id} = {new_value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter update failed for {param_id}: {e}")
            return False
    
    def create_configuration_profile(self,
                                   profile_name: str,
                                   scope: ConfigScope,
                                   parameter_values: Dict[str, Any],
                                   description: str = "") -> str:
        """Create optimized configuration profile"""
        
        try:
            profile_id = f"profile_{int(time.time())}_{hash(profile_name) % 1000}"
            
            with self.config_lock:
                profile = ConfigurationProfile(
                    profile_id=profile_id,
                    profile_name=profile_name,
                    description=description,
                    scope=scope,
                    parameters=parameter_values.copy()
                )
                
                self.configuration_profiles[profile_id] = profile
            
            # ML optimization if enabled
            if self.enable_ml_optimization:
                asyncio.create_task(self._optimize_configuration_profile(profile_id))
            
            self.logger.info(f"Configuration profile created: {profile_name}")
            return profile_id
            
        except Exception as e:
            self.logger.error(f"Configuration profile creation failed: {e}")
            return ""
    
    def apply_configuration_profile(self, profile_id: str) -> bool:
        """Apply configuration profile with rollback capability"""
        
        try:
            with self.config_lock:
                if profile_id not in self.configuration_profiles:
                    self.logger.error(f"Configuration profile not found: {profile_id}")
                    return False
                
                profile = self.configuration_profiles[profile_id]
                
                # Create rollback snapshot
                rollback_snapshot = self._create_configuration_snapshot()
                
                # Apply configuration changes
                applied_changes = {}
                for param_id, new_value in profile.parameters.items():
                    if param_id in self.config_parameters:
                        old_value = self.config_parameters[param_id].current_value
                        if self.update_parameter(param_id, new_value):
                            applied_changes[param_id] = {'old': old_value, 'new': new_value}
                
                if applied_changes:
                    # Store rollback information
                    self.rollback_history.append({
                        'timestamp': datetime.now(),
                        'profile_id': profile_id,
                        'changes': applied_changes,
                        'snapshot': rollback_snapshot
                    })
                    
                    # Update profile usage statistics
                    profile.activation_count += 1
                    profile.last_used = datetime.now()
                    
                    self.config_stats['configurations_applied'] += 1
                    
                    self.logger.info(f"Configuration profile applied: {profile.profile_name}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Configuration profile application failed: {e}")
            return False
    
    async def start_optimization_experiment(self,
                                          objective: OptimizationObjective,
                                          target_parameters: List[str],
                                          max_iterations: int = 100) -> str:
        """Start ML-driven optimization experiment"""
        
        try:
            if not self.enable_ml_optimization:
                self.logger.error("ML optimization is disabled")
                return ""
            
            experiment_id = f"exp_{int(time.time())}_{hash(str(target_parameters)) % 1000}"
            
            # Validate target parameters
            valid_params = [p for p in target_parameters if p in self.config_parameters]
            if not valid_params:
                self.logger.error("No valid target parameters for optimization")
                return ""
            
            # Create baseline configuration
            baseline_config = {
                param_id: self.config_parameters[param_id].current_value
                for param_id in valid_params
            }
            
            # Define parameter space for optimization
            parameter_space = {}
            for param_id in valid_params:
                param = self.config_parameters[param_id]
                if param.data_type == ConfigDataType.INTEGER:
                    parameter_space[param_id] = {
                        'type': 'integer',
                        'min': param.min_value or 1,
                        'max': param.max_value or 1000
                    }
                elif param.data_type == ConfigDataType.FLOAT:
                    parameter_space[param_id] = {
                        'type': 'float',
                        'min': param.min_value or 0.0,
                        'max': param.max_value or 10.0
                    }
                elif param.data_type == ConfigDataType.BOOLEAN:
                    parameter_space[param_id] = {
                        'type': 'boolean',
                        'values': [True, False]
                    }
                elif param.allowed_values:
                    parameter_space[param_id] = {
                        'type': 'categorical',
                        'values': param.allowed_values
                    }
            
            # Create experiment
            experiment = OptimizationExperiment(
                experiment_id=experiment_id,
                objective=objective,
                target_parameters=valid_params,
                baseline_config=baseline_config,
                parameter_space=parameter_space,
                max_iterations=max_iterations
            )
            
            with self.config_lock:
                self.active_experiments[experiment_id] = experiment
            
            # Start optimization process
            asyncio.create_task(self._run_optimization_experiment(experiment_id))
            
            self.logger.info(f"Optimization experiment started: {experiment_id}")
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"Optimization experiment creation failed: {e}")
            return ""
    
    async def _run_optimization_experiment(self, experiment_id: str):
        """Run ML-driven optimization experiment"""
        
        try:
            with self.config_lock:
                if experiment_id not in self.active_experiments:
                    return
                
                experiment = self.active_experiments[experiment_id]
            
            self.logger.info(f"Starting optimization experiment: {experiment_id}")
            
            # Initialize with baseline
            current_config = experiment.baseline_config.copy()
            current_score = await self._evaluate_configuration(current_config, experiment.objective)
            
            experiment.best_config = current_config.copy()
            experiment.best_score = current_score
            
            # Optimization loop
            for iteration in range(experiment.max_iterations):
                if self.shutdown_event.is_set():
                    break
                
                # Generate candidate configuration
                candidate_config = await self._generate_candidate_configuration(
                    experiment, current_config
                )
                
                # Evaluate candidate
                candidate_score = await self._evaluate_configuration(
                    candidate_config, experiment.objective
                )
                
                # Update if better
                improvement = candidate_score - current_score
                if improvement > 0:
                    current_config = candidate_config.copy()
                    current_score = candidate_score
                    
                    # Update best if significant improvement
                    if candidate_score > experiment.best_score + experiment.convergence_threshold:
                        experiment.best_config = candidate_config.copy()
                        experiment.best_score = candidate_score
                        
                        self.logger.info(f"Optimization improvement: {improvement:.4f}")
                
                # Record iteration
                experiment.iteration_history.append({
                    'iteration': iteration,
                    'config': candidate_config.copy(),
                    'score': candidate_score,
                    'improvement': improvement
                })
                
                experiment.current_iteration = iteration + 1
                
                # Check convergence
                if len(experiment.iteration_history) >= self.convergence_patience:
                    recent_improvements = [
                        h['improvement'] for h in experiment.iteration_history[-self.convergence_patience:]
                    ]
                    
                    if max(recent_improvements) < experiment.convergence_threshold:
                        self.logger.info(f"Optimization converged at iteration {iteration}")
                        break
                
                # Brief pause between iterations
                await asyncio.sleep(1)
            
            # Finalize experiment
            await self._finalize_optimization_experiment(experiment_id)
            
        except Exception as e:
            self.logger.error(f"Optimization experiment failed: {e}")
    
    async def _configuration_monitoring_loop(self):
        """Configuration monitoring and adaptation loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Monitor parameter performance
                await self._monitor_parameter_performance()
                
                # Update configuration snapshots
                await self._update_configuration_snapshots()
                
                # Clean up old experiments
                await self._cleanup_completed_experiments()
                
            except Exception as e:
                self.logger.error(f"Configuration monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _ml_optimization_loop(self):
        """ML optimization and insights generation loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.optimization_interval)
                
                if len(self.config_feature_history) >= 50:
                    # Update ML models
                    await self._update_ml_models()
                    
                    # Generate recommendations
                    await self._generate_ml_recommendations()
                    
                    # Analyze parameter interactions
                    await self._analyze_parameter_interactions()
                    
                    # Generate optimization insights
                    await self._generate_optimization_insights()
                
            except Exception as e:
                self.logger.error(f"ML optimization loop error: {e}")
                await asyncio.sleep(30)
    
    def get_configuration_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive configuration management dashboard"""
        
        # Parameter summary
        param_summary = {}
        for param_id, param in self.config_parameters.items():
            param_summary[param_id] = {
                'name': param.name,
                'scope': param.scope.value,
                'current_value': param.current_value,
                'optimal_value': param.optimal_value,
                'optimization_confidence': param.optimization_confidence,
                'sensitivity_score': param.sensitivity_score,
                'adaptation_count': param.adaptation_count,
                'last_changed': param.last_changed.isoformat() if param.last_changed else None
            }
        
        # Active experiments
        experiment_summary = {}
        for exp_id, exp in self.active_experiments.items():
            experiment_summary[exp_id] = {
                'objective': exp.objective.value,
                'target_parameters': exp.target_parameters,
                'current_iteration': exp.current_iteration,
                'best_score': exp.best_score,
                'convergence_rate': exp.convergence_rate
            }
        
        # ML insights
        ml_status = {
            'ml_optimization_enabled': self.enable_ml_optimization,
            'feature_history_size': len(self.config_feature_history),
            'active_experiments': len(self.active_experiments),
            'ml_recommendations': len(self.ml_recommendations),
            'optimization_insights': len(self.optimization_insights)
        }
        
        return {
            'configuration_overview': {
                'total_parameters': len(self.config_parameters),
                'configuration_profiles': len(self.configuration_profiles),
                'active_experiments': len(self.active_experiments),
                'optimization_success_rate': self._calculate_optimization_success_rate()
            },
            'parameters': param_summary,
            'active_experiments': experiment_summary,
            'statistics': self.config_stats.copy(),
            'ml_status': ml_status,
            'recent_insights': self.optimization_insights[-5:] if self.optimization_insights else []
        }
    
    def _calculate_optimization_success_rate(self) -> float:
        """Calculate optimization success rate"""
        
        if not self.optimization_results:
            return 0.0
        
        successful_optimizations = sum(
            1 for result in self.optimization_results
            if result.get('improvement', 0) > self.performance_improvement_threshold
        )
        
        return successful_optimizations / len(self.optimization_results)
    
    async def shutdown(self):
        """Graceful shutdown of configuration manager"""
        
        self.logger.info("Shutting down adaptive configuration manager...")
        
        # Stop active experiments
        for exp_id in list(self.active_experiments.keys()):
            experiment = self.active_experiments[exp_id]
            await self._finalize_optimization_experiment(exp_id)
        
        self.shutdown_event.set()
        await asyncio.sleep(1)
        self.logger.info("Adaptive configuration manager shutdown complete")