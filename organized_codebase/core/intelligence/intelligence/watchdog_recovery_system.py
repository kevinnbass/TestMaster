"""
Watchdog Recovery System (Part 2/3) - TestMaster Advanced ML
Advanced recovery and restart management with ML-driven optimization
Extracted from analytics_watchdog.py (674 lines) → 3 coordinated ML modules
"""

import asyncio
import logging
import subprocess
import signal
import time
import psutil
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Event, Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Union
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .watchdog_ml_monitor import (
    WatchdogAction, ComponentState, MLComponentHealth, 
    MLWatchdogRule, AdvancedWatchdogMLMonitor
)


@dataclass
class RecoveryStrategy:
    """ML-optimized recovery strategy"""
    
    strategy_id: str
    component_name: str
    failure_pattern: str
    recovery_actions: List[WatchdogAction]
    success_probability: float
    execution_time_estimate: int  # seconds
    resource_requirements: Dict[str, float]
    dependencies: List[str] = field(default_factory=list)
    
    # ML Enhancement
    ml_optimized: bool = True
    learning_enabled: bool = True
    adaptation_rate: float = 0.1


@dataclass  
class RecoveryAttempt:
    """Record of recovery attempt with ML analysis"""
    
    attempt_id: str
    component_name: str
    timestamp: datetime
    strategy_used: str
    actions_taken: List[WatchdogAction]
    success: bool
    execution_time: float
    resource_usage: Dict[str, float]
    failure_reason: Optional[str] = None
    
    # ML Enhancement
    predicted_success_rate: float = 0.0
    actual_outcome_score: float = 0.0
    learning_feedback: Dict[str, Any] = field(default_factory=dict)


class AdvancedRecoverySystem:
    """
    ML-enhanced recovery and restart management system
    Part 2/3 of the complete watchdog system
    """
    
    def __init__(self,
                 ml_monitor: AdvancedWatchdogMLMonitor,
                 max_concurrent_recoveries: int = 3,
                 recovery_timeout: int = 300,
                 enable_ml_optimization: bool = True):
        """Initialize advanced recovery system"""
        
        self.ml_monitor = ml_monitor
        self.max_concurrent_recoveries = max_concurrent_recoveries
        self.recovery_timeout = recovery_timeout
        self.enable_ml_optimization = enable_ml_optimization
        
        # ML Models for Recovery Optimization
        self.recovery_classifier: Optional[RandomForestClassifier] = None
        self.recovery_time_predictor: Optional[GradientBoostingRegressor] = None
        self.strategy_clusterer: Optional[KMeans] = None
        
        # ML Feature Processing
        self.feature_scaler = StandardScaler()
        self.recovery_feature_history: deque = deque(maxlen=500)
        
        # Recovery State Management
        self.recovery_strategies: Dict[str, List[RecoveryStrategy]] = {}
        self.active_recoveries: Dict[str, RecoveryAttempt] = {}
        self.recovery_history: deque = deque(maxlen=1000)
        self.recovery_queue: deque = deque()
        
        # Process Management
        self.monitored_processes: Dict[str, Dict[str, Any]] = {}
        self.process_restart_counts: defaultdict = defaultdict(int)
        
        # Custom Recovery Actions
        self.custom_recovery_actions: Dict[str, Callable] = {}
        self.recovery_callbacks: List[Callable] = []
        
        # Configuration
        self.restart_cooldown = 60
        self.max_restart_attempts = 5
        self.cascade_prevention_enabled = True
        
        # Statistics
        self.recovery_stats = {
            'total_recoveries_attempted': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'ml_optimized_recoveries': 0,
            'average_recovery_time': 0.0,
            'cascade_failures_prevented': 0,
            'start_time': datetime.now()
        }
        
        # Synchronization
        self.recovery_lock = RLock()
        self.process_lock = Lock()
        self.shutdown_event = Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models and start recovery manager
        if enable_ml_optimization:
            self._initialize_ml_models()
        
        self._setup_default_strategies()
        asyncio.create_task(self._recovery_manager_loop())
    
    def _initialize_ml_models(self):
        """Initialize ML models for recovery optimization"""
        
        try:
            # Recovery success prediction
            self.recovery_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                random_state=42,
                class_weight='balanced'
            )
            
            # Recovery time prediction
            self.recovery_time_predictor = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                learning_rate=0.1
            )
            
            # Recovery strategy clustering
            self.strategy_clusterer = KMeans(
                n_clusters=5,
                random_state=42,
                n_init=10
            )
            
            self.logger.info("Recovery ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Recovery ML model initialization failed: {e}")
            self.enable_ml_optimization = False
    
    def _setup_default_strategies(self):
        """Setup default recovery strategies"""
        
        # Standard component restart strategy
        standard_restart = RecoveryStrategy(
            strategy_id="standard_restart",
            component_name="*",  # Applies to all components
            failure_pattern="consecutive_failures",
            recovery_actions=[WatchdogAction.RESTART_COMPONENT],
            success_probability=0.7,
            execution_time_estimate=30,
            resource_requirements={'cpu': 0.1, 'memory': 0.05}
        )
        
        # Graceful restart with preparation
        graceful_restart = RecoveryStrategy(
            strategy_id="graceful_restart", 
            component_name="*",
            failure_pattern="degraded_performance",
            recovery_actions=[WatchdogAction.GRACEFUL_SHUTDOWN, WatchdogAction.RESTART_COMPONENT],
            success_probability=0.8,
            execution_time_estimate=60,
            resource_requirements={'cpu': 0.2, 'memory': 0.1}
        )
        
        # Force restart for critical failures
        force_restart = RecoveryStrategy(
            strategy_id="force_restart",
            component_name="*", 
            failure_pattern="critical_failure",
            recovery_actions=[WatchdogAction.FORCE_RESTART],
            success_probability=0.9,
            execution_time_estimate=20,
            resource_requirements={'cpu': 0.3, 'memory': 0.15}
        )
        
        # ML optimization strategy
        ml_optimize = RecoveryStrategy(
            strategy_id="ml_optimize",
            component_name="*",
            failure_pattern="predicted_failure",
            recovery_actions=[WatchdogAction.ML_OPTIMIZE, WatchdogAction.RESTART_COMPONENT],
            success_probability=0.85,
            execution_time_estimate=45,
            resource_requirements={'cpu': 0.15, 'memory': 0.08}
        )
        
        # Service-level restart
        service_restart = RecoveryStrategy(
            strategy_id="service_restart",
            component_name="*",
            failure_pattern="cascade_failure",
            recovery_actions=[WatchdogAction.RESTART_SERVICE],
            success_probability=0.9,
            execution_time_estimate=120,
            resource_requirements={'cpu': 0.5, 'memory': 0.3}
        )
        
        # Store default strategies
        self.recovery_strategies["default"] = [
            standard_restart, graceful_restart, force_restart, 
            ml_optimize, service_restart
        ]
        
        self.logger.info("Default recovery strategies initialized")
    
    async def initiate_recovery(self,
                               component_name: str,
                               failure_reason: str = "health_check_failure",
                               force: bool = False) -> bool:
        """Initiate ML-optimized recovery for component"""
        
        try:
            with self.recovery_lock:
                # Check if recovery is already in progress
                if component_name in self.active_recoveries and not force:
                    self.logger.warning(f"Recovery already in progress for {component_name}")
                    return False
                
                # Check concurrent recovery limit
                if len(self.active_recoveries) >= self.max_concurrent_recoveries and not force:
                    self.logger.warning("Max concurrent recoveries reached, queuing...")
                    await self._queue_recovery(component_name, failure_reason)
                    return True
                
                # Get component health information
                health = self.ml_monitor.component_health.get(component_name)
                if not health:
                    self.logger.error(f"Component not found for recovery: {component_name}")
                    return False
                
                # Select optimal recovery strategy
                strategy = await self._select_recovery_strategy(component_name, health, failure_reason)
                
                if not strategy:
                    self.logger.error(f"No suitable recovery strategy for {component_name}")
                    return False
                
                # Create recovery attempt record
                attempt = RecoveryAttempt(
                    attempt_id=f"recovery_{component_name}_{int(time.time())}",
                    component_name=component_name,
                    timestamp=datetime.now(),
                    strategy_used=strategy.strategy_id,
                    actions_taken=strategy.recovery_actions,
                    success=False,
                    execution_time=0.0,
                    resource_usage={}
                )
                
                # ML prediction for success rate
                if self.enable_ml_optimization:
                    attempt.predicted_success_rate = await self._predict_recovery_success(
                        component_name, strategy, health
                    )
                
                self.active_recoveries[component_name] = attempt
                self.recovery_stats['total_recoveries_attempted'] += 1
                
                # Execute recovery asynchronously
                asyncio.create_task(self._execute_recovery(component_name, strategy, attempt))
                
                return True
                
        except Exception as e:
            self.logger.error(f"Recovery initiation failed for {component_name}: {e}")
            return False
    
    async def _select_recovery_strategy(self,
                                       component_name: str,
                                       health: MLComponentHealth,
                                       failure_reason: str) -> Optional[RecoveryStrategy]:
        """Select optimal recovery strategy using ML analysis"""
        
        try:
            # Get available strategies
            strategies = self.recovery_strategies.get(component_name, []) + \
                        self.recovery_strategies.get("default", [])
            
            if not strategies:
                return None
            
            # ML-based strategy selection
            if self.enable_ml_optimization and len(self.recovery_feature_history) >= 20:
                optimal_strategy = await self._ml_select_strategy(strategies, health, failure_reason)
                if optimal_strategy:
                    return optimal_strategy
            
            # Rule-based fallback selection
            return await self._rule_based_strategy_selection(strategies, health, failure_reason)
            
        except Exception as e:
            self.logger.error(f"Strategy selection failed: {e}")
            return None
    
    async def _ml_select_strategy(self,
                                 strategies: List[RecoveryStrategy],
                                 health: MLComponentHealth,
                                 failure_reason: str) -> Optional[RecoveryStrategy]:
        """ML-based strategy selection"""
        
        try:
            best_strategy = None
            best_score = 0.0
            
            for strategy in strategies:
                # Extract features for ML prediction
                features = await self._extract_recovery_features(health, strategy, failure_reason)
                
                # Predict success probability
                if self.recovery_classifier:
                    success_prob = await self._predict_strategy_success(features)
                else:
                    success_prob = strategy.success_probability
                
                # Predict execution time
                if self.recovery_time_predictor:
                    exec_time = await self._predict_execution_time(features)
                else:
                    exec_time = strategy.execution_time_estimate
                
                # Calculate strategy score (success probability / execution time)
                score = success_prob / max(exec_time, 1.0)
                
                # Resource availability check
                if await self._check_resource_availability(strategy.resource_requirements):
                    score *= 1.2  # Bonus for resource availability
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
            
            return best_strategy
            
        except Exception as e:
            self.logger.error(f"ML strategy selection failed: {e}")
            return None
    
    async def _execute_recovery(self,
                               component_name: str,
                               strategy: RecoveryStrategy,
                               attempt: RecoveryAttempt):
        """Execute recovery strategy with ML monitoring"""
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing recovery for {component_name} using {strategy.strategy_id}")
            
            # Track resource usage
            start_resources = await self._get_system_resources()
            
            # Execute each recovery action
            success = True
            for action in strategy.recovery_actions:
                action_success = await self._execute_recovery_action(component_name, action)
                
                if not action_success:
                    success = False
                    attempt.failure_reason = f"Action failed: {action.value}"
                    break
                
                # Brief pause between actions
                await asyncio.sleep(1)
            
            # Verify recovery success
            if success:
                await asyncio.sleep(5)  # Wait for component to stabilize
                success = await self._verify_recovery_success(component_name)
            
            # Calculate execution time and resource usage
            execution_time = time.time() - start_time
            end_resources = await self._get_system_resources()
            resource_usage = {
                'cpu_delta': end_resources['cpu'] - start_resources['cpu'],
                'memory_delta': end_resources['memory'] - start_resources['memory']
            }
            
            # Update attempt record
            with self.recovery_lock:
                attempt.success = success
                attempt.execution_time = execution_time
                attempt.resource_usage = resource_usage
                
                if success:
                    self.recovery_stats['successful_recoveries'] += 1
                    attempt.actual_outcome_score = 1.0
                else:
                    self.recovery_stats['failed_recoveries'] += 1
                    attempt.actual_outcome_score = 0.0
                
                # ML learning feedback
                if self.enable_ml_optimization:
                    attempt.learning_feedback = await self._generate_learning_feedback(
                        attempt, strategy
                    )
                
                # Move to history and remove from active
                self.recovery_history.append(attempt)
                del self.active_recoveries[component_name]
                
                # Update ML models with new data
                if strategy.learning_enabled:
                    await self._update_ml_models_with_feedback(attempt)
            
            # Trigger callbacks
            for callback in self.recovery_callbacks:
                try:
                    await callback(component_name, success, attempt)
                except Exception as e:
                    self.logger.error(f"Recovery callback error: {e}")
            
            # Update component state
            health = self.ml_monitor.component_health.get(component_name)
            if health:
                if success:
                    health.state = ComponentState.HEALTHY
                    health.consecutive_failures = 0
                    health.restart_count += 1
                    health.last_restart = datetime.now()
                else:
                    health.state = ComponentState.FAILED
            
            if success:
                self.logger.info(f"Recovery successful for {component_name} ({execution_time:.1f}s)")
            else:
                self.logger.error(f"Recovery failed for {component_name} ({execution_time:.1f}s)")
                
                # Attempt escalation if available
                await self._attempt_escalation(component_name, strategy)
            
        except Exception as e:
            self.logger.error(f"Recovery execution error for {component_name}: {e}")
            
            # Cleanup on exception
            with self.recovery_lock:
                attempt.success = False
                attempt.failure_reason = str(e)
                self.recovery_history.append(attempt)
                
                if component_name in self.active_recoveries:
                    del self.active_recoveries[component_name]
    
    async def _execute_recovery_action(self, component_name: str, action: WatchdogAction) -> bool:
        """Execute individual recovery action"""
        
        try:
            if action == WatchdogAction.RESTART_COMPONENT:
                return await self._restart_component(component_name)
            
            elif action == WatchdogAction.RESTART_SERVICE:
                return await self._restart_service(component_name)
            
            elif action == WatchdogAction.RESTART_PROCESS:
                return await self._restart_process(component_name)
            
            elif action == WatchdogAction.GRACEFUL_SHUTDOWN:
                return await self._graceful_shutdown(component_name)
            
            elif action == WatchdogAction.FORCE_RESTART:
                return await self._force_restart(component_name)
            
            elif action == WatchdogAction.ML_OPTIMIZE:
                return await self._ml_optimize_component(component_name)
            
            elif action == WatchdogAction.PREDICTIVE_SCALE:
                return await self._predictive_scale(component_name)
            
            elif action == WatchdogAction.ALERT_ONLY:
                await self._send_alert(component_name, "Recovery action: Alert only")
                return True
            
            else:
                self.logger.warning(f"Unknown recovery action: {action}")
                return False
                
        except Exception as e:
            self.logger.error(f"Recovery action {action} failed for {component_name}: {e}")
            return False
    
    async def _restart_component(self, component_name: str) -> bool:
        """Restart component using custom or default method"""
        
        try:
            # Use custom recovery action if available
            if component_name in self.custom_recovery_actions:
                return await self.custom_recovery_actions[component_name]()
            
            # Get component instance from ML monitor
            rule = self.ml_monitor.watchdog_rules.get(component_name)
            if not rule:
                return False
            
            # Default restart logic - would integrate with actual component restart
            self.logger.info(f"Performing default restart for {component_name}")
            
            # Simulate restart delay
            await asyncio.sleep(2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Component restart failed for {component_name}: {e}")
            return False
    
    async def _ml_optimize_component(self, component_name: str) -> bool:
        """Apply ML-driven optimization to component"""
        
        try:
            health = self.ml_monitor.component_health.get(component_name)
            if not health:
                return False
            
            # Apply optimization suggestions from ML analysis
            optimizations_applied = 0
            
            for suggestion in health.optimization_suggestions:
                if "memory" in suggestion.lower():
                    # Apply memory optimization
                    await self._apply_memory_optimization(component_name)
                    optimizations_applied += 1
                
                elif "cpu" in suggestion.lower():
                    # Apply CPU optimization
                    await self._apply_cpu_optimization(component_name)
                    optimizations_applied += 1
                
                elif "cache" in suggestion.lower():
                    # Apply cache optimization
                    await self._apply_cache_optimization(component_name)
                    optimizations_applied += 1
            
            self.logger.info(f"Applied {optimizations_applied} ML optimizations to {component_name}")
            return optimizations_applied > 0
            
        except Exception as e:
            self.logger.error(f"ML optimization failed for {component_name}: {e}")
            return False
    
    async def _recovery_manager_loop(self):
        """Main recovery management loop"""
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Process recovery queue
                await self._process_recovery_queue()
                
                # Check for stuck recoveries
                await self._check_stuck_recoveries()
                
                # Update recovery statistics
                await self._update_recovery_statistics()
                
                # Cleanup old records
                await self._cleanup_old_recovery_records()
                
            except Exception as e:
                self.logger.error(f"Recovery manager loop error: {e}")
                await asyncio.sleep(5)
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """Get comprehensive recovery system status"""
        
        # Active recoveries
        active_summary = {}
        for comp_name, attempt in self.active_recoveries.items():
            active_summary[comp_name] = {
                'strategy': attempt.strategy_used,
                'started': attempt.timestamp.isoformat(),
                'predicted_success': attempt.predicted_success_rate,
                'actions': [action.value for action in attempt.actions_taken]
            }
        
        # Recent recovery history
        recent_recoveries = []
        for attempt in list(self.recovery_history)[-10:]:
            recent_recoveries.append({
                'component': attempt.component_name,
                'timestamp': attempt.timestamp.isoformat(),
                'strategy': attempt.strategy_used,
                'success': attempt.success,
                'execution_time': attempt.execution_time
            })
        
        return {
            'active_recoveries': active_summary,
            'recovery_queue_size': len(self.recovery_queue),
            'recent_recoveries': recent_recoveries,
            'statistics': self.recovery_stats.copy(),
            'strategies_available': sum(len(strategies) for strategies in self.recovery_strategies.values()),
            'ml_optimization_enabled': self.enable_ml_optimization
        }
    
    async def shutdown(self):
        """Graceful shutdown of recovery system"""
        
        self.logger.info("Shutting down recovery system...")
        
        # Wait for active recoveries to complete (with timeout)
        timeout = 30
        while self.active_recoveries and timeout > 0:
            await asyncio.sleep(1)
            timeout -= 1
        
        self.shutdown_event.set()
        await asyncio.sleep(1)
        self.logger.info("Recovery system shutdown complete")