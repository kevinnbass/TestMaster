"""
Analytics Health Monitor
========================

Comprehensive health monitoring and auto-recovery system for analytics components.
Monitors system health, detects failures, and automatically recovers from issues.

Author: TestMaster Team
"""

import asyncio
import logging
import threading
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from collections import defaultdict, deque
from enum import Enum
import psutil
import gc

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    RECOVERING = "recovering"

class ComponentType(Enum):
    AGGREGATOR = "aggregator"
    VALIDATOR = "validator"
    CORRELATOR = "correlator"
    OPTIMIZER = "optimizer"
    BACKUP_MANAGER = "backup_manager"
    PERFORMANCE_MONITOR = "performance_monitor"
    STREAMING = "streaming"
    PERSISTENCE = "persistence"
    PIPELINE = "pipeline"

class HealthCheck:
    """Represents a health check for a component."""
    
    def __init__(self, name: str, check_func: Callable, interval: int = 60, 
                 timeout: int = 30, critical: bool = False):
        """
        Initialize a health check.
        
        Args:
            name: Name of the health check
            check_func: Function that performs the check
            interval: Check interval in seconds
            timeout: Check timeout in seconds
            critical: Whether this is a critical check
        """
        self.name = name
        self.check_func = check_func
        self.interval = interval
        self.timeout = timeout
        self.critical = critical
        
        # State tracking
        self.last_check = None
        self.last_status = HealthStatus.UNKNOWN
        self.consecutive_failures = 0
        self.total_checks = 0
        self.total_failures = 0
        self.last_error = None
        self.check_history = deque(maxlen=100)
    
    def execute(self) -> Dict[str, Any]:
        """Execute the health check."""
        start_time = time.time()
        self.total_checks += 1
        
        try:
            # Execute the check function with timeout
            result = self._execute_with_timeout()
            
            if result.get('healthy', False):
                self.last_status = HealthStatus.HEALTHY
                self.consecutive_failures = 0
                self.last_error = None
            else:
                self.last_status = HealthStatus.CRITICAL if self.critical else HealthStatus.WARNING
                self.consecutive_failures += 1
                self.total_failures += 1
                self.last_error = result.get('error', 'Check failed')
            
        except Exception as e:
            self.last_status = HealthStatus.CRITICAL if self.critical else HealthStatus.WARNING
            self.consecutive_failures += 1
            self.total_failures += 1
            self.last_error = str(e)
            result = {'healthy': False, 'error': str(e)}
        
        execution_time = time.time() - start_time
        self.last_check = datetime.now()
        
        check_result = {
            'name': self.name,
            'status': self.last_status.value,
            'healthy': self.last_status == HealthStatus.HEALTHY,
            'execution_time': execution_time,
            'consecutive_failures': self.consecutive_failures,
            'last_error': self.last_error,
            'timestamp': self.last_check.isoformat(),
            'details': result
        }
        
        self.check_history.append(check_result)
        return check_result
    
    def _execute_with_timeout(self) -> Dict[str, Any]:
        """Execute check function with timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Health check {self.name} timed out after {self.timeout}s")
        
        try:
            # Set alarm for timeout (Unix only)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
            
            result = self.check_func()
            
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Cancel alarm
            
            return result if isinstance(result, dict) else {'healthy': bool(result)}
            
        except Exception as e:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Cancel alarm
            raise e
    
    def get_stats(self) -> Dict[str, Any]:
        """Get check statistics."""
        success_rate = ((self.total_checks - self.total_failures) / max(self.total_checks, 1)) * 100
        
        return {
            'name': self.name,
            'interval': self.interval,
            'critical': self.critical,
            'last_status': self.last_status.value,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'consecutive_failures': self.consecutive_failures,
            'total_checks': self.total_checks,
            'total_failures': self.total_failures,
            'success_rate': success_rate,
            'last_error': self.last_error
        }

class RecoveryAction:
    """Represents a recovery action for a component."""
    
    def __init__(self, name: str, action_func: Callable, conditions: Dict[str, Any] = None):
        """
        Initialize a recovery action.
        
        Args:
            name: Name of the recovery action
            action_func: Function that performs the recovery
            conditions: Conditions under which to trigger this action
        """
        self.name = name
        self.action_func = action_func
        self.conditions = conditions or {}
        
        # Execution tracking
        self.execution_count = 0
        self.last_execution = None
        self.last_success = None
        self.execution_history = deque(maxlen=50)
    
    def should_execute(self, health_result: Dict[str, Any]) -> bool:
        """Check if this recovery action should be executed."""
        # Check consecutive failures condition
        if 'min_consecutive_failures' in self.conditions:
            if health_result.get('consecutive_failures', 0) < self.conditions['min_consecutive_failures']:
                return False
        
        # Check status condition
        if 'trigger_statuses' in self.conditions:
            if health_result.get('status') not in self.conditions['trigger_statuses']:
                return False
        
        # Check cooldown period
        if 'cooldown_minutes' in self.conditions and self.last_execution:
            cooldown = timedelta(minutes=self.conditions['cooldown_minutes'])
            if datetime.now() - self.last_execution < cooldown:
                return False
        
        return True
    
    def execute(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the recovery action."""
        start_time = time.time()
        self.execution_count += 1
        self.last_execution = datetime.now()
        
        try:
            result = self.action_func(context or {})
            self.last_success = True
            
            execution_result = {
                'name': self.name,
                'success': True,
                'execution_time': time.time() - start_time,
                'timestamp': self.last_execution.isoformat(),
                'result': result
            }
            
        except Exception as e:
            self.last_success = False
            
            execution_result = {
                'name': self.name,
                'success': False,
                'execution_time': time.time() - start_time,
                'timestamp': self.last_execution.isoformat(),
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            logger.error(f"Recovery action {self.name} failed: {e}")
        
        self.execution_history.append(execution_result)
        return execution_result

class AnalyticsHealthMonitor:
    """
    Comprehensive health monitoring system for analytics components.
    """
    
    def __init__(self, check_interval: int = 30):
        """
        Initialize the health monitor.
        
        Args:
            check_interval: Global check interval in seconds
        """
        self.check_interval = check_interval
        self.health_checks = {}
        self.recovery_actions = defaultdict(list)
        self.component_registry = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Health tracking
        self.system_health_history = deque(maxlen=1000)
        self.component_health = defaultdict(lambda: HealthStatus.UNKNOWN)
        self.last_full_check = None
        
        # Recovery tracking
        self.recovery_log = deque(maxlen=200)
        self.recovery_stats = defaultdict(int)
        
        # Alerting
        self.alert_callbacks = []
        self.alert_throttle = defaultdict(datetime)  # Alert throttling
        
        # System resource monitoring
        self.resource_thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'disk_percent': 95.0
        }
        
        logger.info("Analytics Health Monitor initialized")
    
    def register_component(self, component_type: ComponentType, component_instance: Any):
        """Register a component for monitoring."""
        self.component_registry[component_type] = component_instance
        logger.info(f"Registered component for monitoring: {component_type.value}")
    
    def add_health_check(self, component_type: ComponentType, check: HealthCheck):
        """Add a health check for a component."""
        if component_type not in self.health_checks:
            self.health_checks[component_type] = []
        self.health_checks[component_type].append(check)
        logger.info(f"Added health check '{check.name}' for {component_type.value}")
    
    def add_recovery_action(self, component_type: ComponentType, action: RecoveryAction):
        """Add a recovery action for a component."""
        self.recovery_actions[component_type].append(action)
        logger.info(f"Added recovery action '{action.name}' for {component_type.value}")
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a callback function for health alerts."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start the health monitoring service."""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Setup default health checks
        self._setup_default_checks()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop the health monitoring service."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        logger.info("Health monitoring stopped")
    
    def run_health_check(self, component_type: ComponentType = None) -> Dict[str, Any]:
        """
        Run health checks for all components or a specific component.
        
        Args:
            component_type: Specific component to check, or None for all
        
        Returns:
            Health check results
        """
        start_time = time.time()
        results = {}
        
        components_to_check = [component_type] if component_type else self.health_checks.keys()
        
        for comp_type in components_to_check:
            if comp_type not in self.health_checks:
                continue
            
            component_results = []
            component_status = HealthStatus.HEALTHY
            
            for check in self.health_checks[comp_type]:
                try:
                    result = check.execute()
                    component_results.append(result)
                    
                    # Update component status
                    if not result['healthy']:
                        if check.critical or result['status'] == 'critical':
                            component_status = HealthStatus.CRITICAL
                        elif component_status == HealthStatus.HEALTHY:
                            component_status = HealthStatus.WARNING
                    
                    # Trigger recovery if needed
                    self._trigger_recovery(comp_type, result)
                    
                except Exception as e:
                    logger.error(f"Health check execution failed: {e}")
                    component_status = HealthStatus.CRITICAL
            
            self.component_health[comp_type] = component_status
            results[comp_type.value] = {
                'status': component_status.value,
                'checks': component_results,
                'check_count': len(component_results),
                'healthy_checks': sum(1 for r in component_results if r['healthy'])
            }
        
        # Calculate overall system health
        overall_status = self._calculate_overall_health(results)
        
        health_summary = {
            'overall_status': overall_status.value,
            'components': results,
            'check_duration': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'system_resources': self._check_system_resources()
        }
        
        self.last_full_check = datetime.now()
        self.system_health_history.append(health_summary)
        
        # Send alerts if needed
        self._send_alerts(health_summary)
        
        return health_summary
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status summary."""
        if not self.system_health_history:
            return {'status': 'unknown', 'message': 'No health checks performed yet'}
        
        latest = self.system_health_history[-1]
        
        return {
            'overall_status': latest['overall_status'],
            'last_check': latest['timestamp'],
            'monitoring_active': self.monitoring_active,
            'component_count': len(latest['components']),
            'healthy_components': sum(
                1 for comp in latest['components'].values() 
                if comp['status'] == 'healthy'
            ),
            'system_resources': latest.get('system_resources', {}),
            'recent_recoveries': len([
                r for r in self.recovery_log 
                if datetime.fromisoformat(r['timestamp']) > datetime.now() - timedelta(hours=1)
            ])
        }
    
    def get_component_health(self, component_type: ComponentType) -> Dict[str, Any]:
        """Get detailed health information for a specific component."""
        if component_type not in self.health_checks:
            return {'error': f'No health checks registered for {component_type.value}'}
        
        # Get latest check results
        latest_results = []
        for check in self.health_checks[component_type]:
            if check.check_history:
                latest_results.append(check.check_history[-1])
        
        return {
            'component_type': component_type.value,
            'status': self.component_health[component_type].value,
            'check_count': len(self.health_checks[component_type]),
            'latest_results': latest_results,
            'recovery_actions': len(self.recovery_actions[component_type]),
            'check_stats': [check.get_stats() for check in self.health_checks[component_type]]
        }
    
    def force_recovery(self, component_type: ComponentType, action_name: str = None) -> Dict[str, Any]:
        """Force execution of recovery actions for a component."""
        if component_type not in self.recovery_actions:
            return {'error': f'No recovery actions for {component_type.value}'}
        
        recovery_results = []
        
        for action in self.recovery_actions[component_type]:
            if action_name and action.name != action_name:
                continue
            
            try:
                result = action.execute({'forced': True})
                recovery_results.append(result)
                
                # Log recovery
                self.recovery_log.append({
                    'component_type': component_type.value,
                    'action_name': action.name,
                    'timestamp': result['timestamp'],
                    'success': result['success'],
                    'forced': True
                })
                
            except Exception as e:
                logger.error(f"Forced recovery failed: {e}")
                recovery_results.append({
                    'name': action.name,
                    'success': False,
                    'error': str(e)
                })
        
        return {
            'component_type': component_type.value,
            'recovery_results': recovery_results,
            'total_actions': len(recovery_results)
        }
    
    def _setup_default_checks(self):
        """Set up default health checks for known components."""
        # Aggregator health check
        def check_aggregator():
            aggregator = self.component_registry.get(ComponentType.AGGREGATOR)
            if not aggregator:
                return {'healthy': False, 'error': 'Aggregator not registered'}
            
            try:
                # Test basic functionality
                analytics = aggregator.get_comprehensive_analytics()
                return {
                    'healthy': True,
                    'components_count': len(analytics),
                    'cache_active': hasattr(aggregator, '_cache') and bool(aggregator._cache)
                }
            except Exception as e:
                return {'healthy': False, 'error': str(e)}
        
        # Performance monitor health check
        def check_performance_monitor():
            monitor = self.component_registry.get(ComponentType.PERFORMANCE_MONITOR)
            if not monitor:
                return {'healthy': False, 'error': 'Performance monitor not registered'}
            
            try:
                stats = monitor.get_performance_summary()
                return {
                    'healthy': stats.get('monitoring_active', False),
                    'total_operations': stats.get('total_operations', 0),
                    'error_rate': stats.get('overall_error_rate', 0)
                }
            except Exception as e:
                return {'healthy': False, 'error': str(e)}
        
        # System resources check
        def check_system_resources():
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                issues = []
                if cpu_percent > self.resource_thresholds['cpu_percent']:
                    issues.append(f'High CPU usage: {cpu_percent:.1f}%')
                
                if memory.percent > self.resource_thresholds['memory_percent']:
                    issues.append(f'High memory usage: {memory.percent:.1f}%')
                
                return {
                    'healthy': len(issues) == 0,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'issues': issues
                }
            except Exception as e:
                return {'healthy': False, 'error': str(e)}
        
        # Add default checks
        self.add_health_check(
            ComponentType.AGGREGATOR,
            HealthCheck('aggregator_basic', check_aggregator, interval=60, critical=True)
        )
        
        self.add_health_check(
            ComponentType.PERFORMANCE_MONITOR,
            HealthCheck('performance_monitor_basic', check_performance_monitor, interval=60)
        )
        
        self.add_health_check(
            ComponentType.AGGREGATOR,
            HealthCheck('system_resources', check_system_resources, interval=30, critical=True)
        )
        
        # Add default recovery actions
        self._setup_default_recovery_actions()
    
    def _setup_default_recovery_actions(self):
        """Set up default recovery actions."""
        # Memory cleanup action
        def cleanup_memory(context):
            gc.collect()
            return {'action': 'memory_cleanup', 'freed_objects': gc.collect()}
        
        # Cache reset action
        def reset_cache(context):
            aggregator = self.component_registry.get(ComponentType.AGGREGATOR)
            if aggregator and hasattr(aggregator, '_cache'):
                cache_size = len(aggregator._cache)
                aggregator._cache.clear()
                aggregator._cache_timestamps.clear()
                return {'action': 'cache_reset', 'cleared_entries': cache_size}
            return {'action': 'cache_reset', 'result': 'no_cache_found'}
        
        # Component restart action
        def restart_component(context):
            component_type = context.get('component_type')
            if component_type in self.component_registry:
                component = self.component_registry[component_type]
                # Attempt to restart monitoring if available
                if hasattr(component, 'stop_monitoring') and hasattr(component, 'start_monitoring'):
                    component.stop_monitoring()
                    time.sleep(1)
                    component.start_monitoring()
                    return {'action': 'component_restart', 'component': component_type.value}
            return {'action': 'component_restart', 'result': 'restart_not_supported'}
        
        # Add recovery actions
        memory_recovery = RecoveryAction(
            'memory_cleanup',
            cleanup_memory,
            {'min_consecutive_failures': 2, 'cooldown_minutes': 5}
        )
        
        cache_recovery = RecoveryAction(
            'cache_reset',
            reset_cache,
            {'min_consecutive_failures': 3, 'cooldown_minutes': 10}
        )
        
        restart_recovery = RecoveryAction(
            'component_restart',
            restart_component,
            {'min_consecutive_failures': 5, 'cooldown_minutes': 15}
        )
        
        self.add_recovery_action(ComponentType.AGGREGATOR, memory_recovery)
        self.add_recovery_action(ComponentType.AGGREGATOR, cache_recovery)
        self.add_recovery_action(ComponentType.PERFORMANCE_MONITOR, restart_recovery)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Run health checks
                self.run_health_check()
                
                # Sleep until next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(5)  # Back off on error
    
    def _trigger_recovery(self, component_type: ComponentType, health_result: Dict[str, Any]):
        """Trigger recovery actions if conditions are met."""
        if component_type not in self.recovery_actions:
            return
        
        for action in self.recovery_actions[component_type]:
            if action.should_execute(health_result):
                try:
                    result = action.execute({'component_type': component_type, 'health_result': health_result})
                    
                    # Log recovery
                    self.recovery_log.append({
                        'component_type': component_type.value,
                        'action_name': action.name,
                        'timestamp': result['timestamp'],
                        'success': result['success'],
                        'trigger': 'automatic'
                    })
                    
                    self.recovery_stats[f"{component_type.value}_{action.name}"] += 1
                    
                    if result['success']:
                        logger.info(f"Recovery action '{action.name}' executed successfully for {component_type.value}")
                    
                except Exception as e:
                    logger.error(f"Recovery action failed: {e}")
    
    def _calculate_overall_health(self, component_results: Dict[str, Any]) -> HealthStatus:
        """Calculate overall system health from component results."""
        if not component_results:
            return HealthStatus.UNKNOWN
        
        critical_count = sum(1 for comp in component_results.values() if comp['status'] == 'critical')
        warning_count = sum(1 for comp in component_results.values() if comp['status'] == 'warning')
        
        if critical_count > 0:
            return HealthStatus.CRITICAL
        elif warning_count > 0:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3)
            }
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return {}
    
    def _send_alerts(self, health_summary: Dict[str, Any]):
        """Send alerts based on health status."""
        if not self.alert_callbacks:
            return
        
        overall_status = health_summary['overall_status']
        
        # Check if we should throttle alerts
        alert_key = f"overall_{overall_status}"
        if alert_key in self.alert_throttle:
            if datetime.now() - self.alert_throttle[alert_key] < timedelta(minutes=15):
                return  # Throttle alerts
        
        if overall_status in ['critical', 'warning']:
            self.alert_throttle[alert_key] = datetime.now()
            
            for callback in self.alert_callbacks:
                try:
                    callback(health_summary)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        total_checks = sum(
            sum(check.total_checks for check in checks)
            for checks in self.health_checks.values()
        )
        
        total_failures = sum(
            sum(check.total_failures for check in checks)
            for checks in self.health_checks.values()
        )
        
        uptime = (datetime.now() - 
                 self.system_health_history[0]['timestamp'] if self.system_health_history 
                 else datetime.now()).total_seconds() if self.system_health_history else 0
        
        return {
            'monitoring_active': self.monitoring_active,
            'check_interval': self.check_interval,
            'registered_components': len(self.component_registry),
            'total_health_checks': sum(len(checks) for checks in self.health_checks.values()),
            'total_recovery_actions': sum(len(actions) for actions in self.recovery_actions.values()),
            'total_checks_executed': total_checks,
            'total_check_failures': total_failures,
            'overall_success_rate': ((total_checks - total_failures) / max(total_checks, 1)) * 100,
            'recent_recoveries': len(self.recovery_log),
            'recovery_stats': dict(self.recovery_stats),
            'uptime_hours': uptime / 3600,
            'last_full_check': self.last_full_check.isoformat() if self.last_full_check else None
        }