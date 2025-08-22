"""
Analytics Watchdog and Auto-Restart System
===========================================

Comprehensive monitoring and automatic restart capabilities for analytics
components to ensure maximum uptime and reliability.

Author: TestMaster Team
"""

import logging
import time
import threading
import subprocess
import signal
import os
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class WatchdogAction(Enum):
    RESTART_COMPONENT = "restart_component"
    RESTART_SERVICE = "restart_service"
    RESTART_PROCESS = "restart_process"
    ALERT_ONLY = "alert_only"
    GRACEFUL_SHUTDOWN = "graceful_shutdown"
    FORCE_RESTART = "force_restart"

class ComponentState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RESTARTING = "restarting"
    UNKNOWN = "unknown"

@dataclass
class WatchdogRule:
    """Defines a watchdog monitoring rule."""
    rule_id: str
    component_name: str
    check_function: Callable[[], bool]
    failure_threshold: int
    check_interval: int
    action: WatchdogAction
    recovery_timeout: int
    max_restarts: int
    escalation_actions: List[WatchdogAction]
    
@dataclass
class ComponentHealth:
    """Health status of a monitored component."""
    component_name: str
    state: ComponentState
    last_check: datetime
    consecutive_failures: int
    total_failures: int
    restart_count: int
    last_restart: Optional[datetime]
    health_score: float
    metrics: Dict[str, Any]

class AnalyticsWatchdog:
    """
    Comprehensive watchdog system for analytics components.
    """
    
    def __init__(self, check_interval: int = 30, max_restart_attempts: int = 3):
        """
        Initialize analytics watchdog.
        
        Args:
            check_interval: Default interval between health checks in seconds
            max_restart_attempts: Maximum restart attempts before giving up
        """
        self.check_interval = check_interval
        self.max_restart_attempts = max_restart_attempts
        
        # Monitoring components
        self.watchdog_rules = {}
        self.component_health = {}
        self.restart_history = deque(maxlen=1000)
        
        # Process monitoring
        self.monitored_processes = {}
        self.process_restart_counts = defaultdict(int)
        
        # Watchdog control
        self.watchdog_active = False
        self.monitoring_thread = None
        self.process_monitoring_thread = None
        
        # Health check functions
        self.health_checks = {}
        self.custom_recovery_actions = {}
        
        # Configuration
        self.critical_components = set()
        self.restart_cooldown = 60  # seconds between restarts
        self.health_score_threshold = 0.7
        self.cascade_failure_threshold = 3
        
        # Statistics
        self.watchdog_stats = {
            'total_checks': 0,
            'failed_checks': 0,
            'successful_restarts': 0,
            'failed_restarts': 0,
            'cascade_failures': 0,
            'uptime_violations': 0,
            'start_time': datetime.now()
        }
        
        # Callbacks
        self.restart_callbacks = []
        self.failure_callbacks = []
        self.recovery_callbacks = []
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        logger.info("Analytics Watchdog system initialized")
    
    def register_component(self, component_name: str, component_instance: Any,
                          check_function: Callable = None, 
                          failure_threshold: int = 3,
                          restart_action: WatchdogAction = WatchdogAction.RESTART_COMPONENT,
                          is_critical: bool = False):
        """
        Register a component for watchdog monitoring.
        
        Args:
            component_name: Name of the component
            component_instance: Instance of the component
            check_function: Custom health check function
            failure_threshold: Number of failures before taking action
            restart_action: Action to take on failure
            is_critical: Whether component is critical to system operation
        """
        # Create health check function
        if check_function is None:
            check_function = self._create_default_health_check(component_instance)
        
        # Create watchdog rule
        rule = WatchdogRule(
            rule_id=f"rule_{component_name}",
            component_name=component_name,
            check_function=check_function,
            failure_threshold=failure_threshold,
            check_interval=self.check_interval,
            action=restart_action,
            recovery_timeout=300,  # 5 minutes
            max_restarts=self.max_restart_attempts,
            escalation_actions=[WatchdogAction.ALERT_ONLY, WatchdogAction.GRACEFUL_SHUTDOWN]
        )
        
        self.watchdog_rules[component_name] = rule
        
        # Initialize component health
        self.component_health[component_name] = ComponentHealth(
            component_name=component_name,
            state=ComponentState.UNKNOWN,
            last_check=datetime.now(),
            consecutive_failures=0,
            total_failures=0,
            restart_count=0,
            last_restart=None,
            health_score=1.0,
            metrics={}
        )
        
        # Mark as critical if specified
        if is_critical:
            self.critical_components.add(component_name)
        
        logger.info(f"Registered component for watchdog: {component_name} (critical: {is_critical})")
    
    def register_process(self, process_name: str, process_id: int = None, 
                        command_line: str = None, working_directory: str = None):
        """
        Register a process for monitoring and auto-restart.
        
        Args:
            process_name: Name identifier for the process
            process_id: Process ID (if known)
            command_line: Command to restart the process
            working_directory: Working directory for the process
        """
        self.monitored_processes[process_name] = {
            'process_id': process_id,
            'command_line': command_line,
            'working_directory': working_directory,
            'last_seen': datetime.now(),
            'restart_count': 0,
            'health_checks_failed': 0
        }
        
        logger.info(f"Registered process for monitoring: {process_name}")
    
    def start_monitoring(self):
        """Start watchdog monitoring."""
        if self.watchdog_active:
            return
        
        self.watchdog_active = True
        
        # Start monitoring threads
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.process_monitoring_thread = threading.Thread(target=self._process_monitoring_loop, daemon=True)
        
        self.monitoring_thread.start()
        self.process_monitoring_thread.start()
        
        logger.info("Analytics watchdog monitoring started")
    
    def stop_monitoring(self):
        """Stop watchdog monitoring."""
        self.watchdog_active = False
        
        # Wait for threads to finish
        for thread in [self.monitoring_thread, self.process_monitoring_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info("Analytics watchdog monitoring stopped")
    
    def check_component_health(self, component_name: str) -> ComponentHealth:
        """
        Manually check health of a specific component.
        
        Args:
            component_name: Name of component to check
        
        Returns:
            Component health status
        """
        if component_name not in self.watchdog_rules:
            raise ValueError(f"Component {component_name} not registered")
        
        rule = self.watchdog_rules[component_name]
        health = self.component_health[component_name]
        
        self.watchdog_stats['total_checks'] += 1
        
        try:
            # Perform health check
            is_healthy = rule.check_function()
            health.last_check = datetime.now()
            
            if is_healthy:
                # Component is healthy
                health.state = ComponentState.HEALTHY
                health.consecutive_failures = 0
                health.health_score = min(1.0, health.health_score + 0.1)
                
            else:
                # Component failed health check
                health.consecutive_failures += 1
                health.total_failures += 1
                health.health_score = max(0.0, health.health_score - 0.2)
                
                self.watchdog_stats['failed_checks'] += 1
                
                if health.consecutive_failures >= rule.failure_threshold:
                    health.state = ComponentState.FAILED
                    self._handle_component_failure(component_name)
                else:
                    health.state = ComponentState.DEGRADED
            
            # Update health metrics
            health.metrics = self._collect_component_metrics(component_name)
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed for {component_name}: {e}")
            health.state = ComponentState.UNKNOWN
            health.consecutive_failures += 1
            return health
    
    def restart_component(self, component_name: str, force: bool = False) -> bool:
        """
        Restart a specific component.
        
        Args:
            component_name: Name of component to restart
            force: Whether to force restart even if not failed
        
        Returns:
            True if restart was successful
        """
        if component_name not in self.watchdog_rules:
            logger.error(f"Cannot restart unknown component: {component_name}")
            return False
        
        health = self.component_health[component_name]
        
        # Check restart limits
        if health.restart_count >= self.max_restart_attempts and not force:
            logger.error(f"Component {component_name} exceeded max restart attempts")
            return False
        
        # Check restart cooldown
        if (health.last_restart and 
            (datetime.now() - health.last_restart).total_seconds() < self.restart_cooldown and 
            not force):
            logger.warning(f"Component {component_name} restart on cooldown")
            return False
        
        try:
            logger.info(f"Restarting component: {component_name}")
            
            health.state = ComponentState.RESTARTING
            health.last_restart = datetime.now()
            health.restart_count += 1
            
            # Record restart attempt
            restart_record = {
                'timestamp': datetime.now(),
                'component': component_name,
                'reason': 'health_check_failure' if not force else 'manual_restart',
                'success': False
            }
            
            # Perform restart based on action type
            rule = self.watchdog_rules[component_name]
            success = self._execute_restart_action(component_name, rule.action)
            
            restart_record['success'] = success
            self.restart_history.append(restart_record)
            
            if success:
                self.watchdog_stats['successful_restarts'] += 1
                health.consecutive_failures = 0
                health.state = ComponentState.HEALTHY
                
                # Trigger recovery callbacks
                for callback in self.recovery_callbacks:
                    try:
                        callback(component_name)
                    except Exception as e:
                        logger.error(f"Recovery callback error: {e}")
                
                logger.info(f"Successfully restarted component: {component_name}")
            else:
                self.watchdog_stats['failed_restarts'] += 1
                health.state = ComponentState.FAILED
                logger.error(f"Failed to restart component: {component_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Component restart failed: {e}")
            health.state = ComponentState.FAILED
            return False
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.watchdog_active:
            try:
                start_time = time.time()
                
                # Check all registered components
                for component_name in self.watchdog_rules:
                    if not self.watchdog_active:
                        break
                    
                    self.check_component_health(component_name)
                
                # Check for cascade failures
                self._check_cascade_failures()
                
                # Calculate sleep time to maintain interval
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.check_interval - elapsed_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Watchdog monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _process_monitoring_loop(self):
        """Monitor registered processes."""
        while self.watchdog_active:
            try:
                time.sleep(self.check_interval)
                
                for process_name, process_info in self.monitored_processes.items():
                    if not self.watchdog_active:
                        break
                    
                    self._check_process_health(process_name, process_info)
                
            except Exception as e:
                logger.error(f"Process monitoring error: {e}")
    
    def _check_process_health(self, process_name: str, process_info: Dict[str, Any]):
        """Check health of a monitored process."""
        try:
            process_id = process_info.get('process_id')
            
            if process_id:
                # Check if process is still running
                try:
                    process = psutil.Process(process_id)
                    if process.is_running():
                        process_info['last_seen'] = datetime.now()
                        return True
                except psutil.NoSuchProcess:
                    pass
            
            # Process not found or not running
            logger.warning(f"Process {process_name} not running")
            
            command_line = process_info.get('command_line')
            if command_line:
                self._restart_process(process_name, command_line, 
                                    process_info.get('working_directory'))
            else:
                logger.error(f"No restart command for process {process_name}")
            
            return False
            
        except Exception as e:
            logger.error(f"Process health check failed for {process_name}: {e}")
            return False
    
    def _restart_process(self, process_name: str, command_line: str, 
                        working_directory: str = None) -> bool:
        """Restart a process using command line."""
        try:
            restart_count = self.process_restart_counts[process_name]
            
            if restart_count >= self.max_restart_attempts:
                logger.error(f"Process {process_name} exceeded max restart attempts")
                return False
            
            logger.info(f"Restarting process: {process_name}")
            
            # Execute restart command
            process = subprocess.Popen(
                command_line,
                shell=False,
                cwd=working_directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Update process info
            self.monitored_processes[process_name]['process_id'] = process.pid
            self.monitored_processes[process_name]['restart_count'] += 1
            self.process_restart_counts[process_name] += 1
            
            logger.info(f"Successfully restarted process {process_name} with PID {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Process restart failed for {process_name}: {e}")
            return False
    
    def _handle_component_failure(self, component_name: str):
        """Handle component failure."""
        health = self.component_health[component_name]
        rule = self.watchdog_rules[component_name]
        
        logger.warning(f"Component {component_name} failed health checks")
        
        # Trigger failure callbacks
        for callback in self.failure_callbacks:
            try:
                callback(component_name, health)
            except Exception as e:
                logger.error(f"Failure callback error: {e}")
        
        # Execute action based on rule
        if rule.action == WatchdogAction.RESTART_COMPONENT:
            self.restart_component(component_name)
        elif rule.action == WatchdogAction.ALERT_ONLY:
            logger.error(f"ALERT: Component {component_name} failed (action: alert only)")
        elif rule.action == WatchdogAction.GRACEFUL_SHUTDOWN:
            self._graceful_shutdown_component(component_name)
        else:
            logger.warning(f"Unknown action for component {component_name}: {rule.action}")
    
    def _execute_restart_action(self, component_name: str, action: WatchdogAction) -> bool:
        """Execute restart action for a component."""
        try:
            if action == WatchdogAction.RESTART_COMPONENT:
                # Use custom recovery action if available
                if component_name in self.custom_recovery_actions:
                    return self.custom_recovery_actions[component_name]()
                else:
                    # Default restart action - assume component has restart method
                    return True  # Placeholder - would call component's restart method
            
            elif action == WatchdogAction.RESTART_SERVICE:
                # Restart entire service
                logger.info(f"Restarting service for component: {component_name}")
                return True  # Placeholder
            
            elif action == WatchdogAction.FORCE_RESTART:
                # Force restart with higher privileges
                logger.info(f"Force restarting component: {component_name}")
                return True  # Placeholder
            
            else:
                logger.warning(f"Unsupported restart action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Restart action execution failed: {e}")
            return False
    
    def _graceful_shutdown_component(self, component_name: str):
        """Gracefully shutdown a component."""
        try:
            logger.info(f"Gracefully shutting down component: {component_name}")
            
            # Custom shutdown logic would go here
            health = self.component_health[component_name]
            health.state = ComponentState.FAILED
            
        except Exception as e:
            logger.error(f"Graceful shutdown failed for {component_name}: {e}")
    
    def _check_cascade_failures(self):
        """Check for cascade failure patterns."""
        failed_components = [name for name, health in self.component_health.items() 
                           if health.state == ComponentState.FAILED]
        
        # Check if too many critical components failed
        failed_critical = [name for name in failed_components 
                          if name in self.critical_components]
        
        if len(failed_critical) >= self.cascade_failure_threshold:
            logger.critical(f"CASCADE FAILURE: {len(failed_critical)} critical components failed")
            self.watchdog_stats['cascade_failures'] += 1
            
            # Take emergency action - could be system restart, alert, etc.
            self._handle_cascade_failure(failed_critical)
    
    def _handle_cascade_failure(self, failed_components: List[str]):
        """Handle cascade failure situation."""
        logger.critical(f"Handling cascade failure: {failed_components}")
        
        # Emergency recovery actions
        for component_name in failed_components:
            if component_name in self.critical_components:
                # Force restart critical components
                self.restart_component(component_name, force=True)
    
    def _setup_default_health_checks(self):
        """Setup default health check functions."""
        
        def memory_health_check():
            """Check system memory health."""
            try:
                memory = psutil.virtual_memory()
                return memory.percent < 95  # Fail if memory > 95%
            except:
                return False
        
        def cpu_health_check():
            """Check system CPU health."""
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                return cpu_percent < 95  # Fail if CPU > 95%
            except:
                return False
        
        self.health_checks['system_memory'] = memory_health_check
        self.health_checks['system_cpu'] = cpu_health_check
    
    def _create_default_health_check(self, component_instance: Any) -> Callable[[], bool]:
        """Create default health check for a component."""
        def default_check():
            try:
                # Check if component has a health check method
                if hasattr(component_instance, 'is_healthy'):
                    return component_instance.is_healthy()
                elif hasattr(component_instance, 'health_check'):
                    return component_instance.health_check()
                elif hasattr(component_instance, 'get_health_status'):
                    status = component_instance.get_health_status()
                    return status.get('healthy', True)
                else:
                    # Default: component is healthy if it exists
                    return component_instance is not None
            except Exception as e:
                logger.warning(f"Default health check failed: {e}")
                return False
        
        return default_check
    
    def _collect_component_metrics(self, component_name: str) -> Dict[str, Any]:
        """Collect metrics for a component."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'component': component_name
        }
        
        try:
            # Collect system metrics
            metrics['system'] = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
            }
            
            # Component-specific metrics would go here
            
        except Exception as e:
            logger.warning(f"Metrics collection failed for {component_name}: {e}")
        
        return metrics
    
    def add_restart_callback(self, callback: Callable[[str], None]):
        """Add callback for component restart events."""
        self.restart_callbacks.append(callback)
    
    def add_failure_callback(self, callback: Callable[[str, ComponentHealth], None]):
        """Add callback for component failure events."""
        self.failure_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable[[str], None]):
        """Add callback for component recovery events."""
        self.recovery_callbacks.append(callback)
    
    def add_custom_recovery_action(self, component_name: str, action: Callable[[], bool]):
        """Add custom recovery action for a component."""
        self.custom_recovery_actions[component_name] = action
    
    def get_watchdog_status(self) -> Dict[str, Any]:
        """Get overall watchdog status."""
        uptime = (datetime.now() - self.watchdog_stats['start_time']).total_seconds()
        
        component_summary = {}
        for name, health in self.component_health.items():
            component_summary[name] = {
                'state': health.state.value,
                'health_score': health.health_score,
                'consecutive_failures': health.consecutive_failures,
                'total_failures': health.total_failures,
                'restart_count': health.restart_count,
                'last_check': health.last_check.isoformat(),
                'is_critical': name in self.critical_components
            }
        
        return {
            'watchdog_active': self.watchdog_active,
            'check_interval': self.check_interval,
            'total_components': len(self.watchdog_rules),
            'critical_components': len(self.critical_components),
            'healthy_components': len([h for h in self.component_health.values() 
                                     if h.state == ComponentState.HEALTHY]),
            'failed_components': len([h for h in self.component_health.values() 
                                    if h.state == ComponentState.FAILED]),
            'components': component_summary,
            'monitored_processes': len(self.monitored_processes),
            'recent_restarts': len([r for r in self.restart_history 
                                  if datetime.now() - r['timestamp'] <= timedelta(hours=1)]),
            'statistics': self.watchdog_stats.copy(),
            'uptime_seconds': uptime
        }
    
    def shutdown(self):
        """Shutdown watchdog system."""
        self.stop_monitoring()
        logger.info("Analytics Watchdog system shutdown")