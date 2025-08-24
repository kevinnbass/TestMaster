#!/usr/bin/env python3
"""
Automated Optimization & Self-Healing System
Agent B Hours 110-120: Enterprise Integration & Advanced Analytics

Advanced system for automated optimization and self-healing capabilities.
"""

import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import statistics
import hashlib

@dataclass
class OptimizationAction:
    """Optimization action data structure"""
    id: str
    type: str  # 'query_optimization', 'index_creation', 'cache_cleanup', 'resource_scaling'
    category: str  # 'performance', 'storage', 'memory', 'maintenance'
    description: str
    target: str  # Database, query, or system component
    estimated_impact: float  # 0-1 score of expected improvement
    risk_level: str  # 'low', 'medium', 'high'
    auto_apply: bool  # Whether this can be applied automatically
    prerequisites: List[str]
    rollback_action: Optional[str]
    created_timestamp: datetime
    applied_timestamp: Optional[datetime] = None
    success: Optional[bool] = None
    actual_impact: Optional[float] = None

@dataclass
class SystemIssue:
    """System issue detection data"""
    id: str
    type: str  # 'performance_degradation', 'resource_exhaustion', 'connection_failure'
    severity: str  # 'low', 'medium', 'high', 'critical'
    component: str  # System component affected
    description: str
    detected_timestamp: datetime
    auto_healing_possible: bool
    healing_actions: List[str]
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None
    resolution_method: Optional[str] = None

@dataclass
class OptimizationRule:
    """Optimization rule configuration"""
    name: str
    condition: str  # Condition that triggers the rule
    action_type: str
    parameters: Dict[str, Any]
    enabled: bool = True
    auto_apply: bool = False
    cooldown_minutes: int = 30
    max_applications_per_day: int = 5

class AutoOptimizer(ABC):
    """Abstract base class for auto-optimizers"""
    
    @abstractmethod
    def analyze_system(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """Analyze system and return optimization actions"""
        pass
    
    @abstractmethod
    def can_apply_automatically(self, action: OptimizationAction) -> bool:
        """Check if action can be applied automatically"""
        pass
    
    @abstractmethod
    def apply_optimization(self, action: OptimizationAction) -> bool:
        """Apply optimization action"""
        pass
    
    @abstractmethod
    def rollback_optimization(self, action: OptimizationAction) -> bool:
        """Rollback optimization action"""
        pass

class QueryOptimizer(AutoOptimizer):
    """Automated query optimization"""
    
    def __init__(self):
        self.optimization_history = []
        self.query_cache = {}
    
    def analyze_system(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """Analyze query performance and suggest optimizations"""
        actions = []
        
        # Check for slow queries
        if 'query_performance' in metrics:
            avg_query_time = metrics['query_performance'].get('avg_ms', 0)
            
            if avg_query_time > 100:  # Slow queries detected
                action = OptimizationAction(
                    id=f"query_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    type="query_optimization",
                    category="performance",
                    description=f"Optimize slow queries (avg {avg_query_time:.1f}ms)",
                    target="database_queries",
                    estimated_impact=0.3,  # 30% improvement expected
                    risk_level="low",
                    auto_apply=True,
                    prerequisites=[],
                    rollback_action="remove_query_optimizations",
                    created_timestamp=datetime.now()
                )
                actions.append(action)
        
        # Check for missing indexes
        if 'database_metrics' in metrics:
            for db_name, db_metrics in metrics['database_metrics'].items():
                table_count = db_metrics.get('table_count', 0)
                index_count = db_metrics.get('index_count', 0)
                
                if table_count > 0 and index_count / table_count < 0.5:  # Low index ratio
                    action = OptimizationAction(
                        id=f"index_opt_{db_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        type="index_creation",
                        category="performance",
                        description=f"Create missing indexes for {db_name}",
                        target=db_name,
                        estimated_impact=0.4,  # 40% improvement expected
                        risk_level="medium",
                        auto_apply=False,  # Require manual approval for index creation
                        prerequisites=["database_backup"],
                        rollback_action="drop_created_indexes",
                        created_timestamp=datetime.now()
                    )
                    actions.append(action)
        
        return actions
    
    def can_apply_automatically(self, action: OptimizationAction) -> bool:
        """Check if query optimization can be applied automatically"""
        return action.auto_apply and action.risk_level == "low"
    
    def apply_optimization(self, action: OptimizationAction) -> bool:
        """Apply query optimization"""
        try:
            if action.type == "query_optimization":
                # Simulate query optimization
                print(f"[OPTIMIZATION] Applying query optimization: {action.description}")
                time.sleep(0.5)  # Simulate work
                
                # Update query cache settings
                self.query_cache['optimization_applied'] = datetime.now()
                self.query_cache['optimization_type'] = action.type
                
                return True
            
            elif action.type == "index_creation":
                print(f"[OPTIMIZATION] Creating indexes for: {action.target}")
                time.sleep(1.0)  # Simulate index creation
                return True
                
            return False
            
        except Exception as e:
            print(f"[ERROR] Failed to apply optimization {action.id}: {e}")
            return False
    
    def rollback_optimization(self, action: OptimizationAction) -> bool:
        """Rollback query optimization"""
        try:
            print(f"[ROLLBACK] Rolling back optimization: {action.description}")
            
            if action.rollback_action == "remove_query_optimizations":
                self.query_cache.pop('optimization_applied', None)
                self.query_cache.pop('optimization_type', None)
                return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Failed to rollback optimization {action.id}: {e}")
            return False

class ResourceOptimizer(AutoOptimizer):
    """Automated resource optimization"""
    
    def __init__(self):
        self.resource_history = []
        self.optimization_actions_applied = []
    
    def analyze_system(self, metrics: Dict[str, Any]) -> List[OptimizationAction]:
        """Analyze resource usage and suggest optimizations"""
        actions = []
        
        system_metrics = metrics.get('system', {})
        
        # Memory optimization
        memory_percent = system_metrics.get('memory_percent', 0)
        if memory_percent > 85:
            action = OptimizationAction(
                id=f"memory_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type="memory_cleanup",
                category="memory",
                description=f"Clean up memory usage ({memory_percent:.1f}% used)",
                target="system_memory",
                estimated_impact=0.2,  # 20% memory reduction expected
                risk_level="low",
                auto_apply=True,
                prerequisites=[],
                rollback_action="restore_memory_settings",
                created_timestamp=datetime.now()
            )
            actions.append(action)
        
        # CPU optimization
        cpu_percent = system_metrics.get('cpu_percent', 0)
        if cpu_percent > 80:
            action = OptimizationAction(
                id=f"cpu_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                type="cpu_optimization",
                category="performance",
                description=f"Optimize CPU usage ({cpu_percent:.1f}% used)",
                target="system_cpu",
                estimated_impact=0.25,  # 25% CPU reduction expected
                risk_level="medium",
                auto_apply=True,
                prerequisites=[],
                rollback_action="restore_cpu_settings",
                created_timestamp=datetime.now()
            )
            actions.append(action)
        
        # Storage optimization
        if 'database_metrics' in metrics:
            total_size = sum(db.get('size_mb', 0) for db in metrics['database_metrics'].values())
            if total_size > 1000:  # Over 1GB
                action = OptimizationAction(
                    id=f"storage_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    type="storage_cleanup",
                    category="storage",
                    description=f"Clean up database storage ({total_size:.1f}MB used)",
                    target="database_storage",
                    estimated_impact=0.15,  # 15% storage reduction expected
                    risk_level="medium",
                    auto_apply=False,  # Require approval for storage cleanup
                    prerequisites=["full_backup"],
                    rollback_action="restore_from_backup",
                    created_timestamp=datetime.now()
                )
                actions.append(action)
        
        return actions
    
    def can_apply_automatically(self, action: OptimizationAction) -> bool:
        """Check if resource optimization can be applied automatically"""
        return action.auto_apply and action.risk_level in ["low", "medium"]
    
    def apply_optimization(self, action: OptimizationAction) -> bool:
        """Apply resource optimization"""
        try:
            if action.type == "memory_cleanup":
                print(f"[OPTIMIZATION] Applying memory cleanup")
                # Simulate memory cleanup operations
                time.sleep(0.5)
                return True
                
            elif action.type == "cpu_optimization":
                print(f"[OPTIMIZATION] Applying CPU optimization")
                # Simulate CPU optimization
                time.sleep(0.3)
                return True
                
            elif action.type == "storage_cleanup":
                print(f"[OPTIMIZATION] Applying storage cleanup")
                # Simulate storage cleanup
                time.sleep(1.0)
                return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Failed to apply resource optimization {action.id}: {e}")
            return False
    
    def rollback_optimization(self, action: OptimizationAction) -> bool:
        """Rollback resource optimization"""
        try:
            print(f"[ROLLBACK] Rolling back resource optimization: {action.description}")
            time.sleep(0.2)
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to rollback resource optimization {action.id}: {e}")
            return False

class SelfHealingSystem:
    """Self-healing system for automatic issue resolution"""
    
    def __init__(self):
        self.healing_history = []
        self.active_issues = {}
        self.healing_rules = {}
        self.setup_default_healing_rules()
    
    def setup_default_healing_rules(self):
        """Setup default self-healing rules"""
        self.healing_rules = {
            'high_memory_usage': {
                'condition': 'memory_percent > 95',
                'actions': ['restart_services', 'clear_caches', 'garbage_collection'],
                'auto_apply': True,
                'severity_threshold': 'high'
            },
            'connection_failures': {
                'condition': 'connection_errors > 5',
                'actions': ['restart_connection_pool', 'retry_connections'],
                'auto_apply': True,
                'severity_threshold': 'medium'
            },
            'disk_space_exhaustion': {
                'condition': 'disk_free_percent < 5',
                'actions': ['cleanup_temp_files', 'compress_logs', 'archive_old_data'],
                'auto_apply': True,
                'severity_threshold': 'critical'
            },
            'performance_degradation': {
                'condition': 'avg_response_time > 5000',
                'actions': ['restart_slow_processes', 'optimize_queries', 'scale_resources'],
                'auto_apply': False,
                'severity_threshold': 'high'
            }
        }
    
    def detect_issues(self, metrics: Dict[str, Any]) -> List[SystemIssue]:
        """Detect system issues from metrics"""
        issues = []
        current_time = datetime.now()
        
        system_metrics = metrics.get('system', {})
        
        # Memory issues
        memory_percent = system_metrics.get('memory_percent', 0)
        if memory_percent > 95:
            issue = SystemIssue(
                id=f"memory_critical_{current_time.strftime('%Y%m%d_%H%M%S')}",
                type="resource_exhaustion",
                severity="critical",
                component="system_memory",
                description=f"Critical memory usage: {memory_percent:.1f}%",
                detected_timestamp=current_time,
                auto_healing_possible=True,
                healing_actions=self.healing_rules['high_memory_usage']['actions']
            )
            issues.append(issue)
        
        # CPU issues
        cpu_percent = system_metrics.get('cpu_percent', 0)
        if cpu_percent > 95:
            issue = SystemIssue(
                id=f"cpu_critical_{current_time.strftime('%Y%m%d_%H%M%S')}",
                type="performance_degradation",
                severity="critical",
                component="system_cpu",
                description=f"Critical CPU usage: {cpu_percent:.1f}%",
                detected_timestamp=current_time,
                auto_healing_possible=True,
                healing_actions=['throttle_processes', 'scale_cpu']
            )
            issues.append(issue)
        
        # Database connection issues (simulated)
        if 'database_metrics' in metrics:
            for db_name, db_metrics in metrics['database_metrics'].items():
                connection_status = db_metrics.get('connection_status', 'unknown')
                if connection_status in ['error', 'disconnected']:
                    issue = SystemIssue(
                        id=f"db_connection_{db_name}_{current_time.strftime('%Y%m%d_%H%M%S')}",
                        type="connection_failure",
                        severity="high",
                        component=f"database_{db_name}",
                        description=f"Database connection failed: {db_name}",
                        detected_timestamp=current_time,
                        auto_healing_possible=True,
                        healing_actions=self.healing_rules['connection_failures']['actions']
                    )
                    issues.append(issue)
        
        return issues
    
    def apply_healing_action(self, issue: SystemIssue, action: str) -> bool:
        """Apply a healing action for an issue"""
        try:
            print(f"[HEALING] Applying healing action '{action}' for issue: {issue.description}")
            
            if action == "restart_services":
                print("[HEALING] Restarting affected services...")
                time.sleep(1.0)  # Simulate restart time
                return True
                
            elif action == "clear_caches":
                print("[HEALING] Clearing system caches...")
                time.sleep(0.5)
                return True
                
            elif action == "garbage_collection":
                print("[HEALING] Running garbage collection...")
                time.sleep(0.3)
                return True
                
            elif action == "restart_connection_pool":
                print("[HEALING] Restarting connection pool...")
                time.sleep(0.8)
                return True
                
            elif action == "retry_connections":
                print("[HEALING] Retrying failed connections...")
                time.sleep(0.5)
                return True
                
            elif action == "cleanup_temp_files":
                print("[HEALING] Cleaning up temporary files...")
                time.sleep(1.2)
                return True
                
            elif action == "compress_logs":
                print("[HEALING] Compressing log files...")
                time.sleep(2.0)
                return True
                
            elif action == "throttle_processes":
                print("[HEALING] Throttling resource-intensive processes...")
                time.sleep(0.7)
                return True
                
            else:
                print(f"[WARNING] Unknown healing action: {action}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to apply healing action '{action}': {e}")
            return False
    
    def heal_issue(self, issue: SystemIssue) -> bool:
        """Attempt to heal a system issue"""
        if not issue.auto_healing_possible:
            print(f"[WARNING] Issue {issue.id} cannot be auto-healed")
            return False
        
        # Apply healing actions in sequence
        healing_successful = True
        for action in issue.healing_actions:
            if not self.apply_healing_action(issue, action):
                healing_successful = False
                break
        
        if healing_successful:
            issue.resolved = True
            issue.resolved_timestamp = datetime.now()
            issue.resolution_method = "auto_healing"
            self.healing_history.append(issue)
            print(f"[OK] Successfully healed issue: {issue.description}")
        else:
            print(f"[ERROR] Failed to heal issue: {issue.description}")
        
        return healing_successful

class AutomatedOptimizationSystem:
    """Main automated optimization and self-healing system"""
    
    def __init__(self, config_file: str = "auto_optimization_config.json"):
        self.config_file = Path(config_file)
        self.optimization_rules = []
        self.optimization_history = []
        self.active_optimizations = {}
        
        # Initialize optimizers
        self.query_optimizer = QueryOptimizer()
        self.resource_optimizer = ResourceOptimizer()
        self.self_healing_system = SelfHealingSystem()
        
        # System state
        self.auto_optimization_enabled = True
        self.auto_healing_enabled = True
        self.monitoring_active = False
        
        # Performance tracking
        self.optimization_stats = {
            'total_optimizations_applied': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'total_issues_healed': 0,
            'healing_success_rate': 0.0,
            'avg_optimization_impact': 0.0
        }
        
        # Load configuration
        self.load_configuration()
        
        print("[OK] Automated Optimization System initialized")
        print(f"[OK] Auto-optimization: {'Enabled' if self.auto_optimization_enabled else 'Disabled'}")
        print(f"[OK] Auto-healing: {'Enabled' if self.auto_healing_enabled else 'Disabled'}")
    
    def load_configuration(self):
        """Load system configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                self.auto_optimization_enabled = config.get('auto_optimization_enabled', True)
                self.auto_healing_enabled = config.get('auto_healing_enabled', True)
                self.optimization_stats = config.get('optimization_stats', self.optimization_stats)
                
            except Exception as e:
                print(f"[WARNING] Failed to load optimization config: {e}")
    
    def save_configuration(self):
        """Save system configuration"""
        try:
            config = {
                'auto_optimization_enabled': self.auto_optimization_enabled,
                'auto_healing_enabled': self.auto_healing_enabled,
                'optimization_stats': self.optimization_stats
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            print(f"[ERROR] Failed to save optimization config: {e}")
    
    def analyze_and_optimize(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system and apply optimizations"""
        result = {
            'optimizations_identified': 0,
            'optimizations_applied': 0,
            'issues_detected': 0,
            'issues_healed': 0,
            'actions_taken': []
        }
        
        if not self.auto_optimization_enabled:
            return result
        
        # Get optimization suggestions from all optimizers
        all_optimizations = []
        
        # Query optimizations
        query_opts = self.query_optimizer.analyze_system(metrics)
        all_optimizations.extend(query_opts)
        
        # Resource optimizations
        resource_opts = self.resource_optimizer.analyze_system(metrics)
        all_optimizations.extend(resource_opts)
        
        result['optimizations_identified'] = len(all_optimizations)
        
        # Apply automatic optimizations
        for optimization in all_optimizations:
            optimizer = self._get_optimizer_for_action(optimization)
            
            if optimizer and optimizer.can_apply_automatically(optimization):
                success = optimizer.apply_optimization(optimization)
                
                if success:
                    optimization.applied_timestamp = datetime.now()
                    optimization.success = True
                    result['optimizations_applied'] += 1
                    result['actions_taken'].append(f"Applied {optimization.type}: {optimization.description}")
                    
                    # Update stats
                    self.optimization_stats['total_optimizations_applied'] += 1
                    self.optimization_stats['successful_optimizations'] += 1
                else:
                    optimization.success = False
                    self.optimization_stats['failed_optimizations'] += 1
                
                self.optimization_history.append(optimization)
        
        # Self-healing
        if self.auto_healing_enabled:
            issues = self.self_healing_system.detect_issues(metrics)
            result['issues_detected'] = len(issues)
            
            for issue in issues:
                if self.self_healing_system.heal_issue(issue):
                    result['issues_healed'] += 1
                    result['actions_taken'].append(f"Healed {issue.type}: {issue.description}")
                    
                    # Update stats
                    self.optimization_stats['total_issues_healed'] += 1
        
        # Update success rate
        total_healing_attempts = self.optimization_stats['total_issues_healed'] + len(issues) - result['issues_healed']
        if total_healing_attempts > 0:
            self.optimization_stats['healing_success_rate'] = self.optimization_stats['total_issues_healed'] / total_healing_attempts
        
        self.save_configuration()
        return result
    
    def _get_optimizer_for_action(self, action: OptimizationAction) -> Optional[AutoOptimizer]:
        """Get the appropriate optimizer for an action"""
        if action.category in ["performance", "query"]:
            return self.query_optimizer
        elif action.category in ["memory", "storage", "cpu"]:
            return self.resource_optimizer
        else:
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        recent_optimizations = [opt for opt in self.optimization_history[-20:]]
        recent_healing = [issue for issue in self.self_healing_system.healing_history[-20:]]
        
        return {
            'system_status': {
                'auto_optimization_enabled': self.auto_optimization_enabled,
                'auto_healing_enabled': self.auto_healing_enabled,
                'monitoring_active': self.monitoring_active
            },
            'optimization_stats': self.optimization_stats,
            'recent_activity': {
                'recent_optimizations': len(recent_optimizations),
                'recent_healing_actions': len(recent_healing),
                'successful_optimizations_24h': len([opt for opt in recent_optimizations 
                                                   if opt.success and opt.applied_timestamp and 
                                                   opt.applied_timestamp > datetime.now() - timedelta(hours=24)])
            },
            'system_health': {
                'optimization_success_rate': (self.optimization_stats['successful_optimizations'] / 
                                            max(1, self.optimization_stats['total_optimizations_applied'])),
                'healing_success_rate': self.optimization_stats['healing_success_rate'],
                'overall_health_score': self._calculate_system_health_score()
            }
        }
    
    def _calculate_system_health_score(self) -> int:
        """Calculate overall system health score"""
        score = 100
        
        # Deduct for failed optimizations
        failed_rate = (self.optimization_stats['failed_optimizations'] / 
                      max(1, self.optimization_stats['total_optimizations_applied']))
        score -= failed_rate * 30
        
        # Add for successful healing
        healing_rate = self.optimization_stats['healing_success_rate']
        score += healing_rate * 10
        
        # Consider recent activity
        if len(self.optimization_history) > 10:
            recent_successes = sum(1 for opt in self.optimization_history[-10:] if opt.success)
            if recent_successes < 5:
                score -= 20
        
        return max(0, min(100, int(score)))
    
    def start_continuous_optimization(self, interval_seconds: int = 300):
        """Start continuous optimization monitoring"""
        if self.monitoring_active:
            print("[WARNING] Continuous optimization already active")
            return
        
        self.monitoring_active = True
        
        def optimization_loop():
            print(f"[OK] Started continuous optimization (interval: {interval_seconds}s)")
            
            while self.monitoring_active:
                try:
                    # This would normally receive metrics from the monitoring system
                    # For now, we'll simulate some metrics
                    simulated_metrics = self._get_simulated_metrics()
                    
                    # Analyze and optimize
                    result = self.analyze_and_optimize(simulated_metrics)
                    
                    if result['actions_taken']:
                        print(f"[OPTIMIZATION] Applied {result['optimizations_applied']} optimizations, healed {result['issues_healed']} issues")
                        for action in result['actions_taken']:
                            print(f"[OPTIMIZATION] {action}")
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    print(f"[ERROR] Optimization loop error: {e}")
                    time.sleep(interval_seconds * 2)
        
        optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        optimization_thread.start()
    
    def stop_continuous_optimization(self):
        """Stop continuous optimization monitoring"""
        self.monitoring_active = False
        print("[OK] Continuous optimization stopped")
    
    def _get_simulated_metrics(self) -> Dict[str, Any]:
        """Generate simulated metrics for testing"""
        import random
        
        return {
            'system': {
                'cpu_percent': random.uniform(20, 90),
                'memory_percent': random.uniform(30, 95),
                'disk_percent': random.uniform(40, 80)
            },
            'database_metrics': {
                'main_db': {
                    'size_mb': random.uniform(50, 500),
                    'table_count': random.randint(5, 20),
                    'index_count': random.randint(2, 15),
                    'connection_status': random.choice(['connected', 'connected', 'error'])
                }
            },
            'query_performance': {
                'avg_ms': random.uniform(10, 200)
            }
        }
    
    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        status = self.get_system_status()
        
        report = f"""
AUTOMATED OPTIMIZATION SYSTEM REPORT
====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SYSTEM STATUS:
- Auto-Optimization: {'Enabled' if status['system_status']['auto_optimization_enabled'] else 'Disabled'}
- Auto-Healing: {'Enabled' if status['system_status']['auto_healing_enabled'] else 'Disabled'}
- Monitoring Active: {'Yes' if status['system_status']['monitoring_active'] else 'No'}

OPTIMIZATION STATISTICS:
- Total Optimizations Applied: {status['optimization_stats']['total_optimizations_applied']}
- Successful Optimizations: {status['optimization_stats']['successful_optimizations']}
- Failed Optimizations: {status['optimization_stats']['failed_optimizations']}
- Success Rate: {status['system_health']['optimization_success_rate']:.1%}

SELF-HEALING STATISTICS:
- Total Issues Healed: {status['optimization_stats']['total_issues_healed']}
- Healing Success Rate: {status['optimization_stats']['healing_success_rate']:.1%}

RECENT ACTIVITY (24 hours):
- Recent Optimizations: {status['recent_activity']['recent_optimizations']}
- Recent Healing Actions: {status['recent_activity']['recent_healing_actions']}
- Successful Optimizations: {status['recent_activity']['successful_optimizations_24h']}

SYSTEM HEALTH:
- Overall Health Score: {status['system_health']['overall_health_score']}/100
- System Status: {'Excellent' if status['system_health']['overall_health_score'] > 90 else 
                  'Good' if status['system_health']['overall_health_score'] > 75 else 
                  'Fair' if status['system_health']['overall_health_score'] > 60 else 'Poor'}
"""
        
        if self.optimization_history:
            report += f"\nRECENT OPTIMIZATIONS:\n"
            for opt in self.optimization_history[-5:]:
                status_symbol = "[OK]" if opt.success else "[FAIL]"
                report += f"- {status_symbol} {opt.type}: {opt.description}\n"
        
        if self.self_healing_system.healing_history:
            report += f"\nRECENT HEALING ACTIONS:\n"
            for issue in self.self_healing_system.healing_history[-5:]:
                report += f"- Resolved {issue.type}: {issue.description}\n"
        
        return report

def main():
    """Main function for testing automated optimization system"""
    system = AutomatedOptimizationSystem()
    
    print("[OK] Automated Optimization System ready for testing")
    
    # Test with simulated metrics
    test_metrics = {
        'system': {
            'cpu_percent': 85.0,  # High CPU to trigger optimization
            'memory_percent': 90.0,  # High memory to trigger optimization
            'disk_percent': 45.0
        },
        'database_metrics': {
            'test_db': {
                'size_mb': 250.0,
                'table_count': 10,
                'index_count': 3,  # Low index ratio
                'connection_status': 'connected'
            }
        },
        'query_performance': {
            'avg_ms': 150.0  # Slow queries
        }
    }
    
    print("\n[TEST] Analyzing system with high resource usage...")
    result = system.analyze_and_optimize(test_metrics)
    
    print(f"[RESULT] Identified {result['optimizations_identified']} optimizations")
    print(f"[RESULT] Applied {result['optimizations_applied']} optimizations")
    print(f"[RESULT] Detected {result['issues_detected']} issues")
    print(f"[RESULT] Healed {result['issues_healed']} issues")
    
    if result['actions_taken']:
        print("\nACTIONS TAKEN:")
        for action in result['actions_taken']:
            print(f"- {action}")
    
    # Test continuous optimization for a short period
    print("\n[TEST] Starting continuous optimization for 30 seconds...")
    system.start_continuous_optimization(interval_seconds=10)
    time.sleep(30)
    system.stop_continuous_optimization()
    
    # Generate report
    report = system.generate_optimization_report()
    print("\n" + "="*60)
    print(report)
    
    print("\n[OK] Automated Optimization System test completed")

if __name__ == "__main__":
    main()