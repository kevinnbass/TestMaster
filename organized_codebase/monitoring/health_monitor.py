"""
Health Monitor Module
====================

System health monitoring and snapshot management.
Extracted from realtime_performance_monitoring.py for better modularity.

Author: Agent E - Infrastructure Consolidation
"""

import asyncio
import logging
import psutil
import random
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set

from .monitoring_models import SystemHealthSnapshot, SystemType


class HealthMonitor:
    """Monitors overall system health and maintains health snapshots."""
    
    def __init__(self):
        self.logger = logging.getLogger("health_monitor")
        
        # Health snapshots storage
        self.health_snapshots: Dict[SystemType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.current_health: Dict[SystemType, SystemHealthSnapshot] = {}
        
        # Health configuration
        self.health_config = {
            "snapshot_interval_seconds": 60,
            "health_score_threshold_degraded": 70,
            "health_score_threshold_critical": 30,
            "auto_recovery_enabled": True,
            "health_history_hours": 24
        }
        
        # Health statistics
        self.health_stats = {
            "total_snapshots_taken": 0,
            "systems_healthy": 0,
            "systems_degraded": 0,
            "systems_critical": 0,
            "systems_offline": 0,
            "avg_health_score": 100.0,
            "recovery_actions_triggered": 0
        }
        
        # System dependencies
        self.system_dependencies: Dict[SystemType, Set[SystemType]] = {
            SystemType.INTELLIGENCE: {SystemType.ANALYTICS, SystemType.MONITORING},
            SystemType.TESTING: {SystemType.ANALYTICS},
            SystemType.ANALYTICS: {SystemType.MONITORING},
            SystemType.ORCHESTRATION: {SystemType.MONITORING, SystemType.INTEGRATION},
            SystemType.INTEGRATION: {SystemType.MONITORING}
        }
        
        # Recovery actions
        self.recovery_actions: Dict[str, callable] = {}
    
    def register_recovery_action(self, issue_type: str, action: callable):
        """Register recovery action for specific issue type."""
        self.recovery_actions[issue_type] = action
        self.logger.info(f"Registered recovery action for: {issue_type}")
    
    async def take_health_snapshot(self, system: SystemType) -> SystemHealthSnapshot:
        """Take health snapshot for a system."""
        snapshot = SystemHealthSnapshot(system=system)
        
        # Collect system metrics
        snapshot.cpu_usage = psutil.cpu_percent(interval=0.1)
        snapshot.memory_usage = psutil.virtual_memory().percent
        snapshot.disk_usage = psutil.disk_usage('/').percent
        
        # Simulate network usage
        snapshot.network_usage = random.uniform(10, 80)
        
        # Simulate performance metrics
        snapshot.response_time_ms = random.uniform(50, 1500)
        snapshot.throughput_ops_sec = random.uniform(100, 1000)
        snapshot.error_rate_percent = random.uniform(0, 2)
        
        # Simulate detailed metrics
        snapshot.active_connections = random.randint(10, 500)
        snapshot.queue_depth = random.randint(0, 100)
        snapshot.cache_hit_rate = random.uniform(70, 99)
        
        # Simulate issues (for demonstration)
        if random.random() < 0.1:  # 10% chance of warning
            snapshot.warnings.append("High memory usage detected")
        if random.random() < 0.05:  # 5% chance of critical issue
            snapshot.critical_issues.append("Service degradation detected")
            snapshot.active_alerts = random.randint(1, 5)
        
        # Calculate health score
        snapshot.calculate_health_score()
        
        # Store snapshot
        self.health_snapshots[system].append(snapshot)
        self.current_health[system] = snapshot
        
        # Update statistics
        self._update_health_stats()
        
        self.logger.debug(f"Health snapshot taken for {system.value}: score={snapshot.health_score:.1f}")
        
        return snapshot
    
    async def monitor_all_systems(self) -> Dict[SystemType, SystemHealthSnapshot]:
        """Monitor health of all systems."""
        snapshots = {}
        
        for system in SystemType:
            snapshot = await self.take_health_snapshot(system)
            snapshots[system] = snapshot
            
            # Check for recovery actions
            if self.health_config["auto_recovery_enabled"]:
                await self._check_recovery_needed(system, snapshot)
        
        self.health_stats["total_snapshots_taken"] += len(snapshots)
        
        return snapshots
    
    async def _check_recovery_needed(self, system: SystemType, snapshot: SystemHealthSnapshot):
        """Check if recovery actions are needed."""
        if snapshot.status in ["critical", "offline"]:
            # Check for specific issues
            for issue in snapshot.critical_issues:
                if issue in self.recovery_actions:
                    try:
                        action = self.recovery_actions[issue]
                        if asyncio.iscoroutinefunction(action):
                            await action(system, snapshot)
                        else:
                            action(system, snapshot)
                        
                        self.health_stats["recovery_actions_triggered"] += 1
                        self.logger.info(f"Recovery action triggered for {system.value}: {issue}")
                    except Exception as e:
                        self.logger.error(f"Recovery action failed: {e}")
    
    def _update_health_stats(self):
        """Update health statistics."""
        # Count systems by status
        self.health_stats["systems_healthy"] = sum(
            1 for snapshot in self.current_health.values() if snapshot.status == "healthy"
        )
        self.health_stats["systems_degraded"] = sum(
            1 for snapshot in self.current_health.values() if snapshot.status == "degraded"
        )
        self.health_stats["systems_critical"] = sum(
            1 for snapshot in self.current_health.values() if snapshot.status == "critical"
        )
        self.health_stats["systems_offline"] = sum(
            1 for snapshot in self.current_health.values() if snapshot.status == "offline"
        )
        
        # Calculate average health score
        if self.current_health:
            total_score = sum(snapshot.health_score for snapshot in self.current_health.values())
            self.health_stats["avg_health_score"] = total_score / len(self.current_health)
    
    def get_system_health(self, system: SystemType) -> Optional[SystemHealthSnapshot]:
        """Get current health snapshot for a system."""
        return self.current_health.get(system)
    
    def get_all_system_health(self) -> Dict[SystemType, SystemHealthSnapshot]:
        """Get current health snapshots for all systems."""
        return self.current_health.copy()
    
    def get_health_history(self, system: SystemType, hours: Optional[int] = None) -> List[SystemHealthSnapshot]:
        """Get health history for a system."""
        if system not in self.health_snapshots:
            return []
        
        snapshots = list(self.health_snapshots[system])
        
        if hours:
            cutoff = datetime.now() - timedelta(hours=hours)
            snapshots = [s for s in snapshots if s.timestamp >= cutoff]
        
        return snapshots
    
    def get_health_trends(self, system: SystemType) -> Dict[str, Any]:
        """Analyze health trends for a system."""
        history = self.get_health_history(system, hours=24)
        
        if not history:
            return {"status": "no_data"}
        
        health_scores = [s.health_score for s in history]
        cpu_usage = [s.cpu_usage for s in history]
        memory_usage = [s.memory_usage for s in history]
        response_times = [s.response_time_ms for s in history]
        
        return {
            "avg_health_score": sum(health_scores) / len(health_scores),
            "min_health_score": min(health_scores),
            "max_health_score": max(health_scores),
            "health_trend": self._calculate_trend(health_scores),
            "avg_cpu_usage": sum(cpu_usage) / len(cpu_usage),
            "avg_memory_usage": sum(memory_usage) / len(memory_usage),
            "avg_response_time": sum(response_times) / len(response_times),
            "total_alerts": sum(s.active_alerts for s in history),
            "total_critical_issues": sum(len(s.critical_issues) for s in history)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression
        n = len(values)
        x_values = list(range(n))
        
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_xx = sum(x * x for x in x_values)
        
        if n * sum_xx - sum_x * sum_x == 0:
            return "stable"
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        
        if abs(slope) < 0.1:
            return "stable"
        elif slope > 0:
            return "improving"
        else:
            return "degrading"
    
    def check_system_dependencies(self, system: SystemType) -> Dict[str, Any]:
        """Check health of system dependencies."""
        dependencies = self.system_dependencies.get(system, set())
        
        dependency_health = {}
        overall_healthy = True
        
        for dep_system in dependencies:
            if dep_system in self.current_health:
                snapshot = self.current_health[dep_system]
                dependency_health[dep_system.value] = {
                    "status": snapshot.status,
                    "health_score": snapshot.health_score
                }
                
                if snapshot.status in ["critical", "offline"]:
                    overall_healthy = False
        
        return {
            "dependencies": dependency_health,
            "all_dependencies_healthy": overall_healthy,
            "dependency_count": len(dependencies)
        }
    
    def get_health_stats(self) -> Dict[str, Any]:
        """Get health monitoring statistics."""
        return self.health_stats.copy()


__all__ = ['HealthMonitor']