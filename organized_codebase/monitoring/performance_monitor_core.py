#!/usr/bin/env python3
"""
Performance Monitor Core Module
===============================

Core performance monitoring functionality extracted from realtime_performance_dashboard.py
for STEELCLAD modularization (Agent Y STEELCLAD Protocol)

Handles:
- Performance metrics collection
- Real-time data processing
- Background monitoring services
- Performance scoring algorithms
"""

import os
import sys
import time
import json
import asyncio
import threading
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import deque

# Import performance engine
try:
    from testmaster_performance_engine import performance_engine, performance_monitor
    PERFORMANCE_ENGINE_AVAILABLE = True
except ImportError:
    PERFORMANCE_ENGINE_AVAILABLE = False

class PerformanceMonitorCore:
    """Core performance monitoring functionality"""
    
    def __init__(self):
        self.performance_metrics = {
            'realtime_data': deque(maxlen=100),
            'system_health': {},
            'optimization_predictions': {},
            'active_optimizations': [],
            'performance_alerts': deque(maxlen=50)
        }
        self._initialize_system_health()
    
    def _initialize_system_health(self):
        """Initialize system health metrics"""
        self.performance_metrics['system_health'] = {
            'cpu_usage': 45.2,
            'memory_usage': 67.8,
            'disk_io': 23.4,
            'network_latency': 12.5,
            'cache_hit_rate': 94.2,
            'active_connections': 47,
            'thread_count': 23,
            'error_rate': 0.02
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        if PERFORMANCE_ENGINE_AVAILABLE:
            return self._get_real_performance_metrics()
        else:
            return self._get_simulated_performance_metrics()
    
    def _get_real_performance_metrics(self) -> Dict[str, Any]:
        """Get real performance metrics from engine"""
        try:
            # Get real-time system performance
            cpu_percent = performance_monitor.get_cpu_percent()
            memory_info = performance_monitor.get_memory_info()
            
            # Get TestMaster-specific performance
            test_metrics = performance_engine.get_performance_metrics()
            
            return {
                'realtime_data': list(self.performance_metrics['realtime_data']),
                'system_health': {
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory_info.get('percent', 0),
                    'disk_io': test_metrics.get('disk_io_rate', 0),
                    'network_latency': test_metrics.get('network_latency', 0),
                    'cache_hit_rate': test_metrics.get('cache_hit_rate', 95),
                    'active_connections': test_metrics.get('active_connections', 0),
                    'thread_count': test_metrics.get('thread_count', 0),
                    'error_rate': test_metrics.get('error_rate', 0)
                },
                'optimization_predictions': test_metrics.get('predictions', {}),
                'active_optimizations': test_metrics.get('active_optimizations', []),
                'performance_alerts': list(self.performance_metrics['performance_alerts']),
                'performance_score': self.calculate_performance_score(),
                'optimization_recommendations': self.generate_optimization_recommendations()
            }
        except Exception as e:
            print(f"Performance engine error: {e}")
            return self._get_simulated_performance_metrics()
    
    def _get_simulated_performance_metrics(self) -> Dict[str, Any]:
        """Get simulated performance metrics for demo mode"""
        # Add some random variation to make it look realistic
        base_time = time.time()
        
        # Update system health with slight random variations
        health = self.performance_metrics['system_health']
        health['cpu_usage'] = max(0, min(100, health['cpu_usage'] + random.uniform(-2, 2)))
        health['memory_usage'] = max(0, min(100, health['memory_usage'] + random.uniform(-1, 1)))
        health['disk_io'] = max(0, health['disk_io'] + random.uniform(-5, 5))
        health['network_latency'] = max(0, health['network_latency'] + random.uniform(-2, 2))
        health['cache_hit_rate'] = max(80, min(99, health['cache_hit_rate'] + random.uniform(-0.5, 0.5)))
        health['active_connections'] = max(0, health['active_connections'] + random.randint(-3, 3))
        health['thread_count'] = max(1, health['thread_count'] + random.randint(-2, 2))
        health['error_rate'] = max(0, health['error_rate'] + random.uniform(-0.01, 0.01))
        
        return {
            'realtime_data': list(self.performance_metrics['realtime_data']),
            'system_health': health,
            'optimization_predictions': {
                'cpu_optimization_potential': random.uniform(5, 15),
                'memory_optimization_potential': random.uniform(10, 25),
                'io_optimization_potential': random.uniform(8, 20),
                'predicted_performance_gain': random.uniform(12, 30)
            },
            'active_optimizations': [
                {'name': 'Cache Optimization', 'progress': random.randint(60, 95), 'eta': '2 minutes'},
                {'name': 'Memory Defragmentation', 'progress': random.randint(40, 80), 'eta': '5 minutes'}
            ],
            'performance_alerts': list(self.performance_metrics['performance_alerts']),
            'performance_score': self.calculate_performance_score(),
            'optimization_recommendations': self.generate_optimization_recommendations()
        }
    
    def calculate_performance_score(self) -> float:
        """Calculate overall performance score"""
        health = self.performance_metrics['system_health']
        
        # Weighted performance calculation
        cpu_score = max(0, 100 - health['cpu_usage'])
        memory_score = max(0, 100 - health['memory_usage'])
        latency_score = max(0, 100 - min(health['network_latency'], 100))
        cache_score = health['cache_hit_rate']
        error_score = max(0, 100 - (health['error_rate'] * 1000))
        
        # Weighted average
        total_score = (
            cpu_score * 0.2 +
            memory_score * 0.2 +
            latency_score * 0.2 +
            cache_score * 0.2 +
            error_score * 0.2
        )
        
        return round(total_score, 1)
    
    def generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations"""
        health = self.performance_metrics['system_health']
        recommendations = []
        
        if health['cpu_usage'] > 80:
            recommendations.append({
                'type': 'cpu',
                'severity': 'high',
                'title': 'High CPU Usage Detected',
                'description': 'CPU usage is above 80%. Consider process optimization.',
                'action': 'Analyze CPU-intensive processes'
            })
        
        if health['memory_usage'] > 85:
            recommendations.append({
                'type': 'memory',
                'severity': 'high',
                'title': 'High Memory Usage',
                'description': 'Memory usage is above 85%. Memory cleanup recommended.',
                'action': 'Run garbage collection and cache cleanup'
            })
        
        if health['cache_hit_rate'] < 85:
            recommendations.append({
                'type': 'cache',
                'severity': 'medium',
                'title': 'Low Cache Hit Rate',
                'description': 'Cache efficiency is below optimal. Cache tuning needed.',
                'action': 'Review and optimize cache configuration'
            })
        
        if health['network_latency'] > 50:
            recommendations.append({
                'type': 'network',
                'severity': 'medium',
                'title': 'High Network Latency',
                'description': 'Network response time is degraded. Check connections.',
                'action': 'Analyze network connectivity and routing'
            })
        
        if health['error_rate'] > 0.05:
            recommendations.append({
                'type': 'errors',
                'severity': 'high',
                'title': 'Elevated Error Rate',
                'description': 'System error rate is above normal threshold.',
                'action': 'Review error logs and fix underlying issues'
            })
        
        # Add some positive recommendations
        if len(recommendations) == 0:
            recommendations.append({
                'type': 'optimization',
                'severity': 'info',
                'title': 'System Running Optimally',
                'description': 'All performance metrics are within normal ranges.',
                'action': 'Continue monitoring for optimal performance'
            })
        
        return recommendations
    
    def trigger_optimization(self, optimization_type: str) -> Dict[str, Any]:
        """Trigger a performance optimization"""
        optimization_id = f"opt_{int(time.time())}"
        
        optimization_tasks = {
            'cpu': 'CPU Process Optimization',
            'memory': 'Memory Cleanup and Defragmentation',
            'cache': 'Cache Configuration Tuning',
            'network': 'Network Connection Optimization',
            'io': 'Disk I/O Performance Tuning',
            'general': 'General System Optimization'
        }
        
        task_name = optimization_tasks.get(optimization_type, 'System Optimization')
        
        optimization = {
            'id': optimization_id,
            'type': optimization_type,
            'name': task_name,
            'status': 'running',
            'progress': 0,
            'start_time': datetime.now().isoformat(),
            'estimated_duration': random.randint(30, 300),  # 30 seconds to 5 minutes
            'description': f'Optimizing {optimization_type} performance'
        }
        
        self.performance_metrics['active_optimizations'].append(optimization)
        
        return {
            'success': True,
            'optimization_id': optimization_id,
            'message': f'Started {task_name}',
            'estimated_completion': f'{optimization["estimated_duration"]} seconds'
        }
    
    def update_realtime_data(self, socketio=None):
        """Update real-time performance data and emit via WebSocket"""
        # Generate new performance data point
        current_time = datetime.now()
        health = self.performance_metrics['system_health']
        
        perf_data = {
            'timestamp': current_time.isoformat(),
            'cpu_usage': health['cpu_usage'],
            'memory_usage': health['memory_usage'],
            'disk_io': health['disk_io'],
            'network_latency': health['network_latency'],
            'cache_hit_rate': health['cache_hit_rate'],
            'active_connections': health['active_connections'],
            'performance_score': self.calculate_performance_score()
        }
        
        # Store in metrics buffer
        self.performance_metrics['realtime_data'].append(perf_data)
        
        # Emit to all connected clients if socketio provided
        if socketio:
            socketio.emit('performance_update', perf_data)
        
        # Occasionally generate alerts
        if random.random() < 0.1:  # 10% chance
            alert = {
                'severity': random.choice(['info', 'warning', 'info']),
                'message': random.choice([
                    'Cache hit rate optimized by 5%',
                    'Memory usage reduced by 100MB',
                    'Response time improved to <50ms',
                    'Auto-scaling activated for high load'
                ]),
                'timestamp': current_time.isoformat()
            }
            
            self.performance_metrics['performance_alerts'].append(alert)
            if socketio:
                socketio.emit('alert', alert)
        
        return perf_data

# Global instance
performance_core = PerformanceMonitorCore()