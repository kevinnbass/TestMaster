"""
Async Processing Visualization API
=================================

Provides comprehensive async task monitoring, queue management,
and worker process visualization for frontend.

Author: TestMaster Team
"""

import logging
from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import random
import json
import asyncio
import uuid

logger = logging.getLogger(__name__)

class AsyncProcessingAPI:
    """Async Processing and Task Management API endpoints."""
    
    def __init__(self):
        """Initialize Async Processing API."""
        self.blueprint = Blueprint('async_processing', __name__, url_prefix='/api/async')
        self._setup_routes()
        logger.info("Async Processing API initialized")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.blueprint.route('/tasks/active', methods=['GET'])
        def active_tasks():
            """Get currently active async tasks."""
            try:
                # Generate active tasks
                tasks = []
                task_types = [
                    'test_generation', 'code_analysis', 'security_scan',
                    'coverage_calculation', 'file_processing', 'data_export',
                    'backup_operation', 'integration_test', 'performance_benchmark'
                ]
                
                for i in range(random.randint(5, 25)):
                    task_id = str(uuid.uuid4())[:8]
                    task_type = random.choice(task_types)
                    
                    # Task start time (within last 2 hours)
                    start_time = datetime.now() - timedelta(minutes=random.randint(1, 120))
                    
                    # Progress calculation
                    progress = random.uniform(0.1, 0.95)
                    estimated_duration = random.randint(30, 600)  # seconds
                    elapsed = (datetime.now() - start_time).total_seconds()
                    
                    tasks.append({
                        'task_id': task_id,
                        'task_type': task_type,
                        'status': random.choice(['running', 'running', 'waiting', 'suspended']),
                        'progress_percent': round(progress * 100, 1),
                        'start_time': start_time.isoformat(),
                        'estimated_duration_sec': estimated_duration,
                        'elapsed_time_sec': int(elapsed),
                        'remaining_time_sec': max(0, int(estimated_duration - elapsed)),
                        'worker_id': f'worker_{random.randint(1, 8)}',
                        'priority': random.choice(['low', 'normal', 'high', 'critical']),
                        'memory_mb': random.uniform(10, 200),
                        'cpu_percent': random.uniform(5, 80),
                        'input_data_size_mb': random.uniform(1, 100),
                        'output_data_size_mb': random.uniform(0.5, 50),
                        'error_count': random.randint(0, 3),
                        'retry_count': random.randint(0, 2),
                        'metadata': {
                            'source_file': f'file_{random.randint(1, 100)}.py' if random.choice([True, False]) else None,
                            'test_count': random.randint(5, 50) if task_type == 'test_generation' else None,
                            'lines_analyzed': random.randint(100, 5000) if task_type == 'code_analysis' else None
                        }
                    })
                
                # Sort by start time (most recent first)
                tasks.sort(key=lambda x: x['start_time'], reverse=True)
                
                # Calculate summary statistics
                total_tasks = len(tasks)
                running_tasks = len([t for t in tasks if t['status'] == 'running'])
                waiting_tasks = len([t for t in tasks if t['status'] == 'waiting'])
                avg_progress = sum(t['progress_percent'] for t in tasks) / total_tasks if total_tasks > 0 else 0
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'tasks': tasks,
                    'summary': {
                        'total_active_tasks': total_tasks,
                        'running_tasks': running_tasks,
                        'waiting_tasks': waiting_tasks,
                        'suspended_tasks': total_tasks - running_tasks - waiting_tasks,
                        'average_progress': round(avg_progress, 1),
                        'total_memory_mb': sum(t['memory_mb'] for t in tasks),
                        'total_cpu_percent': sum(t['cpu_percent'] for t in tasks),
                        'high_priority_tasks': len([t for t in tasks if t['priority'] in ['high', 'critical']])
                    },
                    'charts': {
                        'task_status_distribution': self._count_by_status(tasks),
                        'task_type_breakdown': self._count_by_type(tasks),
                        'priority_distribution': self._count_by_priority(tasks),
                        'worker_utilization': self._calculate_worker_utilization(tasks),
                        'progress_histogram': self._create_progress_histogram(tasks),
                        'resource_usage_by_task': [
                            {
                                'task_id': t['task_id'][:8],
                                'task_type': t['task_type'],
                                'memory_mb': t['memory_mb'],
                                'cpu_percent': t['cpu_percent'],
                                'progress': t['progress_percent']
                            }
                            for t in sorted(tasks, key=lambda x: x['memory_mb'] + x['cpu_percent'], reverse=True)[:10]
                        ]
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Active tasks failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.blueprint.route('/queues/status', methods=['GET'])
        def queue_status():
            """Get async queue status and metrics."""
            try:
                # Generate queue data
                queues = []
                queue_names = [
                    'test_generation_queue', 'analysis_queue', 'security_scan_queue',
                    'coverage_queue', 'export_queue', 'backup_queue', 'notification_queue'
                ]
                
                for queue_name in queue_names:
                    pending_count = random.randint(0, 100)
                    processing_count = random.randint(0, 10)
                    completed_today = random.randint(50, 500)
                    failed_today = random.randint(0, 20)
                    
                    queues.append({
                        'queue_name': queue_name,
                        'pending_tasks': pending_count,
                        'processing_tasks': processing_count,
                        'completed_today': completed_today,
                        'failed_today': failed_today,
                        'average_wait_time_sec': random.uniform(5, 120),
                        'average_processing_time_sec': random.uniform(30, 300),
                        'throughput_per_hour': random.randint(10, 200),
                        'worker_count': random.randint(2, 8),
                        'max_queue_size': 1000,
                        'queue_utilization': round((pending_count / 1000) * 100, 1),
                        'success_rate': round((completed_today / (completed_today + failed_today)) * 100, 1) if (completed_today + failed_today) > 0 else 100,
                        'last_processed': (datetime.now() - timedelta(minutes=random.randint(0, 60))).isoformat()
                    })
                
                # Dead letter queue
                dlq_data = {
                    'dead_letter_count': random.randint(0, 15),
                    'retry_attempts': random.randint(50, 200),
                    'permanent_failures': random.randint(0, 5),
                    'oldest_message_age_hours': random.uniform(1, 48)
                }
                
                # Queue performance history
                performance_history = []
                for i in range(24):  # Last 24 hours
                    hour = datetime.now() - timedelta(hours=23-i)
                    performance_history.append({
                        'timestamp': hour.isoformat(),
                        'total_processed': random.randint(20, 150),
                        'average_wait_time': random.uniform(10, 80),
                        'queue_depth': random.randint(0, 200),
                        'worker_efficiency': random.uniform(0.7, 0.95)
                    })
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'queues': queues,
                    'dead_letter_queue': dlq_data,
                    'performance_history': performance_history,
                    'overall_metrics': {
                        'total_pending': sum(q['pending_tasks'] for q in queues),
                        'total_processing': sum(q['processing_tasks'] for q in queues),
                        'total_completed_today': sum(q['completed_today'] for q in queues),
                        'total_failed_today': sum(q['failed_today'] for q in queues),
                        'average_queue_utilization': sum(q['queue_utilization'] for q in queues) / len(queues),
                        'overall_success_rate': round(
                            (sum(q['completed_today'] for q in queues) / 
                             (sum(q['completed_today'] for q in queues) + sum(q['failed_today'] for q in queues))) * 100, 1
                        ) if sum(q['completed_today'] for q in queues) + sum(q['failed_today'] for q in queues) > 0 else 100
                    },
                    'charts': {
                        'queue_depth_comparison': [
                            {
                                'queue': q['queue_name'].replace('_', ' ').title(),
                                'pending': q['pending_tasks'],
                                'processing': q['processing_tasks'],
                                'utilization': q['queue_utilization']
                            }
                            for q in queues
                        ],
                        'throughput_analysis': [
                            {
                                'queue': q['queue_name'].replace('_', ' ').title(),
                                'throughput_per_hour': q['throughput_per_hour'],
                                'success_rate': q['success_rate'],
                                'worker_count': q['worker_count']
                            }
                            for q in queues
                        ],
                        'performance_timeline': performance_history[-12:],  # Last 12 hours
                        'wait_time_distribution': self._create_wait_time_distribution(queues)
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Queue status failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.blueprint.route('/workers/status', methods=['GET'])
        def worker_status():
            """Get worker process status and health."""
            try:
                # Generate worker data
                workers = []
                for i in range(random.randint(4, 12)):
                    worker_id = f'worker_{i+1}'
                    is_active = random.choice([True, True, True, False])  # 75% active
                    
                    if is_active:
                        current_task = {
                            'task_id': str(uuid.uuid4())[:8],
                            'task_type': random.choice([
                                'test_generation', 'code_analysis', 'security_scan'
                            ]),
                            'start_time': (datetime.now() - timedelta(minutes=random.randint(1, 30))).isoformat(),
                            'progress': random.uniform(0.1, 0.9)
                        }
                    else:
                        current_task = None
                    
                    start_time = datetime.now() - timedelta(hours=random.randint(1, 48))
                    uptime_hours = (datetime.now() - start_time).total_seconds() / 3600
                    
                    workers.append({
                        'worker_id': worker_id,
                        'status': 'active' if is_active else random.choice(['idle', 'maintenance', 'error']),
                        'current_task': current_task,
                        'tasks_completed_today': random.randint(20, 200),
                        'tasks_failed_today': random.randint(0, 10),
                        'cpu_usage_percent': random.uniform(5, 85) if is_active else random.uniform(0, 15),
                        'memory_usage_mb': random.uniform(50, 300),
                        'uptime_hours': round(uptime_hours, 1),
                        'last_heartbeat': (datetime.now() - timedelta(seconds=random.randint(0, 30))).isoformat(),
                        'error_count': random.randint(0, 5),
                        'restart_count': random.randint(0, 3),
                        'version': f'v{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}',
                        'queue_assignments': random.sample([
                            'test_generation_queue', 'analysis_queue', 'security_scan_queue'
                        ], random.randint(1, 3)),
                        'performance_metrics': {
                            'avg_task_completion_time_sec': random.uniform(30, 180),
                            'task_success_rate': random.uniform(0.85, 0.99),
                            'memory_efficiency': random.uniform(0.7, 0.95),
                            'cpu_efficiency': random.uniform(0.6, 0.9)
                        }
                    })
                
                # Worker pool statistics
                active_workers = [w for w in workers if w['status'] == 'active']
                idle_workers = [w for w in workers if w['status'] == 'idle']
                
                # Scaling metrics
                scaling_metrics = {
                    'target_worker_count': len(workers),
                    'optimal_worker_count': random.randint(6, 10),
                    'scaling_recommendation': random.choice(['scale_up', 'maintain', 'scale_down']),
                    'cpu_threshold_percent': 80,
                    'memory_threshold_mb': 400,
                    'queue_depth_threshold': 50,
                    'auto_scaling_enabled': True,
                    'last_scaling_event': (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat()
                }
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'workers': workers,
                    'worker_pool_summary': {
                        'total_workers': len(workers),
                        'active_workers': len(active_workers),
                        'idle_workers': len(idle_workers),
                        'maintenance_workers': len([w for w in workers if w['status'] == 'maintenance']),
                        'error_workers': len([w for w in workers if w['status'] == 'error']),
                        'total_tasks_completed_today': sum(w['tasks_completed_today'] for w in workers),
                        'total_tasks_failed_today': sum(w['tasks_failed_today'] for w in workers),
                        'average_cpu_usage': sum(w['cpu_usage_percent'] for w in workers) / len(workers),
                        'average_memory_usage': sum(w['memory_usage_mb'] for w in workers) / len(workers),
                        'pool_efficiency': sum(w['performance_metrics']['cpu_efficiency'] for w in workers) / len(workers)
                    },
                    'scaling_metrics': scaling_metrics,
                    'charts': {
                        'worker_status_distribution': self._count_worker_status(workers),
                        'worker_performance_matrix': [
                            {
                                'worker_id': w['worker_id'],
                                'cpu_usage': w['cpu_usage_percent'],
                                'memory_usage': w['memory_usage_mb'],
                                'tasks_completed': w['tasks_completed_today'],
                                'success_rate': w['performance_metrics']['task_success_rate'] * 100,
                                'efficiency': w['performance_metrics']['cpu_efficiency'] * 100
                            }
                            for w in workers
                        ],
                        'resource_utilization_timeline': self._generate_resource_timeline(),
                        'task_completion_rates': [
                            {
                                'worker_id': w['worker_id'],
                                'completion_rate': w['tasks_completed_today'] / max(1, uptime_hours) if (uptime_hours := w['uptime_hours']) > 0 else 0,
                                'success_rate': w['performance_metrics']['task_success_rate'],
                                'avg_completion_time': w['performance_metrics']['avg_task_completion_time_sec']
                            }
                            for w in workers if w['status'] == 'active'
                        ]
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Worker status failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.blueprint.route('/pipelines/flow', methods=['GET'])
        def pipeline_flow():
            """Get async pipeline flow and dependencies."""
            try:
                # Generate pipeline stages
                pipelines = []
                pipeline_names = [
                    'test_generation_pipeline', 'code_analysis_pipeline',
                    'security_assessment_pipeline', 'quality_assurance_pipeline',
                    'deployment_pipeline', 'monitoring_pipeline'
                ]
                
                for pipeline_name in pipeline_names:
                    stages = []
                    stage_count = random.randint(3, 6)
                    
                    for stage_idx in range(stage_count):
                        stage_name = f'stage_{stage_idx + 1}'
                        stage_status = random.choice(['completed', 'running', 'pending', 'failed'])
                        
                        stages.append({
                            'stage_id': f'{pipeline_name}_{stage_name}',
                            'stage_name': stage_name,
                            'stage_type': random.choice([
                                'validation', 'processing', 'analysis', 'notification', 'cleanup'
                            ]),
                            'status': stage_status,
                            'duration_sec': random.randint(10, 300),
                            'start_time': (datetime.now() - timedelta(minutes=random.randint(0, 60))).isoformat(),
                            'dependencies': [f'{pipeline_name}_stage_{stage_idx}'] if stage_idx > 0 else [],
                            'parallel_execution': random.choice([True, False]),
                            'retry_count': random.randint(0, 2),
                            'success_rate': random.uniform(0.85, 0.99),
                            'resource_requirements': {
                                'cpu_cores': random.randint(1, 4),
                                'memory_mb': random.randint(100, 1000),
                                'disk_space_mb': random.randint(50, 500)
                            }
                        })
                    
                    # Calculate pipeline metrics
                    total_duration = sum(s['duration_sec'] for s in stages)
                    completed_stages = len([s for s in stages if s['status'] == 'completed'])
                    
                    pipelines.append({
                        'pipeline_id': pipeline_name,
                        'pipeline_name': pipeline_name.replace('_', ' ').title(),
                        'status': random.choice(['running', 'completed', 'failed', 'paused']),
                        'stages': stages,
                        'total_stages': len(stages),
                        'completed_stages': completed_stages,
                        'progress_percent': round((completed_stages / len(stages)) * 100, 1),
                        'total_duration_sec': total_duration,
                        'estimated_completion': (datetime.now() + timedelta(seconds=random.randint(0, 600))).isoformat(),
                        'priority': random.choice(['low', 'normal', 'high']),
                        'trigger_type': random.choice(['manual', 'scheduled', 'event_driven', 'webhook']),
                        'execution_count_today': random.randint(1, 20),
                        'success_rate_today': random.uniform(0.8, 1.0)
                    })
                
                # Pipeline dependencies graph
                dependency_graph = {
                    'nodes': [],
                    'edges': []
                }
                
                # Add nodes (stages)
                for pipeline in pipelines:
                    for stage in pipeline['stages']:
                        dependency_graph['nodes'].append({
                            'id': stage['stage_id'],
                            'label': f"{pipeline['pipeline_name']} - {stage['stage_name'].title()}",
                            'status': stage['status'],
                            'pipeline': pipeline['pipeline_id'],
                            'type': stage['stage_type']
                        })
                        
                        # Add edges (dependencies)
                        for dep in stage['dependencies']:
                            dependency_graph['edges'].append({
                                'from': dep,
                                'to': stage['stage_id'],
                                'type': 'dependency'
                            })
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'pipelines': pipelines,
                    'dependency_graph': dependency_graph,
                    'pipeline_summary': {
                        'total_pipelines': len(pipelines),
                        'running_pipelines': len([p for p in pipelines if p['status'] == 'running']),
                        'completed_pipelines': len([p for p in pipelines if p['status'] == 'completed']),
                        'failed_pipelines': len([p for p in pipelines if p['status'] == 'failed']),
                        'average_completion_rate': sum(p['progress_percent'] for p in pipelines) / len(pipelines),
                        'total_executions_today': sum(p['execution_count_today'] for p in pipelines),
                        'overall_success_rate': sum(p['success_rate_today'] for p in pipelines) / len(pipelines)
                    },
                    'charts': {
                        'pipeline_status_overview': self._count_pipeline_status(pipelines),
                        'stage_completion_matrix': self._create_stage_matrix(pipelines),
                        'execution_timeline': self._generate_execution_timeline(),
                        'resource_allocation': self._calculate_pipeline_resources(pipelines),
                        'dependency_complexity': {
                            'total_stages': len(dependency_graph['nodes']),
                            'total_dependencies': len(dependency_graph['edges']),
                            'avg_dependencies_per_stage': len(dependency_graph['edges']) / max(1, len(dependency_graph['nodes'])),
                            'parallel_stages': len([n for n in dependency_graph['nodes'] if len([e for e in dependency_graph['edges'] if e['to'] == n['id']]) == 0])
                        }
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Pipeline flow failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def _count_by_status(self, tasks):
        """Count tasks by status."""
        counts = {}
        for task in tasks:
            status = task['status']
            counts[status] = counts.get(status, 0) + 1
        return [{'status': k, 'count': v} for k, v in counts.items()]
    
    def _count_by_type(self, tasks):
        """Count tasks by type."""
        counts = {}
        for task in tasks:
            task_type = task['task_type']
            counts[task_type] = counts.get(task_type, 0) + 1
        return [{'type': k, 'count': v} for k, v in counts.items()]
    
    def _count_by_priority(self, tasks):
        """Count tasks by priority."""
        counts = {}
        for task in tasks:
            priority = task['priority']
            counts[priority] = counts.get(priority, 0) + 1
        return [{'priority': k, 'count': v} for k, v in counts.items()]
    
    def _calculate_worker_utilization(self, tasks):
        """Calculate worker utilization."""
        worker_usage = {}
        for task in tasks:
            worker_id = task['worker_id']
            if worker_id not in worker_usage:
                worker_usage[worker_id] = {'task_count': 0, 'cpu_total': 0, 'memory_total': 0}
            worker_usage[worker_id]['task_count'] += 1
            worker_usage[worker_id]['cpu_total'] += task['cpu_percent']
            worker_usage[worker_id]['memory_total'] += task['memory_mb']
        
        return [
            {
                'worker_id': k,
                'task_count': v['task_count'],
                'avg_cpu': round(v['cpu_total'] / v['task_count'], 1),
                'total_memory': round(v['memory_total'], 1)
            }
            for k, v in worker_usage.items()
        ]
    
    def _create_progress_histogram(self, tasks):
        """Create progress histogram."""
        buckets = {'0-25%': 0, '25-50%': 0, '50-75%': 0, '75-100%': 0}
        for task in tasks:
            progress = task['progress_percent']
            if progress < 25:
                buckets['0-25%'] += 1
            elif progress < 50:
                buckets['25-50%'] += 1
            elif progress < 75:
                buckets['50-75%'] += 1
            else:
                buckets['75-100%'] += 1
        
        return [{'range': k, 'count': v} for k, v in buckets.items()]
    
    def _create_wait_time_distribution(self, queues):
        """Create wait time distribution."""
        return [
            {
                'queue': q['queue_name'].replace('_', ' ').title(),
                'avg_wait_time': q['average_wait_time_sec'],
                'avg_processing_time': q['average_processing_time_sec'],
                'efficiency_ratio': round(q['average_processing_time_sec'] / (q['average_wait_time_sec'] + q['average_processing_time_sec']), 2)
            }
            for q in queues
        ]
    
    def _count_worker_status(self, workers):
        """Count workers by status."""
        counts = {}
        for worker in workers:
            status = worker['status']
            counts[status] = counts.get(status, 0) + 1
        return [{'status': k, 'count': v} for k, v in counts.items()]
    
    def _generate_resource_timeline(self):
        """Generate resource utilization timeline."""
        timeline = []
        for i in range(12):  # Last 12 hours
            hour = datetime.now() - timedelta(hours=11-i)
            timeline.append({
                'timestamp': hour.isoformat(),
                'cpu_usage': random.uniform(20, 80),
                'memory_usage': random.uniform(30, 70),
                'active_workers': random.randint(4, 8),
                'task_throughput': random.randint(50, 200)
            })
        return timeline
    
    def _count_pipeline_status(self, pipelines):
        """Count pipelines by status."""
        counts = {}
        for pipeline in pipelines:
            status = pipeline['status']
            counts[status] = counts.get(status, 0) + 1
        return [{'status': k, 'count': v} for k, v in counts.items()]
    
    def _create_stage_matrix(self, pipelines):
        """Create stage completion matrix."""
        matrix = []
        for pipeline in pipelines:
            completed = len([s for s in pipeline['stages'] if s['status'] == 'completed'])
            running = len([s for s in pipeline['stages'] if s['status'] == 'running'])
            pending = len([s for s in pipeline['stages'] if s['status'] == 'pending'])
            failed = len([s for s in pipeline['stages'] if s['status'] == 'failed'])
            
            matrix.append({
                'pipeline': pipeline['pipeline_name'],
                'completed': completed,
                'running': running,
                'pending': pending,
                'failed': failed,
                'total': len(pipeline['stages'])
            })
        return matrix
    
    def _generate_execution_timeline(self):
        """Generate pipeline execution timeline."""
        timeline = []
        for i in range(7):  # Last 7 days
            day = datetime.now() - timedelta(days=6-i)
            timeline.append({
                'date': day.strftime('%Y-%m-%d'),
                'total_executions': random.randint(10, 50),
                'successful_executions': random.randint(8, 45),
                'failed_executions': random.randint(0, 5),
                'avg_execution_time_min': random.uniform(5, 30)
            })
        return timeline
    
    def _calculate_pipeline_resources(self, pipelines):
        """Calculate resource allocation for pipelines."""
        resources = []
        for pipeline in pipelines:
            total_cpu = sum(s['resource_requirements']['cpu_cores'] for s in pipeline['stages'])
            total_memory = sum(s['resource_requirements']['memory_mb'] for s in pipeline['stages'])
            total_disk = sum(s['resource_requirements']['disk_space_mb'] for s in pipeline['stages'])
            
            resources.append({
                'pipeline': pipeline['pipeline_name'],
                'cpu_cores': total_cpu,
                'memory_mb': total_memory,
                'disk_mb': total_disk,
                'resource_intensity': total_cpu * 0.4 + (total_memory / 1000) * 0.4 + (total_disk / 1000) * 0.2
            })
        
        return resources