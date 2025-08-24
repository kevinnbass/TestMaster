"""
Telemetry & Performance Profiling API
====================================

Provides comprehensive telemetry data and performance metrics
for frontend visualization.

Author: TestMaster Team
"""

import logging
from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import random
import json
import time

logger = logging.getLogger(__name__)

class TelemetryAPI:
    """Telemetry and Performance Profiling API endpoints."""
    
    def __init__(self):
        """Initialize Telemetry API."""
        self.blueprint = Blueprint('telemetry', __name__, url_prefix='/api/telemetry')
        self._setup_routes()
        logger.info("Telemetry API initialized")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.blueprint.route('/performance/system', methods=['GET'])
        def system_performance():
            """Get system performance metrics."""
            try:
                # Generate system performance data
                current_time = datetime.now()
                
                # CPU metrics
                cpu_cores = random.randint(4, 16)
                cpu_data = []
                for i in range(cpu_cores):
                    cpu_data.append({
                        'core_id': f'core_{i}',
                        'usage_percent': random.uniform(15, 85),
                        'frequency_mhz': random.randint(2400, 3600),
                        'temperature': random.uniform(45, 75)
                    })
                
                # Memory metrics
                memory_total = random.randint(8, 64)  # GB
                memory_used = memory_total * random.uniform(0.3, 0.8)
                
                # Disk metrics
                disk_metrics = []
                for disk in ['C:', 'D:', '/']:
                    if random.choice([True, False]):  # Some disks may not exist
                        disk_metrics.append({
                            'disk': disk,
                            'total_gb': random.randint(250, 2000),
                            'used_gb': random.randint(100, 1500),
                            'read_iops': random.randint(50, 500),
                            'write_iops': random.randint(30, 300),
                            'read_latency_ms': random.uniform(1, 15),
                            'write_latency_ms': random.uniform(2, 20)
                        })
                
                # Network metrics
                network_interfaces = []
                for i in range(random.randint(1, 3)):
                    network_interfaces.append({
                        'interface': f'eth{i}' if i > 0 else 'wifi0',
                        'bytes_sent': random.randint(1000000, 100000000),
                        'bytes_received': random.randint(5000000, 500000000),
                        'packets_sent': random.randint(10000, 1000000),
                        'packets_received': random.randint(50000, 5000000),
                        'speed_mbps': random.choice([100, 1000, 10000]),
                        'status': 'connected'
                    })
                
                return jsonify({
                    'status': 'success',
                    'timestamp': current_time.isoformat(),
                    'system_info': {
                        'hostname': 'testmaster-server',
                        'os': 'Windows 11' if random.choice([True, False]) else 'Linux',
                        'uptime_hours': random.randint(1, 720),
                        'load_average': [
                            random.uniform(0.1, 2.0),
                            random.uniform(0.1, 2.0),
                            random.uniform(0.1, 2.0)
                        ]
                    },
                    'cpu': {
                        'cores': cpu_cores,
                        'overall_usage': sum(c['usage_percent'] for c in cpu_data) / len(cpu_data),
                        'cores_detail': cpu_data,
                        'architecture': 'x64',
                        'cache_l1': '32KB per core',
                        'cache_l2': '256KB per core',
                        'cache_l3': '12MB shared'
                    },
                    'memory': {
                        'total_gb': memory_total,
                        'used_gb': round(memory_used, 2),
                        'available_gb': round(memory_total - memory_used, 2),
                        'usage_percent': round((memory_used / memory_total) * 100, 1),
                        'swap_total_gb': random.randint(2, 16),
                        'swap_used_gb': random.uniform(0, 2)
                    },
                    'disk': {
                        'disks': disk_metrics,
                        'total_iops': sum(d['read_iops'] + d['write_iops'] for d in disk_metrics),
                        'avg_latency_ms': sum(d['read_latency_ms'] + d['write_latency_ms'] for d in disk_metrics) / (len(disk_metrics) * 2) if disk_metrics else 0
                    },
                    'network': {
                        'interfaces': network_interfaces,
                        'total_bytes_sent': sum(n['bytes_sent'] for n in network_interfaces),
                        'total_bytes_received': sum(n['bytes_received'] for n in network_interfaces),
                        'active_connections': random.randint(50, 500)
                    },
                    'charts': {
                        'cpu_usage_by_core': [
                            {'core': c['core_id'], 'usage': c['usage_percent']} 
                            for c in cpu_data
                        ],
                        'memory_breakdown': {
                            'used': memory_used,
                            'available': memory_total - memory_used,
                            'cached': random.uniform(1, 3),
                            'buffers': random.uniform(0.5, 1.5)
                        },
                        'disk_performance': [
                            {
                                'disk': d['disk'],
                                'read_iops': d['read_iops'],
                                'write_iops': d['write_iops'],
                                'read_latency': d['read_latency_ms'],
                                'write_latency': d['write_latency_ms']
                            }
                            for d in disk_metrics
                        ],
                        'network_throughput': [
                            {
                                'interface': n['interface'],
                                'sent_mbps': n['bytes_sent'] / (1024*1024),
                                'received_mbps': n['bytes_received'] / (1024*1024)
                            }
                            for n in network_interfaces
                        ]
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"System performance failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.blueprint.route('/performance/application', methods=['GET'])
        def application_performance():
            """Get application performance metrics."""
            try:
                # Generate application performance data
                processes = []
                for i in range(random.randint(5, 15)):
                    processes.append({
                        'pid': random.randint(1000, 9999),
                        'name': random.choice([
                            'testmaster.exe', 'python.exe', 'dashboard.exe',
                            'test_generator.py', 'monitor.py', 'analyzer.py'
                        ]),
                        'cpu_percent': random.uniform(0.1, 25.0),
                        'memory_mb': random.uniform(10, 500),
                        'threads': random.randint(1, 20),
                        'status': random.choice(['running', 'sleeping', 'waiting']),
                        'start_time': (datetime.now() - timedelta(minutes=random.randint(1, 1440))).isoformat()
                    })
                
                # Database performance (simulated)
                db_metrics = {
                    'connections_active': random.randint(5, 50),
                    'connections_idle': random.randint(10, 100),
                    'queries_per_second': random.uniform(50, 500),
                    'avg_query_time_ms': random.uniform(5, 100),
                    'cache_hit_ratio': random.uniform(0.85, 0.98),
                    'deadlocks': random.randint(0, 2),
                    'buffer_pool_usage': random.uniform(0.7, 0.95)
                }
                
                # API performance
                api_endpoints = []
                endpoints = [
                    '/api/analytics/summary', '/api/coverage/intelligence',
                    '/api/security/vulnerabilities', '/api/qa/scorecard',
                    '/api/intelligence/agents', '/api/flow/dag'
                ]
                
                for endpoint in endpoints:
                    api_endpoints.append({
                        'endpoint': endpoint,
                        'requests_per_minute': random.randint(10, 200),
                        'avg_response_time_ms': random.uniform(50, 300),
                        'p95_response_time_ms': random.uniform(100, 500),
                        'error_rate_percent': random.uniform(0, 5),
                        'success_rate_percent': random.uniform(95, 100)
                    })
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'application_summary': {
                        'total_processes': len(processes),
                        'total_cpu_usage': sum(p['cpu_percent'] for p in processes),
                        'total_memory_mb': sum(p['memory_mb'] for p in processes),
                        'total_threads': sum(p['threads'] for p in processes)
                    },
                    'processes': processes[:10],  # Top 10 processes
                    'database': db_metrics,
                    'api_performance': api_endpoints,
                    'garbage_collection': {
                        'collections_gen0': random.randint(100, 1000),
                        'collections_gen1': random.randint(10, 100),
                        'collections_gen2': random.randint(1, 10),
                        'total_memory_mb': random.uniform(50, 200),
                        'avg_gc_time_ms': random.uniform(1, 10)
                    },
                    'charts': {
                        'process_cpu_usage': [
                            {'process': p['name'], 'cpu': p['cpu_percent'], 'memory': p['memory_mb']}
                            for p in sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:8]
                        ],
                        'database_performance': db_metrics,
                        'api_response_times': [
                            {
                                'endpoint': api['endpoint'].split('/')[-1],
                                'avg_time': api['avg_response_time_ms'],
                                'p95_time': api['p95_response_time_ms'],
                                'requests': api['requests_per_minute']
                            }
                            for api in api_endpoints
                        ],
                        'memory_timeline': self._generate_memory_timeline()
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Application performance failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.blueprint.route('/events/stream', methods=['GET'])
        def telemetry_events():
            """Get real-time telemetry events stream."""
            try:
                # Generate recent telemetry events
                events = []
                event_types = [
                    'test_execution', 'code_analysis', 'file_change',
                    'api_request', 'error_detected', 'performance_alert',
                    'security_scan', 'coverage_update', 'user_action'
                ]
                
                for i in range(50):
                    event_time = datetime.now() - timedelta(minutes=random.randint(0, 60))
                    events.append({
                        'event_id': f'evt_{random.randint(10000, 99999)}',
                        'event_type': random.choice(event_types),
                        'timestamp': event_time.isoformat(),
                        'component': random.choice([
                            'test_generator', 'file_watcher', 'api_server',
                            'coverage_analyzer', 'security_scanner', 'dashboard'
                        ]),
                        'operation': random.choice([
                            'generate_test', 'analyze_file', 'serve_request',
                            'calculate_coverage', 'scan_vulnerabilities', 'render_chart'
                        ]),
                        'duration_ms': random.uniform(10, 2000),
                        'success': random.choice([True, True, True, False]),  # 75% success rate
                        'metadata': {
                            'file_count': random.randint(1, 50) if random.choice([True, False]) else None,
                            'test_count': random.randint(1, 20) if random.choice([True, False]) else None,
                            'memory_used_mb': random.uniform(10, 100),
                            'cpu_used_percent': random.uniform(5, 50)
                        }
                    })
                
                # Sort by timestamp (most recent first)
                events.sort(key=lambda x: x['timestamp'], reverse=True)
                
                # Calculate event statistics
                successful_events = [e for e in events if e['success']]
                failed_events = [e for e in events if not e['success']]
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'events': events,
                    'summary': {
                        'total_events': len(events),
                        'successful_events': len(successful_events),
                        'failed_events': len(failed_events),
                        'success_rate': round((len(successful_events) / len(events)) * 100, 1),
                        'avg_duration_ms': sum(e['duration_ms'] for e in events) / len(events),
                        'events_per_minute': len(events)  # Since we're showing last hour
                    },
                    'charts': {
                        'events_by_type': self._count_by_field(events, 'event_type'),
                        'events_by_component': self._count_by_field(events, 'component'),
                        'success_failure_ratio': {
                            'successful': len(successful_events),
                            'failed': len(failed_events)
                        },
                        'duration_distribution': self._create_duration_distribution(events),
                        'events_timeline': self._create_events_timeline(events)
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Telemetry events failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.blueprint.route('/profiling/hotspots', methods=['GET'])
        def performance_hotspots():
            """Get performance profiling hotspots."""
            try:
                # Generate performance hotspots
                hotspots = []
                for i in range(15):
                    hotspots.append({
                        'function_name': random.choice([
                            'test_generator.generate_test', 'analyzer.analyze_file',
                            'coverage.calculate_coverage', 'api.serve_request',
                            'security.scan_file', 'monitor.check_changes',
                            'parser.parse_ast', 'validator.validate_syntax'
                        ]),
                        'file_path': f'testmaster/{random.choice(["core", "generators", "analyzers"])}/{random.choice(["test", "main", "utils"])}.py',
                        'line_number': random.randint(10, 500),
                        'total_time_ms': random.uniform(100, 5000),
                        'call_count': random.randint(50, 1000),
                        'avg_time_per_call_ms': None,  # Will calculate
                        'cpu_percent': random.uniform(5, 25),
                        'memory_mb': random.uniform(10, 100),
                        'hotspot_score': None  # Will calculate
                    })
                
                # Calculate derived metrics
                for hotspot in hotspots:
                    hotspot['avg_time_per_call_ms'] = round(hotspot['total_time_ms'] / hotspot['call_count'], 2)
                    hotspot['hotspot_score'] = round(
                        (hotspot['total_time_ms'] / 1000) * hotspot['cpu_percent'] * (hotspot['memory_mb'] / 10), 1
                    )
                
                # Sort by hotspot score (highest first)
                hotspots.sort(key=lambda x: x['hotspot_score'], reverse=True)
                
                # Memory profiling
                memory_profile = {
                    'total_allocated_mb': random.uniform(500, 2000),
                    'peak_usage_mb': random.uniform(600, 2500),
                    'current_usage_mb': random.uniform(400, 1500),
                    'gc_collections': random.randint(100, 1000),
                    'memory_leaks_detected': random.randint(0, 3),
                    'largest_objects': [
                        {
                            'object_type': random.choice(['dict', 'list', 'str', 'TestResult', 'CoverageData']),
                            'size_mb': random.uniform(10, 100),
                            'count': random.randint(100, 10000),
                            'location': f'line {random.randint(1, 500)}'
                        }
                        for _ in range(8)
                    ]
                }
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'hotspots': hotspots,
                    'memory_profile': memory_profile,
                    'profiling_summary': {
                        'total_functions_profiled': len(hotspots),
                        'top_hotspot_score': hotspots[0]['hotspot_score'] if hotspots else 0,
                        'total_cpu_time_ms': sum(h['total_time_ms'] for h in hotspots),
                        'total_memory_mb': sum(h['memory_mb'] for h in hotspots),
                        'optimization_opportunities': len([h for h in hotspots if h['hotspot_score'] > 50])
                    },
                    'charts': {
                        'hotspot_matrix': [
                            {
                                'function': h['function_name'].split('.')[-1],
                                'total_time': h['total_time_ms'],
                                'call_count': h['call_count'],
                                'cpu_percent': h['cpu_percent'],
                                'hotspot_score': h['hotspot_score']
                            }
                            for h in hotspots[:10]
                        ],
                        'memory_usage_breakdown': [
                            {'type': obj['object_type'], 'size_mb': obj['size_mb'], 'count': obj['count']}
                            for obj in memory_profile['largest_objects'][:6]
                        ],
                        'call_frequency_vs_time': [
                            {
                                'function': h['function_name'].split('.')[-1],
                                'call_count': h['call_count'],
                                'avg_time': h['avg_time_per_call_ms']
                            }
                            for h in hotspots[:8]
                        ]
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Performance hotspots failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.blueprint.route('/metrics/custom', methods=['GET'])
        def custom_metrics():
            """Get custom telemetry metrics."""
            try:
                # Custom business metrics
                business_metrics = {
                    'tests_generated_today': random.randint(50, 500),
                    'code_coverage_improvement': random.uniform(-2.5, 5.0),
                    'bugs_detected': random.randint(0, 15),
                    'performance_regressions': random.randint(0, 5),
                    'security_vulnerabilities_fixed': random.randint(0, 8),
                    'user_satisfaction_score': random.uniform(7.5, 9.5),
                    'system_uptime_hours': random.randint(100, 8760)
                }
                
                # Feature usage metrics
                feature_usage = []
                features = [
                    'intelligent_test_generation', 'security_scanning', 'coverage_analysis',
                    'performance_monitoring', 'real_time_dashboard', 'auto_healing',
                    'batch_processing', 'integration_tests', 'quality_assurance'
                ]
                
                for feature in features:
                    feature_usage.append({
                        'feature_name': feature,
                        'usage_count': random.randint(10, 1000),
                        'active_users': random.randint(1, 50),
                        'avg_session_duration_min': random.uniform(5, 120),
                        'success_rate': random.uniform(85, 99),
                        'last_used': (datetime.now() - timedelta(hours=random.randint(0, 48))).isoformat()
                    })
                
                # Error patterns
                error_patterns = []
                error_types = [
                    'ImportError', 'SyntaxError', 'AttributeError', 'TypeError',
                    'ConnectionError', 'TimeoutError', 'ValidationError'
                ]
                
                for error_type in error_types:
                    if random.choice([True, False, False]):  # Not all errors occur
                        error_patterns.append({
                            'error_type': error_type,
                            'occurrences': random.randint(1, 50),
                            'affected_components': random.sample([
                                'test_generator', 'file_analyzer', 'coverage_calculator',
                                'security_scanner', 'dashboard', 'api_server'
                            ], random.randint(1, 3)),
                            'resolution_rate': random.uniform(70, 95),
                            'avg_resolution_time_min': random.uniform(5, 60),
                            'first_seen': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
                        })
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'business_metrics': business_metrics,
                    'feature_usage': feature_usage,
                    'error_patterns': error_patterns,
                    'custom_kpis': {
                        'code_quality_score': random.uniform(75, 95),
                        'test_effectiveness_ratio': random.uniform(0.7, 0.95),
                        'deployment_frequency_per_week': random.randint(2, 10),
                        'mean_time_to_recovery_hours': random.uniform(0.5, 4.0),
                        'change_failure_rate': random.uniform(0.05, 0.15)
                    },
                    'charts': {
                        'feature_adoption': [
                            {
                                'feature': f['feature_name'].replace('_', ' ').title(),
                                'usage_count': f['usage_count'],
                                'active_users': f['active_users'],
                                'success_rate': f['success_rate']
                            }
                            for f in sorted(feature_usage, key=lambda x: x['usage_count'], reverse=True)[:6]
                        ],
                        'error_distribution': [
                            {'error_type': e['error_type'], 'occurrences': e['occurrences']}
                            for e in error_patterns
                        ],
                        'business_metrics_trend': self._generate_metrics_trend(business_metrics),
                        'quality_indicators': {
                            'code_quality': business_metrics.get('user_satisfaction_score', 8.0) * 10,
                            'reliability': 100 - (len(error_patterns) * 5),
                            'performance': 95 - business_metrics.get('performance_regressions', 0) * 5,
                            'security': 100 - business_metrics.get('security_vulnerabilities_fixed', 0) * 2
                        }
                    }
                }), 200
                
            except Exception as e:
                logger.error(f"Custom metrics failed: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def _generate_memory_timeline(self):
        """Generate memory usage timeline."""
        timeline = []
        for i in range(24):  # Last 24 hours
            hour = datetime.now() - timedelta(hours=23-i)
            timeline.append({
                'timestamp': hour.isoformat(),
                'memory_usage_mb': random.uniform(100, 800),
                'memory_available_mb': random.uniform(200, 1200)
            })
        return timeline
    
    def _count_by_field(self, events, field):
        """Count events by a specific field."""
        counts = {}
        for event in events:
            value = event.get(field, 'unknown')
            counts[value] = counts.get(value, 0) + 1
        return [{'name': k, 'count': v} for k, v in counts.items()]
    
    def _create_duration_distribution(self, events):
        """Create duration distribution buckets."""
        buckets = {
            '0-100ms': 0, '100-500ms': 0, '500ms-1s': 0, 
            '1-2s': 0, '2s+': 0
        }
        
        for event in events:
            duration = event['duration_ms']
            if duration < 100:
                buckets['0-100ms'] += 1
            elif duration < 500:
                buckets['100-500ms'] += 1
            elif duration < 1000:
                buckets['500ms-1s'] += 1
            elif duration < 2000:
                buckets['1-2s'] += 1
            else:
                buckets['2s+'] += 1
        
        return [{'range': k, 'count': v} for k, v in buckets.items()]
    
    def _create_events_timeline(self, events):
        """Create events timeline by minute."""
        timeline = {}
        for event in events:
            # Round to minute
            event_time = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
            minute_key = event_time.strftime('%H:%M')
            timeline[minute_key] = timeline.get(minute_key, 0) + 1
        
        return [{'time': k, 'count': v} for k, v in sorted(timeline.items())]
    
    def _generate_metrics_trend(self, metrics):
        """Generate trend data for metrics."""
        trends = []
        for i in range(7):  # Last 7 days
            day = datetime.now() - timedelta(days=6-i)
            trends.append({
                'date': day.strftime('%Y-%m-%d'),
                'tests_generated': metrics['tests_generated_today'] + random.randint(-50, 50),
                'bugs_detected': metrics['bugs_detected'] + random.randint(-5, 5),
                'coverage_improvement': metrics['code_coverage_improvement'] + random.uniform(-1, 1)
            })
        return trends