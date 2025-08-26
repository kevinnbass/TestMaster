"""
Enterprise ML API - TestMaster Advanced ML
==========================================
"""API Endpoints Module - Split from ml_api.py"""


import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import asdict

from flask import Blueprint, request, jsonify, current_app
from flask_restx import Api, Resource, fields, Namespace
from marshmallow import Schema, fields as ma_fields, ValidationError

# ML Module Integration
from ..ml.ml_orchestrator import get_ml_orchestration_status, get_ml_integration_insights
from ..ml.integration_analysis import get_integration_analysis

# Enterprise ML Infrastructure Integration

    
    def get_monitoring_dashboard():
        global _monitoring_dashboard
        if _monitoring_dashboard is None:
            _monitoring_dashboard = MLMonitoringDashboard()
        return _monitoring_dashboard
    
    def get_auto_scaling_system():
        global _auto_scaling_system
        if _auto_scaling_system is None:
            _auto_scaling_system = MLAutoScaling()
        return _auto_scaling_system
    
    def get_infrastructure_orchestrator():
        global _infrastructure_orchestrator
        if _infrastructure_orchestrator is None:
            _infrastructure_orchestrator = MLInfrastructureOrchestrator()
        return _infrastructure_orchestrator

    # Enterprise Infrastructure Models
    infrastructure_status_model = api.model('InfrastructureStatus', {
        'cluster_overview': fields.Raw(description='Cluster resource overview'),
        'services': fields.Raw(description='Service deployment status'),
        'deployments': fields.Raw(description='Deployment history and metrics'),
        'cost_analysis': fields.Raw(description='Cost analysis and optimization'),
        'service_mesh': fields.Raw(description='Service mesh configuration'),
        'disaster_recovery': fields.Raw(description='Disaster recovery status')
    })
    
    scaling_summary_model = api.model('ScalingSummary', {
        'system_overview': fields.Raw(description='System-wide scaling metrics'),
        'recent_activity': fields.Raw(description='Recent scaling events'),
        'resource_pools': fields.Raw(description='Resource pool utilization'),
        'predictions': fields.Raw(description='Scaling predictions'),
        'cost_optimization': fields.Raw(description='Cost optimization metrics')
    })
    
    monitoring_summary_model = api.model('MonitoringSummary', {
        'overall_health': fields.Float(description='Overall system health score'),
        'module_status': fields.Raw(description='Individual module health'),
        'recent_alerts': fields.List(fields.Raw, description='Recent system alerts'),
        'performance_metrics': fields.Raw(description='Performance metrics'),
        'predictions': fields.Raw(description='Predictive analysis results')
    })

    @enterprise_ns.route('/monitoring/dashboard')
    class MonitoringDashboard(Resource):
        @api.marshal_with(monitoring_summary_model)
        def get(self):
            """Get comprehensive monitoring dashboard data"""
            try:
                dashboard = get_monitoring_dashboard()
                
                # Get current metrics and health
                health_data = dashboard._calculate_system_health()
                current_metrics = dashboard._get_current_metrics()
                recent_alerts = list(dashboard.alerts)[-10:]  # Last 10 alerts
                predictions = dashboard._get_system_predictions()
                
                summary_data = {
                    'overall_health': health_data['overall_score'],
                    'module_status': {
                        module: {
                            'health_score': dashboard.system_health.get(module, 100),
                            'status': info['status'],
                            'last_update': info['last_update'].isoformat() if isinstance(info['last_update'], datetime) else str(info['last_update'])
                        }
                        for module, info in dashboard.ml_modules.items()
                    },
                    'recent_alerts': [
                        {
                            'alert_id': alert.alert_id,
                            'severity': alert.severity,
                            'module_name': alert.module_name,
                            'message': alert.message,
                            'timestamp': alert.timestamp.isoformat(),
                            'auto_resolved': alert.auto_resolved
                        }
                        for alert in recent_alerts
                    ],
                    'performance_metrics': current_metrics,
                    'predictions': predictions
                }
                
                return {
                    'success': True,
                    'data': summary_data,
                    'timestamp': datetime.now().isoformat()
                }, 200
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message': 'Failed to retrieve monitoring dashboard data'
                }, 500

    @enterprise_ns.route('/monitoring/console')
    class MonitoringConsole(Resource):
        def get(self):
            """Get console-friendly monitoring summary"""
            try:
                dashboard = get_monitoring_dashboard()
                console_summary = dashboard.get_console_summary()
                
                return {
                    'success': True,
                    'data': {
                        'console_output': console_summary,
                        'format': 'text/plain'
                    },
                    'timestamp': datetime.now().isoformat()
                }, 200
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message': 'Failed to retrieve console monitoring summary'
                }, 500

    @enterprise_ns.route('/scaling/summary')
    class AutoScalingSummary(Resource):
        @api.marshal_with(scaling_summary_model)
        def get(self):
            """Get comprehensive auto-scaling system summary"""
            try:
                scaler = get_auto_scaling_system()
                summary = scaler.get_scaling_summary()
                
                return {
                    'success': True,
                    'data': summary,
                    'timestamp': datetime.now().isoformat()
                }, 200
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message': 'Failed to retrieve auto-scaling summary'
                }, 500

    @enterprise_ns.route('/scaling/metrics/<module_name>')
    class ModuleScalingMetrics(Resource):
        def get(self, module_name):
            """Get scaling metrics for a specific module"""
            try:
                scaler = get_auto_scaling_system()
                
                if module_name not in scaler.ml_modules:
                    return {
                        'success': False,
                        'error': 'Module not found',
                        'message': f'ML module {module_name} not found in scaling system'
                    }, 404
                
                current_metrics = scaler.current_metrics.get(module_name)
                module_config = scaler.ml_modules[module_name]
                
                if current_metrics is None:
                    return {
                        'success': False,
                        'error': 'No metrics available',
                        'message': f'No current metrics available for {module_name}'
                    }, 404
                
                return {
                    'success': True,
                    'data': {
                        'module_name': module_name,
                        'current_metrics': asdict(current_metrics),
                        'module_config': module_config,
                        'scaling_recommendation': scaler._make_scaling_decision(module_name)
                    },
                    'timestamp': datetime.now().isoformat()
                }, 200
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message': f'Failed to retrieve metrics for module {module_name}'
                }, 500

    @enterprise_ns.route('/infrastructure/status')
    class InfrastructureStatus(Resource):
        @api.marshal_with(infrastructure_status_model)
        def get(self):
            """Get comprehensive infrastructure status"""
            try:
                orchestrator = get_infrastructure_orchestrator()
                status = orchestrator.get_infrastructure_status()
                
                return {
                    'success': True,
                    'data': status,
                    'timestamp': datetime.now().isoformat()
                }, 200
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message': 'Failed to retrieve infrastructure status'
                }, 500

    @enterprise_ns.route('/infrastructure/deploy')
    class ServiceDeployment(Resource):
        def post(self):
            """Deploy or update a service"""
            try:
                data = request.get_json() or {}
                
                service_name = data.get('service_name')
                new_version = data.get('version')
                strategy = data.get('strategy', 'blue_green')
                
                if not service_name or not new_version:
                    return {
                        'success': False,
                        'error': 'Missing required parameters',
                        'message': 'service_name and version are required'
                    }, 400
                
                orchestrator = get_infrastructure_orchestrator()
                
                # Convert strategy string to enum
                from ..ml.enterprise.ml_infrastructure_orchestrator import DeploymentStrategy
                strategy_map = {
                    'blue_green': DeploymentStrategy.BLUE_GREEN,
                    'canary': DeploymentStrategy.CANARY,
                    'rolling': DeploymentStrategy.ROLLING,
                    'recreate': DeploymentStrategy.RECREATE
                }
                
                deployment_strategy = strategy_map.get(strategy, DeploymentStrategy.BLUE_GREEN)
                deployment_id = orchestrator.deploy_service(service_name, new_version, deployment_strategy)
                
                return {
                    'success': True,
                    'data': {
                        'deployment_id': deployment_id,
                        'service_name': service_name,
                        'target_version': new_version,
                        'strategy': strategy,
                        'status': 'initiated'
                    },
                    'timestamp': datetime.now().isoformat()
                }, 200
                
            except ValueError as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message': 'Invalid deployment parameters'
                }, 400
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message': 'Failed to initiate deployment'
                }, 500

    @enterprise_ns.route('/infrastructure/nodes')
    class InfrastructureNodes(Resource):
        def get(self):
            """Get all infrastructure nodes status"""
            try:
                orchestrator = get_infrastructure_orchestrator()
                
                nodes_data = {}
                for node_id, node in orchestrator.infrastructure_nodes.items():
                    nodes_data[node_id] = {
                        'node_id': node.node_id,
                        'provider': node.provider.value,
                        'zone': node.zone,
                        'instance_type': node.instance_type,
                        'cpu_cores': node.cpu_cores,
                        'memory_gb': node.memory_gb,
                        'gpu_count': node.gpu_count,
                        'gpu_type': node.gpu_type,
                        'status': node.status,
                        'utilization': node.utilization,
                        'cost_per_hour': node.cost_per_hour
                    }
                
                return {
                    'success': True,
                    'data': {
                        'total_nodes': len(nodes_data),
                        'nodes': nodes_data
                    },
                    'timestamp': datetime.now().isoformat()
                }, 200
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message': 'Failed to retrieve infrastructure nodes'
                }, 500

    @enterprise_ns.route('/metrics/real-time')
    class RealTimeMetrics(Resource):
        def get(self):
            """Get real-time metrics across all enterprise systems"""
            try:
                # Collect metrics from all enterprise systems
                dashboard = get_monitoring_dashboard()
                scaler = get_auto_scaling_system()
                orchestrator = get_infrastructure_orchestrator()
                
                # Monitoring metrics
                monitoring_health = dashboard._calculate_system_health()
                current_metrics = dashboard._get_current_metrics()
                
                # Scaling metrics
                scaling_summary = scaler.get_scaling_summary()
                
                # Infrastructure metrics
                infra_status = orchestrator.get_infrastructure_status()
                
                real_time_data = {
                    'monitoring': {
                        'overall_health': monitoring_health['overall_score'],
                        'status': monitoring_health['status'],
                        'active_modules': monitoring_health['active_modules'],
                        'total_modules': monitoring_health['total_modules']
                    },
                    'scaling': {
                        'total_instances': scaling_summary['system_overview']['total_instances'],
                        'hourly_cost': scaling_summary['system_overview']['current_hourly_cost'],
                        'avg_cpu_utilization': scaling_summary['system_overview']['average_cpu_utilization'],
                        'recent_scaling_events': scaling_summary['recent_activity']['scaling_events_last_hour']
                    },
                    'infrastructure': {
                        'active_nodes': infra_status['cluster_overview']['active_nodes'],
                        'total_cpu_cores': infra_status['cluster_overview']['total_cpu_cores'],
                        'total_memory_gb': infra_status['cluster_overview']['total_memory_gb'],
                        'deployment_success_rate': infra_status['deployments']['deployment_success_rate']
                    },
                    'system_overview': {
                        'total_services': len(orchestrator.ml_service_definitions),
                        'enterprise_ml_modules': 19,
                        'infrastructure_nodes': len(orchestrator.infrastructure_nodes),
                        'monitoring_alerts': len(dashboard.alerts)
                    }
                }
                
                return {
                    'success': True,
                    'data': real_time_data,
                    'timestamp': datetime.now().isoformat(),
                    'refresh_interval': 30  # Recommended refresh interval in seconds
                }, 200
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message': 'Failed to retrieve real-time metrics'
                }, 500

else:
    # Fallback endpoints when enterprise ML is not available
    @enterprise_ns.route('/status')
    class EnterpriseStatus(Resource):
        def get(self):
            """Get enterprise ML infrastructure availability status"""
            return {
                'success': False,
                'error': 'Enterprise ML infrastructure not available',
                'message': 'Enterprise ML modules are not installed or configured',
                'available_features': [],
                'recommendation': 'Install enterprise ML dependencies to access advanced infrastructure features'
            }, 503