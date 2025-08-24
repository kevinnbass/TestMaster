"""
Phase 3 API Endpoints
=====================

Production-ready API endpoints for enterprise deployment and advanced UI/UX.
Provides comprehensive REST API for all Phase 3 components.

Author: TestMaster Team
"""

from flask import Blueprint, request, jsonify, session
import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
phase3_bp = Blueprint('phase3', __name__, url_prefix='/api/phase3')

# Global instances (would use dependency injection in production)
deployment_instances = {}
registry_instances = {}
swarm_instances = {}
studio_sessions = {}
agentverse_instances = {}
dashboard_instances = {}


# ==================== Deployment Endpoints ====================

@phase3_bp.route('/deployments', methods=['GET', 'POST'])
def manage_deployments():
    """Manage enterprise deployments"""
    if request.method == 'GET':
        # List all deployments
        deployments = []
        for deploy_id, deployment in deployment_instances.items():
            status = deployment.get_deployment_status()
            deployments.append({
                "deployment_id": deploy_id,
                "mode": status["mode"],
                "status": status["status"],
                "services": len(status["services"]),
                "instances": status["total_instances"],
                "uptime": status["uptime_seconds"]
            })
        
        return jsonify({
            "deployments": deployments,
            "count": len(deployments)
        })
    
    elif request.method == 'POST':
        # Create new deployment
        data = request.json
        
        from deployment import EnterpriseTestDeployment, DeploymentMode
        
        mode_str = data.get('mode', 'production')
        mode = DeploymentMode[mode_str.upper()]
        
        deployment = EnterpriseTestDeployment(mode)
        deployment_instances[deployment.deployment_id] = deployment
        
        return jsonify({
            "deployment_id": deployment.deployment_id,
            "status": "created",
            "mode": mode.value
        }), 201


@phase3_bp.route('/deployments/<deployment_id>', methods=['GET', 'DELETE'])
def deployment_detail(deployment_id):
    """Get or delete a specific deployment"""
    if deployment_id not in deployment_instances:
        return jsonify({"error": "Deployment not found"}), 404
    
    deployment = deployment_instances[deployment_id]
    
    if request.method == 'GET':
        return jsonify(deployment.get_deployment_status())
    
    elif request.method == 'DELETE':
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(deployment.shutdown())
        loop.close()
        
        if success:
            del deployment_instances[deployment_id]
            return jsonify({"status": "deleted"})
        
        return jsonify({"error": "Failed to shutdown deployment"}), 500


@phase3_bp.route('/deployments/<deployment_id>/scale', methods=['POST'])
def scale_service(deployment_id):
    """Scale a service in the deployment"""
    if deployment_id not in deployment_instances:
        return jsonify({"error": "Deployment not found"}), 404
    
    deployment = deployment_instances[deployment_id]
    data = request.json
    
    service_id = data.get('service_id')
    target_replicas = data.get('target_replicas')
    
    if not service_id or target_replicas is None:
        return jsonify({"error": "Missing service_id or target_replicas"}), 400
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    success = loop.run_until_complete(
        deployment.scale_service(service_id, target_replicas)
    )
    loop.close()
    
    if success:
        return jsonify({"status": "scaled", "replicas": target_replicas})
    
    return jsonify({"error": "Failed to scale service"}), 500


@phase3_bp.route('/deployments/<deployment_id>/health', methods=['GET'])
def deployment_health(deployment_id):
    """Get deployment health status"""
    if deployment_id not in deployment_instances:
        return jsonify({"error": "Deployment not found"}), 404
    
    deployment = deployment_instances[deployment_id]
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    health = loop.run_until_complete(deployment.perform_health_check())
    loop.close()
    
    return jsonify(health)


# ==================== Service Registry Endpoints ====================

@phase3_bp.route('/registry', methods=['GET', 'POST'])
def manage_registry():
    """Manage service registry"""
    if request.method == 'GET':
        # Get registry status
        registries = []
        for reg_id, registry in registry_instances.items():
            status = registry.get_registry_status()
            registries.append({
                "registry_id": reg_id,
                "services": status["total_services"],
                "healthy": status["healthy_services"],
                "zones": status["zones"]
            })
        
        return jsonify({
            "registries": registries,
            "count": len(registries)
        })
    
    elif request.method == 'POST':
        # Create new registry
        from deployment import ServiceRegistry
        
        registry = ServiceRegistry()
        registry_instances[registry.registry_id] = registry
        
        return jsonify({
            "registry_id": registry.registry_id,
            "status": "created"
        }), 201


@phase3_bp.route('/registry/<registry_id>/services', methods=['GET', 'POST'])
def registry_services(registry_id):
    """Manage services in registry"""
    if registry_id not in registry_instances:
        return jsonify({"error": "Registry not found"}), 404
    
    registry = registry_instances[registry_id]
    
    if request.method == 'GET':
        # Discover services
        params = request.args
        services = registry.discover_services(
            service_type=params.get('type'),
            zone=params.get('zone'),
            health_only=params.get('health_only', 'true').lower() == 'true'
        )
        
        return jsonify({
            "services": [
                {
                    "service_id": s.service_id,
                    "name": s.name,
                    "type": s.service_type,
                    "health": s.health_status.value,
                    "endpoints": len(s.endpoints)
                }
                for s in services
            ],
            "count": len(services)
        })
    
    elif request.method == 'POST':
        # Register new service
        from deployment import ServiceDescriptor, ServiceEndpoint
        
        data = request.json
        
        endpoints = [
            ServiceEndpoint(
                host=ep.get('host', 'localhost'),
                port=ep.get('port', 8080),
                zone=ep.get('zone', 'default')
            )
            for ep in data.get('endpoints', [])
        ]
        
        service = ServiceDescriptor(
            name=data.get('name', 'Service'),
            service_type=data.get('type', 'test_service'),
            endpoints=endpoints,
            capabilities=data.get('capabilities', [])
        )
        
        service_id = registry.register_service(service)
        
        return jsonify({
            "service_id": service_id,
            "status": "registered"
        }), 201


# ==================== Swarm Orchestration Endpoints ====================

@phase3_bp.route('/swarms', methods=['GET', 'POST'])
def manage_swarms():
    """Manage swarm orchestrators"""
    if request.method == 'GET':
        # List all swarms
        swarms = []
        for swarm_id, swarm in swarm_instances.items():
            status = swarm.get_swarm_status()
            swarms.append({
                "orchestrator_id": swarm_id,
                "name": status["config"]["name"],
                "agents": status["agents"]["total"],
                "tasks": status["tasks"]["total"],
                "performance": status["performance"]["average_agent_score"]
            })
        
        return jsonify({
            "swarms": swarms,
            "count": len(swarms)
        })
    
    elif request.method == 'POST':
        # Create new swarm
        from deployment import SwarmOrchestrator, SwarmConfig
        
        data = request.json
        
        config = SwarmConfig(
            name=data.get('name', 'TestSwarm'),
            min_agents=data.get('min_agents', 3),
            max_agents=data.get('max_agents', 50),
            task_distribution_strategy=data.get('strategy', 'performance_weighted')
        )
        
        swarm = SwarmOrchestrator(config)
        swarm_instances[swarm.orchestrator_id] = swarm
        
        # Start swarm
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(swarm.start())
        loop.close()
        
        return jsonify({
            "orchestrator_id": swarm.orchestrator_id,
            "status": "started"
        }), 201


@phase3_bp.route('/swarms/<swarm_id>/tasks', methods=['GET', 'POST'])
def swarm_tasks(swarm_id):
    """Manage swarm tasks"""
    if swarm_id not in swarm_instances:
        return jsonify({"error": "Swarm not found"}), 404
    
    swarm = swarm_instances[swarm_id]
    
    if request.method == 'GET':
        # Get task list
        tasks = []
        for task_id, task in swarm.tasks.items():
            tasks.append({
                "task_id": task_id,
                "type": task.task_type,
                "status": task.status.value,
                "priority": task.priority,
                "assigned_to": task.assigned_to
            })
        
        return jsonify({
            "tasks": tasks,
            "count": len(tasks)
        })
    
    elif request.method == 'POST':
        # Submit new task
        from deployment import SwarmTask
        
        data = request.json
        
        task = SwarmTask(
            task_type=data.get('type', 'test_execution'),
            priority=data.get('priority', 5),
            payload=data.get('payload', {}),
            dependencies=data.get('dependencies', [])
        )
        
        task_id = swarm.submit_task(task)
        
        return jsonify({
            "task_id": task_id,
            "status": "submitted"
        }), 201


# ==================== Studio Interface Endpoints ====================

@phase3_bp.route('/studio/sessions', methods=['GET', 'POST'])
def studio_sessions_api():
    """Manage studio sessions"""
    if request.method == 'GET':
        # List sessions
        sessions = []
        for session_id, studio_session in studio_sessions.items():
            sessions.append({
                "session_id": session_id,
                "user_id": studio_session.user_id,
                "workflows": len(studio_session.workflows),
                "mode": studio_session.interaction_mode.value,
                "last_activity": studio_session.last_activity.isoformat()
            })
        
        return jsonify({
            "sessions": sessions,
            "count": len(sessions)
        })
    
    elif request.method == 'POST':
        # Create new session
        from ui_ux import StudioInterface, InteractionMode
        
        data = request.json
        
        # Get or create studio instance
        if 'studio' not in globals():
            global studio
            studio = StudioInterface()
        
        mode = InteractionMode[data.get('mode', 'VISUAL').upper()]
        session_id = studio.create_session(
            data.get('user_id', 'anonymous'),
            mode
        )
        
        studio_sessions[session_id] = studio.get_session(session_id)
        
        return jsonify({
            "session_id": session_id,
            "status": "created"
        }), 201


@phase3_bp.route('/studio/sessions/<session_id>/workflows', methods=['GET', 'POST'])
def studio_workflows(session_id):
    """Manage workflows in a studio session"""
    if session_id not in studio_sessions:
        return jsonify({"error": "Session not found"}), 404
    
    if request.method == 'GET':
        # Get workflows
        studio_session = studio_sessions[session_id]
        workflows = []
        
        for workflow in studio_session.workflows:
            workflows.append({
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "nodes": len(workflow.nodes),
                "edges": len(workflow.edges)
            })
        
        return jsonify({
            "workflows": workflows,
            "count": len(workflows)
        })
    
    elif request.method == 'POST':
        # Create workflow
        data = request.json
        
        workflow_id = studio.create_workflow(
            session_id,
            data.get('name', 'New Workflow'),
            data.get('description', '')
        )
        
        return jsonify({
            "workflow_id": workflow_id,
            "status": "created"
        }), 201


# ==================== AgentVerse UI Endpoints ====================

@phase3_bp.route('/agentverse', methods=['GET', 'POST'])
def agentverse_api():
    """Manage AgentVerse visualizations"""
    if request.method == 'GET':
        # List visualizations
        visualizations = []
        for viz_id, agentverse in agentverse_instances.items():
            status = agentverse.get_ui_status()
            visualizations.append({
                "ui_id": viz_id,
                "visualizations": status["visualizations"],
                "agents": status["total_agents"],
                "interactions": status["total_interactions"]
            })
        
        return jsonify({
            "visualizations": visualizations,
            "count": len(visualizations)
        })
    
    elif request.method == 'POST':
        # Create new AgentVerse UI
        from ui_ux import AgentVerseUI
        
        ui = AgentVerseUI()
        agentverse_instances[ui.ui_id] = ui
        
        viz_id = ui.create_visualization(
            request.json.get('name', 'Default Visualization')
        )
        
        return jsonify({
            "ui_id": ui.ui_id,
            "viz_id": viz_id,
            "status": "created"
        }), 201


@phase3_bp.route('/agentverse/<ui_id>/agents', methods=['GET', 'POST'])
def agentverse_agents(ui_id):
    """Manage agents in AgentVerse"""
    if ui_id not in agentverse_instances:
        return jsonify({"error": "AgentVerse UI not found"}), 404
    
    ui = agentverse_instances[ui_id]
    
    if request.method == 'GET':
        # Get agents
        if not ui.active_viz:
            return jsonify({"agents": [], "count": 0})
        
        agents = []
        for agent_id, agent in ui.active_viz.graph.nodes.items():
            agents.append({
                "agent_id": agent_id,
                "name": agent.name,
                "type": agent.type,
                "status": agent.status,
                "position": agent.position
            })
        
        return jsonify({
            "agents": agents,
            "count": len(agents)
        })
    
    elif request.method == 'POST':
        # Add agent
        data = request.json
        
        agent = ui.add_agent(
            data.get('agent_id', f"agent_{len(ui.active_viz.graph.nodes)}"),
            data.get('name', 'Agent'),
            data.get('type', 'default')
        )
        
        return jsonify({
            "agent_id": agent.agent_id,
            "status": "added"
        }), 201


# ==================== Dashboard Endpoints ====================

@phase3_bp.route('/dashboards', methods=['GET', 'POST'])
def dashboards_api():
    """Manage interactive dashboards"""
    if request.method == 'GET':
        # List dashboards
        dashboards = []
        for dash_id, dashboard in dashboard_instances.items():
            status = dashboard.get_dashboard_status()
            dashboards.append({
                "dashboard_id": dash_id,
                "name": status["name"],
                "widgets": status["widget_count"],
                "alerts": status["active_alerts"]
            })
        
        return jsonify({
            "dashboards": dashboards,
            "count": len(dashboards)
        })
    
    elif request.method == 'POST':
        # Create new dashboard
        from ui_ux import InteractiveDashboard
        
        data = request.json
        
        dashboard = InteractiveDashboard(
            data.get('name', 'New Dashboard')
        )
        dashboard_instances[dashboard.dashboard_id] = dashboard
        
        return jsonify({
            "dashboard_id": dashboard.dashboard_id,
            "status": "created"
        }), 201


@phase3_bp.route('/dashboards/<dashboard_id>/widgets', methods=['GET', 'POST'])
def dashboard_widgets(dashboard_id):
    """Manage dashboard widgets"""
    if dashboard_id not in dashboard_instances:
        return jsonify({"error": "Dashboard not found"}), 404
    
    dashboard = dashboard_instances[dashboard_id]
    
    if request.method == 'GET':
        # Get widgets
        widgets = []
        for widget_id, widget in dashboard.widgets.items():
            widgets.append({
                "widget_id": widget_id,
                "type": widget.widget_type.value,
                "title": widget.title,
                "position": widget.position,
                "size": widget.size
            })
        
        return jsonify({
            "widgets": widgets,
            "count": len(widgets)
        })
    
    elif request.method == 'POST':
        # Add widget
        from ui_ux import RealtimeChart, MetricsPanel, ControlPanel
        from ui_ux.interactive_dashboard import WidgetType, ChartType
        
        data = request.json
        widget_type = WidgetType[data.get('type', 'METRIC').upper()]
        
        if widget_type == WidgetType.CHART:
            widget = RealtimeChart(
                title=data.get('title', 'Chart'),
                chart_type=ChartType[data.get('chart_type', 'LINE').upper()],
                position=data.get('position', {"x": 0, "y": 0}),
                size=data.get('size', {"width": 4, "height": 3})
            )
        elif widget_type == WidgetType.METRIC:
            widget = MetricsPanel(
                title=data.get('title', 'Metrics'),
                position=data.get('position', {"x": 0, "y": 0}),
                size=data.get('size', {"width": 4, "height": 2})
            )
        elif widget_type == WidgetType.CONTROL:
            widget = ControlPanel(
                title=data.get('title', 'Controls'),
                position=data.get('position', {"x": 0, "y": 0}),
                size=data.get('size', {"width": 4, "height": 2})
            )
        else:
            from ui_ux import DashboardWidget
            widget = DashboardWidget(
                widget_type=widget_type,
                title=data.get('title', 'Widget'),
                position=data.get('position', {"x": 0, "y": 0}),
                size=data.get('size', {"width": 4, "height": 3})
            )
        
        widget_id = dashboard.add_widget(widget)
        
        return jsonify({
            "widget_id": widget_id,
            "status": "added"
        }), 201


# ==================== Status Endpoints ====================

@phase3_bp.route('/status', methods=['GET'])
def phase3_status():
    """Get overall Phase 3 system status"""
    return jsonify({
        "phase": "3",
        "name": "Production Enhancement",
        "components": {
            "deployments": len(deployment_instances),
            "registries": len(registry_instances),
            "swarms": len(swarm_instances),
            "studio_sessions": len(studio_sessions),
            "agentverse_uis": len(agentverse_instances),
            "dashboards": len(dashboard_instances)
        },
        "features": {
            "enterprise_deployment": True,
            "service_registry": True,
            "swarm_orchestration": True,
            "visual_studio": True,
            "agent_visualization": True,
            "interactive_dashboards": True
        },
        "api_version": "3.0.0",
        "status": "operational"
    })


# ==================== Error Handlers ====================

@phase3_bp.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404


@phase3_bp.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


@phase3_bp.errorhandler(Exception)
def handle_exception(error):
    logger.error(f"Unhandled exception: {error}")
    return jsonify({"error": str(error)}), 500