
# AGENT D SECURITY INTEGRATION
try:
    from SECURITY_PATCHES.api_security_framework import APISecurityFramework
    from SECURITY_PATCHES.authentication_framework import SecurityFramework
    _security_framework = SecurityFramework()
    _api_security = APISecurityFramework()
    _SECURITY_ENABLED = True
except ImportError:
    _SECURITY_ENABLED = False
    print("Security frameworks not available - running without protection")

def apply_security_middleware():
    """Apply security middleware to requests"""
    if not _SECURITY_ENABLED:
        return True, {}
    
    from flask import request
    request_data = {
        'ip_address': request.remote_addr,
        'endpoint': request.path,
        'method': request.method,
        'user_agent': request.headers.get('User-Agent', ''),
        'body': request.get_json() if request.is_json else {},
        'query_params': dict(request.args),
        'headers': dict(request.headers)
    }
    
    return _api_security.validate_request(request_data)

"""
Phase 2 API Endpoints
=====================

API endpoints for Phase 2 multi-agent testing framework and enhanced monitoring.
Provides integration layer for frontend components and external systems.

Author: TestMaster Team
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Blueprint, request, jsonify
from functools import wraps

# Import Phase 2 components
try:
    # Try importing from testmaster package
    from testmaster.agents.team import (
        TestingTeam, TeamConfiguration, TeamRole, TeamWorkflow, 
        STANDARD_TESTING_WORKFLOW
    )
    from testmaster.agents.supervisor import TestingSupervisor, SupervisorMode
    from testmaster.monitoring import (
        EnhancedTestMonitor, MonitoringMode,
        PerformanceMonitoringAgent, QualityMonitoringAgent,
        SecurityMonitoringAgent, CollaborationMonitoringAgent
    )
    PHASE2_AVAILABLE = True
except ImportError:
    try:
        # Fallback to relative imports
        from ...agents.team import (
            TestingTeam, TeamConfiguration, TeamRole, TeamWorkflow, 
            STANDARD_TESTING_WORKFLOW
        )
        from ...agents.supervisor import TestingSupervisor, SupervisorMode
        from ...monitoring import (
            EnhancedTestMonitor, MonitoringMode,
            PerformanceMonitoringAgent, QualityMonitoringAgent,
            SecurityMonitoringAgent, CollaborationMonitoringAgent
        )
        PHASE2_AVAILABLE = True
    except ImportError as e:
        logging.warning(f"Phase 2 components not available: {e}")
        PHASE2_AVAILABLE = False
        # Create dummy classes to prevent errors
        class TestingTeam: pass
        class EnhancedTestMonitor: pass

# Create blueprint
phase2_bp = Blueprint('phase2', __name__)
logger = logging.getLogger(__name__)

# Global state for Phase 2 components
active_teams: Dict[str, TestingTeam] = {}
active_monitors: Dict[str, EnhancedTestMonitor] = {}

def async_route(f):
    """Decorator to run async functions in Flask routes"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

def require_phase2(f):
    """Decorator to check if Phase 2 is available"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not PHASE2_AVAILABLE:
            return jsonify({
                "status": "error",
                "error": "Phase 2 components not available"
            }), 503
        return f(*args, **kwargs)
    return wrapper

# =============================================================================
# Testing Team API Endpoints
# =============================================================================

@phase2_bp.route('/teams', methods=['GET'])
@require_phase2
def list_teams():
    """List all active testing teams"""
    try:
        teams_data = {}
        for team_id, team in active_teams.items():
            teams_data[team_id] = team.get_team_status()
        
        return jsonify({
            "status": "success",
            "data": {
                "teams": teams_data,
                "total_count": len(teams_data)
            }
        })
    except Exception as e:
        logger.error(f"Failed to list teams: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@phase2_bp.route('/teams', methods=['POST'])
@require_phase2
@async_route
async def create_team():
    """Create a new testing team"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"status": "error", "error": "No data provided"}), 400
        
        # Parse team configuration
        team_type = data.get("type", "standard")
        roles = data.get("roles", [])
        supervisor_mode = data.get("supervisor_mode", "guided")
        
        if team_type == "standard":
            team = TestingTeam.create_standard_team()
        elif team_type == "minimal":
            team = TestingTeam.create_minimal_team()
        else:
            # Custom team configuration
            config = TeamConfiguration(
                roles=[TeamRole(role) for role in roles],
                supervisor_mode=SupervisorMode(supervisor_mode),
                workflow_type=data.get("workflow_type", "standard")
            )
            team = TestingTeam(config)
        
        # Start the team
        session_id = data.get("session_id")
        await team.start_team(session_id)
        
        # Store team
        team_id = team.team_id
        active_teams[team_id] = team
        
        return jsonify({
            "status": "success",
            "data": {
                "team_id": team_id,
                "team_status": team.get_team_status()
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to create team: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@phase2_bp.route('/teams/<team_id>', methods=['GET'])
@require_phase2
def get_team(team_id):
    """Get details for a specific team"""
    try:
        if team_id not in active_teams:
            return jsonify({"status": "error", "error": "Team not found"}), 404
        
        team = active_teams[team_id]
        return jsonify({
            "status": "success",
            "data": team.get_team_status()
        })
    except Exception as e:
        logger.error(f"Failed to get team {team_id}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@phase2_bp.route('/teams/<team_id>/execute', methods=['POST'])
@require_phase2
@async_route
async def execute_team_workflow(team_id):
    """Execute a workflow with a testing team"""
    try:
        if team_id not in active_teams:
            return jsonify({"status": "error", "error": "Team not found"}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "error": "No data provided"}), 400
        
        team = active_teams[team_id]
        target_path = data.get("target_path", ".")
        workflow_type = data.get("workflow", "standard")
        
        # Select workflow
        if workflow_type == "standard":
            workflow = STANDARD_TESTING_WORKFLOW
        else:
            # Custom workflow
            workflow = TeamWorkflow(
                name=data.get("workflow_name", "Custom Workflow"),
                phases=data.get("phases", []),
                success_criteria=data.get("success_criteria", {})
            )
        
        # Execute workflow
        result = await team.execute_workflow(workflow, target_path)
        
        return jsonify({
            "status": "success",
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Failed to execute workflow for team {team_id}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@phase2_bp.route('/teams/<team_id>', methods=['DELETE'])
@require_phase2
@async_route
async def stop_team(team_id):
    """Stop and remove a testing team"""
    try:
        if team_id not in active_teams:
            return jsonify({"status": "error", "error": "Team not found"}), 404
        
        team = active_teams[team_id]
        await team.stop_team()
        
        # Remove from active teams
        del active_teams[team_id]
        
        return jsonify({
            "status": "success",
            "message": f"Team {team_id} stopped and removed"
        })
        
    except Exception as e:
        logger.error(f"Failed to stop team {team_id}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

# =============================================================================
# Enhanced Monitoring API Endpoints
# =============================================================================

@phase2_bp.route('/monitoring', methods=['GET'])
@require_phase2
def list_monitors():
    """List all active monitoring instances"""
    try:
        monitors_data = {}
        for monitor_id, monitor in active_monitors.items():
            monitors_data[monitor_id] = monitor.get_monitoring_status()
        
        return jsonify({
            "status": "success",
            "data": {
                "monitors": monitors_data,
                "total_count": len(monitors_data)
            }
        })
    except Exception as e:
        logger.error(f"Failed to list monitors: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@phase2_bp.route('/monitoring', methods=['POST'])
@require_phase2
@async_route
async def create_monitor():
    """Create a new enhanced monitoring instance"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"status": "error", "error": "No data provided"}), 400
        
        # Create enhanced monitor
        mode = MonitoringMode(data.get("mode", "interactive"))
        monitor = EnhancedTestMonitor(mode)
        
        # Add monitoring agents
        agents_config = data.get("agents", ["performance", "quality", "security", "collaboration"])
        
        for agent_type in agents_config:
            if agent_type == "performance":
                monitor.add_monitoring_agent(PerformanceMonitoringAgent())
            elif agent_type == "quality":
                monitor.add_monitoring_agent(QualityMonitoringAgent())
            elif agent_type == "security":
                monitor.add_monitoring_agent(SecurityMonitoringAgent())
            elif agent_type == "collaboration":
                monitor.add_monitoring_agent(CollaborationMonitoringAgent())
        
        # Start monitoring
        session_id = data.get("session_id")
        await monitor.start_monitoring(session_id)
        
        # Store monitor
        monitor_id = monitor.monitor_id
        active_monitors[monitor_id] = monitor
        
        return jsonify({
            "status": "success",
            "data": {
                "monitor_id": monitor_id,
                "monitor_status": monitor.get_monitoring_status()
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to create monitor: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@phase2_bp.route('/monitoring/<monitor_id>', methods=['GET'])
@require_phase2
def get_monitor(monitor_id):
    """Get details for a specific monitor"""
    try:
        if monitor_id not in active_monitors:
            return jsonify({"status": "error", "error": "Monitor not found"}), 404
        
        monitor = active_monitors[monitor_id]
        return jsonify({
            "status": "success",
            "data": monitor.get_monitoring_status()
        })
    except Exception as e:
        logger.error(f"Failed to get monitor {monitor_id}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@phase2_bp.route('/monitoring/<monitor_id>/chat', methods=['POST'])
@require_phase2
@async_route
async def chat_with_monitor(monitor_id):
    """Chat with monitoring system"""
    try:
        if monitor_id not in active_monitors:
            return jsonify({"status": "error", "error": "Monitor not found"}), 404
        
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"status": "error", "error": "Message is required"}), 400
        
        monitor = active_monitors[monitor_id]
        message = data["message"]
        sender = data.get("sender", "user")
        
        # Process conversational query
        response = await monitor.process_conversation(message, sender)
        
        return jsonify({
            "status": "success",
            "data": {
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to process chat for monitor {monitor_id}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@phase2_bp.route('/monitoring/<monitor_id>/analyze', methods=['POST'])
@require_phase2
@async_route
async def analyze_data(monitor_id):
    """Analyze data with multi-modal analyzer"""
    try:
        if monitor_id not in active_monitors:
            return jsonify({"status": "error", "error": "Monitor not found"}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "error": "No data provided"}), 400
        
        monitor = active_monitors[monitor_id]
        
        analysis_data = data.get("data")
        data_type = data.get("type", "json")
        context = data.get("context", {})
        
        # Perform analysis
        result = await monitor.analyze_data(analysis_data, data_type, context)
        
        return jsonify({
            "status": "success",
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Failed to analyze data for monitor {monitor_id}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@phase2_bp.route('/monitoring/<monitor_id>', methods=['DELETE'])
@require_phase2
@async_route
async def stop_monitor(monitor_id):
    """Stop and remove a monitoring instance"""
    try:
        if monitor_id not in active_monitors:
            return jsonify({"status": "error", "error": "Monitor not found"}), 404
        
        monitor = active_monitors[monitor_id]
        await monitor.stop_monitoring()
        
        # Remove from active monitors
        del active_monitors[monitor_id]
        
        return jsonify({
            "status": "success",
            "message": f"Monitor {monitor_id} stopped and removed"
        })
        
    except Exception as e:
        logger.error(f"Failed to stop monitor {monitor_id}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

# =============================================================================
# Phase 2 Integration Endpoints
# =============================================================================

@phase2_bp.route('/status', methods=['GET'])
@require_phase2
def get_phase2_status():
    """Get overall Phase 2 system status"""
    try:
        return jsonify({
            "status": "success",
            "data": {
                "phase2_available": PHASE2_AVAILABLE,
                "active_teams": len(active_teams),
                "active_monitors": len(active_monitors),
                "components": {
                    "multi_agent_framework": True,
                    "enhanced_monitoring": True,
                    "conversational_interface": True,
                    "multi_modal_analysis": True
                },
                "timestamp": datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Failed to get Phase 2 status: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@phase2_bp.route('/workflows/templates', methods=['GET'])
@require_phase2
def get_workflow_templates():
    """Get available workflow templates"""
    try:
        templates = {
            "standard": {
                "name": "Standard Testing Workflow",
                "description": "Comprehensive testing workflow with all phases",
                "phases": [phase["name"] for phase in STANDARD_TESTING_WORKFLOW.phases],
                "estimated_duration": "15-30 minutes"
            },
            "minimal": {
                "name": "Minimal Testing Workflow",
                "description": "Basic testing workflow for simple projects",
                "phases": ["requirements_analysis", "test_implementation", "test_execution"],
                "estimated_duration": "5-10 minutes"
            },
            "quality_focused": {
                "name": "Quality-Focused Workflow",
                "description": "Workflow emphasizing quality assurance and review",
                "phases": ["requirements_analysis", "test_design", "test_implementation", "quality_review", "optimization"],
                "estimated_duration": "20-40 minutes"
            }
        }
        
        return jsonify({
            "status": "success",
            "data": {
                "templates": templates,
                "total_count": len(templates)
            }
        })
    except Exception as e:
        logger.error(f"Failed to get workflow templates: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@phase2_bp.route('/capabilities', methods=['GET'])
@require_phase2
def get_phase2_capabilities():
    """Get Phase 2 capabilities and features"""
    try:
        capabilities = {
            "multi_agent_testing": {
                "description": "Role-based testing with hierarchical supervision",
                "features": [
                    "Test Architect for strategy and design",
                    "Test Engineer for implementation",
                    "Quality Assurance Agent for review",
                    "Test Executor for execution",
                    "Test Coordinator for orchestration"
                ],
                "supervision_modes": ["autonomous", "guided", "directed", "collaborative", "hierarchical"]
            },
            "enhanced_monitoring": {
                "description": "Multi-modal monitoring with conversational interface",
                "features": [
                    "Performance monitoring with trend analysis",
                    "Quality monitoring with coverage tracking",
                    "Security monitoring with vulnerability detection",
                    "Collaboration monitoring with communication analysis"
                ],
                "interaction_modes": ["passive", "interactive", "proactive", "conversational"]
            },
            "workflow_orchestration": {
                "description": "Intelligent workflow coordination and execution",
                "features": [
                    "Dynamic task assignment",
                    "Performance-based optimization",
                    "Quality gate enforcement",
                    "Resource allocation management"
                ]
            }
        }
        
        return jsonify({
            "status": "success",
            "data": capabilities
        })
    except Exception as e:
        logger.error(f"Failed to get Phase 2 capabilities: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

# Health check for Phase 2
@phase2_bp.route('/health', methods=['GET'])
def phase2_health_check():
    """Health check for Phase 2 components"""
    return jsonify({
        "status": "healthy" if PHASE2_AVAILABLE else "unavailable",
        "phase2_available": PHASE2_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    })

# Initialize Phase 2 API
def init_phase2_api():
    """Initialize Phase 2 API components"""
    if PHASE2_AVAILABLE:
        logger.info("Phase 2 API initialized successfully")
        return True
    else:
        logger.warning("Phase 2 API not available - components missing")
        return False

# Export blueprint and initialization function
__all__ = ['phase2_bp', 'init_phase2_api']