
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
TestMaster Orchestration Flask Blueprint
=======================================

Flask blueprint wrapper for the orchestration API.
Provides compatibility with the existing Flask-based dashboard server.

Author: TestMaster Team
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Blueprint, request, jsonify
from functools import wraps

# Import core orchestration components
try:
    from core.orchestration import (
        orchestration_engine, TestAgent, OrchestrationTask,
        AgentStatus, OrchestrationMode
    )
    from core.observability import global_observability
    from core.tools import global_tool_registry, ValidationLevel
    ORCHESTRATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Orchestration modules not available: {e}")
    ORCHESTRATION_AVAILABLE = False

# Create blueprint
orchestration_bp = Blueprint('orchestration', __name__)
logger = logging.getLogger(__name__)

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

def require_orchestration(f):
    """Decorator to check if orchestration is available"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not ORCHESTRATION_AVAILABLE:
            return jsonify({
                "status": "error",
                "error": "Orchestration system not available"
            }), 503
        return f(*args, **kwargs)
    return wrapper

@orchestration_bp.route('/status', methods=['GET'])
@require_orchestration
def get_orchestration_status():
    """Get current orchestration system status"""
    try:
        status = orchestration_engine.get_orchestration_status()
        observability_status = global_observability.get_observability_status()
        
        return jsonify({
            "status": "success",
            "data": {
                "orchestration": status,
                "observability": observability_status,
                "timestamp": datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Failed to get orchestration status: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@orchestration_bp.route('/agents', methods=['GET'])
@require_orchestration
def list_agents():
    """List all registered agents"""
    try:
        agents = {}
        for agent_id, agent in orchestration_engine.agents.items():
            agents[agent_id] = {
                "name": agent.name,
                "role": agent.role,
                "capabilities": agent.capabilities,
                "status": agent.status.value,
                "current_task": agent.current_task,
                "performance_metrics": agent.performance_metrics,
                "dependencies": agent.dependencies
            }
        
        return jsonify({
            "status": "success",
            "data": {
                "agents": agents,
                "total_count": len(agents)
            }
        })
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@orchestration_bp.route('/agents', methods=['POST'])
@require_orchestration
def create_agent():
    """Create and register a new agent"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"status": "error", "error": "No data provided"}), 400
        
        required_fields = ['name', 'role', 'capabilities']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    "status": "error", 
                    "error": f"Missing required field: {field}"
                }), 400
        
        agent = TestAgent(
            agent_id="",  # Will be auto-generated
            name=data['name'],
            role=data['role'],
            capabilities=data['capabilities'],
            dependencies=data.get('dependencies', [])
        )
        
        agent_id = orchestration_engine.register_agent(agent)
        
        return jsonify({
            "status": "success",
            "data": {
                "agent_id": agent_id,
                "agent": {
                    "name": agent.name,
                    "role": agent.role,
                    "capabilities": agent.capabilities,
                    "status": agent.status.value
                }
            }
        })
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@orchestration_bp.route('/agents/<agent_id>', methods=['GET'])
@require_orchestration
def get_agent(agent_id):
    """Get details for a specific agent"""
    try:
        agent = orchestration_engine.agents.get(agent_id)
        if not agent:
            return jsonify({"status": "error", "error": "Agent not found"}), 404
        
        return jsonify({
            "status": "success",
            "data": {
                "agent_id": agent_id,
                "name": agent.name,
                "role": agent.role,
                "capabilities": agent.capabilities,
                "status": agent.status.value,
                "current_task": agent.current_task,
                "performance_metrics": agent.performance_metrics,
                "execution_history": agent.execution_history[-10:]  # Last 10 executions
            }
        })
    except Exception as e:
        logger.error(f"Failed to get agent {agent_id}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@orchestration_bp.route('/crews/<crew_type>', methods=['POST'])
@require_orchestration
def create_test_crew(crew_type):
    """Create a specialized test crew"""
    try:
        valid_crew_types = ["standard", "performance", "security"]
        if crew_type not in valid_crew_types:
            return jsonify({
                "status": "error",
                "error": f"Invalid crew type. Must be one of: {valid_crew_types}"
            }), 400
        
        crew_agents = orchestration_engine.create_test_crew(crew_type)
        
        crew_data = {}
        for role, agent in crew_agents.items():
            crew_data[role] = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "role": agent.role,
                "capabilities": agent.capabilities
            }
        
        return jsonify({
            "status": "success",
            "data": {
                "crew_type": crew_type,
                "agents": crew_data,
                "total_agents": len(crew_agents)
            }
        })
    except Exception as e:
        logger.error(f"Failed to create {crew_type} crew: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@orchestration_bp.route('/tasks', methods=['GET'])
@require_orchestration
def list_tasks():
    """List all orchestration tasks"""
    try:
        tasks = {}
        for task_id, task in orchestration_engine.tasks.items():
            tasks[task_id] = {
                "name": task.name,
                "agent_id": task.agent_id,
                "status": task.status,
                "dependencies": task.dependencies,
                "timeout": task.timeout,
                "execution_time": task.execution_time,
                "created_at": task.created_at.isoformat(),
                "error": task.error
            }
        
        return jsonify({
            "status": "success",
            "data": {
                "tasks": tasks,
                "total_count": len(tasks)
            }
        })
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@orchestration_bp.route('/sessions/execute', methods=['POST'])
@require_orchestration
@async_route
async def execute_orchestration_session():
    """Execute an orchestration session"""
    try:
        data = request.get_json()
        
        if not data or 'session_config' not in data:
            return jsonify({
                "status": "error", 
                "error": "session_config is required"
            }), 400
        
        session_config = data['session_config']
        mode_str = data.get('mode', 'workflow')
        
        # Validate execution mode
        valid_modes = ["sequential", "parallel", "workflow", "adaptive"]
        if mode_str not in valid_modes:
            return jsonify({
                "status": "error",
                "error": f"Invalid execution mode. Must be one of: {valid_modes}"
            }), 400
        
        mode = OrchestrationMode(mode_str)
        
        # Start observability session
        session_name = session_config.get("name", f"orchestration_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        obs_session_id = global_observability.start_test_session(
            session_name,
            metadata={"type": "orchestration", "mode": mode_str}
        )
        
        # Add observability session ID to config
        session_config = session_config.copy()
        session_config["observability_session_id"] = obs_session_id
        
        # Execute session
        try:
            result = await orchestration_engine.execute_orchestration_session(
                session_config, mode
            )
            global_observability.end_test_session(obs_session_id, "completed")
        except Exception as e:
            global_observability.end_test_session(obs_session_id, "failed")
            raise
        
        return jsonify({
            "status": "success",
            "data": {
                "session_id": session_config.get("session_id"),
                "observability_session_id": obs_session_id,
                "mode": mode_str,
                "result": result,
                "status": "completed"
            }
        })
    except Exception as e:
        logger.error(f"Failed to execute session: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@orchestration_bp.route('/tools', methods=['GET'])
@require_orchestration
def list_available_tools():
    """List all available type-safe tools"""
    try:
        tools = global_tool_registry.get_available_tools()
        
        return jsonify({
            "status": "success",
            "data": {
                "tools": tools,
                "total_count": len(tools)
            }
        })
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@orchestration_bp.route('/tools/execute', methods=['POST'])
@require_orchestration
@async_route
async def execute_tool():
    """Execute a type-safe tool"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"status": "error", "error": "No data provided"}), 400
        
        tool_name = data.get('tool_name')
        input_data = data.get('input_data', {})
        validation_level_str = data.get('validation_level', 'strict')
        
        if not tool_name:
            return jsonify({"status": "error", "error": "tool_name is required"}), 400
        
        # Validate validation level
        valid_levels = ["strict", "moderate", "lenient"]
        if validation_level_str not in valid_levels:
            return jsonify({
                "status": "error",
                "error": f"Invalid validation level. Must be one of: {valid_levels}"
            }), 400
        
        validation_level = ValidationLevel(validation_level_str.upper())
        
        # Execute tool
        result = await global_tool_registry.execute_tool_safe(
            tool_name,
            input_data,
            validation_level
        )
        
        return jsonify({
            "status": "success",
            "data": {
                "tool_name": tool_name,
                "execution_id": result.execution_id,
                "tool_status": result.status.value,
                "result": result.result,
                "error": result.error,
                "execution_time": result.execution_time,
                "timestamp": result.timestamp.isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Failed to execute tool: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@orchestration_bp.route('/observability/sessions', methods=['GET'])
@require_orchestration
def list_observability_sessions():
    """List all observability sessions"""
    try:
        sessions = {}
        for session_id, session in global_observability.sessions.items():
            sessions[session_id] = {
                "session_id": session.session_id,
                "name": session.name,
                "status": session.status,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "duration": session.duration,
                "metadata": session.metadata,
                "tags": session.tags,
                "actions_count": len(session.actions),
                "llm_calls_count": len(session.llm_calls)
            }
        
        return jsonify({
            "status": "success",
            "data": {
                "sessions": sessions,
                "total_count": len(sessions)
            }
        })
    except Exception as e:
        logger.error(f"Failed to list observability sessions: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@orchestration_bp.route('/observability/sessions/<session_id>/replay', methods=['GET'])
@require_orchestration
def get_session_replay(session_id):
    """Get detailed session replay data"""
    try:
        replay_data = global_observability.generate_session_replay(session_id)
        
        if "error" in replay_data:
            return jsonify({"status": "error", "error": replay_data["error"]}), 404
        
        return jsonify({
            "status": "success",
            "data": replay_data
        })
    except Exception as e:
        logger.error(f"Failed to get session replay {session_id}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@orchestration_bp.route('/metrics', methods=['GET'])
@require_orchestration
def get_system_metrics():
    """Get comprehensive system metrics"""
    try:
        orchestration_status = orchestration_engine.get_orchestration_status()
        observability_status = global_observability.get_observability_status()
        
        # Calculate additional metrics
        total_tools = len(global_tool_registry.tools)
        
        return jsonify({
            "status": "success",
            "data": {
                "orchestration_metrics": orchestration_status["metrics"],
                "observability_metrics": observability_status["analytics"],
                "system_health": {
                    "orchestration": orchestration_status["system_health"],
                    "observability": observability_status["system_health"]
                },
                "resource_usage": {
                    "active_agents": orchestration_status["active_sessions"],
                    "active_sessions": observability_status["active_sessions"],
                    "total_tools": total_tools
                },
                "timestamp": datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@orchestration_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "orchestration_available": ORCHESTRATION_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

# Initialize orchestration API
def init_orchestration_api():
    """Initialize the orchestration API components"""
    if ORCHESTRATION_AVAILABLE:
        logger.info("Orchestration API initialized successfully")
        return True
    else:
        logger.warning("Orchestration API not available - some imports failed")
        return False

# Export blueprint and initialization function
__all__ = ['orchestration_bp', 'init_orchestration_api']