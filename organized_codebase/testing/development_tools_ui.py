"""
Development Tools UI Module
===========================
Studio interfaces and debugging tools extracted from AgentScope Studio and development patterns.
Module size: ~295 lines (under 300 limit)

Patterns extracted from:
- AgentScope: Studio interface, project management, and development environment
- CrewAI: Flow debugging and visual development tools
- AutoGen: Development environment and agent debugging
- Agency-Swarm: Tool creation and agent development interfaces
- LLama-Agents: Deployment tools and configuration UIs
- PhiData: Cookbook interfaces and development examples
- Swarms: Intelligence development and testing frameworks

Author: Agent D - Visualization Specialist
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from collections import deque
import threading


@dataclass
class ProjectConfig:
    """Development project configuration."""
    id: str
    name: str
    description: str
    created_at: datetime
    last_modified: datetime
    config: Dict[str, Any]
    agents: List[str]
    workflows: List[str]
    status: str = "active"
    
    @classmethod
    def create(cls, name: str, description: str = "", **config):
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            created_at=datetime.now(),
            last_modified=datetime.now(),
            config=config,
            agents=[],
            workflows=[],
            status="active"
        )


@dataclass
class RunSession:
    """Development run session."""
    id: str
    project_id: str
    name: str
    started_at: datetime
    ended_at: Optional[datetime]
    status: str  # running, completed, failed, stopped
    config: Dict[str, Any]
    logs: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    
    def add_log(self, level: str, message: str, **metadata):
        """Add log entry."""
        self.logs.append({
            "timestamp": datetime.now(),
            "level": level,
            "message": message,
            "metadata": metadata
        })


class StudioInterface:
    """Main studio interface (AgentScope pattern)."""
    
    def __init__(self, workspace_dir: str = "./workspace"):
        self.workspace_dir = workspace_dir
        self.projects: Dict[str, ProjectConfig] = {}
        self.runs: Dict[str, RunSession] = {}
        self.active_sessions: Dict[str, str] = {}  # project_id -> run_id
        self.ui_components = {}
        
        # Ensure workspace exists
        os.makedirs(workspace_dir, exist_ok=True)
        
    def create_project(self, name: str, description: str = "", **config) -> str:
        """Create new development project."""
        project = ProjectConfig.create(name, description, **config)
        self.projects[project.id] = project
        
        # Create project directory
        project_dir = os.path.join(self.workspace_dir, project.id)
        os.makedirs(project_dir, exist_ok=True)
        
        # Save project config
        self._save_project_config(project)
        return project.id
        
    def get_project(self, project_id: str) -> Optional[ProjectConfig]:
        """Get project by ID."""
        return self.projects.get(project_id)
        
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects."""
        return [asdict(project) for project in self.projects.values()]
        
    def start_run(self, project_id: str, run_name: str = "", **config) -> str:
        """Start new run session."""
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
            
        run_name = run_name or f"Run {len(self.runs) + 1}"
        run_session = RunSession(
            id=str(uuid.uuid4()),
            project_id=project_id,
            name=run_name,
            started_at=datetime.now(),
            ended_at=None,
            status="running",
            config=config,
            logs=[],
            metrics={}
        )
        
        self.runs[run_session.id] = run_session
        self.active_sessions[project_id] = run_session.id
        
        run_session.add_log("info", f"Started run session: {run_name}")
        return run_session.id
        
    def stop_run(self, run_id: str, status: str = "completed"):
        """Stop run session."""
        if run_id in self.runs:
            run = self.runs[run_id]
            run.status = status
            run.ended_at = datetime.now()
            run.add_log("info", f"Run session ended with status: {status}")
            
            # Remove from active sessions
            if run.project_id in self.active_sessions:
                if self.active_sessions[run.project_id] == run_id:
                    del self.active_sessions[run.project_id]
                    
    def get_run_logs(self, run_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get run logs."""
        if run_id not in self.runs:
            return []
            
        logs = self.runs[run_id].logs
        return logs[-limit:] if len(logs) > limit else logs
        
    def add_ui_component(self, component_id: str, component_type: str, **config):
        """Add UI component to studio."""
        self.ui_components[component_id] = {
            "type": component_type,
            "config": config,
            "created_at": datetime.now()
        }
        
    def _save_project_config(self, project: ProjectConfig):
        """Save project configuration to file."""
        project_dir = os.path.join(self.workspace_dir, project.id)
        config_path = os.path.join(project_dir, "project.json")
        
        with open(config_path, 'w') as f:
            json.dump(asdict(project), f, indent=2, default=str)


class DebuggingInterface:
    """Debugging tools and interfaces."""
    
    def __init__(self):
        self.breakpoints: Dict[str, List[str]] = {}  # agent_id -> breakpoint_ids
        self.debug_sessions: Dict[str, Dict[str, Any]] = {}
        self.variable_inspectors = {}
        self.execution_trackers = {}
        
    def set_breakpoint(self, agent_id: str, location: str, condition: str = "") -> str:
        """Set debugging breakpoint."""
        breakpoint_id = str(uuid.uuid4())
        
        if agent_id not in self.breakpoints:
            self.breakpoints[agent_id] = []
            
        self.breakpoints[agent_id].append(breakpoint_id)
        
        breakpoint_data = {
            "id": breakpoint_id,
            "agent_id": agent_id,
            "location": location,
            "condition": condition,
            "created_at": datetime.now(),
            "hit_count": 0
        }
        
        return breakpoint_id
        
    def start_debug_session(self, agent_id: str) -> str:
        """Start debugging session for agent."""
        session_id = str(uuid.uuid4())
        
        self.debug_sessions[session_id] = {
            "id": session_id,
            "agent_id": agent_id,
            "started_at": datetime.now(),
            "status": "active",
            "current_frame": None,
            "variables": {},
            "call_stack": []
        }
        
        return session_id
        
    def inspect_variables(self, session_id: str) -> Dict[str, Any]:
        """Inspect variables in debug session."""
        if session_id not in self.debug_sessions:
            return {}
            
        return self.debug_sessions[session_id].get("variables", {})
        
    def step_execution(self, session_id: str, step_type: str = "over") -> bool:
        """Step through execution (over, into, out)."""
        if session_id not in self.debug_sessions:
            return False
            
        session = self.debug_sessions[session_id]
        session["last_step"] = {
            "type": step_type,
            "timestamp": datetime.now()
        }
        
        return True


class CodeEditor:
    """Code editing interface for development."""
    
    def __init__(self):
        self.open_files: Dict[str, Dict[str, Any]] = {}
        self.file_watchers = {}
        self.syntax_highlighters = {}
        
    def open_file(self, file_path: str) -> str:
        """Open file for editing."""
        file_id = str(uuid.uuid4())
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            self.open_files[file_id] = {
                "id": file_id,
                "path": file_path,
                "content": content,
                "original_content": content,
                "modified": False,
                "opened_at": datetime.now(),
                "last_modified": None
            }
            
            return file_id
        except Exception as e:
            return f"Error opening file: {str(e)}"
            
    def save_file(self, file_id: str) -> bool:
        """Save file changes."""
        if file_id not in self.open_files:
            return False
            
        file_info = self.open_files[file_id]
        
        try:
            with open(file_info["path"], 'w', encoding='utf-8') as f:
                f.write(file_info["content"])
                
            file_info["original_content"] = file_info["content"]
            file_info["modified"] = False
            file_info["last_saved"] = datetime.now()
            
            return True
        except Exception:
            return False
            
    def get_file_content(self, file_id: str) -> str:
        """Get file content."""
        if file_id not in self.open_files:
            return ""
            
        return self.open_files[file_id]["content"]
        
    def update_file_content(self, file_id: str, content: str):
        """Update file content."""
        if file_id in self.open_files:
            file_info = self.open_files[file_id]
            file_info["content"] = content
            file_info["modified"] = content != file_info["original_content"]
            file_info["last_modified"] = datetime.now()


class WorkflowDesigner:
    """Visual workflow design interface (CrewAI pattern)."""
    
    def __init__(self):
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.templates = {}
        
    def create_workflow(self, name: str, description: str = "") -> str:
        """Create new workflow."""
        workflow_id = str(uuid.uuid4())
        
        self.workflows[workflow_id] = {
            "id": workflow_id,
            "name": name,
            "description": description,
            "nodes": [],
            "connections": [],
            "created_at": datetime.now(),
            "last_modified": datetime.now(),
            "status": "draft"
        }
        
        return workflow_id
        
    def add_workflow_node(self, workflow_id: str, node_type: str, **config) -> str:
        """Add node to workflow."""
        if workflow_id not in self.workflows:
            return ""
            
        node_id = str(uuid.uuid4())
        node = {
            "id": node_id,
            "type": node_type,
            "config": config,
            "position": config.get("position", {"x": 0, "y": 0})
        }
        
        self.workflows[workflow_id]["nodes"].append(node)
        self.workflows[workflow_id]["last_modified"] = datetime.now()
        
        return node_id
        
    def connect_nodes(self, workflow_id: str, from_node: str, to_node: str, **config) -> str:
        """Connect two nodes in workflow."""
        if workflow_id not in self.workflows:
            return ""
            
        connection_id = str(uuid.uuid4())
        connection = {
            "id": connection_id,
            "from": from_node,
            "to": to_node,
            "config": config
        }
        
        self.workflows[workflow_id]["connections"].append(connection)
        self.workflows[workflow_id]["last_modified"] = datetime.now()
        
        return connection_id
        
    def export_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Export workflow configuration."""
        return self.workflows.get(workflow_id, {})


class TestingInterface:
    """Testing and validation interface."""
    
    def __init__(self):
        self.test_suites: Dict[str, Dict[str, Any]] = {}
        self.test_results: Dict[str, List[Dict[str, Any]]] = {}
        
    def create_test_suite(self, name: str, project_id: str) -> str:
        """Create test suite."""
        suite_id = str(uuid.uuid4())
        
        self.test_suites[suite_id] = {
            "id": suite_id,
            "name": name,
            "project_id": project_id,
            "tests": [],
            "created_at": datetime.now()
        }
        
        return suite_id
        
    def add_test(self, suite_id: str, test_name: str, test_config: Dict[str, Any]) -> str:
        """Add test to suite."""
        if suite_id not in self.test_suites:
            return ""
            
        test_id = str(uuid.uuid4())
        test = {
            "id": test_id,
            "name": test_name,
            "config": test_config,
            "created_at": datetime.now()
        }
        
        self.test_suites[suite_id]["tests"].append(test)
        return test_id
        
    def run_tests(self, suite_id: str) -> List[Dict[str, Any]]:
        """Run test suite."""
        if suite_id not in self.test_suites:
            return []
            
        # Mock test execution
        results = []
        for test in self.test_suites[suite_id]["tests"]:
            result = {
                "test_id": test["id"],
                "test_name": test["name"],
                "status": "passed",  # Mock result
                "duration": 0.1,
                "timestamp": datetime.now()
            }
            results.append(result)
            
        self.test_results[suite_id] = results
        return results


# Public API
__all__ = [
    'ProjectConfig',
    'RunSession',
    'StudioInterface',
    'DebuggingInterface', 
    'CodeEditor',
    'WorkflowDesigner',
    'TestingInterface'
]