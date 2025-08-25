"""
Studio Interface System
=======================

AutoGen Studio-inspired interface for visual agent orchestration.
Provides drag-and-drop workflow creation and real-time monitoring.

Author: TestMaster Team
"""

import json
import uuid
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Configure logging
logger = logging.getLogger(__name__)


class InteractionMode(Enum):
    """Modes of interaction in the studio"""
    VISUAL = "visual"  # Drag-and-drop interface
    CODE = "code"  # Code-based configuration
    HYBRID = "hybrid"  # Both visual and code
    CONVERSATIONAL = "conversational"  # Natural language
    GUIDED = "guided"  # Step-by-step wizard


class VisualizationType(Enum):
    """Types of visualizations available"""
    WORKFLOW = "workflow"
    AGENT_GRAPH = "agent_graph"
    TIMELINE = "timeline"
    METRICS = "metrics"
    DEPENDENCY_TREE = "dependency_tree"
    HEATMAP = "heatmap"
    SANKEY = "sankey"


@dataclass
class StudioWorkflow:
    """Workflow definition in the studio"""
    workflow_id: str = field(default_factory=lambda: f"workflow_{uuid.uuid4().hex[:12]}")
    name: str = ""
    description: str = ""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node_type: str, position: Dict[str, float], 
                 config: Dict[str, Any] = None) -> str:
        """Add a node to the workflow"""
        node_id = f"node_{uuid.uuid4().hex[:8]}"
        node = {
            "id": node_id,
            "type": node_type,
            "position": position,
            "config": config or {},
            "status": "inactive"
        }
        self.nodes.append(node)
        self.modified_at = datetime.now()
        return node_id
    
    def add_edge(self, source: str, target: str, 
                 condition: Optional[str] = None) -> str:
        """Add an edge between nodes"""
        edge_id = f"edge_{uuid.uuid4().hex[:8]}"
        edge = {
            "id": edge_id,
            "source": source,
            "target": target,
            "condition": condition
        }
        self.edges.append(edge)
        self.modified_at = datetime.now()
        return edge_id
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate workflow structure"""
        errors = []
        
        # Check for orphaned nodes
        node_ids = {node["id"] for node in self.nodes}
        for edge in self.edges:
            if edge["source"] not in node_ids:
                errors.append(f"Edge source '{edge['source']}' not found")
            if edge["target"] not in node_ids:
                errors.append(f"Edge target '{edge['target']}' not found")
        
        # Check for cycles
        if self._has_cycle():
            errors.append("Workflow contains cycles")
        
        # Check for required configurations
        for node in self.nodes:
            if node["type"] == "test_agent" and "agent_id" not in node.get("config", {}):
                errors.append(f"Node {node['id']} missing required agent_id")
        
        return len(errors) == 0, errors
    
    def _has_cycle(self) -> bool:
        """Check if workflow has cycles"""
        # Build adjacency list
        graph = {node["id"]: [] for node in self.nodes}
        for edge in self.edges:
            if edge["source"] in graph:
                graph[edge["source"]].append(edge["target"])
        
        # DFS to detect cycle
        visited = set()
        rec_stack = set()
        
        def has_cycle_util(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle_util(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle_util(node):
                    return True
        
        return False


@dataclass
class StudioSession:
    """Active studio session"""
    session_id: str = field(default_factory=lambda: f"session_{uuid.uuid4().hex[:12]}")
    user_id: str = ""
    workflows: List[StudioWorkflow] = field(default_factory=list)
    active_workflow: Optional[StudioWorkflow] = None
    interaction_mode: InteractionMode = InteractionMode.VISUAL
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    undo_stack: List[Dict[str, Any]] = field(default_factory=list)
    redo_stack: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def record_action(self, action: Dict[str, Any]):
        """Record an action for undo/redo"""
        self.undo_stack.append(action)
        self.redo_stack.clear()
        self.last_activity = datetime.now()
        
        # Limit stack size
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)
    
    def undo(self) -> Optional[Dict[str, Any]]:
        """Undo last action"""
        if self.undo_stack:
            action = self.undo_stack.pop()
            self.redo_stack.append(action)
            return action
        return None
    
    def redo(self) -> Optional[Dict[str, Any]]:
        """Redo last undone action"""
        if self.redo_stack:
            action = self.redo_stack.pop()
            self.undo_stack.append(action)
            return action
        return None


class StudioInterface:
    """
    Advanced studio interface with AutoGen Studio patterns.
    
    Features:
    - Visual workflow builder
    - Drag-and-drop agent configuration
    - Real-time execution monitoring
    - Code generation from visual workflows
    - Interactive debugging
    - Template library
    - Collaboration features
    """
    
    def __init__(self):
        self.studio_id = f"studio_{uuid.uuid4().hex[:12]}"
        self.sessions: Dict[str, StudioSession] = {}
        self.workflow_templates: Dict[str, StudioWorkflow] = {}
        self.agent_library: Dict[str, Dict[str, Any]] = {}
        self.execution_engine = None
        self.visualization_engine = None
        self._load_default_templates()
        self._load_agent_library()
    
    def _load_default_templates(self):
        """Load default workflow templates"""
        # Simple test workflow
        simple_test = StudioWorkflow(
            name="Simple Test Workflow",
            description="Basic test execution workflow"
        )
        simple_test.add_node("start", {"x": 100, "y": 100})
        simple_test.add_node("test_agent", {"x": 300, "y": 100}, 
                           {"agent_id": "test_executor"})
        simple_test.add_node("end", {"x": 500, "y": 100})
        simple_test.add_edge(simple_test.nodes[0]["id"], simple_test.nodes[1]["id"])
        simple_test.add_edge(simple_test.nodes[1]["id"], simple_test.nodes[2]["id"])
        self.workflow_templates["simple_test"] = simple_test
        
        # Parallel test workflow
        parallel_test = StudioWorkflow(
            name="Parallel Test Workflow",
            description="Parallel test execution with multiple agents"
        )
        parallel_test.add_node("start", {"x": 100, "y": 200})
        parallel_test.add_node("split", {"x": 200, "y": 200})
        parallel_test.add_node("test_agent", {"x": 350, "y": 100}, 
                              {"agent_id": "unit_tester"})
        parallel_test.add_node("test_agent", {"x": 350, "y": 200}, 
                              {"agent_id": "integration_tester"})
        parallel_test.add_node("test_agent", {"x": 350, "y": 300}, 
                              {"agent_id": "performance_tester"})
        parallel_test.add_node("join", {"x": 500, "y": 200})
        parallel_test.add_node("end", {"x": 600, "y": 200})
        self.workflow_templates["parallel_test"] = parallel_test
    
    def _load_agent_library(self):
        """Load available agents"""
        self.agent_library = {
            "test_executor": {
                "name": "Test Executor",
                "type": "executor",
                "capabilities": ["unit_test", "integration_test"],
                "icon": "🧪",
                "color": "#4CAF50"
            },
            "test_analyzer": {
                "name": "Test Analyzer",
                "type": "analyzer",
                "capabilities": ["coverage_analysis", "quality_metrics"],
                "icon": "📊",
                "color": "#2196F3"
            },
            "test_reporter": {
                "name": "Test Reporter",
                "type": "reporter",
                "capabilities": ["html_report", "json_report", "pdf_report"],
                "icon": "📝",
                "color": "#FF9800"
            },
            "test_optimizer": {
                "name": "Test Optimizer",
                "type": "optimizer",
                "capabilities": ["test_selection", "parallel_optimization"],
                "icon": "⚡",
                "color": "#9C27B0"
            }
        }
    
    def create_session(self, user_id: str, 
                      mode: InteractionMode = InteractionMode.VISUAL) -> str:
        """Create a new studio session"""
        session = StudioSession(
            user_id=user_id,
            interaction_mode=mode
        )
        self.sessions[session.session_id] = session
        
        logger.info(f"Created studio session {session.session_id} for user {user_id}")
        return session.session_id
    
    def get_session(self, session_id: str) -> Optional[StudioSession]:
        """Get a studio session"""
        return self.sessions.get(session_id)
    
    def create_workflow(self, session_id: str, name: str, 
                       description: str = "") -> Optional[str]:
        """Create a new workflow in a session"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        workflow = StudioWorkflow(
            name=name,
            description=description
        )
        session.workflows.append(workflow)
        session.active_workflow = workflow
        
        session.record_action({
            "type": "create_workflow",
            "workflow_id": workflow.workflow_id,
            "timestamp": datetime.now().isoformat()
        })
        
        return workflow.workflow_id
    
    def load_template(self, session_id: str, template_name: str) -> Optional[str]:
        """Load a workflow template"""
        session = self.get_session(session_id)
        if not session or template_name not in self.workflow_templates:
            return None
        
        # Clone template
        template = self.workflow_templates[template_name]
        workflow = StudioWorkflow(
            name=f"{template.name} (Copy)",
            description=template.description,
            nodes=template.nodes.copy(),
            edges=template.edges.copy(),
            configuration=template.configuration.copy()
        )
        
        session.workflows.append(workflow)
        session.active_workflow = workflow
        
        return workflow.workflow_id
    
    def add_agent_node(self, session_id: str, agent_id: str, 
                      position: Dict[str, float]) -> Optional[str]:
        """Add an agent node to the active workflow"""
        session = self.get_session(session_id)
        if not session or not session.active_workflow:
            return None
        
        if agent_id not in self.agent_library:
            return None
        
        agent_info = self.agent_library[agent_id]
        node_id = session.active_workflow.add_node(
            "test_agent",
            position,
            {
                "agent_id": agent_id,
                "name": agent_info["name"],
                "capabilities": agent_info["capabilities"]
            }
        )
        
        session.record_action({
            "type": "add_node",
            "node_id": node_id,
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat()
        })
        
        return node_id
    
    def connect_nodes(self, session_id: str, source_id: str, 
                     target_id: str, condition: Optional[str] = None) -> Optional[str]:
        """Connect two nodes in the workflow"""
        session = self.get_session(session_id)
        if not session or not session.active_workflow:
            return None
        
        edge_id = session.active_workflow.add_edge(source_id, target_id, condition)
        
        session.record_action({
            "type": "add_edge",
            "edge_id": edge_id,
            "source": source_id,
            "target": target_id,
            "timestamp": datetime.now().isoformat()
        })
        
        return edge_id
    
    def generate_code(self, session_id: str) -> Optional[str]:
        """Generate Python code from visual workflow"""
        session = self.get_session(session_id)
        if not session or not session.active_workflow:
            return None
        
        workflow = session.active_workflow
        
        # Validate workflow first
        is_valid, errors = workflow.validate()
        if not is_valid:
            logger.error(f"Workflow validation failed: {errors}")
            return None
        
        # Generate code
        code_lines = [
            "# Generated workflow code",
            f"# Workflow: {workflow.name}",
            f"# Generated at: {datetime.now().isoformat()}",
            "",
            "import asyncio",
            "from testmaster import AgentOrchestrator, TestAgent",
            "",
            f"async def run_{workflow.workflow_id.replace('-', '_')}():",
            "    # Initialize orchestrator",
            "    orchestrator = AgentOrchestrator()",
            ""
        ]
        
        # Add agents
        for node in workflow.nodes:
            if node["type"] == "test_agent":
                agent_id = node["config"].get("agent_id", "unknown")
                code_lines.append(f"    # Add {node['config'].get('name', 'agent')}")
                code_lines.append(f"    agent_{node['id']} = TestAgent('{agent_id}')")
                code_lines.append(f"    orchestrator.add_agent(agent_{node['id']})")
                code_lines.append("")
        
        # Add workflow logic
        code_lines.append("    # Execute workflow")
        code_lines.append("    results = await orchestrator.execute()")
        code_lines.append("    return results")
        code_lines.append("")
        code_lines.append("# Run workflow")
        code_lines.append("if __name__ == '__main__':")
        code_lines.append("    results = asyncio.run(run_workflow())")
        code_lines.append("    print(results)")
        
        return "\n".join(code_lines)
    
    def execute_workflow(self, session_id: str, 
                        callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute a workflow"""
        session = self.get_session(session_id)
        if not session or not session.active_workflow:
            return {"error": "No active workflow"}
        
        workflow = session.active_workflow
        
        # Validate workflow
        is_valid, errors = workflow.validate()
        if not is_valid:
            return {"error": f"Workflow validation failed: {errors}"}
        
        # Simulate execution (would use actual execution engine)
        execution_result = {
            "workflow_id": workflow.workflow_id,
            "status": "completed",
            "started_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "nodes_executed": len(workflow.nodes),
            "results": {}
        }
        
        # Execute each node (simplified)
        for node in workflow.nodes:
            node_result = {
                "status": "success",
                "output": f"Executed {node['type']} node"
            }
            execution_result["results"][node["id"]] = node_result
            
            # Update node status
            node["status"] = "completed"
            
            # Callback for real-time updates
            if callback:
                callback({
                    "event": "node_completed",
                    "node_id": node["id"],
                    "result": node_result
                })
        
        return execution_result
    
    def get_visualization(self, session_id: str, 
                         viz_type: VisualizationType) -> Dict[str, Any]:
        """Get visualization data for the workflow"""
        session = self.get_session(session_id)
        if not session or not session.active_workflow:
            return {}
        
        workflow = session.active_workflow
        
        if viz_type == VisualizationType.WORKFLOW:
            return {
                "nodes": workflow.nodes,
                "edges": workflow.edges,
                "layout": "dagre"  # Layout algorithm
            }
        
        elif viz_type == VisualizationType.AGENT_GRAPH:
            # Build agent interaction graph
            agents = {}
            interactions = []
            
            for node in workflow.nodes:
                if node["type"] == "test_agent":
                    agent_id = node["config"].get("agent_id")
                    if agent_id:
                        agents[node["id"]] = self.agent_library.get(agent_id, {})
            
            for edge in workflow.edges:
                if edge["source"] in agents and edge["target"] in agents:
                    interactions.append({
                        "source": edge["source"],
                        "target": edge["target"],
                        "weight": 1
                    })
            
            return {
                "agents": agents,
                "interactions": interactions
            }
        
        elif viz_type == VisualizationType.TIMELINE:
            # Build execution timeline
            events = []
            for i, node in enumerate(workflow.nodes):
                events.append({
                    "timestamp": i * 1000,  # Simulated timestamps
                    "event": f"Execute {node['type']}",
                    "node_id": node["id"]
                })
            
            return {"events": events}
        
        return {}
    
    def collaborate(self, session_id: str, 
                   action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle collaboration actions"""
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
        
        if action == "share":
            # Generate shareable link
            share_token = f"share_{uuid.uuid4().hex[:12]}"
            return {
                "share_token": share_token,
                "url": f"/studio/shared/{share_token}"
            }
        
        elif action == "comment":
            # Add comment to workflow
            if session.active_workflow:
                comment = {
                    "id": f"comment_{uuid.uuid4().hex[:8]}",
                    "user_id": session.user_id,
                    "text": data.get("text", ""),
                    "node_id": data.get("node_id"),
                    "timestamp": datetime.now().isoformat()
                }
                
                if "comments" not in session.active_workflow.metadata:
                    session.active_workflow.metadata["comments"] = []
                
                session.active_workflow.metadata["comments"].append(comment)
                return comment
        
        elif action == "cursor":
            # Share cursor position for real-time collaboration
            return {
                "user_id": session.user_id,
                "position": data.get("position", {"x": 0, "y": 0}),
                "timestamp": datetime.now().isoformat()
            }
        
        return {}
    
    def export_workflow(self, session_id: str, format: str = "json") -> Optional[str]:
        """Export workflow in various formats"""
        session = self.get_session(session_id)
        if not session or not session.active_workflow:
            return None
        
        workflow = session.active_workflow
        
        if format == "json":
            return json.dumps({
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "description": workflow.description,
                "nodes": workflow.nodes,
                "edges": workflow.edges,
                "configuration": workflow.configuration,
                "version": workflow.version,
                "exported_at": datetime.now().isoformat()
            }, indent=2, default=str)
        
        elif format == "yaml":
            # Would use yaml library in production
            yaml_lines = [
                f"workflow_id: {workflow.workflow_id}",
                f"name: {workflow.name}",
                f"description: {workflow.description}",
                "nodes:",
            ]
            for node in workflow.nodes:
                yaml_lines.append(f"  - id: {node['id']}")
                yaml_lines.append(f"    type: {node['type']}")
            
            return "\n".join(yaml_lines)
        
        elif format == "python":
            return self.generate_code(session_id)
        
        return None
    
    def get_studio_status(self) -> Dict[str, Any]:
        """Get studio status and statistics"""
        active_sessions = sum(
            1 for s in self.sessions.values()
            if (datetime.now() - s.last_activity).total_seconds() < 3600
        )
        
        return {
            "studio_id": self.studio_id,
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "templates_available": len(self.workflow_templates),
            "agents_available": len(self.agent_library),
            "features": {
                "visual_builder": True,
                "code_generation": True,
                "real_time_execution": True,
                "collaboration": True,
                "template_library": True,
                "undo_redo": True,
                "export_import": True
            }
        }