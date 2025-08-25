"""
AgentVerse UI System
====================

AgentVerse-inspired visualization and interaction system.
Provides immersive agent world visualization and interaction.

Author: TestMaster Team
"""

import json
import uuid
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import math
import logging

# Configure logging
logger = logging.getLogger(__name__)


class LayoutAlgorithm(Enum):
    """Graph layout algorithms"""
    FORCE_DIRECTED = "force_directed"
    HIERARCHICAL = "hierarchical"
    CIRCULAR = "circular"
    GRID = "grid"
    RADIAL = "radial"
    SPRING = "spring"


class InteractionType(Enum):
    """Types of agent interactions"""
    MESSAGE = "message"
    TASK_ASSIGNMENT = "task_assignment"
    RESULT_SHARING = "result_sharing"
    KNOWLEDGE_TRANSFER = "knowledge_transfer"
    COORDINATION = "coordination"
    CONFLICT_RESOLUTION = "conflict_resolution"


@dataclass
class AgentCard:
    """Visual representation of an agent"""
    agent_id: str
    name: str
    type: str
    status: str = "idle"
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    color: str = "#4CAF50"
    icon: str = "🤖"
    size: float = 1.0
    capabilities: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_position(self, x: float, y: float, z: float = 0):
        """Update agent position in 3D space"""
        self.position = {"x": x, "y": y, "z": z}
    
    def get_distance_to(self, other: 'AgentCard') -> float:
        """Calculate distance to another agent"""
        dx = self.position["x"] - other.position["x"]
        dy = self.position["y"] - other.position["y"]
        dz = self.position["z"] - other.position["z"]
        return math.sqrt(dx*dx + dy*dy + dz*dz)


@dataclass
class InteractionEdge:
    """Represents an interaction between agents"""
    edge_id: str = field(default_factory=lambda: f"edge_{uuid.uuid4().hex[:8]}")
    source: str = ""
    target: str = ""
    interaction_type: InteractionType = InteractionType.MESSAGE
    weight: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    animated: bool = True
    color: str = "#2196F3"
    style: str = "solid"  # solid, dashed, dotted
    
    def get_age_seconds(self) -> float:
        """Get age of interaction in seconds"""
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass
class WorkflowDiagram:
    """Visual workflow representation"""
    workflow_id: str = field(default_factory=lambda: f"workflow_{uuid.uuid4().hex[:12]}")
    name: str = ""
    stages: List[Dict[str, Any]] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    current_stage: Optional[str] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_stage(self, name: str, agents: List[str], duration: float = 0) -> str:
        """Add a workflow stage"""
        stage_id = f"stage_{uuid.uuid4().hex[:8]}"
        stage = {
            "id": stage_id,
            "name": name,
            "agents": agents,
            "duration": duration,
            "status": "pending",
            "started_at": None,
            "completed_at": None
        }
        self.stages.append(stage)
        return stage_id
    
    def update_progress(self):
        """Update workflow progress"""
        if not self.stages:
            self.progress = 0.0
            return
        
        completed = sum(1 for s in self.stages if s["status"] == "completed")
        self.progress = (completed / len(self.stages)) * 100


class InteractionGraph:
    """Agent interaction graph"""
    
    def __init__(self):
        self.nodes: Dict[str, AgentCard] = {}
        self.edges: List[InteractionEdge] = []
        self.layout_algorithm = LayoutAlgorithm.FORCE_DIRECTED
        self.bounds = {"width": 1000, "height": 800, "depth": 500}
        self.center = {"x": 500, "y": 400, "z": 250}
        
    def add_agent(self, agent: AgentCard):
        """Add an agent to the graph"""
        self.nodes[agent.agent_id] = agent
        self._update_layout()
    
    def add_interaction(self, source_id: str, target_id: str, 
                       interaction_type: InteractionType, data: Dict[str, Any] = None):
        """Add an interaction between agents"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        
        edge = InteractionEdge(
            source=source_id,
            target=target_id,
            interaction_type=interaction_type,
            data=data or {}
        )
        self.edges.append(edge)
        
        # Update connections
        self.nodes[source_id].connections.append(target_id)
        if source_id != target_id:
            self.nodes[target_id].connections.append(source_id)
        
        return edge.edge_id
    
    def _update_layout(self):
        """Update graph layout based on algorithm"""
        if not self.nodes:
            return
        
        if self.layout_algorithm == LayoutAlgorithm.FORCE_DIRECTED:
            self._force_directed_layout()
        elif self.layout_algorithm == LayoutAlgorithm.HIERARCHICAL:
            self._hierarchical_layout()
        elif self.layout_algorithm == LayoutAlgorithm.CIRCULAR:
            self._circular_layout()
        elif self.layout_algorithm == LayoutAlgorithm.GRID:
            self._grid_layout()
        elif self.layout_algorithm == LayoutAlgorithm.RADIAL:
            self._radial_layout()
    
    def _force_directed_layout(self):
        """Apply force-directed layout algorithm"""
        # Simplified force-directed layout
        iterations = 50
        k = math.sqrt((self.bounds["width"] * self.bounds["height"]) / len(self.nodes))
        
        for _ in range(iterations):
            forces = {node_id: {"x": 0, "y": 0} for node_id in self.nodes}
            
            # Calculate repulsive forces
            for id1, node1 in self.nodes.items():
                for id2, node2 in self.nodes.items():
                    if id1 != id2:
                        dx = node1.position["x"] - node2.position["x"]
                        dy = node1.position["y"] - node2.position["y"]
                        dist = max(math.sqrt(dx*dx + dy*dy), 0.01)
                        
                        force = k * k / dist
                        forces[id1]["x"] += (dx / dist) * force
                        forces[id1]["y"] += (dy / dist) * force
            
            # Calculate attractive forces for connected nodes
            for edge in self.edges:
                if edge.source in self.nodes and edge.target in self.nodes:
                    node1 = self.nodes[edge.source]
                    node2 = self.nodes[edge.target]
                    
                    dx = node2.position["x"] - node1.position["x"]
                    dy = node2.position["y"] - node1.position["y"]
                    dist = max(math.sqrt(dx*dx + dy*dy), 0.01)
                    
                    force = dist * dist / k
                    forces[edge.source]["x"] += (dx / dist) * force * 0.1
                    forces[edge.source]["y"] += (dy / dist) * force * 0.1
                    forces[edge.target]["x"] -= (dx / dist) * force * 0.1
                    forces[edge.target]["y"] -= (dy / dist) * force * 0.1
            
            # Apply forces
            for node_id, force in forces.items():
                node = self.nodes[node_id]
                node.position["x"] = max(0, min(self.bounds["width"], 
                                               node.position["x"] + force["x"] * 0.1))
                node.position["y"] = max(0, min(self.bounds["height"], 
                                               node.position["y"] + force["y"] * 0.1))
    
    def _hierarchical_layout(self):
        """Apply hierarchical layout algorithm"""
        # Build hierarchy levels
        levels = {}
        visited = set()
        
        # Find root nodes (no incoming edges)
        roots = set(self.nodes.keys())
        for edge in self.edges:
            roots.discard(edge.target)
        
        if not roots:
            roots = {list(self.nodes.keys())[0]}
        
        # BFS to assign levels
        queue = [(root, 0) for root in roots]
        while queue:
            node_id, level = queue.pop(0)
            if node_id in visited:
                continue
            
            visited.add(node_id)
            if level not in levels:
                levels[level] = []
            levels[level].append(node_id)
            
            # Add children
            for edge in self.edges:
                if edge.source == node_id and edge.target not in visited:
                    queue.append((edge.target, level + 1))
        
        # Position nodes by level
        if levels:
            level_height = self.bounds["height"] / (len(levels) + 1)
            
            for level, nodes in levels.items():
                y = (level + 1) * level_height
                x_spacing = self.bounds["width"] / (len(nodes) + 1)
                
                for i, node_id in enumerate(nodes):
                    if node_id in self.nodes:
                        self.nodes[node_id].update_position(
                            (i + 1) * x_spacing, y
                        )
    
    def _circular_layout(self):
        """Apply circular layout algorithm"""
        if not self.nodes:
            return
        
        angle_step = 2 * math.pi / len(self.nodes)
        radius = min(self.bounds["width"], self.bounds["height"]) * 0.4
        
        for i, node in enumerate(self.nodes.values()):
            angle = i * angle_step
            x = self.center["x"] + radius * math.cos(angle)
            y = self.center["y"] + radius * math.sin(angle)
            node.update_position(x, y)
    
    def _grid_layout(self):
        """Apply grid layout algorithm"""
        if not self.nodes:
            return
        
        grid_size = math.ceil(math.sqrt(len(self.nodes)))
        cell_width = self.bounds["width"] / grid_size
        cell_height = self.bounds["height"] / grid_size
        
        for i, node in enumerate(self.nodes.values()):
            row = i // grid_size
            col = i % grid_size
            x = (col + 0.5) * cell_width
            y = (row + 0.5) * cell_height
            node.update_position(x, y)
    
    def _radial_layout(self):
        """Apply radial layout algorithm"""
        if not self.nodes:
            return
        
        # Find most connected node as center
        connectivity = {
            node_id: len(node.connections)
            for node_id, node in self.nodes.items()
        }
        
        if connectivity:
            center_id = max(connectivity, key=connectivity.get)
            
            # Position center node
            self.nodes[center_id].update_position(
                self.center["x"], self.center["y"]
            )
            
            # Position other nodes in rings
            remaining = [n for n in self.nodes.keys() if n != center_id]
            if remaining:
                angle_step = 2 * math.pi / len(remaining)
                radius = min(self.bounds["width"], self.bounds["height"]) * 0.3
                
                for i, node_id in enumerate(remaining):
                    angle = i * angle_step
                    x = self.center["x"] + radius * math.cos(angle)
                    y = self.center["y"] + radius * math.sin(angle)
                    self.nodes[node_id].update_position(x, y)
    
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data for visualization rendering"""
        return {
            "nodes": [
                {
                    "id": node.agent_id,
                    "label": node.name,
                    "x": node.position["x"],
                    "y": node.position["y"],
                    "z": node.position["z"],
                    "color": node.color,
                    "icon": node.icon,
                    "size": node.size * 20,
                    "status": node.status,
                    "capabilities": node.capabilities,
                    "metrics": node.metrics
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "id": edge.edge_id,
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.interaction_type.value,
                    "weight": edge.weight,
                    "color": edge.color,
                    "style": edge.style,
                    "animated": edge.animated,
                    "age": edge.get_age_seconds()
                }
                for edge in self.edges
            ],
            "layout": self.layout_algorithm.value,
            "bounds": self.bounds
        }


class AgentVisualization:
    """Agent visualization manager"""
    
    def __init__(self):
        self.viz_id = f"viz_{uuid.uuid4().hex[:12]}"
        self.graph = InteractionGraph()
        self.workflows: Dict[str, WorkflowDiagram] = {}
        self.animation_queue: List[Dict[str, Any]] = []
        self.view_settings = {
            "zoom": 1.0,
            "rotation": {"x": 0, "y": 0, "z": 0},
            "camera_position": {"x": 500, "y": 400, "z": 1000},
            "show_labels": True,
            "show_metrics": True,
            "show_connections": True,
            "animation_speed": 1.0
        }
    
    def create_agent_card(self, agent_id: str, name: str, 
                         agent_type: str, **kwargs) -> AgentCard:
        """Create a new agent card"""
        card = AgentCard(
            agent_id=agent_id,
            name=name,
            type=agent_type,
            **kwargs
        )
        self.graph.add_agent(card)
        return card
    
    def create_workflow(self, name: str) -> WorkflowDiagram:
        """Create a new workflow diagram"""
        workflow = WorkflowDiagram(name=name)
        self.workflows[workflow.workflow_id] = workflow
        return workflow
    
    def animate_interaction(self, source_id: str, target_id: str, 
                          animation_type: str = "pulse"):
        """Add animation to interaction"""
        self.animation_queue.append({
            "type": animation_type,
            "source": source_id,
            "target": target_id,
            "timestamp": datetime.now().isoformat(),
            "duration": 1000  # milliseconds
        })
    
    def update_view(self, settings: Dict[str, Any]):
        """Update view settings"""
        self.view_settings.update(settings)
    
    def get_scene_data(self) -> Dict[str, Any]:
        """Get complete scene data for rendering"""
        return {
            "graph": self.graph.get_visualization_data(),
            "workflows": {
                wf_id: {
                    "id": wf.workflow_id,
                    "name": wf.name,
                    "stages": wf.stages,
                    "transitions": wf.transitions,
                    "progress": wf.progress,
                    "current_stage": wf.current_stage
                }
                for wf_id, wf in self.workflows.items()
            },
            "animations": self.animation_queue[-10:],  # Last 10 animations
            "view_settings": self.view_settings
        }


class AgentVerseUI:
    """
    Main AgentVerse UI system with immersive visualization.
    
    Features:
    - 3D agent world visualization
    - Real-time interaction tracking
    - Dynamic graph layouts
    - Workflow visualization
    - Performance metrics overlay
    - Interactive agent cards
    - Collaboration paths
    """
    
    def __init__(self):
        self.ui_id = f"agentverse_{uuid.uuid4().hex[:12]}"
        self.visualizations: Dict[str, AgentVisualization] = {}
        self.active_viz: Optional[AgentVisualization] = None
        self.interaction_history: List[Dict[str, Any]] = []
        self.performance_data: Dict[str, List[float]] = {}
        
    def create_visualization(self, name: str) -> str:
        """Create a new visualization"""
        viz = AgentVisualization()
        self.visualizations[viz.viz_id] = viz
        self.active_viz = viz
        
        logger.info(f"Created visualization {viz.viz_id}: {name}")
        return viz.viz_id
    
    def add_agent(self, agent_id: str, name: str, 
                  agent_type: str, **kwargs) -> AgentCard:
        """Add an agent to the current visualization"""
        if not self.active_viz:
            self.create_visualization("Default")
        
        return self.active_viz.create_agent_card(agent_id, name, agent_type, **kwargs)
    
    def track_interaction(self, source_id: str, target_id: str, 
                         interaction_type: InteractionType, data: Dict[str, Any] = None):
        """Track an interaction between agents"""
        if not self.active_viz:
            return
        
        edge_id = self.active_viz.graph.add_interaction(
            source_id, target_id, interaction_type, data
        )
        
        # Add to history
        self.interaction_history.append({
            "edge_id": edge_id,
            "source": source_id,
            "target": target_id,
            "type": interaction_type.value,
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
        
        # Animate the interaction
        self.active_viz.animate_interaction(source_id, target_id)
        
        return edge_id
    
    def update_agent_status(self, agent_id: str, status: str, metrics: Dict[str, Any] = None):
        """Update agent status and metrics"""
        if not self.active_viz:
            return
        
        if agent_id in self.active_viz.graph.nodes:
            agent = self.active_viz.graph.nodes[agent_id]
            agent.status = status
            
            if metrics:
                agent.metrics.update(metrics)
                
                # Track performance data
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if key not in self.performance_data:
                            self.performance_data[key] = []
                        self.performance_data[key].append(value)
    
    def set_layout(self, algorithm: LayoutAlgorithm):
        """Change graph layout algorithm"""
        if self.active_viz:
            self.active_viz.graph.layout_algorithm = algorithm
            self.active_viz.graph._update_layout()
    
    def get_agent_details(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed agent information"""
        if not self.active_viz or agent_id not in self.active_viz.graph.nodes:
            return None
        
        agent = self.active_viz.graph.nodes[agent_id]
        
        # Calculate interaction statistics
        sent_count = sum(1 for e in self.active_viz.graph.edges if e.source == agent_id)
        received_count = sum(1 for e in self.active_viz.graph.edges if e.target == agent_id)
        
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "type": agent.type,
            "status": agent.status,
            "position": agent.position,
            "capabilities": agent.capabilities,
            "metrics": agent.metrics,
            "connections": agent.connections,
            "interaction_stats": {
                "sent": sent_count,
                "received": received_count,
                "total": sent_count + received_count
            },
            "metadata": agent.metadata
        }
    
    def get_interaction_matrix(self) -> Dict[str, Any]:
        """Get agent interaction matrix"""
        if not self.active_viz:
            return {}
        
        matrix = {}
        agents = list(self.active_viz.graph.nodes.keys())
        
        for source in agents:
            matrix[source] = {}
            for target in agents:
                count = sum(
                    1 for e in self.active_viz.graph.edges
                    if e.source == source and e.target == target
                )
                matrix[source][target] = count
        
        return {
            "agents": agents,
            "matrix": matrix,
            "total_interactions": len(self.active_viz.graph.edges)
        }
    
    def export_visualization(self, format: str = "json") -> str:
        """Export visualization data"""
        if not self.active_viz:
            return ""
        
        data = {
            "viz_id": self.active_viz.viz_id,
            "exported_at": datetime.now().isoformat(),
            "scene": self.active_viz.get_scene_data(),
            "interaction_history": self.interaction_history[-100:],  # Last 100
            "performance_data": self.performance_data
        }
        
        if format == "json":
            return json.dumps(data, indent=2, default=str)
        
        return str(data)
    
    def get_ui_status(self) -> Dict[str, Any]:
        """Get UI system status"""
        return {
            "ui_id": self.ui_id,
            "visualizations": len(self.visualizations),
            "active_viz": self.active_viz.viz_id if self.active_viz else None,
            "total_agents": len(self.active_viz.graph.nodes) if self.active_viz else 0,
            "total_interactions": len(self.interaction_history),
            "performance_metrics": list(self.performance_data.keys()),
            "features": {
                "3d_visualization": True,
                "real_time_tracking": True,
                "multiple_layouts": True,
                "workflow_support": True,
                "performance_overlay": True,
                "interaction_history": True,
                "export_capability": True
            }
        }