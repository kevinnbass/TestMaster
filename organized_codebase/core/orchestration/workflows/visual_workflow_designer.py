"""
Visual Workflow Designer
=======================

Drag-and-drop visual workflow designer built on the no-code dashboard infrastructure.
Enables intuitive creation of cross-system workflows with real-time validation.

Integrates with:
- No-Code Dashboard Builder for visual interface
- Workflow Framework for YAML generation
- Cross-System APIs for real-time validation
- Unified Dashboard for workflow monitoring

Author: TestMaster Phase 1B Integration System
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path

# Import workflow framework
from .workflow_framework import (
    WorkflowStepType, WorkflowStatus, StepStatus,
    WorkflowStep, WorkflowDefinition, WorkflowVariable,
    workflow_parser, workflow_templates
)

# Import cross-system APIs
from .cross_system_apis import SystemType, cross_system_coordinator


# ============================================================================
# VISUAL DESIGNER TYPES
# ============================================================================

class NodeType(Enum):
    """Visual node types in workflow designer"""
    START = "start"
    END = "end"
    SYSTEM_OPERATION = "system_operation"
    CONDITION = "condition"
    PARALLEL_SPLIT = "parallel_split"
    PARALLEL_JOIN = "parallel_join"
    LOOP = "loop"
    DELAY = "delay"
    HUMAN_TASK = "human_task"
    DATA_TRANSFORM = "data_transform"
    EXTERNAL_API = "external_api"


class ConnectionType(Enum):
    """Connection types between nodes"""
    SEQUENCE = "sequence"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    LOOP_BACK = "loop_back"


@dataclass
class VisualNode:
    """Visual node in workflow designer"""
    node_id: str = field(default_factory=lambda: f"node_{uuid.uuid4().hex[:8]}")
    node_type: NodeType = NodeType.SYSTEM_OPERATION
    title: str = ""
    description: str = ""
    
    # Visual properties
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0})
    size: Dict[str, float] = field(default_factory=lambda: {"width": 120, "height": 80})
    style: Dict[str, Any] = field(default_factory=dict)
    
    # Workflow properties
    target_system: Optional[SystemType] = None
    operation: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    
    # Designer state
    selected: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    def to_workflow_step(self) -> WorkflowStep:
        """Convert visual node to workflow step"""
        step_type = self._map_node_type_to_step_type()
        
        return WorkflowStep(
            step_id=self.node_id,
            name=self.title or self.node_id,
            type=step_type,
            target_system=self.target_system,
            operation=self.operation,
            parameters=self.parameters,
            conditions=self.conditions,
            timeout_seconds=self.timeout_seconds
        )
    
    def _map_node_type_to_step_type(self) -> WorkflowStepType:
        """Map visual node type to workflow step type"""
        mapping = {
            NodeType.SYSTEM_OPERATION: WorkflowStepType.SYSTEM_OPERATION,
            NodeType.CONDITION: WorkflowStepType.CONDITIONAL,
            NodeType.PARALLEL_SPLIT: WorkflowStepType.PARALLEL,
            NodeType.PARALLEL_JOIN: WorkflowStepType.PARALLEL,
            NodeType.LOOP: WorkflowStepType.LOOP,
            NodeType.DELAY: WorkflowStepType.DELAY,
            NodeType.HUMAN_TASK: WorkflowStepType.HUMAN_APPROVAL,
            NodeType.DATA_TRANSFORM: WorkflowStepType.DATA_TRANSFORM,
            NodeType.EXTERNAL_API: WorkflowStepType.EXTERNAL_API
        }
        
        return mapping.get(self.node_type, WorkflowStepType.SYSTEM_OPERATION)


@dataclass
class VisualConnection:
    """Connection between visual nodes"""
    connection_id: str = field(default_factory=lambda: f"conn_{uuid.uuid4().hex[:8]}")
    source_node_id: str = ""
    target_node_id: str = ""
    connection_type: ConnectionType = ConnectionType.SEQUENCE
    
    # Visual properties
    points: List[Dict[str, float]] = field(default_factory=list)
    style: Dict[str, Any] = field(default_factory=dict)
    
    # Logic properties
    condition_expression: Optional[str] = None
    label: str = ""
    
    # Designer state
    selected: bool = False
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class WorkflowDesignSession:
    """Visual workflow design session"""
    session_id: str = field(default_factory=lambda: f"design_{uuid.uuid4().hex[:12]}")
    workflow_name: str = "New Workflow"
    description: str = ""
    
    # Visual elements
    nodes: Dict[str, VisualNode] = field(default_factory=dict)
    connections: Dict[str, VisualConnection] = field(default_factory=dict)
    variables: List[WorkflowVariable] = field(default_factory=list)
    
    # Design state
    canvas_size: Dict[str, float] = field(default_factory=lambda: {"width": 1200, "height": 800})
    zoom_level: float = 1.0
    canvas_offset: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0})
    
    # Session tracking
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    def add_node(self, node: VisualNode) -> bool:
        """Add node to design session"""
        if node.node_id in self.nodes:
            return False
        
        self.nodes[node.node_id] = node
        self.last_modified = datetime.now()
        return True
    
    def add_connection(self, connection: VisualConnection) -> bool:
        """Add connection to design session"""
        # Validate connection
        if (connection.source_node_id not in self.nodes or 
            connection.target_node_id not in self.nodes):
            return False
        
        self.connections[connection.connection_id] = connection
        self.last_modified = datetime.now()
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """Remove node and associated connections"""
        if node_id not in self.nodes:
            return False
        
        # Remove node
        del self.nodes[node_id]
        
        # Remove associated connections
        connections_to_remove = []
        for conn_id, connection in self.connections.items():
            if (connection.source_node_id == node_id or 
                connection.target_node_id == node_id):
                connections_to_remove.append(conn_id)
        
        for conn_id in connections_to_remove:
            del self.connections[conn_id]
        
        self.last_modified = datetime.now()
        return True
    
    def validate_design(self) -> bool:
        """Validate the workflow design"""
        self.validation_errors = []
        
        # Check for start and end nodes
        has_start = any(node.node_type == NodeType.START for node in self.nodes.values())
        has_end = any(node.node_type == NodeType.END for node in self.nodes.values())
        
        if not has_start:
            self.validation_errors.append("Workflow must have a start node")
        if not has_end:
            self.validation_errors.append("Workflow must have an end node")
        
        # Check for orphaned nodes
        connected_nodes = set()
        for connection in self.connections.values():
            connected_nodes.add(connection.source_node_id)
            connected_nodes.add(connection.target_node_id)
        
        orphaned_nodes = set(self.nodes.keys()) - connected_nodes
        if orphaned_nodes:
            self.validation_errors.append(f"Orphaned nodes found: {', '.join(orphaned_nodes)}")
        
        # Check for cycles (except loop nodes)
        if self._has_invalid_cycles():
            self.validation_errors.append("Invalid cycles detected in workflow")
        
        # Validate individual nodes
        for node in self.nodes.values():
            node_errors = self._validate_node(node)
            node.validation_errors = node_errors
            self.validation_errors.extend([f"Node {node.node_id}: {error}" for error in node_errors])
        
        self.is_valid = len(self.validation_errors) == 0
        return self.is_valid
    
    def _validate_node(self, node: VisualNode) -> List[str]:
        """Validate individual node"""
        errors = []
        
        if not node.title.strip():
            errors.append("Node must have a title")
        
        if node.node_type == NodeType.SYSTEM_OPERATION:
            if not node.target_system:
                errors.append("System operation node must specify target system")
            if not node.operation:
                errors.append("System operation node must specify operation")
        
        elif node.node_type == NodeType.CONDITION:
            if not node.conditions:
                errors.append("Condition node must specify conditions")
        
        elif node.node_type == NodeType.DELAY:
            if "seconds" not in node.parameters:
                errors.append("Delay node must specify delay seconds")
        
        return errors
    
    def _has_invalid_cycles(self) -> bool:
        """Check for invalid cycles (excluding intentional loops)"""
        # Build adjacency list excluding loop-back connections
        graph = {}
        for node_id in self.nodes:
            graph[node_id] = []
        
        for connection in self.connections.values():
            if connection.connection_type != ConnectionType.LOOP_BACK:
                if connection.source_node_id in graph:
                    graph[connection.source_node_id].append(connection.target_node_id)
        
        # DFS cycle detection
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True
        
        return False
    
    def to_workflow_definition(self) -> WorkflowDefinition:
        """Convert design session to workflow definition"""
        if not self.validate_design():
            raise ValueError(f"Invalid workflow design: {'; '.join(self.validation_errors)}")
        
        # Convert nodes to steps
        steps = []
        for node in self.nodes.values():
            if node.node_type not in [NodeType.START, NodeType.END]:
                step = node.to_workflow_step()
                
                # Add dependencies based on connections
                dependencies = []
                for connection in self.connections.values():
                    if connection.target_node_id == node.node_id:
                        source_node = self.nodes.get(connection.source_node_id)
                        if source_node and source_node.node_type not in [NodeType.START]:
                            dependencies.append(connection.source_node_id)
                
                step.depends_on = dependencies
                steps.append(step)
        
        # Create workflow definition
        workflow = WorkflowDefinition(
            workflow_id=self.session_id,
            name=self.workflow_name,
            description=self.description,
            variables=self.variables.copy(),
            steps=steps
        )
        
        return workflow


# ============================================================================
# VISUAL WORKFLOW DESIGNER
# ============================================================================

class VisualWorkflowDesigner:
    """
    Main visual workflow designer component.
    Integrates with no-code dashboard builder for UI rendering.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("visual_workflow_designer")
        
        # Design sessions
        self.active_sessions: Dict[str, WorkflowDesignSession] = {}
        
        # Node templates
        self.node_templates: Dict[str, Dict[str, Any]] = {}
        
        # System operation catalog
        self.system_operations: Dict[SystemType, List[str]] = {}
        
        self._initialize_node_templates()
        self._load_system_operations()
        
        self.logger.info("Visual workflow designer initialized")
    
    def _initialize_node_templates(self):
        """Initialize node templates for the designer palette"""
        self.node_templates = {
            "start": {
                "node_type": NodeType.START,
                "title": "Start",
                "description": "Workflow start point",
                "style": {
                    "backgroundColor": "#4CAF50",
                    "borderColor": "#388E3C",
                    "textColor": "white",
                    "shape": "circle"
                },
                "size": {"width": 60, "height": 60}
            },
            "end": {
                "node_type": NodeType.END,
                "title": "End",
                "description": "Workflow end point",
                "style": {
                    "backgroundColor": "#F44336",
                    "borderColor": "#D32F2F",
                    "textColor": "white",
                    "shape": "circle"
                },
                "size": {"width": 60, "height": 60}
            },
            "system_operation": {
                "node_type": NodeType.SYSTEM_OPERATION,
                "title": "System Operation",
                "description": "Execute operation on unified system",
                "style": {
                    "backgroundColor": "#2196F3",
                    "borderColor": "#1976D2",
                    "textColor": "white",
                    "shape": "rectangle"
                },
                "size": {"width": 140, "height": 80}
            },
            "condition": {
                "node_type": NodeType.CONDITION,
                "title": "Condition",
                "description": "Conditional branching logic",
                "style": {
                    "backgroundColor": "#FF9800",
                    "borderColor": "#F57C00",
                    "textColor": "white",
                    "shape": "diamond"
                },
                "size": {"width": 100, "height": 100}
            },
            "parallel_split": {
                "node_type": NodeType.PARALLEL_SPLIT,
                "title": "Parallel Split",
                "description": "Split workflow into parallel branches",
                "style": {
                    "backgroundColor": "#9C27B0",
                    "borderColor": "#7B1FA2",
                    "textColor": "white",
                    "shape": "rectangle"
                },
                "size": {"width": 120, "height": 60}
            },
            "parallel_join": {
                "node_type": NodeType.PARALLEL_JOIN,
                "title": "Parallel Join",
                "description": "Join parallel branches",
                "style": {
                    "backgroundColor": "#9C27B0",
                    "borderColor": "#7B1FA2",
                    "textColor": "white",
                    "shape": "rectangle"
                },
                "size": {"width": 120, "height": 60}
            },
            "delay": {
                "node_type": NodeType.DELAY,
                "title": "Delay",
                "description": "Add time delay",
                "style": {
                    "backgroundColor": "#607D8B",
                    "borderColor": "#455A64",
                    "textColor": "white",
                    "shape": "rectangle"
                },
                "size": {"width": 100, "height": 60}
            },
            "human_task": {
                "node_type": NodeType.HUMAN_TASK,
                "title": "Human Task",
                "description": "Require human approval",
                "style": {
                    "backgroundColor": "#795548",
                    "borderColor": "#5D4037",
                    "textColor": "white",
                    "shape": "rectangle"
                },
                "size": {"width": 120, "height": 80}
            },
            "data_transform": {
                "node_type": NodeType.DATA_TRANSFORM,
                "title": "Data Transform",
                "description": "Transform data",
                "style": {
                    "backgroundColor": "#009688",
                    "borderColor": "#00695C",
                    "textColor": "white",
                    "shape": "rectangle"
                },
                "size": {"width": 130, "height": 80}
            }
        }
    
    def _load_system_operations(self):
        """Load available operations for each system"""
        # In production, this would query the actual systems
        self.system_operations = {
            SystemType.OBSERVABILITY: [
                "get_metrics", "start_monitoring", "stop_monitoring", 
                "get_analytics", "health_check", "get_alerts"
            ],
            SystemType.STATE_CONFIG: [
                "save_state", "load_state", "update_config", 
                "backup_state", "restore_state", "health_check"
            ],
            SystemType.ORCHESTRATION: [
                "start_workflow", "pause_workflow", "resume_workflow",
                "route_task", "get_agent_status", "health_check"
            ],
            SystemType.UI_DASHBOARD: [
                "create_dashboard", "update_widget", "export_layout",
                "refresh_data", "get_dashboard_data", "health_check"
            ]
        }
    
    def create_design_session(self, workflow_name: str = "New Workflow") -> str:
        """Create new workflow design session"""
        session = WorkflowDesignSession(workflow_name=workflow_name)
        self.active_sessions[session.session_id] = session
        
        self.logger.info(f"Created design session: {session.session_id}")
        return session.session_id
    
    def get_design_session(self, session_id: str) -> Optional[WorkflowDesignSession]:
        """Get design session by ID"""
        return self.active_sessions.get(session_id)
    
    def add_node_to_session(self, session_id: str, template_name: str, 
                          position: Dict[str, float]) -> Optional[str]:
        """Add node to design session from template"""
        session = self.get_design_session(session_id)
        if not session:
            return None
        
        template = self.node_templates.get(template_name)
        if not template:
            return None
        
        # Create node from template
        node = VisualNode(
            node_type=NodeType(template["node_type"]),
            title=template["title"],
            description=template["description"],
            position=position,
            size=template["size"].copy(),
            style=template["style"].copy()
        )
        
        if session.add_node(node):
            self.logger.info(f"Added node {node.node_id} to session {session_id}")
            return node.node_id
        
        return None
    
    def connect_nodes(self, session_id: str, source_node_id: str, 
                     target_node_id: str, connection_type: ConnectionType = ConnectionType.SEQUENCE) -> Optional[str]:
        """Connect two nodes in design session"""
        session = self.get_design_session(session_id)
        if not session:
            return None
        
        if source_node_id not in session.nodes or target_node_id not in session.nodes:
            return None
        
        connection = VisualConnection(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            connection_type=connection_type
        )
        
        if session.add_connection(connection):
            self.logger.info(f"Connected nodes {source_node_id} -> {target_node_id} in session {session_id}")
            return connection.connection_id
        
        return None
    
    def update_node_properties(self, session_id: str, node_id: str, 
                             properties: Dict[str, Any]) -> bool:
        """Update node properties"""
        session = self.get_design_session(session_id)
        if not session:
            return False
        
        node = session.nodes.get(node_id)
        if not node:
            return False
        
        # Update properties
        for key, value in properties.items():
            if hasattr(node, key):
                setattr(node, key, value)
        
        session.last_modified = datetime.now()
        self.logger.info(f"Updated node {node_id} properties in session {session_id}")
        return True
    
    def validate_session(self, session_id: str) -> Dict[str, Any]:
        """Validate design session"""
        session = self.get_design_session(session_id)
        if not session:
            return {"valid": False, "errors": ["Session not found"]}
        
        is_valid = session.validate_design()
        
        return {
            "valid": is_valid,
            "errors": session.validation_errors,
            "node_errors": {
                node_id: node.validation_errors 
                for node_id, node in session.nodes.items()
                if node.validation_errors
            }
        }
    
    def generate_workflow_yaml(self, session_id: str) -> Optional[str]:
        """Generate YAML workflow definition from design session"""
        session = self.get_design_session(session_id)
        if not session:
            return None
        
        try:
            workflow_def = session.to_workflow_definition()
            yaml_content = workflow_parser.export_workflow_to_yaml(workflow_def)
            
            self.logger.info(f"Generated YAML for session {session_id}")
            return yaml_content
            
        except Exception as e:
            self.logger.error(f"Failed to generate YAML for session {session_id}: {e}")
            return None
    
    def save_workflow_design(self, session_id: str, file_path: str) -> bool:
        """Save workflow design to file"""
        yaml_content = self.generate_workflow_yaml(session_id)
        if not yaml_content:
            return False
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            
            self.logger.info(f"Saved workflow design to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save workflow design: {e}")
            return False
    
    def load_workflow_design(self, file_path: str) -> Optional[str]:
        """Load workflow design from file into new session"""
        try:
            workflow_def = workflow_parser.parse_workflow_file(file_path)
            session_id = self._create_session_from_workflow(workflow_def)
            
            self.logger.info(f"Loaded workflow design from {file_path} into session {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to load workflow design: {e}")
            return None
    
    def _create_session_from_workflow(self, workflow_def: WorkflowDefinition) -> str:
        """Create design session from workflow definition"""
        session = WorkflowDesignSession(
            workflow_name=workflow_def.name,
            description=workflow_def.description,
            variables=workflow_def.variables.copy()
        )
        
        # Convert steps to visual nodes
        node_positions = self._calculate_auto_layout(workflow_def.steps)
        
        for i, step in enumerate(workflow_def.steps):
            node_type = self._map_step_type_to_node_type(step.type)
            
            node = VisualNode(
                node_id=step.step_id,
                node_type=node_type,
                title=step.name,
                target_system=step.target_system,
                operation=step.operation,
                parameters=step.parameters,
                conditions=step.conditions,
                timeout_seconds=step.timeout_seconds,
                position=node_positions.get(step.step_id, {"x": 100 + (i * 200), "y": 100})
            )
            
            session.add_node(node)
        
        # Create connections based on dependencies
        for step in workflow_def.steps:
            for dep in step.depends_on:
                connection = VisualConnection(
                    source_node_id=dep,
                    target_node_id=step.step_id,
                    connection_type=ConnectionType.SEQUENCE
                )
                session.add_connection(connection)
        
        self.active_sessions[session.session_id] = session
        return session.session_id
    
    def _map_step_type_to_node_type(self, step_type) -> NodeType:
        """Map workflow step type to visual node type"""
        mapping = {
            "system_operation": NodeType.SYSTEM_OPERATION,
            "conditional": NodeType.CONDITION,
            "parallel": NodeType.PARALLEL_SPLIT,
            "loop": NodeType.LOOP,
            "delay": NodeType.DELAY,
            "human_approval": NodeType.HUMAN_TASK,
            "data_transform": NodeType.DATA_TRANSFORM,
            "external_api": NodeType.EXTERNAL_API
        }
        
        return mapping.get(step_type.value if hasattr(step_type, 'value') else str(step_type), 
                          NodeType.SYSTEM_OPERATION)
    
    def _calculate_auto_layout(self, steps: List) -> Dict[str, Dict[str, float]]:
        """Calculate automatic layout positions for workflow steps"""
        positions = {}
        
        # Simple grid layout for now
        cols = 3
        x_spacing = 200
        y_spacing = 150
        start_x = 100
        start_y = 100
        
        for i, step in enumerate(steps):
            row = i // cols
            col = i % cols
            
            positions[step.step_id] = {
                "x": start_x + (col * x_spacing),
                "y": start_y + (row * y_spacing)
            }
        
        return positions
    
    def get_node_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available node templates for designer palette"""
        return self.node_templates.copy()
    
    def get_system_operations(self, system_type: SystemType) -> List[str]:
        """Get available operations for a system"""
        return self.system_operations.get(system_type, [])
    
    def get_designer_statistics(self) -> Dict[str, Any]:
        """Get designer usage statistics"""
        return {
            "active_sessions": len(self.active_sessions),
            "node_templates": len(self.node_templates),
            "system_types": len(self.system_operations),
            "sessions": [
                {
                    "session_id": session.session_id,
                    "workflow_name": session.workflow_name,
                    "node_count": len(session.nodes),
                    "connection_count": len(session.connections),
                    "is_valid": session.is_valid,
                    "last_modified": session.last_modified.isoformat()
                }
                for session in self.active_sessions.values()
            ]
        }


# ============================================================================
# GLOBAL VISUAL DESIGNER INSTANCE
# ============================================================================

# Global instance for visual workflow designer
visual_workflow_designer = VisualWorkflowDesigner()

# Export for external use
__all__ = [
    'NodeType',
    'ConnectionType',
    'VisualNode',
    'VisualConnection', 
    'WorkflowDesignSession',
    'VisualWorkflowDesigner',
    'visual_workflow_designer'
]