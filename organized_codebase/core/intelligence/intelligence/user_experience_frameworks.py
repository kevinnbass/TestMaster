"""
User Experience Frameworks Module
=================================
Navigation, workflows, and user journey patterns from LLama-Agents and AgentScope.
Module size: ~299 lines (under 300 limit)

Patterns extracted from:
- LLama-Agents: Modern deployment UI and user-friendly interfaces  
- AgentScope: Studio navigation and project management workflows
- CrewAI: Flow-based user interactions and workflow design
- AutoGen: Clean chat interfaces and user interaction patterns
- Agency-Swarm: Tool creation workflows and user guidance
- PhiData: Cookbook navigation and example-driven UX
- Swarms: Intelligence coordination interfaces and user experience

Author: Agent D - Visualization Specialist
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from collections import deque
from abc import ABC, abstractmethod


@dataclass
class NavigationItem:
    """Navigation menu item."""
    id: str
    label: str
    path: str
    icon: str = ""
    children: List['NavigationItem'] = None
    permissions: List[str] = None
    metadata: Dict[str, Any] = None
    active: bool = False
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.permissions is None:
            self.permissions = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WorkflowStep:
    """Individual workflow step."""
    id: str
    title: str
    description: str
    step_type: str  # form, action, review, confirmation
    required: bool = True
    config: Dict[str, Any] = None
    validation: Dict[str, Any] = None
    next_step: str = ""
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}
        if self.validation is None:
            self.validation = {}


@dataclass
class UserJourney:
    """Complete user journey definition."""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    entry_points: List[str]
    completion_criteria: Dict[str, Any]
    analytics_events: List[str] = None
    
    def __post_init__(self):
        if self.analytics_events is None:
            self.analytics_events = []


class NavigationSystem:
    """Advanced navigation system (LLama-Agents pattern)."""
    
    def __init__(self):
        self.navigation_items: Dict[str, NavigationItem] = {}
        self.breadcrumbs: Dict[str, List[str]] = {}
        self.user_permissions: Dict[str, List[str]] = {}
        self.navigation_history: Dict[str, deque] = {}
        
    def add_navigation_item(self, item: NavigationItem, parent_id: str = "") -> str:
        """Add navigation item to hierarchy."""
        self.navigation_items[item.id] = item
        
        if parent_id and parent_id in self.navigation_items:
            parent = self.navigation_items[parent_id]
            parent.children.append(item)
            
        return item.id
        
    def build_navigation_tree(self, user_id: str = "") -> List[Dict[str, Any]]:
        """Build navigation tree for user with permissions."""
        user_perms = self.user_permissions.get(user_id, [])
        
        def build_item(item: NavigationItem) -> Optional[Dict[str, Any]]:
            # Check permissions
            if item.permissions and not any(perm in user_perms for perm in item.permissions):
                return None
                
            nav_item = {
                "id": item.id,
                "label": item.label,
                "path": item.path,
                "icon": item.icon,
                "active": item.active,
                "metadata": item.metadata,
                "children": []
            }
            
            # Add children
            for child in item.children:
                child_item = build_item(child)
                if child_item:
                    nav_item["children"].append(child_item)
                    
            return nav_item
            
        # Build root level items
        root_items = [item for item in self.navigation_items.values() 
                     if not any(item in other.children for other in self.navigation_items.values())]
        
        navigation_tree = []
        for item in root_items:
            built_item = build_item(item)
            if built_item:
                navigation_tree.append(built_item)
                
        return navigation_tree
        
    def set_active_path(self, path: str, user_id: str = ""):
        """Set active navigation path."""
        # Deactivate all items
        for item in self.navigation_items.values():
            item.active = False
            
        # Find and activate matching item
        matching_item = next((item for item in self.navigation_items.values() 
                            if item.path == path), None)
        
        if matching_item:
            matching_item.active = True
            self._update_breadcrumbs(matching_item.id, user_id)
            self._track_navigation(path, user_id)
            
    def get_breadcrumbs(self, user_id: str = "") -> List[Dict[str, Any]]:
        """Get breadcrumb navigation."""
        breadcrumb_ids = self.breadcrumbs.get(user_id, [])
        
        breadcrumbs = []
        for item_id in breadcrumb_ids:
            if item_id in self.navigation_items:
                item = self.navigation_items[item_id]
                breadcrumbs.append({
                    "id": item.id,
                    "label": item.label,
                    "path": item.path
                })
                
        return breadcrumbs
        
    def set_user_permissions(self, user_id: str, permissions: List[str]):
        """Set user permissions for navigation."""
        self.user_permissions[user_id] = permissions
        
    def _update_breadcrumbs(self, item_id: str, user_id: str):
        """Update breadcrumb trail."""
        # Build path to root
        path_to_root = []
        current_id = item_id
        
        while current_id:
            path_to_root.append(current_id)
            # Find parent
            parent = next((item for item in self.navigation_items.values() 
                         if any(child.id == current_id for child in item.children)), None)
            current_id = parent.id if parent else None
            
        self.breadcrumbs[user_id] = list(reversed(path_to_root))
        
    def _track_navigation(self, path: str, user_id: str):
        """Track navigation for analytics."""
        if user_id not in self.navigation_history:
            self.navigation_history[user_id] = deque(maxlen=100)
            
        self.navigation_history[user_id].append({
            "path": path,
            "timestamp": datetime.now()
        })


class WorkflowEngine:
    """Workflow management system (CrewAI pattern)."""
    
    def __init__(self):
        self.workflows: Dict[str, UserJourney] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.step_handlers: Dict[str, Callable] = {}
        
    def register_workflow(self, journey: UserJourney):
        """Register user journey workflow."""
        self.workflows[journey.id] = journey
        
    def start_workflow(self, workflow_id: str, user_id: str, initial_data: Dict[str, Any] = None) -> str:
        """Start workflow session for user."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        session_id = str(uuid.uuid4())
        workflow = self.workflows[workflow_id]
        
        self.active_sessions[session_id] = {
            "session_id": session_id,
            "workflow_id": workflow_id,
            "user_id": user_id,
            "current_step": workflow.steps[0].id if workflow.steps else "",
            "step_data": initial_data or {},
            "completed_steps": [],
            "started_at": datetime.now(),
            "status": "active"
        }
        
        return session_id
        
    def get_current_step(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow step for session."""
        if session_id not in self.active_sessions:
            return None
            
        session = self.active_sessions[session_id]
        workflow = self.workflows[session["workflow_id"]]
        
        current_step_id = session["current_step"]
        current_step = next((step for step in workflow.steps 
                           if step.id == current_step_id), None)
        
        if not current_step:
            return None
            
        return {
            "step": asdict(current_step),
            "session_info": {
                "session_id": session_id,
                "progress": len(session["completed_steps"]) / len(workflow.steps) * 100,
                "total_steps": len(workflow.steps),
                "completed_steps": len(session["completed_steps"])
            }
        }
        
    def complete_step(self, session_id: str, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete current workflow step."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
            
        session = self.active_sessions[session_id]
        workflow = self.workflows[session["workflow_id"]]
        
        current_step_id = session["current_step"]
        current_step = next((step for step in workflow.steps 
                           if step.id == current_step_id), None)
        
        if not current_step:
            return {"error": "Current step not found"}
            
        # Validate step data
        if not self._validate_step_data(current_step, step_data):
            return {"error": "Step validation failed"}
            
        # Store step data and mark as completed
        session["step_data"][current_step_id] = step_data
        session["completed_steps"].append(current_step_id)
        
        # Determine next step
        next_step_id = current_step.next_step
        if not next_step_id:
            # Find next step in sequence
            current_index = next((i for i, step in enumerate(workflow.steps) 
                                if step.id == current_step_id), -1)
            if current_index >= 0 and current_index < len(workflow.steps) - 1:
                next_step_id = workflow.steps[current_index + 1].id
                
        if next_step_id:
            session["current_step"] = next_step_id
            return {"success": True, "next_step": next_step_id}
        else:
            # Workflow complete
            session["status"] = "completed"
            session["completed_at"] = datetime.now()
            return {"success": True, "workflow_completed": True}
            
    def get_workflow_progress(self, session_id: str) -> Dict[str, Any]:
        """Get workflow progress information."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
            
        session = self.active_sessions[session_id]
        workflow = self.workflows[session["workflow_id"]]
        
        return {
            "session_id": session_id,
            "workflow_name": workflow.name,
            "total_steps": len(workflow.steps),
            "completed_steps": len(session["completed_steps"]),
            "current_step": session["current_step"],
            "progress_percentage": len(session["completed_steps"]) / len(workflow.steps) * 100,
            "status": session["status"],
            "started_at": session["started_at"]
        }
        
    def register_step_handler(self, step_type: str, handler: Callable):
        """Register handler for specific step type."""
        self.step_handlers[step_type] = handler
        
    def _validate_step_data(self, step: WorkflowStep, data: Dict[str, Any]) -> bool:
        """Validate step data against step requirements."""
        if not step.validation:
            return True
            
        # Simple validation - could be more sophisticated
        required_fields = step.validation.get("required_fields", [])
        for field in required_fields:
            if field not in data or not data[field]:
                return False
                
        return True


class ResponsiveUIFramework:
    """Responsive UI framework for different screen sizes."""
    
    def __init__(self):
        self.breakpoints = {
            "mobile": 768,
            "tablet": 1024, 
            "desktop": 1200,
            "wide": 1600
        }
        self.component_layouts = {}
        self.theme_configs = {}
        
    def register_component_layout(self, component_id: str, layouts: Dict[str, Dict[str, Any]]):
        """Register responsive layouts for component."""
        self.component_layouts[component_id] = layouts
        
    def get_component_layout(self, component_id: str, screen_width: int) -> Dict[str, Any]:
        """Get appropriate layout for screen size."""
        if component_id not in self.component_layouts:
            return {}
            
        layouts = self.component_layouts[component_id]
        
        # Determine breakpoint
        if screen_width < self.breakpoints["mobile"]:
            breakpoint = "mobile"
        elif screen_width < self.breakpoints["tablet"]:
            breakpoint = "tablet"
        elif screen_width < self.breakpoints["desktop"]:
            breakpoint = "desktop"
        else:
            breakpoint = "wide"
            
        # Return best matching layout
        for bp in [breakpoint, "desktop", "tablet", "mobile"]:
            if bp in layouts:
                return layouts[bp]
                
        return {}
        
    def set_theme_config(self, theme_name: str, config: Dict[str, Any]):
        """Set theme configuration."""
        self.theme_configs[theme_name] = config
        
    def get_theme_config(self, theme_name: str) -> Dict[str, Any]:
        """Get theme configuration."""
        return self.theme_configs.get(theme_name, {})
        
    def generate_css_grid(self, layout_config: Dict[str, Any]) -> str:
        """Generate CSS Grid configuration."""
        grid_template = layout_config.get("grid_template", "1fr")
        gap = layout_config.get("gap", "16px")
        
        css = f"""
        .grid-container {{
            display: grid;
            grid-template-columns: {grid_template};
            gap: {gap};
            width: 100%;
            height: 100%;
        }}
        """
        
        # Add responsive rules
        for breakpoint, width in self.breakpoints.items():
            if breakpoint in layout_config:
                bp_config = layout_config[breakpoint]
                bp_grid = bp_config.get("grid_template", grid_template)
                bp_gap = bp_config.get("gap", gap)
                
                css += f"""
                @media (max-width: {width}px) {{
                    .grid-container {{
                        grid-template-columns: {bp_grid};
                        gap: {bp_gap};
                    }}
                }}
                """
                
        return css


class AccessibilityManager:
    """Accessibility features management."""
    
    def __init__(self):
        self.a11y_config = {
            "high_contrast": False,
            "large_text": False,
            "reduce_motion": False,
            "screen_reader": False,
            "keyboard_navigation": True
        }
        self.aria_labels = {}
        self.focus_order = []
        
    def set_accessibility_option(self, option: str, enabled: bool):
        """Set accessibility option."""
        if option in self.a11y_config:
            self.a11y_config[option] = enabled
            
    def get_accessibility_css(self) -> str:
        """Generate accessibility CSS."""
        css = ""
        
        if self.a11y_config["high_contrast"]:
            css += """
            body { filter: contrast(150%); }
            """
            
        if self.a11y_config["large_text"]:
            css += """
            * { font-size: 120% !important; }
            """
            
        if self.a11y_config["reduce_motion"]:
            css += """
            * { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
            """
            
        return css
        
    def add_aria_label(self, element_id: str, label: str, description: str = ""):
        """Add ARIA label for element."""
        self.aria_labels[element_id] = {
            "label": label,
            "description": description
        }
        
    def get_aria_attributes(self, element_id: str) -> Dict[str, str]:
        """Get ARIA attributes for element."""
        if element_id not in self.aria_labels:
            return {}
            
        attrs = {"aria-label": self.aria_labels[element_id]["label"]}
        
        if self.aria_labels[element_id]["description"]:
            attrs["aria-describedby"] = f"{element_id}-description"
            
        return attrs


# Public API
__all__ = [
    'NavigationItem',
    'WorkflowStep',
    'UserJourney',
    'NavigationSystem',
    'WorkflowEngine',
    'ResponsiveUIFramework',
    'AccessibilityManager'
]