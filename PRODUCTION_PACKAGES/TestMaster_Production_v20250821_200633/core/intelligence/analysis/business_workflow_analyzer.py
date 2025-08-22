"""
Business Workflow Analysis Component
====================================

Analyzes workflow patterns, state machines, and business processes.
Part of modularized business_analyzer system.
"""

import ast
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict

from .business_base import (
    WorkflowState, DomainEntity, BusinessEvent, 
    BusinessAnalysisConfiguration
)


class BusinessWorkflowAnalyzer:
    """Analyzes business workflows and state machines"""
    
    def __init__(self, config: BusinessAnalysisConfiguration):
        self.config = config
    
    def analyze_workflows(self, python_files: List[Path]) -> Dict[str, Any]:
        """Analyze workflow patterns in the code"""
        workflow_analysis = {
            "workflows": [],
            "workflow_steps": [],
            "transitions": [],
            "approval_flows": [],
            "process_flows": [],
            "workflow_patterns": defaultdict(int)
        }
        
        for file_path in python_files:
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Look for workflow classes
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if any(pattern in node.name.lower() for pattern in ["workflow", "process", "flow", "pipeline"]):
                                workflow = self._analyze_workflow_class(node, file_path)
                                workflow_analysis["workflows"].append(workflow)
                                
                        # Look for workflow functions
                        elif isinstance(node, ast.FunctionDef):
                            if any(pattern in node.name.lower() for pattern in ["transition", "approve", "reject", "submit", "process"]):
                                step = self._analyze_workflow_step(node, file_path)
                                workflow_analysis["workflow_steps"].append(step)
                                
                                # Categorize workflow type
                                if "approve" in node.name.lower() or "reject" in node.name.lower():
                                    workflow_analysis["approval_flows"].append(step)
                                elif "process" in node.name.lower():
                                    workflow_analysis["process_flows"].append(step)
                                    
                        # Look for state transitions
                        elif isinstance(node, ast.If):
                            transition = self._detect_state_transition(node, file_path)
                            if transition:
                                workflow_analysis["transitions"].append(transition)
                                
            except Exception as e:
                print(f"Error analyzing workflows in {file_path}: {e}")
                
        # Identify workflow patterns
        for workflow in workflow_analysis["workflows"]:
            if "sequential" in str(workflow).lower():
                workflow_analysis["workflow_patterns"]["sequential"] += 1
            elif "parallel" in str(workflow).lower():
                workflow_analysis["workflow_patterns"]["parallel"] += 1
            elif "conditional" in str(workflow).lower():
                workflow_analysis["workflow_patterns"]["conditional"] += 1
                
        return workflow_analysis
    
    def detect_state_machines(self, python_files: List[Path]) -> Dict[str, Any]:
        """Detect and analyze state machine implementations"""
        state_machines = {
            "state_machines": [],
            "states": [],
            "transitions": [],
            "guards": [],
            "actions": [],
            "state_patterns": defaultdict(int)
        }
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = self._parse_file(file_path)
                if tree:
                    # Look for state machine classes
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if any(pattern in node.name.lower() for pattern in ["state", "machine", "fsm"]):
                                machine = self._analyze_state_machine(node, file_path)
                                state_machines["state_machines"].append(machine)
                                
                        # Look for state enums or constants
                        elif isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    if "state" in target.id.lower():
                                        states = self._extract_states(node)
                                        state_machines["states"].extend(states)
                                        
                        # Look for transition methods
                        elif isinstance(node, ast.FunctionDef):
                            if "transition" in node.name.lower():
                                transition = self._analyze_transition(node, file_path)
                                state_machines["transitions"].append(transition)
                            elif "guard" in node.name.lower() or "can_" in node.name.lower():
                                guard = self._analyze_guard(node, file_path)
                                state_machines["guards"].append(guard)
                            elif "on_enter" in node.name.lower() or "on_exit" in node.name.lower():
                                action = self._analyze_state_action(node, file_path)
                                state_machines["actions"].append(action)
                                
                # Detect state patterns
                if re.search(r"STATE_\w+", content):
                    state_machines["state_patterns"]["enum_states"] += 1
                if re.search(r"transition_table", content, re.IGNORECASE):
                    state_machines["state_patterns"]["transition_table"] += 1
                if re.search(r"state_diagram", content, re.IGNORECASE):
                    state_machines["state_patterns"]["documented"] += 1
                    
            except Exception as e:
                print(f"Error detecting state machines in {file_path}: {e}")
                
        return state_machines
    
    def extract_domain_model(self, python_files: List[Path]) -> Dict[str, Any]:
        """Extract domain model and entities"""
        domain_model = {
            "entities": [],
            "value_objects": [],
            "aggregates": [],
            "repositories": [],
            "services": [],
            "domain_events": [],
            "relationships": [],
            "invariants": []
        }
        
        for file_path in python_files:
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # Identify domain entities
                            entity = self._analyze_domain_entity(node, file_path)
                            if entity:
                                # Categorize entity type
                                if "entity" in node.name.lower():
                                    domain_model["entities"].append(entity)
                                elif "value" in node.name.lower() or "vo" in node.name.lower():
                                    domain_model["value_objects"].append(entity)
                                elif "aggregate" in node.name.lower() or "root" in node.name.lower():
                                    domain_model["aggregates"].append(entity)
                                elif "repository" in node.name.lower():
                                    domain_model["repositories"].append(entity)
                                elif "service" in node.name.lower():
                                    domain_model["services"].append(entity)
                                else:
                                    # Check if it looks like a domain entity
                                    if self._is_domain_entity(node):
                                        domain_model["entities"].append(entity)
                                        
                            # Extract relationships
                            relationships = self._extract_entity_relationships(node, file_path)
                            domain_model["relationships"].extend(relationships)
                            
                            # Extract invariants
                            invariants = self._extract_invariants(node, file_path)
                            domain_model["invariants"].extend(invariants)
                            
                        # Look for domain events
                        elif isinstance(node, ast.FunctionDef):
                            if "event" in node.name.lower() or "publish" in node.name.lower():
                                event = self._extract_domain_event(node, file_path)
                                if event:
                                    domain_model["domain_events"].append(event)
                                    
            except Exception as e:
                print(f"Error extracting domain model from {file_path}: {e}")
                
        return domain_model
    
    def extract_business_events(self, python_files: List[Path]) -> Dict[str, Any]:
        """Extract business events and event handlers"""
        events = {
            "event_definitions": [],
            "event_handlers": [],
            "event_publishers": [],
            "event_subscribers": [],
            "event_flows": [],
            "event_patterns": defaultdict(int)
        }
        
        for file_path in python_files:
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        # Look for event classes
                        if isinstance(node, ast.ClassDef):
                            if "event" in node.name.lower():
                                event_def = self._analyze_event_class(node, file_path)
                                events["event_definitions"].append(event_def)
                                
                        # Look for event handlers
                        elif isinstance(node, ast.FunctionDef):
                            if any(pattern in node.name.lower() for pattern in ["handle", "on_", "process_event"]):
                                handler = self._analyze_event_handler(node, file_path)
                                events["event_handlers"].append(handler)
                            elif "publish" in node.name.lower() or "emit" in node.name.lower():
                                publisher = self._analyze_event_publisher(node, file_path)
                                events["event_publishers"].append(publisher)
                            elif "subscribe" in node.name.lower() or "listen" in node.name.lower():
                                subscriber = self._analyze_event_subscriber(node, file_path)
                                events["event_subscribers"].append(subscriber)
                                
                        # Look for decorator-based event handling
                        if isinstance(node, ast.FunctionDef):
                            for decorator in node.decorator_list:
                                if isinstance(decorator, ast.Name):
                                    if "event" in decorator.id.lower():
                                        events["event_patterns"]["decorator_based"] += 1
                                        
            except Exception as e:
                print(f"Error extracting business events from {file_path}: {e}")
                
        # Analyze event flows
        events["event_flows"] = self._analyze_event_flows(events)
        
        return events
    
    def _parse_file(self, file_path: Path) -> Optional[ast.AST]:
        """Parse a Python file and return its AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return ast.parse(content)
        except Exception:
            return None
    
    def _is_domain_entity(self, node: ast.ClassDef) -> bool:
        """Check if a class represents a domain entity"""
        # Check for entity-like attributes
        has_id = False
        has_business_methods = False
        
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                if child.name == "__init__":
                    # Check for id attribute
                    for arg in child.args.args:
                        if "id" in arg.arg.lower():
                            has_id = True
                elif not child.name.startswith("_"):
                    # Public method that might be business logic
                    has_business_methods = True
                    
        return has_id or has_business_methods
    
    # Simplified placeholder methods for complex analysis functions
    def _analyze_workflow_class(self, node: ast.ClassDef, file_path: Path) -> Dict:
        """Analyze workflow class"""
        return {"class": node.name, "location": str(file_path), "type": "workflow"}
    
    def _analyze_workflow_step(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze workflow step"""
        return {"function": node.name, "location": str(file_path), "type": "step"}
    
    def _detect_state_transition(self, node: ast.If, file_path: Path) -> Optional[Dict]:
        """Detect state transition"""
        # Check if the conditional involves state changes
        condition_str = str(node.test) if hasattr(node, 'test') else ""
        if any(state_word in condition_str.lower() for state_word in ["state", "status", "phase"]):
            return {"type": "transition", "location": str(file_path)}
        return None
    
    def _analyze_state_machine(self, node: ast.ClassDef, file_path: Path) -> Dict:
        """Analyze state machine"""
        return {"class": node.name, "location": str(file_path), "type": "state_machine"}
    
    def _extract_states(self, node: ast.Assign) -> List[Dict]:
        """Extract states from assignment"""
        return [{"name": "extracted_state", "type": "state"}]
    
    def _analyze_transition(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze transition function"""
        return {"function": node.name, "location": str(file_path), "type": "transition"}
    
    def _analyze_guard(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze guard function"""
        return {"function": node.name, "location": str(file_path), "type": "guard"}
    
    def _analyze_state_action(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze state action"""
        return {"function": node.name, "location": str(file_path), "type": "action"}
    
    def _analyze_domain_entity(self, node: ast.ClassDef, file_path: Path) -> Optional[Dict]:
        """Analyze domain entity"""
        if self._is_domain_entity(node):
            return {"class": node.name, "location": str(file_path), "type": "entity"}
        return None
    
    def _extract_entity_relationships(self, node: ast.ClassDef, file_path: Path) -> List[Dict]:
        """Extract entity relationships"""
        return []
    
    def _extract_invariants(self, node: ast.ClassDef, file_path: Path) -> List[Dict]:
        """Extract invariants"""
        return []
    
    def _extract_domain_event(self, node: ast.FunctionDef, file_path: Path) -> Optional[Dict]:
        """Extract domain event"""
        return {"function": node.name, "location": str(file_path), "type": "event"}
    
    def _analyze_event_class(self, node: ast.ClassDef, file_path: Path) -> Dict:
        """Analyze event class"""
        return {"class": node.name, "location": str(file_path), "type": "event_definition"}
    
    def _analyze_event_handler(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze event handler"""
        return {"function": node.name, "location": str(file_path), "type": "handler"}
    
    def _analyze_event_publisher(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze event publisher"""
        return {"function": node.name, "location": str(file_path), "type": "publisher"}
    
    def _analyze_event_subscriber(self, node: ast.FunctionDef, file_path: Path) -> Dict:
        """Analyze event subscriber"""
        return {"function": node.name, "location": str(file_path), "type": "subscriber"}
    
    def _analyze_event_flows(self, events: Dict) -> List[Dict]:
        """Analyze event flows"""
        return [{"flow": "event_flow_analysis", "type": "flow"}]


# Export
__all__ = ['BusinessWorkflowAnalyzer']