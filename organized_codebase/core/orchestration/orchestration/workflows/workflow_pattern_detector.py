"""
Workflow Pattern Detector for State Machines

This module detects and analyzes workflow patterns, state machines, and business
process flows in code. It identifies state transitions, workflow stages, and
process orchestration patterns.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging

from ..base import BaseAnalyzer


class WorkflowType(Enum):
    """Types of workflow patterns"""
    STATE_MACHINE = "state_machine"
    BUSINESS_PROCESS = "business_process"
    APPROVAL_WORKFLOW = "approval_workflow"
    ORDER_PROCESSING = "order_processing"
    DATA_PIPELINE = "data_pipeline"
    USER_JOURNEY = "user_journey"
    AUTHENTICATION_FLOW = "authentication_flow"
    PAYMENT_FLOW = "payment_flow"
    NOTIFICATION_FLOW = "notification_flow"
    ERROR_HANDLING_FLOW = "error_handling_flow"
    AUDIT_TRAIL = "audit_trail"
    BATCH_PROCESSING = "batch_processing"


class StateType(Enum):
    """Types of states in workflows"""
    INITIAL = "initial"
    INTERMEDIATE = "intermediate"
    DECISION = "decision"
    TERMINAL = "terminal"
    ERROR = "error"
    WAITING = "waiting"
    PROCESSING = "processing"
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"


class TransitionType(Enum):
    """Types of state transitions"""
    AUTOMATIC = "automatic"
    USER_TRIGGERED = "user_triggered"
    TIME_BASED = "time_based"
    CONDITION_BASED = "condition_based"
    EVENT_DRIVEN = "event_driven"
    ERROR_TRIGGERED = "error_triggered"


@dataclass
class WorkflowState:
    """Represents a state in a workflow"""
    name: str
    state_type: StateType
    description: str
    entry_actions: List[str]
    exit_actions: List[str]
    validations: List[str]
    timeouts: List[str]
    file_path: str
    line_number: int


@dataclass
class StateTransition:
    """Represents a transition between states"""
    from_state: str
    to_state: str
    trigger: str
    transition_type: TransitionType
    conditions: List[str]
    actions: List[str]
    guards: List[str]
    file_path: str
    line_number: int


@dataclass
class WorkflowPattern:
    """Represents a complete workflow pattern"""
    name: str
    workflow_type: WorkflowType
    states: List[WorkflowState]
    transitions: List[StateTransition]
    initial_state: Optional[str]
    final_states: List[str]
    error_states: List[str]
    complexity_score: int
    is_deterministic: bool
    has_loops: bool
    has_error_handling: bool
    has_timeouts: bool
    has_persistence: bool
    file_path: str
    line_number: int


@dataclass
class WorkflowIssue:
    """Represents an issue in workflow design"""
    workflow_name: str
    issue_type: str
    severity: str  # critical, high, medium, low
    description: str
    impact: str
    recommendation: str
    file_path: str
    line_number: int
    states_affected: List[str]


class WorkflowPatternDetector(BaseAnalyzer):
    """Detects workflow patterns and state machines"""
    
    def __init__(self):
        super().__init__()
        self.workflows: List[WorkflowPattern] = []
        self.states: List[WorkflowState] = []
        self.transitions: List[StateTransition] = []
        self.issues: List[WorkflowIssue] = []
        
        # Workflow framework patterns
        self.workflow_frameworks = {
            "django_fsm": ["from django_fsm", "FSMField", "@transition"],
            "transitions": ["from transitions", "Machine", "State"],
            "pytransitions": ["import transitions", "state_machine"],
            "statemachine": ["from statemachine", "StateMachine", "State"],
            "workflow": ["import workflow", "from workflow"],
            "celery": ["from celery", "@task", "chain", "group"],
            "airflow": ["from airflow", "DAG", "PythonOperator"],
            "prefect": ["from prefect", "@task", "@flow"],
            "dask": ["from dask", "delayed", "compute"],
        }
        
        # State keywords and patterns
        self.state_keywords = [
            "state", "status", "phase", "stage", "step",
            "pending", "processing", "completed", "failed",
            "approved", "rejected", "cancelled", "active",
            "inactive", "draft", "published", "archived"
        ]
        
        # Transition keywords
        self.transition_keywords = [
            "transition", "change", "move", "progress",
            "advance", "proceed", "complete", "finish",
            "approve", "reject", "cancel", "activate",
            "deactivate", "submit", "confirm", "retry"
        ]
        
        # Workflow patterns by industry/domain
        self.domain_patterns = {
            "ecommerce": {
                "order_states": ["cart", "checkout", "payment", "confirmed", 
                               "processing", "shipped", "delivered", "cancelled"],
                "payment_states": ["pending", "authorized", "captured", 
                                 "failed", "refunded", "disputed"]
            },
            "hr": {
                "application_states": ["submitted", "screening", "interview", 
                                     "offer", "accepted", "rejected"],
                "leave_states": ["requested", "pending", "approved", "rejected", "taken"]
            },
            "finance": {
                "loan_states": ["application", "review", "approved", "funded", 
                              "repaying", "closed", "defaulted"],
                "transaction_states": ["pending", "processing", "completed", 
                                     "failed", "reversed"]
            },
            "content": {
                "content_states": ["draft", "review", "approved", "published", 
                                 "archived", "deleted"],
                "moderation_states": ["submitted", "pending", "approved", 
                                    "rejected", "flagged"]
            }
        }
        
        # Anti-patterns in workflow design
        self.workflow_antipatterns = {
            "god_state": {
                "description": "State with too many transitions",
                "threshold": 10,
                "severity": "high"
            },
            "unreachable_state": {
                "description": "State with no incoming transitions",
                "severity": "medium"
            },
            "dead_end_state": {
                "description": "Non-terminal state with no outgoing transitions",
                "severity": "high"
            },
            "missing_error_handling": {
                "description": "No error states or error transitions",
                "severity": "medium"
            },
            "non_deterministic": {
                "description": "Multiple transitions possible from same state",
                "severity": "low"
            },
            "infinite_loop": {
                "description": "Possible infinite loops in state machine",
                "severity": "high"
            }
        }
        
    def analyze(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Analyze workflow patterns and state machines"""
        if file_path:
            self._analyze_file(Path(file_path))
        else:
            self._analyze_directory()
            
        return self._generate_report()
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single file for workflow patterns"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Check if file contains workflow-related code
            if not self._contains_workflow_code(content):
                return
            
            # Detect workflow frameworks
            frameworks = self._detect_frameworks(content)
            
            # Analyze different types of workflows
            self._analyze_state_machines(tree, content, str(file_path))
            self._analyze_business_processes(tree, content, str(file_path))
            self._analyze_enum_based_workflows(tree, content, str(file_path))
            self._analyze_class_based_workflows(tree, content, str(file_path))
            self._analyze_conditional_workflows(tree, content, str(file_path))
            
            # Detect workflow issues
            self._detect_workflow_issues(str(file_path))
            
        except Exception as e:
            logging.error(f"Error analyzing {file_path}: {e}")
    
    def _contains_workflow_code(self, content: str) -> bool:
        """Check if file contains workflow-related code"""
        workflow_indicators = (
            self.state_keywords + self.transition_keywords +
            ["workflow", "process", "fsm", "state_machine", "status"]
        )
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in workflow_indicators)
    
    def _detect_frameworks(self, content: str) -> Set[str]:
        """Detect workflow frameworks used"""
        frameworks = set()
        for framework, patterns in self.workflow_frameworks.items():
            for pattern in patterns:
                if pattern in content:
                    frameworks.add(framework)
                    break
        return frameworks
    
    def _analyze_state_machines(self, tree: ast.AST, content: str,
                               file_path: str) -> None:
        """Analyze explicit state machine implementations"""
        for node in ast.walk(tree):
            # Django FSM patterns
            if isinstance(node, ast.FunctionDef):
                decorators = [ast.unparse(d) for d in node.decorator_list]
                if any("transition" in dec for dec in decorators):
                    self._analyze_fsm_transition(node, content, file_path)
            
            # Class-based state machines
            elif isinstance(node, ast.ClassDef):
                if self._is_state_machine_class(node):
                    self._analyze_state_machine_class(node, content, file_path)
                
                # Enum-based states
                elif any(base.id == "Enum" for base in node.bases 
                        if isinstance(base, ast.Name)):
                    if self._is_state_enum(node):
                        self._analyze_state_enum(node, content, file_path)
    
    def _is_state_machine_class(self, node: ast.ClassDef) -> bool:
        """Check if class represents a state machine"""
        class_name_lower = node.name.lower()
        state_machine_indicators = [
            "statemachine", "state_machine", "fsm", "workflow",
            "process", "flow", "machine"
        ]
        return any(indicator in class_name_lower 
                  for indicator in state_machine_indicators)
    
    def _is_state_enum(self, node: ast.ClassDef) -> bool:
        """Check if enum represents states"""
        # Check class name
        class_name_lower = node.name.lower()
        if any(keyword in class_name_lower for keyword in 
              ["state", "status", "phase", "stage"]):
            return True
        
        # Check enum values
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        value_name_lower = target.id.lower()
                        if any(keyword in value_name_lower 
                              for keyword in self.state_keywords):
                            return True
        return False
    
    def _analyze_state_machine_class(self, node: ast.ClassDef, content: str,
                                   file_path: str) -> None:
        """Analyze a state machine class"""
        states = []
        transitions = []
        
        # Extract states and transitions from methods
        for method in node.body:
            if isinstance(method, ast.FunctionDef):
                method_name_lower = method.name.lower()
                
                # State-related methods
                if any(keyword in method_name_lower for keyword in self.state_keywords):
                    state = WorkflowState(
                        name=method.name,
                        state_type=self._infer_state_type(method.name),
                        description=self._extract_docstring(method),
                        entry_actions=self._extract_entry_actions(method),
                        exit_actions=self._extract_exit_actions(method),
                        validations=self._extract_validations(method),
                        timeouts=self._extract_timeouts(method),
                        file_path=file_path,
                        line_number=method.lineno
                    )
                    states.append(state)
                
                # Transition-related methods
                elif any(keyword in method_name_lower 
                        for keyword in self.transition_keywords):
                    transition = self._analyze_transition_method(method, file_path)
                    if transition:
                        transitions.append(transition)
        
        # Create workflow pattern
        if states or transitions:
            workflow = WorkflowPattern(
                name=node.name,
                workflow_type=self._infer_workflow_type(node.name, states),
                states=states,
                transitions=transitions,
                initial_state=self._find_initial_state(states, transitions),
                final_states=self._find_final_states(states, transitions),
                error_states=self._find_error_states(states),
                complexity_score=self._calculate_complexity(states, transitions),
                is_deterministic=self._is_deterministic(transitions),
                has_loops=self._has_loops(transitions),
                has_error_handling=len(self._find_error_states(states)) > 0,
                has_timeouts=any(state.timeouts for state in states),
                has_persistence=self._has_persistence(node),
                file_path=file_path,
                line_number=node.lineno
            )
            self.workflows.append(workflow)
            self.states.extend(states)
            self.transitions.extend(transitions)
    
    def _analyze_state_enum(self, node: ast.ClassDef, content: str,
                           file_path: str) -> None:
        """Analyze state enumeration"""
        states = []
        
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        state = WorkflowState(
                            name=target.id,
                            state_type=self._infer_state_type(target.id),
                            description=f"State from enum {node.name}",
                            entry_actions=[],
                            exit_actions=[],
                            validations=[],
                            timeouts=[],
                            file_path=file_path,
                            line_number=item.lineno
                        )
                        states.append(state)
        
        # Look for transitions in the same file
        transitions = self._find_enum_transitions(node.name, content, file_path)
        
        if states:
            workflow = WorkflowPattern(
                name=f"{node.name}_workflow",
                workflow_type=self._infer_workflow_type(node.name, states),
                states=states,
                transitions=transitions,
                initial_state=self._find_initial_state(states, transitions),
                final_states=self._find_final_states(states, transitions),
                error_states=self._find_error_states(states),
                complexity_score=self._calculate_complexity(states, transitions),
                is_deterministic=self._is_deterministic(transitions),
                has_loops=self._has_loops(transitions),
                has_error_handling=len(self._find_error_states(states)) > 0,
                has_timeouts=False,
                has_persistence=False,
                file_path=file_path,
                line_number=node.lineno
            )
            self.workflows.append(workflow)
            self.states.extend(states)
            self.transitions.extend(transitions)
    
    def _find_enum_transitions(self, enum_name: str, content: str,
                              file_path: str) -> List[StateTransition]:
        """Find transitions that use enum states"""
        transitions = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            line_clean = line.strip()
            
            # Look for assignment patterns like: status = Status.APPROVED
            if f"{enum_name}." in line_clean and "=" in line_clean:
                # Extract state change
                if " = " in line_clean:
                    left, right = line_clean.split(" = ", 1)
                    if f"{enum_name}." in right:
                        to_state = right.replace(f"{enum_name}.", "")
                        
                        transition = StateTransition(
                            from_state="unknown",
                            to_state=to_state,
                            trigger="assignment",
                            transition_type=TransitionType.AUTOMATIC,
                            conditions=[],
                            actions=[line_clean],
                            guards=[],
                            file_path=file_path,
                            line_number=i
                        )
                        transitions.append(transition)
        
        return transitions
    
    def _analyze_conditional_workflows(self, tree: ast.AST, content: str,
                                     file_path: str) -> None:
        """Analyze conditional-based workflows (if/elif chains)"""
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if self._is_workflow_conditional(node, content):
                    workflow = self._extract_conditional_workflow(node, content, file_path)
                    if workflow:
                        self.workflows.append(workflow)
    
    def _is_workflow_conditional(self, node: ast.If, content: str) -> bool:
        """Check if if-statement represents workflow logic"""
        # Check if conditions involve state-like variables
        conditions_str = ast.unparse(node.test).lower()
        if any(keyword in conditions_str for keyword in self.state_keywords):
            return True
        
        # Check if multiple elif branches (suggesting state transitions)
        elif_count = 0
        current = node
        while hasattr(current, 'orelse') and current.orelse:
            if isinstance(current.orelse[0], ast.If):
                elif_count += 1
                current = current.orelse[0]
            else:
                break
        
        return elif_count >= 2  # At least 3 branches total
    
    def _extract_conditional_workflow(self, node: ast.If, content: str,
                                    file_path: str) -> Optional[WorkflowPattern]:
        """Extract workflow from conditional logic"""
        states = set()
        transitions = []
        
        # Analyze if/elif chain
        current = node
        branch_count = 0
        
        while current:
            branch_count += 1
            
            # Extract condition
            condition = ast.unparse(current.test)
            
            # Look for state references in condition
            condition_lower = condition.lower()
            for keyword in self.state_keywords:
                if keyword in condition_lower:
                    states.add(keyword)
            
            # Extract actions in branch
            actions = []
            for stmt in current.body:
                if isinstance(stmt, ast.Assign):
                    assign_str = ast.unparse(stmt)
                    actions.append(assign_str)
                    
                    # Look for state assignments
                    for keyword in self.state_keywords:
                        if keyword in assign_str.lower():
                            states.add(keyword)
            
            # Create transition
            if actions:
                transition = StateTransition(
                    from_state="condition_check",
                    to_state=f"branch_{branch_count}",
                    trigger=condition,
                    transition_type=TransitionType.CONDITION_BASED,
                    conditions=[condition],
                    actions=actions,
                    guards=[],
                    file_path=file_path,
                    line_number=current.lineno
                )
                transitions.append(transition)
            
            # Move to next elif/else
            if hasattr(current, 'orelse') and current.orelse:
                if isinstance(current.orelse[0], ast.If):
                    current = current.orelse[0]
                else:
                    # Handle else clause
                    else_actions = []
                    for stmt in current.orelse:
                        if isinstance(stmt, ast.Assign):
                            else_actions.append(ast.unparse(stmt))
                    
                    if else_actions:
                        transition = StateTransition(
                            from_state="condition_check",
                            to_state="else_branch",
                            trigger="default",
                            transition_type=TransitionType.AUTOMATIC,
                            conditions=[],
                            actions=else_actions,
                            guards=[],
                            file_path=file_path,
                            line_number=current.orelse[0].lineno
                        )
                        transitions.append(transition)
                    break
            else:
                break
        
        if len(states) >= 2 and transitions:
            # Create state objects
            state_objects = [
                WorkflowState(
                    name=state,
                    state_type=self._infer_state_type(state),
                    description=f"Conditional workflow state",
                    entry_actions=[],
                    exit_actions=[],
                    validations=[],
                    timeouts=[],
                    file_path=file_path,
                    line_number=node.lineno
                )
                for state in states
            ]
            
            return WorkflowPattern(
                name=f"conditional_workflow_{node.lineno}",
                workflow_type=WorkflowType.BUSINESS_PROCESS,
                states=state_objects,
                transitions=transitions,
                initial_state=list(states)[0] if states else None,
                final_states=[],
                error_states=[],
                complexity_score=len(transitions),
                is_deterministic=True,
                has_loops=False,
                has_error_handling=False,
                has_timeouts=False,
                has_persistence=False,
                file_path=file_path,
                line_number=node.lineno
            )
        
        return None
    
    def _infer_state_type(self, state_name: str) -> StateType:
        """Infer state type from state name"""
        name_lower = state_name.lower()
        
        if any(keyword in name_lower for keyword in 
              ["initial", "start", "begin", "new", "created"]):
            return StateType.INITIAL
        elif any(keyword in name_lower for keyword in 
                ["final", "end", "complete", "finished", "done"]):
            return StateType.TERMINAL
        elif any(keyword in name_lower for keyword in 
                ["error", "failed", "exception", "invalid"]):
            return StateType.ERROR
        elif any(keyword in name_lower for keyword in 
                ["pending", "wait", "suspended", "paused"]):
            return StateType.WAITING
        elif any(keyword in name_lower for keyword in 
                ["process", "active", "running", "executing"]):
            return StateType.PROCESSING
        elif any(keyword in name_lower for keyword in 
                ["approved", "accepted", "confirmed"]):
            return StateType.APPROVED
        elif any(keyword in name_lower for keyword in 
                ["rejected", "denied", "cancelled"]):
            return StateType.REJECTED
        elif any(keyword in name_lower for keyword in 
                ["decision", "choice", "branch"]):
            return StateType.DECISION
        else:
            return StateType.INTERMEDIATE
    
    def _infer_workflow_type(self, name: str, states: List[WorkflowState]) -> WorkflowType:
        """Infer workflow type from name and states"""
        name_lower = name.lower()
        
        # Check name patterns
        if any(keyword in name_lower for keyword in 
              ["order", "purchase", "cart", "payment"]):
            return WorkflowType.ORDER_PROCESSING
        elif any(keyword in name_lower for keyword in 
                ["approval", "review", "authorize"]):
            return WorkflowType.APPROVAL_WORKFLOW
        elif any(keyword in name_lower for keyword in 
                ["auth", "login", "signup", "register"]):
            return WorkflowType.AUTHENTICATION_FLOW
        elif any(keyword in name_lower for keyword in 
                ["payment", "transaction", "billing"]):
            return WorkflowType.PAYMENT_FLOW
        elif any(keyword in name_lower for keyword in 
                ["notification", "alert", "message"]):
            return WorkflowType.NOTIFICATION_FLOW
        elif any(keyword in name_lower for keyword in 
                ["user", "journey", "experience"]):
            return WorkflowType.USER_JOURNEY
        elif any(keyword in name_lower for keyword in 
                ["batch", "job", "task"]):
            return WorkflowType.BATCH_PROCESSING
        elif any(keyword in name_lower for keyword in 
                ["audit", "log", "trace"]):
            return WorkflowType.AUDIT_TRAIL
        elif any(keyword in name_lower for keyword in 
                ["pipeline", "etl", "data"]):
            return WorkflowType.DATA_PIPELINE
        
        # Check state patterns
        state_names = [s.name.lower() for s in states]
        if any("error" in name for name in state_names):
            return WorkflowType.ERROR_HANDLING_FLOW
        
        return WorkflowType.BUSINESS_PROCESS
    
    def _calculate_complexity(self, states: List[WorkflowState], 
                            transitions: List[StateTransition]) -> int:
        """Calculate workflow complexity score"""
        complexity = 0
        
        # Base complexity from states and transitions
        complexity += len(states) * 2
        complexity += len(transitions)
        
        # Additional complexity for decision points
        decision_states = [s for s in states if s.state_type == StateType.DECISION]
        complexity += len(decision_states) * 3
        
        # Complexity for conditions and guards
        for transition in transitions:
            complexity += len(transition.conditions)
            complexity += len(transition.guards)
        
        # Complexity for error handling
        error_states = [s for s in states if s.state_type == StateType.ERROR]
        if error_states:
            complexity += len(error_states)
        else:
            complexity += 5  # Penalty for no error handling
        
        return complexity
    
    def _is_deterministic(self, transitions: List[StateTransition]) -> bool:
        """Check if workflow is deterministic"""
        # Group transitions by from_state
        state_transitions = {}
        for transition in transitions:
            from_state = transition.from_state
            if from_state not in state_transitions:
                state_transitions[from_state] = []
            state_transitions[from_state].append(transition)
        
        # Check for multiple transitions from same state with overlapping conditions
        for from_state, trans_list in state_transitions.items():
            if len(trans_list) > 1:
                # If multiple transitions but with mutually exclusive conditions
                # it's still deterministic (would need more sophisticated analysis)
                return False
        
        return True
    
    def _has_loops(self, transitions: List[StateTransition]) -> bool:
        """Check if workflow has loops"""
        # Build adjacency list
        graph = {}
        for transition in transitions:
            from_state = transition.from_state
            to_state = transition.to_state
            
            if from_state not in graph:
                graph[from_state] = []
            graph[from_state].append(to_state)
        
        # Check for cycles using DFS
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
        
        for state in graph:
            if state not in visited:
                if has_cycle(state):
                    return True
        
        return False
    
    def _find_initial_state(self, states: List[WorkflowState], 
                           transitions: List[StateTransition]) -> Optional[str]:
        """Find the initial state of the workflow"""
        # Look for explicitly marked initial states
        for state in states:
            if state.state_type == StateType.INITIAL:
                return state.name
        
        # Find state with no incoming transitions
        incoming = set()
        outgoing = set()
        
        for transition in transitions:
            incoming.add(transition.to_state)
            outgoing.add(transition.from_state)
        
        # States that have outgoing but no incoming transitions
        initial_candidates = outgoing - incoming
        return next(iter(initial_candidates)) if initial_candidates else None
    
    def _find_final_states(self, states: List[WorkflowState], 
                          transitions: List[StateTransition]) -> List[str]:
        """Find final states of the workflow"""
        # Look for explicitly marked terminal states
        terminal_states = [s.name for s in states if s.state_type == StateType.TERMINAL]
        
        # Find states with no outgoing transitions
        incoming = set()
        outgoing = set()
        
        for transition in transitions:
            incoming.add(transition.to_state)
            outgoing.add(transition.from_state)
        
        # States that have incoming but no outgoing transitions
        final_candidates = incoming - outgoing
        
        # Combine explicitly terminal and structurally final states
        all_final = set(terminal_states) | final_candidates
        return list(all_final)
    
    def _find_error_states(self, states: List[WorkflowState]) -> List[str]:
        """Find error states in the workflow"""
        return [s.name for s in states if s.state_type == StateType.ERROR]
    
    def _has_persistence(self, node: ast.AST) -> bool:
        """Check if workflow has persistence capabilities"""
        persistence_indicators = [
            "save", "persist", "database", "db", "store",
            "commit", "transaction", "session", "model"
        ]
        
        node_str = ast.unparse(node).lower()
        return any(indicator in node_str for indicator in persistence_indicators)
    
    def _detect_workflow_issues(self, file_path: str) -> None:
        """Detect issues in workflow designs"""
        for workflow in self.workflows:
            if workflow.file_path != file_path:
                continue
            
            # Check for god states
            self._check_god_states(workflow)
            
            # Check for unreachable states
            self._check_unreachable_states(workflow)
            
            # Check for dead-end states
            self._check_dead_end_states(workflow)
            
            # Check error handling
            self._check_error_handling(workflow)
            
            # Check for infinite loops
            self._check_infinite_loops(workflow)
            
            # Check for missing validations
            self._check_missing_validations(workflow)
    
    def _check_god_states(self, workflow: WorkflowPattern) -> None:
        """Check for states with too many transitions"""
        # Count outgoing transitions per state
        outgoing_count = {}
        for transition in workflow.transitions:
            from_state = transition.from_state
            outgoing_count[from_state] = outgoing_count.get(from_state, 0) + 1
        
        threshold = self.workflow_antipatterns["god_state"]["threshold"]
        for state_name, count in outgoing_count.items():
            if count >= threshold:
                self.issues.append(WorkflowIssue(
                    workflow_name=workflow.name,
                    issue_type="god_state",
                    severity="high",
                    description=f"State '{state_name}' has {count} outgoing transitions",
                    impact="Complex state logic, difficult to maintain",
                    recommendation="Split into multiple states or use sub-workflows",
                    file_path=workflow.file_path,
                    line_number=workflow.line_number,
                    states_affected=[state_name]
                ))
    
    def _check_unreachable_states(self, workflow: WorkflowPattern) -> None:
        """Check for states with no incoming transitions"""
        incoming_states = set()
        all_states = set(s.name for s in workflow.states)
        
        for transition in workflow.transitions:
            incoming_states.add(transition.to_state)
        
        # Initial state is reachable by definition
        if workflow.initial_state:
            incoming_states.add(workflow.initial_state)
        
        unreachable = all_states - incoming_states
        for state_name in unreachable:
            if state_name != workflow.initial_state:  # Don't flag initial state
                self.issues.append(WorkflowIssue(
                    workflow_name=workflow.name,
                    issue_type="unreachable_state",
                    severity="medium",
                    description=f"State '{state_name}' has no incoming transitions",
                    impact="State can never be reached during workflow execution",
                    recommendation="Add transition to this state or remove if unused",
                    file_path=workflow.file_path,
                    line_number=workflow.line_number,
                    states_affected=[state_name]
                ))
    
    def _check_dead_end_states(self, workflow: WorkflowPattern) -> None:
        """Check for non-terminal states with no outgoing transitions"""
        outgoing_states = set()
        terminal_states = set(workflow.final_states)
        all_states = set(s.name for s in workflow.states)
        
        for transition in workflow.transitions:
            outgoing_states.add(transition.from_state)
        
        dead_end_states = all_states - outgoing_states - terminal_states
        for state_name in dead_end_states:
            self.issues.append(WorkflowIssue(
                workflow_name=workflow.name,
                issue_type="dead_end_state",
                severity="high",
                description=f"Non-terminal state '{state_name}' has no outgoing transitions",
                impact="Workflow will get stuck in this state",
                recommendation="Add transitions from this state or mark as terminal",
                file_path=workflow.file_path,
                line_number=workflow.line_number,
                states_affected=[state_name]
            ))
    
    def _check_error_handling(self, workflow: WorkflowPattern) -> None:
        """Check for missing error handling"""
        if not workflow.error_states:
            self.issues.append(WorkflowIssue(
                workflow_name=workflow.name,
                issue_type="missing_error_handling",
                severity="medium",
                description="No error states defined in workflow",
                impact="No graceful handling of error conditions",
                recommendation="Add error states and error transitions",
                file_path=workflow.file_path,
                line_number=workflow.line_number,
                states_affected=[]
            ))
    
    def _check_infinite_loops(self, workflow: WorkflowPattern) -> None:
        """Check for potential infinite loops"""
        if workflow.has_loops and not workflow.has_timeouts:
            self.issues.append(WorkflowIssue(
                workflow_name=workflow.name,
                issue_type="infinite_loop_risk",
                severity="high",
                description="Workflow has loops but no timeout mechanisms",
                impact="Risk of infinite loops causing system hang",
                recommendation="Add timeout conditions or loop counters",
                file_path=workflow.file_path,
                line_number=workflow.line_number,
                states_affected=[]
            ))
    
    def _check_missing_validations(self, workflow: WorkflowPattern) -> None:
        """Check for missing state validations"""
        states_without_validation = []
        
        for state in workflow.states:
            if not state.validations and state.state_type not in [
                StateType.INITIAL, StateType.TERMINAL, StateType.ERROR
            ]:
                states_without_validation.append(state.name)
        
        if states_without_validation:
            self.issues.append(WorkflowIssue(
                workflow_name=workflow.name,
                issue_type="missing_validations",
                severity="low",
                description="Some states lack validation logic",
                impact="Invalid state transitions may be allowed",
                recommendation="Add validation logic to state entry/exit",
                file_path=workflow.file_path,
                line_number=workflow.line_number,
                states_affected=states_without_validation
            ))
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive workflow analysis report"""
        # Calculate statistics
        total_workflows = len(self.workflows)
        total_states = len(self.states)
        total_transitions = len(self.transitions)
        total_issues = len(self.issues)
        
        # Group by workflow type
        workflow_types = {}
        for workflow in self.workflows:
            wf_type = workflow.workflow_type.value
            workflow_types[wf_type] = workflow_types.get(wf_type, 0) + 1
        
        # Calculate complexity statistics
        if self.workflows:
            avg_complexity = sum(w.complexity_score for w in self.workflows) / total_workflows
            max_complexity = max(w.complexity_score for w in self.workflows)
            min_complexity = min(w.complexity_score for w in self.workflows)
        else:
            avg_complexity = max_complexity = min_complexity = 0
        
        # Group issues by severity
        issues_by_severity = {}
        for issue in self.issues:
            severity = issue.severity
            issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
        
        return {
            "summary": {
                "total_workflows": total_workflows,
                "total_states": total_states,
                "total_transitions": total_transitions,
                "total_issues": total_issues,
                "average_complexity": round(avg_complexity, 2),
                "max_complexity": max_complexity,
                "min_complexity": min_complexity,
                "deterministic_workflows": sum(1 for w in self.workflows if w.is_deterministic),
                "workflows_with_loops": sum(1 for w in self.workflows if w.has_loops),
                "workflows_with_error_handling": sum(1 for w in self.workflows if w.has_error_handling),
            },
            "workflows": [
                {
                    "name": w.name,
                    "type": w.workflow_type.value,
                    "file": w.file_path,
                    "line": w.line_number,
                    "states_count": len(w.states),
                    "transitions_count": len(w.transitions),
                    "complexity_score": w.complexity_score,
                    "initial_state": w.initial_state,
                    "final_states": w.final_states,
                    "error_states": w.error_states,
                    "is_deterministic": w.is_deterministic,
                    "has_loops": w.has_loops,
                    "has_error_handling": w.has_error_handling,
                    "has_timeouts": w.has_timeouts,
                    "has_persistence": w.has_persistence,
                }
                for w in self.workflows
            ],
            "workflow_types": workflow_types,
            "states": [
                {
                    "name": s.name,
                    "type": s.state_type.value,
                    "description": s.description,
                    "file": s.file_path,
                    "line": s.line_number,
                    "entry_actions": s.entry_actions,
                    "exit_actions": s.exit_actions,
                    "validations": s.validations,
                    "timeouts": s.timeouts,
                }
                for s in self.states
            ],
            "transitions": [
                {
                    "from_state": t.from_state,
                    "to_state": t.to_state,
                    "trigger": t.trigger,
                    "type": t.transition_type.value,
                    "conditions": t.conditions,
                    "actions": t.actions,
                    "guards": t.guards,
                    "file": t.file_path,
                    "line": t.line_number,
                }
                for t in self.transitions
            ],
            "issues": [
                {
                    "workflow": i.workflow_name,
                    "type": i.issue_type,
                    "severity": i.severity,
                    "description": i.description,
                    "impact": i.impact,
                    "recommendation": i.recommendation,
                    "file": i.file_path,
                    "line": i.line_number,
                    "states_affected": i.states_affected,
                }
                for i in sorted(self.issues, 
                              key=lambda x: {"critical": 0, "high": 1, 
                                           "medium": 2, "low": 3}[x.severity])
            ],
            "issues_by_severity": issues_by_severity,
            "recommendations": self._generate_recommendations(),
        }
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate workflow improvement recommendations"""
        recommendations = []
        
        # Check for complex workflows
        complex_workflows = [w for w in self.workflows if w.complexity_score > 20]
        if complex_workflows:
            recommendations.append({
                "category": "Complexity",
                "priority": "medium",
                "recommendation": "Simplify complex workflows",
                "impact": "Improve maintainability and reduce bugs",
                "affected_workflows": [w.name for w in complex_workflows]
            })
        
        # Check for missing error handling
        no_error_handling = [w for w in self.workflows if not w.has_error_handling]
        if no_error_handling:
            recommendations.append({
                "category": "Error Handling",
                "priority": "high",
                "recommendation": "Add error states and error handling",
                "impact": "Improve system resilience",
                "affected_workflows": [w.name for w in no_error_handling]
            })
        
        # Check for non-deterministic workflows
        non_deterministic = [w for w in self.workflows if not w.is_deterministic]
        if non_deterministic:
            recommendations.append({
                "category": "Determinism",
                "priority": "medium",
                "recommendation": "Make workflows deterministic",
                "impact": "Improve predictability and testing",
                "affected_workflows": [w.name for w in non_deterministic]
            })
        
        # Check for workflows without persistence
        no_persistence = [w for w in self.workflows if not w.has_persistence]
        if no_persistence:
            recommendations.append({
                "category": "Persistence",
                "priority": "low",
                "recommendation": "Consider adding state persistence",
                "impact": "Enable workflow recovery after failures",
                "affected_workflows": [w.name for w in no_persistence]
            })
        
        return recommendations