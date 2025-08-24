"""
Hierarchical Test Planning (HTP) Reasoning System

Implements hierarchical reasoning for systematic test generation planning.
Previously mislabeled as "Tree-of-Thought" - corrected as per integration roadmap.

This system breaks down complex test generation into hierarchical plans with
multiple levels of abstraction, evaluation, and refinement.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
import json
import time
from datetime import datetime
import heapq
from abc import ABC, abstractmethod


class PlanningStrategy(Enum):
    """Different strategies for hierarchical planning exploration."""
    BREADTH_FIRST = "breadth_first"      # Explore all options at each level
    DEPTH_FIRST = "depth_first"          # Go deep on promising paths
    BEST_FIRST = "best_first"            # Always expand best node (A* like)
    MONTE_CARLO = "monte_carlo"          # Random sampling with backpropagation
    BEAM_SEARCH = "beam_search"          # Keep only top-k paths at each level


@dataclass
class EvaluationCriteria:
    """Criteria for evaluating planning nodes."""
    name: str
    weight: float = 1.0
    evaluator: Optional[Callable] = None
    description: str = ""
    
    def evaluate(self, plan: Any) -> float:
        """Evaluate a plan based on this criterion."""
        if self.evaluator:
            return self.evaluator(plan) * self.weight
        return 0.0


@dataclass
class PlanningNode:
    """A single node in the hierarchical planning tree."""
    id: str
    content: Dict[str, Any]
    parent: Optional['PlanningNode'] = None
    children: List['PlanningNode'] = field(default_factory=list)
    
    # Evaluation scores
    raw_scores: Dict[str, float] = field(default_factory=dict)
    aggregate_score: float = 0.0
    confidence: float = 0.0
    
    # Node state
    depth: int = 0
    is_terminal: bool = False
    is_pruned: bool = False
    visit_count: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child: 'PlanningNode'):
        """Add a child node."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
    
    def update_score(self, criterion: str, score: float):
        """Update score for a specific criterion."""
        self.raw_scores[criterion] = score
        
    def calculate_aggregate_score(self, criteria: List[EvaluationCriteria]) -> float:
        """Calculate aggregate score from all criteria."""
        total_score = 0.0
        total_weight = 0.0
        
        for criterion in criteria:
            if criterion.name in self.raw_scores:
                total_score += self.raw_scores[criterion.name] * criterion.weight
                total_weight += criterion.weight
        
        self.aggregate_score = total_score / max(total_weight, 1.0)
        return self.aggregate_score
    
    def get_path(self) -> List['PlanningNode']:
        """Get path from root to this node."""
        path = []
        current = self
        while current:
            path.insert(0, current)
            current = current.parent
        return path
    
    def prune(self):
        """Mark this node as pruned."""
        self.is_pruned = True
        for child in self.children:
            child.prune()


@dataclass
class PlanningTree:
    """The complete hierarchical planning tree."""
    root: PlanningNode
    nodes: Dict[str, PlanningNode] = field(default_factory=dict)
    
    # Tree statistics
    total_nodes: int = 0
    max_depth: int = 0
    pruned_branches: int = 0
    
    def __post_init__(self):
        """Initialize tree with root node."""
        self.nodes[self.root.id] = self.root
        self.total_nodes = 1
    
    def add_node(self, parent_id: str, node: PlanningNode) -> PlanningNode:
        """Add a node to the tree."""
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id} not found")
        
        parent = self.nodes[parent_id]
        parent.add_child(node)
        self.nodes[node.id] = node
        self.total_nodes += 1
        self.max_depth = max(self.max_depth, node.depth)
        
        return node
    
    def get_leaf_nodes(self) -> List[PlanningNode]:
        """Get all leaf nodes (no children or terminal)."""
        leaves = []
        for node in self.nodes.values():
            if (not node.children or node.is_terminal) and not node.is_pruned:
                leaves.append(node)
        return leaves
    
    def get_best_plan(self) -> List[PlanningNode]:
        """Get the plan with highest aggregate score."""
        best_leaf = None
        best_score = float('-inf')
        
        for leaf in self.get_leaf_nodes():
            if leaf.aggregate_score > best_score:
                best_score = leaf.aggregate_score
                best_leaf = leaf
        
        if best_leaf:
            return best_leaf.get_path()
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tree statistics."""
        return {
            'total_nodes': self.total_nodes,
            'max_depth': self.max_depth,
            'leaf_nodes': len(self.get_leaf_nodes()),
            'pruned_branches': self.pruned_branches,
            'average_branching_factor': self._calculate_avg_branching()
        }
    
    def _calculate_avg_branching(self) -> float:
        """Calculate average branching factor."""
        non_leaf_nodes = [n for n in self.nodes.values() if n.children]
        if not non_leaf_nodes:
            return 0.0
        return sum(len(n.children) for n in non_leaf_nodes) / len(non_leaf_nodes)


class PlanGenerator(ABC):
    """Abstract base class for generating new plans."""
    
    @abstractmethod
    def generate(self, parent_node: PlanningNode, context: Dict[str, Any]) -> List[PlanningNode]:
        """Generate child plans from a parent node."""
        pass


class PlanEvaluator(ABC):
    """Abstract base class for evaluating plans."""
    
    @abstractmethod
    def evaluate(self, node: PlanningNode, criteria: List[EvaluationCriteria]) -> float:
        """Evaluate a planning node."""
        pass


class HierarchicalTestPlanner:
    """Main Hierarchical Test Planning reasoning engine."""
    
    def __init__(self,
                 plan_generator: PlanGenerator,
                 plan_evaluator: PlanEvaluator,
                 strategy: PlanningStrategy = PlanningStrategy.BEST_FIRST,
                 max_depth: int = 5,
                 max_iterations: int = 100,
                 beam_width: int = 3):
        
        self.plan_generator = plan_generator
        self.plan_evaluator = plan_evaluator
        self.strategy = strategy
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.beam_width = beam_width
        
        # Evaluation criteria
        self.criteria: List[EvaluationCriteria] = []
        
        # Statistics
        self.iterations = 0
        self.nodes_evaluated = 0
        self.nodes_generated = 0
        
        print(f"Hierarchical Test Planner initialized")
        print(f"   Strategy: {strategy.value}")
        print(f"   Max depth: {max_depth}")
        print(f"   Max iterations: {max_iterations}")
    
    def add_criterion(self, criterion: EvaluationCriteria):
        """Add an evaluation criterion."""
        self.criteria.append(criterion)
    
    def plan(self, initial_plan: Dict[str, Any], context: Dict[str, Any] = None) -> PlanningTree:
        """Execute Hierarchical Test Planning."""
        context = context or {}
        
        # Initialize tree with root
        root = PlanningNode(
            id="root",
            content=initial_plan,
            depth=0
        )
        tree = PlanningTree(root=root)
        
        print(f"\nStarting Hierarchical Test Planning...")
        
        # Execute strategy
        if self.strategy == PlanningStrategy.BREADTH_FIRST:
            self._breadth_first_search(tree, context)
        elif self.strategy == PlanningStrategy.DEPTH_FIRST:
            self._depth_first_search(tree, context)
        elif self.strategy == PlanningStrategy.BEST_FIRST:
            self._best_first_search(tree, context)
        elif self.strategy == PlanningStrategy.BEAM_SEARCH:
            self._beam_search(tree, context)
        elif self.strategy == PlanningStrategy.MONTE_CARLO:
            self._monte_carlo_search(tree, context)
        
        print(f"\nPlanning complete:")
        print(f"   Iterations: {self.iterations}")
        print(f"   Nodes generated: {self.nodes_generated}")
        print(f"   Nodes evaluated: {self.nodes_evaluated}")
        print(f"   Tree depth: {tree.max_depth}")
        print(f"   Total nodes: {tree.total_nodes}")
        
        return tree
    
    def _breadth_first_search(self, tree: PlanningTree, context: Dict[str, Any]):
        """Breadth-first exploration of planning tree."""
        current_level = [tree.root]
        
        for depth in range(self.max_depth):
            if self.iterations >= self.max_iterations:
                break
            
            next_level = []
            
            for node in current_level:
                if node.is_pruned:
                    continue
                
                # Generate children
                children = self._expand_node(node, tree, context)
                next_level.extend(children)
                
                self.iterations += 1
                
                if self.iterations >= self.max_iterations:
                    break
            
            current_level = next_level
            
            if not current_level:
                break
    
    def _best_first_search(self, tree: PlanningTree, context: Dict[str, Any]):
        """Best-first (A*-like) exploration of planning tree."""
        # Priority queue: (negative score for max-heap behavior, node)
        frontier = [(-tree.root.aggregate_score, tree.root.id, tree.root)]
        explored = set()
        
        while frontier and self.iterations < self.max_iterations:
            _, _, node = heapq.heappop(frontier)
            
            if node.id in explored:
                continue
            
            explored.add(node.id)
            
            if node.is_pruned or node.depth >= self.max_depth:
                continue
            
            # Expand node
            children = self._expand_node(node, tree, context)
            
            # Add children to frontier
            for child in children:
                if child.id not in explored:
                    heapq.heappush(frontier, (-child.aggregate_score, child.id, child))
            
            self.iterations += 1
    
    def _beam_search(self, tree: PlanningTree, context: Dict[str, Any]):
        """Beam search with limited beam width."""
        current_beam = [tree.root]
        
        for depth in range(self.max_depth):
            if self.iterations >= self.max_iterations:
                break
            
            next_candidates = []
            
            # Expand all nodes in current beam
            for node in current_beam:
                if node.is_pruned:
                    continue
                
                children = self._expand_node(node, tree, context)
                next_candidates.extend(children)
                
                self.iterations += 1
                
                if self.iterations >= self.max_iterations:
                    break
            
            # Keep only top beam_width candidates
            next_candidates.sort(key=lambda n: n.aggregate_score, reverse=True)
            current_beam = next_candidates[:self.beam_width]
            
            if not current_beam:
                break
    
    def _expand_node(self, node: PlanningNode, tree: PlanningTree, context: Dict[str, Any]) -> List[PlanningNode]:
        """Expand a node by generating and evaluating children."""
        if node.is_terminal or node.is_pruned:
            return []
        
        # Generate children
        children = self.plan_generator.generate(node, context)
        self.nodes_generated += len(children)
        
        # Add to tree and evaluate
        for child in children:
            tree.add_node(node.id, child)
            
            # Evaluate child
            score = self.plan_evaluator.evaluate(child, self.criteria)
            child.calculate_aggregate_score(self.criteria)
            self.nodes_evaluated += 1
        
        return children


def get_hierarchical_planner() -> HierarchicalTestPlanner:
    """Get hierarchical test planner instance."""
    from .test_plan_generator import TestPlanGenerator, TestPlanEvaluator
    
    generator = TestPlanGenerator()
    evaluator = TestPlanEvaluator()
    
    return HierarchicalTestPlanner(
        plan_generator=generator,
        plan_evaluator=evaluator,
        strategy=PlanningStrategy.BEST_FIRST,
        max_depth=4,
        max_iterations=50
    )