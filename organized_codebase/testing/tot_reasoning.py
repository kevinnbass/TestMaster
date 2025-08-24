"""
Core Tree-of-Thought Reasoning System

Implements the Tree-of-Thought algorithm for systematic reasoning about test generation.
Directly adapted from Swarm and Agency Swarm's multi-agent reasoning patterns.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
import json
import time
from datetime import datetime
import heapq
from abc import ABC, abstractmethod


class ReasoningStrategy(Enum):
    """Different strategies for tree exploration."""
    BREADTH_FIRST = "breadth_first"      # Explore all options at each level
    DEPTH_FIRST = "depth_first"          # Go deep on promising paths
    BEST_FIRST = "best_first"            # Always expand best node (A* like)
    MONTE_CARLO = "monte_carlo"          # Random sampling with backpropagation
    BEAM_SEARCH = "beam_search"          # Keep only top-k paths at each level


@dataclass
class EvaluationCriteria:
    """Criteria for evaluating thought nodes."""
    name: str
    weight: float = 1.0
    evaluator: Optional[Callable] = None
    description: str = ""
    
    def evaluate(self, thought: Any) -> float:
        """Evaluate a thought based on this criterion."""
        if self.evaluator:
            return self.evaluator(thought) * self.weight
        return 0.0


@dataclass
class ThoughtNode:
    """A single node in the thought tree."""
    id: str
    content: Dict[str, Any]
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = field(default_factory=list)
    
    # Evaluation scores
    scores: Dict[str, float] = field(default_factory=dict)
    aggregate_score: float = 0.0
    
    # Metadata
    depth: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    visits: int = 0
    
    # State
    is_terminal: bool = False
    is_expanded: bool = False
    is_pruned: bool = False
    
    def add_child(self, child: 'ThoughtNode') -> 'ThoughtNode':
        """Add a child node."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
        return child
    
    def get_path(self) -> List['ThoughtNode']:
        """Get the path from root to this node."""
        path = []
        current = self
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def prune_subtree(self):
        """Prune this node and all descendants."""
        self.is_pruned = True
        for child in self.children:
            child.prune_subtree()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            'id': self.id,
            'content': self.content,
            'scores': self.scores,
            'aggregate_score': self.aggregate_score,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'is_expanded': self.is_expanded,
            'is_pruned': self.is_pruned,
            'children': [child.id for child in self.children]
        }


@dataclass
class ThoughtTree:
    """The complete tree of thoughts."""
    root: ThoughtNode
    nodes: Dict[str, ThoughtNode] = field(default_factory=dict)
    
    # Tree statistics
    total_nodes: int = 0
    max_depth: int = 0
    pruned_branches: int = 0
    
    def __post_init__(self):
        """Initialize tree with root node."""
        self.nodes[self.root.id] = self.root
        self.total_nodes = 1
    
    def add_node(self, parent_id: str, node: ThoughtNode) -> ThoughtNode:
        """Add a node to the tree."""
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id} not found")
        
        parent = self.nodes[parent_id]
        parent.add_child(node)
        self.nodes[node.id] = node
        self.total_nodes += 1
        self.max_depth = max(self.max_depth, node.depth)
        
        return node
    
    def get_leaf_nodes(self) -> List[ThoughtNode]:
        """Get all leaf nodes (no children or terminal)."""
        leaves = []
        for node in self.nodes.values():
            if (not node.children or node.is_terminal) and not node.is_pruned:
                leaves.append(node)
        return leaves
    
    def get_best_path(self) -> List[ThoughtNode]:
        """Get the path with highest aggregate score."""
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
    
    def visualize(self, max_depth: int = None) -> str:
        """Generate a text visualization of the tree."""
        lines = []
        self._visualize_node(self.root, lines, "", True, max_depth, 0)
        return '\n'.join(lines)
    
    def _visualize_node(self, node: ThoughtNode, lines: List[str], prefix: str, 
                        is_last: bool, max_depth: Optional[int], current_depth: int):
        """Recursively visualize nodes."""
        if max_depth and current_depth > max_depth:
            return
        
        # Node representation
        connector = "└── " if is_last else "├── "
        node_str = f"{node.id} (score: {node.aggregate_score:.2f})"
        if node.is_terminal:
            node_str += " [TERMINAL]"
        if node.is_pruned:
            node_str += " [PRUNED]"
        
        lines.append(prefix + connector + node_str)
        
        # Prepare prefix for children
        extension = "    " if is_last else "│   "
        
        # Add children
        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            self._visualize_node(child, lines, prefix + extension, 
                                is_last_child, max_depth, current_depth + 1)


class ThoughtGenerator(ABC):
    """Abstract base class for generating new thoughts."""
    
    @abstractmethod
    def generate(self, parent_node: ThoughtNode, context: Dict[str, Any]) -> List[ThoughtNode]:
        """Generate child thoughts from a parent node."""
        pass


class ThoughtEvaluator(ABC):
    """Abstract base class for evaluating thoughts."""
    
    @abstractmethod
    def evaluate(self, node: ThoughtNode, criteria: List[EvaluationCriteria]) -> float:
        """Evaluate a thought node."""
        pass


class TreeOfThoughtReasoner:
    """Main Tree-of-Thought reasoning engine."""
    
    def __init__(self,
                 thought_generator: ThoughtGenerator,
                 thought_evaluator: ThoughtEvaluator,
                 strategy: ReasoningStrategy = ReasoningStrategy.BEST_FIRST,
                 max_depth: int = 5,
                 max_iterations: int = 100,
                 beam_width: int = 3):
        
        self.thought_generator = thought_generator
        self.thought_evaluator = thought_evaluator
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
        
        print(f"Tree-of-Thought Reasoner initialized")
        print(f"   Strategy: {strategy.value}")
        print(f"   Max depth: {max_depth}")
        print(f"   Max iterations: {max_iterations}")
    
    def add_criterion(self, criterion: EvaluationCriteria):
        """Add an evaluation criterion."""
        self.criteria.append(criterion)
    
    def reason(self, initial_thought: Dict[str, Any], context: Dict[str, Any] = None) -> ThoughtTree:
        """Execute Tree-of-Thought reasoning."""
        context = context or {}
        
        # Initialize tree with root
        root = ThoughtNode(
            id="root",
            content=initial_thought,
            depth=0
        )
        tree = ThoughtTree(root=root)
        
        print(f"\nStarting Tree-of-Thought reasoning...")
        
        # Execute strategy
        if self.strategy == ReasoningStrategy.BREADTH_FIRST:
            self._breadth_first_search(tree, context)
        elif self.strategy == ReasoningStrategy.DEPTH_FIRST:
            self._depth_first_search(tree, context)
        elif self.strategy == ReasoningStrategy.BEST_FIRST:
            self._best_first_search(tree, context)
        elif self.strategy == ReasoningStrategy.BEAM_SEARCH:
            self._beam_search(tree, context)
        elif self.strategy == ReasoningStrategy.MONTE_CARLO:
            self._monte_carlo_search(tree, context)
        
        print(f"\nReasoning complete:")
        print(f"   Iterations: {self.iterations}")
        print(f"   Nodes generated: {self.nodes_generated}")
        print(f"   Nodes evaluated: {self.nodes_evaluated}")
        print(f"   Tree depth: {tree.max_depth}")
        print(f"   Total nodes: {tree.total_nodes}")
        
        return tree
    
    def _breadth_first_search(self, tree: ThoughtTree, context: Dict[str, Any]):
        """Breadth-first exploration of thought tree."""
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
    
    def _depth_first_search(self, tree: ThoughtTree, context: Dict[str, Any]):
        """Depth-first exploration of thought tree."""
        stack = [tree.root]
        
        while stack and self.iterations < self.max_iterations:
            node = stack.pop()
            
            if node.is_pruned or node.depth >= self.max_depth:
                continue
            
            # Generate and evaluate children
            children = self._expand_node(node, tree, context)
            
            # Add children to stack (reverse order for left-to-right exploration)
            stack.extend(reversed(children))
            
            self.iterations += 1
    
    def _best_first_search(self, tree: ThoughtTree, context: Dict[str, Any]):
        """Best-first (A*-like) exploration of thought tree."""
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
            
            # Generate and evaluate children
            children = self._expand_node(node, tree, context)
            
            # Add children to frontier
            for child in children:
                if child.id not in explored:
                    heapq.heappush(frontier, (-child.aggregate_score, child.id, child))
            
            self.iterations += 1
    
    def _beam_search(self, tree: ThoughtTree, context: Dict[str, Any]):
        """Beam search exploration keeping only top-k paths."""
        current_beam = [tree.root]
        
        for depth in range(self.max_depth):
            if self.iterations >= self.max_iterations:
                break
            
            all_children = []
            
            # Generate children for all nodes in beam
            for node in current_beam:
                if node.is_pruned:
                    continue
                
                children = self._expand_node(node, tree, context)
                all_children.extend(children)
                
                self.iterations += 1
                
                if self.iterations >= self.max_iterations:
                    break
            
            if not all_children:
                break
            
            # Keep only top beam_width children
            all_children.sort(key=lambda x: x.aggregate_score, reverse=True)
            current_beam = all_children[:self.beam_width]
            
            # Prune nodes not in beam
            for child in all_children[self.beam_width:]:
                child.prune_subtree()
                tree.pruned_branches += 1
    
    def _monte_carlo_search(self, tree: ThoughtTree, context: Dict[str, Any]):
        """Monte Carlo Tree Search with UCB1 selection."""
        import math
        import random
        
        def ucb1_score(node: ThoughtNode, parent_visits: int, c: float = 1.414) -> float:
            """Calculate UCB1 score for node selection."""
            if node.visits == 0:
                return float('inf')
            
            exploitation = node.aggregate_score / node.visits
            exploration = c * math.sqrt(math.log(parent_visits) / node.visits)
            return exploitation + exploration
        
        for _ in range(self.max_iterations):
            # Selection: traverse tree using UCB1
            current = tree.root
            path = [current]
            
            while current.children and not current.is_terminal:
                # Select child with highest UCB1 score
                best_child = max(current.children, 
                               key=lambda c: ucb1_score(c, current.visits))
                current = best_child
                path.append(current)
            
            # Expansion: add new child if not terminal
            if not current.is_terminal and current.depth < self.max_depth:
                children = self._expand_node(current, tree, context)
                if children:
                    current = random.choice(children)
                    path.append(current)
            
            # Simulation: evaluate the node
            score = self.thought_evaluator.evaluate(current, self.criteria)
            
            # Backpropagation: update scores along path
            for node in path:
                node.visits += 1
                node.aggregate_score = ((node.aggregate_score * (node.visits - 1) + score) 
                                       / node.visits)
            
            self.iterations += 1
    
    def _expand_node(self, node: ThoughtNode, tree: ThoughtTree, context: Dict[str, Any]) -> List[ThoughtNode]:
        """Expand a node by generating and evaluating children."""
        if node.is_expanded:
            return node.children
        
        # Generate children
        children = self.thought_generator.generate(node, context)
        self.nodes_generated += len(children)
        
        # Add to tree and evaluate
        evaluated_children = []
        for child in children:
            # Add to tree
            tree_child = tree.add_node(node.id, child)
            
            # Evaluate
            score = self.thought_evaluator.evaluate(tree_child, self.criteria)
            tree_child.aggregate_score = score
            self.nodes_evaluated += 1
            
            evaluated_children.append(tree_child)
        
        node.is_expanded = True
        return evaluated_children


class SimpleThoughtGenerator(ThoughtGenerator):
    """Simple implementation of thought generator for testing."""
    
    def generate(self, parent_node: ThoughtNode, context: Dict[str, Any]) -> List[ThoughtNode]:
        """Generate simple child thoughts."""
        children = []
        
        # Generate 2-3 child thoughts
        num_children = 2 if parent_node.depth < 2 else 3
        
        for i in range(num_children):
            child = ThoughtNode(
                id=f"{parent_node.id}_child_{i}",
                content={
                    'thought': f"Expansion of {parent_node.id}",
                    'option': i,
                    'depth': parent_node.depth + 1
                }
            )
            children.append(child)
        
        return children


class SimpleThoughtEvaluator(ThoughtEvaluator):
    """Simple implementation of thought evaluator for testing."""
    
    def evaluate(self, node: ThoughtNode, criteria: List[EvaluationCriteria]) -> float:
        """Evaluate a thought node."""
        total_score = 0.0
        total_weight = 0.0
        
        for criterion in criteria:
            score = criterion.evaluate(node)
            total_score += score
            total_weight += criterion.weight
        
        if total_weight > 0:
            return total_score / total_weight
        
        # Default evaluation based on depth (prefer deeper thoughts)
        return 1.0 / (1.0 + node.depth)