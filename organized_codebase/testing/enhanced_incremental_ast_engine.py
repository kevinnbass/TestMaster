#!/usr/bin/env python3
"""
Enhanced Incremental AST Analysis Engine
=======================================
Advanced incremental AST analysis with fine-grained change detection,
semantic-aware diffing, and optimized re-analysis strategies.
"""

import ast
import os
import time
import hashlib
import pickle
import threading
import asyncio
import logging
import difflib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, PriorityQueue
from enum import Enum
import weakref

# Import the base classes - create minimal implementations if not available
try:
    from testmaster.analysis.comprehensive_analysis.realtime_analysis.realtime_ast_engine import (
        RealtimeASTEngine, ChangeType, Priority, FileChange, AnalysisResult, AnalysisTask,
        CacheManager, DependencyTracker, FileWatcher, AnalysisWorker
    )
except ImportError:
    # Create minimal base implementations
    from enum import Enum
    from dataclasses import dataclass
    from typing import Any, Dict, List, Optional
    
    class ChangeType(Enum):
        ADD = "add"
        MODIFY = "modify"
        DELETE = "delete"
    
    class Priority(Enum):
        LOW = 1
        MEDIUM = 2
        HIGH = 3
        CRITICAL = 4
    
    @dataclass
    class FileChange:
        path: str
        change_type: ChangeType
        timestamp: float
        content: Optional[str] = None
    
    @dataclass
    class AnalysisResult:
        file_path: str
        ast_tree: Optional[Any] = None
        errors: List[str] = field(default_factory=list)
        metrics: Dict[str, Any] = field(default_factory=dict)
        timestamp: float = 0.0
    
    @dataclass
    class AnalysisTask:
        file_path: str
        priority: Priority
        timestamp: float
    
    class RealtimeASTEngine:
        def __init__(self, config=None):
            self.cache = {}
            self.config = config or {}
            self.cache_manager = CacheManager()
            self.dependency_tracker = DependencyTracker()
            self.file_watcher = FileWatcher()
            self.workers = []
            self.task_queue = Queue()
            self.cache_dir = '.cache'  # Default cache directory
            
    class CacheManager:
        def __init__(self):
            self.cache = {}
            
    class DependencyTracker:
        def __init__(self):
            self.dependencies = {}
            
    class FileWatcher:
        def __init__(self):
            self.watched_files = set()
            
    class AnalysisWorker:
        def __init__(self):
            self.tasks = []

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ASTNode:
    """Represents an AST node with metadata for incremental analysis."""
    node_type: str
    node_id: str
    line_start: int
    line_end: int
    content_hash: str
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)
    semantic_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CodeChange:
    """Represents a fine-grained code change."""
    change_type: str  # 'added', 'removed', 'modified'
    node_id: str
    old_node: Optional[ASTNode] = None
    new_node: Optional[ASTNode] = None
    affected_lines: Tuple[int, int] = (0, 0)
    impact_scope: str = 'local'  # 'local', 'module', 'global'
    dependency_changes: Set[str] = field(default_factory=set)

@dataclass
class IncrementalResult:
    """Result of incremental analysis."""
    file_path: str
    changes: List[CodeChange]
    affected_nodes: Set[str]
    reanalyzed_portions: List[Tuple[int, int]]  # Line ranges
    analysis_time: float
    cache_efficiency: float
    semantic_changes: Dict[str, Any]

class SemanticDiffer:
    """Performs semantic-aware AST diffing for minimal reanalysis."""
    
    def __init__(self):
        self.node_id_counter = 0
    
    def diff_ast_trees(self, old_tree: ast.AST, new_tree: ast.AST, 
                      old_content: str, new_content: str) -> List[CodeChange]:
        """Perform semantic diffing between AST trees."""
        old_nodes = self._extract_semantic_nodes(old_tree, old_content)
        new_nodes = self._extract_semantic_nodes(new_tree, new_content)
        
        changes = []
        
        # Find removed nodes
        for old_id, old_node in old_nodes.items():
            if old_id not in new_nodes:
                changes.append(CodeChange(
                    change_type='removed',
                    node_id=old_id,
                    old_node=old_node,
                    affected_lines=(old_node.line_start, old_node.line_end),
                    impact_scope=self._determine_impact_scope(old_node)
                ))
        
        # Find added nodes
        for new_id, new_node in new_nodes.items():
            if new_id not in old_nodes:
                changes.append(CodeChange(
                    change_type='added',
                    node_id=new_id,
                    new_node=new_node,
                    affected_lines=(new_node.line_start, new_node.line_end),
                    impact_scope=self._determine_impact_scope(new_node)
                ))
        
        # Find modified nodes
        for node_id in old_nodes.keys() & new_nodes.keys():
            old_node = old_nodes[node_id]
            new_node = new_nodes[node_id]
            
            if old_node.content_hash != new_node.content_hash:
                dependency_changes = (old_node.dependencies ^ new_node.dependencies)
                
                changes.append(CodeChange(
                    change_type='modified',
                    node_id=node_id,
                    old_node=old_node,
                    new_node=new_node,
                    affected_lines=(new_node.line_start, new_node.line_end),
                    impact_scope=self._determine_impact_scope(new_node),
                    dependency_changes=dependency_changes
                ))
        
        return changes
    
    def _extract_semantic_nodes(self, tree: ast.AST, content: str) -> Dict[str, ASTNode]:
        """Extract semantically significant nodes from AST."""
        nodes = {}
        lines = content.split('\n')
        
        def extract_node(node, parent_id=None):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                node_id = self._generate_node_id(node)
                
                # Calculate content hash for semantic comparison
                node_lines = lines[node.lineno-1:getattr(node, 'end_lineno', node.lineno)]
                node_content = '\n'.join(node_lines)
                content_hash = hashlib.md5(node_content.encode()).hexdigest()
                
                # Extract dependencies
                dependencies = self._extract_node_dependencies(node)
                
                # Extract semantic information
                semantic_info = self._extract_semantic_info(node)
                
                ast_node = ASTNode(
                    node_type=type(node).__name__,
                    node_id=node_id,
                    line_start=node.lineno,
                    line_end=getattr(node, 'end_lineno', node.lineno),
                    content_hash=content_hash,
                    parent=parent_id,
                    dependencies=dependencies,
                    semantic_info=semantic_info
                )
                
                nodes[node_id] = ast_node
                
                # Recursively extract child nodes
                for child in ast.iter_child_nodes(node):
                    extract_node(child, node_id)
            else:
                # Continue with non-semantic nodes
                for child in ast.iter_child_nodes(node):
                    extract_node(child, parent_id)
        
        extract_node(tree)
        return nodes
    
    def _generate_node_id(self, node: ast.AST) -> str:
        """Generate unique identifier for AST node."""
        if hasattr(node, 'name'):
            base_id = f"{type(node).__name__}_{node.name}_{node.lineno}"
        else:
            base_id = f"{type(node).__name__}_{node.lineno}"
        
        return hashlib.md5(base_id.encode()).hexdigest()[:12]
    
    def _extract_node_dependencies(self, node: ast.AST) -> Set[str]:
        """Extract dependencies for a specific node."""
        dependencies = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if isinstance(child.ctx, (ast.Load, ast.Store)):
                    dependencies.add(child.id)
            elif isinstance(child, ast.Attribute):
                if isinstance(child.ctx, ast.Load):
                    # Build full attribute path
                    attr_path = []
                    current = child
                    while hasattr(current, 'attr'):
                        attr_path.append(current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        attr_path.append(current.id)
                        dependencies.add('.'.join(reversed(attr_path)))
        
        return dependencies
    
    def _extract_semantic_info(self, node: ast.AST) -> Dict[str, Any]:
        """Extract semantic information from node."""
        info = {}
        
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info.update({
                'function_name': node.name,
                'arg_count': len(node.args.args),
                'has_defaults': bool(node.args.defaults),
                'has_varargs': bool(node.args.vararg),
                'has_kwargs': bool(node.args.kwarg),
                'is_async': isinstance(node, ast.AsyncFunctionDef),
                'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                'return_annotation': ast.unparse(node.returns) if node.returns else None
            })
        elif isinstance(node, ast.ClassDef):
            info.update({
                'class_name': node.name,
                'base_classes': [b.id if isinstance(b, ast.Name) else str(b) for b in node.bases],
                'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                'methods': [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            })
        
        return info
    
    def _determine_impact_scope(self, node: ASTNode) -> str:
        """Determine the impact scope of a node change."""
        if node.node_type == 'ClassDef':
            return 'module'
        elif node.node_type in ['FunctionDef', 'AsyncFunctionDef']:
            # Check if it's a public function or has external dependencies
            if node.semantic_info.get('function_name', '').startswith('_'):
                return 'local'
            else:
                return 'module'
        else:
            return 'local'

class IncrementalAnalysisCache:
    """Enhanced cache for incremental analysis results."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.node_cache = {}  # node_id -> analysis_result
        self.dependency_cache = {}  # node_id -> dependencies
        self.semantic_cache = {}  # semantic_signature -> analysis_result
        self.lock = threading.RLock()
    
    def get_node_analysis(self, node_id: str, content_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis for a specific node."""
        with self.lock:
            cache_key = f"{node_id}_{content_hash}"
            return self.node_cache.get(cache_key)
    
    def store_node_analysis(self, node_id: str, content_hash: str, analysis: Dict[str, Any]):
        """Store analysis result for a specific node."""
        with self.lock:
            cache_key = f"{node_id}_{content_hash}"
            self.node_cache[cache_key] = analysis
    
    def get_semantic_analysis(self, semantic_signature: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis based on semantic signature."""
        with self.lock:
            return self.semantic_cache.get(semantic_signature)
    
    def store_semantic_analysis(self, semantic_signature: str, analysis: Dict[str, Any]):
        """Store analysis result based on semantic signature."""
        with self.lock:
            self.semantic_cache[semantic_signature] = analysis
    
    def invalidate_node(self, node_id: str):
        """Invalidate all cache entries for a node."""
        with self.lock:
            keys_to_remove = [k for k in self.node_cache.keys() if k.startswith(f"{node_id}_")]
            for key in keys_to_remove:
                del self.node_cache[key]

class EnhancedIncrementalASTEngine(RealtimeASTEngine):
    """Enhanced incremental AST analysis engine with fine-grained change detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Enhanced components
        self.semantic_differ = SemanticDiffer()
        self.incremental_cache = IncrementalAnalysisCache(
            os.path.join(self.cache_dir, 'incremental')
        )
        
        # Incremental analysis state
        self.file_nodes = {}  # file_path -> Dict[node_id, ASTNode]
        self.node_analysis_cache = {}  # node_id -> analysis_result
        self.semantic_signatures = {}  # node_id -> semantic_signature
        
        # Performance tracking
        self.incremental_stats = {
            'full_reanalysis_count': 0,
            'partial_reanalysis_count': 0,
            'nodes_reused': 0,
            'total_time_saved': 0.0,
            'cache_efficiency': 0.0
        }
        
        # Configuration for incremental analysis
        self.incremental_config = {
            'min_change_threshold': 0.1,  # Minimum change ratio to trigger reanalysis
            'semantic_caching': True,
            'parallel_node_analysis': True,
            'dependency_propagation': True
        }
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_incremental(self, file_path: str, old_content: str = None) -> IncrementalResult:
        """Perform incremental analysis of a file."""
        start_time = time.time()
        
        try:
            # Read current content
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return None
            
            new_content = file_path_obj.read_text(encoding='utf-8')
            
            # Parse new AST
            try:
                new_tree = ast.parse(new_content)
            except SyntaxError as e:
                self.logger.warning(f"Syntax error in {file_path}: {e}")
                return None
            
            # Get old content if not provided
            if old_content is None:
                old_result = self.get_result(file_path)
                if old_result:
                    old_content = self._reconstruct_content_from_cache(file_path)
                else:
                    # First analysis - perform full analysis
                    return self._perform_full_incremental_analysis(file_path, new_content, new_tree, start_time)
            
            # Parse old AST
            try:
                old_tree = ast.parse(old_content)
            except SyntaxError:
                # If old content has syntax errors, perform full analysis
                return self._perform_full_incremental_analysis(file_path, new_content, new_tree, start_time)
            
            # Perform semantic diffing
            changes = self.semantic_differ.diff_ast_trees(old_tree, new_tree, old_content, new_content)
            
            if not changes:
                # No semantic changes detected
                return IncrementalResult(
                    file_path=file_path,
                    changes=[],
                    affected_nodes=set(),
                    reanalyzed_portions=[],
                    analysis_time=time.time() - start_time,
                    cache_efficiency=1.0,
                    semantic_changes={}
                )
            
            # Perform incremental analysis
            return self._perform_incremental_analysis(
                file_path, changes, new_content, new_tree, start_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in incremental analysis of {file_path}: {e}")
            return None
    
    def _perform_full_incremental_analysis(self, file_path: str, content: str, 
                                         tree: ast.AST, start_time: float) -> IncrementalResult:
        """Perform full analysis when incremental analysis is not possible."""
        self.incremental_stats['full_reanalysis_count'] += 1
        
        # Extract all nodes
        nodes = self.semantic_differ._extract_semantic_nodes(tree, content)
        self.file_nodes[file_path] = nodes
        
        # Analyze all nodes
        affected_nodes = set(nodes.keys())
        analysis_data = self._perform_ast_analysis(tree, content, file_path)
        
        # Store node-level analysis results
        for node_id, node in nodes.items():
            node_analysis = self._analyze_individual_node(node, content, tree)
            self.incremental_cache.store_node_analysis(node_id, node.content_hash, node_analysis)
        
        analysis_time = time.time() - start_time
        
        return IncrementalResult(
            file_path=file_path,
            changes=[],
            affected_nodes=affected_nodes,
            reanalyzed_portions=[(1, len(content.split('\n')))],
            analysis_time=analysis_time,
            cache_efficiency=0.0,
            semantic_changes=analysis_data
        )
    
    def _perform_incremental_analysis(self, file_path: str, changes: List[CodeChange],
                                    content: str, tree: ast.AST, start_time: float) -> IncrementalResult:
        """Perform incremental analysis based on detected changes."""
        self.incremental_stats['partial_reanalysis_count'] += 1
        
        affected_nodes = set()
        reanalyzed_portions = []
        nodes_reused = 0
        
        # Update node cache with changes
        current_nodes = self.file_nodes.get(file_path, {})
        new_nodes = self.semantic_differ._extract_semantic_nodes(tree, content)
        
        # Process each change
        for change in changes:
            affected_nodes.add(change.node_id)
            
            if change.change_type in ['added', 'modified']:
                # Reanalyze affected node
                if change.new_node:
                    node_analysis = self._analyze_individual_node(change.new_node, content, tree)
                    self.incremental_cache.store_node_analysis(
                        change.node_id, change.new_node.content_hash, node_analysis
                    )
                    
                    reanalyzed_portions.append(change.affected_lines)
                    
                    # Propagate to dependent nodes if enabled
                    if self.incremental_config['dependency_propagation']:
                        dependent_nodes = self._find_dependent_nodes(change.node_id, current_nodes)
                        affected_nodes.update(dependent_nodes)
            
            elif change.change_type == 'removed':
                # Invalidate cache for removed node
                self.incremental_cache.invalidate_node(change.node_id)
                
                if change.old_node:
                    reanalyzed_portions.append(change.affected_lines)
        
        # Count reused nodes
        for node_id in new_nodes:
            if node_id not in affected_nodes:
                nodes_reused += 1
        
        # Update file nodes
        self.file_nodes[file_path] = new_nodes
        
        # Calculate cache efficiency
        total_nodes = len(new_nodes)
        cache_efficiency = nodes_reused / total_nodes if total_nodes > 0 else 0.0
        
        # Perform analysis on affected portions only
        semantic_changes = self._analyze_affected_portions(
            tree, content, affected_nodes, reanalyzed_portions
        )
        
        analysis_time = time.time() - start_time
        self.incremental_stats['nodes_reused'] += nodes_reused
        self.incremental_stats['cache_efficiency'] = (
            self.incremental_stats['cache_efficiency'] * 0.9 + cache_efficiency * 0.1
        )
        
        return IncrementalResult(
            file_path=file_path,
            changes=changes,
            affected_nodes=affected_nodes,
            reanalyzed_portions=reanalyzed_portions,
            analysis_time=analysis_time,
            cache_efficiency=cache_efficiency,
            semantic_changes=semantic_changes
        )
    
    def _analyze_individual_node(self, node: ASTNode, content: str, tree: ast.AST) -> Dict[str, Any]:
        """Analyze an individual AST node."""
        # Check semantic cache first
        semantic_sig = self._generate_semantic_signature(node)
        cached_analysis = self.incremental_cache.get_semantic_analysis(semantic_sig)
        
        if cached_analysis and self.incremental_config['semantic_caching']:
            return cached_analysis
        
        # Perform node-specific analysis
        lines = content.split('\n')
        node_content = '\n'.join(lines[node.line_start-1:node.line_end])
        
        try:
            node_ast = ast.parse(node_content)
        except SyntaxError:
            return {'error': 'syntax_error', 'node_type': node.node_type}
        
        analysis = {
            'node_type': node.node_type,
            'semantic_info': node.semantic_info,
            'dependencies': list(node.dependencies),
            'complexity': self._calculate_node_complexity(node_ast),
            'metrics': self._calculate_node_metrics(node_content),
            'issues': self._detect_node_issues(node_ast, node_content)
        }
        
        # Cache the result
        if self.incremental_config['semantic_caching']:
            self.incremental_cache.store_semantic_analysis(semantic_sig, analysis)
        
        return analysis
    
    def _generate_semantic_signature(self, node: ASTNode) -> str:
        """Generate semantic signature for caching."""
        signature_data = {
            'node_type': node.node_type,
            'semantic_info': node.semantic_info,
            'dependencies': sorted(list(node.dependencies))
        }
        
        signature_str = str(signature_data)
        return hashlib.md5(signature_str.encode()).hexdigest()
    
    def _find_dependent_nodes(self, node_id: str, nodes: Dict[str, ASTNode]) -> Set[str]:
        """Find nodes that depend on the given node."""
        dependents = set()
        
        for other_id, other_node in nodes.items():
            if node_id in other_node.dependencies:
                dependents.add(other_id)
        
        return dependents
    
    def _analyze_affected_portions(self, tree: ast.AST, content: str, 
                                 affected_nodes: Set[str], 
                                 reanalyzed_portions: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze only the affected portions of the code."""
        # Create a minimal analysis focusing on changed areas
        analysis = {
            'affected_nodes': list(affected_nodes),
            'reanalyzed_lines': len(set().union(*[
                range(start, end + 1) for start, end in reanalyzed_portions
            ])) if reanalyzed_portions else 0,
            'change_impact': self._assess_change_impact(affected_nodes, reanalyzed_portions),
            'updated_metrics': self._calculate_updated_metrics(tree, content, reanalyzed_portions)
        }
        
        return analysis
    
    def _assess_change_impact(self, affected_nodes: Set[str], 
                            reanalyzed_portions: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Assess the impact of the changes."""
        total_affected_lines = sum(end - start + 1 for start, end in reanalyzed_portions)
        
        return {
            'affected_node_count': len(affected_nodes),
            'affected_line_count': total_affected_lines,
            'impact_level': 'low' if total_affected_lines < 10 else 'medium' if total_affected_lines < 50 else 'high'
        }
    
    def _calculate_node_complexity(self, node_ast: ast.AST) -> Dict[str, Any]:
        """Calculate complexity metrics for a single node."""
        complexity = 1
        max_depth = 0
        
        def calculate_complexity_and_depth(node, depth=0):
            nonlocal complexity, max_depth
            max_depth = max(max_depth, depth)
            
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            
            for child in ast.iter_child_nodes(node):
                calculate_complexity_and_depth(child, depth + 1)
        
        calculate_complexity_and_depth(node_ast)
        
        return {
            'cyclomatic_complexity': complexity,
            'nesting_depth': max_depth
        }
    
    def _calculate_node_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate basic metrics for a node."""
        lines = content.split('\n')
        
        return {
            'line_count': len(lines),
            'non_empty_lines': len([line for line in lines if line.strip()]),
            'comment_lines': len([line for line in lines if line.strip().startswith('#')]),
            'character_count': len(content)
        }
    
    def _detect_node_issues(self, node_ast: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Detect issues in a specific node."""
        issues = []
        
        # Simplified issue detection for node level
        for node in ast.walk(node_ast):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ['eval', 'exec']:
                    issues.append({
                        'type': 'security',
                        'severity': 'high',
                        'message': f'Use of {node.func.id}() detected',
                        'line': node.lineno
                    })
        
        return issues
    
    def _calculate_updated_metrics(self, tree: ast.AST, content: str, 
                                 reanalyzed_portions: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Calculate metrics for updated portions only."""
        if not reanalyzed_portions:
            return {}
        
        lines = content.split('\n')
        affected_lines = set()
        
        for start, end in reanalyzed_portions:
            affected_lines.update(range(start - 1, end))  # Convert to 0-based indexing
        
        affected_content = '\n'.join(lines[i] for i in sorted(affected_lines) if i < len(lines))
        
        return {
            'updated_line_count': len(affected_lines),
            'updated_character_count': len(affected_content),
            'updated_non_empty_lines': len([line for line in affected_content.split('\n') if line.strip()])
        }
    
    def _reconstruct_content_from_cache(self, file_path: str) -> Optional[str]:
        """Attempt to reconstruct old content from cache (simplified)."""
        # In a real implementation, you might store the original content
        # For now, we'll return None to trigger full analysis
        return None
    
    def get_incremental_statistics(self) -> Dict[str, Any]:
        """Get statistics about incremental analysis performance."""
        total_analysis = (self.incremental_stats['full_reanalysis_count'] + 
                         self.incremental_stats['partial_reanalysis_count'])
        
        if total_analysis == 0:
            return self.incremental_stats
        
        return {
            **self.incremental_stats,
            'incremental_ratio': self.incremental_stats['partial_reanalysis_count'] / total_analysis,
            'average_cache_efficiency': self.incremental_stats['cache_efficiency'],
            'total_analyses': total_analysis
        }
    
    def optimize_cache(self):
        """Optimize cache performance by cleaning up unused entries."""
        # This could implement more sophisticated cache optimization
        self.logger.info("Optimizing incremental analysis cache...")
        
        # Clear old semantic signatures that haven't been used recently
        # Implementation would track usage patterns and clean accordingly
        
        self.logger.info("Cache optimization completed")


def main():
    """Demo and testing of enhanced incremental AST engine."""
    # Configuration for the enhanced engine
    config = {
        'watch_paths': ['.'],
        'cache_dir': '.enhanced_ast_cache',
        'num_workers': 2,
        'max_cache_size': 500
    }
    
    # Create enhanced engine
    engine = EnhancedIncrementalASTEngine(config)
    
    try:
        # Start the engine
        engine.start()
        
        # Test incremental analysis on a sample file
        test_file = 'test_incremental.py'
        
        # Create initial test content
        initial_content = '''
def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    return "Hello"

class TestClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
'''
        
        # Write test file
        with open(test_file, 'w') as f:
            f.write(initial_content)
        
        # Perform initial analysis
        logger.info("Performing initial analysis...")
        initial_result = engine.analyze(test_file)
        logger.info(f"Initial analysis completed: {initial_result}")
        
        # Modify the content
        modified_content = '''
def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    print("This is a modification!")
    return "Hello"

class TestClass:
    def __init__(self):
        self.value = 42
        self.name = "test"  # Added field
    
    def get_value(self):
        return self.value
    
    def get_name(self):  # Added method
        return self.name
'''
        
        # Write modified content
        with open(test_file, 'w') as f:
            f.write(modified_content)
        
        # Wait a bit for file watcher to detect change
        time.sleep(1.5)
        
        # Perform incremental analysis
        logger.info("Performing incremental analysis...")
        incremental_result = engine.analyze_incremental(test_file, initial_content)
        
        if incremental_result:
            logger.info(f"Incremental analysis completed:")
            logger.info(f"  Changes detected: {len(incremental_result.changes)}")
            logger.info(f"  Affected nodes: {len(incremental_result.affected_nodes)}")
            logger.info(f"  Analysis time: {incremental_result.analysis_time:.3f}s")
            logger.info(f"  Cache efficiency: {incremental_result.cache_efficiency:.2%}")
            
            for change in incremental_result.changes:
                logger.info(f"  Change: {change.change_type} - {change.node_id} (lines {change.affected_lines})")
        
        # Get statistics
        stats = engine.get_incremental_statistics()
        logger.info(f"Incremental statistics: {stats}")
        
        # Clean up
        os.remove(test_file)
        
    finally:
        engine.stop()
        logger.info("Enhanced incremental AST engine demo completed")


if __name__ == "__main__":
    main()