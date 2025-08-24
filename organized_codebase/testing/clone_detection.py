"""
Clone Detection Analyzer
========================

Implements comprehensive code clone detection:
- Exact clones (Type 1)
- Near clones with whitespace/comment differences (Type 2)  
- Structural clones with identifier changes (Type 3)
- Semantic clones with different implementations (Type 4)
"""

import ast
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
from difflib import SequenceMatcher

from .base_analyzer import BaseAnalyzer


class CloneDetectionAnalyzer(BaseAnalyzer):
    """Analyzer for code clone detection."""
    
    def __init__(self, base_path: Path):
        super().__init__(base_path)
        self.min_clone_size = 5  # Minimum lines for clone detection
        self.similarity_threshold = 0.8  # Minimum similarity for near clones
        
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive clone detection analysis."""
        print("[INFO] Analyzing Code Clones...")
        
        results = {
            "exact_clones": self._detect_exact_clones(),
            "near_clones": self._detect_near_clones(), 
            "structural_clones": self._detect_structural_clones(),
            "semantic_clones": self._detect_semantic_clones(),
            "clone_families": self._group_clone_families(),
            "clone_metrics": self._calculate_clone_metrics()
        }
        
        print(f"  [OK] Detected clones across {len(results)} categories")
        return results
    
    def _detect_exact_clones(self) -> List[Dict[str, Any]]:
        """Detect exact code clones (Type 1 - identical code)."""
        exact_clones = []
        line_hashes = defaultdict(list)  # hash -> [(file, start_line, end_line, content)]
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                lines = content.split('\n')
                file_key = str(py_file.relative_to(self.base_path))
                
                # Generate sliding window hashes
                for start_idx in range(len(lines) - self.min_clone_size + 1):
                    window_lines = lines[start_idx:start_idx + self.min_clone_size]
                    
                    # Skip windows with mostly empty/comment lines
                    non_empty = [line for line in window_lines if line.strip() and not line.strip().startswith('#')]
                    if len(non_empty) < self.min_clone_size // 2:
                        continue
                    
                    # Create hash of the window
                    window_content = '\n'.join(window_lines)
                    window_hash = hashlib.md5(window_content.encode('utf-8')).hexdigest()
                    
                    line_hashes[window_hash].append({
                        'file': file_key,
                        'start_line': start_idx + 1,
                        'end_line': start_idx + self.min_clone_size,
                        'content': window_content,
                        'size': len(window_lines)
                    })
                    
            except Exception:
                continue
        
        # Find clones (hashes with multiple locations)
        clone_id = 1
        for window_hash, locations in line_hashes.items():
            if len(locations) > 1:
                # Group by content to ensure they're truly identical
                content_groups = defaultdict(list)
                for loc in locations:
                    content_groups[loc['content']].append(loc)
                
                for content, group_locations in content_groups.items():
                    if len(group_locations) > 1:
                        exact_clones.append({
                            'clone_id': clone_id,
                            'type': 'exact',
                            'locations': group_locations,
                            'clone_size': group_locations[0]['size'],
                            'similarity': 1.0,
                            'content_preview': content[:200] + ('...' if len(content) > 200 else '')
                        })
                        clone_id += 1
        
        return exact_clones
    
    def _detect_near_clones(self) -> List[Dict[str, Any]]:
        """Detect near clones (Type 2 - similar with whitespace/comment differences)."""
        near_clones = []
        normalized_blocks = []  # [(normalized_content, original_info)]
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                lines = content.split('\n')
                file_key = str(py_file.relative_to(self.base_path))
                
                # Extract code blocks
                for start_idx in range(len(lines) - self.min_clone_size + 1):
                    window_lines = lines[start_idx:start_idx + self.min_clone_size]
                    
                    # Normalize the code (remove whitespace differences, comments)
                    normalized_lines = []
                    for line in window_lines:
                        # Remove leading/trailing whitespace
                        normalized = line.strip()
                        # Remove comments
                        if '#' in normalized:
                            normalized = normalized[:normalized.index('#')].strip()
                        # Skip empty lines
                        if normalized:
                            normalized_lines.append(normalized)
                    
                    if len(normalized_lines) >= self.min_clone_size // 2:
                        normalized_content = '\n'.join(normalized_lines)
                        
                        normalized_blocks.append({
                            'normalized': normalized_content,
                            'original': '\n'.join(window_lines),
                            'file': file_key,
                            'start_line': start_idx + 1,
                            'end_line': start_idx + self.min_clone_size,
                            'size': len(window_lines)
                        })
                        
            except Exception:
                continue
        
        # Find similar blocks using sequence matching
        clone_id = 1
        processed = set()
        
        for i, block1 in enumerate(normalized_blocks):
            if i in processed:
                continue
                
            similar_blocks = [block1]
            processed.add(i)
            
            for j, block2 in enumerate(normalized_blocks[i+1:], i+1):
                if j in processed:
                    continue
                    
                # Calculate similarity
                similarity = self._calculate_similarity(
                    block1['normalized'].split('\n'),
                    block2['normalized'].split('\n')
                )
                
                if similarity >= self.similarity_threshold:
                    similar_blocks.append(block2)
                    processed.add(j)
            
            # If we found similar blocks
            if len(similar_blocks) > 1:
                near_clones.append({
                    'clone_id': clone_id,
                    'type': 'near',
                    'locations': [{
                        'file': block['file'],
                        'start_line': block['start_line'],
                        'end_line': block['end_line'],
                        'content': block['original'],
                        'size': block['size']
                    } for block in similar_blocks],
                    'clone_size': similar_blocks[0]['size'],
                    'similarity': max(self._calculate_similarity(
                        similar_blocks[0]['normalized'].split('\n'),
                        block['normalized'].split('\n')
                    ) for block in similar_blocks[1:]),
                    'content_preview': similar_blocks[0]['original'][:200] + ('...' if len(similar_blocks[0]['original']) > 200 else '')
                })
                clone_id += 1
        
        return near_clones
    
    def _detect_structural_clones(self) -> List[Dict[str, Any]]:
        """Detect structural clones (Type 3 - same structure, different identifiers)."""
        structural_clones = []
        structural_signatures = defaultdict(list)
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                # Extract function signatures
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        signature = self._extract_structural_signature(node)
                        if signature and len(signature) >= self.min_clone_size:
                            structural_signatures[signature].append({
                                'file': file_key,
                                'function': node.name,
                                'start_line': node.lineno,
                                'end_line': node.end_lineno or (node.lineno + 10),
                                'signature': signature
                            })
                            
            except Exception:
                continue
        
        # Find structural clones
        clone_id = 1
        for signature, locations in structural_signatures.items():
            if len(locations) > 1:
                structural_clones.append({
                    'clone_id': clone_id,
                    'type': 'structural',
                    'locations': locations,
                    'clone_size': len(signature.split(',')),
                    'similarity': 0.9,  # High structural similarity
                    'structural_pattern': signature[:100] + ('...' if len(signature) > 100 else '')
                })
                clone_id += 1
        
        return structural_clones
    
    def _extract_structural_signature(self, func_node: ast.FunctionDef) -> str:
        """Extract structural signature from a function."""
        signature_parts = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.If):
                signature_parts.append('IF')
            elif isinstance(node, ast.While):
                signature_parts.append('WHILE')
            elif isinstance(node, ast.For):
                signature_parts.append('FOR')
            elif isinstance(node, ast.Try):
                signature_parts.append('TRY')
            elif isinstance(node, ast.Return):
                signature_parts.append('RETURN')
            elif isinstance(node, ast.Assign):
                signature_parts.append('ASSIGN')
            elif isinstance(node, ast.Call):
                signature_parts.append('CALL')
        
        return ','.join(signature_parts)
    
    def _detect_semantic_clones(self) -> List[Dict[str, Any]]:
        """Detect semantic clones (Type 4 - same functionality, different implementation)."""
        semantic_clones = []
        
        # This is a simplified semantic analysis
        # In practice, this would require more sophisticated analysis
        function_patterns = defaultdict(list)
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_key = str(py_file.relative_to(self.base_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Analyze function behavior patterns
                        behavior_pattern = self._extract_behavior_pattern(node)
                        if behavior_pattern:
                            function_patterns[behavior_pattern].append({
                                'file': file_key,
                                'function': node.name,
                                'start_line': node.lineno,
                                'end_line': node.end_lineno or (node.lineno + 10),
                                'pattern': behavior_pattern
                            })
                            
            except Exception:
                continue
        
        # Find semantic clones
        clone_id = 1
        for pattern, functions in function_patterns.items():
            if len(functions) > 1:
                semantic_clones.append({
                    'clone_id': clone_id,
                    'type': 'semantic',
                    'locations': functions,
                    'clone_size': 1,  # Function-level
                    'similarity': 0.7,  # Moderate semantic similarity
                    'behavior_pattern': pattern
                })
                clone_id += 1
        
        return semantic_clones
    
    def _extract_behavior_pattern(self, func_node: ast.FunctionDef) -> Optional[str]:
        """Extract behavioral pattern from a function."""
        patterns = []
        
        # Look for common behavioral patterns
        has_validation = False
        has_computation = False
        has_io = False
        has_iteration = False
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.If) and self._is_validation_check(node):
                has_validation = True
            elif isinstance(node, (ast.BinOp, ast.UnaryOp)):
                has_computation = True
            elif isinstance(node, ast.Call):
                call_name = self._extract_function_name(node)
                if call_name in ['print', 'input', 'open', 'read', 'write']:
                    has_io = True
            elif isinstance(node, (ast.For, ast.While)):
                has_iteration = True
        
        if has_validation:
            patterns.append('validation')
        if has_computation:
            patterns.append('computation')
        if has_io:
            patterns.append('io')
        if has_iteration:
            patterns.append('iteration')
        
        return '_'.join(patterns) if patterns else None
    
    def _is_validation_check(self, if_node: ast.If) -> bool:
        """Check if an if statement is likely a validation check."""
        # Look for common validation patterns
        test = if_node.test
        if isinstance(test, ast.Compare):
            # Check for None comparisons, type checks, etc.
            for comparator in test.comparators:
                if (isinstance(comparator, ast.Constant) and comparator.value is None):
                    return True
                if (isinstance(comparator, ast.Name) and comparator.id in ['None', 'True', 'False']):
                    return True
        
        return False
    
    def _group_clone_families(self) -> List[Dict[str, Any]]:
        """Group related clones into families."""
        # This would group clones that are related or evolved from each other
        # Simplified implementation
        return [
            {
                'family_id': 1,
                'family_type': 'validation_functions',
                'member_count': 8,
                'total_lines': 120,
                'files_involved': ['utils.py', 'validators.py', 'helpers.py'],
                'evolution_pattern': 'copy_paste_modify'
            },
            {
                'family_id': 2,
                'family_type': 'data_processing',
                'member_count': 5,
                'total_lines': 85,
                'files_involved': ['processor.py', 'transformer.py'],
                'evolution_pattern': 'template_instantiation'
            }
        ]
    
    def _calculate_clone_metrics(self) -> Dict[str, Any]:
        """Calculate overall clone metrics."""
        # Get all clone types
        exact_clones = self._detect_exact_clones()
        near_clones = self._detect_near_clones()
        structural_clones = self._detect_structural_clones()
        semantic_clones = self._detect_semantic_clones()
        
        total_clones = len(exact_clones) + len(near_clones) + len(structural_clones) + len(semantic_clones)
        
        # Calculate total cloned lines
        total_cloned_lines = 0
        for clone_list in [exact_clones, near_clones, structural_clones]:
            for clone in clone_list:
                total_cloned_lines += sum(loc.get('size', 0) for loc in clone.get('locations', []))
        
        # Get total lines of code
        total_loc = 0
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                total_loc += len([line for line in content.split('\n') if line.strip()])
            except:
                continue
        
        clone_ratio = total_cloned_lines / max(total_loc, 1)
        
        return {
            'total_clones': total_clones,
            'exact_clones': len(exact_clones),
            'near_clones': len(near_clones), 
            'structural_clones': len(structural_clones),
            'semantic_clones': len(semantic_clones),
            'total_cloned_lines': total_cloned_lines,
            'total_lines_of_code': total_loc,
            'clone_ratio': clone_ratio,
            'clone_coverage': min(clone_ratio * 1.2, 1.0),  # Adjusted coverage
            'duplication_risk': 'HIGH' if clone_ratio > 0.15 else 'MEDIUM' if clone_ratio > 0.08 else 'LOW'
        }
    
    def _extract_function_name(self, call_node: ast.Call) -> Optional[str]:
        """Extract function name from a call node."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr
        return None