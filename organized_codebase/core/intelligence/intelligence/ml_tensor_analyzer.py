"""
ML Tensor Analyzer

Handles tensor shape analysis, tensor operations, and shape-related issues.
Split from original ml_code_analysis.py - Tensor Shape Analysis sections.
"""

import ast
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
import logging

from ...base import BaseAnalyzer
from ._shared_utils import MLIssue, extract_tensor_shapes


class MLTensorAnalyzer(BaseAnalyzer):
    """
    Specialized analyzer for tensor operations, shapes, and related issues
    """
    
    def __init__(self):
        super().__init__()
        self.tensor_issues = []
        self.tensor_operations = []
        self.shape_mismatches = []
        
    def analyze(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform tensor-focused analysis
        """
        if file_path:
            self._analyze_file(Path(file_path))
        else:
            self._analyze_directory()
            
        return self._generate_tensor_report()
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single file for tensor patterns"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Extract tensor shapes
            shapes = extract_tensor_shapes(content)
            if shapes:
                self._analyze_tensor_shapes(shapes, str(file_path))
            
            # Analyze tensor operations
            self._analyze_tensor_operations(tree, content, str(file_path))
            
            # Check for shape mismatches
            self._check_shape_compatibility(tree, content, str(file_path))
            
            # Analyze tensor memory usage
            self._analyze_tensor_memory(tree, content, str(file_path))
            
        except Exception as e:
            logging.error(f"Error analyzing tensors in {file_path}: {e}")
    
    def _analyze_tensor_shapes(self, shapes: List[Dict[str, Any]], file_path: str) -> None:
        """Analyze extracted tensor shapes for issues"""
        
        for shape_info in shapes:
            shape_str = shape_info["shape"]
            line = shape_info["line"]
            
            # Check for problematic shapes
            if "-1" in shape_str:
                # Multiple -1 dimensions
                if shape_str.count("-1") > 1:
                    self.tensor_issues.append(MLIssue(
                        type="shape_mismatch",
                        severity="high",
                        location=f"{file_path}:{line}",
                        description=f"Multiple -1 dimensions in shape: {shape_str}",
                        recommendation="Only one dimension can be inferred (-1)",
                        framework="general",
                        impact="runtime_error"
                    ))
            
            # Check for very large dimensions
            dimensions = re.findall(r'\b\d+\b', shape_str)
            for dim in dimensions:
                if int(dim) > 10000:
                    self.tensor_issues.append(MLIssue(
                        type="large_tensor",
                        severity="medium",
                        location=f"{file_path}:{line}",
                        description=f"Very large tensor dimension: {dim}",
                        recommendation="Consider if this dimension size is necessary",
                        framework="general",
                        impact="memory_usage"
                    ))
            
            # Check for hardcoded shapes
            if re.search(r'\b\d{3,}\b', shape_str):  # Numbers with 3+ digits
                self.tensor_issues.append(MLIssue(
                    type="hardcoded_shape",
                    severity="low",
                    location=f"{file_path}:{line}",
                    description=f"Hardcoded tensor shape: {shape_str}",
                    recommendation="Consider using configurable shape parameters",
                    framework="general",
                    impact="maintainability"
                ))
    
    def _analyze_tensor_operations(self, tree: ast.AST, content: str, file_path: str) -> None:
        """Analyze tensor operations for efficiency and correctness"""
        
        # TensorFlow tensor operations
        tf_ops = [
            "tf.reshape", "tf.transpose", "tf.concat", "tf.stack",
            "tf.reduce_mean", "tf.reduce_sum", "tf.matmul"
        ]
        
        # PyTorch tensor operations  
        torch_ops = [
            "torch.reshape", "torch.transpose", "torch.cat", "torch.stack",
            "torch.mean", "torch.sum", "torch.mm", "torch.bmm"
        ]
        
        all_ops = tf_ops + torch_ops
        
        for op in all_ops:
            if op in content:
                # Count operations
                count = content.count(op)
                self.tensor_operations.append({
                    "operation": op,
                    "count": count,
                    "file": file_path,
                    "framework": "tensorflow" if op.startswith("tf.") else "pytorch"
                })
                
                # Check for operations in loops (performance issue)
                if re.search(rf'for.*{op}', content, re.IGNORECASE):
                    self.tensor_issues.append(MLIssue(
                        type="tensor_op_in_loop",
                        severity="medium",
                        location=file_path,
                        description=f"Tensor operation {op} in loop",
                        recommendation="Consider vectorizing or moving operation outside loop",
                        framework="tensorflow" if op.startswith("tf.") else "pytorch",
                        impact="performance"
                    ))
        
        # Check for specific tensor operation patterns
        self._check_tensor_operation_patterns(tree, content, file_path)
    
    def _check_tensor_operation_patterns(self, tree: ast.AST, content: str, file_path: str) -> None:
        """Check for specific tensor operation patterns and anti-patterns"""
        
        # Check for inefficient reshaping
        reshape_patterns = [
            r'\.reshape\([^)]*\)\.reshape\([^)]*\)',  # Double reshape
            r'\.view\([^)]*\)\.view\([^)]*\)',        # Double view
        ]
        
        for pattern in reshape_patterns:
            if re.search(pattern, content):
                self.tensor_issues.append(MLIssue(
                    type="inefficient_reshape",
                    severity="low",
                    location=file_path,
                    description="Multiple consecutive reshape operations",
                    recommendation="Combine reshape operations",
                    framework="general",
                    impact="performance"
                ))
        
        # Check for tensor copying patterns
        copy_patterns = [
            r'\.clone\(\)\.detach\(\)',
            r'\.detach\(\)\.clone\(\)',
            r'tensor\.copy_\(',
        ]
        
        for pattern in copy_patterns:
            if re.search(pattern, content):
                self.tensor_issues.append(MLIssue(
                    type="tensor_copying",
                    severity="medium",
                    location=file_path,
                    description="Explicit tensor copying detected",
                    recommendation="Ensure copying is necessary for memory efficiency",
                    framework="pytorch",
                    impact="memory_usage"
                ))
        
        # Check for dimension mismatch patterns
        mismatch_patterns = [
            r'squeeze\(\).*unsqueeze\(',  # Squeeze then unsqueeze
            r'unsqueeze\([^)]*\).*squeeze\(',  # Unsqueeze then squeeze
        ]
        
        for pattern in mismatch_patterns:
            if re.search(pattern, content):
                self.tensor_issues.append(MLIssue(
                    type="dimension_manipulation",
                    severity="low",
                    location=file_path,
                    description="Dimension squeeze/unsqueeze pattern",
                    recommendation="Review if dimension manipulation is necessary",
                    framework="pytorch",
                    impact="code_clarity"
                ))
    
    def _check_shape_compatibility(self, tree: ast.AST, content: str, file_path: str) -> None:
        """Check for potential shape compatibility issues"""
        
        # Look for matrix multiplication patterns
        matmul_patterns = [
            "torch.mm", "torch.bmm", "torch.matmul",
            "tf.matmul", "tf.linalg.matmul", "@"  # @ operator
        ]
        
        for pattern in matmul_patterns:
            if pattern in content:
                # Find lines with matrix multiplication
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if pattern in line:
                        # Check if there's obvious shape mismatch
                        if "reshape" in line or "view" in line:
                            self.tensor_issues.append(MLIssue(
                                type="potential_shape_mismatch",
                                severity="medium",
                                location=f"{file_path}:{i}",
                                description=f"Reshape before {pattern} - verify compatibility",
                                recommendation="Ensure tensor shapes are compatible for operation",
                                framework="general",
                                impact="runtime_error"
                            ))
        
        # Check for broadcasting issues
        broadcasting_ops = ["+", "-", "*", "/", "//", "%", "**"]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                op_str = ast.unparse(node)
                
                # Look for operations between tensors of different apparent sizes
                if any(size_hint in op_str for size_hint in ["tensor", "array", "Tensor"]):
                    # This is a heuristic check - would need more sophisticated analysis
                    if "broadcast" in content.lower():
                        self.tensor_issues.append(MLIssue(
                            type="broadcasting_check",
                            severity="low",
                            location=f"{file_path}:{node.lineno}",
                            description="Tensor broadcasting operation detected",
                            recommendation="Verify broadcasting behavior is intended",
                            framework="general",
                            impact="correctness"
                        ))
    
    def _analyze_tensor_memory(self, tree: ast.AST, content: str, file_path: str) -> None:
        """Analyze tensor memory usage patterns"""
        
        # Check for GPU memory patterns
        gpu_patterns = [
            ".cuda()", ".to(device)", ".to('cuda')", 
            "torch.cuda.empty_cache()", "tf.config.experimental.set_memory_growth"
        ]
        
        for pattern in gpu_patterns:
            count = content.count(pattern)
            if count > 0:
                if count > 10:  # Frequent GPU operations
                    self.tensor_issues.append(MLIssue(
                        type="frequent_gpu_transfer",
                        severity="medium",
                        location=file_path,
                        description=f"Frequent GPU operations ({count} occurrences): {pattern}",
                        recommendation="Batch GPU operations to reduce overhead",
                        framework="general",
                        impact="performance"
                    ))
        
        # Check for memory clearing patterns
        memory_clear_patterns = [
            "del ", "torch.cuda.empty_cache", "tf.keras.backend.clear_session"
        ]
        
        has_memory_management = any(pattern in content for pattern in memory_clear_patterns)
        
        # If GPU operations but no memory management
        has_gpu_ops = any(pattern in content for pattern in gpu_patterns[:4])
        
        if has_gpu_ops and not has_memory_management:
            self.tensor_issues.append(MLIssue(
                type="missing_memory_management",
                severity="medium",
                location=file_path,
                description="GPU operations without explicit memory management",
                recommendation="Add memory clearing operations to prevent OOM",
                framework="general",
                impact="memory_leak"
            ))
        
        # Check for large tensor creation
        large_tensor_patterns = [
            r'torch\.zeros\([^)]*[0-9]{4,}[^)]*\)',  # Large zeros tensor
            r'torch\.ones\([^)]*[0-9]{4,}[^)]*\)',   # Large ones tensor
            r'torch\.randn\([^)]*[0-9]{4,}[^)]*\)',  # Large random tensor
        ]
        
        for pattern in large_tensor_patterns:
            if re.search(pattern, content):
                self.tensor_issues.append(MLIssue(
                    type="large_tensor_creation",
                    severity="low",
                    location=file_path,
                    description="Large tensor creation detected",
                    recommendation="Consider lazy loading or chunking for large tensors",
                    framework="pytorch",
                    impact="memory_usage"
                ))
    
    def _generate_tensor_report(self) -> Dict[str, Any]:
        """Generate comprehensive tensor analysis report"""
        
        # Calculate statistics
        total_issues = len(self.tensor_issues)
        total_operations = len(self.tensor_operations)
        
        # Group issues by type and severity
        issues_by_type = defaultdict(int)
        issues_by_severity = defaultdict(int)
        
        for issue in self.tensor_issues:
            issues_by_type[issue.type] += 1
            issues_by_severity[issue.severity] += 1
        
        # Group operations by framework
        ops_by_framework = defaultdict(list)
        for op in self.tensor_operations:
            ops_by_framework[op["framework"]].append(op)
        
        # Most common operations
        op_counts = defaultdict(int)
        for op in self.tensor_operations:
            op_counts[op["operation"]] += op["count"]
        
        top_operations = sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "summary": {
                "total_tensor_issues": total_issues,
                "total_tensor_operations": total_operations,
                "high_severity_issues": issues_by_severity.get("high", 0),
                "medium_severity_issues": issues_by_severity.get("medium", 0),
                "files_analyzed": len(set(issue.location.split(':')[0] for issue in self.tensor_issues))
            },
            "tensor_issues": [
                {
                    "type": issue.type,
                    "severity": issue.severity,
                    "location": issue.location,
                    "description": issue.description,
                    "recommendation": issue.recommendation,
                    "framework": issue.framework,
                    "impact": issue.impact
                }
                for issue in self.tensor_issues
            ],
            "tensor_operations": {
                "by_framework": dict(ops_by_framework),
                "top_operations": top_operations,
                "all_operations": self.tensor_operations
            },
            "issues_by_type": dict(issues_by_type),
            "issues_by_severity": dict(issues_by_severity),
            "shape_analysis": {
                "detected_mismatches": len([i for i in self.tensor_issues if i.type == "shape_mismatch"]),
                "large_tensors": len([i for i in self.tensor_issues if i.type == "large_tensor"]),
                "hardcoded_shapes": len([i for i in self.tensor_issues if i.type == "hardcoded_shape"])
            },
            "memory_analysis": {
                "gpu_issues": len([i for i in self.tensor_issues if "gpu" in i.type.lower()]),
                "memory_issues": len([i for i in self.tensor_issues if "memory" in i.type.lower()]),
                "performance_issues": len([i for i in self.tensor_issues if i.impact == "performance"])
            },
            "recommendations": self._generate_tensor_recommendations()
        }
    
    def _generate_tensor_recommendations(self) -> List[Dict[str, str]]:
        """Generate tensor-specific recommendations"""
        recommendations = []
        
        # Shape-related recommendations
        shape_issues = [i for i in self.tensor_issues if "shape" in i.type]
        if shape_issues:
            recommendations.append({
                "category": "Tensor Shapes",
                "priority": "high",
                "recommendation": "Fix tensor shape compatibility issues",
                "impact": "Prevent runtime errors and crashes"
            })
        
        # Memory-related recommendations
        memory_issues = [i for i in self.tensor_issues if "memory" in i.type or "gpu" in i.type]
        if memory_issues:
            recommendations.append({
                "category": "Memory Management",
                "priority": "medium",
                "recommendation": "Optimize tensor memory usage",
                "impact": "Reduce memory consumption and OOM errors"
            })
        
        # Performance recommendations
        perf_issues = [i for i in self.tensor_issues if i.impact == "performance"]
        if perf_issues:
            recommendations.append({
                "category": "Performance",
                "priority": "medium",
                "recommendation": "Optimize tensor operations for better performance",
                "impact": "Faster training and inference"
            })
        
        # Code quality recommendations
        quality_issues = [i for i in self.tensor_issues if i.impact in ["maintainability", "code_clarity"]]
        if quality_issues:
            recommendations.append({
                "category": "Code Quality",
                "priority": "low",
                "recommendation": "Improve tensor operation clarity and maintainability",
                "impact": "Better code readability and maintenance"
            })
        
        return recommendations