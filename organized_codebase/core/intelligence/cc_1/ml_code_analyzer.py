"""
Machine Learning Code Analysis Module
Analyzes ML/AI code for tensor shapes, model architecture, and data pipeline issues
Extracted from archive and integrated into core intelligence system
"""

import ast
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import json
from datetime import datetime


@dataclass
class MLIssue:
    """Represents an ML-specific code issue"""
    type: str
    severity: str
    location: str
    description: str
    recommendation: str
    framework: str
    impact: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'type': self.type,
            'severity': self.severity,
            'location': self.location,
            'description': self.description,
            'recommendation': self.recommendation,
            'framework': self.framework,
            'impact': self.impact
        }


@dataclass
class MLAnalysisResult:
    """Complete ML analysis result"""
    frameworks_detected: List[str]
    issues: List[MLIssue]
    architecture_summary: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'frameworks_detected': self.frameworks_detected,
            'issues': [issue.to_dict() for issue in self.issues],
            'architecture_summary': self.architecture_summary,
            'performance_metrics': self.performance_metrics,
            'recommendations': self.recommendations
        }


class MLCodeAnalyzer:
    """
    Analyzes machine learning code for common issues, best practices,
    and potential problems in model architecture and data pipelines
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # ML frameworks detection patterns
        self.ml_frameworks = {
            "tensorflow": ["import tensorflow", "from tensorflow"],
            "pytorch": ["import torch", "from torch"],
            "keras": ["import keras", "from keras"],
            "sklearn": ["import sklearn", "from sklearn"],
            "xgboost": ["import xgboost", "from xgboost"],
            "lightgbm": ["import lightgbm", "from lightgbm"],
            "pandas": ["import pandas", "from pandas"],
            "numpy": ["import numpy", "from numpy"],
            "transformers": ["import transformers", "from transformers"],
            "jax": ["import jax", "from jax"],
            "fastai": ["import fastai", "from fastai"],
            "catboost": ["import catboost", "from catboost"],
            "huggingface": ["import datasets", "from datasets"]
        }
        
        # Common ML issues patterns
        self.ml_antipatterns = {
            "data_leakage": [
                r"fit.*test",
                r"train.*after.*test",
                r"preprocessing.*test.*train",
                r"scaler\.fit\(X_test\)",
                r"normalize.*test.*before.*split"
            ],
            "shape_mismatch": [
                r"reshape.*-1.*-1",  # Multiple -1 in reshape
                r"view.*-1.*-1",
                r"tensor\.size\(\).*mismatch",
                r"dimension.*mismatch"
            ],
            "memory_issues": [
                r"\.to\(.*device.*\).*for",  # Moving to GPU in loop
                r"torch\.cuda\.empty_cache\(\).*loop",
                r"gradient.*accumulate.*without.*clear",
                r"large.*tensor.*creation.*loop"
            ],
            "training_issues": [
                r"optimizer\.zero_grad\(\).*missing",
                r"loss\.backward\(\).*twice",
                r"model\.eval\(\).*missing",
                r"model\.train\(\).*missing.*after.*eval",
                r"no_grad.*context.*training"
            ],
            "evaluation_issues": [
                r"model\.eval\(\).*missing.*evaluation",
                r"torch\.no_grad\(\).*missing.*inference",
                r"dropout.*enabled.*inference",
                r"batch_norm.*training.*inference"
            ]
        }
        
        # Security patterns for ML
        self.ml_security_patterns = [
            r"pickle\.load\(",  # Unsafe pickle loading
            r"torch\.load\(.*map_location=None\)",  # Unsafe model loading
            r"eval\(.*model.*\)",  # Code execution in model definitions
            r"exec\(.*architecture.*\)",
            r"joblib\.load\(.*verify=False\)"
        ]
        
        self.ml_issues = []
        self.detected_frameworks = set()
        
    def analyze_project(self, project_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive ML code analysis on a project
        """
        try:
            project_path = Path(project_path)
            self.ml_issues = []
            self.detected_frameworks = set()
            
            # Analyze Python files
            for py_file in project_path.rglob("*.py"):
                if self._should_analyze_file(py_file):
                    self._analyze_file(py_file)
            
            # Analyze Jupyter notebooks if present
            for nb_file in project_path.rglob("*.ipynb"):
                self._analyze_notebook(nb_file)
            
            result = MLAnalysisResult(
                frameworks_detected=list(self.detected_frameworks),
                issues=self.ml_issues,
                architecture_summary=self._generate_architecture_summary(),
                performance_metrics=self._calculate_performance_metrics(),
                recommendations=self._generate_recommendations()
            )
            
            return {
                'analysis_result': result.to_dict(),
                'summary': self._generate_summary(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'analysis_result': None,
                'timestamp': datetime.now().isoformat()
            }
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determine if a file should be analyzed for ML patterns"""
        # Focus on files likely to contain ML code
        ml_indicators = [
            'model', 'train', 'neural', 'deep', 'learning', 'ml', 'ai',
            'classification', 'regression', 'clustering', 'pipeline',
            'feature', 'dataset', 'preprocessing'
        ]
        
        file_str = str(file_path).lower()
        
        # Skip obvious non-ML files
        skip_patterns = [
            '__pycache__', '.git', 'node_modules', 'venv', '.venv'
        ]
        
        if any(pattern in file_str for pattern in skip_patterns):
            return False
        
        # Include files with ML indicators or all Python files in smaller projects
        return (any(indicator in file_str for indicator in ml_indicators) or
                file_path.suffix == '.py')
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file for ML patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Detect ML frameworks
            self._detect_frameworks(content)
            
            # Analyze different ML aspects
            self._analyze_imports(tree, file_path)
            self._analyze_data_handling(tree, content, file_path)
            self._analyze_model_definition(tree, content, file_path)
            self._analyze_training_patterns(tree, content, file_path)
            self._analyze_evaluation_patterns(tree, content, file_path)
            self._analyze_ml_security(content, file_path)
            self._analyze_tensor_operations(tree, content, file_path)
            self._analyze_gpu_usage(tree, content, file_path)
            
        except Exception as e:
            # Log error but continue analysis
            pass
    
    def _analyze_notebook(self, nb_path: Path):
        """Analyze Jupyter notebook for ML patterns"""
        try:
            import json
            with open(nb_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            # Extract code from cells
            code_content = ""
            for cell in notebook.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        code_content += '\n'.join(source) + '\n'
                    else:
                        code_content += str(source) + '\n'
            
            if code_content.strip():
                # Analyze notebook code
                self._detect_frameworks(code_content)
                self._analyze_notebook_patterns(code_content, nb_path)
                
        except Exception as e:
            # Skip notebook analysis if failed
            pass
    
    def _detect_frameworks(self, content: str):
        """Detect ML frameworks used in the code"""
        for framework, patterns in self.ml_frameworks.items():
            for pattern in patterns:
                if pattern in content:
                    self.detected_frameworks.add(framework)
    
    def _analyze_imports(self, tree: ast.AST, file_path: Path):
        """Analyze import patterns for ML issues"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Check for deprecated imports
                    if 'keras' in alias.name and alias.name != 'tensorflow.keras':
                        issue = MLIssue(
                            type="deprecated_import",
                            severity="medium",
                            location=f"{file_path}:{node.lineno}",
                            description=f"Using standalone Keras import: {alias.name}",
                            recommendation="Use tensorflow.keras instead of standalone keras",
                            framework="keras",
                            impact="compatibility"
                        )
                        self.ml_issues.append(issue)
    
    def _analyze_data_handling(self, tree: ast.AST, content: str, file_path: Path):
        """Analyze data handling patterns"""
        # Check for data leakage patterns
        for pattern_type, patterns in self.ml_antipatterns.items():
            if pattern_type == "data_leakage":
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        issue = MLIssue(
                            type="data_leakage",
                            severity="high",
                            location=str(file_path),
                            description=f"Potential data leakage detected: {match}",
                            recommendation="Ensure test data is never used for training or preprocessing fitting",
                            framework="general",
                            impact="model_validity"
                        )
                        self.ml_issues.append(issue)
        
        # Check for proper train/test split
        if "train_test_split" in content:
            if "random_state" not in content:
                issue = MLIssue(
                    type="reproducibility",
                    severity="medium",
                    location=str(file_path),
                    description="train_test_split without random_state parameter",
                    recommendation="Add random_state parameter for reproducible splits",
                    framework="sklearn",
                    impact="reproducibility"
                )
                self.ml_issues.append(issue)
    
    def _analyze_model_definition(self, tree: ast.AST, content: str, file_path: Path):
        """Analyze model architecture definition"""
        # Check for common PyTorch model issues
        if "torch" in content:
            # Check for missing super().__init__()
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if any("Module" in base.id if hasattr(base, 'id') else False 
                          for base in node.bases):
                        # This is a PyTorch module
                        init_method = None
                        for item in node.body:
                            if (isinstance(item, ast.FunctionDef) and 
                                item.name == "__init__"):
                                init_method = item
                                break
                        
                        if init_method:
                            # Check for super().__init__() call
                            has_super_init = False
                            for stmt in init_method.body:
                                if (isinstance(stmt, ast.Expr) and
                                    isinstance(stmt.value, ast.Call)):
                                    if hasattr(stmt.value.func, 'attr') and stmt.value.func.attr == '__init__':
                                        has_super_init = True
                                        break
                            
                            if not has_super_init:
                                issue = MLIssue(
                                    type="missing_super_init",
                                    severity="high",
                                    location=f"{file_path}:{init_method.lineno}",
                                    description="PyTorch module missing super().__init__() call",
                                    recommendation="Add super().__init__() at the beginning of __init__ method",
                                    framework="pytorch",
                                    impact="functionality"
                                )
                                self.ml_issues.append(issue)
        
        # Check for TensorFlow/Keras model issues
        if any(fw in content for fw in ["tensorflow", "keras"]):
            # Check for missing model compilation
            if "Sequential" in content or "Model" in content:
                if "compile" not in content:
                    issue = MLIssue(
                        type="missing_compilation",
                        severity="medium",
                        location=str(file_path),
                        description="Model defined but not compiled",
                        recommendation="Add model.compile() with optimizer, loss, and metrics",
                        framework="tensorflow",
                        impact="functionality"
                    )
                    self.ml_issues.append(issue)
    
    def _analyze_training_patterns(self, tree: ast.AST, content: str, file_path: Path):
        """Analyze training loop patterns"""
        # Check for training issues
        for pattern_type, patterns in self.ml_antipatterns.items():
            if pattern_type == "training_issues":
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        issue = MLIssue(
                            type="training_issue",
                            severity="high",
                            location=str(file_path),
                            description=f"Training issue detected: {match}",
                            recommendation="Follow proper training loop patterns",
                            framework="pytorch" if "torch" in content else "general",
                            impact="training_effectiveness"
                        )
                        self.ml_issues.append(issue)
        
        # Check for missing gradient clipping in RNN/Transformer models
        if any(pattern in content.lower() for pattern in ["rnn", "lstm", "gru", "transformer"]):
            if "clip_grad" not in content:
                issue = MLIssue(
                    type="missing_gradient_clipping",
                    severity="medium",
                    location=str(file_path),
                    description="RNN/Transformer model without gradient clipping",
                    recommendation="Add gradient clipping to prevent exploding gradients",
                    framework="pytorch" if "torch" in content else "tensorflow",
                    impact="training_stability"
                )
                self.ml_issues.append(issue)
    
    def _analyze_evaluation_patterns(self, tree: ast.AST, content: str, file_path: Path):
        """Analyze model evaluation patterns"""
        # Check for evaluation issues
        for pattern_type, patterns in self.ml_antipatterns.items():
            if pattern_type == "evaluation_issues":
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        issue = MLIssue(
                            type="evaluation_issue",
                            severity="medium",
                            location=str(file_path),
                            description=f"Evaluation issue detected: {match}",
                            recommendation="Ensure proper evaluation mode and context",
                            framework="pytorch" if "torch" in content else "general",
                            impact="evaluation_accuracy"
                        )
                        self.ml_issues.append(issue)
    
    def _analyze_ml_security(self, content: str, file_path: Path):
        """Analyze ML-specific security issues"""
        for pattern in self.ml_security_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                issue = MLIssue(
                    type="security_issue",
                    severity="critical",
                    location=str(file_path),
                    description=f"Unsafe ML operation detected: {match}",
                    recommendation="Use safe loading methods and validate model sources",
                    framework="general",
                    impact="security"
                )
                self.ml_issues.append(issue)
    
    def _analyze_tensor_operations(self, tree: ast.AST, content: str, file_path: Path):
        """Analyze tensor operations for shape issues"""
        # Check for shape mismatch patterns
        for pattern_type, patterns in self.ml_antipatterns.items():
            if pattern_type == "shape_mismatch":
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        issue = MLIssue(
                            type="shape_mismatch",
                            severity="high",
                            location=str(file_path),
                            description=f"Potential shape mismatch: {match}",
                            recommendation="Verify tensor dimensions and use proper reshape operations",
                            framework="pytorch" if "torch" in content else "tensorflow",
                            impact="runtime_error"
                        )
                        self.ml_issues.append(issue)
    
    def _analyze_gpu_usage(self, tree: ast.AST, content: str, file_path: Path):
        """Analyze GPU usage patterns"""
        if "cuda" in content or "device" in content:
            # Check for inefficient GPU usage
            for pattern_type, patterns in self.ml_antipatterns.items():
                if pattern_type == "memory_issues":
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            issue = MLIssue(
                                type="gpu_inefficiency",
                                severity="medium",
                                location=str(file_path),
                                description=f"Inefficient GPU usage: {match}",
                                recommendation="Optimize GPU memory usage and tensor movements",
                                framework="pytorch" if "torch" in content else "tensorflow",
                                impact="performance"
                            )
                            self.ml_issues.append(issue)
    
    def _analyze_notebook_patterns(self, content: str, nb_path: Path):
        """Analyze Jupyter notebook specific patterns"""
        # Check for common notebook issues
        if content.count("import") > 10:
            issue = MLIssue(
                type="notebook_organization",
                severity="low",
                location=str(nb_path),
                description="Many import statements scattered throughout notebook",
                recommendation="Organize imports at the top of the notebook",
                framework="general",
                impact="maintainability"
            )
            self.ml_issues.append(issue)
    
    def _generate_architecture_summary(self) -> Dict[str, Any]:
        """Generate architecture summary"""
        return {
            "frameworks_used": list(self.detected_frameworks),
            "total_issues": len(self.ml_issues),
            "issues_by_severity": self._count_by_severity(),
            "issues_by_type": self._count_by_type(),
            "frameworks_coverage": len(self.detected_frameworks)
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance-related metrics"""
        performance_issues = [
            issue for issue in self.ml_issues 
            if issue.impact in ["performance", "training_effectiveness", "training_stability"]
        ]
        
        return {
            "performance_issues_count": len(performance_issues),
            "critical_performance_issues": len([
                issue for issue in performance_issues 
                if issue.severity == "critical"
            ]),
            "gpu_related_issues": len([
                issue for issue in self.ml_issues 
                if "gpu" in issue.type or "cuda" in issue.description.lower()
            ])
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Framework-specific recommendations
        if "pytorch" in self.detected_frameworks:
            recommendations.append("Follow PyTorch best practices for model definition and training")
        
        if "tensorflow" in self.detected_frameworks:
            recommendations.append("Ensure proper model compilation and TensorFlow 2.x patterns")
        
        # Issue-based recommendations
        critical_issues = [issue for issue in self.ml_issues if issue.severity == "critical"]
        if critical_issues:
            recommendations.append(f"Address {len(critical_issues)} critical security/functionality issues immediately")
        
        security_issues = [issue for issue in self.ml_issues if issue.type == "security_issue"]
        if security_issues:
            recommendations.append("Implement secure model loading and validation practices")
        
        return recommendations
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate analysis summary"""
        return {
            "frameworks_detected": len(self.detected_frameworks),
            "total_issues": len(self.ml_issues),
            "severity_breakdown": self._count_by_severity(),
            "framework_breakdown": self._count_by_framework(),
            "top_issue_types": self._get_top_issue_types()
        }
    
    def _count_by_severity(self) -> Dict[str, int]:
        """Count issues by severity"""
        severity_count = defaultdict(int)
        for issue in self.ml_issues:
            severity_count[issue.severity] += 1
        return dict(severity_count)
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count issues by type"""
        type_count = defaultdict(int)
        for issue in self.ml_issues:
            type_count[issue.type] += 1
        return dict(type_count)
    
    def _count_by_framework(self) -> Dict[str, int]:
        """Count issues by framework"""
        framework_count = defaultdict(int)
        for issue in self.ml_issues:
            framework_count[issue.framework] += 1
        return dict(framework_count)
    
    def _get_top_issue_types(self) -> List[Tuple[str, int]]:
        """Get top issue types"""
        type_count = self._count_by_type()
        return sorted(type_count.items(), key=lambda x: x[1], reverse=True)[:5]


# Export
__all__ = ['MLCodeAnalyzer', 'MLIssue', 'MLAnalysisResult']