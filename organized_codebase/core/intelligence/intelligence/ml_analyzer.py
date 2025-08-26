"""
ML Code Analysis Module - Integrated from Archive
Analyzes ML/AI code for tensor shapes, model architecture, and data pipeline issues
"""

import ast
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import json
import logging


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


class MLCodeAnalyzer:
    """
    Analyzes machine learning code for common issues, best practices,
    and potential problems in model architecture and data pipelines
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
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
            "jax": ["import jax", "from jax"]
        }
        
        # Common ML issues patterns
        self.ml_antipatterns = {
            "data_leakage": [
                "fit.*test",
                "train.*after.*test",
                "preprocessing.*test.*train"
            ],
            "shape_mismatch": [
                "reshape.*-1.*-1",  # Multiple -1 in reshape
                "view.*-1.*-1"
            ],
            "memory_issues": [
                "to.*cuda.*for",  # Moving to GPU in loop
                "gradient.*accumulate",
                "detach.*missing"
            ],
            "training_issues": [
                "optimizer.*zero_grad.*missing",
                "backward.*twice",
                "eval.*missing"
            ]
        }
        
        self.ml_issues = []
        
    def analyze(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive ML code analysis
        """
        self.base_path = Path(path) if path else Path.cwd()
        
        results = {
            "framework_detection": self._detect_ml_frameworks(),
            "tensor_shape_analysis": self._analyze_tensor_shapes(),
            "model_architecture": self._analyze_model_architecture(),
            "data_pipeline": self._analyze_data_pipeline(),
            "training_loop_analysis": self._analyze_training_loops(),
            "data_leakage_detection": self._detect_data_leakage(),
            "preprocessing_analysis": self._analyze_preprocessing(),
            "hyperparameter_analysis": self._analyze_hyperparameters(),
            "gpu_optimization": self._analyze_gpu_usage(),
            "model_serialization": self._analyze_model_serialization(),
            "reproducibility_check": self._check_reproducibility(),
            "performance_bottlenecks": self._identify_ml_bottlenecks(),
            "best_practices": self._check_ml_best_practices(),
            "security_analysis": self._analyze_ml_security(),
            "summary": self._generate_ml_summary()
        }
        
        return results
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files in the project"""
        return list(self.base_path.rglob("*.py"))
    
    def _parse_file(self, file_path: Path) -> Optional[ast.AST]:
        """Parse a Python file into an AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return ast.parse(f.read())
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")
            return None
    
    def _detect_ml_frameworks(self) -> Dict[str, Any]:
        """Detect which ML frameworks are being used"""
        framework_usage = {
            "detected_frameworks": [],
            "framework_versions": {},
            "framework_conflicts": [],
            "usage_statistics": defaultdict(int)
        }
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for framework, patterns in self.ml_frameworks.items():
                    for pattern in patterns:
                        if pattern in content:
                            framework_usage["detected_frameworks"].append({
                                "framework": framework,
                                "file": str(file_path),
                                "import_style": pattern
                            })
                            framework_usage["usage_statistics"][framework] += 1
                            break
                            
                # Check for version specifications
                version_patterns = {
                    "tensorflow": r"tensorflow[=><!]+(\d+\.\d+)",
                    "torch": r"torch[=><!]+(\d+\.\d+)",
                    "sklearn": r"scikit-learn[=><!]+(\d+\.\d+)"
                }
                
                for framework, pattern in version_patterns.items():
                    matches = re.findall(pattern, content)
                    if matches:
                        framework_usage["framework_versions"][framework] = matches[0]
                        
            except Exception as e:
                self.logger.error(f"Error detecting frameworks in {file_path}: {e}")
                
        # Check for potential conflicts
        if "tensorflow" in framework_usage["usage_statistics"] and "pytorch" in framework_usage["usage_statistics"]:
            framework_usage["framework_conflicts"].append({
                "conflict": "Mixed TensorFlow and PyTorch usage",
                "recommendation": "Consider standardizing on one framework",
                "severity": "medium"
            })
            
        return framework_usage
    
    def _analyze_tensor_shapes(self) -> Dict[str, Any]:
        """Analyze tensor shape operations and potential issues"""
        shape_analysis = {
            "shape_operations": [],
            "shape_mismatches": [],
            "dynamic_shapes": [],
            "broadcasting_issues": [],
            "shape_assertions": [],
            "recommendations": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        # Detect reshape operations
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Attribute):
                                if node.func.attr in ["reshape", "view", "resize", "transpose"]:
                                    shape_op = self._analyze_shape_operation(node, file_path)
                                    shape_analysis["shape_operations"].append(shape_op)
                                    
                                    # Check for potential issues
                                    if shape_op.get("has_multiple_unknown"):
                                        shape_analysis["shape_mismatches"].append({
                                            "location": str(file_path),
                                            "operation": node.func.attr,
                                            "issue": "Multiple -1 dimensions in reshape",
                                            "severity": "high",
                                            "fix": "Only one dimension can be inferred (-1)"
                                        })
                                        
                                # Check for broadcasting operations
                                elif node.func.attr in ["matmul", "bmm", "dot", "mul"]:
                                    broadcast_check = self._check_broadcasting(node, file_path)
                                    if broadcast_check.get("potential_issue"):
                                        shape_analysis["broadcasting_issues"].append(broadcast_check)
                                        
                        # Check for shape assertions
                        if isinstance(node, ast.Assert):
                            if self._is_shape_assertion(node):
                                shape_analysis["shape_assertions"].append({
                                    "location": str(file_path),
                                    "assertion": self._safe_unparse(node.test),
                                    "purpose": "shape validation"
                                })
                                
            except Exception as e:
                self.logger.error(f"Error analyzing tensor shapes in {file_path}: {e}")
                
        # Generate recommendations
        if not shape_analysis["shape_assertions"]:
            shape_analysis["recommendations"].append({
                "type": "missing_shape_validation",
                "description": "Add shape assertions to validate tensor dimensions",
                "priority": "medium"
            })
            
        if shape_analysis["broadcasting_issues"]:
            shape_analysis["recommendations"].append({
                "type": "explicit_broadcasting",
                "description": "Make broadcasting operations explicit for clarity",
                "priority": "high"
            })
            
        return shape_analysis
    
    def _analyze_model_architecture(self) -> Dict[str, Any]:
        """Analyze ML model architecture patterns and issues"""
        architecture_analysis = {
            "model_definitions": [],
            "layer_analysis": [],
            "activation_functions": defaultdict(int),
            "regularization": [],
            "architecture_issues": [],
            "complexity_metrics": {}
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        # Detect model class definitions
                        if isinstance(node, ast.ClassDef):
                            if self._is_model_class(node):
                                model_info = self._analyze_model_class(node, file_path)
                                architecture_analysis["model_definitions"].append(model_info)
                                
                                # Analyze layers
                                layers = self._extract_model_layers(node)
                                architecture_analysis["layer_analysis"].extend(layers)
                                
                                # Check for architecture issues
                                issues = self._check_architecture_issues(node, layers)
                                architecture_analysis["architecture_issues"].extend(issues)
                                
                        # Track activation functions
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Name):
                                if node.func.id in ["ReLU", "Sigmoid", "Tanh", "Softmax", "GELU", "LeakyReLU"]:
                                    architecture_analysis["activation_functions"][node.func.id] += 1
                                    
                            elif isinstance(node.func, ast.Attribute):
                                if node.func.attr in ["relu", "sigmoid", "tanh", "softmax", "gelu"]:
                                    architecture_analysis["activation_functions"][node.func.attr] += 1
                                    
                        # Detect regularization
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Attribute):
                                if node.func.attr in ["Dropout", "BatchNorm", "LayerNorm", "L1", "L2"]:
                                    architecture_analysis["regularization"].append({
                                        "type": node.func.attr,
                                        "location": str(file_path),
                                        "parameters": self._extract_call_params(node)
                                    })
                                    
            except Exception as e:
                self.logger.error(f"Error analyzing model architecture in {file_path}: {e}")
                
        # Calculate complexity metrics
        if architecture_analysis["model_definitions"]:
            architecture_analysis["complexity_metrics"] = {
                "total_models": len(architecture_analysis["model_definitions"]),
                "avg_layers_per_model": self._calculate_avg_layers(architecture_analysis["layer_analysis"]),
                "max_depth": self._calculate_max_depth(architecture_analysis["layer_analysis"]),
                "parameter_count_estimate": self._estimate_parameters(architecture_analysis["layer_analysis"])
            }
            
        return architecture_analysis
    
    def _analyze_data_pipeline(self) -> Dict[str, Any]:
        """Analyze data loading and preprocessing pipelines"""
        pipeline_analysis = {
            "data_loaders": [],
            "preprocessing_steps": [],
            "augmentation_techniques": [],
            "batch_processing": [],
            "pipeline_issues": [],
            "optimization_opportunities": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        # Detect DataLoader usage
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Name):
                                if "DataLoader" in node.func.id or "Dataset" in node.func.id:
                                    loader_info = self._analyze_data_loader(node, file_path)
                                    pipeline_analysis["data_loaders"].append(loader_info)
                                    
                                    # Check for common issues
                                    if loader_info.get("num_workers", 0) == 0:
                                        pipeline_analysis["pipeline_issues"].append({
                                            "type": "single_threaded_loading",
                                            "location": str(file_path),
                                            "impact": "Slow data loading",
                                            "fix": "Set num_workers > 0 for parallel loading"
                                        })
                                        
                        # Detect preprocessing operations
                        if isinstance(node, ast.FunctionDef):
                            if any(keyword in node.name.lower() for keyword in ["preprocess", "transform", "normalize"]):
                                preprocessing = self._analyze_preprocessing_function(node, file_path)
                                pipeline_analysis["preprocessing_steps"].append(preprocessing)
                                
                        # Detect augmentation
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Attribute):
                                if any(aug in node.func.attr for aug in ["flip", "rotate", "crop", "augment"]):
                                    pipeline_analysis["augmentation_techniques"].append({
                                        "technique": node.func.attr,
                                        "location": str(file_path),
                                        "parameters": self._extract_call_params(node)
                                    })
                                    
            except Exception as e:
                self.logger.error(f"Error analyzing data pipeline in {file_path}: {e}")
                
        # Identify optimization opportunities
        if not pipeline_analysis["data_loaders"]:
            pipeline_analysis["optimization_opportunities"].append({
                "type": "missing_data_loader",
                "description": "Consider using DataLoader for efficient batch processing",
                "impact": "performance"
            })
            
        if not any("cache" in str(step).lower() for step in pipeline_analysis["preprocessing_steps"]):
            pipeline_analysis["optimization_opportunities"].append({
                "type": "missing_caching",
                "description": "Consider caching preprocessed data",
                "impact": "performance"
            })
            
        return pipeline_analysis
    
    def _analyze_training_loops(self) -> Dict[str, Any]:
        """Analyze training loop implementations"""
        training_analysis = {
            "training_loops": [],
            "optimization_setup": [],
            "loss_functions": [],
            "metrics_tracking": [],
            "training_issues": [],
            "best_practices_violations": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        # Detect training functions
                        if isinstance(node, ast.FunctionDef):
                            if "train" in node.name.lower():
                                loop_analysis = self._analyze_training_function(node, file_path)
                                training_analysis["training_loops"].append(loop_analysis)
                                
                                # Check for common issues
                                issues = self._check_training_issues(node)
                                training_analysis["training_issues"].extend(issues)
                                
                        # Detect optimizer setup
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Name):
                                if "Optimizer" in node.func.id or node.func.id in ["Adam", "SGD", "RMSprop"]:
                                    training_analysis["optimization_setup"].append({
                                        "optimizer": node.func.id,
                                        "location": str(file_path),
                                        "parameters": self._extract_call_params(node)
                                    })
                                    
                            # Detect loss functions
                            if isinstance(node.func, ast.Attribute):
                                if "loss" in node.func.attr.lower() or node.func.attr in ["CrossEntropyLoss", "MSELoss", "BCELoss"]:
                                    training_analysis["loss_functions"].append({
                                        "type": node.func.attr,
                                        "location": str(file_path)
                                    })
                                    
            except Exception as e:
                self.logger.error(f"Error analyzing training loops in {file_path}: {e}")
                
        return training_analysis
    
    def _detect_data_leakage(self) -> Dict[str, Any]:
        """Detect potential data leakage issues"""
        leakage_analysis = {
            "potential_leaks": [],
            "preprocessing_order_issues": [],
            "test_contamination": [],
            "recommendations": []
        }
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for antipatterns
                for pattern_type, patterns in self.ml_antipatterns.items():
                    if pattern_type == "data_leakage":
                        for pattern in patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                leakage_analysis["potential_leaks"].append({
                                    "file": str(file_path),
                                    "pattern": pattern,
                                    "risk": "high",
                                    "description": "Potential data leakage detected"
                                })
                                
            except Exception as e:
                self.logger.error(f"Error detecting data leakage in {file_path}: {e}")
                
        return leakage_analysis
    
    def _analyze_preprocessing(self) -> Dict[str, Any]:
        """Analyze data preprocessing approaches"""
        return {
            "preprocessing_methods": [],
            "normalization_techniques": [],
            "feature_engineering": [],
            "data_validation": []
        }
    
    def _analyze_hyperparameters(self) -> Dict[str, Any]:
        """Analyze hyperparameter configuration"""
        return {
            "hyperparameter_configs": [],
            "tuning_approaches": [],
            "hardcoded_values": [],
            "recommendations": []
        }
    
    def _analyze_gpu_usage(self) -> Dict[str, Any]:
        """Analyze GPU optimization and usage"""
        gpu_analysis = {
            "gpu_operations": [],
            "memory_management": [],
            "optimization_issues": [],
            "cuda_kernels": []
        }
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for GPU operations
                gpu_patterns = [
                    r"\.cuda\(\)",
                    r"\.to\(['\"]*cuda",
                    r"torch\.cuda\.",
                    r"with torch\.cuda\."
                ]
                
                for pattern in gpu_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        gpu_analysis["gpu_operations"].append({
                            "file": str(file_path),
                            "operations": len(matches),
                            "pattern": pattern
                        })
                        
            except Exception as e:
                self.logger.error(f"Error analyzing GPU usage in {file_path}: {e}")
                
        return gpu_analysis
    
    def _analyze_model_serialization(self) -> Dict[str, Any]:
        """Analyze model saving and loading patterns"""
        return {
            "save_patterns": [],
            "load_patterns": [],
            "checkpoint_management": [],
            "versioning": []
        }
    
    def _check_reproducibility(self) -> Dict[str, Any]:
        """Check for reproducibility best practices"""
        reproducibility = {
            "seed_setting": [],
            "deterministic_operations": [],
            "random_state_management": [],
            "issues": []
        }
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for seed setting
                seed_patterns = [
                    r"random\.seed",
                    r"numpy\.random\.seed",
                    r"torch\.manual_seed",
                    r"tf\.random\.set_seed"
                ]
                
                for pattern in seed_patterns:
                    if re.search(pattern, content):
                        reproducibility["seed_setting"].append({
                            "file": str(file_path),
                            "pattern": pattern
                        })
                        
            except Exception as e:
                self.logger.error(f"Error checking reproducibility in {file_path}: {e}")
                
        if not reproducibility["seed_setting"]:
            reproducibility["issues"].append({
                "type": "missing_seed",
                "description": "No random seed setting found",
                "recommendation": "Set random seeds for reproducibility"
            })
            
        return reproducibility
    
    def _identify_ml_bottlenecks(self) -> Dict[str, Any]:
        """Identify performance bottlenecks in ML code"""
        return {
            "computation_bottlenecks": [],
            "memory_bottlenecks": [],
            "io_bottlenecks": [],
            "optimization_suggestions": []
        }
    
    def _check_ml_best_practices(self) -> Dict[str, Any]:
        """Check adherence to ML best practices"""
        return {
            "practices_followed": [],
            "violations": [],
            "recommendations": []
        }
    
    def _analyze_ml_security(self) -> Dict[str, Any]:
        """Analyze ML-specific security concerns"""
        return {
            "model_poisoning_risks": [],
            "adversarial_robustness": [],
            "privacy_concerns": [],
            "security_recommendations": []
        }
    
    def _generate_ml_summary(self) -> Dict[str, Any]:
        """Generate comprehensive ML analysis summary"""
        return {
            "total_issues": len(self.ml_issues),
            "critical_issues": sum(1 for issue in self.ml_issues if issue.severity == "critical"),
            "frameworks_used": [],
            "analysis_timestamp": None,
            "recommendations_count": 0
        }
    
    # Helper methods
    def _safe_unparse(self, node: ast.AST) -> str:
        """Safely unparse an AST node"""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                return str(node)
        except:
            return str(node)
    
    def _analyze_shape_operation(self, node: ast.Call, file_path: Path) -> Dict[str, Any]:
        """Analyze a shape operation"""
        result = {
            "operation": node.func.attr if isinstance(node.func, ast.Attribute) else str(node.func),
            "location": str(file_path),
            "has_multiple_unknown": False
        }
        
        # Check for multiple -1 in arguments
        unknown_count = 0
        for arg in node.args:
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.operand, ast.Constant):
                if arg.operand.value == 1:
                    unknown_count += 1
                    
        result["has_multiple_unknown"] = unknown_count > 1
        return result
    
    def _check_broadcasting(self, node: ast.Call, file_path: Path) -> Dict[str, Any]:
        """Check for broadcasting issues"""
        return {
            "operation": node.func.attr if isinstance(node.func, ast.Attribute) else str(node.func),
            "location": str(file_path),
            "potential_issue": False
        }
    
    def _is_shape_assertion(self, node: ast.Assert) -> bool:
        """Check if an assertion is related to shape validation"""
        if hasattr(node.test, 'func'):
            if isinstance(node.test.func, ast.Attribute):
                return "shape" in node.test.func.attr.lower()
        return False
    
    def _is_model_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is likely a model definition"""
        # Check for common model base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                if any(name in base.id for name in ["Model", "Module", "Layer"]):
                    return True
            elif isinstance(base, ast.Attribute):
                if any(name in base.attr for name in ["Model", "Module", "Layer"]):
                    return True
        return False
    
    def _analyze_model_class(self, node: ast.ClassDef, file_path: Path) -> Dict[str, Any]:
        """Analyze a model class definition"""
        return {
            "name": node.name,
            "location": str(file_path),
            "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
            "has_forward": any(m.name == "forward" for m in node.body if isinstance(m, ast.FunctionDef)),
            "has_init": any(m.name == "__init__" for m in node.body if isinstance(m, ast.FunctionDef))
        }
    
    def _extract_model_layers(self, node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract layer definitions from a model class"""
        layers = []
        for item in ast.walk(node):
            if isinstance(item, ast.Call):
                if isinstance(item.func, ast.Attribute):
                    if any(layer in item.func.attr for layer in ["Linear", "Conv", "LSTM", "GRU", "Embedding"]):
                        layers.append({
                            "type": item.func.attr,
                            "class": node.name
                        })
        return layers
    
    def _check_architecture_issues(self, node: ast.ClassDef, layers: List[Dict]) -> List[Dict[str, Any]]:
        """Check for architecture-related issues"""
        issues = []
        
        # Check for missing forward method
        if not any(m.name == "forward" for m in node.body if isinstance(m, ast.FunctionDef)):
            issues.append({
                "type": "missing_forward",
                "class": node.name,
                "severity": "high",
                "description": "Model class missing forward() method"
            })
            
        return issues
    
    def _extract_call_params(self, node: ast.Call) -> Dict[str, Any]:
        """Extract parameters from a function call"""
        params = {}
        for keyword in node.keywords:
            params[keyword.arg] = self._safe_unparse(keyword.value)
        return params
    
    def _analyze_data_loader(self, node: ast.Call, file_path: Path) -> Dict[str, Any]:
        """Analyze DataLoader configuration"""
        config = {
            "location": str(file_path),
            "type": node.func.id if isinstance(node.func, ast.Name) else str(node.func)
        }
        
        # Extract parameters
        for keyword in node.keywords:
            if keyword.arg in ["batch_size", "num_workers", "shuffle", "pin_memory"]:
                config[keyword.arg] = self._safe_unparse(keyword.value)
                
        return config
    
    def _analyze_preprocessing_function(self, node: ast.FunctionDef, file_path: Path) -> Dict[str, Any]:
        """Analyze a preprocessing function"""
        return {
            "name": node.name,
            "location": str(file_path),
            "parameters": [arg.arg for arg in node.args.args],
            "has_docstring": ast.get_docstring(node) is not None
        }
    
    def _analyze_training_function(self, node: ast.FunctionDef, file_path: Path) -> Dict[str, Any]:
        """Analyze a training function"""
        return {
            "name": node.name,
            "location": str(file_path),
            "has_validation": "val" in node.name.lower() or any("val" in self._safe_unparse(n) for n in ast.walk(node))
        }
    
    def _check_training_issues(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Check for common training issues"""
        issues = []
        
        # Check for gradient clearing
        has_zero_grad = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and "zero_grad" in n.func.attr
            for n in ast.walk(node)
        )
        
        if not has_zero_grad:
            issues.append({
                "type": "missing_zero_grad",
                "function": node.name,
                "severity": "high",
                "description": "Training loop may be missing optimizer.zero_grad()"
            })
            
        return issues
    
    def _calculate_avg_layers(self, layers: List[Dict]) -> float:
        """Calculate average layers per model"""
        if not layers:
            return 0.0
        model_counts = defaultdict(int)
        for layer in layers:
            if "class" in layer:
                model_counts[layer["class"]] += 1
        return sum(model_counts.values()) / len(model_counts) if model_counts else 0.0
    
    def _calculate_max_depth(self, layers: List[Dict]) -> int:
        """Calculate maximum model depth"""
        if not layers:
            return 0
        model_counts = defaultdict(int)
        for layer in layers:
            if "class" in layer:
                model_counts[layer["class"]] += 1
        return max(model_counts.values()) if model_counts else 0
    
    def _estimate_parameters(self, layers: List[Dict]) -> int:
        """Estimate total parameter count"""
        # Simplified estimation
        param_estimates = {
            "Linear": 1000000,
            "Conv": 500000,
            "LSTM": 2000000,
            "GRU": 1500000,
            "Embedding": 10000000
        }
        
        total = 0
        for layer in layers:
            layer_type = layer.get("type", "")
            for key, value in param_estimates.items():
                if key in layer_type:
                    total += value
                    break
                    
        return total