from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor
"""
TestMaster ML Analysis Utilities

Machine learning specific utility functions for framework detection,
pattern analysis, and code understanding.
"""

import ast
import re
from typing import Dict, Any, List, Optional
from collections import defaultdict
from .shared_utils import ML_FRAMEWORKS


def extract_tensor_shapes(code_content: str) -> List[Dict[str, Any]]:
    """Extract tensor shape information from code"""
    shapes = []
    
    # Common shape patterns
    shape_patterns = [
        r'\.shape\s*=\s*(\[[^\]]+\])',
        r'reshape\s*\(\s*([^)]+)\)',
        r'view\s*\(\s*([^)]+)\)',
        r'size\s*=\s*(\([^)]+\))',
    ]
    
    for pattern in shape_patterns:
        matches = re.finditer(pattern, code_content, re.IGNORECASE)
        for match in matches:
            shapes.append({
                "pattern": pattern,
                "shape": match.group(1),
                "line": code_content[:match.start()].count('\n') + 1
            })
    
    return shapes


def detect_ml_frameworks_in_content(content: str) -> Dict[str, List[str]]:
    """Detect ML frameworks in code content"""
    detected = defaultdict(list)
    
    for framework, patterns in ML_FRAMEWORKS.items():
        for pattern in patterns:
            if pattern in content:
                detected[framework].append(pattern)
    
    return dict(detected)


def analyze_import_patterns(tree: ast.AST) -> Dict[str, Any]:
    """Analyze import patterns for ML frameworks"""
    imports = {
        "standard_imports": [],
        "from_imports": [],
        "alias_imports": [],
        "star_imports": []
    }
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_info = {
                    "module": alias.name,
                    "alias": alias.asname,
                    "line": node.lineno
                }
                
                if alias.asname:
                    imports["alias_imports"].append(import_info)
                else:
                    imports["standard_imports"].append(import_info)
                    
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    import_info = {
                        "module": node.module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno
                    }
                    
                    if alias.name == "*":
                        imports["star_imports"].append(import_info)
                    else:
                        imports["from_imports"].append(import_info)
    
    return imports


def check_version_compatibility(frameworks: List[str]) -> List[Dict[str, Any]]:
    """Check for framework version compatibility issues"""
    conflicts = []
    
    # Known compatibility issues
    compatibility_matrix = {
        ("tensorflow", "pytorch"): {
            "issue": "Mixed TensorFlow and PyTorch usage",
            "recommendation": "Consider standardizing on one framework",
            "severity": "medium"
        },
        ("tensorflow", "keras"): {
            "issue": "Separate Keras import with TensorFlow 2.x",
            "recommendation": "Use tf.keras instead of separate keras",
            "severity": "low"
        }
    }
    
    for fw1 in frameworks:
        for fw2 in frameworks:
            if fw1 != fw2:
                key = tuple(sorted([fw1, fw2]))
                if key in compatibility_matrix:
                    conflicts.append(compatibility_matrix[key])
    
    return conflicts


def extract_hyperparameters(tree: ast.AST) -> List[Dict[str, Any]]:
    """Extract hyperparameter assignments from code"""
    hyperparams = []
    
    # Common hyperparameter names
    hyperparam_names = [
        "learning_rate", "lr", "batch_size", "epochs", "hidden_size",
        "num_layers", "dropout", "weight_decay", "momentum", "beta1", "beta2"
    ]
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if any(param in target.id.lower() for param in hyperparam_names):
                        try:
                            value = ast.literal_SafeCodeExecutor.safe_SafeCodeExecutor.safe_eval(node.value)
                            hyperparams.append({
                                "name": target.id,
                                "value": value,
                                "line": node.lineno,
                                "type": type(value).__name__
                            })
                        except (ValueError, TypeError):
                            # Can't evaluate the value
                            hyperparams.append({
                                "name": target.id,
                                "value": ast.unparse(node.value),
                                "line": node.lineno,
                                "type": "expression"
                            })
    
    return hyperparams


def identify_model_patterns(tree: ast.AST) -> List[Dict[str, Any]]:
    """Identify common ML model patterns"""
    patterns = []
    
    # Sequential model pattern
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_str = ast.unparse(node)
            
            # TensorFlow/Keras patterns
            if "Sequential" in call_str:
                patterns.append({
                    "pattern": "Sequential Model",
                    "framework": "tensorflow/keras",
                    "line": node.lineno,
                    "description": "Sequential neural network architecture"
                })
            
            # PyTorch patterns
            elif "nn.Module" in call_str:
                patterns.append({
                    "pattern": "PyTorch Module",
                    "framework": "pytorch",
                    "line": node.lineno,
                    "description": "Custom PyTorch neural network module"
                })
            
            # Sklearn patterns
            elif any(clf in call_str for clf in ["Classifier", "Regressor", "SVM", "RandomForest"]):
                patterns.append({
                    "pattern": "Scikit-learn Model",
                    "framework": "sklearn",
                    "line": node.lineno,
                    "description": "Traditional machine learning model"
                })
    
    return patterns