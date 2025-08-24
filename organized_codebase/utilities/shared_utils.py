from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor
"""
TestMaster Core Shared Utilities

Common data structures, constants, and patterns used across all analysis modules.
Consolidated from multiple _shared_utils.py files throughout the system.
"""

import ast
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict


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


@dataclass
class AnalysisIssue:
    """Common issue structure for all analyzers"""
    issue_type: str
    severity: str
    location: str
    description: str
    recommendation: str
    impact: str


# ML frameworks detection patterns
ML_FRAMEWORKS = {
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
ML_ANTIPATTERNS = {
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

# TensorFlow-specific patterns
TENSORFLOW_PATTERNS = {
    "session_usage": [
        "tf.Session",
        "sess.run",
        "tf.placeholder"
    ],
    "eager_execution": [
        "tf.enable_eager_execution",
        "tf.executing_eagerly"
    ],
    "v1_vs_v2": [
        "tf.compat.v1",
        "tf.keras",
        "tf.function"
    ]
}

# PyTorch-specific patterns
PYTORCH_PATTERNS = {
    "tensor_operations": [
        "torch.tensor",
        ".cuda()",
        ".to(device)"
    ],
    "autograd": [
        "torch.autograd",
        "requires_grad",
        "backward()"
    ],
    "jit_compilation": [
        "torch.jit",
        "@torch.jit.script"
    ]
}

# Data science patterns
DATA_SCIENCE_PATTERNS = {
    "preprocessing": [
        "train_test_split",
        "StandardScaler",
        "LabelEncoder"
    ],
    "feature_engineering": [
        "SelectKBest",
        "PCA",
        "feature_selection"
    ],
    "validation": [
        "cross_val_score",
        "GridSearchCV",
        "validation_curve"
    ]
}

# Analysis patterns for different domains
BUSINESS_ANALYSIS_PATTERNS = {
    "business_logic": [
        "calculate_roi",
        "process_payment", 
        "validate_business_rule"
    ],
    "workflow": [
        "approve_workflow",
        "trigger_event",
        "state_transition"
    ]
}

SEMANTIC_ANALYSIS_PATTERNS = {
    "naming": [
        "meaningful_names",
        "consistent_naming",
        "domain_terminology"
    ],
    "structure": [
        "semantic_cohesion",
        "logical_grouping",
        "abstraction_level"
    ]
}

DEBT_ANALYSIS_PATTERNS = {
    "code_smells": [
        "large_class",
        "long_method",
        "duplicate_code"
    ],
    "architectural": [
        "circular_dependency",
        "god_object",
        "feature_envy"
    ]
}

METAPROG_ANALYSIS_PATTERNS = {
    "metaprogramming": [
        "dynamic_method",
        "code_generation", 
        "reflection_usage"
    ],
    "advanced": [
        "metaclass",
        "descriptor",
        "decorator_factory"
    ]
}

ENERGY_ANALYSIS_PATTERNS = {
    "performance": [
        "inefficient_loop",
        "memory_leak",
        "cpu_intensive"
    ],
    "optimization": [
        "caching_opportunity",
        "lazy_loading",
        "batch_processing"
    ]
}

# Common constants
COMMON_PATTERNS = {
    "analysis": BUSINESS_ANALYSIS_PATTERNS,
    "semantic": SEMANTIC_ANALYSIS_PATTERNS,
    "debt": DEBT_ANALYSIS_PATTERNS, 
    "metaprog": METAPROG_ANALYSIS_PATTERNS,
    "energy": ENERGY_ANALYSIS_PATTERNS
}