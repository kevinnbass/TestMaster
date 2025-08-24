"""
Data Leakage Detector for ML Workflows

This module detects various forms of data leakage in machine learning workflows,
including temporal leakage, target leakage, and feature leakage that can lead to
overly optimistic model performance estimates.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging

from ..base import BaseAnalyzer


class LeakageType(Enum):
    """Types of data leakage"""
    TARGET_LEAKAGE = "target_leakage"
    TEMPORAL_LEAKAGE = "temporal_leakage"
    FEATURE_LEAKAGE = "feature_leakage"
    GROUP_LEAKAGE = "group_leakage"
    DUPLICATE_LEAKAGE = "duplicate_leakage"
    PREPROCESSING_LEAKAGE = "preprocessing_leakage"
    LOOKAHEAD_BIAS = "lookahead_bias"
    SELECTION_BIAS = "selection_bias"
    SURVIVAL_BIAS = "survival_bias"
    LABEL_LEAKAGE = "label_leakage"


class Severity(Enum):
    """Severity levels for leakage issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class LeakageIssue:
    """Represents a data leakage issue"""
    leakage_type: LeakageType
    severity: Severity
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    impact: str
    recommendation: str
    confidence: float  # 0.0 to 1.0
    false_positive_risk: float  # 0.0 to 1.0
    estimated_performance_impact: str  # percentage or description


@dataclass
class DataSplit:
    """Represents a data split operation"""
    method: str
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    is_temporal: bool
    is_stratified: bool
    random_state: Optional[int]
    file_path: str
    line_number: int


@dataclass
class FeatureEngineering:
    """Represents feature engineering operations"""
    operation: str
    features_created: List[str]
    features_used: List[str]
    is_fit_before_split: bool
    is_transform_after_split: bool
    file_path: str
    line_number: int


@dataclass
class ModelTraining:
    """Represents model training operations"""
    model_type: str
    features: List[str]
    target: str
    uses_future_data: bool
    uses_target_derived_features: bool
    file_path: str
    line_number: int


class DataLeakageDetector(BaseAnalyzer):
    """Detects data leakage in ML workflows"""
    
    def __init__(self):
        super().__init__()
        self.leakage_issues: List[LeakageIssue] = []
        self.data_splits: List[DataSplit] = []
        self.feature_engineering: List[FeatureEngineering] = []
        self.model_training: List[ModelTraining] = []
        
        # ML frameworks and their patterns
        self.ml_frameworks = {
            "sklearn": ["from sklearn", "import sklearn"],
            "pandas": ["import pandas", "from pandas"],
            "numpy": ["import numpy", "from numpy"],
            "tensorflow": ["import tensorflow", "from tensorflow"],
            "pytorch": ["import torch", "from torch"],
            "xgboost": ["import xgboost", "from xgboost"],
            "lightgbm": ["import lightgbm", "from lightgbm"],
            "catboost": ["import catboost", "from catboost"],
        }
        
        # Leakage-prone patterns
        self.leakage_patterns = {
            # Target leakage patterns
            LeakageType.TARGET_LEAKAGE: {
                "patterns": [
                    r"df\[.*\]\s*=.*target",  # Using target to create features
                    r"features.*\.merge.*target",  # Merging target into features
                    r".*target.*\.shift\(-",  # Using future target values
                    r"groupby.*target.*transform",  # Target-based transformations
                    r"fillna.*target",  # Filling nulls with target info
                ],
                "description": "Features derived from target variable",
                "impact": "Artificially inflated model performance"
            },
            
            # Temporal leakage patterns
            LeakageType.TEMPORAL_LEAKAGE: {
                "patterns": [
                    r"sort_values.*ascending=False",  # Sorting by date descending
                    r"\.shift\(-\d+\)",  # Using future values
                    r"rolling.*\.shift\(-",  # Future rolling statistics
                    r"\.iloc\[-\d+:\]",  # Using last observations
                    r"max\(.*date.*\)",  # Using maximum date
                ],
                "description": "Using future information to predict past",
                "impact": "Model will not work in production"
            },
            
            # Feature leakage patterns
            LeakageType.FEATURE_LEAKAGE: {
                "patterns": [
                    r"StandardScaler\(\)\.fit\(X\)",  # Fitting scaler on all data
                    r"\.fit\(X_train\+X_test\)",  # Fitting on combined data
                    r"fillna\(.*\.mean\(\)\)",  # Using overall mean to fill nulls
                    r"normalize.*fit\(.*\)",  # Normalizing all data together
                    r"PCA\(\)\.fit\(X\)",  # PCA on all data
                ],
                "description": "Preprocessing fitted on test data",
                "impact": "Optimistic performance estimates"
            },
            
            # Group leakage patterns
            LeakageType.GROUP_LEAKAGE: {
                "patterns": [
                    r"train_test_split.*random_state",  # Random split with groups
                    r"sample\(.*random_state",  # Random sampling with groups
                    r"shuffle=True",  # Shuffling grouped data
                ],
                "description": "Random splits breaking group structure",
                "impact": "Overestimated performance due to data similarity"
            },
            
            # Duplicate leakage patterns
            LeakageType.DUPLICATE_LEAKAGE: {
                "patterns": [
                    r"train_test_split.*\)",  # Split without deduplication
                    r"(?!.*drop_duplicates).*split",  # Split without checking duplicates
                ],
                "description": "Identical samples in train and test",
                "impact": "Inflated performance metrics"
            }
        }
        
        # Preprocessing operations that can cause leakage
        self.preprocessing_ops = {
            "scalers": ["StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer"],
            "encoders": ["LabelEncoder", "OneHotEncoder", "OrdinalEncoder"],
            "imputers": ["SimpleImputer", "KNNImputer", "IterativeImputer"],
            "selectors": ["SelectKBest", "RFE", "SelectFromModel"],
            "transformers": ["PCA", "TruncatedSVD", "FactorAnalysis"],
        }
        
        # Time-related column names that indicate temporal data
        self.temporal_columns = [
            "date", "time", "timestamp", "created", "updated",
            "year", "month", "day", "hour", "minute",
            "datetime", "period", "epoch", "ts"
        ]
        
        # Target-like column names
        self.target_like_columns = [
            "target", "label", "y", "class", "category",
            "outcome", "result", "prediction", "score",
            "rating", "price", "amount", "value"
        ]
        
    def analyze(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Analyze data leakage in ML workflows"""
        if file_path:
            self._analyze_file(Path(file_path))
        else:
            self._analyze_directory()
            
        return self._generate_report()
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single file for data leakage"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Check if file contains ML code
            if not self._contains_ml_code(content):
                return
            
            # Analyze data splits
            self._analyze_data_splits(tree, content, str(file_path))
            
            # Analyze feature engineering
            self._analyze_feature_engineering(tree, content, str(file_path))
            
            # Analyze model training
            self._analyze_model_training(tree, content, str(file_path))
            
            # Detect specific leakage patterns
            self._detect_leakage_patterns(content, str(file_path))
            
            # Analyze preprocessing order
            self._analyze_preprocessing_order(tree, content, str(file_path))
            
            # Check temporal consistency
            self._check_temporal_consistency(tree, content, str(file_path))
            
            # Detect duplicate handling
            self._detect_duplicate_handling(tree, content, str(file_path))
            
        except Exception as e:
            logging.error(f"Error analyzing {file_path}: {e}")
    
    def _contains_ml_code(self, content: str) -> bool:
        """Check if file contains ML-related code"""
        ml_indicators = [
            "train_test_split", "cross_val_score", "fit", "predict",
            "classifier", "regressor", "model", "sklearn", "ml"
        ]
        return any(indicator in content.lower() for indicator in ml_indicators)
    
    def _analyze_data_splits(self, tree: ast.AST, content: str, 
                            file_path: str) -> None:
        """Analyze data splitting operations"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_str = ast.unparse(node)
                
                # Detect train_test_split
                if "train_test_split" in call_str:
                    self._analyze_train_test_split(node, content, file_path)
                
                # Detect cross-validation
                elif any(cv in call_str for cv in 
                        ["cross_val_score", "KFold", "StratifiedKFold"]):
                    self._analyze_cross_validation(node, content, file_path)
                
                # Detect temporal splits
                elif any(temp in call_str.lower() for temp in 
                        ["timeseries", "temporal", "date"]):
                    self._analyze_temporal_split(node, content, file_path)
    
    def _analyze_train_test_split(self, node: ast.Call, content: str,
                                 file_path: str) -> None:
        """Analyze train_test_split usage"""
        call_str = ast.unparse(node)
        
        # Extract parameters
        test_size = self._extract_parameter(call_str, "test_size", 0.25)
        random_state = self._extract_parameter(call_str, "random_state", None)
        shuffle = self._extract_parameter(call_str, "shuffle", True)
        stratify = self._extract_parameter(call_str, "stratify", None)
        
        # Check for issues
        issues = []
        
        # No random state
        if random_state is None:
            issues.append(LeakageIssue(
                leakage_type=LeakageType.SELECTION_BIAS,
                severity=Severity.MEDIUM,
                description="No random state set for train_test_split",
                file_path=file_path,
                line_number=node.lineno,
                code_snippet=call_str,
                impact="Results not reproducible",
                recommendation="Set random_state parameter",
                confidence=0.8,
                false_positive_risk=0.1,
                estimated_performance_impact="Reproducibility issue only"
            ))
        
        # Shuffle without considering groups
        if shuffle and "stratify" not in call_str:
            # Check if data might have groups
            if self._might_have_groups(content):
                issues.append(LeakageIssue(
                    leakage_type=LeakageType.GROUP_LEAKAGE,
                    severity=Severity.HIGH,
                    description="Random shuffle may break group structure",
                    file_path=file_path,
                    line_number=node.lineno,
                    code_snippet=call_str,
                    impact="Overoptimistic performance due to similar samples",
                    recommendation="Use GroupKFold or group-aware splitting",
                    confidence=0.6,
                    false_positive_risk=0.3,
                    estimated_performance_impact="5-15% performance overestimate"
                ))
        
        # Test size too small
        if isinstance(test_size, (int, float)) and test_size < 0.15:
            issues.append(LeakageIssue(
                leakage_type=LeakageType.SELECTION_BIAS,
                severity=Severity.MEDIUM,
                description=f"Test set too small ({test_size:.1%})",
                file_path=file_path,
                line_number=node.lineno,
                code_snippet=call_str,
                impact="Unreliable performance estimates",
                recommendation="Use at least 15-20% for test set",
                confidence=0.7,
                false_positive_risk=0.2,
                estimated_performance_impact="High variance in performance"
            ))
        
        self.leakage_issues.extend(issues)
        
        # Record data split
        data_split = DataSplit(
            method="train_test_split",
            train_ratio=1 - test_size if isinstance(test_size, (int, float)) else 0.75,
            validation_ratio=0.0,
            test_ratio=test_size if isinstance(test_size, (int, float)) else 0.25,
            is_temporal=False,
            is_stratified=stratify is not None,
            random_state=random_state,
            file_path=file_path,
            line_number=node.lineno
        )
        self.data_splits.append(data_split)
    
    def _extract_parameter(self, call_str: str, param: str, default: Any) -> Any:
        """Extract parameter value from function call string"""
        pattern = rf"{param}\s*=\s*([^,\)]+)"
        match = re.search(pattern, call_str)
        if match:
            value_str = match.group(1).strip()
            
            # Try to parse common types
            if value_str == "None":
                return None
            elif value_str == "True":
                return True
            elif value_str == "False":
                return False
            elif value_str.replace(".", "").isdigit():
                return float(value_str) if "." in value_str else int(value_str)
            else:
                return value_str
        
        return default
    
    def _might_have_groups(self, content: str) -> bool:
        """Check if data might have group structure"""
        group_indicators = [
            "customer", "user", "patient", "subject", "id",
            "group", "cluster", "session", "visit", "transaction"
        ]
        return any(indicator in content.lower() for indicator in group_indicators)
    
    def _analyze_feature_engineering(self, tree: ast.AST, content: str,
                                   file_path: str) -> None:
        """Analyze feature engineering operations"""
        preprocessing_order = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_str = ast.unparse(node)
                
                # Check for preprocessing operations
                for op_type, operations in self.preprocessing_ops.items():
                    for operation in operations:
                        if operation in call_str:
                            preprocessing_order.append({
                                "operation": operation,
                                "line": node.lineno,
                                "call": call_str
                            })
        
        # Analyze preprocessing order for leakage
        self._check_preprocessing_leakage(preprocessing_order, file_path)
    
    def _check_preprocessing_leakage(self, operations: List[Dict], 
                                   file_path: str) -> None:
        """Check for preprocessing leakage"""
        fit_operations = []
        transform_operations = []
        
        for op in operations:
            call = op["call"]
            if ".fit(" in call:
                fit_operations.append(op)
            elif ".transform(" in call:
                transform_operations.append(op)
        
        # Check if fit operations happen on all data
        for fit_op in fit_operations:
            call = fit_op["call"]
            
            # Look for patterns indicating fitting on all data
            if any(pattern in call for pattern in ["X)", "data)", "df)"]):
                # Check if this happens before splitting
                split_lines = [split.line_number for split in self.data_splits 
                              if split.file_path == file_path]
                
                if not split_lines or fit_op["line"] < min(split_lines):
                    self.leakage_issues.append(LeakageIssue(
                        leakage_type=LeakageType.PREPROCESSING_LEAKAGE,
                        severity=Severity.HIGH,
                        description=f"Preprocessor fitted on all data before split",
                        file_path=file_path,
                        line_number=fit_op["line"],
                        code_snippet=fit_op["call"],
                        impact="Information from test set leaks into training",
                        recommendation="Fit preprocessor only on training data",
                        confidence=0.8,
                        false_positive_risk=0.2,
                        estimated_performance_impact="2-10% performance overestimate"
                    ))
    
    def _detect_leakage_patterns(self, content: str, file_path: str) -> None:
        """Detect specific leakage patterns in code"""
        lines = content.split('\n')
        
        for leakage_type, pattern_info in self.leakage_patterns.items():
            for pattern in pattern_info["patterns"]:
                for i, line in enumerate(lines, 1):
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        # Calculate confidence based on pattern specificity
                        confidence = 0.7 if leakage_type == LeakageType.TARGET_LEAKAGE else 0.5
                        
                        self.leakage_issues.append(LeakageIssue(
                            leakage_type=leakage_type,
                            severity=self._determine_severity(leakage_type),
                            description=pattern_info["description"],
                            file_path=file_path,
                            line_number=i,
                            code_snippet=line.strip(),
                            impact=pattern_info["impact"],
                            recommendation=self._get_recommendation(leakage_type),
                            confidence=confidence,
                            false_positive_risk=0.3,
                            estimated_performance_impact=self._get_impact_estimate(leakage_type)
                        ))
    
    def _determine_severity(self, leakage_type: LeakageType) -> Severity:
        """Determine severity based on leakage type"""
        severity_map = {
            LeakageType.TARGET_LEAKAGE: Severity.CRITICAL,
            LeakageType.TEMPORAL_LEAKAGE: Severity.CRITICAL,
            LeakageType.PREPROCESSING_LEAKAGE: Severity.HIGH,
            LeakageType.FEATURE_LEAKAGE: Severity.HIGH,
            LeakageType.GROUP_LEAKAGE: Severity.HIGH,
            LeakageType.DUPLICATE_LEAKAGE: Severity.MEDIUM,
            LeakageType.LOOKAHEAD_BIAS: Severity.HIGH,
            LeakageType.SELECTION_BIAS: Severity.MEDIUM,
            LeakageType.SURVIVAL_BIAS: Severity.MEDIUM,
            LeakageType.LABEL_LEAKAGE: Severity.HIGH,
        }
        return severity_map.get(leakage_type, Severity.MEDIUM)
    
    def _get_recommendation(self, leakage_type: LeakageType) -> str:
        """Get recommendation for leakage type"""
        recommendations = {
            LeakageType.TARGET_LEAKAGE: "Remove target-derived features or use temporal cutoff",
            LeakageType.TEMPORAL_LEAKAGE: "Ensure features only use past information",
            LeakageType.PREPROCESSING_LEAKAGE: "Fit preprocessors only on training data",
            LeakageType.FEATURE_LEAKAGE: "Apply feature engineering after data split",
            LeakageType.GROUP_LEAKAGE: "Use group-aware splitting (GroupKFold)",
            LeakageType.DUPLICATE_LEAKAGE: "Remove duplicates before splitting",
            LeakageType.LOOKAHEAD_BIAS: "Use only past information for predictions",
            LeakageType.SELECTION_BIAS: "Use proper sampling and randomization",
            LeakageType.SURVIVAL_BIAS: "Account for censoring and dropout patterns",
            LeakageType.LABEL_LEAKAGE: "Separate labeling from feature engineering",
        }
        return recommendations.get(leakage_type, "Review data processing pipeline")
    
    def _get_impact_estimate(self, leakage_type: LeakageType) -> str:
        """Get performance impact estimate for leakage type"""
        impact_estimates = {
            LeakageType.TARGET_LEAKAGE: "50-100% performance overestimate",
            LeakageType.TEMPORAL_LEAKAGE: "20-80% performance overestimate", 
            LeakageType.PREPROCESSING_LEAKAGE: "2-10% performance overestimate",
            LeakageType.FEATURE_LEAKAGE: "5-15% performance overestimate",
            LeakageType.GROUP_LEAKAGE: "5-15% performance overestimate",
            LeakageType.DUPLICATE_LEAKAGE: "10-30% performance overestimate",
            LeakageType.LOOKAHEAD_BIAS: "10-50% performance overestimate",
            LeakageType.SELECTION_BIAS: "Variable, reproducibility issues",
            LeakageType.SURVIVAL_BIAS: "5-20% performance overestimate",
            LeakageType.LABEL_LEAKAGE: "10-40% performance overestimate",
        }
        return impact_estimates.get(leakage_type, "Unknown impact")
    
    def _analyze_preprocessing_order(self, tree: ast.AST, content: str,
                                   file_path: str) -> None:
        """Analyze the order of preprocessing operations"""
        # Track the sequence of operations
        operations_sequence = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_str = ast.unparse(node)
                
                # Check for data splitting
                if "train_test_split" in call_str:
                    operations_sequence.append({
                        "type": "split",
                        "line": node.lineno,
                        "operation": "train_test_split"
                    })
                
                # Check for preprocessing
                for op_type, operations in self.preprocessing_ops.items():
                    for operation in operations:
                        if operation in call_str and ".fit(" in call_str:
                            operations_sequence.append({
                                "type": "preprocessing",
                                "line": node.lineno,
                                "operation": operation
                            })
        
        # Sort by line number
        operations_sequence.sort(key=lambda x: x["line"])
        
        # Check if preprocessing happens before splitting
        split_line = None
        for op in operations_sequence:
            if op["type"] == "split":
                split_line = op["line"]
                break
        
        if split_line:
            preprocessing_before_split = [
                op for op in operations_sequence
                if op["type"] == "preprocessing" and op["line"] < split_line
            ]
            
            for op in preprocessing_before_split:
                self.leakage_issues.append(LeakageIssue(
                    leakage_type=LeakageType.PREPROCESSING_LEAKAGE,
                    severity=Severity.HIGH,
                    description=f"{op['operation']} fitted before data split",
                    file_path=file_path,
                    line_number=op["line"],
                    code_snippet=f"{op['operation']} preprocessing",
                    impact="Test data information leaks into training",
                    recommendation="Move preprocessing after data split",
                    confidence=0.9,
                    false_positive_risk=0.1,
                    estimated_performance_impact="2-10% performance overestimate"
                ))
    
    def _check_temporal_consistency(self, tree: ast.AST, content: str,
                                  file_path: str) -> None:
        """Check for temporal consistency in time series data"""
        # Look for temporal column usage
        temporal_columns_found = []
        
        for col in self.temporal_columns:
            if col in content.lower():
                temporal_columns_found.append(col)
        
        if not temporal_columns_found:
            return
        
        # Check for temporal leakage patterns
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()
            
            # Check for future data usage
            if any(col in line_lower for col in temporal_columns_found):
                # Look for problematic patterns
                if any(pattern in line for pattern in [".shift(-", "max(", "last("]):
                    self.leakage_issues.append(LeakageIssue(
                        leakage_type=LeakageType.TEMPORAL_LEAKAGE,
                        severity=Severity.CRITICAL,
                        description="Potential use of future information",
                        file_path=file_path,
                        line_number=i,
                        code_snippet=line.strip(),
                        impact="Model will not work in production",
                        recommendation="Use only past information relative to prediction time",
                        confidence=0.7,
                        false_positive_risk=0.3,
                        estimated_performance_impact="20-80% performance overestimate"
                    ))
    
    def _detect_duplicate_handling(self, tree: ast.AST, content: str,
                                  file_path: str) -> None:
        """Detect duplicate handling in data preparation"""
        has_split = "train_test_split" in content
        has_dedup = any(pattern in content for pattern in 
                       ["drop_duplicates", "duplicated", "unique"])
        
        if has_split and not has_dedup:
            # Look for the split line
            lines = content.split('\n')
            split_line = None
            for i, line in enumerate(lines, 1):
                if "train_test_split" in line:
                    split_line = i
                    break
            
            if split_line:
                self.leakage_issues.append(LeakageIssue(
                    leakage_type=LeakageType.DUPLICATE_LEAKAGE,
                    severity=Severity.MEDIUM,
                    description="No duplicate removal before train/test split",
                    file_path=file_path,
                    line_number=split_line,
                    code_snippet="train_test_split without deduplication",
                    impact="Identical samples may appear in train and test",
                    recommendation="Remove duplicates before splitting data",
                    confidence=0.5,
                    false_positive_risk=0.4,
                    estimated_performance_impact="5-20% performance overestimate"
                ))
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive data leakage report"""
        # Sort issues by severity and confidence
        sorted_issues = sorted(
            self.leakage_issues,
            key=lambda x: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3}[x.severity.value],
                -x.confidence
            )
        )
        
        # Calculate statistics
        total_issues = len(self.leakage_issues)
        critical_issues = sum(1 for i in self.leakage_issues 
                            if i.severity == Severity.CRITICAL)
        high_issues = sum(1 for i in self.leakage_issues 
                         if i.severity == Severity.HIGH)
        
        # Group issues by type
        issues_by_type = {}
        for issue in self.leakage_issues:
            leakage_type = issue.leakage_type.value
            if leakage_type not in issues_by_type:
                issues_by_type[leakage_type] = []
            issues_by_type[leakage_type].append(issue)
        
        # Calculate confidence statistics
        avg_confidence = (sum(i.confidence for i in self.leakage_issues) / total_issues
                         if total_issues > 0 else 0)
        
        high_confidence_issues = sum(1 for i in self.leakage_issues 
                                   if i.confidence > 0.7)
        
        return {
            "summary": {
                "total_issues": total_issues,
                "critical_issues": critical_issues,
                "high_issues": high_issues,
                "medium_issues": sum(1 for i in self.leakage_issues 
                                   if i.severity == Severity.MEDIUM),
                "low_issues": sum(1 for i in self.leakage_issues 
                                if i.severity == Severity.LOW),
                "average_confidence": round(avg_confidence, 3),
                "high_confidence_issues": high_confidence_issues,
                "files_analyzed": len(set(i.file_path for i in self.leakage_issues)),
            },
            "leakage_issues": [
                {
                    "type": issue.leakage_type.value,
                    "severity": issue.severity.value,
                    "description": issue.description,
                    "file": issue.file_path,
                    "line": issue.line_number,
                    "code": issue.code_snippet,
                    "impact": issue.impact,
                    "recommendation": issue.recommendation,
                    "confidence": issue.confidence,
                    "false_positive_risk": issue.false_positive_risk,
                    "performance_impact": issue.estimated_performance_impact,
                }
                for issue in sorted_issues
            ],
            "issues_by_type": {
                leakage_type: len(issues)
                for leakage_type, issues in issues_by_type.items()
            },
            "data_splits": [
                {
                    "method": split.method,
                    "train_ratio": split.train_ratio,
                    "validation_ratio": split.validation_ratio,
                    "test_ratio": split.test_ratio,
                    "is_temporal": split.is_temporal,
                    "is_stratified": split.is_stratified,
                    "random_state": split.random_state,
                    "file": split.file_path,
                    "line": split.line_number,
                }
                for split in self.data_splits
            ],
            "recommendations": self._generate_recommendations(),
        }
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Critical issues first
        critical_issues = [i for i in self.leakage_issues 
                          if i.severity == Severity.CRITICAL]
        if critical_issues:
            recommendations.append({
                "priority": "critical",
                "category": "Data Leakage",
                "recommendation": "Fix critical data leakage issues immediately",
                "impact": "These issues will cause complete model failure in production",
                "issues_count": len(critical_issues),
                "estimated_effort": "1-2 days per issue"
            })
        
        # Preprocessing leakage
        preprocessing_issues = [i for i in self.leakage_issues 
                              if i.leakage_type == LeakageType.PREPROCESSING_LEAKAGE]
        if preprocessing_issues:
            recommendations.append({
                "priority": "high",
                "category": "Preprocessing",
                "recommendation": "Move all preprocessing after data split",
                "impact": "Reduce performance overestimation by 2-10%",
                "issues_count": len(preprocessing_issues),
                "estimated_effort": "4-8 hours"
            })
        
        # Group leakage
        group_issues = [i for i in self.leakage_issues 
                       if i.leakage_type == LeakageType.GROUP_LEAKAGE]
        if group_issues:
            recommendations.append({
                "priority": "high",
                "category": "Data Splitting",
                "recommendation": "Implement group-aware data splitting",
                "impact": "Reduce performance overestimation by 5-15%",
                "issues_count": len(group_issues),
                "estimated_effort": "2-4 hours"
            })
        
        # Temporal issues
        temporal_issues = [i for i in self.leakage_issues 
                          if i.leakage_type == LeakageType.TEMPORAL_LEAKAGE]
        if temporal_issues:
            recommendations.append({
                "priority": "critical",
                "category": "Time Series",
                "recommendation": "Implement proper temporal validation",
                "impact": "Prevent 20-80% performance overestimation",
                "issues_count": len(temporal_issues),
                "estimated_effort": "1-2 days"
            })
        
        # General recommendations
        if self.leakage_issues:
            recommendations.append({
                "priority": "medium",
                "category": "Best Practices",
                "recommendation": "Implement comprehensive data validation pipeline",
                "impact": "Prevent future leakage issues",
                "issues_count": 0,
                "estimated_effort": "1-2 weeks"
            })
        
        return recommendations