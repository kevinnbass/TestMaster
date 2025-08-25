"""
ML Data Analyzer

Handles data pipeline analysis, preprocessing patterns, and data-related issues.
Split from original ml_code_analysis.py - Data Pipeline & Preprocessing sections.
"""

import ast
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
import logging

from ...base import BaseAnalyzer
from ._shared_utils import MLIssue, DATA_SCIENCE_PATTERNS


class MLDataAnalyzer(BaseAnalyzer):
    """
    Specialized analyzer for ML data pipelines, preprocessing, and data-related patterns
    """
    
    def __init__(self):
        super().__init__()
        self.data_issues = []
        self.data_patterns = []
        self.preprocessing_steps = []
        self.data_leakage_risks = []
        
    def analyze(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform data-focused analysis
        """
        if file_path:
            self._analyze_file(Path(file_path))
        else:
            self._analyze_directory()
            
        return self._generate_data_report()
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single file for data patterns"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Analyze data loading patterns
            self._analyze_data_loading(tree, content, str(file_path))
            
            # Analyze preprocessing
            self._analyze_preprocessing(tree, content, str(file_path))
            
            # Check for data leakage
            self._detect_data_leakage(tree, content, str(file_path))
            
            # Analyze data validation
            self._analyze_data_validation(tree, content, str(file_path))
            
            # Check data pipeline efficiency
            self._analyze_data_efficiency(tree, content, str(file_path))
            
        except Exception as e:
            logging.error(f"Error analyzing data in {file_path}: {e}")
    
    def _analyze_data_loading(self, tree: ast.AST, content: str, file_path: str) -> None:
        """Analyze data loading patterns"""
        
        # Data loading patterns
        loading_patterns = {
            "pandas": ["pd.read_csv", "pd.read_excel", "pd.read_json", "pd.read_parquet"],
            "numpy": ["np.load", "np.loadtxt", "np.genfromtxt"],
            "torch": ["torch.load", "DataLoader", "Dataset"],
            "tensorflow": ["tf.data", "tf.io", "tfds.load"],
            "sklearn": ["load_iris", "load_boston", "load_digits", "fetch_"]
        }
        
        for framework, patterns in loading_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    self.data_patterns.append({
                        "type": "data_loading",
                        "framework": framework,
                        "pattern": pattern,
                        "file": file_path,
                        "count": content.count(pattern)
                    })
        
        # Check for data loading issues
        self._check_data_loading_issues(content, file_path)
    
    def _check_data_loading_issues(self, content: str, file_path: str) -> None:
        """Check for data loading issues"""
        
        # Large file loading without chunking
        if "read_csv" in content:
            if "chunksize" not in content and "nrows" not in content:
                # Check for potential large file indicators
                large_file_indicators = ["GB", "million", "large", "big_data"]
                if any(indicator in content.lower() for indicator in large_file_indicators):
                    self.data_issues.append(MLIssue(
                        type="large_file_no_chunking",
                        severity="medium",
                        location=file_path,
                        description="Large file loading without chunking",
                        recommendation="Use chunksize parameter for large files",
                        framework="pandas",
                        impact="memory_usage"
                    ))
        
        # Missing error handling for data loading
        has_data_loading = any(pattern in content for pattern in 
                              ["read_csv", "read_excel", "load_data", "DataLoader"])
        has_error_handling = "try:" in content or "except" in content
        
        if has_data_loading and not has_error_handling:
            self.data_issues.append(MLIssue(
                type="missing_data_error_handling",
                severity="medium",
                location=file_path,
                description="Data loading without error handling",
                recommendation="Add try-except blocks for data loading operations",
                framework="general",
                impact="robustness"
            ))
        
        # Hardcoded file paths
        file_path_patterns = [r'["\'][A-Za-z]:\\[^"\']+["\']', r'["\'][^"\']*\.[csv|xlsx|json|parquet]["\']']
        for pattern in file_path_patterns:
            if re.search(pattern, content):
                self.data_issues.append(MLIssue(
                    type="hardcoded_file_path",
                    severity="low",
                    location=file_path,
                    description="Hardcoded file paths detected",
                    recommendation="Use configuration files or environment variables",
                    framework="general",
                    impact="maintainability"
                ))
    
    def _analyze_preprocessing(self, tree: ast.AST, content: str, file_path: str) -> None:
        """Analyze preprocessing patterns"""
        
        # Preprocessing patterns from DATA_SCIENCE_PATTERNS
        for pattern_type, patterns in DATA_SCIENCE_PATTERNS.items():
            for pattern in patterns:
                if pattern in content:
                    self.preprocessing_steps.append({
                        "type": pattern_type,
                        "pattern": pattern,
                        "file": file_path,
                        "count": content.count(pattern)
                    })
        
        # Additional preprocessing patterns
        additional_patterns = {
            "scaling": ["StandardScaler", "MinMaxScaler", "RobustScaler", "normalize"],
            "encoding": ["LabelEncoder", "OneHotEncoder", "get_dummies", "categorical"],
            "imputation": ["fillna", "dropna", "SimpleImputer", "KNNImputer"],
            "feature_selection": ["SelectKBest", "RFE", "SelectFromModel", "VarianceThreshold"],
            "dimensionality_reduction": ["PCA", "TruncatedSVD", "UMAP", "TSNE"],
            "text_processing": ["CountVectorizer", "TfidfVectorizer", "tokenize", "lemmatize"]
        }
        
        for prep_type, patterns in additional_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    self.preprocessing_steps.append({
                        "type": prep_type,
                        "pattern": pattern,
                        "file": file_path,
                        "count": content.count(pattern)
                    })
        
        # Check preprocessing order and issues
        self._check_preprocessing_issues(content, file_path)
    
    def _check_preprocessing_issues(self, content: str, file_path: str) -> None:
        """Check for preprocessing issues"""
        
        # Preprocessing before train/test split (data leakage)
        split_patterns = ["train_test_split", "StratifiedKFold", "KFold"]
        preprocessing_patterns = ["fit_transform", "StandardScaler", "MinMaxScaler"]
        
        has_split = any(pattern in content for pattern in split_patterns)
        has_preprocessing = any(pattern in content for pattern in preprocessing_patterns)
        
        if has_preprocessing and has_split:
            # Simple heuristic: check if preprocessing appears before split in text
            split_positions = []
            prep_positions = []
            
            for pattern in split_patterns:
                pos = content.find(pattern)
                if pos != -1:
                    split_positions.append(pos)
            
            for pattern in preprocessing_patterns:
                pos = content.find(pattern)
                if pos != -1:
                    prep_positions.append(pos)
            
            if prep_positions and split_positions:
                if min(prep_positions) < min(split_positions):
                    self.data_leakage_risks.append({
                        "type": "preprocessing_before_split",
                        "severity": "critical",
                        "file": file_path,
                        "description": "Preprocessing applied before train/test split"
                    })
        
        # Missing data validation
        validation_patterns = ["isnull", "isna", "info()", "describe()", "dtypes"]
        has_validation = any(pattern in content for pattern in validation_patterns)
        
        if has_preprocessing and not has_validation:
            self.data_issues.append(MLIssue(
                type="missing_data_validation",
                severity="medium",
                location=file_path,
                description="Preprocessing without data validation",
                recommendation="Add data validation before preprocessing",
                framework="general",
                impact="data_quality"
            ))
        
        # Inconsistent preprocessing
        scaler_count = content.count("Scaler")
        if scaler_count > 1:
            different_scalers = sum(1 for scaler in ["StandardScaler", "MinMaxScaler", "RobustScaler"] 
                                  if scaler in content)
            if different_scalers > 1:
                self.data_issues.append(MLIssue(
                    type="inconsistent_scaling",
                    severity="low",
                    location=file_path,
                    description="Multiple different scalers used",
                    recommendation="Use consistent scaling method",
                    framework="sklearn",
                    impact="model_performance"
                ))
    
    def _detect_data_leakage(self, tree: ast.AST, content: str, file_path: str) -> None:
        """Detect potential data leakage patterns"""
        
        # Target leakage patterns
        target_leakage_patterns = [
            r"df\[.*target.*\].*=.*df\[.*\]",  # Using target to create features
            r".*merge.*target",  # Merging target
            r".*target.*shift\(-",  # Future target values
        ]
        
        for pattern in target_leakage_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                self.data_leakage_risks.append({
                    "type": "target_leakage",
                    "severity": "critical",
                    "file": file_path,
                    "description": f"Potential target leakage: {pattern}"
                })
        
        # Temporal leakage patterns
        temporal_patterns = [
            r"sort.*ascending=False",  # Sorting by date descending
            r"\.tail\(\)",  # Using last observations
            r"max\(.*date\)",  # Using maximum date
        ]
        
        for pattern in temporal_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                self.data_leakage_risks.append({
                    "type": "temporal_leakage",
                    "severity": "high",
                    "file": file_path,
                    "description": f"Potential temporal leakage: {pattern}"
                })
        
        # Feature leakage patterns
        feature_patterns = [
            r"fit\(X\)",  # Fitting on all data
            r"transform\(.*test.*train\)",  # Transforming test with train
        ]
        
        for pattern in feature_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                self.data_leakage_risks.append({
                    "type": "feature_leakage",
                    "severity": "high",
                    "file": file_path,
                    "description": f"Potential feature leakage: {pattern}"
                })
    
    def _analyze_data_validation(self, tree: ast.AST, content: str, file_path: str) -> None:
        """Analyze data validation patterns"""
        
        # Data quality checks
        quality_patterns = {
            "null_check": ["isnull", "isna", "notna", "notnull"],
            "duplicate_check": ["duplicated", "drop_duplicates"],
            "type_check": ["dtypes", "astype", "infer_objects"],
            "range_check": ["between", "clip", "where"],
            "outlier_check": ["quantile", "IQR", "outlier", "zscore"],
            "consistency_check": ["unique", "nunique", "value_counts"]
        }
        
        for check_type, patterns in quality_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    self.data_patterns.append({
                        "type": f"validation_{check_type}",
                        "pattern": pattern,
                        "file": file_path,
                        "count": content.count(pattern)
                    })
        
        # Data profiling
        profiling_patterns = ["info()", "describe()", "head()", "tail()", "sample()"]
        for pattern in profiling_patterns:
            if pattern in content:
                self.data_patterns.append({
                    "type": "data_profiling",
                    "pattern": pattern,
                    "file": file_path,
                    "count": content.count(pattern)
                })
    
    def _analyze_data_efficiency(self, tree: ast.AST, content: str, file_path: str) -> None:
        """Analyze data pipeline efficiency"""
        
        # Inefficient patterns
        inefficient_patterns = {
            "iterrows": [r"\.iterrows\(\)", r"for.*in.*\.iterrows"],
            "apply_with_lambda": [r"\.apply\(lambda"],
            "chained_operations": [r"\]\[.*\]\[.*\]\["],  # Multiple bracket operations
            "unnecessary_copy": [r"\.copy\(\)\.copy\(\)", r"\.copy\(\).*\.copy\(\)"],
        }
        
        for pattern_type, patterns in inefficient_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    self.data_issues.append(MLIssue(
                        type=f"inefficient_{pattern_type}",
                        severity="medium",
                        location=file_path,
                        description=f"Inefficient data operation: {pattern_type}",
                        recommendation=self._get_efficiency_recommendation(pattern_type),
                        framework="pandas",
                        impact="performance"
                    ))
        
        # Memory usage patterns
        memory_patterns = [
            "memory_usage", "info(memory_usage=True)", "sys.getsizeof"
        ]
        
        has_memory_monitoring = any(pattern in content for pattern in memory_patterns)
        has_large_data = any(indicator in content.lower() 
                            for indicator in ["large", "big", "memory", "GB", "TB"])
        
        if has_large_data and not has_memory_monitoring:
            self.data_issues.append(MLIssue(
                type="no_memory_monitoring",
                severity="low",
                location=file_path,
                description="Large data processing without memory monitoring",
                recommendation="Add memory usage monitoring",
                framework="general",
                impact="resource_usage"
            ))
    
    def _get_efficiency_recommendation(self, pattern_type: str) -> str:
        """Get efficiency recommendation for pattern type"""
        recommendations = {
            "iterrows": "Use vectorized operations instead of iterrows()",
            "apply_with_lambda": "Use vectorized operations or built-in functions",
            "chained_operations": "Use method chaining or intermediate variables",
            "unnecessary_copy": "Remove unnecessary copy() calls"
        }
        return recommendations.get(pattern_type, "Optimize data operations")
    
    def _generate_data_report(self) -> Dict[str, Any]:
        """Generate comprehensive data analysis report"""
        
        # Calculate statistics
        total_issues = len(self.data_issues)
        total_patterns = len(self.data_patterns)
        total_preprocessing = len(self.preprocessing_steps)
        total_leakage_risks = len(self.data_leakage_risks)
        
        # Group issues by type and severity
        issues_by_type = defaultdict(int)
        issues_by_severity = defaultdict(int)
        
        for issue in self.data_issues:
            issues_by_type[issue.type] += 1
            issues_by_severity[issue.severity] += 1
        
        # Group patterns by type
        patterns_by_type = defaultdict(list)
        for pattern in self.data_patterns:
            patterns_by_type[pattern["type"]].append(pattern)
        
        # Group preprocessing by type
        preprocessing_by_type = defaultdict(list)
        for step in self.preprocessing_steps:
            preprocessing_by_type[step["type"]].append(step)
        
        # Analyze data leakage risks
        leakage_by_type = defaultdict(list)
        for risk in self.data_leakage_risks:
            leakage_by_type[risk["type"]].append(risk)
        
        return {
            "summary": {
                "total_data_issues": total_issues,
                "total_data_patterns": total_patterns,
                "total_preprocessing_steps": total_preprocessing,
                "total_leakage_risks": total_leakage_risks,
                "critical_leakage_risks": len([r for r in self.data_leakage_risks if r["severity"] == "critical"]),
                "high_severity_issues": issues_by_severity.get("high", 0),
                "medium_severity_issues": issues_by_severity.get("medium", 0),
                "files_analyzed": len(set(issue.location for issue in self.data_issues))
            },
            "data_issues": [
                {
                    "type": issue.type,
                    "severity": issue.severity,
                    "location": issue.location,
                    "description": issue.description,
                    "recommendation": issue.recommendation,
                    "framework": issue.framework,
                    "impact": issue.impact
                }
                for issue in self.data_issues
            ],
            "data_patterns": {
                "by_type": dict(patterns_by_type),
                "all_patterns": self.data_patterns
            },
            "preprocessing_analysis": {
                "by_type": dict(preprocessing_by_type),
                "all_steps": self.preprocessing_steps,
                "preprocessing_coverage": len(set(step["type"] for step in self.preprocessing_steps))
            },
            "data_leakage_analysis": {
                "by_type": dict(leakage_by_type),
                "all_risks": self.data_leakage_risks,
                "risk_severity_distribution": {
                    "critical": len([r for r in self.data_leakage_risks if r["severity"] == "critical"]),
                    "high": len([r for r in self.data_leakage_risks if r["severity"] == "high"]),
                    "medium": len([r for r in self.data_leakage_risks if r["severity"] == "medium"])
                }
            },
            "issues_by_type": dict(issues_by_type),
            "issues_by_severity": dict(issues_by_severity),
            "recommendations": self._generate_data_recommendations()
        }
    
    def _generate_data_recommendations(self) -> List[Dict[str, str]]:
        """Generate data-specific recommendations"""
        recommendations = []
        
        # Data leakage recommendations
        if self.data_leakage_risks:
            critical_leakage = [r for r in self.data_leakage_risks if r["severity"] == "critical"]
            if critical_leakage:
                recommendations.append({
                    "category": "Data Leakage",
                    "priority": "critical",
                    "recommendation": "Fix critical data leakage issues immediately",
                    "impact": "Prevent invalid model performance estimates"
                })
        
        # Data quality recommendations
        quality_issues = [i for i in self.data_issues if "validation" in i.type or "quality" in i.impact]
        if quality_issues:
            recommendations.append({
                "category": "Data Quality",
                "priority": "high",
                "recommendation": "Improve data validation and quality checks",
                "impact": "Better data quality and model reliability"
            })
        
        # Performance recommendations
        perf_issues = [i for i in self.data_issues if "inefficient" in i.type or i.impact == "performance"]
        if perf_issues:
            recommendations.append({
                "category": "Data Processing Performance",
                "priority": "medium",
                "recommendation": "Optimize data processing operations",
                "impact": "Faster data pipeline execution"
            })
        
        # Preprocessing recommendations
        if len(self.preprocessing_steps) > 10:
            recommendations.append({
                "category": "Preprocessing Complexity",
                "priority": "low",
                "recommendation": "Consider simplifying preprocessing pipeline",
                "impact": "Reduced complexity and better maintainability"
            })
        
        # Data loading recommendations
        loading_issues = [i for i in self.data_issues if "loading" in i.type or "file" in i.type]
        if loading_issues:
            recommendations.append({
                "category": "Data Loading",
                "priority": "medium",
                "recommendation": "Improve data loading practices",
                "impact": "More robust and efficient data loading"
            })
        
        return recommendations