from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor
"""
Anomaly Detection for Unusual Code Patterns

This module implements machine learning-based anomaly detection to identify
unusual, suspicious, or non-standard code patterns that deviate from normal
coding practices and established patterns in the codebase.
"""

import ast
import re
import os
import math
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .base import BaseAnalyzer


class AnomalyType(Enum):
    """Types of code anomalies"""
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    STATISTICAL = "statistical"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    BEHAVIORAL = "behavioral"


@dataclass
class CodeAnomaly:
    """Represents a detected code anomaly"""
    anomaly_type: AnomalyType
    severity: str  # "low", "medium", "high", "critical"
    location: str
    description: str
    confidence: float  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    anomaly_score: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureVector:
    """Feature vector for code analysis"""
    structural_features: List[float] = field(default_factory=list)
    complexity_features: List[float] = field(default_factory=list)
    lexical_features: List[float] = field(default_factory=list)
    semantic_features: List[float] = field(default_factory=list)
    behavioral_features: List[float] = field(default_factory=list)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML processing"""
        all_features = (
            self.structural_features +
            self.complexity_features +
            self.lexical_features +
            self.semantic_features +
            self.behavioral_features
        )
        return np.array(all_features, dtype=np.float32)


class CodeNormalizer:
    """Normalizes code for consistent analysis"""
    
    def normalize_identifiers(self, code: str) -> str:
        """Normalize identifier names to focus on structure"""
        # Replace all identifiers with generic names
        normalized = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', 'IDENTIFIER', code)
        # Replace string literals
        normalized = re.sub(r'["\'][^"\']*["\']', 'STRING', normalized)
        # Replace numeric literals
        normalized = re.sub(r'\b\d+\.?\d*\b', 'NUMBER', normalized)
        return normalized
    
    def extract_structure_pattern(self, node: ast.AST) -> str:
        """Extract structural pattern from AST node"""
        if isinstance(node, ast.FunctionDef):
            return f"FunctionDef({len(node.args.args)}args,{len(node.body)}body)"
        elif isinstance(node, ast.ClassDef):
            return f"ClassDef({len(node.bases)}bases,{len(node.body)}body)"
        elif isinstance(node, ast.If):
            return f"If({len(node.orelse)}else)"
        elif isinstance(node, ast.For):
            return f"For({len(node.orelse)}else)"
        elif isinstance(node, ast.While):
            return f"While({len(node.orelse)}else)"
        else:
            return type(node).__name__


class FeatureExtractor:
    """Extracts features from code for anomaly detection"""
    
    def __init__(self):
        self.normalizer = CodeNormalizer()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100, 
            ngram_range=(1, 2),
            stop_words='english'
        )
        self._initialize_known_patterns()
    
    def _initialize_known_patterns(self):
        """Initialize patterns known to be normal or suspicious"""
        self.suspicious_patterns = {
            'eval_usage': re.compile(r'\beval\s*\('),
            'exec_usage': re.compile(r'\bexec\s*\('),
            'import_star': re.compile(r'from\s+\w+\s+import\s+\*'),
            'bare_except': re.compile(r'except\s*:'),
            'global_usage': re.compile(r'\bglobal\s+\w+'),
            'nonlocal_usage': re.compile(r'\bnonlocal\s+\w+'),
            'magic_numbers': re.compile(r'\b\d{4,}\b'),  # Large numbers
            'complex_regex': re.compile(r'r["\'][^"\']{50,}["\']'),  # Very long regex
        }
        
        self.normal_patterns = {
            'imports': re.compile(r'^import\s+\w+|^from\s+\w+\s+import\s+\w+'),
            'docstrings': re.compile(r'""".*?"""', re.DOTALL),
            'comments': re.compile(r'#.*$', re.MULTILINE),
            'standard_functions': re.compile(r'\b(print|len|range|enumerate|zip|map|filter)\s*\('),
        }
    
    def extract_features(self, file_path: str, content: str, tree: ast.AST) -> FeatureVector:
        """Extract comprehensive features from code"""
        features = FeatureVector()
        
        # Structural features
        features.structural_features = self._extract_structural_features(tree)
        
        # Complexity features
        features.complexity_features = self._extract_complexity_features(tree, content)
        
        # Lexical features
        features.lexical_features = self._extract_lexical_features(content)
        
        # Semantic features
        features.semantic_features = self._extract_semantic_features(tree, content)
        
        # Behavioral features
        features.behavioral_features = self._extract_behavioral_features(tree, content)
        
        return features
    
    def _extract_structural_features(self, tree: ast.AST) -> List[float]:
        """Extract structural features from AST"""
        features = []
        
        # Count different node types
        node_counts = defaultdict(int)
        max_depth = 0
        
        def count_nodes(node, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            node_counts[type(node).__name__] += 1
            for child in ast.iter_child_nodes(node):
                count_nodes(child, depth + 1)
        
        count_nodes(tree)
        
        # Normalize counts by total nodes
        total_nodes = sum(node_counts.values())
        
        # Key structural metrics
        features.extend([
            node_counts.get('FunctionDef', 0) / max(total_nodes, 1),
            node_counts.get('ClassDef', 0) / max(total_nodes, 1),
            node_counts.get('If', 0) / max(total_nodes, 1),
            node_counts.get('For', 0) / max(total_nodes, 1),
            node_counts.get('While', 0) / max(total_nodes, 1),
            node_counts.get('Try', 0) / max(total_nodes, 1),
            node_counts.get('With', 0) / max(total_nodes, 1),
            node_counts.get('Lambda', 0) / max(total_nodes, 1),
            max_depth / max(total_nodes, 1),  # Normalized depth
            total_nodes  # Raw count for scale reference
        ])
        
        return features
    
    def _extract_complexity_features(self, tree: ast.AST, content: str) -> List[float]:
        """Extract complexity-related features"""
        features = []
        
        # Cyclomatic complexity
        cyclomatic = self._calculate_cyclomatic_complexity(tree)
        
        # Halstead metrics
        halstead = self._calculate_halstead_metrics(tree)
        
        # Line-based metrics
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        features.extend([
            cyclomatic,
            halstead.get('vocabulary', 0),
            halstead.get('volume', 0),
            halstead.get('difficulty', 0),
            len(lines),  # Total lines
            len(non_empty_lines),  # Non-empty lines
            len(non_empty_lines) / max(len(lines), 1),  # Code density
        ])
        
        return features
    
    def _extract_lexical_features(self, content: str) -> List[float]:
        """Extract lexical and textual features"""
        features = []
        
        # Basic text metrics
        words = re.findall(r'\b\w+\b', content)
        unique_words = set(words)
        
        # Character-level metrics
        features.extend([
            len(content),  # Total characters
            len(words),  # Total words
            len(unique_words),  # Unique words
            len(unique_words) / max(len(words), 1),  # Vocabulary diversity
            content.count('\n'),  # Line count
            content.count(' '),  # Space count
            content.count('\t'),  # Tab count
        ])
        
        # Suspicious pattern counts
        for pattern_name, pattern in self.suspicious_patterns.items():
            matches = len(pattern.findall(content))
            features.append(matches)
        
        # Normal pattern counts
        for pattern_name, pattern in self.normal_patterns.items():
            matches = len(pattern.findall(content))
            features.append(matches)
        
        return features
    
    def _extract_semantic_features(self, tree: ast.AST, content: str) -> List[float]:
        """Extract semantic features"""
        features = []
        
        # Import analysis
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend([f"{module}.{alias.name}" for alias in node.names])
        
        # Function/method analysis
        functions = []
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        features.extend([
            len(imports),  # Number of imports
            len(set(imports)),  # Unique imports
            len(functions),  # Number of functions
            len(set(functions)),  # Unique function names
            len(classes),  # Number of classes
            len(set(classes)),  # Unique class names
        ])
        
        # Name analysis (looking for unusual naming patterns)
        all_names = functions + classes + [imp.split('.')[-1] for imp in imports]
        if all_names:
            avg_name_length = sum(len(name) for name in all_names) / len(all_names)
            features.append(avg_name_length)
        else:
            features.append(0)
        
        return features
    
    def _extract_behavioral_features(self, tree: ast.AST, content: str) -> List[float]:
        """Extract behavioral pattern features"""
        features = []
        
        # Control flow patterns
        if_else_ratio = 0
        try_except_ratio = 0
        loop_nesting = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if node.orelse:
                    if_else_ratio += 1
            elif isinstance(node, ast.Try):
                if node.handlers or node.orelse or node.finalbody:
                    try_except_ratio += 1
        
        # Calculate nesting depth for loops
        def calculate_loop_nesting(node, depth=0):
            nonlocal loop_nesting
            if isinstance(node, (ast.For, ast.While)):
                loop_nesting = max(loop_nesting, depth + 1)
                depth += 1
            for child in ast.iter_child_nodes(node):
                calculate_loop_nesting(child, depth)
        
        calculate_loop_nesting(tree)
        
        features.extend([
            if_else_ratio,
            try_except_ratio,
            loop_nesting,
        ])
        
        # File-level behavioral patterns
        features.extend([
            1 if 'main' in content else 0,  # Has main function/guard
            1 if '__name__' in content else 0,  # Uses name guard
            1 if 'logging' in content else 0,  # Uses logging
            1 if 'TODO' in content or 'FIXME' in content else 0,  # Has TODOs
        ])
        
        return features
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.Break, ast.Continue)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
        
        return complexity
    
    def _calculate_halstead_metrics(self, tree: ast.AST) -> Dict[str, float]:
        """Calculate Halstead complexity metrics"""
        operators = set()
        operands = set()
        operator_count = 0
        operand_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                operators.add(type(node.op).__name__)
                operator_count += 1
            elif isinstance(node, ast.UnaryOp):
                operators.add(type(node.op).__name__)
                operator_count += 1
            elif isinstance(node, ast.Compare):
                for op in node.ops:
                    operators.add(type(op).__name__)
                    operator_count += 1
            elif isinstance(node, ast.Name):
                operands.add(node.id)
                operand_count += 1
            elif isinstance(node, (ast.Str, ast.Num, ast.Constant)):
                operands.add(str(type(node).__name__))
                operand_count += 1
        
        n1 = len(operators)  # Unique operators
        n2 = len(operands)   # Unique operands
        N1 = operator_count  # Total operators
        N2 = operand_count   # Total operands
        
        vocabulary = n1 + n2
        length = N1 + N2
        
        if n1 > 0 and n2 > 0:
            volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
            difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        else:
            volume = 0
            difficulty = 0
        
        return {
            'vocabulary': vocabulary,
            'length': length,
            'volume': volume,
            'difficulty': difficulty
        }


class AnomalyDetectionAnalyzer(BaseAnalyzer):
    """
    Advanced anomaly detection analyzer for unusual code patterns
    
    Uses machine learning techniques to identify code that deviates
    significantly from normal patterns and established best practices.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.feature_extractor = FeatureExtractor()
        self.normalizer = CodeNormalizer()
        self.anomalies = []
        self.normal_patterns = []
        self.feature_vectors = []
        
        # ML models
        self.isolation_forest = None
        self.dbscan = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% variance
        
        # Thresholds
        self.anomaly_threshold = config.get('anomaly_threshold', -0.1) if config else -0.1
        self.contamination = config.get('contamination', 0.1) if config else 0.1
        
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive anomaly detection analysis
        """
        self.logger.info("Starting anomaly detection analysis...")
        
        if file_path:
            self._analyze_file(Path(file_path))
        else:
            self._analyze_directory()
        
        # Train models and detect anomalies
        self._train_models()
        self._detect_anomalies()
        
        return self._generate_report()
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Extract features
            features = self.feature_extractor.extract_features(str(file_path), content, tree)
            
            # Store for later processing
            self.feature_vectors.append({
                'file_path': str(file_path),
                'features': features,
                'content': content,
                'tree': tree
            })
            
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
    
    def _train_models(self) -> None:
        """Train anomaly detection models"""
        if not self.feature_vectors:
            self.logger.warning("No feature vectors available for training")
            return
        
        # Convert features to numpy array
        feature_matrix = np.array([fv['features'].to_array() for fv in self.feature_vectors])
        
        # Handle missing values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        # Scale features
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Apply PCA for dimensionality reduction
        feature_matrix_pca = self.pca.fit_transform(feature_matrix_scaled)
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        self.isolation_forest.fit(feature_matrix_pca)
        
        # Train DBSCAN for clustering-based anomaly detection
        self.dbscan = DBSCAN(eps=0.5, min_samples=3)
        cluster_labels = self.dbscan.fit_predict(feature_matrix_pca)
        
        # Store processed features
        for i, fv in enumerate(self.feature_vectors):
            fv['features_scaled'] = feature_matrix_scaled[i]
            fv['features_pca'] = feature_matrix_pca[i]
            fv['cluster_label'] = cluster_labels[i]
    
    def _detect_anomalies(self) -> None:
        """Detect anomalies using trained models"""
        if not self.isolation_forest:
            return
        
        for fv in self.feature_vectors:
            file_path = fv['file_path']
            features_pca = fv['features_pca'].reshape(1, -1)
            
            # Isolation Forest anomaly score
            anomaly_score = self.isolation_forest.decision_function(features_pca)[0]
            is_anomaly = self.isolation_forest.predict(features_pca)[0] == -1
            
            # DBSCAN outlier (cluster label -1)
            is_outlier = fv['cluster_label'] == -1
            
            # Rule-based anomaly detection
            rule_anomalies = self._detect_rule_based_anomalies(fv)
            
            # Statistical anomaly detection
            statistical_anomalies = self._detect_statistical_anomalies(fv)
            
            # Combine all anomaly indicators
            if is_anomaly or is_outlier or rule_anomalies or statistical_anomalies:
                severity = self._calculate_severity(anomaly_score, is_anomaly, is_outlier, 
                                                 len(rule_anomalies), len(statistical_anomalies))
                
                anomaly = CodeAnomaly(
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    location=file_path,
                    description=self._generate_anomaly_description(fv, is_anomaly, is_outlier),
                    confidence=self._calculate_confidence(anomaly_score, is_anomaly, is_outlier),
                    anomaly_score=anomaly_score,
                    evidence=self._collect_evidence(fv, rule_anomalies, statistical_anomalies),
                    recommendations=self._generate_recommendations(fv, rule_anomalies, statistical_anomalies),
                    context={
                        'is_isolation_anomaly': is_anomaly,
                        'is_dbscan_outlier': is_outlier,
                        'rule_anomalies': len(rule_anomalies),
                        'statistical_anomalies': len(statistical_anomalies)
                    }
                )
                
                self.anomalies.append(anomaly)
    
    def _detect_rule_based_anomalies(self, fv: Dict[str, Any]) -> List[str]:
        """Detect anomalies using predefined rules"""
        anomalies = []
        content = fv['content']
        tree = fv['tree']
        
        # Check for suspicious patterns
        if re.search(r'\beval\s*\(', content):
            anomalies.append("Use of SafeCodeExecutor.safe_SafeCodeExecutor.safe_eval() function detected")
        
        if re.search(r'\bexec\s*\(', content):
            anomalies.append("Use of SafeCodeExecutor.safe_exec() function detected")
        
        if re.search(r'from\s+\w+\s+import\s+\*', content):
            anomalies.append("Wildcard import detected")
        
        # Check for unusual structure patterns
        function_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
        if function_count > 50:
            anomalies.append(f"Unusually high function count: {function_count}")
        
        # Check for very long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.body) > 100:
                    anomalies.append(f"Very long function detected: {node.name} ({len(node.body)} statements)")
        
        # Check for deep nesting
        max_depth = self._calculate_max_nesting_depth(tree)
        if max_depth > 8:
            anomalies.append(f"Excessive nesting depth: {max_depth}")
        
        return anomalies
    
    def _detect_statistical_anomalies(self, fv: Dict[str, Any]) -> List[str]:
        """Detect statistical anomalies"""
        anomalies = []
        features = fv['features']
        
        # Check for statistical outliers in feature values
        feature_array = features.to_array()
        
        # Z-score based detection
        if len(self.feature_vectors) > 3:  # Need enough samples
            all_features = np.array([f['features'].to_array() for f in self.feature_vectors])
            mean_features = np.mean(all_features, axis=0)
            std_features = np.std(all_features, axis=0)
            
            z_scores = np.abs((feature_array - mean_features) / (std_features + 1e-8))
            outlier_indices = np.where(z_scores > 3)[0]  # 3-sigma rule
            
            if len(outlier_indices) > 5:  # More than 5 features are outliers
                anomalies.append(f"Multiple statistical outliers detected: {len(outlier_indices)} features")
        
        return anomalies
    
    def _calculate_max_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        def get_depth(node, current_depth=0):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)):
                current_depth += 1
            
            max_child_depth = current_depth
            for child in ast.iter_child_nodes(node):
                max_child_depth = max(max_child_depth, get_depth(child, current_depth))
            
            return max_child_depth
        
        return get_depth(tree)
    
    def _calculate_severity(self, anomaly_score: float, is_anomaly: bool, is_outlier: bool, 
                          rule_count: int, stat_count: int) -> str:
        """Calculate anomaly severity"""
        score = 0
        
        if is_anomaly:
            score += 2
        if is_outlier:
            score += 2
        score += rule_count
        score += stat_count
        
        # Adjust by anomaly score
        if anomaly_score < -0.5:
            score += 2
        elif anomaly_score < -0.2:
            score += 1
        
        if score >= 6:
            return "critical"
        elif score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        else:
            return "low"
    
    def _calculate_confidence(self, anomaly_score: float, is_anomaly: bool, is_outlier: bool) -> float:
        """Calculate confidence in anomaly detection"""
        confidence = 0.5  # Base confidence
        
        if is_anomaly:
            confidence += 0.3
        if is_outlier:
            confidence += 0.2
        
        # Adjust by score magnitude
        score_magnitude = abs(anomaly_score)
        if score_magnitude > 0.5:
            confidence += 0.2
        elif score_magnitude > 0.2:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_anomaly_description(self, fv: Dict[str, Any], is_anomaly: bool, is_outlier: bool) -> str:
        """Generate description for detected anomaly"""
        descriptions = []
        
        if is_anomaly:
            descriptions.append("Statistical anomaly detected by Isolation Forest")
        if is_outlier:
            descriptions.append("Outlier detected by clustering analysis")
        
        if not descriptions:
            descriptions.append("Unusual code pattern detected")
        
        return "; ".join(descriptions)
    
    def _collect_evidence(self, fv: Dict[str, Any], rule_anomalies: List[str], 
                         statistical_anomalies: List[str]) -> List[str]:
        """Collect evidence for the anomaly"""
        evidence = []
        evidence.extend(rule_anomalies)
        evidence.extend(statistical_anomalies)
        
        # Add feature-based evidence
        features = fv['features']
        feature_array = features.to_array()
        
        if len(feature_array) > 0:
            evidence.append(f"Feature vector magnitude: {np.linalg.norm(feature_array):.2f}")
        
        return evidence
    
    def _generate_recommendations(self, fv: Dict[str, Any], rule_anomalies: List[str], 
                                statistical_anomalies: List[str]) -> List[str]:
        """Generate recommendations for addressing anomalies"""
        recommendations = []
        
        if any("eval" in anomaly for anomaly in rule_anomalies):
            recommendations.append("Replace SafeCodeExecutor.safe_SafeCodeExecutor.safe_eval() with safer alternatives like ast.literal_SafeCodeExecutor.safe_SafeCodeExecutor.safe_eval()")
        
        if any("exec" in anomaly for anomaly in rule_anomalies):
            recommendations.append("Avoid SafeCodeExecutor.safe_exec() or implement strict sandboxing")
        
        if any("import *" in anomaly for anomaly in rule_anomalies):
            recommendations.append("Use explicit imports instead of wildcard imports")
        
        if any("function count" in anomaly for anomaly in rule_anomalies):
            recommendations.append("Consider splitting file into multiple modules")
        
        if any("nesting" in anomaly for anomaly in rule_anomalies):
            recommendations.append("Refactor to reduce nesting depth using early returns or helper functions")
        
        if statistical_anomalies:
            recommendations.append("Review code structure for consistency with project patterns")
        
        if not recommendations:
            recommendations.append("Manual review recommended to assess code quality and security")
        
        return recommendations
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive anomaly detection report"""
        # Categorize anomalies by type and severity
        by_severity = defaultdict(list)
        by_type = defaultdict(list)
        
        for anomaly in self.anomalies:
            by_severity[anomaly.severity].append(anomaly)
            by_type[anomaly.anomaly_type.value].append(anomaly)
        
        # Calculate statistics
        total_files = len(self.feature_vectors)
        anomalous_files = len(self.anomalies)
        anomaly_rate = anomalous_files / total_files if total_files > 0 else 0
        
        # Risk assessment
        risk_score = self._calculate_overall_risk()
        
        return {
            "summary": {
                "total_files_analyzed": total_files,
                "anomalous_files": anomalous_files,
                "anomaly_rate": anomaly_rate,
                "overall_risk_score": risk_score,
                "detection_models_trained": bool(self.isolation_forest),
                "feature_dimensions": len(self.feature_vectors[0]['features'].to_array()) if self.feature_vectors else 0
            },
            "anomalies_by_severity": {
                severity: [
                    {
                        "file": anomaly.location,
                        "type": anomaly.anomaly_type.value,
                        "description": anomaly.description,
                        "confidence": anomaly.confidence,
                        "score": anomaly.anomaly_score,
                        "evidence": anomaly.evidence,
                        "recommendations": anomaly.recommendations
                    }
                    for anomaly in anomalies
                ]
                for severity, anomalies in by_severity.items()
            },
            "anomalies_by_type": {
                anomaly_type: len(anomalies)
                for anomaly_type, anomalies in by_type.items()
            },
            "detailed_anomalies": [
                {
                    "file": anomaly.location,
                    "type": anomaly.anomaly_type.value,
                    "severity": anomaly.severity,
                    "description": anomaly.description,
                    "confidence": anomaly.confidence,
                    "anomaly_score": anomaly.anomaly_score,
                    "evidence": anomaly.evidence,
                    "recommendations": anomaly.recommendations,
                    "context": anomaly.context
                }
                for anomaly in self.anomalies
            ],
            "model_info": {
                "isolation_forest_trained": bool(self.isolation_forest),
                "dbscan_clusters": len(set(fv.get('cluster_label', -1) for fv in self.feature_vectors)) - (1 if -1 in [fv.get('cluster_label', -1) for fv in self.feature_vectors] else 0),
                "outlier_count": sum(1 for fv in self.feature_vectors if fv.get('cluster_label', -1) == -1),
                "feature_dimensions_original": len(self.feature_vectors[0]['features'].to_array()) if self.feature_vectors else 0,
                "feature_dimensions_pca": self.pca.n_components_ if hasattr(self.pca, 'n_components_') else 0
            },
            "recommendations": self._generate_overall_recommendations(),
            "risk_assessment": {
                "overall_risk": risk_score,
                "risk_level": self._get_risk_level(risk_score),
                "critical_issues": len(by_severity.get("critical", [])),
                "high_issues": len(by_severity.get("high", [])),
                "recommendations": self._generate_risk_mitigation_recommendations()
            }
        }
    
    def _calculate_overall_risk(self) -> float:
        """Calculate overall risk score based on detected anomalies"""
        if not self.anomalies:
            return 0.0
        
        severity_weights = {
            "critical": 4.0,
            "high": 3.0,
            "medium": 2.0,
            "low": 1.0
        }
        
        total_weight = sum(severity_weights.get(anomaly.severity, 1.0) * anomaly.confidence 
                          for anomaly in self.anomalies)
        
        # Normalize by number of files
        total_files = len(self.feature_vectors)
        risk_score = total_weight / max(total_files, 1)
        
        return min(risk_score, 10.0)  # Cap at 10
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score >= 7:
            return "CRITICAL"
        elif risk_score >= 5:
            return "HIGH"
        elif risk_score >= 3:
            return "MEDIUM"
        elif risk_score >= 1:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall recommendations based on all detected anomalies"""
        recommendations = []
        
        if len(self.anomalies) > len(self.feature_vectors) * 0.3:
            recommendations.append("High anomaly rate detected - consider comprehensive code review")
        
        severity_counts = Counter(anomaly.severity for anomaly in self.anomalies)
        
        if severity_counts.get("critical", 0) > 0:
            recommendations.append("Critical anomalies detected - immediate action required")
        
        if severity_counts.get("high", 0) > 3:
            recommendations.append("Multiple high-severity anomalies - prioritize fixes")
        
        # Add specific recommendations based on anomaly types
        common_recommendations = Counter()
        for anomaly in self.anomalies:
            for rec in anomaly.recommendations:
                common_recommendations[rec] += 1
        
        # Include most common recommendations
        for rec, count in common_recommendations.most_common(3):
            if count > 1:
                recommendations.append(f"{rec} (affects {count} files)")
        
        return recommendations
    
    def _generate_risk_mitigation_recommendations(self) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        critical_count = len([a for a in self.anomalies if a.severity == "critical"])
        high_count = len([a for a in self.anomalies if a.severity == "high"])
        
        if critical_count > 0:
            recommendations.append(f"Address {critical_count} critical anomalies immediately")
        
        if high_count > 0:
            recommendations.append(f"Review and fix {high_count} high-severity anomalies")
        
        recommendations.extend([
            "Implement automated code quality gates",
            "Consider additional static analysis tools",
            "Establish coding standards and enforcement",
            "Regular security code reviews",
            "Developer training on secure coding practices"
        ])
        
        return recommendations