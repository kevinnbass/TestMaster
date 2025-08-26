"""
ML Core Analyzer

Handles ML framework detection, basic analysis, and coordination with other ML analyzers.
Split from original ml_code_analysis.py - Framework Detection & Core Analysis sections.
"""

import ast
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
import logging

from ...base import BaseAnalyzer
from ._shared_utils import (
    MLIssue, ML_FRAMEWORKS, ML_ANTIPATTERNS, 
    detect_ml_frameworks_in_content, analyze_import_patterns, 
    check_version_compatibility
)


class MLCoreAnalyzer(BaseAnalyzer):
    """
    Core ML code analyzer responsible for framework detection,
    basic analysis, and coordination with specialized analyzers.
    """
    
    def __init__(self):
        super().__init__()
        self.ml_issues = []
        self.detected_frameworks = {}
        
    def analyze(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform core ML analysis including framework detection
        """
        if file_path:
            self._analyze_file(Path(file_path))
        else:
            self._analyze_directory()
            
        return self._generate_core_report()
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single file for ML patterns"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Framework detection
            frameworks = self._detect_ml_frameworks_in_file(content, str(file_path))
            if frameworks:
                self.detected_frameworks[str(file_path)] = frameworks
            
            # Basic ML pattern analysis
            self._analyze_ml_patterns(tree, content, str(file_path))
            
            # Check for common ML issues
            self._check_ml_antipatterns(content, str(file_path))
            
        except Exception as e:
            logging.error(f"Error analyzing {file_path}: {e}")
    
    def _detect_ml_frameworks_in_file(self, content: str, file_path: str) -> Dict[str, Any]:
        """
        Detect which ML frameworks are being used in a specific file
        """
        framework_usage = {
            "detected_frameworks": [],
            "framework_versions": {},
            "framework_conflicts": [],
            "usage_statistics": defaultdict(int),
            "import_patterns": {}
        }
        
        # Parse AST for detailed import analysis
        try:
            tree = ast.parse(content)
            framework_usage["import_patterns"] = analyze_import_patterns(tree)
        except SyntaxError:
            pass
        
        # Detect frameworks using content patterns
        detected = detect_ml_frameworks_in_content(content)
        
        for framework, patterns in detected.items():
            framework_usage["detected_frameworks"].append({
                "framework": framework,
                "file": file_path,
                "import_patterns": patterns,
                "count": len(patterns)
            })
            framework_usage["usage_statistics"][framework] = len(patterns)
        
        # Check for version specifications
        version_patterns = {
            "tensorflow": r"tensorflow[=><!]+(\d+\.\d+)",
            "torch": r"torch[=><!]+(\d+\.\d+)",
            "sklearn": r"scikit-learn[=><!]+(\d+\.\d+)",
            "numpy": r"numpy[=><!]+(\d+\.\d+)",
            "pandas": r"pandas[=><!]+(\d+\.\d+)"
        }
        
        for framework, pattern in version_patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                framework_usage["framework_versions"][framework] = matches[0]
        
        # Check for potential conflicts
        detected_fw_names = list(detected.keys())
        conflicts = check_version_compatibility(detected_fw_names)
        framework_usage["framework_conflicts"].extend(conflicts)
        
        return framework_usage
    
    def _analyze_ml_patterns(self, tree: ast.AST, content: str, file_path: str) -> None:
        """Analyze general ML patterns in code"""
        
        # Check for data science workflow patterns
        ds_patterns = {
            "data_loading": ["read_csv", "load_data", "pd.read", "np.load"],
            "preprocessing": ["train_test_split", "StandardScaler", "normalize"],
            "model_training": ["fit(", "train(", "compile("],
            "evaluation": ["score(", "evaluate(", "predict(", "accuracy"],
            "visualization": ["plt.", "sns.", "plot(", "hist(", "scatter("]
        }
        
        for pattern_type, patterns in ds_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    # Count occurrences
                    count = content.count(pattern)
                    if count > 0:
                        self.ml_issues.append(MLIssue(
                            type=f"pattern_detected",
                            severity="info",
                            location=f"{file_path}",
                            description=f"Detected {pattern_type} pattern: {pattern}",
                            recommendation=f"Ensure proper {pattern_type} practices",
                            framework="general",
                            impact="informational"
                        ))
        
        # Check for GPU usage patterns
        gpu_patterns = [".cuda()", ".to(device)", "tf.device", "with tf.device"]
        for pattern in gpu_patterns:
            if pattern in content:
                self.ml_issues.append(MLIssue(
                    type="gpu_usage",
                    severity="info", 
                    location=file_path,
                    description=f"GPU usage detected: {pattern}",
                    recommendation="Ensure proper GPU memory management",
                    framework="general",
                    impact="performance"
                ))
        
        # Check for distributed training patterns
        distributed_patterns = [
            "DistributedDataParallel", "DataParallel", "distribute_strategy",
            "tf.distribute", "torch.distributed"
        ]
        for pattern in distributed_patterns:
            if pattern in content:
                self.ml_issues.append(MLIssue(
                    type="distributed_training",
                    severity="info",
                    location=file_path,
                    description=f"Distributed training pattern: {pattern}",
                    recommendation="Verify distributed training configuration",
                    framework="general", 
                    impact="scalability"
                ))
    
    def _check_ml_antipatterns(self, content: str, file_path: str) -> None:
        """Check for common ML anti-patterns"""
        
        for antipattern_type, patterns in ML_ANTIPATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    severity = self._get_antipattern_severity(antipattern_type)
                    
                    self.ml_issues.append(MLIssue(
                        type=antipattern_type,
                        severity=severity,
                        location=file_path,
                        description=f"Potential {antipattern_type} detected: {pattern}",
                        recommendation=self._get_antipattern_recommendation(antipattern_type),
                        framework="general",
                        impact=self._get_antipattern_impact(antipattern_type)
                    ))
    
    def _get_antipattern_severity(self, antipattern_type: str) -> str:
        """Get severity level for anti-pattern type"""
        severity_map = {
            "data_leakage": "critical",
            "shape_mismatch": "high",
            "memory_issues": "medium",
            "training_issues": "high"
        }
        return severity_map.get(antipattern_type, "medium")
    
    def _get_antipattern_recommendation(self, antipattern_type: str) -> str:
        """Get recommendation for anti-pattern type"""
        recommendations = {
            "data_leakage": "Ensure preprocessing is applied only to training data",
            "shape_mismatch": "Verify tensor dimensions before operations",
            "memory_issues": "Optimize GPU memory usage and tensor management",
            "training_issues": "Follow proper training loop patterns"
        }
        return recommendations.get(antipattern_type, "Review ML best practices")
    
    def _get_antipattern_impact(self, antipattern_type: str) -> str:
        """Get impact description for anti-pattern type"""
        impacts = {
            "data_leakage": "Invalid model performance estimates",
            "shape_mismatch": "Runtime errors and crashes",
            "memory_issues": "OOM errors and poor performance",
            "training_issues": "Incorrect model training"
        }
        return impacts.get(antipattern_type, "Potential quality issues")
    
    def _check_reproducibility(self, content: str, file_path: str) -> Dict[str, Any]:
        """Check for reproducibility best practices"""
        reproducibility = {
            "random_seed_set": False,
            "deterministic_algorithms": False,
            "version_pinning": False,
            "issues": []
        }
        
        # Check for random seed setting
        seed_patterns = [
            "random.seed", "np.random.seed", "torch.manual_seed",
            "tf.random.set_seed", "random_state="
        ]
        
        for pattern in seed_patterns:
            if pattern in content:
                reproducibility["random_seed_set"] = True
                break
        
        if not reproducibility["random_seed_set"]:
            self.ml_issues.append(MLIssue(
                type="reproducibility",
                severity="medium",
                location=file_path,
                description="No random seed setting detected",
                recommendation="Set random seeds for reproducible results",
                framework="general",
                impact="reproducibility"
            ))
        
        # Check for deterministic algorithms
        deterministic_patterns = [
            "torch.backends.cudnn.deterministic",
            "tf.config.experimental.enable_deterministic_ops"
        ]
        
        for pattern in deterministic_patterns:
            if pattern in content:
                reproducibility["deterministic_algorithms"] = True
                break
        
        # Check for version pinning in requirements
        if "==" in content or "requirements" in file_path.lower():
            reproducibility["version_pinning"] = True
        
        return reproducibility
    
    def _analyze_performance_patterns(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Analyze performance-related patterns"""
        performance_issues = []
        
        # Performance anti-patterns
        perf_antipatterns = {
            "loop_inefficiency": [
                "for.*in.*range.*len",  # for i in range(len(x))
                "for.*enumerate.*range"   # for i, x in enumerate(range(...))
            ],
            "memory_inefficiency": [
                "\.append.*loop",  # Appending in loops
                "list.*comprehension.*large"
            ],
            "gpu_inefficiency": [
                "\.cpu\(\).*\.cuda\(\)",  # CPU-GPU transfers
                "for.*\.to\(device\)"  # Moving to device in loop
            ]
        }
        
        for issue_type, patterns in perf_antipatterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    performance_issues.append({
                        "type": issue_type,
                        "pattern": pattern,
                        "file": file_path,
                        "severity": "medium"
                    })
        
        return performance_issues
    
    def _generate_core_report(self) -> Dict[str, Any]:
        """Generate core ML analysis report"""
        
        # Aggregate framework statistics
        all_frameworks = set()
        framework_files = {}
        
        for file_path, framework_data in self.detected_frameworks.items():
            for fw_info in framework_data.get("detected_frameworks", []):
                framework = fw_info["framework"]
                all_frameworks.add(framework)
                
                if framework not in framework_files:
                    framework_files[framework] = []
                framework_files[framework].append(file_path)
        
        # Issue statistics
        issues_by_type = defaultdict(int)
        issues_by_severity = defaultdict(int)
        
        for issue in self.ml_issues:
            issues_by_type[issue.type] += 1
            issues_by_severity[issue.severity] += 1
        
        return {
            "summary": {
                "frameworks_detected": list(all_frameworks),
                "total_ml_files": len(self.detected_frameworks),
                "total_issues": len(self.ml_issues),
                "critical_issues": issues_by_severity.get("critical", 0),
                "high_issues": issues_by_severity.get("high", 0),
                "medium_issues": issues_by_severity.get("medium", 0)
            },
            "framework_detection": {
                "detected_frameworks": list(all_frameworks),
                "framework_usage": framework_files,
                "framework_details": self.detected_frameworks
            },
            "issues": [
                {
                    "type": issue.type,
                    "severity": issue.severity,
                    "location": issue.location,
                    "description": issue.description,
                    "recommendation": issue.recommendation,
                    "framework": issue.framework,
                    "impact": issue.impact
                }
                for issue in self.ml_issues
            ],
            "issues_by_type": dict(issues_by_type),
            "issues_by_severity": dict(issues_by_severity),
            "recommendations": self._generate_core_recommendations()
        }
    
    def _generate_core_recommendations(self) -> List[Dict[str, str]]:
        """Generate core recommendations based on analysis"""
        recommendations = []
        
        # Framework recommendations
        if len(self.detected_frameworks) > 0:
            all_frameworks = set()
            for framework_data in self.detected_frameworks.values():
                for fw_info in framework_data.get("detected_frameworks", []):
                    all_frameworks.add(fw_info["framework"])
            
            if len(all_frameworks) > 3:
                recommendations.append({
                    "category": "Framework Management",
                    "priority": "medium",
                    "recommendation": "Consider reducing framework dependencies",
                    "impact": "Simplified development and deployment"
                })
        
        # Issue-based recommendations
        critical_issues = [i for i in self.ml_issues if i.severity == "critical"]
        if critical_issues:
            recommendations.append({
                "category": "Critical Issues",
                "priority": "high", 
                "recommendation": "Address critical ML issues immediately",
                "impact": "Prevent model failures and data leakage"
            })
        
        high_issues = [i for i in self.ml_issues if i.severity == "high"]
        if high_issues:
            recommendations.append({
                "category": "High Priority Issues",
                "priority": "high",
                "recommendation": "Fix high-severity ML issues",
                "impact": "Improve model reliability and performance"
            })
        
        return recommendations