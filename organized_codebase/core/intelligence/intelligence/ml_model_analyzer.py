from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor
"""
ML Model Analyzer

Handles model architecture analysis, training loops, and model-specific patterns.
Split from original ml_code_analysis.py - Model Architecture & Training sections.
"""

import ast
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
import logging

from ...base import BaseAnalyzer
from ._shared_utils import MLIssue, identify_model_patterns, extract_hyperparameters


class MLModelAnalyzer(BaseAnalyzer):
    """
    Specialized analyzer for ML model architecture, training loops, and model patterns
    """
    
    def __init__(self):
        super().__init__()
        self.model_issues = []
        self.model_architectures = []
        self.training_patterns = []
        self.hyperparameters = []
        
    def analyze(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform model-focused analysis
        """
        if file_path:
            self._analyze_file(Path(file_path))
        else:
            self._analyze_directory()
            
        return self._generate_model_report()
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single file for model patterns"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Analyze model architectures
            architectures = identify_model_patterns(tree)
            self.model_architectures.extend(architectures)
            
            # Extract hyperparameters
            hyperparams = extract_hyperparameters(tree)
            self.hyperparameters.extend(hyperparams)
            
            # Analyze training loops
            self._analyze_training_loops(tree, content, str(file_path))
            
            # Check model serialization
            self._analyze_model_serialization(tree, content, str(file_path))
            
            # Analyze model evaluation
            self._analyze_model_evaluation(tree, content, str(file_path))
            
            # Check for model optimization patterns
            self._analyze_model_optimization(tree, content, str(file_path))
            
        except Exception as e:
            logging.error(f"Error analyzing model in {file_path}: {e}")
    
    def _analyze_training_loops(self, tree: ast.AST, content: str, file_path: str) -> None:
        """Analyze training loop patterns and issues"""
        
        # Training loop patterns
        training_patterns = {
            "basic_loop": ["for epoch in", "for batch in", "for step in"],
            "optimizer_usage": ["optimizer.zero_grad", "optimizer.step", "loss.backward"],
            "model_modes": ["model.train()", "model.SafeCodeExecutor.safe_SafeCodeExecutor.safe_eval()", "torch.no_grad()"],
            "loss_computation": ["criterion(", "loss_fn(", "F.cross_entropy", "F.mse_loss"]
        }
        
        for pattern_type, patterns in training_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    self.training_patterns.append({
                        "type": pattern_type,
                        "pattern": pattern,
                        "file": file_path,
                        "count": content.count(pattern)
                    })
        
        # Check for training loop issues
        self._check_training_issues(content, file_path)
        
        # Check for proper gradient handling
        self._check_gradient_handling(content, file_path)
        
        # Check for learning rate scheduling
        self._check_lr_scheduling(content, file_path)
    
    def _check_training_issues(self, content: str, file_path: str) -> None:
        """Check for common training loop issues"""
        
        # Missing optimizer.zero_grad()
        if "loss.backward()" in content and "optimizer.zero_grad()" not in content:
            self.model_issues.append(MLIssue(
                type="missing_zero_grad",
                severity="high",
                location=file_path,
                description="loss.backward() without optimizer.zero_grad()",
                recommendation="Add optimizer.zero_grad() before loss computation",
                framework="pytorch",
                impact="incorrect_gradients"
            ))
        
        # Missing model.train()/model.SafeCodeExecutor.safe_SafeCodeExecutor.safe_eval()
        has_training_loop = any(pattern in content for pattern in ["for epoch", "for batch"])
        has_mode_setting = any(pattern in content for pattern in ["model.train()", "model.SafeCodeExecutor.safe_SafeCodeExecutor.safe_eval()"])
        
        if has_training_loop and not has_mode_setting:
            self.model_issues.append(MLIssue(
                type="missing_model_mode",
                severity="medium",
                location=file_path,
                description="Training loop without model.train()/model.SafeCodeExecutor.safe_SafeCodeExecutor.safe_eval()",
                recommendation="Set model mode with model.train() and model.SafeCodeExecutor.safe_SafeCodeExecutor.safe_eval()",
                framework="pytorch",
                impact="incorrect_behavior"
            ))
        
        # Double backward calls
        if content.count("backward()") > content.count("zero_grad()") + 1:
            self.model_issues.append(MLIssue(
                type="double_backward",
                severity="high",
                location=file_path,
                description="Multiple backward() calls without proper gradient clearing",
                recommendation="Ensure one backward() per optimization step",
                framework="pytorch",
                impact="gradient_accumulation_error"
            ))
        
        # Loss not moving to correct device
        if ".cuda()" in content and "loss" in content:
            if not re.search(r'loss.*\.to\(', content) and not re.search(r'criterion.*\.to\(', content):
                self.model_issues.append(MLIssue(
                    type="loss_device_mismatch",
                    severity="medium",
                    location=file_path,
                    description="GPU usage detected but loss function device not specified",
                    recommendation="Move loss function to appropriate device",
                    framework="pytorch",
                    impact="device_error"
                ))
    
    def _check_gradient_handling(self, content: str, file_path: str) -> None:
        """Check gradient handling patterns"""
        
        # Gradient clipping
        if "torch.nn.utils.clip_grad" in content:
            self.training_patterns.append({
                "type": "gradient_clipping",
                "pattern": "gradient_clipping",
                "file": file_path,
                "count": content.count("clip_grad")
            })
        
        # Gradient accumulation
        if "accumulate" in content.lower() and "gradient" in content.lower():
            self.training_patterns.append({
                "type": "gradient_accumulation",
                "pattern": "gradient_accumulation",
                "file": file_path,
                "count": 1
            })
        
        # Manual gradient computation
        if "autograd.grad" in content:
            self.model_issues.append(MLIssue(
                type="manual_gradients",
                severity="low",
                location=file_path,
                description="Manual gradient computation detected",
                recommendation="Ensure manual gradients are necessary",
                framework="pytorch",
                impact="complexity"
            ))
    
    def _check_lr_scheduling(self, content: str, file_path: str) -> None:
        """Check learning rate scheduling patterns"""
        
        lr_schedulers = [
            "StepLR", "ExponentialLR", "CosineAnnealingLR", 
            "ReduceLROnPlateau", "CyclicLR", "OneCycleLR"
        ]
        
        for scheduler in lr_schedulers:
            if scheduler in content:
                self.training_patterns.append({
                    "type": "lr_scheduling",
                    "pattern": scheduler,
                    "file": file_path,
                    "count": content.count(scheduler)
                })
        
        # Check if training loop but no LR scheduling
        has_training = "optimizer.step()" in content
        has_scheduling = any(scheduler in content for scheduler in lr_schedulers)
        
        if has_training and not has_scheduling:
            self.model_issues.append(MLIssue(
                type="no_lr_scheduling",
                severity="low",
                location=file_path,
                description="Training without learning rate scheduling",
                recommendation="Consider adding learning rate scheduling",
                framework="general",
                impact="suboptimal_training"
            ))
    
    def _analyze_model_serialization(self, tree: ast.AST, content: str, file_path: str) -> None:
        """Analyze model serialization patterns"""
        
        # PyTorch serialization
        torch_save_patterns = ["torch.save", "torch.load", "state_dict", "load_state_dict"]
        
        for pattern in torch_save_patterns:
            if pattern in content:
                self.training_patterns.append({
                    "type": "serialization",
                    "pattern": pattern,
                    "file": file_path,
                    "count": content.count(pattern)
                })
        
        # TensorFlow serialization
        tf_save_patterns = ["model.save", "tf.saved_model", "model.save_weights", "model.load_weights"]
        
        for pattern in tf_save_patterns:
            if pattern in content:
                self.training_patterns.append({
                    "type": "serialization",
                    "pattern": pattern,
                    "file": file_path,
                    "count": content.count(pattern)
                })
        
        # Check for serialization issues
        if "torch.save" in content:
            # Check if saving entire model vs state_dict
            if content.count("torch.save") > content.count("state_dict"):
                self.model_issues.append(MLIssue(
                    type="inefficient_serialization",
                    severity="low",
                    location=file_path,
                    description="Saving entire model instead of state_dict",
                    recommendation="Use model.state_dict() for better compatibility",
                    framework="pytorch",
                    impact="compatibility"
                ))
    
    def _analyze_model_evaluation(self, tree: ast.AST, content: str, file_path: str) -> None:
        """Analyze model evaluation patterns"""
        
        # Evaluation metrics
        eval_metrics = [
            "accuracy", "precision", "recall", "f1_score", "roc_auc",
            "mean_squared_error", "mean_absolute_error", "r2_score"
        ]
        
        detected_metrics = []
        for metric in eval_metrics:
            if metric in content:
                detected_metrics.append(metric)
        
        if detected_metrics:
            self.training_patterns.append({
                "type": "evaluation_metrics",
                "pattern": ", ".join(detected_metrics),
                "file": file_path,
                "count": len(detected_metrics)
            })
        
        # Check for proper evaluation mode
        if "model.SafeCodeExecutor.safe_SafeCodeExecutor.safe_eval()" in content or "torch.no_grad()" in content:
            self.training_patterns.append({
                "type": "proper_evaluation",
                "pattern": "evaluation_mode",
                "file": file_path,
                "count": 1
            })
        
        # Validation during training
        if "validation" in content.lower() and ("for epoch" in content or "for batch" in content):
            self.training_patterns.append({
                "type": "validation_during_training",
                "pattern": "validation_loop",
                "file": file_path,
                "count": 1
            })
        
        # Check for evaluation issues
        has_evaluation = any(metric in content for metric in eval_metrics)
        has_prediction = "predict" in content or "inference" in content
        
        if has_prediction and not has_evaluation:
            self.model_issues.append(MLIssue(
                type="missing_evaluation",
                severity="medium",
                location=file_path,
                description="Model prediction without evaluation metrics",
                recommendation="Add evaluation metrics to assess model performance",
                framework="general",
                impact="unknown_performance"
            ))
    
    def _analyze_model_optimization(self, tree: ast.AST, content: str, file_path: str) -> None:
        """Analyze model optimization patterns"""
        
        # Optimization techniques
        opt_techniques = {
            "mixed_precision": ["amp.autocast", "GradScaler", "torch.cuda.amp"],
            "data_parallel": ["DataParallel", "DistributedDataParallel", "torch.nn.parallel"],
            "gradient_checkpointing": ["checkpoint", "gradient_checkpointing"],
            "model_compression": ["prune", "quantize", "distill"],
            "early_stopping": ["early_stopping", "EarlyStopping"],
            "batch_optimization": ["DataLoader", "batch_size", "num_workers"]
        }
        
        for tech_type, patterns in opt_techniques.items():
            for pattern in patterns:
                if pattern in content:
                    self.training_patterns.append({
                        "type": f"optimization_{tech_type}",
                        "pattern": pattern,
                        "file": file_path,
                        "count": content.count(pattern)
                    })
        
        # Check for optimization opportunities
        self._check_optimization_opportunities(content, file_path)
    
    def _check_optimization_opportunities(self, content: str, file_path: str) -> None:
        """Check for optimization opportunities"""
        
        # Large batch sizes without optimization
        if "batch_size" in content:
            batch_matches = re.findall(r'batch_size\s*=\s*(\d+)', content)
            for batch_size in batch_matches:
                if int(batch_size) > 64:
                    # Check if mixed precision or other optimizations are used
                    has_optimization = any(opt in content for opt in 
                                         ["amp", "DataParallel", "gradient_checkpointing"])
                    
                    if not has_optimization:
                        self.model_issues.append(MLIssue(
                            type="large_batch_no_optimization",
                            severity="low",
                            location=file_path,
                            description=f"Large batch size ({batch_size}) without optimization",
                            recommendation="Consider mixed precision or gradient checkpointing",
                            framework="general",
                            impact="memory_usage"
                        ))
        
        # GPU usage without mixed precision
        if ".cuda()" in content and "amp" not in content:
            self.model_issues.append(MLIssue(
                type="gpu_no_mixed_precision",
                severity="low",
                location=file_path,
                description="GPU usage without mixed precision training",
                recommendation="Consider using mixed precision for faster training",
                framework="pytorch",
                impact="performance"
            ))
        
        # Training without validation
        has_training = "optimizer.step()" in content
        has_validation = "validation" in content.lower() or "val_" in content
        
        if has_training and not has_validation:
            self.model_issues.append(MLIssue(
                type="no_validation",
                severity="medium",
                location=file_path,
                description="Training without validation monitoring",
                recommendation="Add validation loop to monitor overfitting",
                framework="general",
                impact="overfitting_risk"
            ))
    
    def _generate_model_report(self) -> Dict[str, Any]:
        """Generate comprehensive model analysis report"""
        
        # Calculate statistics
        total_issues = len(self.model_issues)
        total_architectures = len(self.model_architectures)
        total_patterns = len(self.training_patterns)
        total_hyperparams = len(self.hyperparameters)
        
        # Group issues by type and severity
        issues_by_type = defaultdict(int)
        issues_by_severity = defaultdict(int)
        
        for issue in self.model_issues:
            issues_by_type[issue.type] += 1
            issues_by_severity[issue.severity] += 1
        
        # Group patterns by type
        patterns_by_type = defaultdict(list)
        for pattern in self.training_patterns:
            patterns_by_type[pattern["type"]].append(pattern)
        
        # Group architectures by framework
        archs_by_framework = defaultdict(list)
        for arch in self.model_architectures:
            archs_by_framework[arch["framework"]].append(arch)
        
        # Analyze hyperparameters
        hyperparam_analysis = self._analyze_hyperparameter_patterns()
        
        return {
            "summary": {
                "total_model_issues": total_issues,
                "total_architectures": total_architectures,
                "total_training_patterns": total_patterns,
                "total_hyperparameters": total_hyperparams,
                "high_severity_issues": issues_by_severity.get("high", 0),
                "medium_severity_issues": issues_by_severity.get("medium", 0),
                "files_analyzed": len(set(issue.location for issue in self.model_issues))
            },
            "model_issues": [
                {
                    "type": issue.type,
                    "severity": issue.severity,
                    "location": issue.location,
                    "description": issue.description,
                    "recommendation": issue.recommendation,
                    "framework": issue.framework,
                    "impact": issue.impact
                }
                for issue in self.model_issues
            ],
            "model_architectures": {
                "by_framework": dict(archs_by_framework),
                "all_architectures": self.model_architectures
            },
            "training_patterns": {
                "by_type": dict(patterns_by_type),
                "all_patterns": self.training_patterns
            },
            "hyperparameter_analysis": hyperparam_analysis,
            "issues_by_type": dict(issues_by_type),
            "issues_by_severity": dict(issues_by_severity),
            "recommendations": self._generate_model_recommendations()
        }
    
    def _analyze_hyperparameter_patterns(self) -> Dict[str, Any]:
        """Analyze hyperparameter patterns and values"""
        
        # Group hyperparameters by type
        param_types = defaultdict(list)
        param_values = defaultdict(list)
        
        for param in self.hyperparameters:
            param_name = param["name"].lower()
            
            # Categorize parameters
            if "lr" in param_name or "learning_rate" in param_name:
                param_types["learning_rate"].append(param)
            elif "batch" in param_name:
                param_types["batch_size"].append(param)
            elif "epoch" in param_name:
                param_types["epochs"].append(param)
            elif "dropout" in param_name:
                param_types["regularization"].append(param)
            elif any(opt in param_name for opt in ["momentum", "beta", "weight_decay"]):
                param_types["optimizer"].append(param)
            else:
                param_types["other"].append(param)
            
            # Collect values for analysis
            if param["type"] in ["int", "float"]:
                param_values[param_name].append(param["value"])
        
        # Analyze value distributions
        value_analysis = {}
        for param_name, values in param_values.items():
            if values:
                value_analysis[param_name] = {
                    "count": len(values),
                    "unique_values": len(set(values)),
                    "min": min(values) if all(isinstance(v, (int, float)) for v in values) else None,
                    "max": max(values) if all(isinstance(v, (int, float)) for v in values) else None,
                    "values": list(set(values))
                }
        
        return {
            "by_type": dict(param_types),
            "value_analysis": value_analysis,
            "total_parameters": len(self.hyperparameters),
            "hardcoded_parameters": len([p for p in self.hyperparameters if p["type"] != "expression"])
        }
    
    def _generate_model_recommendations(self) -> List[Dict[str, str]]:
        """Generate model-specific recommendations"""
        recommendations = []
        
        # Training issues
        training_issues = [i for i in self.model_issues if "training" in i.type or "grad" in i.type]
        if training_issues:
            recommendations.append({
                "category": "Training Loop",
                "priority": "high",
                "recommendation": "Fix training loop issues",
                "impact": "Correct model training and convergence"
            })
        
        # Optimization opportunities
        opt_issues = [i for i in self.model_issues if "optimization" in i.type or "performance" in i.impact]
        if opt_issues:
            recommendations.append({
                "category": "Performance Optimization",
                "priority": "medium",
                "recommendation": "Implement performance optimizations",
                "impact": "Faster training and inference"
            })
        
        # Evaluation improvements
        eval_issues = [i for i in self.model_issues if "evaluation" in i.type or "validation" in i.type]
        if eval_issues:
            recommendations.append({
                "category": "Model Evaluation",
                "priority": "medium",
                "recommendation": "Improve model evaluation practices",
                "impact": "Better understanding of model performance"
            })
        
        # Serialization issues
        serial_issues = [i for i in self.model_issues if "serialization" in i.type]
        if serial_issues:
            recommendations.append({
                "category": "Model Serialization",
                "priority": "low",
                "recommendation": "Optimize model saving and loading",
                "impact": "Better model compatibility and deployment"
            })
        
        return recommendations