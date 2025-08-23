#!/usr/bin/env python3
"""
🏗️ MODULE: Pattern Analyzer - Extracted from Enhanced Intelligence Linkage
==========================================================================

📋 PURPOSE:
    Comprehensive design pattern detection engine for identifying design patterns,
    anti-patterns, architectural patterns, and pattern-based clustering analysis.

🎯 CORE FUNCTIONALITY:
    • Design pattern detection (Singleton, Factory, Observer, etc.)
    • Anti-pattern identification (God Class, Long Parameter List, etc.)
    • Architectural pattern recognition (MVC, Repository, Service Layer, etc.)
    • Pattern-based clustering and density analysis

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 [2025-08-23] | Agent Z | 🔧 STEELCLAD
   └─ Goal: Extract pattern analysis from enhanced_intelligence_linkage.py
   └─ Changes: Modularized pattern analysis with ~140 lines of focused functionality
   └─ Impact: Reduces main intelligence linkage size while maintaining pattern detection

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent Z (STEELCLAD extraction)
🔧 Language: Python
📦 Dependencies: re, pathlib
🎯 Integration Points: EnhancedLinkageAnalyzer class
⚡ Performance Notes: Optimized for large-scale pattern detection
🔒 Security Notes: Safe pattern matching with comprehensive detection
"""

import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

class PatternAnalyzer:
    """Comprehensive design pattern detection and analysis engine."""
    
    def __init__(self):
        self.design_patterns = self._load_design_patterns()
        self.anti_patterns = self._load_anti_patterns()
        self.architectural_patterns = self._load_architectural_patterns()
        self.pattern_weights = self._load_pattern_weights()
    
    def _load_design_patterns(self) -> Dict[str, List[str]]:
        """Load design pattern detection patterns."""
        return {
            "singleton": [
                r"_instance.*=.*None", r"__new__.*cls\)", r"class.*Singleton",
                r"if.*_instance.*is.*None", r"_instance.*=.*cls\("
            ],
            "factory": [
                r"create_", r"make_", r"build_", r"get_instance",
                r"factory", r"Factory", r"def.*create.*\(.*type"
            ],
            "observer": [
                r"notify", r"subscribe", r"observer", r"Observer",
                r"add_observer", r"remove_observer", r"update.*observer"
            ],
            "decorator": [
                r"@\w+", r"def wrapper", r"def.*decorator",
                r"functools\.wraps", r"@property", r"@staticmethod"
            ],
            "strategy": [
                r"strategy", r"Strategy", r"algorithm", r"Algorithm",
                r"def.*execute", r"interface.*strategy"
            ],
            "command": [
                r"execute", r"command", r"Command", r"def.*execute\(",
                r"undo", r"redo", r"invoke"
            ],
            "adapter": [
                r"adapter", r"Adapter", r"adapt", r"convert",
                r"interface.*adapt", r"legacy.*convert"
            ],
            "facade": [
                r"facade", r"Facade", r"unified.*interface",
                r"simplified.*access", r"wrapper.*complex"
            ],
            "proxy": [
                r"proxy", r"Proxy", r"surrogate", r"placeholder",
                r"virtual.*proxy", r"remote.*proxy"
            ],
            "builder": [
                r"builder", r"Builder", r"build", r"construct",
                r"step.*by.*step", r"fluent.*interface"
            ],
            "prototype": [
                r"clone", r"copy", r"prototype", r"Prototype",
                r"deep.*copy", r"shallow.*copy"
            ],
            "template_method": [
                r"template", r"Template", r"abstract.*method",
                r"def.*step", r"skeleton.*algorithm"
            ]
        }
    
    def _load_anti_patterns(self) -> Dict[str, List[str]]:
        """Load anti-pattern detection patterns."""
        return {
            "god_class": [
                r"class.*{.*\n.*{.*\n.*{.*\n.*{.*\n.*{",  # Very long class indication
                r"def.*\n.*def.*\n.*def.*\n.*def.*\n.*def.*\n.*def"  # Many methods
            ],
            "long_parameter_list": [
                r"def \w+\([^)]{50,}", r"def \w+\([^)]*,.*,.*,.*,.*,",
                r"def.*\(.*\w+.*,.*\w+.*,.*\w+.*,.*\w+.*,.*\w+.*,"
            ],
            "dead_code": [
                r"#.*unused", r"#.*dead", r"#.*remove", r"#.*delete",
                r"#.*deprecated", r"pass\s*#.*dead"
            ],
            "magic_numbers": [
                r"\b\d{2,}\b", r"\b\d+\.\d{3,}\b"
            ],
            "deep_nesting": [
                r"\n\s{20,}", r"\n\s{24,}", r"\n\s{28,}"
            ],
            "duplicate_code": [
                r"(.{30,})\n.*\1", r"def \w+.*\n.*def \w+.*\n.*def \w+"
            ],
            "large_interface": [
                r"def.*\(\).*:\s*pass.*def.*\(\).*:\s*pass.*def.*\(\).*:\s*pass"
            ],
            "lazy_class": [
                r"class \w+.*:\s*pass", r"class \w+.*:\s*def __init__"
            ],
            "inappropriate_intimacy": [
                r"\._\w+", r"friend", r"access.*private"
            ],
            "feature_envy": [
                r"other\.\w+.*other\.\w+.*other\.\w+"
            ]
        }
    
    def _load_architectural_patterns(self) -> Dict[str, List[str]]:
        """Load architectural pattern detection patterns."""
        return {
            "mvc": [
                r"model", r"view", r"controller", r"Model", r"View", r"Controller",
                r"render", r"template", r"business.*logic"
            ],
            "mvp": [
                r"presenter", r"Presenter", r"view.*interface", r"passive.*view"
            ],
            "mvvm": [
                r"viewmodel", r"ViewModel", r"binding", r"observable"
            ],
            "repository": [
                r"repository", r"Repository", r"data.*access", r"persistence",
                r"find.*by", r"save", r"delete.*by"
            ],
            "service_layer": [
                r"service", r"Service", r"business.*service", r"application.*service",
                r"use.*case", r"workflow"
            ],
            "data_access_object": [
                r"dao", r"DAO", r"data.*access.*object", r"crud"
            ],
            "layered_architecture": [
                r"layer", r"Layer", r"tier", r"Tier", r"presentation.*layer",
                r"business.*layer", r"data.*layer"
            ],
            "microservices": [
                r"microservice", r"service.*mesh", r"api.*gateway",
                r"distributed.*system", r"service.*discovery"
            ],
            "event_driven": [
                r"event", r"Event", r"publish", r"subscribe", r"message.*queue",
                r"event.*bus", r"handler.*event"
            ],
            "pipe_and_filter": [
                r"pipeline", r"Pipeline", r"filter", r"Filter", r"process.*chain",
                r"transform.*chain"
            ]
        }
    
    def _load_pattern_weights(self) -> Dict[str, float]:
        """Load pattern scoring weights."""
        return {
            # Design patterns (positive weight)
            "singleton": 0.8,
            "factory": 1.0,
            "observer": 1.2,
            "decorator": 0.9,
            "strategy": 1.1,
            "command": 1.0,
            "adapter": 0.9,
            "facade": 1.0,
            "proxy": 0.8,
            "builder": 1.1,
            "prototype": 0.7,
            "template_method": 1.0,
            
            # Anti-patterns (negative weight)
            "god_class": -2.0,
            "long_parameter_list": -1.5,
            "dead_code": -1.8,
            "magic_numbers": -0.5,
            "deep_nesting": -1.2,
            "duplicate_code": -1.6,
            "large_interface": -1.3,
            "lazy_class": -0.8,
            "inappropriate_intimacy": -1.0,
            "feature_envy": -0.9,
            
            # Architectural patterns (positive weight)
            "mvc": 1.5,
            "repository": 1.3,
            "service_layer": 1.4,
            "event_driven": 1.2,
            "microservices": 1.6
        }
    
    def analyze_pattern_dimensions(self, python_files: List[Path], base_dir: str) -> Dict[str, Any]:
        """Comprehensive pattern analysis of codebase."""
        pattern_results = {
            "design_patterns": {},
            "anti_patterns": {},
            "architectural_patterns": {},
            "pattern_based_clustering": {},
            "pattern_metrics": {}
        }
        
        base_path = Path(base_dir)
        total_files = len(python_files)
        
        print(f"Pattern Analysis: Processing {total_files} files...")
        
        for i, py_file in enumerate(python_files):
            if i % 100 == 0:  # Progress update every 100 files
                print(f"  Pattern analysis progress: {i}/{total_files} ({i/total_files*100:.1f}%)")
            
            try:
                relative_path = str(py_file.relative_to(base_path))
                
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Design pattern detection
                patterns = self.detect_design_patterns(content)
                pattern_results["design_patterns"][relative_path] = patterns
                
                # Anti-pattern detection
                anti_patterns = self.detect_anti_patterns(content)
                pattern_results["anti_patterns"][relative_path] = anti_patterns
                
                # Architectural pattern detection
                arch_patterns = self.detect_architectural_patterns(content)
                pattern_results["architectural_patterns"][relative_path] = arch_patterns
                
            except Exception as e:
                print(f"  Error processing {py_file}: {e}")
                continue
        
        # Pattern-based clustering
        pattern_results["pattern_based_clustering"] = self.cluster_by_patterns(
            pattern_results["design_patterns"],
            pattern_results["architectural_patterns"]
        )
        
        # Calculate pattern metrics
        pattern_results["pattern_metrics"] = self.calculate_pattern_metrics(
            pattern_results
        )
        
        print("Pattern Analysis: Complete!")
        return pattern_results
    
    def detect_design_patterns(self, content: str) -> Dict[str, Any]:
        """Detect design patterns in code."""
        detected_patterns = {}
        pattern_scores = {}
        pattern_locations = {}
        
        for pattern_name, patterns in self.design_patterns.items():
            count = 0
            locations = []
            
            for pattern in patterns:
                try:
                    matches = list(re.finditer(pattern, content, re.IGNORECASE))
                    count += len(matches)
                    
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1
                        locations.append({
                            "line": line_number,
                            "text": match.group(0)[:50] + "..." if len(match.group(0)) > 50 else match.group(0),
                            "pattern": pattern
                        })
                except re.error:
                    continue
            
            pattern_scores[pattern_name] = count
            if count > 0:
                detected_patterns[pattern_name] = count
                pattern_locations[pattern_name] = locations[:3]  # Limit to first 3 locations
        
        pattern_density = len(detected_patterns) / len(self.design_patterns) if self.design_patterns else 0
        
        return {
            "detected_patterns": list(detected_patterns.keys()),
            "pattern_scores": pattern_scores,
            "pattern_locations": pattern_locations,
            "pattern_density": pattern_density,
            "design_score": self._calculate_design_score(detected_patterns)
        }
    
    def detect_anti_patterns(self, content: str) -> Dict[str, Any]:
        """Detect anti-patterns in code."""
        detected_anti_patterns = {}
        anti_pattern_scores = {}
        anti_pattern_locations = {}
        
        for pattern_name, patterns in self.anti_patterns.items():
            count = 0
            locations = []
            
            # Special handling for certain anti-patterns
            if pattern_name == "god_class":
                lines = len(content.splitlines())
                if lines > 1000:
                    count = 1
                    locations.append({"line": 1, "text": f"Class has {lines} lines", "severity": "high"})
            elif pattern_name == "magic_numbers":
                # More sophisticated magic number detection
                numbers = re.findall(r'\b(?<![\d.])\d{2,}(?![\d.])\b', content)
                # Filter out common non-magic numbers
                magic_numbers = [n for n in numbers if int(n) not in [0, 1, 2, 10, 100, 1000]]
                count = len(magic_numbers)
                if count > 0:
                    locations = [{"line": 0, "text": f"Found {count} potential magic numbers", "numbers": magic_numbers[:5]}]
            else:
                for pattern in patterns:
                    try:
                        matches = list(re.finditer(pattern, content, re.IGNORECASE | re.DOTALL))
                        count += len(matches)
                        
                        for match in matches:
                            line_number = content[:match.start()].count('\n') + 1
                            locations.append({
                                "line": line_number,
                                "text": match.group(0)[:50] + "..." if len(match.group(0)) > 50 else match.group(0),
                                "pattern": pattern
                            })
                    except re.error:
                        continue
            
            anti_pattern_scores[pattern_name] = count
            if count > 0:
                detected_anti_patterns[pattern_name] = count
                anti_pattern_locations[pattern_name] = locations[:3]  # Limit to first 3 locations
        
        severity = self._calculate_anti_pattern_severity(detected_anti_patterns)
        
        return {
            "detected_anti_patterns": list(detected_anti_patterns.keys()),
            "anti_pattern_scores": anti_pattern_scores,
            "anti_pattern_locations": anti_pattern_locations,
            "anti_pattern_severity": severity,
            "design_debt_score": self._calculate_design_debt_score(detected_anti_patterns)
        }
    
    def detect_architectural_patterns(self, content: str) -> Dict[str, Any]:
        """Detect architectural patterns in code."""
        detected_patterns = {}
        pattern_scores = {}
        pattern_confidence = {}
        
        for pattern_name, patterns in self.architectural_patterns.items():
            count = 0
            confidence_factors = []
            
            for pattern in patterns:
                try:
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    count += matches
                    if matches > 0:
                        confidence_factors.append(matches)
                except re.error:
                    continue
            
            pattern_scores[pattern_name] = count
            if count > 0:
                detected_patterns[pattern_name] = count
                # Calculate confidence based on multiple pattern matches
                confidence = min(1.0, sum(confidence_factors) / (len(patterns) * 2))
                pattern_confidence[pattern_name] = confidence
        
        architecture_score = self._calculate_architecture_score(detected_patterns)
        
        return {
            "detected_patterns": list(detected_patterns.keys()),
            "pattern_scores": pattern_scores,
            "pattern_confidence": pattern_confidence,
            "architecture_score": architecture_score,
            "dominant_architecture": max(detected_patterns, key=detected_patterns.get) if detected_patterns else None
        }
    
    def cluster_by_patterns(self, design_patterns: Dict[str, Dict], 
                          architectural_patterns: Dict[str, Dict]) -> Dict[str, Any]:
        """Cluster files by detected patterns."""
        clusters = {
            "design_pattern_clusters": defaultdict(list),
            "architectural_clusters": defaultdict(list),
            "pattern_combinations": defaultdict(list)
        }
        
        # Cluster by design patterns
        for file_path, pattern_data in design_patterns.items():
            detected = pattern_data.get("detected_patterns", [])
            for pattern in detected:
                clusters["design_pattern_clusters"][pattern].append(file_path)
        
        # Cluster by architectural patterns
        for file_path, pattern_data in architectural_patterns.items():
            detected = pattern_data.get("detected_patterns", [])
            for pattern in detected:
                clusters["architectural_clusters"][pattern].append(file_path)
        
        # Find pattern combinations
        for file_path in design_patterns.keys():
            design_detected = design_patterns.get(file_path, {}).get("detected_patterns", [])
            arch_detected = architectural_patterns.get(file_path, {}).get("detected_patterns", [])
            
            if design_detected and arch_detected:
                combination = f"{arch_detected[0]}_with_{design_detected[0]}"
                clusters["pattern_combinations"][combination].append(file_path)
        
        return {
            "design_pattern_clusters": dict(clusters["design_pattern_clusters"]),
            "architectural_clusters": dict(clusters["architectural_clusters"]),
            "pattern_combinations": dict(clusters["pattern_combinations"]),
            "clustering_summary": self._generate_clustering_summary(clusters)
        }
    
    def calculate_pattern_metrics(self, pattern_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive pattern metrics."""
        design_patterns = pattern_results.get("design_patterns", {})
        anti_patterns = pattern_results.get("anti_patterns", {})
        architectural_patterns = pattern_results.get("architectural_patterns", {})
        
        if not design_patterns:
            return {"error": "No pattern data available"}
        
        total_files = len(design_patterns)
        
        # Count patterns across all files
        total_design_patterns = sum(
            len(data.get("detected_patterns", [])) for data in design_patterns.values()
        )
        total_anti_patterns = sum(
            len(data.get("detected_anti_patterns", [])) for data in anti_patterns.values()
        )
        total_arch_patterns = sum(
            len(data.get("detected_patterns", [])) for data in architectural_patterns.values()
        )
        
        # Calculate averages
        avg_design_patterns = total_design_patterns / max(total_files, 1)
        avg_anti_patterns = total_anti_patterns / max(total_files, 1)
        avg_arch_patterns = total_arch_patterns / max(total_files, 1)
        
        # Pattern distribution
        design_distribution = defaultdict(int)
        anti_distribution = defaultdict(int)
        arch_distribution = defaultdict(int)
        
        for data in design_patterns.values():
            for pattern in data.get("detected_patterns", []):
                design_distribution[pattern] += 1
        
        for data in anti_patterns.values():
            for pattern in data.get("detected_anti_patterns", []):
                anti_distribution[pattern] += 1
        
        for data in architectural_patterns.values():
            for pattern in data.get("detected_patterns", []):
                arch_distribution[pattern] += 1
        
        # Quality score based on patterns
        pattern_quality_score = self._calculate_pattern_quality_score(
            total_design_patterns, total_anti_patterns, total_arch_patterns
        )
        
        return {
            "total_files_analyzed": total_files,
            "total_design_patterns": total_design_patterns,
            "total_anti_patterns": total_anti_patterns,
            "total_architectural_patterns": total_arch_patterns,
            "average_design_patterns_per_file": avg_design_patterns,
            "average_anti_patterns_per_file": avg_anti_patterns,
            "average_architectural_patterns_per_file": avg_arch_patterns,
            "design_pattern_distribution": dict(design_distribution),
            "anti_pattern_distribution": dict(anti_distribution),
            "architectural_pattern_distribution": dict(arch_distribution),
            "pattern_quality_score": pattern_quality_score,
            "most_common_design_pattern": max(design_distribution, key=design_distribution.get) if design_distribution else None,
            "most_common_anti_pattern": max(anti_distribution, key=anti_distribution.get) if anti_distribution else None,
            "pattern_health": "good" if pattern_quality_score > 0.6 else "needs_improvement"
        }
    
    def _calculate_design_score(self, detected_patterns: Dict[str, int]) -> float:
        """Calculate design pattern score."""
        score = 0.0
        for pattern_name, count in detected_patterns.items():
            weight = self.pattern_weights.get(pattern_name, 0.5)
            score += count * weight
        return min(10.0, score)  # Cap at 10
    
    def _calculate_anti_pattern_severity(self, detected_anti_patterns: Dict[str, int]) -> str:
        """Calculate anti-pattern severity level."""
        total_score = 0
        for pattern_name, count in detected_anti_patterns.items():
            weight = abs(self.pattern_weights.get(pattern_name, -1.0))
            total_score += count * weight
        
        if total_score <= 2:
            return "low"
        elif total_score <= 5:
            return "medium"
        elif total_score <= 10:
            return "high"
        else:
            return "critical"
    
    def _calculate_design_debt_score(self, detected_anti_patterns: Dict[str, int]) -> float:
        """Calculate design debt score."""
        debt_score = 0.0
        for pattern_name, count in detected_anti_patterns.items():
            weight = abs(self.pattern_weights.get(pattern_name, -1.0))
            debt_score += count * weight
        return debt_score
    
    def _calculate_architecture_score(self, detected_patterns: Dict[str, int]) -> float:
        """Calculate architectural pattern score."""
        score = 0.0
        for pattern_name, count in detected_patterns.items():
            weight = self.pattern_weights.get(pattern_name, 1.0)
            score += count * weight
        return min(15.0, score)  # Cap at 15
    
    def _generate_clustering_summary(self, clusters: Dict[str, defaultdict]) -> Dict[str, Any]:
        """Generate clustering summary."""
        return {
            "design_clusters_count": len(clusters["design_pattern_clusters"]),
            "architectural_clusters_count": len(clusters["architectural_clusters"]),
            "pattern_combinations_count": len(clusters["pattern_combinations"]),
            "largest_cluster_size": max(
                [len(files) for files in clusters["design_pattern_clusters"].values()] +
                [len(files) for files in clusters["architectural_clusters"].values()],
                default=0
            )
        }
    
    def _calculate_pattern_quality_score(self, design_patterns: int, 
                                       anti_patterns: int, arch_patterns: int) -> float:
        """Calculate overall pattern quality score."""
        positive_score = design_patterns * 0.5 + arch_patterns * 0.7
        negative_score = anti_patterns * 1.2
        
        net_score = positive_score - negative_score
        total_patterns = design_patterns + anti_patterns + arch_patterns
        
        if total_patterns == 0:
            return 0.5  # Neutral score
        
        # Normalize to 0-1 range
        normalized_score = (net_score / total_patterns) + 0.5
        return max(0.0, min(1.0, normalized_score))

def create_pattern_analyzer() -> PatternAnalyzer:
    """Factory function to create a configured pattern analyzer."""
    return PatternAnalyzer()