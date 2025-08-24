#!/usr/bin/env python3
"""
🏗️ MODULE: Semantic Analyzer - Extracted from Enhanced Intelligence Linkage
===========================================================================

📋 PURPOSE:
    AI-powered semantic analysis engine for developer intent classification,
    conceptual element extraction, and purpose-based linkage analysis.

🎯 CORE FUNCTIONALITY:
    • Developer intent classification using pattern-based analysis
    • Conceptual element extraction from code structure
    • Semantic clustering and purpose-based linkage
    • Multi-dimensional semantic relationship mapping

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 [2025-08-23] | Agent Z | 🔧 STEELCLAD
   └─ Goal: Extract semantic analysis from enhanced_intelligence_linkage.py
   └─ Changes: Modularized semantic analysis with ~200 lines of focused functionality
   └─ Impact: Reduces main intelligence linkage size while maintaining semantic analysis

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent Z (STEELCLAD extraction)
🔧 Language: Python
📦 Dependencies: ast, re, pathlib
🎯 Integration Points: EnhancedLinkageAnalyzer class
⚡ Performance Notes: Optimized for large-scale codebase analysis
🔒 Security Notes: Safe AST parsing with error handling
"""

import ast
import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

class SemanticAnalyzer:
    """AI-powered semantic analysis engine for code intent classification."""
    
    def __init__(self):
        self.semantic_patterns = self._load_semantic_patterns()
        self.intent_patterns = self._load_intent_patterns()
        self.concept_extractors = self._load_concept_extractors()
    
    def _load_semantic_patterns(self) -> Dict[str, List[str]]:
        """Load semantic patterns for analysis."""
        return {
            "high_level_abstractions": [
                r"class.*Manager", r"class.*Service", r"class.*Controller",
                r"class.*Handler", r"class.*Provider", r"class.*Factory"
            ],
            "data_structures": [
                r"class.*Model", r"class.*Entity", r"class.*DTO",
                r"@dataclass", r"typing\.", r"Dict\[", r"List\["
            ],
            "functional_patterns": [
                r"def.*_async", r"async def", r"await ", r"yield ",
                r"lambda ", r"map\(", r"filter\(", r"reduce\("
            ],
            "integration_patterns": [
                r"requests\.", r"httpx\.", r"aiohttp\.",
                r"@app\.", r"@router\.", r"@api\."
            ]
        }
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load developer intent classification patterns."""
        return {
            "data_processing": [
                r"def process_", r"def transform_", r"\.map\(", r"\.filter\(", 
                r"pandas", r"numpy", r"dataframe", r"series"
            ],
            "api_endpoint": [
                r"@app\.route", r"@router\.", r"FastAPI", r"Flask", 
                r"def get_", r"def post_", r"def put_", r"def delete_"
            ],
            "authentication": [
                r"login", r"auth", r"token", r"jwt", r"password", 
                r"@login_required", r"session", r"credential"
            ],
            "validation": [
                r"validate", r"schema", r"pydantic", r"marshmallow", 
                r"assert", r"check_", r"verify_"
            ],
            "configuration": [
                r"config", r"settings", r"environ", r"\.env", 
                r"yaml", r"json", r"toml", r"ini"
            ],
            "testing": [
                r"test_", r"pytest", r"unittest", r"mock", 
                r"assert", r"@patch", r"fixture"
            ],
            "utilities": [
                r"def helper_", r"def util_", r"def format_", 
                r"def convert_", r"def parse_", r"def serialize_"
            ],
            "orchestration": [
                r"workflow", r"pipeline", r"orchestrat", r"coordinate", 
                r"schedule", r"queue", r"task"
            ],
            "monitoring": [
                r"log", r"metric", r"monitor", r"alert", r"track", 
                r"observ", r"telemetry", r"instrument"
            ],
            "security": [
                r"encrypt", r"decrypt", r"hash", r"secure", r"vulnerab", 
                r"sanitiz", r"escap", r"permission"
            ],
            "integration": [
                r"integrate", r"connect", r"api", r"client", r"service", 
                r"webhook", r"callback", r"bridge"
            ],
            "analysis": [
                r"analyz", r"predict", r"model", r"algorithm", r"statistic", 
                r"machine_learning", r"ml_", r"ai_"
            ],
            "database": [
                r"query", r"sql", r"database", r"db_", r"orm", 
                r"migrate", r"schema", r"table"
            ],
            "ui_frontend": [
                r"render", r"template", r"html", r"css", r"javascript", 
                r"react", r"vue", r"angular"
            ],
            "background_processing": [
                r"celery", r"worker", r"job", r"background", r"async", 
                r"queue", r"schedule"
            ]
        }
    
    def _load_concept_extractors(self) -> Dict[str, callable]:
        """Load concept extraction functions."""
        return {
            "domain_entities": self._extract_domain_entities,
            "business_concepts": self._extract_business_concepts,
            "technical_concepts": self._extract_technical_concepts,
            "architectural_patterns": self._extract_architectural_patterns
        }
    
    def analyze_semantic_dimensions(self, python_files: List[Path], base_dir: str) -> Dict[str, Any]:
        """Comprehensive semantic analysis of codebase."""
        semantic_results = {
            "intent_classifications": {},
            "semantic_clusters": {},
            "conceptual_relationships": {},
            "purpose_based_linkage": {},
            "semantic_metrics": {}
        }
        
        base_path = Path(base_dir)
        total_files = len(python_files)
        
        print(f"Semantic Analysis: Processing {total_files} files...")
        
        for i, py_file in enumerate(python_files):
            if i % 100 == 0:  # Progress update every 100 files
                print(f"  Semantic analysis progress: {i}/{total_files} ({i/total_files*100:.1f}%)")
            
            try:
                relative_path = str(py_file.relative_to(base_path))
                
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Semantic intent classification
                intent = self.classify_developer_intent(content)
                semantic_results["intent_classifications"][relative_path] = intent
                
                # Conceptual relationship mapping
                concepts = self.extract_conceptual_elements(content)
                semantic_results["conceptual_relationships"][relative_path] = concepts
                
            except Exception as e:
                print(f"  Error processing {py_file}: {e}")
                continue
        
        # Build semantic clusters
        semantic_results["semantic_clusters"] = self.cluster_by_semantics(
            semantic_results["intent_classifications"]
        )
        
        # Purpose-based linkage
        semantic_results["purpose_based_linkage"] = self.build_purpose_linkage(
            semantic_results["intent_classifications"]
        )
        
        # Calculate semantic metrics
        semantic_results["semantic_metrics"] = self.calculate_semantic_metrics(
            semantic_results
        )
        
        print("Semantic Analysis: Complete!")
        return semantic_results
    
    def classify_developer_intent(self, content: str) -> Dict[str, Any]:
        """Classify the developer's intent using semantic analysis."""
        intents = {}
        
        # Initialize all intent scores
        for intent_name in self.intent_patterns.keys():
            intents[intent_name] = 0
        
        # Pattern-based intent detection
        for intent, intent_patterns in self.intent_patterns.items():
            for pattern in intent_patterns:
                try:
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    intents[intent] += matches
                except re.error:
                    continue
        
        # Calculate total matches and determine dominant intent
        total_matches = max(sum(intents.values()), 1)
        dominant_intent = max(intents, key=intents.get)
        confidence = intents[dominant_intent] / total_matches
        
        # Calculate normalized scores
        normalized_intents = {k: v / total_matches for k, v in intents.items()}
        
        return {
            "primary_intent": dominant_intent,
            "confidence": confidence,
            "all_intents": normalized_intents,
            "total_intent_signals": total_matches,
            "secondary_intents": sorted(
                [(k, v) for k, v in normalized_intents.items() if k != dominant_intent],
                key=lambda x: x[1],
                reverse=True
            )[:3]  # Top 3 secondary intents
        }
    
    def extract_conceptual_elements(self, content: str) -> Dict[str, List[str]]:
        """Extract conceptual elements from code."""
        concepts = {
            "domain_entities": [],
            "business_concepts": [],
            "technical_concepts": [],
            "architectural_patterns": []
        }
        
        # Extract concepts using AST analysis
        try:
            tree = ast.parse(content)
            
            for concept_type, extractor in self.concept_extractors.items():
                concepts[concept_type] = extractor(tree, content)
                
        except SyntaxError:
            # If AST parsing fails, use regex-based extraction
            concepts = self._fallback_concept_extraction(content)
        except Exception as e:
            print(f"Error in concept extraction: {e}")
        
        return concepts
    
    def _extract_domain_entities(self, tree: ast.AST, content: str) -> List[str]:
        """Extract domain entities from AST."""
        entities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Look for domain-specific class naming patterns
                if any(suffix in node.name for suffix in ['Model', 'Entity', 'Data', 'Record']):
                    entities.append(node.name)
                elif node.name[0].isupper() and len(node.name) > 3:  # PascalCase classes
                    entities.append(node.name)
        
        return list(set(entities))
    
    def _extract_business_concepts(self, tree: ast.AST, content: str) -> List[str]:
        """Extract business concepts from AST."""
        business_concepts = []
        
        business_keywords = [
            'business', 'process', 'handle', 'manage', 'service', 'workflow',
            'calculate', 'validate', 'approve', 'submit', 'review', 'analyze'
        ]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name_lower = node.name.lower()
                if any(keyword in func_name_lower for keyword in business_keywords):
                    business_concepts.append(node.name)
            elif isinstance(node, ast.ClassDef):
                class_name_lower = node.name.lower()
                if any(keyword in class_name_lower for keyword in business_keywords):
                    business_concepts.append(node.name)
        
        return list(set(business_concepts))
    
    def _extract_technical_concepts(self, tree: ast.AST, content: str) -> List[str]:
        """Extract technical concepts from AST."""
        technical_concepts = []
        
        technical_keywords = [
            'parser', 'serializer', 'adapter', 'factory', 'builder', 'observer',
            'strategy', 'decorator', 'proxy', 'singleton', 'facade'
        ]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name_lower = node.name.lower()
                if any(keyword in func_name_lower for keyword in technical_keywords):
                    technical_concepts.append(node.name)
                elif func_name_lower.startswith(('_', '__')):  # Private/dunder methods
                    technical_concepts.append(node.name)
        
        return list(set(technical_concepts))
    
    def _extract_architectural_patterns(self, tree: ast.AST, content: str) -> List[str]:
        """Extract architectural patterns from AST."""
        patterns = []
        
        pattern_indicators = {
            "MVC": ["controller", "model", "view"],
            "Repository": ["repository", "repo"],
            "Service Layer": ["service", "business"],
            "Factory": ["factory", "create", "build"],
            "Observer": ["observer", "notify", "subscribe"],
            "Decorator": ["decorator", "wrapper", "wrap"],
            "Adapter": ["adapter", "adapt"],
            "Strategy": ["strategy", "algorithm"]
        }
        
        content_lower = content.lower()
        
        for pattern_name, indicators in pattern_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                patterns.append(pattern_name)
        
        return list(set(patterns))
    
    def _fallback_concept_extraction(self, content: str) -> Dict[str, List[str]]:
        """Fallback concept extraction using regex when AST fails."""
        concepts = {
            "domain_entities": [],
            "business_concepts": [],
            "technical_concepts": [],
            "architectural_patterns": []
        }
        
        # Simple regex-based extraction
        class_pattern = r'class\s+([A-Z][a-zA-Z0-9_]*)'
        function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        classes = re.findall(class_pattern, content)
        functions = re.findall(function_pattern, content)
        
        concepts["domain_entities"] = [c for c in classes if not c.startswith('_')]
        concepts["technical_concepts"] = [f for f in functions if f.startswith('_')]
        concepts["business_concepts"] = [f for f in functions if not f.startswith('_')]
        
        return concepts
    
    def cluster_by_semantics(self, intent_classifications: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Cluster files by semantic similarity."""
        clusters = defaultdict(list)
        
        for file_path, intent_data in intent_classifications.items():
            primary_intent = intent_data.get("primary_intent", "unknown")
            clusters[primary_intent].append(file_path)
        
        return dict(clusters)
    
    def build_purpose_linkage(self, intent_classifications: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """Build purpose-based linkage between files."""
        purpose_links = defaultdict(list)
        
        # Group files by primary intent
        intent_groups = defaultdict(list)
        for file_path, intent_data in intent_classifications.items():
            primary_intent = intent_data.get("primary_intent", "unknown")
            intent_groups[primary_intent].append(file_path)
        
        # Create links between files with related purposes
        related_intents = {
            "api_endpoint": ["authentication", "validation", "data_processing"],
            "data_processing": ["database", "validation", "analysis"],
            "authentication": ["security", "validation", "configuration"],
            "testing": ["validation", "utilities", "configuration"],
            "monitoring": ["analysis", "utilities", "integration"],
            "orchestration": ["integration", "background_processing", "monitoring"]
        }
        
        for primary_intent, files in intent_groups.items():
            related = related_intents.get(primary_intent, [])
            
            for file_path in files:
                links = []
                
                # Find related files
                for related_intent in related:
                    for related_file in intent_groups.get(related_intent, []):
                        if related_file != file_path:
                            confidence = intent_classifications[file_path].get("confidence", 0)
                            links.append({
                                "target": related_file,
                                "relationship": f"{primary_intent}_to_{related_intent}",
                                "strength": confidence * 0.8  # Slightly lower strength for purpose links
                            })
                
                purpose_links[file_path] = links
        
        return dict(purpose_links)
    
    def calculate_semantic_metrics(self, semantic_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate semantic analysis metrics."""
        intent_classifications = semantic_results.get("intent_classifications", {})
        
        if not intent_classifications:
            return {"error": "No intent classifications available"}
        
        # Intent distribution
        intent_counts = defaultdict(int)
        confidence_scores = []
        
        for file_data in intent_classifications.values():
            primary_intent = file_data.get("primary_intent", "unknown")
            confidence = file_data.get("confidence", 0)
            
            intent_counts[primary_intent] += 1
            confidence_scores.append(confidence)
        
        total_files = len(intent_classifications)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            "total_files_analyzed": total_files,
            "intent_distribution": dict(intent_counts),
            "most_common_intent": max(intent_counts, key=intent_counts.get) if intent_counts else "none",
            "average_confidence": avg_confidence,
            "high_confidence_files": len([s for s in confidence_scores if s > 0.7]),
            "low_confidence_files": len([s for s in confidence_scores if s < 0.3]),
            "unique_intents_found": len(intent_counts),
            "semantic_diversity": len(intent_counts) / max(total_files, 1)
        }
    
    def get_semantic_summary(self, semantic_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get semantic analysis summary."""
        metrics = semantic_results.get("semantic_metrics", {})
        clusters = semantic_results.get("semantic_clusters", {})
        
        return {
            "analysis_type": "semantic",
            "files_processed": metrics.get("total_files_analyzed", 0),
            "primary_findings": {
                "dominant_intent": metrics.get("most_common_intent", "unknown"),
                "confidence_level": "high" if metrics.get("average_confidence", 0) > 0.6 else "medium",
                "diversity_score": metrics.get("semantic_diversity", 0),
                "cluster_count": len(clusters)
            },
            "recommendations": self._generate_semantic_recommendations(semantic_results)
        }
    
    def _generate_semantic_recommendations(self, semantic_results: Dict[str, Any]) -> List[str]:
        """Generate semantic analysis recommendations."""
        recommendations = []
        
        metrics = semantic_results.get("semantic_metrics", {})
        avg_confidence = metrics.get("average_confidence", 0)
        low_confidence_files = metrics.get("low_confidence_files", 0)
        
        if avg_confidence < 0.5:
            recommendations.append("Consider improving code documentation and naming conventions to increase semantic clarity")
        
        if low_confidence_files > 5:
            recommendations.append("Multiple files have unclear purpose - consider refactoring for better semantic intent")
        
        clusters = semantic_results.get("semantic_clusters", {})
        if len(clusters) > 15:
            recommendations.append("High semantic diversity detected - consider architectural consolidation")
        
        return recommendations

def create_semantic_analyzer() -> SemanticAnalyzer:
    """Factory function to create a configured semantic analyzer."""
    return SemanticAnalyzer()