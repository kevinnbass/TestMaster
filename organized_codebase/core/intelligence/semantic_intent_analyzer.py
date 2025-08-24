"""
Semantic Intent Analysis Component
==================================

Analyzes and recognizes developer intent from code elements.
Part of modularized semantic_analyzer system.
"""

import ast
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict

from .semantic_base import (
    SemanticIntent, IntentType, SemanticConfiguration
)


class SemanticIntentAnalyzer:
    """Analyzes semantic intent of code elements"""
    
    def __init__(self, config: SemanticConfiguration):
        self.config = config
        self.semantic_intents = []
    
    def recognize_intent(self, python_files: List[Path]) -> Dict[str, Any]:
        """Recognize developer intent from code elements"""
        intent_analysis = {
            "recognized_intents": [],
            "intent_distribution": defaultdict(int),
            "confidence_scores": [],
            "ambiguous_intents": []
        }
        
        self.semantic_intents = []
        
        for file_path in python_files:
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Analyze functions
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            intent = self._analyze_function_intent(node, file_path)
                            self.semantic_intents.append(intent)
                            intent_analysis["recognized_intents"].append(intent)
                            intent_analysis["intent_distribution"][intent.primary_intent.value] += 1
                            intent_analysis["confidence_scores"].append(intent.confidence)
                            
                            if intent.confidence < 0.7:
                                intent_analysis["ambiguous_intents"].append({
                                    "name": intent.name,
                                    "location": str(file_path),
                                    "possible_intents": [i.value for i in intent.secondary_intents]
                                })
                                
                        # Analyze classes
                        elif isinstance(node, ast.ClassDef):
                            intent = self._analyze_class_intent(node, file_path)
                            self.semantic_intents.append(intent)
                            intent_analysis["recognized_intents"].append(intent)
                            intent_analysis["intent_distribution"][intent.primary_intent.value] += 1
                            
            except Exception as e:
                print(f"Error recognizing intent in {file_path}: {e}")
                
        return intent_analysis
    
    def extract_semantic_signatures(self, python_files: List[Path]) -> Dict[str, Any]:
        """Extract semantic signatures from code"""
        signatures = {
            "function_signatures": [],
            "class_signatures": [],
            "module_signatures": [],
            "signature_patterns": defaultdict(list)
        }
        
        for file_path in python_files:
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Extract module signature
                    module_sig = self._create_module_signature(tree, file_path)
                    signatures["module_signatures"].append(module_sig)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_sig = self._create_function_signature(node)
                            signatures["function_signatures"].append({
                                "name": node.name,
                                "signature": func_sig,
                                "location": str(file_path)
                            })
                            signatures["signature_patterns"][func_sig].append(node.name)
                            
                        elif isinstance(node, ast.ClassDef):
                            class_sig = self._create_class_signature(node)
                            signatures["class_signatures"].append({
                                "name": node.name,
                                "signature": class_sig,
                                "location": str(file_path)
                            })
                            
            except Exception as e:
                print(f"Error extracting signatures from {file_path}: {e}")
                
        return signatures
    
    def classify_code_purpose(self, python_files: List[Path]) -> Dict[str, Any]:
        """Classify the purpose of code sections"""
        purpose_classification = {
            "business_logic": [],
            "infrastructure": [],
            "utilities": [],
            "data_access": [],
            "presentation": [],
            "integration": []
        }
        
        for file_path in python_files:
            try:
                # Classify based on file location and content
                file_purpose = self._determine_file_purpose(file_path)
                
                if file_purpose:
                    purpose_classification[file_purpose].append(str(file_path))
                    
            except Exception as e:
                print(f"Error classifying {file_path}: {e}")
                
        return purpose_classification
    
    def check_intent_consistency(self, python_files: List[Path]) -> Dict[str, Any]:
        """Check consistency of intent across codebase"""
        consistency_analysis = {
            "consistent_patterns": [],
            "inconsistencies": [],
            "naming_intent_mismatches": [],
            "structural_inconsistencies": []
        }
        
        # Analyze consistency across similar functions
        function_groups = defaultdict(list)
        
        for file_path in python_files:
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Group by similar names
                            base_name = self._extract_base_name(node.name)
                            function_groups[base_name].append({
                                "name": node.name,
                                "location": str(file_path),
                                "structure": self._get_function_structure(node)
                            })
                            
            except Exception as e:
                print(f"Error checking consistency in {file_path}: {e}")
                
        # Check for inconsistencies
        for base_name, functions in function_groups.items():
            if len(functions) > 1:
                if not self._are_structures_consistent(functions):
                    consistency_analysis["structural_inconsistencies"].append({
                        "base_name": base_name,
                        "functions": functions
                    })
                    
        return consistency_analysis
    
    def _parse_file(self, file_path: Path) -> Optional[ast.AST]:
        """Parse a Python file into an AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return ast.parse(f.read())
        except Exception:
            return None
    
    def _analyze_function_intent(self, node: ast.FunctionDef, file_path: Path) -> SemanticIntent:
        """Analyze the intent of a function"""
        primary_intent = IntentType.UNKNOWN
        secondary_intents = []
        confidence = 0.0
        
        # Analyze function name
        name_lower = node.name.lower()
        intent_scores = defaultdict(float)
        
        for intent_type, keywords in self.config.intent_keywords.items():
            for keyword in keywords:
                if keyword in name_lower:
                    intent_scores[intent_type] += 1.0
                    
        # Analyze function body
        for child in ast.walk(node):
            for intent_type, keywords in self.config.intent_keywords.items():
                if isinstance(child, ast.Name):
                    if any(keyword in child.id.lower() for keyword in keywords):
                        intent_scores[intent_type] += 0.5
                        
        # Determine primary intent
        if intent_scores:
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
            primary_intent = sorted_intents[0][0]
            confidence = min(sorted_intents[0][1] / 5.0, 1.0)  # Normalize confidence
            
            # Get secondary intents
            secondary_intents = [intent for intent, score in sorted_intents[1:3] if score > 0]
            
        return SemanticIntent(
            element_type="function",
            name=node.name,
            location=str(file_path),
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            confidence=confidence,
            semantic_signature=self._create_function_signature(node),
            context={},
            relationships=[]
        )
    
    def _analyze_class_intent(self, node: ast.ClassDef, file_path: Path) -> SemanticIntent:
        """Analyze the intent of a class"""
        return SemanticIntent(
            element_type="class",
            name=node.name,
            location=str(file_path),
            primary_intent=self._determine_class_intent(node),
            secondary_intents=[],
            confidence=0.8,
            semantic_signature=self._create_class_signature(node),
            context={},
            relationships=[]
        )
    
    def _determine_class_intent(self, node: ast.ClassDef) -> IntentType:
        """Determine the primary intent of a class"""
        name_lower = node.name.lower()
        
        # Check common patterns
        if any(pattern in name_lower for pattern in ["model", "entity", "dto"]):
            return IntentType.PERSISTENCE
        elif any(pattern in name_lower for pattern in ["service", "manager", "handler"]):
            return IntentType.ORCHESTRATION
        elif any(pattern in name_lower for pattern in ["validator", "checker"]):
            return IntentType.VALIDATION
        elif any(pattern in name_lower for pattern in ["controller", "endpoint", "api"]):
            return IntentType.API_ENDPOINT
        elif "test" in name_lower:
            return IntentType.TESTING
        else:
            return IntentType.UNKNOWN
    
    def _create_function_signature(self, node: ast.FunctionDef) -> str:
        """Create a semantic signature for a function"""
        params = [arg.arg for arg in node.args.args]
        return f"func:{node.name}({','.join(params)})"
    
    def _create_class_signature(self, node: ast.ClassDef) -> str:
        """Create a semantic signature for a class"""
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        return f"class:{node.name}[{','.join(methods[:3])}]"
    
    def _create_module_signature(self, tree: ast.AST, file_path: Path) -> Dict[str, Any]:
        """Create a semantic signature for a module"""
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        
        return {
            "module": str(file_path),
            "classes": classes[:5],
            "functions": functions[:5],
            "signature": f"module:{file_path.stem}[{len(classes)}c,{len(functions)}f]"
        }
    
    def _determine_file_purpose(self, file_path: Path) -> Optional[str]:
        """Determine the purpose of a file based on location and content"""
        path_str = str(file_path).lower()
        
        for purpose, indicators in self.config.purpose_indicators.items():
            if any(indicator in path_str for indicator in indicators):
                return purpose
        
        return None
    
    def _extract_base_name(self, name: str) -> str:
        """Extract base name from function name"""
        # Remove common prefixes/suffixes
        for prefix in ["get_", "set_", "is_", "has_", "can_"]:
            if name.startswith(prefix):
                return name[len(prefix):]
        return name
    
    def _get_function_structure(self, node: ast.FunctionDef) -> Dict:
        """Get structural information about a function"""
        return {
            "params": len(node.args.args),
            "lines": len(node.body),
            "has_return": any(isinstance(n, ast.Return) for n in ast.walk(node))
        }
    
    def _are_structures_consistent(self, functions: List[Dict]) -> bool:
        """Check if function structures are consistent"""
        if not functions:
            return True
        first_structure = functions[0]["structure"]
        return all(f["structure"] == first_structure for f in functions)
    
    def get_dominant_intent(self) -> str:
        """Get the most common intent type"""
        if not self.semantic_intents:
            return "unknown"
        intent_counts = defaultdict(int)
        for intent in self.semantic_intents:
            intent_counts[intent.primary_intent.value] += 1
        return max(intent_counts, key=intent_counts.get) if intent_counts else "unknown"
    
    def get_intents_by_type(self, intent_type: IntentType) -> List[SemanticIntent]:
        """Get all intents of a specific type"""
        return [intent for intent in self.semantic_intents 
                if intent.primary_intent == intent_type]


# Export
__all__ = ['SemanticIntentAnalyzer']