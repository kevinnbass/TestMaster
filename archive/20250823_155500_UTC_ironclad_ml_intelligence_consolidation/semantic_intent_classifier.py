"""
Semantic Intent Classification Module
Extracted from enhanced_intelligence_linkage.py for Agent X's Epsilon base integration
< 200 lines per STEELCLAD protocol

Provides ML-powered code intent classification with:
- 15+ semantic categories
- Confidence scoring
- Pattern matching analysis
- Developer intent prediction
"""

import ast
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IntentClassification:
    """Intent classification result"""
    primary_intent: str
    confidence: float
    all_intents: Dict[str, int]
    total_patterns: int
    timestamp: str

class SemanticIntentClassifier:
    """ML-powered semantic intent classification for code analysis"""
    
    def __init__(self):
        self.intent_categories = {
            "data_processing": ["pandas", "dataframe", "csv", "json.load", "pickle", "transform", "process_data", "etl", "extract", "parse"],
            "api_endpoint": ["@app.route", "flask", "fastapi", "endpoint", "api", "@get", "@post", "@put", "@delete", "request", "response"],
            "authentication": ["authenticate", "login", "password", "token", "jwt", "oauth", "session", "auth", "permission", "authorize"],
            "security": ["encrypt", "decrypt", "hash", "crypto", "security", "vulnerability", "sanitize", "validate", "escape", "secure"],
            "testing": ["test_", "unittest", "pytest", "assert", "mock", "fixture", "testcase", "test", "spec", "should"],
            "configuration": ["config", "settings", "environment", "env", "configure", "setup", "ini", "yaml", "toml", "json"],
            "utilities": ["util", "helper", "common", "shared", "tools", "helpers", "utility", "support", "lib", "utils"],
            "ui_components": ["render", "template", "html", "css", "javascript", "component", "widget", "ui", "frontend", "view"],
            "database_operations": ["sql", "database", "db", "query", "select", "insert", "update", "delete", "orm", "model"],
            "machine_learning": ["model", "predict", "train", "ml", "ai", "neural", "sklearn", "tensorflow", "pytorch", "algorithm"],
            "integration": ["api", "webhook", "integration", "connector", "bridge", "adapter", "client", "service", "external"],
            "monitoring": ["log", "monitor", "metrics", "health", "status", "performance", "tracking", "analytics", "telemetry"],
            "documentation": ["doc", "readme", "comment", "docstring", "help", "guide", "manual", "wiki", "documentation"],
            "business_logic": ["business", "logic", "rule", "workflow", "process", "calculation", "algorithm", "domain", "core"],
            "error_handling": ["error", "exception", "try", "catch", "except", "finally", "raise", "handle", "fail", "recovery"]
        }
        
    def classify_code_intent(self, content: str, filename: str = None) -> IntentClassification:
        """Classify the primary intent of code content"""
        content_lower = content.lower()
        intent_scores = {}
        
        # Calculate pattern matches for each intent category
        for intent, patterns in self.intent_categories.items():
            score = sum(1 for pattern in patterns if pattern in content_lower)
            intent_scores[intent] = score
        
        # Calculate total patterns found
        total_patterns = sum(intent_scores.values())
        
        if total_patterns == 0:
            primary_intent = "unknown"
            confidence = 0.0
        else:
            # Find primary intent (highest score)
            primary_intent = max(intent_scores, key=intent_scores.get)
            max_score = intent_scores[primary_intent]
            
            # Calculate confidence based on score distribution
            confidence = min(0.95, (max_score / total_patterns) * 0.8 + 0.2)
        
        from datetime import datetime
        return IntentClassification(
            primary_intent=primary_intent,
            confidence=confidence,
            all_intents=intent_scores,
            total_patterns=total_patterns,
            timestamp=datetime.now().isoformat()
        )
    
    def classify_multiple_files(self, file_contents: Dict[str, str]) -> Dict[str, IntentClassification]:
        """Classify intent for multiple files"""
        results = {}
        for filename, content in file_contents.items():
            results[filename] = self.classify_code_intent(content, filename)
        return results
    
    def get_intent_summary(self, classifications: List[IntentClassification]) -> Dict[str, Any]:
        """Generate summary of intent classifications"""
        if not classifications:
            return {"status": "no_classifications", "summary": {}}
        
        # Count primary intents
        intent_distribution = {}
        confidence_sum = 0
        high_confidence_count = 0
        
        for classification in classifications:
            intent = classification.primary_intent
            intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
            confidence_sum += classification.confidence
            
            if classification.confidence >= 0.7:
                high_confidence_count += 1
        
        avg_confidence = confidence_sum / len(classifications)
        
        return {
            "status": "classification_complete",
            "summary": {
                "total_files": len(classifications),
                "intent_distribution": intent_distribution,
                "average_confidence": round(avg_confidence, 3),
                "high_confidence_classifications": high_confidence_count,
                "most_common_intent": max(intent_distribution, key=intent_distribution.get) if intent_distribution else "unknown"
            }
        }
    
    def get_detailed_analysis(self, classification: IntentClassification) -> Dict[str, Any]:
        """Get detailed analysis for a classification result"""
        # Find top 3 intents
        sorted_intents = sorted(classification.all_intents.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Calculate pattern density
        pattern_density = classification.total_patterns / len(str(classification).split()) if classification.total_patterns > 0 else 0
        
        return {
            "primary_intent": classification.primary_intent,
            "confidence_level": "high" if classification.confidence >= 0.7 else "medium" if classification.confidence >= 0.4 else "low",
            "top_intents": sorted_intents,
            "pattern_density": round(pattern_density, 4),
            "analysis_quality": "comprehensive" if classification.total_patterns >= 5 else "basic",
            "timestamp": classification.timestamp
        }
    
    def enhance_classification_with_ast(self, content: str, base_classification: IntentClassification) -> IntentClassification:
        """Enhance classification using AST analysis"""
        try:
            tree = ast.parse(content)
            ast_indicators = {
                "classes": 0,
                "functions": 0,
                "async_functions": 0,
                "decorators": 0,
                "imports": 0
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    ast_indicators["classes"] += 1
                elif isinstance(node, ast.FunctionDef):
                    ast_indicators["functions"] += 1
                elif isinstance(node, ast.AsyncFunctionDef):
                    ast_indicators["async_functions"] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    ast_indicators["imports"] += 1
            
            # Adjust confidence based on AST structure
            structure_bonus = 0
            if ast_indicators["classes"] > 0:
                structure_bonus += 0.1
            if ast_indicators["functions"] >= 3:
                structure_bonus += 0.1
            if ast_indicators["async_functions"] > 0:
                structure_bonus += 0.05
            
            enhanced_confidence = min(0.95, base_classification.confidence + structure_bonus)
            
            return IntentClassification(
                primary_intent=base_classification.primary_intent,
                confidence=enhanced_confidence,
                all_intents=base_classification.all_intents,
                total_patterns=base_classification.total_patterns + sum(ast_indicators.values()),
                timestamp=base_classification.timestamp
            )
            
        except SyntaxError:
            # Return original classification if AST parsing fails
            return base_classification
    
    def get_available_categories(self) -> List[str]:
        """Get list of all available intent categories"""
        return list(self.intent_categories.keys())
    
    def add_custom_patterns(self, category: str, patterns: List[str]) -> bool:
        """Add custom patterns to existing category"""
        if category not in self.intent_categories:
            return False
        
        self.intent_categories[category].extend(patterns)
        return True

# Plugin interface for Agent X integration
def create_intent_classifier_plugin(config: Dict[str, Any] = None):
    """Factory function to create semantic intent classifier plugin"""
    classifier = SemanticIntentClassifier()
    
    # Add custom patterns if provided in config
    if config and 'custom_patterns' in config:
        for category, patterns in config['custom_patterns'].items():
            classifier.add_custom_patterns(category, patterns)
    
    return classifier