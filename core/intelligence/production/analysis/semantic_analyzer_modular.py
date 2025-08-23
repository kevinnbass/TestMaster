"""
Modularized Semantic Analyzer
=============================

Main orchestrator for semantic code analysis using modular components.
Replaces the original 952-line semantic_analyzer.py.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .semantic_base import SemanticConfiguration
from .semantic_intent_analyzer import SemanticIntentAnalyzer
from .semantic_pattern_detector import SemanticPatternDetector
from .semantic_relationship_analyzer import SemanticRelationshipAnalyzer


class SemanticAnalyzer:
    """
    Orchestrates semantic code analysis across all components.
    
    This modularized version maintains all functionality of the original
    while being split into focused, maintainable components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the semantic analyzer."""
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize configuration
        self.semantic_config = SemanticConfiguration()
        
        # Initialize components
        self.intent_analyzer = SemanticIntentAnalyzer(self.semantic_config)
        self.pattern_detector = SemanticPatternDetector(self.semantic_config)
        self.relationship_analyzer = SemanticRelationshipAnalyzer(self.semantic_config)
        
        # Results storage
        self.semantic_intents = []
        self.conceptual_patterns = []
        
        # Configuration
        self.base_path = None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the analyzer."""
        logger = logging.getLogger("semantic_analyzer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive semantic analysis.
        
        Args:
            path: Optional path to analyze. If None, uses current directory.
        """
        self.base_path = Path(path) if path else Path.cwd()
        self.logger.info(f"Starting semantic analysis for: {self.base_path}")
        
        # Get Python files
        python_files = self._get_python_files()
        self.logger.info(f"Found {len(python_files)} Python files to analyze")
        
        # Run analyses using components
        results = {
            "intent_recognition": self.intent_analyzer.recognize_intent(python_files),
            "semantic_signatures": self.intent_analyzer.extract_semantic_signatures(python_files),
            "conceptual_patterns": self.pattern_detector.identify_conceptual_patterns(python_files),
            "semantic_relationships": self.relationship_analyzer.analyze_semantic_relationships(python_files),
            "purpose_classification": self.intent_analyzer.classify_code_purpose(python_files),
            "naming_semantics": self.relationship_analyzer.analyze_naming_semantics(python_files),
            "behavioral_patterns": self.pattern_detector.identify_behavioral_patterns(python_files),
            "domain_concepts": self.pattern_detector.extract_domain_concepts(python_files),
            "semantic_clustering": self.pattern_detector.perform_semantic_clustering(python_files),
            "intent_consistency": self.intent_analyzer.check_intent_consistency(python_files),
            "semantic_quality": self.relationship_analyzer.assess_semantic_quality(python_files),
            "summary": self._generate_semantic_summary()
        }
        
        # Store results for summary generation
        self.semantic_intents = self.intent_analyzer.semantic_intents
        self.conceptual_patterns = self.pattern_detector.conceptual_patterns
        
        self.logger.info(f"Analysis complete: {len(self.semantic_intents)} intents recognized")
        
        return results
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files in the project"""
        python_files = []
        
        if self.base_path.exists():
            for file_path in self.base_path.rglob("*.py"):
                # Skip common excluded directories
                if not self._should_exclude(file_path):
                    python_files.append(file_path)
        
        return python_files
    
    def _should_exclude(self, file_path: Path) -> bool:
        """Check if file should be excluded from analysis"""
        exclude_patterns = [
            '__pycache__', '.git', '.pytest_cache', 'venv', 'env',
            'build', 'dist', '*.egg-info', '.venv', 'node_modules'
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in exclude_patterns)
    
    def _generate_semantic_summary(self) -> Dict[str, Any]:
        """Generate a summary of semantic analysis"""
        summary = {
            "total_intents_recognized": len(self.semantic_intents),
            "dominant_intent_type": self._get_dominant_intent(),
            "pattern_count": len(self.conceptual_patterns),
            "semantic_complexity": self._calculate_semantic_complexity(),
            "recommendations": self._generate_semantic_recommendations(),
            "quality_metrics": self._calculate_quality_metrics()
        }
        
        return summary
    
    def _get_dominant_intent(self) -> str:
        """Get the most common intent type"""
        if not self.semantic_intents:
            return "unknown"
        
        from collections import defaultdict
        intent_counts = defaultdict(int)
        for intent in self.semantic_intents:
            intent_counts[intent.primary_intent.value] += 1
        
        return max(intent_counts, key=intent_counts.get) if intent_counts else "unknown"
    
    def _calculate_semantic_complexity(self) -> float:
        """Calculate overall semantic complexity"""
        if not self.semantic_intents:
            return 0.0
        
        unique_intents = len(set(i.primary_intent for i in self.semantic_intents))
        total_intents = len(self.semantic_intents)
        
        return unique_intents / total_intents if total_intents > 0 else 0.0
    
    def _generate_semantic_recommendations(self) -> List[str]:
        """Generate recommendations based on semantic analysis"""
        recommendations = []
        
        complexity = self._calculate_semantic_complexity()
        if complexity > 0.8:
            recommendations.append(
                "Consider reducing semantic complexity by grouping related functionality"
            )
        
        pattern_count = len(self.conceptual_patterns)
        if pattern_count < 5:
            recommendations.append(
                "Consider implementing more design patterns for better structure"
            )
        
        # Check for low confidence intents
        low_confidence_count = sum(1 for intent in self.semantic_intents if intent.confidence < 0.7)
        if low_confidence_count > len(self.semantic_intents) * 0.3:
            recommendations.append(
                "Improve code clarity to make intent more obvious"
            )
        
        return recommendations
    
    def _calculate_quality_metrics(self) -> Dict[str, float]:
        """Calculate overall quality metrics"""
        if not self.semantic_intents:
            return {"overall": 0.0, "clarity": 0.0, "consistency": 0.0}
        
        # Calculate average confidence as clarity metric
        avg_confidence = sum(intent.confidence for intent in self.semantic_intents) / len(self.semantic_intents)
        
        # Calculate consistency based on intent distribution
        from collections import Counter
        intent_distribution = Counter(intent.primary_intent for intent in self.semantic_intents)
        max_intent_count = max(intent_distribution.values()) if intent_distribution else 0
        consistency = max_intent_count / len(self.semantic_intents) if self.semantic_intents else 0
        
        overall = (avg_confidence + consistency) / 2
        
        return {
            "overall": overall,
            "clarity": avg_confidence,
            "consistency": consistency
        }
    
    def get_intents_by_type(self, intent_type_name: str) -> List:
        """Get all intents of a specific type"""
        from .semantic_base import IntentType
        
        try:
            intent_type = IntentType(intent_type_name)
            return [intent for intent in self.semantic_intents 
                    if intent.primary_intent == intent_type]
        except ValueError:
            return []
    
    def get_low_confidence_intents(self, threshold: float = 0.7) -> List:
        """Get intents with confidence below threshold"""
        return [intent for intent in self.semantic_intents 
                if intent.confidence < threshold]
    
    def export_report(self, format: str = 'json') -> str:
        """Export semantic analysis report in specified format"""
        results = self.analyze() if not self.semantic_intents else self._generate_semantic_summary()
        
        if format == 'json':
            import json
            return json.dumps(results, indent=2, default=str)
        elif format == 'markdown':
            return self._format_as_markdown(results)
        else:
            return str(results)
    
    def _format_as_markdown(self, results: Dict[str, Any]) -> str:
        """Format results as markdown"""
        summary = results if isinstance(results, dict) and 'total_intents_recognized' in results else results.get('summary', {})
        
        md = f"""# Semantic Analysis Report

## Summary
- **Total Intents Recognized**: {summary.get('total_intents_recognized', 0)}
- **Dominant Intent Type**: {summary.get('dominant_intent_type', 'unknown')}
- **Pattern Count**: {summary.get('pattern_count', 0)}
- **Semantic Complexity**: {summary.get('semantic_complexity', 0.0):.2f}

## Quality Metrics
"""
        
        quality_metrics = summary.get('quality_metrics', {})
        for metric, value in quality_metrics.items():
            md += f"- **{metric.title()}**: {value:.2f}\n"
        
        md += "\n## Recommendations\n"
        for rec in summary.get('recommendations', []):
            md += f"- {rec}\n"
        
        return md


# Export
__all__ = ['SemanticAnalyzer']