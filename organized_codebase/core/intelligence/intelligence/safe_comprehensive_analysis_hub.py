"""
Safe Comprehensive Analysis Hub - Complete Analysis Consolidation
=================================================================

Unified analysis hub that safely handles missing dependencies and provides
consolidated access to all available analysis capabilities.

This version focuses on consolidating what exists rather than assuming
all analysis components are available.

Author: Agent A - The Architect  
Date: 2025
Status: SAFE CONSOLIDATION
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from pathlib import Path
import threading
import time
import importlib
import sys


class AnalysisType(Enum):
    """Types of analysis supported"""
    TECHNICAL_DEBT = "technical_debt"
    BUSINESS_LOGIC = "business_logic"
    SEMANTIC_STRUCTURE = "semantic_structure"
    ML_CODE = "ml_code"
    PATTERN_RECOGNITION = "pattern_recognition"
    CROSS_SYSTEM = "cross_system"
    COMPREHENSIVE = "comprehensive"


class AnalysisPriority(Enum):
    """Analysis priority levels"""
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class AnalysisResult:
    """Unified analysis result structure"""
    result_id: str = field(default_factory=lambda: f"result_{int(time.time() * 1000000)}")
    request_id: str = ""
    analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE
    status: str = "pending"  # pending, running, completed, failed
    findings: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    execution_time: timedelta = field(default_factory=lambda: timedelta())
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Dict[str, Any] = field(default_factory=dict)


class SafeAnalyzerLoader:
    """Safely loads available analyzers without breaking on missing dependencies"""
    
    def __init__(self):
        self.available_analyzers = {}
        self.failed_imports = {}
        self.logger = logging.getLogger(__name__)
        
    def load_analyzer(self, analyzer_name: str, module_path: str) -> Optional[Any]:
        """Safely load an analyzer class"""
        try:
            # Try to import the module
            module = importlib.import_module(module_path)
            
            # Get the analyzer class
            analyzer_class = getattr(module, analyzer_name)
            
            # Try to instantiate it
            analyzer_instance = analyzer_class()
            
            self.available_analyzers[analyzer_name] = analyzer_instance
            self.logger.info(f"Successfully loaded analyzer: {analyzer_name}")
            return analyzer_instance
            
        except Exception as e:
            self.failed_imports[analyzer_name] = str(e)
            self.logger.warning(f"Failed to load analyzer {analyzer_name}: {e}")
            return None
    
    def get_available_analyzers(self) -> Dict[str, Any]:
        """Get all successfully loaded analyzers"""
        return self.available_analyzers.copy()
    
    def get_failed_imports(self) -> Dict[str, str]:
        """Get information about failed imports"""
        return self.failed_imports.copy()


class SafeComprehensiveAnalysisHub:
    """
    Safe Comprehensive Analysis Hub
    
    Provides unified access to all available analysis capabilities
    without breaking when some components are missing.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize safe analyzer loader
        self.analyzer_loader = SafeAnalyzerLoader()
        
        # Load available analyzers
        self._load_available_analyzers()
        
        # Request and result management
        self.completed_results: Dict[str, AnalysisResult] = {}
        
        # Performance metrics
        self.performance_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'available_analyzers': len(self.analyzer_loader.available_analyzers),
            'failed_imports': len(self.analyzer_loader.failed_imports)
        }
        
    def _load_available_analyzers(self):
        """Load all available analyzers safely"""
        
        # Define analyzers to attempt loading
        analyzers_to_load = [
            # Technical debt analyzers
            ('TechnicalDebtAnalyzer', 'core.intelligence.analysis.technical_debt_analyzer'),
            
            # Business analyzers  
            ('BusinessAnalyzer', 'core.intelligence.analysis.business_analyzer'),
            
            # Semantic analyzers
            ('SemanticAnalyzer', 'core.intelligence.analysis.semantic_analyzer'),
            
            # ML analyzers
            ('MLCodeAnalyzer', 'core.intelligence.analysis.ml_code_analyzer'),
        ]
        
        for analyzer_name, module_path in analyzers_to_load:
            self.analyzer_loader.load_analyzer(analyzer_name, module_path)
        
        self.logger.info(f"Loaded {len(self.analyzer_loader.available_analyzers)} analyzers successfully")
        if self.analyzer_loader.failed_imports:
            self.logger.warning(f"Failed to load {len(self.analyzer_loader.failed_imports)} analyzers")
    
    async def analyze_comprehensive(
        self,
        target_path: str,
        analysis_types: Optional[List[AnalysisType]] = None,
        priority: AnalysisPriority = AnalysisPriority.MEDIUM,
        config: Optional[Dict] = None
    ) -> AnalysisResult:
        """
        Perform comprehensive analysis using all available analyzers
        """
        
        if analysis_types is None:
            analysis_types = [AnalysisType.COMPREHENSIVE]
        
        start_time = datetime.now()
        
        try:
            # Collect findings from available analyzers
            consolidated_findings = {}
            
            available_analyzers = self.analyzer_loader.get_available_analyzers()
            
            for analyzer_name, analyzer_instance in available_analyzers.items():
                try:
                    # Try to run analysis with each available analyzer
                    if hasattr(analyzer_instance, 'analyze_project'):
                        result = analyzer_instance.analyze_project(target_path)
                    elif hasattr(analyzer_instance, 'analyze'):
                        result = analyzer_instance.analyze(target_path)
                    elif hasattr(analyzer_instance, 'analyze_code'):
                        result = analyzer_instance.analyze_code(target_path)
                    else:
                        # Create a basic result
                        result = {
                            'analyzer': analyzer_name,
                            'status': 'no_analysis_method',
                            'message': f'{analyzer_name} loaded but no standard analysis method found'
                        }
                    
                    consolidated_findings[analyzer_name] = result
                    
                except Exception as e:
                    consolidated_findings[analyzer_name] = {
                        'error': str(e),
                        'status': 'analysis_failed'
                    }
            
            # Create result
            result = AnalysisResult(
                analysis_type=AnalysisType.COMPREHENSIVE,
                status="completed",
                findings=self._format_findings(consolidated_findings),
                metrics=self._calculate_metrics(consolidated_findings),
                recommendations=self._generate_recommendations(consolidated_findings),
                confidence_score=self._calculate_confidence(consolidated_findings),
                execution_time=datetime.now() - start_time,
                raw_data=consolidated_findings
            )
            
            # Store result
            self.completed_results[result.result_id] = result
            
            # Update performance metrics
            self.performance_metrics['total_analyses'] += 1
            self.performance_metrics['successful_analyses'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {str(e)}")
            
            # Create failed result
            result = AnalysisResult(
                analysis_type=AnalysisType.COMPREHENSIVE,
                status="failed",
                findings=[{'error': str(e)}],
                execution_time=datetime.now() - start_time
            )
            
            self.performance_metrics['total_analyses'] += 1
            self.performance_metrics['failed_analyses'] += 1
            
            return result
    
    def _format_findings(self, consolidated_findings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format findings for consistent output"""
        formatted = []
        
        for analyzer_name, findings in consolidated_findings.items():
            if isinstance(findings, dict):
                formatted.append({
                    'analyzer': analyzer_name,
                    'findings': findings,
                    'summary': self._summarize_findings(findings)
                })
        
        return formatted
    
    def _summarize_findings(self, findings: Dict[str, Any]) -> str:
        """Generate summary of findings"""
        if 'error' in findings:
            return f"Analysis failed: {findings['error']}"
        
        if 'status' in findings:
            return f"Status: {findings['status']}"
        
        # Count significant findings
        summary_parts = []
        for key, value in findings.items():
            if isinstance(value, (list, set)) and len(value) > 0:
                summary_parts.append(f"{len(value)} {key}")
            elif isinstance(value, (int, float)) and value > 0:
                summary_parts.append(f"{key}: {value}")
        
        return "; ".join(summary_parts) if summary_parts else "Analysis completed"
    
    def _calculate_metrics(self, consolidated_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall metrics from consolidated findings"""
        metrics = {
            'analyzers_used': len(consolidated_findings),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'available_analyzers': len(self.analyzer_loader.available_analyzers),
            'failed_imports': len(self.analyzer_loader.failed_imports)
        }
        
        for analyzer_name, findings in consolidated_findings.items():
            if isinstance(findings, dict) and 'error' not in findings:
                metrics['successful_analyses'] += 1
            else:
                metrics['failed_analyses'] += 1
        
        return metrics
    
    def _generate_recommendations(self, consolidated_findings: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = set()
        
        # Collect recommendations from successful analyses
        for analyzer_name, findings in consolidated_findings.items():
            if isinstance(findings, dict) and 'recommendations' in findings:
                if isinstance(findings['recommendations'], list):
                    recommendations.update(findings['recommendations'])
        
        # Add meta-recommendations based on analysis
        if len(consolidated_findings) > 3:
            recommendations.add("Multiple analysis types completed - consider regular monitoring")
        
        failed_count = sum(1 for f in consolidated_findings.values() 
                          if isinstance(f, dict) and 'error' in f)
        if failed_count > 0:
            recommendations.add(f"Some analyzers failed ({failed_count}) - check dependencies")
        
        return list(recommendations)
    
    def _calculate_confidence(self, consolidated_findings: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        if not consolidated_findings:
            return 0.0
        
        successful_analyses = sum(1 for findings in consolidated_findings.values() 
                                if isinstance(findings, dict) and 'error' not in findings)
        
        return successful_analyses / len(consolidated_findings)
    
    def get_consolidated_analysis_capabilities(self) -> Dict[str, Any]:
        """Get information about consolidated analysis capabilities"""
        available_analyzers = self.analyzer_loader.get_available_analyzers()
        failed_imports = self.analyzer_loader.get_failed_imports()
        
        return {
            'total_analyzers_attempted': len(available_analyzers) + len(failed_imports),
            'total_analyzers_loaded': len(available_analyzers),
            'total_analyzers_failed': len(failed_imports),
            'available_analyzers': list(available_analyzers.keys()),
            'failed_analyzers': list(failed_imports.keys()),
            'supported_analysis_types': [t.value for t in AnalysisType],
            'integration_status': f'PARTIAL - {len(available_analyzers)} components available',
            'access_method': 'SafeComprehensiveAnalysisHub.analyze_comprehensive()'
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the analysis hub"""
        return self.performance_metrics.copy()


# Convenience function for direct access
async def analyze_project_safe(
    target_path: str,
    analysis_types: Optional[List[AnalysisType]] = None,
    config: Optional[Dict] = None
) -> AnalysisResult:
    """
    Convenience function for safe comprehensive project analysis
    """
    
    hub = SafeComprehensiveAnalysisHub(config)
    return await hub.analyze_comprehensive(target_path, analysis_types)


# Main execution for demonstration
async def demonstrate_safe_analysis():
    """Demonstrate the safe comprehensive analysis hub"""
    
    print("=" * 80)
    print("SAFE COMPREHENSIVE ANALYSIS HUB")
    print("Consolidating Available Analysis Components")
    print("=" * 80)
    print()
    
    # Initialize the safe hub
    hub = SafeComprehensiveAnalysisHub()
    
    # Show capabilities
    capabilities = hub.get_consolidated_analysis_capabilities()
    
    print("Analysis Hub Status:")
    print(f"  Analyzers Attempted: {capabilities['total_analyzers_attempted']}")
    print(f"  Analyzers Loaded: {capabilities['total_analyzers_loaded']}")
    print(f"  Analyzers Failed: {capabilities['total_analyzers_failed']}")
    print(f"  Integration Status: {capabilities['integration_status']}")
    print()
    
    print("Available Analyzers:")
    for analyzer in capabilities['available_analyzers']:
        print(f"  - {analyzer}")
    print()
    
    if capabilities['failed_analyzers']:
        print("Failed Analyzers:")
        for analyzer in capabilities['failed_analyzers']:
            print(f"  - {analyzer}")
        print()
    
    # Simulate analysis
    target_path = "."
    
    print(f"Performing safe comprehensive analysis on: {target_path}")
    print()
    
    result = await hub.analyze_comprehensive(target_path)
    
    # Display results
    print("=" * 80)
    print("SAFE CONSOLIDATION COMPLETE")
    print("=" * 80)
    print()
    
    print(f"Analysis ID: {result.result_id}")
    print(f"Status: {result.status}")
    print(f"Execution Time: {result.execution_time}")
    print(f"Confidence Score: {result.confidence_score:.2%}")
    print()
    
    print("Analysis Coverage:")
    for finding in result.findings:
        print(f"  - {finding['analyzer']}: {finding['summary']}")
    print()
    
    print("Key Metrics:")
    for metric, value in result.metrics.items():
        print(f"  {metric}: {value}")
    print()
    
    if result.recommendations:
        print("Recommendations:")
        for i, recommendation in enumerate(result.recommendations[:5], 1):
            print(f"  {i}. {recommendation}")
    print()
    
    print("=" * 80)
    print("SAFE ANALYSIS CONSOLIDATION COMPLETE")
    print("Working with available components only")
    print("=" * 80)


if __name__ == "__main__":
    # Run the safe analysis demonstration
    asyncio.run(demonstrate_safe_analysis())