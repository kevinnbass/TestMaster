"""
Intelligent Code Optimization Package - Enterprise-grade AI-powered code analysis

This package provides comprehensive intelligent code optimization capabilities
including performance analysis, security assessment, quality evaluation, and
automated improvement recommendations with machine learning insights.

Key Components:
- Optimization Models: Core data structures and enums for optimization analysis
- Optimization Engine: Main AI-powered analysis engine with ML pattern recognition
- Performance Analyzer: Advanced performance bottleneck detection and optimization
- Security Optimizer: Comprehensive security vulnerability analysis and remediation
- Quality Analyzer: Code quality assessment with maintainability recommendations

Enterprise Features:
- Multi-layered analysis with confidence scoring and risk assessment
- Machine learning pattern recognition for continuous improvement
- OWASP Top 10 compliance and CWE mapping for security vulnerabilities
- Algorithmic complexity analysis with Big O notation recommendations
- Design pattern recognition and anti-pattern detection
- Comprehensive metrics calculation including Maintainability Index
"""

from .optimization_models import (
    # Core enums
    OptimizationType,
    OptimizationPriority, 
    OptimizationStrategy,
    RecommendationStatus,
    AnalysisType,
    
    # Data models
    OptimizationRecommendation,
    OptimizationResult,
    OptimizationSession,
    OptimizationContext,
    PerformanceMetrics,
    QualityMetrics,
    SecurityMetrics,
    RiskAssessment,
    LearningEntry,
    
    # Factory functions
    create_optimization_recommendation,
    create_performance_metrics,
    create_optimization_session,
    create_risk_assessment,
    create_learning_entry,
    
    # Utility functions
    get_priority_weight,
    calculate_improvement_score,
    sort_recommendations_by_impact,
    
    # Constants
    DEFAULT_PERFORMANCE_THRESHOLDS,
    DEFAULT_QUALITY_THRESHOLDS,
    OPTIMIZATION_TYPE_WEIGHTS
)

from .optimization_engine import (
    OptimizationEngine,
    create_optimization_engine
)

from .performance_analyzer import (
    PerformanceAnalyzer,
    PerformancePattern,
    AlgorithmicComplexity,
    create_performance_analyzer
)

from .security_optimizer import (
    SecurityOptimizer,
    SecurityVulnerability,
    SecurityPattern,
    create_security_optimizer
)

from .quality_analyzer import (
    QualityAnalyzer,
    QualityIssue,
    ComplexityMetrics,
    DesignPattern,
    create_quality_analyzer
)

__all__ = [
    # Main components
    'OptimizationEngine',
    'PerformanceAnalyzer',
    'SecurityOptimizer',
    'QualityAnalyzer',
    
    # Factory functions
    'create_optimization_engine',
    'create_performance_analyzer', 
    'create_security_optimizer',
    'create_quality_analyzer',
    
    # Core enums
    'OptimizationType',
    'OptimizationPriority',
    'OptimizationStrategy',
    'RecommendationStatus',
    'AnalysisType',
    
    # Data models
    'OptimizationRecommendation',
    'OptimizationResult',
    'OptimizationSession',
    'OptimizationContext',
    'PerformanceMetrics',
    'QualityMetrics',
    'SecurityMetrics',
    'RiskAssessment',
    'LearningEntry',
    
    # Analysis-specific models
    'PerformancePattern',
    'AlgorithmicComplexity',
    'SecurityVulnerability',
    'SecurityPattern',
    'QualityIssue',
    'ComplexityMetrics',
    'DesignPattern',
    
    # Model factory functions
    'create_optimization_recommendation',
    'create_performance_metrics',
    'create_optimization_session',
    'create_risk_assessment',
    'create_learning_entry',
    
    # Utility functions
    'get_priority_weight',
    'calculate_improvement_score',
    'sort_recommendations_by_impact',
    
    # Constants
    'DEFAULT_PERFORMANCE_THRESHOLDS',
    'DEFAULT_QUALITY_THRESHOLDS',
    'OPTIMIZATION_TYPE_WEIGHTS',
    
    # Convenience imports
    'quick_analysis',
    'comprehensive_analysis',
    'security_analysis',
    'performance_analysis',
    'quality_analysis'
]

# Version information
__version__ = '1.0.0'
__author__ = 'TestMaster Intelligence Team'
__description__ = 'Enterprise-grade AI-powered code optimization with comprehensive analysis capabilities'


def quick_analysis(file_path: str, analysis_types: list = None) -> OptimizationSession:
    """
    Perform quick code analysis with default settings
    
    Args:
        file_path: Path to the code file to analyze
        analysis_types: List of analysis types to perform (default: all)
        
    Returns:
        Optimization session with results
        
    Example:
        >>> from core.optimization import quick_analysis
        >>> session = quick_analysis('my_module.py')
        >>> print(f"Found {len(session.recommendations)} recommendations")
    """
    engine = create_optimization_engine()
    context = OptimizationContext(
        project_name="Quick Analysis",
        language="python",
        environment="development"
    )
    
    session = create_optimization_session(
        project_name="Quick Analysis",
        target_files=[file_path],
        analysis_types=analysis_types or [
            AnalysisType.STATIC_ANALYSIS,
            AnalysisType.PERFORMANCE_PROFILING,
            AnalysisType.SECURITY_SCAN,
            AnalysisType.QUALITY_METRICS
        ]
    )
    session.context = context
    
    return session


async def comprehensive_analysis(
    file_paths: list,
    project_name: str = "Comprehensive Analysis",
    include_learning: bool = True
) -> OptimizationSession:
    """
    Perform comprehensive analysis across multiple files
    
    Args:
        file_paths: List of file paths to analyze
        project_name: Name of the project being analyzed
        include_learning: Whether to enable machine learning insights
        
    Returns:
        Comprehensive optimization session with detailed analysis
        
    Example:
        >>> import asyncio
        >>> from core.optimization import comprehensive_analysis
        >>> session = await comprehensive_analysis(['module1.py', 'module2.py'])
        >>> high_priority = [r for r in session.recommendations if r.priority == OptimizationPriority.HIGH]
    """
    engine = create_optimization_engine(learning_enabled=include_learning)
    performance_analyzer = create_performance_analyzer()
    security_optimizer = create_security_optimizer()
    quality_analyzer = create_quality_analyzer()
    
    context = OptimizationContext(
        project_name=project_name,
        language="python",
        environment="development"
    )
    
    session = create_optimization_session(
        project_name=project_name,
        target_files=file_paths,
        analysis_types=[
            AnalysisType.STATIC_ANALYSIS,
            AnalysisType.PERFORMANCE_PROFILING,
            AnalysisType.SECURITY_SCAN,
            AnalysisType.COMPLEXITY_ANALYSIS,
            AnalysisType.QUALITY_METRICS,
            AnalysisType.ARCHITECTURE_ANALYSIS
        ]
    )
    session.context = context
    
    # Perform comprehensive analysis on each file
    all_recommendations = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            import ast
            tree = ast.parse(code)
            
            # Multi-analyzer approach
            engine_recs = await engine.analyze_code(file_path, context)
            perf_recs = await performance_analyzer.analyze_performance(code, tree, file_path)
            security_recs = await security_optimizer.analyze_security(code, tree, file_path)
            quality_recs = await quality_analyzer.analyze_quality(code, tree, file_path)
            
            all_recommendations.extend(engine_recs)
            all_recommendations.extend(perf_recs)
            all_recommendations.extend(security_recs)
            all_recommendations.extend(quality_recs)
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    # Sort and deduplicate recommendations
    session.recommendations = sort_recommendations_by_impact(all_recommendations)
    
    return session


async def security_analysis(file_path: str) -> list:
    """
    Perform focused security analysis on a single file
    
    Args:
        file_path: Path to the code file to analyze
        
    Returns:
        List of security-focused optimization recommendations
        
    Example:
        >>> import asyncio
        >>> from core.optimization import security_analysis
        >>> security_recs = await security_analysis('webapp.py')
        >>> critical_vulns = [r for r in security_recs if r.priority == OptimizationPriority.CRITICAL]
    """
    security_optimizer = create_security_optimizer()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        import ast
        tree = ast.parse(code)
        
        return await security_optimizer.analyze_security(code, tree, file_path)
    
    except Exception as e:
        print(f"Error in security analysis: {e}")
        return []


async def performance_analysis(file_path: str) -> list:
    """
    Perform focused performance analysis on a single file
    
    Args:
        file_path: Path to the code file to analyze
        
    Returns:
        List of performance-focused optimization recommendations
        
    Example:
        >>> import asyncio
        >>> from core.optimization import performance_analysis
        >>> perf_recs = await performance_analysis('algorithms.py')
        >>> bottlenecks = [r for r in perf_recs if 'bottleneck' in r.description.lower()]
    """
    performance_analyzer = create_performance_analyzer()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        import ast
        tree = ast.parse(code)
        
        return await performance_analyzer.analyze_performance(code, tree, file_path)
    
    except Exception as e:
        print(f"Error in performance analysis: {e}")
        return []


async def quality_analysis(file_path: str) -> list:
    """
    Perform focused quality analysis on a single file
    
    Args:
        file_path: Path to the code file to analyze
        
    Returns:
        List of quality-focused optimization recommendations
        
    Example:
        >>> import asyncio
        >>> from core.optimization import quality_analysis
        >>> quality_recs = await quality_analysis('legacy_code.py')
        >>> complexity_issues = [r for r in quality_recs if 'complexity' in r.title.lower()]
    """
    quality_analyzer = create_quality_analyzer()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        import ast
        tree = ast.parse(code)
        
        return await quality_analyzer.analyze_quality(code, tree, file_path)
    
    except Exception as e:
        print(f"Error in quality analysis: {e}")
        return []


# Convenience factory for complete optimization suite
def create_optimization_suite(
    learning_enabled: bool = True,
    performance_focus: bool = True,
    security_focus: bool = True,
    quality_focus: bool = True
) -> dict:
    """
    Create a complete optimization analysis suite
    
    Args:
        learning_enabled: Enable machine learning insights
        performance_focus: Include performance analysis
        security_focus: Include security analysis
        quality_focus: Include quality analysis
        
    Returns:
        Dictionary containing all enabled analyzers
        
    Example:
        >>> from core.optimization import create_optimization_suite
        >>> suite = create_optimization_suite()
        >>> engine = suite['engine']
        >>> performance = suite.get('performance')
    """
    suite = {
        'engine': create_optimization_engine(learning_enabled)
    }
    
    if performance_focus:
        suite['performance'] = create_performance_analyzer()
    
    if security_focus:
        suite['security'] = create_security_optimizer()
    
    if quality_focus:
        suite['quality'] = create_quality_analyzer()
    
    return suite


# Module initialization
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())