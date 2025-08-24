#!/usr/bin/env python3
"""
TestMaster Module Contribution Analysis - Phase 2: Hours 46-50
Agent B - Documentation & Modularization Excellence

Comprehensive module contribution analysis system.
Assesses module importance, value, business impact, technical value,
and strategic significance across the TestMaster framework.
"""

import ast
import logging
import os
import json
import re
import inspect
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from datetime import datetime
import statistics
from collections import defaultdict, Counter
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModuleMetrics:
    """Comprehensive module metrics for contribution analysis."""
    module_id: str
    lines_of_code: int
    functions_count: int
    classes_count: int
    complexity_score: float
    documentation_coverage: float
    test_coverage_estimate: float
    dependency_fan_in: int
    dependency_fan_out: int
    api_surface_area: int
    maintainability_index: float

@dataclass
class BusinessValue:
    """Business value assessment for modules."""
    user_facing_features: int
    business_logic_density: float
    integration_points: int
    data_processing_capability: float
    automation_value: float
    innovation_factor: float
    market_differentiation: float
    cost_savings_potential: float

@dataclass
class TechnicalValue:
    """Technical value assessment for modules."""
    architectural_importance: float
    performance_impact: float
    security_contribution: float
    scalability_enablement: float
    maintainability_contribution: float
    reusability_factor: float
    technology_advancement: float
    infrastructure_value: float

@dataclass
class StrategicImportance:
    """Strategic importance assessment for modules."""
    competitive_advantage: float
    innovation_enablement: float
    future_extensibility: float
    platform_value: float
    ecosystem_integration: float
    intellectual_property_value: float
    market_positioning: float
    long_term_viability: float

@dataclass
class ModuleContribution:
    """Complete module contribution analysis."""
    module_id: str
    module_metrics: ModuleMetrics
    business_value: BusinessValue
    technical_value: TechnicalValue
    strategic_importance: StrategicImportance
    overall_contribution_score: float
    contribution_rank: int
    value_category: str  # critical, high, medium, low
    recommendations: List[str]
    investment_priority: int

@dataclass
class ContributionCluster:
    """Cluster of modules with related contributions."""
    cluster_name: str
    modules: List[str]
    collective_value: float
    synergy_factor: float
    business_impact: float
    technical_impact: float
    strategic_impact: float
    optimization_opportunities: List[str]

class ContributionAnalyzer:
    """Comprehensive module contribution analysis system."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.module_contributions: Dict[str, ModuleContribution] = {}
        self.contribution_clusters: List[ContributionCluster] = []
        self.module_contents: Dict[str, str] = {}
        self.contribution_metrics = {
            'total_modules_analyzed': 0,
            'critical_modules': 0,
            'high_value_modules': 0,
            'medium_value_modules': 0,
            'low_value_modules': 0,
            'average_contribution_score': 0.0,
            'business_value_distribution': {},
            'technical_value_distribution': {},
            'strategic_importance_distribution': {}
        }
        
    def analyze_all_contributions(self) -> Dict[str, Any]:
        """Perform comprehensive contribution analysis across the framework."""
        logger.info("🔍 Starting comprehensive module contribution analysis...")
        
        # Critical modules for contribution analysis
        critical_modules = [
            "core/intelligence/__init__.py",
            "core/intelligence/testing/__init__.py", 
            "core/intelligence/api/__init__.py",
            "core/intelligence/analytics/__init__.py",
            "testmaster_orchestrator.py",
            "intelligent_test_builder.py",
            "enhanced_self_healing_verifier.py",
            "agentic_test_monitor.py",
            "parallel_converter.py",
            "config/__init__.py"
        ]
        
        # Load module contents and analyze each module
        self._load_module_contents(critical_modules)
        
        for module_path in critical_modules:
            if module_path in self.module_contents:
                logger.info(f"Analyzing contribution: {module_path}")
                contribution = self._analyze_module_contribution(module_path)
                self.module_contributions[module_path] = contribution
        
        # Rank modules by contribution
        self._rank_module_contributions()
        
        # Identify contribution clusters
        self._identify_contribution_clusters()
        
        # Calculate contribution metrics
        self._calculate_contribution_metrics()
        
        return self._compile_contribution_analysis_results()
    
    def _load_module_contents(self, modules: List[str]):
        """Load content of all modules for contribution analysis."""
        for module_path in modules:
            try:
                full_path = self.base_path / module_path
                if full_path.exists():
                    with open(full_path, 'r', encoding='utf-8') as f:
                        self.module_contents[module_path] = f.read()
                    logger.info(f"Loaded module: {module_path}")
                else:
                    logger.warning(f"Module not found: {module_path}")
            except Exception as e:
                logger.error(f"Error loading module {module_path}: {e}")
    
    def _analyze_module_contribution(self, module_path: str) -> ModuleContribution:
        """Analyze a single module's contribution comprehensively."""
        content = self.module_contents[module_path]
        
        # Calculate module metrics
        metrics = self._calculate_module_metrics(module_path, content)
        
        # Assess business value
        business_value = self._assess_business_value(module_path, content)
        
        # Assess technical value
        technical_value = self._assess_technical_value(module_path, content)
        
        # Assess strategic importance
        strategic_importance = self._assess_strategic_importance(module_path, content)
        
        # Calculate overall contribution score
        overall_score = self._calculate_overall_contribution_score(
            business_value, technical_value, strategic_importance
        )
        
        # Generate recommendations
        recommendations = self._generate_module_recommendations(
            module_path, metrics, business_value, technical_value, strategic_importance
        )
        
        # Determine value category
        value_category = self._determine_value_category(overall_score)
        
        return ModuleContribution(
            module_id=module_path,
            module_metrics=metrics,
            business_value=business_value,
            technical_value=technical_value,
            strategic_importance=strategic_importance,
            overall_contribution_score=overall_score,
            contribution_rank=0,  # Will be set during ranking
            value_category=value_category,
            recommendations=recommendations,
            investment_priority=self._calculate_investment_priority(overall_score, metrics)
        )
    
    def _calculate_module_metrics(self, module_path: str, content: str) -> ModuleMetrics:
        """Calculate comprehensive module metrics."""
        lines = content.split('\n')
        lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        try:
            tree = ast.parse(content)
            functions_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            classes_count = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        except:
            functions_count = len(re.findall(r'def\s+\w+', content))
            classes_count = len(re.findall(r'class\s+\w+', content))
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(content)
        
        # Calculate documentation coverage
        doc_coverage = self._calculate_documentation_coverage(content)
        
        # Estimate test coverage
        test_coverage = self._estimate_test_coverage(module_path, content)
        
        # Calculate dependencies
        fan_in, fan_out = self._calculate_dependencies(module_path, content)
        
        # Calculate API surface area
        api_surface = self._calculate_api_surface_area(content)
        
        # Calculate maintainability index
        maintainability = self._calculate_maintainability_index(
            lines_of_code, complexity_score, doc_coverage
        )
        
        return ModuleMetrics(
            module_id=module_path,
            lines_of_code=lines_of_code,
            functions_count=functions_count,
            classes_count=classes_count,
            complexity_score=complexity_score,
            documentation_coverage=doc_coverage,
            test_coverage_estimate=test_coverage,
            dependency_fan_in=fan_in,
            dependency_fan_out=fan_out,
            api_surface_area=api_surface,
            maintainability_index=maintainability
        )
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate code complexity score."""
        complexity = 1.0  # Base complexity
        
        # Cyclomatic complexity indicators
        complexity += len(re.findall(r'\bif\b', content)) * 1.0
        complexity += len(re.findall(r'\bfor\b', content)) * 1.0
        complexity += len(re.findall(r'\bwhile\b', content)) * 1.0
        complexity += len(re.findall(r'\btry\b', content)) * 1.0
        complexity += len(re.findall(r'\bexcept\b', content)) * 1.0
        complexity += len(re.findall(r'\bwith\b', content)) * 0.5
        
        # Nested complexity
        nested_patterns = [
            r'if.*if', r'for.*for', r'try.*try', r'def.*def'
        ]
        for pattern in nested_patterns:
            complexity += len(re.findall(pattern, content, re.DOTALL)) * 2.0
        
        # Normalize by lines of code
        lines = len(content.split('\n'))
        return complexity / max(lines, 1) * 100
    
    def _calculate_documentation_coverage(self, content: str) -> float:
        """Calculate documentation coverage percentage."""
        # Count docstrings
        docstrings = len(re.findall(r'""".*?"""', content, re.DOTALL))
        docstrings += len(re.findall(r"'''.*?'''", content, re.DOTALL))
        
        # Count functions and classes that should have docstrings
        functions = len(re.findall(r'def\s+\w+', content))
        classes = len(re.findall(r'class\s+\w+', content))
        
        total_items = functions + classes
        if total_items == 0:
            return 1.0
        
        return min(docstrings / total_items, 1.0)
    
    def _estimate_test_coverage(self, module_path: str, content: str) -> float:
        """Estimate test coverage based on module characteristics."""
        # Look for test files
        test_indicators = 0
        
        # Check if it's a test module itself
        if 'test' in module_path.lower():
            return 1.0
        
        # Look for assertions and test-like patterns
        test_indicators += len(re.findall(r'assert\s+', content))
        test_indicators += len(re.findall(r'test_\w+', content))
        test_indicators += len(re.findall(r'@pytest\.|@unittest\.', content))
        
        # Estimate based on module type
        if 'intelligence' in module_path:
            base_coverage = 0.7  # Core modules likely well-tested
        elif 'config' in module_path:
            base_coverage = 0.5  # Configuration modules moderately tested
        else:
            base_coverage = 0.6  # Application modules reasonably tested
        
        # Adjust based on test indicators
        coverage_adjustment = min(test_indicators * 0.1, 0.3)
        return min(base_coverage + coverage_adjustment, 1.0)
    
    def _calculate_dependencies(self, module_path: str, content: str) -> Tuple[int, int]:
        """Calculate fan-in and fan-out dependencies."""
        # Fan-out: modules this module depends on
        imports = re.findall(r'(?:from\s+(\S+)\s+import|import\s+(\S+))', content)
        fan_out = len(set(imp[0] or imp[1] for imp in imports if imp[0] or imp[1]))
        
        # Fan-in: estimated based on module importance (simplified)
        if 'intelligence' in module_path and '__init__' in module_path:
            fan_in = 25  # Core intelligence modules are widely used
        elif 'orchestrator' in module_path:
            fan_in = 15  # Orchestrator is used by many components
        elif 'config' in module_path:
            fan_in = 20  # Configuration is widely used
        else:
            fan_in = 5  # Application modules have moderate usage
        
        return fan_in, fan_out
    
    def _calculate_api_surface_area(self, content: str) -> int:
        """Calculate the API surface area (public functions and classes)."""
        public_functions = len(re.findall(r'def\s+([a-zA-Z][a-zA-Z0-9_]*)\s*\(', content))
        public_classes = len(re.findall(r'class\s+([A-Z][a-zA-Z0-9_]*)', content))
        
        # Subtract private members (starting with _)
        private_functions = len(re.findall(r'def\s+_\w+', content))
        private_classes = len(re.findall(r'class\s+_\w+', content))
        
        return max(0, public_functions + public_classes - private_functions - private_classes)
    
    def _calculate_maintainability_index(self, loc: int, complexity: float, doc_coverage: float) -> float:
        """Calculate maintainability index (0-100 scale)."""
        # Based on Microsoft's maintainability index formula (simplified)
        volume = loc * math.log2(max(loc, 1)) if loc > 0 else 1
        mi = max(0, (171 - 5.2 * math.log(volume) - 0.23 * complexity + 16.2 * math.log(max(loc, 1))) * 100 / 171)
        
        # Adjust for documentation coverage
        mi_adjusted = mi * (0.7 + 0.3 * doc_coverage)
        
        return min(100, max(0, mi_adjusted))
    
    def _assess_business_value(self, module_path: str, content: str) -> BusinessValue:
        """Assess business value of the module."""
        # User-facing features
        user_features = 0
        if 'api' in module_path:
            user_features = 8  # API modules directly serve users
        elif 'intelligence' in module_path:
            user_features = 6  # Intelligence features provide user value
        elif 'builder' in module_path or 'orchestrator' in module_path:
            user_features = 7  # Core functionality users interact with
        else:
            user_features = 3  # Supporting functionality
        
        # Business logic density
        business_logic = self._calculate_business_logic_density(content)
        
        # Integration points
        integration_points = self._count_integration_points(content)
        
        # Data processing capability
        data_processing = self._assess_data_processing_capability(content)
        
        # Automation value
        automation_value = self._assess_automation_value(module_path, content)
        
        # Innovation factor
        innovation_factor = self._assess_innovation_factor(module_path, content)
        
        # Market differentiation
        market_diff = self._assess_market_differentiation(module_path)
        
        # Cost savings potential
        cost_savings = self._assess_cost_savings_potential(module_path, content)
        
        return BusinessValue(
            user_facing_features=user_features,
            business_logic_density=business_logic,
            integration_points=integration_points,
            data_processing_capability=data_processing,
            automation_value=automation_value,
            innovation_factor=innovation_factor,
            market_differentiation=market_diff,
            cost_savings_potential=cost_savings
        )
    
    def _calculate_business_logic_density(self, content: str) -> float:
        """Calculate density of business logic in the module."""
        business_indicators = [
            r'def\s+(create|process|analyze|generate|validate|transform)',
            r'class\s+\w*(Manager|Service|Handler|Processor)',
            r'business|logic|rule|workflow|process',
            r'calculate|compute|determine|decide'
        ]
        
        score = 0
        for pattern in business_indicators:
            score += len(re.findall(pattern, content, re.IGNORECASE))
        
        # Normalize by content length
        lines = len(content.split('\n'))
        return min(score / max(lines, 1) * 100, 1.0)
    
    def _count_integration_points(self, content: str) -> int:
        """Count integration points (APIs, external connections)."""
        integration_patterns = [
            r'@app\.route|@api\.|@endpoint',
            r'requests\.|urllib\.|httpx\.',
            r'import\s+(requests|urllib|httpx|aiohttp)',
            r'def\s+\w*api\w*|def\s+\w*endpoint\w*',
            r'async\s+def\s+\w*(get|post|put|delete)',
            r'\.json\(\)|\.serialize\(\)|\.deserialize\(\)'
        ]
        
        count = 0
        for pattern in integration_patterns:
            count += len(re.findall(pattern, content, re.IGNORECASE))
        
        return count
    
    def _assess_data_processing_capability(self, content: str) -> float:
        """Assess data processing and transformation capabilities."""
        data_patterns = [
            r'pandas|numpy|scipy|sklearn',
            r'json\.|yaml\.|csv\.|xml\.',
            r'parse|transform|convert|serialize|deserialize',
            r'data|dataset|dataframe|array|matrix',
            r'filter|map|reduce|aggregate|group'
        ]
        
        score = 0
        for pattern in data_patterns:
            score += len(re.findall(pattern, content, re.IGNORECASE))
        
        return min(score / 10.0, 1.0)
    
    def _assess_automation_value(self, module_path: str, content: str) -> float:
        """Assess automation value provided by the module."""
        automation_score = 0.0
        
        # Automated test generation
        if 'test' in module_path and 'builder' in module_path:
            automation_score += 0.9
        
        # Self-healing capabilities
        if 'healing' in module_path or 'verifier' in module_path:
            automation_score += 0.8
        
        # Orchestration and workflow automation
        if 'orchestrator' in module_path:
            automation_score += 0.85
        
        # Parallel processing automation
        if 'parallel' in module_path or 'concurrent' in content:
            automation_score += 0.7
        
        # Monitoring automation
        if 'monitor' in module_path:
            automation_score += 0.6
        
        # Configuration automation
        if 'config' in module_path:
            automation_score += 0.5
        
        return min(automation_score, 1.0)
    
    def _assess_innovation_factor(self, module_path: str, content: str) -> float:
        """Assess innovation and cutting-edge technology factor."""
        innovation_indicators = [
            r'ai|ml|machine\s+learning|artificial\s+intelligence',
            r'neural|network|deep\s+learning|llm|gpt',
            r'async|await|asyncio',
            r'self.*heal|adaptive|intelligent|smart',
            r'agentic|autonomous|auto.*generat',
            r'gemini|openai|anthropic'
        ]
        
        score = 0
        for pattern in innovation_indicators:
            score += len(re.findall(pattern, content, re.IGNORECASE))
        
        # Bonus for specific innovative modules
        if 'intelligence' in module_path:
            score += 5
        if 'agentic' in module_path:
            score += 4
        if 'self_healing' in module_path:
            score += 3
        
        return min(score / 10.0, 1.0)
    
    def _assess_market_differentiation(self, module_path: str) -> float:
        """Assess market differentiation potential of the module."""
        differentiation_factors = {
            'intelligence': 0.9,  # AI/ML intelligence is highly differentiating
            'agentic': 0.85,      # Autonomous agents are cutting-edge
            'self_healing': 0.8,   # Self-healing is innovative
            'orchestrator': 0.7,   # Advanced orchestration
            'parallel': 0.6,       # Parallel processing
            'analytics': 0.75,     # Advanced analytics
            'api': 0.5,           # APIs are standard
            'config': 0.3         # Configuration is commodity
        }
        
        score = 0.4  # Base score
        for keyword, factor in differentiation_factors.items():
            if keyword in module_path.lower():
                score = max(score, factor)
        
        return score
    
    def _assess_cost_savings_potential(self, module_path: str, content: str) -> float:
        """Assess potential for cost savings through automation."""
        savings_indicators = [
            r'automat|generat|build|create|process',
            r'parallel|concurrent|async|batch',
            r'optimi|effic|perform|speed|fast',
            r'reduc|minimi|eliminat|save',
            r'scale|scalab|elastic'
        ]
        
        score = 0
        for pattern in savings_indicators:
            score += len(re.findall(pattern, content, re.IGNORECASE))
        
        # Module-specific savings potential
        module_savings = {
            'orchestrator': 0.8,    # Workflow automation saves manual effort
            'parallel': 0.7,       # Parallel processing saves time
            'self_healing': 0.9,    # Self-healing reduces maintenance costs
            'test.*builder': 0.85,  # Automated test generation saves QA time
            'monitor': 0.6,         # Monitoring reduces debugging time
            'intelligence': 0.75    # AI intelligence reduces manual analysis
        }
        
        for pattern, savings in module_savings.items():
            if re.search(pattern, module_path, re.IGNORECASE):
                score += savings * 10
        
        return min(score / 20.0, 1.0)
    
    def _assess_technical_value(self, module_path: str, content: str) -> TechnicalValue:
        """Assess technical value of the module."""
        # Architectural importance
        arch_importance = self._assess_architectural_importance(module_path)
        
        # Performance impact
        performance_impact = self._assess_performance_impact(content)
        
        # Security contribution
        security_contrib = self._assess_security_contribution(content)
        
        # Scalability enablement
        scalability = self._assess_scalability_enablement(content)
        
        # Maintainability contribution
        maintainability = self._assess_maintainability_contribution(content)
        
        # Reusability factor
        reusability = self._assess_reusability_factor(module_path, content)
        
        # Technology advancement
        tech_advancement = self._assess_technology_advancement(content)
        
        # Infrastructure value
        infrastructure = self._assess_infrastructure_value(module_path)
        
        return TechnicalValue(
            architectural_importance=arch_importance,
            performance_impact=performance_impact,
            security_contribution=security_contrib,
            scalability_enablement=scalability,
            maintainability_contribution=maintainability,
            reusability_factor=reusability,
            technology_advancement=tech_advancement,
            infrastructure_value=infrastructure
        )
    
    def _assess_architectural_importance(self, module_path: str) -> float:
        """Assess architectural importance of the module."""
        importance_scores = {
            'core/intelligence/__init__.py': 1.0,  # Central hub
            'config/__init__.py': 0.9,             # Configuration foundation
            'testmaster_orchestrator.py': 0.85,    # Workflow orchestration
            'core/intelligence/api/__init__.py': 0.8,  # API layer
            'core/intelligence/testing/__init__.py': 0.8,  # Testing infrastructure
            'core/intelligence/analytics/__init__.py': 0.75,  # Analytics infrastructure
        }
        
        return importance_scores.get(module_path, 0.5)
    
    def _assess_performance_impact(self, content: str) -> float:
        """Assess performance impact of the module."""
        performance_indicators = [
            r'async|await|asyncio|concurrent|parallel',
            r'cache|caching|memoiz|optimi|performance',
            r'batch|bulk|stream|pipeline',
            r'thread|process|worker|pool',
            r'efficient|fast|speed|quick'
        ]
        
        score = 0
        for pattern in performance_indicators:
            score += len(re.findall(pattern, content, re.IGNORECASE))
        
        return min(score / 15.0, 1.0)
    
    def _assess_security_contribution(self, content: str) -> float:
        """Assess security contribution of the module."""
        security_indicators = [
            r'secur|auth|authent|authoriz|token',
            r'encrypt|decrypt|hash|sign|verify',
            r'sanitiz|validat|escape|clean',
            r'permission|access|role|privilege',
            r'audit|log|monitor|track'
        ]
        
        score = 0
        for pattern in security_indicators:
            score += len(re.findall(pattern, content, re.IGNORECASE))
        
        return min(score / 10.0, 1.0)
    
    def _assess_scalability_enablement(self, content: str) -> float:
        """Assess how the module enables scalability."""
        scalability_indicators = [
            r'scale|scalab|distribut|cluster',
            r'parallel|concurrent|async|thread',
            r'queue|pool|worker|batch',
            r'microservice|service|component',
            r'load|balance|partition|shard'
        ]
        
        score = 0
        for pattern in scalability_indicators:
            score += len(re.findall(pattern, content, re.IGNORECASE))
        
        return min(score / 12.0, 1.0)
    
    def _assess_maintainability_contribution(self, content: str) -> float:
        """Assess maintainability contribution of the module."""
        maintainability_indicators = [
            r'""".*?"""',  # Docstrings
            r'test|assert|mock|stub',
            r'log|debug|trace|monitor',
            r'config|setting|parameter',
            r'interface|abstract|protocol'
        ]
        
        score = 0
        for pattern in maintainability_indicators:
            score += len(re.findall(pattern, content, re.IGNORECASE | re.DOTALL))
        
        return min(score / 20.0, 1.0)
    
    def _assess_reusability_factor(self, module_path: str, content: str) -> float:
        """Assess reusability factor of the module."""
        reusability_score = 0.5  # Base score
        
        # High reusability indicators
        if 'core' in module_path:
            reusability_score += 0.3
        if 'intelligence' in module_path:
            reusability_score += 0.2
        if 'config' in module_path:
            reusability_score += 0.25
        
        # Look for generic patterns
        generic_patterns = [
            r'class\s+\w*(Base|Abstract|Interface)',
            r'def\s+(get|set|create|build|make)_\w+',
            r'@abstractmethod|@property|@classmethod|@staticmethod'
        ]
        
        for pattern in generic_patterns:
            reusability_score += len(re.findall(pattern, content)) * 0.05
        
        return min(reusability_score, 1.0)
    
    def _assess_technology_advancement(self, content: str) -> float:
        """Assess technology advancement level of the module."""
        advanced_tech_indicators = [
            r'ai|ml|machine\s+learning|neural|deep',
            r'async|await|asyncio|concurrent\.futures',
            r'typing|dataclass|protocol|generic',
            r'pathlib|contextlib|functools|itertools',
            r'pydantic|fastapi|starlette|uvicorn'
        ]
        
        score = 0
        for pattern in advanced_tech_indicators:
            score += len(re.findall(pattern, content, re.IGNORECASE))
        
        return min(score / 15.0, 1.0)
    
    def _assess_infrastructure_value(self, module_path: str) -> float:
        """Assess infrastructure value of the module."""
        infrastructure_scores = {
            'config': 0.9,           # Configuration infrastructure
            'intelligence': 0.8,     # Intelligence infrastructure
            'api': 0.7,             # API infrastructure
            'orchestrator': 0.75,    # Workflow infrastructure
            'monitor': 0.6          # Monitoring infrastructure
        }
        
        score = 0.3  # Base infrastructure value
        for keyword, value in infrastructure_scores.items():
            if keyword in module_path.lower():
                score = max(score, value)
        
        return score
    
    def _assess_strategic_importance(self, module_path: str, content: str) -> StrategicImportance:
        """Assess strategic importance of the module."""
        # Competitive advantage
        competitive_advantage = self._assess_competitive_advantage(module_path)
        
        # Innovation enablement
        innovation_enablement = self._assess_innovation_enablement(content)
        
        # Future extensibility
        future_extensibility = self._assess_future_extensibility(content)
        
        # Platform value
        platform_value = self._assess_platform_value(module_path)
        
        # Ecosystem integration
        ecosystem_integration = self._assess_ecosystem_integration(content)
        
        # Intellectual property value
        ip_value = self._assess_ip_value(module_path, content)
        
        # Market positioning
        market_positioning = self._assess_market_positioning(module_path)
        
        # Long-term viability
        long_term_viability = self._assess_long_term_viability(content)
        
        return StrategicImportance(
            competitive_advantage=competitive_advantage,
            innovation_enablement=innovation_enablement,
            future_extensibility=future_extensibility,
            platform_value=platform_value,
            ecosystem_integration=ecosystem_integration,
            intellectual_property_value=ip_value,
            market_positioning=market_positioning,
            long_term_viability=long_term_viability
        )
    
    def _assess_competitive_advantage(self, module_path: str) -> float:
        """Assess competitive advantage provided by the module."""
        advantage_scores = {
            'intelligence': 0.95,    # AI intelligence is highly competitive
            'agentic': 0.9,         # Autonomous agents are cutting-edge
            'self_healing': 0.85,    # Self-healing is innovative
            'orchestrator': 0.7,     # Advanced orchestration
            'analytics': 0.75,       # Advanced analytics
            'api': 0.4,             # APIs are commoditized
            'config': 0.3           # Configuration is standard
        }
        
        score = 0.3  # Base competitive advantage
        for keyword, value in advantage_scores.items():
            if keyword in module_path.lower():
                score = max(score, value)
        
        return score
    
    def _assess_innovation_enablement(self, content: str) -> float:
        """Assess how the module enables innovation."""
        innovation_patterns = [
            r'extensib|plugin|hook|callback',
            r'interface|abstract|protocol',
            r'factory|builder|strategy|observer',
            r'configur|parameter|setting|option',
            r'ai|ml|intelligent|adaptive|learning'
        ]
        
        score = 0
        for pattern in innovation_patterns:
            score += len(re.findall(pattern, content, re.IGNORECASE))
        
        return min(score / 15.0, 1.0)
    
    def _assess_future_extensibility(self, content: str) -> float:
        """Assess future extensibility of the module."""
        extensibility_indicators = [
            r'class\s+\w*(Base|Abstract|Interface|Protocol)',
            r'@abstractmethod|@property|@classmethod',
            r'plugin|extension|hook|callback|event',
            r'configur|parameter|setting|option',
            r'factory|builder|registry|manager'
        ]
        
        score = 0
        for pattern in extensibility_indicators:
            score += len(re.findall(pattern, content, re.IGNORECASE))
        
        return min(score / 20.0, 1.0)
    
    def _assess_platform_value(self, module_path: str) -> float:
        """Assess platform value of the module."""
        platform_scores = {
            'core/intelligence': 0.9,    # Core platform component
            'api': 0.8,                  # API platform layer
            'orchestrator': 0.75,        # Orchestration platform
            'config': 0.7,              # Configuration platform
            'testing': 0.65             # Testing platform
        }
        
        score = 0.3  # Base platform value
        for keyword, value in platform_scores.items():
            if keyword in module_path.lower():
                score = max(score, value)
        
        return score
    
    def _assess_ecosystem_integration(self, content: str) -> float:
        """Assess ecosystem integration capabilities."""
        ecosystem_indicators = [
            r'import\s+(requests|urllib|httpx|aiohttp)',
            r'api|endpoint|service|client|sdk',
            r'json|yaml|xml|csv|protocol|format',
            r'auth|oauth|token|key|secret',
            r'webhook|callback|event|notification'
        ]
        
        score = 0
        for pattern in ecosystem_indicators:
            score += len(re.findall(pattern, content, re.IGNORECASE))
        
        return min(score / 12.0, 1.0)
    
    def _assess_ip_value(self, module_path: str, content: str) -> float:
        """Assess intellectual property value."""
        ip_indicators = [
            r'algorithm|heuristic|model|intelligence',
            r'ai|ml|neural|learning|adaptive',
            r'pattern|strategy|approach|method',
            r'optimi|efficien|performance|smart',
            r'innovation|novel|unique|proprietary'
        ]
        
        score = 0
        for pattern in ip_indicators:
            score += len(re.findall(pattern, content, re.IGNORECASE))
        
        # Bonus for specific high-IP modules
        if 'intelligence' in module_path:
            score += 10
        if 'agentic' in module_path:
            score += 8
        if 'self_healing' in module_path:
            score += 6
        
        return min(score / 25.0, 1.0)
    
    def _assess_market_positioning(self, module_path: str) -> float:
        """Assess market positioning value."""
        positioning_scores = {
            'intelligence': 0.9,     # AI/ML positioning
            'agentic': 0.85,        # Autonomous systems positioning
            'self_healing': 0.8,     # Self-healing positioning
            'orchestrator': 0.7,     # Workflow automation positioning
            'analytics': 0.75,       # Analytics positioning
            'api': 0.5,             # Standard API positioning
            'config': 0.3           # Basic infrastructure positioning
        }
        
        score = 0.3  # Base market positioning
        for keyword, value in positioning_scores.items():
            if keyword in module_path.lower():
                score = max(score, value)
        
        return score
    
    def _assess_long_term_viability(self, content: str) -> float:
        """Assess long-term viability of the module."""
        viability_indicators = [
            r'standard|protocol|interface|spec',
            r'maintai|support|updat|evolv',
            r'test|coverage|quality|reliab',
            r'document|comment|docstring',
            r'version|compatib|migration|legacy'
        ]
        
        score = 0
        for pattern in viability_indicators:
            score += len(re.findall(pattern, content, re.IGNORECASE))
        
        return min(score / 15.0, 1.0)
    
    def _calculate_overall_contribution_score(self, business: BusinessValue, 
                                            technical: TechnicalValue, 
                                            strategic: StrategicImportance) -> float:
        """Calculate overall contribution score (0-100)."""
        # Weighted combination of all value dimensions
        business_score = (
            business.user_facing_features * 0.15 +
            business.business_logic_density * 0.1 +
            business.integration_points * 0.1 +
            business.data_processing_capability * 0.1 +
            business.automation_value * 0.2 +
            business.innovation_factor * 0.15 +
            business.market_differentiation * 0.1 +
            business.cost_savings_potential * 0.1
        ) / 8.0 * 10  # Scale to 0-10
        
        technical_score = (
            technical.architectural_importance +
            technical.performance_impact +
            technical.security_contribution +
            technical.scalability_enablement +
            technical.maintainability_contribution +
            technical.reusability_factor +
            technical.technology_advancement +
            technical.infrastructure_value
        ) / 8.0 * 10  # Scale to 0-10
        
        strategic_score = (
            strategic.competitive_advantage +
            strategic.innovation_enablement +
            strategic.future_extensibility +
            strategic.platform_value +
            strategic.ecosystem_integration +
            strategic.intellectual_property_value +
            strategic.market_positioning +
            strategic.long_term_viability
        ) / 8.0 * 10  # Scale to 0-10
        
        # Weighted overall score
        overall = (business_score * 0.4 + technical_score * 0.35 + strategic_score * 0.25) * 10
        return min(100, max(0, overall))
    
    def _determine_value_category(self, score: float) -> str:
        """Determine value category based on contribution score."""
        if score >= 80:
            return "critical"
        elif score >= 65:
            return "high"
        elif score >= 45:
            return "medium"
        else:
            return "low"
    
    def _calculate_investment_priority(self, score: float, metrics: ModuleMetrics) -> int:
        """Calculate investment priority (1-10, 10 being highest)."""
        base_priority = int(score / 10)  # 0-10 based on contribution score
        
        # Adjust based on maintainability
        if metrics.maintainability_index < 50:
            base_priority -= 1  # Lower priority for hard-to-maintain modules
        elif metrics.maintainability_index > 80:
            base_priority += 1  # Higher priority for well-maintained modules
        
        # Adjust based on technical debt indicators
        if metrics.complexity_score > 15:
            base_priority -= 1  # Lower priority for overly complex modules
        
        if metrics.documentation_coverage < 0.5:
            base_priority -= 1  # Lower priority for poorly documented modules
        
        return max(1, min(10, base_priority))
    
    def _generate_module_recommendations(self, module_path: str, metrics: ModuleMetrics,
                                       business: BusinessValue, technical: TechnicalValue,
                                       strategic: StrategicImportance) -> List[str]:
        """Generate specific recommendations for the module."""
        recommendations = []
        
        # Documentation recommendations
        if metrics.documentation_coverage < 0.7:
            recommendations.append("Improve documentation coverage with comprehensive docstrings")
        
        # Complexity recommendations
        if metrics.complexity_score > 15:
            recommendations.append("Refactor to reduce complexity and improve maintainability")
        
        # Test coverage recommendations
        if metrics.test_coverage_estimate < 0.8:
            recommendations.append("Increase test coverage for better reliability")
        
        # Business value recommendations
        if business.automation_value < 0.6:
            recommendations.append("Enhance automation capabilities to increase business value")
        
        # Technical value recommendations
        if technical.performance_impact < 0.5:
            recommendations.append("Optimize performance to increase technical value")
        
        # Strategic recommendations
        if strategic.future_extensibility < 0.6:
            recommendations.append("Improve extensibility for future enhancements")
        
        # Module-specific recommendations
        if 'intelligence' in module_path:
            recommendations.append("Leverage AI/ML capabilities for competitive advantage")
        
        if 'api' in module_path:
            recommendations.append("Enhance API design and documentation for better developer experience")
        
        if 'config' in module_path:
            recommendations.append("Implement configuration validation and better error handling")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _rank_module_contributions(self):
        """Rank modules by their contribution scores."""
        sorted_contributions = sorted(
            self.module_contributions.items(),
            key=lambda x: x[1].overall_contribution_score,
            reverse=True
        )
        
        for rank, (module_id, contribution) in enumerate(sorted_contributions, 1):
            contribution.contribution_rank = rank
    
    def _identify_contribution_clusters(self):
        """Identify clusters of modules with related contributions."""
        # Intelligence Core Cluster
        intelligence_cluster = self._create_intelligence_cluster()
        if intelligence_cluster:
            self.contribution_clusters.append(intelligence_cluster)
        
        # Application Layer Cluster
        application_cluster = self._create_application_cluster()
        if application_cluster:
            self.contribution_clusters.append(application_cluster)
        
        # Infrastructure Cluster
        infrastructure_cluster = self._create_infrastructure_cluster()
        if infrastructure_cluster:
            self.contribution_clusters.append(infrastructure_cluster)
    
    def _create_intelligence_cluster(self) -> Optional[ContributionCluster]:
        """Create intelligence framework cluster."""
        intelligence_modules = [
            module_id for module_id in self.module_contributions.keys()
            if 'intelligence' in module_id.lower()
        ]
        
        if len(intelligence_modules) >= 2:
            # Calculate collective metrics
            collective_value = statistics.mean([
                self.module_contributions[mod].overall_contribution_score
                for mod in intelligence_modules
            ])
            
            business_impact = statistics.mean([
                (self.module_contributions[mod].business_value.automation_value +
                 self.module_contributions[mod].business_value.innovation_factor) / 2
                for mod in intelligence_modules
            ])
            
            technical_impact = statistics.mean([
                (self.module_contributions[mod].technical_value.architectural_importance +
                 self.module_contributions[mod].technical_value.technology_advancement) / 2
                for mod in intelligence_modules
            ])
            
            strategic_impact = statistics.mean([
                (self.module_contributions[mod].strategic_importance.competitive_advantage +
                 self.module_contributions[mod].strategic_importance.innovation_enablement) / 2
                for mod in intelligence_modules
            ])
            
            return ContributionCluster(
                cluster_name="Intelligence Framework Core",
                modules=intelligence_modules,
                collective_value=collective_value,
                synergy_factor=0.85,
                business_impact=business_impact,
                technical_impact=technical_impact,
                strategic_impact=strategic_impact,
                optimization_opportunities=[
                    "Enhance cross-intelligence module integration",
                    "Standardize intelligence interfaces",
                    "Implement shared intelligence capabilities"
                ]
            )
        return None
    
    def _create_application_cluster(self) -> Optional[ContributionCluster]:
        """Create application layer cluster."""
        application_modules = [
            module_id for module_id in self.module_contributions.keys()
            if any(term in module_id.lower() for term in 
                  ['orchestrator', 'builder', 'verifier', 'converter', 'monitor'])
        ]
        
        if len(application_modules) >= 2:
            collective_value = statistics.mean([
                self.module_contributions[mod].overall_contribution_score
                for mod in application_modules
            ])
            
            return ContributionCluster(
                cluster_name="Application Processing Layer",
                modules=application_modules,
                collective_value=collective_value,
                synergy_factor=0.75,
                business_impact=0.7,
                technical_impact=0.6,
                strategic_impact=0.65,
                optimization_opportunities=[
                    "Improve inter-application coordination",
                    "Standardize processing interfaces",
                    "Enhanced error handling and recovery"
                ]
            )
        return None
    
    def _create_infrastructure_cluster(self) -> Optional[ContributionCluster]:
        """Create infrastructure cluster."""
        infrastructure_modules = [
            module_id for module_id in self.module_contributions.keys()
            if any(term in module_id.lower() for term in ['config', 'api'])
        ]
        
        if len(infrastructure_modules) >= 1:
            collective_value = statistics.mean([
                self.module_contributions[mod].overall_contribution_score
                for mod in infrastructure_modules
            ])
            
            return ContributionCluster(
                cluster_name="Infrastructure Foundation",
                modules=infrastructure_modules,
                collective_value=collective_value,
                synergy_factor=0.8,
                business_impact=0.5,
                technical_impact=0.85,
                strategic_impact=0.7,
                optimization_opportunities=[
                    "Enhance infrastructure reliability",
                    "Improve configuration management",
                    "Strengthen API design and documentation"
                ]
            )
        return None
    
    def _calculate_contribution_metrics(self):
        """Calculate comprehensive contribution metrics."""
        self.contribution_metrics['total_modules_analyzed'] = len(self.module_contributions)
        
        # Count modules by value category
        categories = defaultdict(int)
        scores = []
        
        for contribution in self.module_contributions.values():
            categories[contribution.value_category] += 1
            scores.append(contribution.overall_contribution_score)
        
        self.contribution_metrics['critical_modules'] = categories['critical']
        self.contribution_metrics['high_value_modules'] = categories['high']
        self.contribution_metrics['medium_value_modules'] = categories['medium']
        self.contribution_metrics['low_value_modules'] = categories['low']
        
        if scores:
            self.contribution_metrics['average_contribution_score'] = statistics.mean(scores)
        
        # Value distribution analysis
        business_values = []
        technical_values = []
        strategic_values = []
        
        for contribution in self.module_contributions.values():
            bv = contribution.business_value
            tv = contribution.technical_value
            sv = contribution.strategic_importance
            
            business_avg = (bv.automation_value + bv.innovation_factor + bv.market_differentiation) / 3
            technical_avg = (tv.architectural_importance + tv.performance_impact + tv.technology_advancement) / 3
            strategic_avg = (sv.competitive_advantage + sv.innovation_enablement + sv.platform_value) / 3
            
            business_values.append(business_avg)
            technical_values.append(technical_avg)
            strategic_values.append(strategic_avg)
        
        if business_values:
            self.contribution_metrics['business_value_distribution'] = {
                'mean': statistics.mean(business_values),
                'min': min(business_values),
                'max': max(business_values),
                'std': statistics.stdev(business_values) if len(business_values) > 1 else 0
            }
        
        if technical_values:
            self.contribution_metrics['technical_value_distribution'] = {
                'mean': statistics.mean(technical_values),
                'min': min(technical_values),
                'max': max(technical_values),
                'std': statistics.stdev(technical_values) if len(technical_values) > 1 else 0
            }
        
        if strategic_values:
            self.contribution_metrics['strategic_importance_distribution'] = {
                'mean': statistics.mean(strategic_values),
                'min': min(strategic_values),
                'max': max(strategic_values),
                'std': statistics.stdev(strategic_values) if len(strategic_values) > 1 else 0
            }
    
    def _compile_contribution_analysis_results(self) -> Dict[str, Any]:
        """Compile comprehensive contribution analysis results."""
        return {
            "analysis_metadata": {
                "analyzer": "Agent B - Module Contribution Analysis",
                "phase": "Hours 46-50",
                "modules_analyzed": len(self.module_contributions),
                "total_contribution_score": sum(c.overall_contribution_score for c in self.module_contributions.values())
            },
            "module_contributions": {
                module_id: {
                    "contribution_score": contribution.overall_contribution_score,
                    "contribution_rank": contribution.contribution_rank,
                    "value_category": contribution.value_category,
                    "investment_priority": contribution.investment_priority,
                    "business_value_summary": {
                        "automation_value": contribution.business_value.automation_value,
                        "innovation_factor": contribution.business_value.innovation_factor,
                        "market_differentiation": contribution.business_value.market_differentiation,
                        "cost_savings_potential": contribution.business_value.cost_savings_potential
                    },
                    "technical_value_summary": {
                        "architectural_importance": contribution.technical_value.architectural_importance,
                        "performance_impact": contribution.technical_value.performance_impact,
                        "technology_advancement": contribution.technical_value.technology_advancement,
                        "infrastructure_value": contribution.technical_value.infrastructure_value
                    },
                    "strategic_importance_summary": {
                        "competitive_advantage": contribution.strategic_importance.competitive_advantage,
                        "innovation_enablement": contribution.strategic_importance.innovation_enablement,
                        "platform_value": contribution.strategic_importance.platform_value,
                        "long_term_viability": contribution.strategic_importance.long_term_viability
                    },
                    "module_metrics": {
                        "lines_of_code": contribution.module_metrics.lines_of_code,
                        "complexity_score": contribution.module_metrics.complexity_score,
                        "maintainability_index": contribution.module_metrics.maintainability_index,
                        "api_surface_area": contribution.module_metrics.api_surface_area
                    },
                    "recommendations": contribution.recommendations
                }
                for module_id, contribution in self.module_contributions.items()
            },
            "contribution_clusters": [
                {
                    "name": cluster.cluster_name,
                    "modules": cluster.modules,
                    "collective_value": cluster.collective_value,
                    "synergy_factor": cluster.synergy_factor,
                    "business_impact": cluster.business_impact,
                    "technical_impact": cluster.technical_impact,
                    "strategic_impact": cluster.strategic_impact,
                    "optimization_opportunities": cluster.optimization_opportunities
                }
                for cluster in self.contribution_clusters
            ],
            "contribution_metrics": self.contribution_metrics,
            "investment_recommendations": self._generate_investment_recommendations()
        }
    
    def _generate_investment_recommendations(self) -> List[Dict[str, Any]]:
        """Generate investment recommendations based on contribution analysis."""
        recommendations = []
        
        # High-value modules needing investment
        high_value_modules = [
            (module_id, contrib) for module_id, contrib in self.module_contributions.items()
            if contrib.value_category in ['critical', 'high'] and contrib.investment_priority >= 7
        ]
        
        if high_value_modules:
            recommendations.append({
                "category": "high_value_investment",
                "priority": "high",
                "description": f"Invest in {len(high_value_modules)} high-value modules",
                "modules": [module_id for module_id, _ in high_value_modules],
                "actions": [
                    "Enhance documentation and testing",
                    "Optimize performance and scalability",
                    "Strengthen competitive advantages"
                ]
            })
        
        # Innovation opportunities
        innovation_modules = [
            module_id for module_id, contrib in self.module_contributions.items()
            if contrib.strategic_importance.innovation_enablement > 0.7
        ]
        
        if innovation_modules:
            recommendations.append({
                "category": "innovation_investment",
                "priority": "medium",
                "description": f"Leverage {len(innovation_modules)} modules for innovation",
                "modules": innovation_modules,
                "actions": [
                    "Expand AI/ML capabilities",
                    "Implement cutting-edge features",
                    "Enhance automation capabilities"
                ]
            })
        
        # Technical debt reduction
        debt_modules = [
            module_id for module_id, contrib in self.module_contributions.items()
            if contrib.module_metrics.maintainability_index < 60
        ]
        
        if debt_modules:
            recommendations.append({
                "category": "technical_debt_reduction",
                "priority": "medium",
                "description": f"Address technical debt in {len(debt_modules)} modules",
                "modules": debt_modules,
                "actions": [
                    "Refactor complex code",
                    "Improve documentation",
                    "Enhance test coverage"
                ]
            })
        
        return recommendations
    
    def export_contribution_analysis(self, output_file: str):
        """Export comprehensive contribution analysis results."""
        results = self._compile_contribution_analysis_results()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Contribution analysis results exported to {output_file}")

def main():
    """Run comprehensive module contribution analysis."""
    analyzer = ContributionAnalyzer()
    
    logger.info("Starting Agent B Phase 2 Hours 46-50: Module Contribution Analysis")
    
    # Perform comprehensive contribution analysis
    results = analyzer.analyze_all_contributions()
    
    # Export detailed results
    analyzer.export_contribution_analysis("contribution_analysis_results.json")
    
    # Print summary
    print(f"""
Module Contribution Analysis Complete!

Analysis Summary:
├── Modules Analyzed: {results['analysis_metadata']['modules_analyzed']}
├── Critical Modules: {results['contribution_metrics'].get('critical_modules', 0)}
├── High Value Modules: {results['contribution_metrics'].get('high_value_modules', 0)}
├── Average Contribution Score: {results['contribution_metrics'].get('average_contribution_score', 0.0):.1f}
└── Investment Recommendations: {len(results.get('investment_recommendations', []))}

Contribution analysis results saved to contribution_analysis_results.json
""")

if __name__ == "__main__":
    main()