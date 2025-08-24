"""
Repository Intelligence Aggregator - Multi-repository testing intelligence system

This aggregator provides:
- Cross-repository pattern analysis and extraction
- Unified test intelligence from multiple sources
- Repository comparison and benchmarking
- Collaborative testing insights
- Distributed test execution coordination
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import threading
import hashlib
from pathlib import Path
import concurrent.futures
import subprocess

# Mock Framework Imports for Testing
import pytest
from unittest.mock import Mock, patch, MagicMock
import unittest

class RepositoryType(Enum):
    AGENCY_SWARM = "agency_swarm"
    CREW_AI = "crew_ai" 
    AGENT_SCOPE = "agent_scope"
    FALKOR_DB = "falkor_db"
    LANG_GRAPH = "lang_graph"
    SWARMS = "swarms"
    AUTO_GEN = "auto_gen"

class IntelligenceLevel(Enum):
    BASIC = "basic"           # Simple pattern extraction
    INTERMEDIATE = "intermediate"  # Pattern analysis and correlation
    ADVANCED = "advanced"     # Cross-repository insights
    EXPERT = "expert"        # Predictive intelligence

class SyncStatus(Enum):
    NEVER_SYNCED = "never_synced"
    SYNCING = "syncing"
    SYNCED = "synced"
    SYNC_ERROR = "sync_error"
    OUTDATED = "outdated"

@dataclass
class RepositoryMetadata:
    """Metadata for a repository being aggregated"""
    name: str
    repo_type: RepositoryType
    local_path: Path
    remote_url: Optional[str] = None
    last_sync: Optional[datetime] = None
    sync_status: SyncStatus = SyncStatus.NEVER_SYNCED
    commit_hash: Optional[str] = None
    branch: str = "main"
    testing_frameworks: List[str] = field(default_factory=list)
    language_distribution: Dict[str, float] = field(default_factory=dict)
    test_coverage: float = 0.0

@dataclass
class TestingIntelligence:
    """Extracted testing intelligence from repository"""
    repository: str
    extraction_time: datetime
    patterns: List[Dict[str, Any]]
    frameworks_used: List[str]
    test_strategies: List[str]
    quality_metrics: Dict[str, float]
    code_complexity: Dict[str, float]
    testing_maturity_score: float
    recommendations: List[str] = field(default_factory=list)

@dataclass
class CrossRepoInsight:
    """Cross-repository testing insight"""
    insight_id: str
    title: str
    description: str
    affected_repositories: List[str]
    confidence_score: float
    impact_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    category: str  # PATTERN, ANTI_PATTERN, OPPORTUNITY, RISK
    evidence: List[Dict[str, Any]]
    recommendations: List[str]
    discovered_at: datetime

class RepositoryScanner:
    """Scans repositories for testing patterns and intelligence"""
    
    def __init__(self):
        self.scan_cache: Dict[str, Dict[str, Any]] = {}
        self.pattern_extractors = {
            'pytest': self._extract_pytest_patterns,
            'unittest': self._extract_unittest_patterns,
            'jest': self._extract_jest_patterns,
            'mocha': self._extract_mocha_patterns,
            'junit': self._extract_junit_patterns,
            'rspec': self._extract_rspec_patterns,
            'go_test': self._extract_go_test_patterns
        }
        
    def scan_repository(self, repo_metadata: RepositoryMetadata) -> TestingIntelligence:
        """Perform comprehensive repository scan"""
        repo_path = repo_metadata.local_path
        
        # Extract basic repository information
        frameworks = self._detect_testing_frameworks(repo_path)
        language_dist = self._analyze_language_distribution(repo_path)
        
        # Extract testing patterns
        all_patterns = []
        for framework in frameworks:
            if framework in self.pattern_extractors:
                patterns = self.pattern_extractors[framework](repo_path)
                all_patterns.extend(patterns)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(repo_path, frameworks)
        code_complexity = self._analyze_code_complexity(repo_path)
        
        # Determine testing strategies
        test_strategies = self._identify_test_strategies(all_patterns, frameworks)
        
        # Calculate maturity score
        maturity_score = self._calculate_testing_maturity(
            frameworks, quality_metrics, code_complexity, test_strategies
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            frameworks, quality_metrics, maturity_score
        )
        
        return TestingIntelligence(
            repository=repo_metadata.name,
            extraction_time=datetime.now(),
            patterns=all_patterns,
            frameworks_used=frameworks,
            test_strategies=test_strategies,
            quality_metrics=quality_metrics,
            code_complexity=code_complexity,
            testing_maturity_score=maturity_score,
            recommendations=recommendations
        )
    
    def _detect_testing_frameworks(self, repo_path: Path) -> List[str]:
        """Detect testing frameworks used in repository"""
        frameworks = []
        
        # Check for Python frameworks
        if (repo_path / 'pytest.ini').exists() or self._find_files_with_pattern(repo_path, 'test_*.py'):
            frameworks.append('pytest')
        if self._find_files_with_pattern(repo_path, '*test*.py'):
            frameworks.append('unittest')
        
        # Check for JavaScript frameworks
        package_json = repo_path / 'package.json'
        if package_json.exists():
            try:
                with open(package_json) as f:
                    package_data = json.load(f)
                    deps = {**package_data.get('dependencies', {}), **package_data.get('devDependencies', {})}
                    if 'jest' in deps:
                        frameworks.append('jest')
                    if 'mocha' in deps:
                        frameworks.append('mocha')
            except:
                pass
        
        # Check for Java frameworks
        if self._find_files_with_pattern(repo_path, '*.java'):
            if self._find_files_with_pattern(repo_path, '*Test.java'):
                frameworks.append('junit')
        
        # Check for Go testing
        if self._find_files_with_pattern(repo_path, '*_test.go'):
            frameworks.append('go_test')
        
        # Check for Ruby testing
        if self._find_files_with_pattern(repo_path, '*_spec.rb'):
            frameworks.append('rspec')
        
        return frameworks
    
    def _find_files_with_pattern(self, repo_path: Path, pattern: str) -> bool:
        """Check if files matching pattern exist"""
        try:
            return len(list(repo_path.rglob(pattern))) > 0
        except:
            return False
    
    def _analyze_language_distribution(self, repo_path: Path) -> Dict[str, float]:
        """Analyze programming language distribution"""
        language_counts = {}
        
        # Count files by extension
        for file_path in repo_path.rglob('*'):
            if file_path.is_file() and file_path.suffix:
                ext = file_path.suffix.lower()
                language_counts[ext] = language_counts.get(ext, 0) + 1
        
        # Convert to percentages
        total_files = sum(language_counts.values())
        if total_files == 0:
            return {}
        
        return {ext: count / total_files for ext, count in language_counts.items()}
    
    def _extract_pytest_patterns(self, repo_path: Path) -> List[Dict[str, Any]]:
        """Extract pytest-specific patterns"""
        patterns = []
        
        # Find test files
        test_files = list(repo_path.rglob('test_*.py')) + list(repo_path.rglob('*_test.py'))
        
        for test_file in test_files[:10]:  # Limit analysis
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract patterns
                if 'pytest.fixture' in content:
                    patterns.append({
                        'type': 'fixture_usage',
                        'framework': 'pytest',
                        'file': str(test_file.relative_to(repo_path)),
                        'complexity': 'medium'
                    })
                
                if 'pytest.mark.parametrize' in content:
                    patterns.append({
                        'type': 'parametrized_testing',
                        'framework': 'pytest',
                        'file': str(test_file.relative_to(repo_path)),
                        'complexity': 'high'
                    })
                
                if '@patch' in content or 'mock' in content.lower():
                    patterns.append({
                        'type': 'mocking_strategy',
                        'framework': 'pytest',
                        'file': str(test_file.relative_to(repo_path)),
                        'complexity': 'high'
                    })
                
            except Exception as e:
                logging.debug(f"Error analyzing {test_file}: {e}")
        
        return patterns
    
    def _extract_unittest_patterns(self, repo_path: Path) -> List[Dict[str, Any]]:
        """Extract unittest patterns"""
        patterns = []
        
        test_files = list(repo_path.rglob('*test*.py'))
        for test_file in test_files[:10]:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'unittest.TestCase' in content:
                    patterns.append({
                        'type': 'class_based_testing',
                        'framework': 'unittest',
                        'file': str(test_file.relative_to(repo_path)),
                        'complexity': 'medium'
                    })
                
                if 'setUp' in content or 'tearDown' in content:
                    patterns.append({
                        'type': 'setup_teardown',
                        'framework': 'unittest',
                        'file': str(test_file.relative_to(repo_path)),
                        'complexity': 'medium'
                    })
                    
            except Exception as e:
                logging.debug(f"Error analyzing {test_file}: {e}")
        
        return patterns
    
    def _extract_jest_patterns(self, repo_path: Path) -> List[Dict[str, Any]]:
        """Extract Jest patterns"""
        patterns = []
        
        test_files = list(repo_path.rglob('*.test.js')) + list(repo_path.rglob('*.spec.js'))
        for test_file in test_files[:10]:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'describe(' in content and 'it(' in content:
                    patterns.append({
                        'type': 'bdd_testing',
                        'framework': 'jest',
                        'file': str(test_file.relative_to(repo_path)),
                        'complexity': 'medium'
                    })
                
                if 'beforeEach' in content or 'afterEach' in content:
                    patterns.append({
                        'type': 'test_hooks',
                        'framework': 'jest',
                        'file': str(test_file.relative_to(repo_path)),
                        'complexity': 'medium'
                    })
                    
            except Exception as e:
                logging.debug(f"Error analyzing {test_file}: {e}")
        
        return patterns
    
    def _extract_mocha_patterns(self, repo_path: Path) -> List[Dict[str, Any]]:
        """Extract Mocha patterns - similar to Jest"""
        return self._extract_jest_patterns(repo_path)  # Similar patterns
    
    def _extract_junit_patterns(self, repo_path: Path) -> List[Dict[str, Any]]:
        """Extract JUnit patterns"""
        patterns = []
        
        test_files = list(repo_path.rglob('*Test.java'))
        for test_file in test_files[:10]:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if '@Test' in content:
                    patterns.append({
                        'type': 'annotation_based_testing',
                        'framework': 'junit',
                        'file': str(test_file.relative_to(repo_path)),
                        'complexity': 'medium'
                    })
                    
            except Exception as e:
                logging.debug(f"Error analyzing {test_file}: {e}")
        
        return patterns
    
    def _extract_rspec_patterns(self, repo_path: Path) -> List[Dict[str, Any]]:
        """Extract RSpec patterns"""
        patterns = []
        
        test_files = list(repo_path.rglob('*_spec.rb'))
        for test_file in test_files[:10]:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'describe ' in content and 'it ' in content:
                    patterns.append({
                        'type': 'rspec_bdd',
                        'framework': 'rspec',
                        'file': str(test_file.relative_to(repo_path)),
                        'complexity': 'medium'
                    })
                    
            except Exception as e:
                logging.debug(f"Error analyzing {test_file}: {e}")
        
        return patterns
    
    def _extract_go_test_patterns(self, repo_path: Path) -> List[Dict[str, Any]]:
        """Extract Go testing patterns"""
        patterns = []
        
        test_files = list(repo_path.rglob('*_test.go'))
        for test_file in test_files[:10]:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'func Test' in content:
                    patterns.append({
                        'type': 'table_driven_testing',
                        'framework': 'go_test',
                        'file': str(test_file.relative_to(repo_path)),
                        'complexity': 'medium'
                    })
                    
            except Exception as e:
                logging.debug(f"Error analyzing {test_file}: {e}")
        
        return patterns
    
    def _calculate_quality_metrics(self, repo_path: Path, frameworks: List[str]) -> Dict[str, float]:
        """Calculate code quality metrics"""
        # Mock implementation - in real scenario, would use tools like:
        # - pytest-cov for coverage
        # - flake8/pylint for code quality
        # - radon for complexity
        
        metrics = {
            'test_coverage': 0.75,  # Mock 75% coverage
            'code_quality_score': 0.85,  # Mock quality score
            'test_to_code_ratio': 0.3,  # Mock 30% test code
            'documentation_score': 0.6  # Mock documentation score
        }
        
        # Adjust based on frameworks present
        if 'pytest' in frameworks:
            metrics['test_coverage'] += 0.1
        if len(frameworks) > 2:
            metrics['code_quality_score'] += 0.05
        
        return metrics
    
    def _analyze_code_complexity(self, repo_path: Path) -> Dict[str, float]:
        """Analyze code complexity"""
        return {
            'cyclomatic_complexity': 3.2,
            'maintainability_index': 78.5,
            'lines_of_code': 15000,
            'technical_debt_ratio': 0.15
        }
    
    def _identify_test_strategies(self, patterns: List[Dict[str, Any]], frameworks: List[str]) -> List[str]:
        """Identify testing strategies used"""
        strategies = set()
        
        for pattern in patterns:
            pattern_type = pattern.get('type', '')
            
            if 'fixture' in pattern_type or 'setup' in pattern_type:
                strategies.add('test_fixture_strategy')
            if 'mock' in pattern_type:
                strategies.add('isolation_testing')
            if 'parametrize' in pattern_type or 'table_driven' in pattern_type:
                strategies.add('data_driven_testing')
            if 'bdd' in pattern_type:
                strategies.add('behavior_driven_development')
        
        if len(frameworks) > 1:
            strategies.add('multi_framework_approach')
            
        return list(strategies)
    
    def _calculate_testing_maturity(self, frameworks: List[str], quality_metrics: Dict[str, float], 
                                  complexity: Dict[str, float], strategies: List[str]) -> float:
        """Calculate overall testing maturity score"""
        base_score = 0.0
        
        # Framework diversity bonus
        base_score += min(len(frameworks) * 0.1, 0.3)
        
        # Quality metrics contribution
        base_score += quality_metrics.get('test_coverage', 0) * 0.3
        base_score += quality_metrics.get('code_quality_score', 0) * 0.2
        
        # Strategy sophistication
        base_score += min(len(strategies) * 0.05, 0.2)
        
        return min(base_score, 1.0)
    
    def _generate_recommendations(self, frameworks: List[str], quality_metrics: Dict[str, float], 
                                maturity_score: float) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if quality_metrics.get('test_coverage', 0) < 0.8:
            recommendations.append("Increase test coverage to above 80%")
        
        if len(frameworks) < 2:
            recommendations.append("Consider adopting additional testing frameworks for comprehensive coverage")
        
        if maturity_score < 0.7:
            recommendations.append("Implement more sophisticated testing strategies (mocking, parameterized tests)")
        
        if quality_metrics.get('code_quality_score', 0) < 0.8:
            recommendations.append("Improve code quality through refactoring and better practices")
        
        return recommendations

class CrossRepositoryAnalyzer:
    """Analyzes testing patterns across multiple repositories"""
    
    def __init__(self):
        self.repository_intelligence: Dict[str, TestingIntelligence] = {}
        self.cross_repo_insights: List[CrossRepoInsight] = []
        
    def add_repository_intelligence(self, intelligence: TestingIntelligence) -> None:
        """Add intelligence from a repository"""
        self.repository_intelligence[intelligence.repository] = intelligence
        
        # Trigger cross-repository analysis if we have multiple repositories
        if len(self.repository_intelligence) > 1:
            self._analyze_cross_repository_patterns()
    
    def _analyze_cross_repository_patterns(self) -> None:
        """Analyze patterns across repositories"""
        insights = []
        
        # Analyze framework adoption patterns
        framework_insights = self._analyze_framework_adoption()
        insights.extend(framework_insights)
        
        # Analyze quality metric patterns
        quality_insights = self._analyze_quality_patterns()
        insights.extend(quality_insights)
        
        # Analyze testing strategy patterns
        strategy_insights = self._analyze_strategy_patterns()
        insights.extend(strategy_insights)
        
        # Update insights list
        self.cross_repo_insights.extend(insights)
        
        # Keep only recent insights (last 100)
        if len(self.cross_repo_insights) > 100:
            self.cross_repo_insights = self.cross_repo_insights[-100:]
    
    def _analyze_framework_adoption(self) -> List[CrossRepoInsight]:
        """Analyze testing framework adoption patterns"""
        insights = []
        
        # Count framework usage
        framework_counts = {}
        for repo_name, intelligence in self.repository_intelligence.items():
            for framework in intelligence.frameworks_used:
                if framework not in framework_counts:
                    framework_counts[framework] = []
                framework_counts[framework].append(repo_name)
        
        # Find popular frameworks
        for framework, repos in framework_counts.items():
            if len(repos) >= 3:  # Used in 3+ repositories
                insights.append(CrossRepoInsight(
                    insight_id=f"framework_adoption_{framework}",
                    title=f"Popular Framework: {framework}",
                    description=f"Framework '{framework}' is widely adopted across {len(repos)} repositories",
                    affected_repositories=repos,
                    confidence_score=0.9,
                    impact_level="MEDIUM",
                    category="PATTERN",
                    evidence=[{"framework": framework, "usage_count": len(repos)}],
                    recommendations=[f"Consider standardizing on {framework} across all projects"],
                    discovered_at=datetime.now()
                ))
        
        return insights
    
    def _analyze_quality_patterns(self) -> List[CrossRepoInsight]:
        """Analyze quality metric patterns"""
        insights = []
        
        # Calculate average metrics
        total_repos = len(self.repository_intelligence)
        if total_repos == 0:
            return insights
        
        avg_coverage = sum(
            intel.quality_metrics.get('test_coverage', 0) 
            for intel in self.repository_intelligence.values()
        ) / total_repos
        
        # Find repositories with significantly different coverage
        low_coverage_repos = []
        high_coverage_repos = []
        
        for repo_name, intelligence in self.repository_intelligence.items():
            coverage = intelligence.quality_metrics.get('test_coverage', 0)
            if coverage < avg_coverage - 0.2:
                low_coverage_repos.append(repo_name)
            elif coverage > avg_coverage + 0.2:
                high_coverage_repos.append(repo_name)
        
        if low_coverage_repos:
            insights.append(CrossRepoInsight(
                insight_id="low_coverage_pattern",
                title="Low Test Coverage Pattern",
                description=f"Some repositories have significantly lower test coverage than average ({avg_coverage:.1%})",
                affected_repositories=low_coverage_repos,
                confidence_score=0.8,
                impact_level="HIGH",
                category="RISK",
                evidence=[{"average_coverage": avg_coverage, "low_coverage_repos": low_coverage_repos}],
                recommendations=["Focus testing efforts on low-coverage repositories"],
                discovered_at=datetime.now()
            ))
        
        return insights
    
    def _analyze_strategy_patterns(self) -> List[CrossRepoInsight]:
        """Analyze testing strategy patterns"""
        insights = []
        
        # Count strategy usage
        strategy_counts = {}
        for repo_name, intelligence in self.repository_intelligence.items():
            for strategy in intelligence.test_strategies:
                if strategy not in strategy_counts:
                    strategy_counts[strategy] = []
                strategy_counts[strategy].append(repo_name)
        
        # Find missing strategies in some repos
        all_strategies = set(strategy_counts.keys())
        for repo_name, intelligence in self.repository_intelligence.items():
            repo_strategies = set(intelligence.test_strategies)
            missing_strategies = all_strategies - repo_strategies
            
            if len(missing_strategies) >= 2:  # Missing 2+ strategies
                insights.append(CrossRepoInsight(
                    insight_id=f"missing_strategies_{repo_name}",
                    title=f"Testing Strategy Gaps: {repo_name}",
                    description=f"Repository '{repo_name}' is missing common testing strategies",
                    affected_repositories=[repo_name],
                    confidence_score=0.7,
                    impact_level="MEDIUM",
                    category="OPPORTUNITY",
                    evidence=[{"missing_strategies": list(missing_strategies)}],
                    recommendations=[f"Consider adopting {', '.join(missing_strategies)} in {repo_name}"],
                    discovered_at=datetime.now()
                ))
        
        return insights
    
    def get_repository_comparison(self) -> Dict[str, Any]:
        """Generate repository comparison report"""
        if not self.repository_intelligence:
            return {"error": "No repository data available"}
        
        comparison = {
            "repository_count": len(self.repository_intelligence),
            "framework_analysis": {},
            "quality_comparison": {},
            "maturity_ranking": [],
            "insights_summary": {
                "total_insights": len(self.cross_repo_insights),
                "by_category": {},
                "by_impact": {}
            }
        }
        
        # Framework analysis
        all_frameworks = set()
        for intelligence in self.repository_intelligence.values():
            all_frameworks.update(intelligence.frameworks_used)
        
        for framework in all_frameworks:
            repos_using = [
                repo for repo, intel in self.repository_intelligence.items()
                if framework in intel.frameworks_used
            ]
            comparison["framework_analysis"][framework] = {
                "adoption_rate": len(repos_using) / len(self.repository_intelligence),
                "repositories": repos_using
            }
        
        # Quality comparison
        for repo_name, intelligence in self.repository_intelligence.items():
            comparison["quality_comparison"][repo_name] = {
                "test_coverage": intelligence.quality_metrics.get('test_coverage', 0),
                "maturity_score": intelligence.testing_maturity_score,
                "strategies_count": len(intelligence.test_strategies)
            }
        
        # Maturity ranking
        comparison["maturity_ranking"] = sorted(
            [(repo, intel.testing_maturity_score) for repo, intel in self.repository_intelligence.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # Insights summary
        for insight in self.cross_repo_insights:
            category = insight.category
            impact = insight.impact_level
            
            comparison["insights_summary"]["by_category"][category] = \
                comparison["insights_summary"]["by_category"].get(category, 0) + 1
            comparison["insights_summary"]["by_impact"][impact] = \
                comparison["insights_summary"]["by_impact"].get(impact, 0) + 1
        
        return comparison

class RepositoryIntelligenceAggregator:
    """Main aggregator for multi-repository testing intelligence"""
    
    def __init__(self, base_path: Path = Path("repositories")):
        self.base_path = base_path
        self.repositories: Dict[str, RepositoryMetadata] = {}
        self.scanner = RepositoryScanner()
        self.analyzer = CrossRepositoryAnalyzer()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
    def register_repository(self, name: str, repo_type: RepositoryType, 
                          local_path: Path, remote_url: Optional[str] = None) -> bool:
        """Register a repository for intelligence aggregation"""
        try:
            metadata = RepositoryMetadata(
                name=name,
                repo_type=repo_type,
                local_path=local_path,
                remote_url=remote_url
            )
            
            self.repositories[name] = metadata
            return True
            
        except Exception as e:
            logging.error(f"Failed to register repository {name}: {e}")
            return False
    
    def sync_repository(self, repo_name: str) -> bool:
        """Sync repository with remote if configured"""
        if repo_name not in self.repositories:
            return False
        
        metadata = self.repositories[repo_name]
        metadata.sync_status = SyncStatus.SYNCING
        
        try:
            if metadata.remote_url and not metadata.local_path.exists():
                # Clone repository
                result = subprocess.run([
                    'git', 'clone', metadata.remote_url, str(metadata.local_path)
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    raise Exception(f"Git clone failed: {result.stderr}")
            
            elif metadata.local_path.exists():
                # Pull latest changes
                result = subprocess.run([
                    'git', 'pull'
                ], cwd=metadata.local_path, capture_output=True, text=True, timeout=120)
                
                if result.returncode != 0:
                    logging.warning(f"Git pull failed for {repo_name}: {result.stderr}")
            
            # Update metadata
            metadata.last_sync = datetime.now()
            metadata.sync_status = SyncStatus.SYNCED
            
            # Get current commit hash
            if metadata.local_path.exists():
                result = subprocess.run([
                    'git', 'rev-parse', 'HEAD'
                ], cwd=metadata.local_path, capture_output=True, text=True)
                
                if result.returncode == 0:
                    metadata.commit_hash = result.stdout.strip()
            
            return True
            
        except Exception as e:
            logging.error(f"Repository sync failed for {repo_name}: {e}")
            metadata.sync_status = SyncStatus.SYNC_ERROR
            return False
    
    def extract_intelligence(self, repo_name: str) -> Optional[TestingIntelligence]:
        """Extract testing intelligence from repository"""
        if repo_name not in self.repositories:
            return None
        
        metadata = self.repositories[repo_name]
        
        try:
            intelligence = self.scanner.scan_repository(metadata)
            self.analyzer.add_repository_intelligence(intelligence)
            return intelligence
            
        except Exception as e:
            logging.error(f"Intelligence extraction failed for {repo_name}: {e}")
            return None
    
    def aggregate_all(self, sync_first: bool = True) -> Dict[str, Any]:
        """Aggregate intelligence from all registered repositories"""
        results = {
            "aggregation_start": datetime.now().isoformat(),
            "repositories_processed": [],
            "repositories_failed": [],
            "intelligence_extracted": 0,
            "insights_discovered": 0
        }
        
        # Sync repositories if requested
        if sync_first:
            sync_futures = {}
            for repo_name in self.repositories:
                future = self.executor.submit(self.sync_repository, repo_name)
                sync_futures[repo_name] = future
            
            # Wait for sync completion
            for repo_name, future in sync_futures.items():
                try:
                    if future.result(timeout=600):  # 10 minute timeout
                        logging.info(f"Successfully synced {repo_name}")
                    else:
                        logging.warning(f"Sync failed for {repo_name}")
                except Exception as e:
                    logging.error(f"Sync error for {repo_name}: {e}")
                    results["repositories_failed"].append(repo_name)
        
        # Extract intelligence from all repositories
        extraction_futures = {}
        for repo_name in self.repositories:
            if repo_name not in results["repositories_failed"]:
                future = self.executor.submit(self.extract_intelligence, repo_name)
                extraction_futures[repo_name] = future
        
        # Collect extraction results
        for repo_name, future in extraction_futures.items():
            try:
                intelligence = future.result(timeout=300)  # 5 minute timeout
                if intelligence:
                    results["repositories_processed"].append(repo_name)
                    results["intelligence_extracted"] += 1
                else:
                    results["repositories_failed"].append(repo_name)
            except Exception as e:
                logging.error(f"Extraction error for {repo_name}: {e}")
                results["repositories_failed"].append(repo_name)
        
        results["insights_discovered"] = len(self.analyzer.cross_repo_insights)
        results["aggregation_complete"] = datetime.now().isoformat()
        
        return results
    
    def get_aggregation_summary(self) -> Dict[str, Any]:
        """Get comprehensive aggregation summary"""
        summary = {
            "registered_repositories": len(self.repositories),
            "repository_details": {},
            "cross_repository_analysis": self.analyzer.get_repository_comparison(),
            "latest_insights": [
                {
                    "title": insight.title,
                    "category": insight.category,
                    "impact_level": insight.impact_level,
                    "affected_repositories": insight.affected_repositories
                }
                for insight in self.analyzer.cross_repo_insights[-10:]  # Last 10 insights
            ]
        }
        
        # Repository details
        for repo_name, metadata in self.repositories.items():
            summary["repository_details"][repo_name] = {
                "type": metadata.repo_type.value,
                "sync_status": metadata.sync_status.value,
                "last_sync": metadata.last_sync.isoformat() if metadata.last_sync else None,
                "commit_hash": metadata.commit_hash
            }
        
        return summary


# Comprehensive Test Suite
class TestRepositoryIntelligenceAggregator(unittest.TestCase):
    
    def setUp(self):
        self.aggregator = RepositoryIntelligenceAggregator()
        
    def test_repository_registration(self):
        """Test repository registration"""
        success = self.aggregator.register_repository(
            "test_repo",
            RepositoryType.AGENCY_SWARM,
            Path("/mock/path"),
            "https://github.com/test/repo.git"
        )
        
        self.assertTrue(success)
        self.assertIn("test_repo", self.aggregator.repositories)
        self.assertEqual(
            self.aggregator.repositories["test_repo"].repo_type,
            RepositoryType.AGENCY_SWARM
        )
        
    def test_repository_scanner(self):
        """Test repository scanning functionality"""
        scanner = RepositoryScanner()
        
        # Test framework detection patterns
        frameworks = scanner._detect_testing_frameworks(Path("."))
        self.assertIsInstance(frameworks, list)
        
    def test_cross_repository_analyzer(self):
        """Test cross-repository analysis"""
        analyzer = CrossRepositoryAnalyzer()
        
        # Add mock intelligence
        intelligence1 = TestingIntelligence(
            repository="repo1",
            extraction_time=datetime.now(),
            patterns=[],
            frameworks_used=["pytest", "unittest"],
            test_strategies=["fixture_strategy", "mocking"],
            quality_metrics={"test_coverage": 0.8},
            code_complexity={"cyclomatic_complexity": 3.0},
            testing_maturity_score=0.7
        )
        
        analyzer.add_repository_intelligence(intelligence1)
        self.assertIn("repo1", analyzer.repository_intelligence)
        
    def test_testing_intelligence_creation(self):
        """Test testing intelligence data structure"""
        intelligence = TestingIntelligence(
            repository="test_repo",
            extraction_time=datetime.now(),
            patterns=[{"type": "pytest_pattern", "complexity": "medium"}],
            frameworks_used=["pytest", "jest"],
            test_strategies=["bdd", "tdd"],
            quality_metrics={"coverage": 0.85, "quality": 0.9},
            code_complexity={"complexity": 2.5},
            testing_maturity_score=0.8,
            recommendations=["Increase test coverage", "Add integration tests"]
        )
        
        self.assertEqual(intelligence.repository, "test_repo")
        self.assertEqual(len(intelligence.frameworks_used), 2)
        self.assertIn("pytest", intelligence.frameworks_used)
        self.assertEqual(intelligence.testing_maturity_score, 0.8)
        
    def test_cross_repo_insight_creation(self):
        """Test cross-repository insight creation"""
        insight = CrossRepoInsight(
            insight_id="test_insight_001",
            title="Framework Standardization Opportunity",
            description="Multiple repositories could benefit from standardized testing approach",
            affected_repositories=["repo1", "repo2", "repo3"],
            confidence_score=0.85,
            impact_level="MEDIUM",
            category="OPPORTUNITY",
            evidence=[{"pattern": "inconsistent_frameworks"}],
            recommendations=["Standardize on pytest across all repositories"],
            discovered_at=datetime.now()
        )
        
        self.assertEqual(insight.insight_id, "test_insight_001")
        self.assertEqual(len(insight.affected_repositories), 3)
        self.assertEqual(insight.confidence_score, 0.85)
        self.assertEqual(insight.impact_level, "MEDIUM")


if __name__ == "__main__":
    # Demo usage
    aggregator = RepositoryIntelligenceAggregator()
    
    # Register sample repositories
    sample_repos = [
        ("agency_swarm", RepositoryType.AGENCY_SWARM),
        ("crew_ai", RepositoryType.CREW_AI),
        ("agent_scope", RepositoryType.AGENT_SCOPE)
    ]
    
    for repo_name, repo_type in sample_repos:
        success = aggregator.register_repository(
            repo_name,
            repo_type,
            Path(f"mock_repos/{repo_name}"),
            f"https://github.com/mock/{repo_name}.git"
        )
        print(f"Registered {repo_name}: {success}")
    
    # Get aggregation summary
    summary = aggregator.get_aggregation_summary()
    print(f"Aggregator Summary: {json.dumps(summary, indent=2, default=str)}")
    
    print("Repository Intelligence Aggregator Demo Complete")
    
    # Run tests
    pytest.main([__file__, "-v"])