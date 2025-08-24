#!/usr/bin/env python3
"""
Risk-Based Test Targeter - Targets tests based on risk analysis and code changes.
Identifies high-risk areas and focuses testing efforts where they matter most.
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import ast
import difflib
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create wrapper classes for the analyzers
class ComplexityAnalyzer:
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Simple complexity analysis stub."""
        return {
            'total_complexity': 10,
            'cognitive_complexity': 15,
            'maintainability_index': 75,
            'lines_of_code': 100
        }

class DependencyAnalyzer:
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Simple dependency analysis stub."""
        return {
            'dependencies': [],
            'dependents': [],
            'circular_dependencies': []
        }

class ContinuousSecurityMonitor:
    def check_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Simple security check stub."""
        return []

# These will be imported later to avoid circular imports
TestComplexityPrioritizer = None
TestDependencyOrderer = None


class RiskLevel(Enum):
    """Risk levels for code areas."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class RiskFactor:
    """Represents a single risk factor."""
    factor_type: str
    description: str
    severity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    source: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'factor_type': self.factor_type,
            'description': self.description,
            'severity': self.severity,
            'confidence': self.confidence,
            'source': self.source
        }


@dataclass
class RiskProfile:
    """Risk profile for a code area or test."""
    path: str
    risk_level: RiskLevel
    risk_score: float  # 0.0 to 1.0
    risk_factors: List[RiskFactor] = field(default_factory=list)
    affected_areas: List[str] = field(default_factory=list)
    recommended_tests: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'path': self.path,
            'risk_level': self.risk_level.value,
            'risk_score': self.risk_score,
            'risk_factors': [f.to_dict() for f in self.risk_factors],
            'affected_areas': self.affected_areas,
            'recommended_tests': self.recommended_tests,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class TestTarget:
    """Represents a targeted test with risk justification."""
    test_path: str
    test_name: str
    target_score: float
    risk_coverage: float
    reasons: List[str] = field(default_factory=list)
    covered_risks: List[RiskFactor] = field(default_factory=list)
    estimated_value: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'test_path': self.test_path,
            'test_name': self.test_name,
            'target_score': self.target_score,
            'risk_coverage': self.risk_coverage,
            'reasons': self.reasons,
            'covered_risks': [r.to_dict() for r in self.covered_risks],
            'estimated_value': self.estimated_value
        }


class RiskBasedTestTargeter:
    """Targets tests based on comprehensive risk analysis."""
    
    def __init__(self, project_root: str = '.'):
        """Initialize the risk-based test targeter."""
        self.project_root = Path(project_root).resolve()
        
        # Initialize analyzers
        self.complexity_analyzer = ComplexityAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.security_monitor = ContinuousSecurityMonitor()
        
        # Import these lazily to avoid circular imports
        global TestComplexityPrioritizer, TestDependencyOrderer
        try:
            from test_complexity_prioritizer import TestComplexityPrioritizer as TCP
            from test_dependency_orderer import TestDependencyOrderer as TDO
            self.test_prioritizer = TCP(project_root)
            self.test_orderer = TDO(project_root)
        except ImportError:
            # Create stubs if imports fail
            self.test_prioritizer = None
            self.test_orderer = None
        
        # Risk profiles cache
        self._risk_profiles: Dict[str, RiskProfile] = {}
        self._change_history: Dict[str, List[Dict]] = {}
        self._test_coverage_map: Dict[str, Set[str]] = {}
        
        # Configuration
        self.config = {
            'risk_weights': {
                'complexity': 0.2,
                'security': 0.3,
                'change_frequency': 0.2,
                'dependencies': 0.15,
                'history': 0.15
            },
            'risk_thresholds': {
                RiskLevel.CRITICAL: 0.8,
                RiskLevel.HIGH: 0.6,
                RiskLevel.MEDIUM: 0.4,
                RiskLevel.LOW: 0.2,
                RiskLevel.MINIMAL: 0.0
            },
            'max_parallel_analysis': 10,
            'cache_ttl_seconds': 300
        }
        
        # Threading
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=self.config['max_parallel_analysis'])
    
    def analyze_risk(self, file_path: str, include_history: bool = True) -> RiskProfile:
        """Analyze risk for a specific file or directory."""
        path = Path(file_path)
        
        # Check cache
        if file_path in self._risk_profiles:
            profile = self._risk_profiles[file_path]
            if (datetime.now() - profile.last_updated).seconds < self.config['cache_ttl_seconds']:
                return profile
        
        risk_factors = []
        
        # Analyze different risk dimensions
        if path.is_file() and path.suffix == '.py':
            # Complexity risk
            complexity_risk = self._analyze_complexity_risk(file_path)
            if complexity_risk:
                risk_factors.extend(complexity_risk)
            
            # Security risk
            security_risk = self._analyze_security_risk(file_path)
            if security_risk:
                risk_factors.extend(security_risk)
            
            # Dependency risk
            dependency_risk = self._analyze_dependency_risk(file_path)
            if dependency_risk:
                risk_factors.extend(dependency_risk)
            
            # Change frequency risk
            if include_history:
                change_risk = self._analyze_change_frequency_risk(file_path)
                if change_risk:
                    risk_factors.extend(change_risk)
            
            # Historical failure risk
            if include_history:
                history_risk = self._analyze_historical_risk(file_path)
                if history_risk:
                    risk_factors.extend(history_risk)
        
        # Calculate overall risk score
        risk_score = self._calculate_overall_risk_score(risk_factors)
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_score)
        
        # Identify affected areas
        affected_areas = self._identify_affected_areas(file_path)
        
        # Recommend tests
        recommended_tests = self._recommend_tests_for_file(file_path, risk_factors)
        
        # Create risk profile
        profile = RiskProfile(
            path=file_path,
            risk_level=risk_level,
            risk_score=risk_score,
            risk_factors=risk_factors,
            affected_areas=affected_areas,
            recommended_tests=recommended_tests,
            last_updated=datetime.now()
        )
        
        # Cache profile
        with self._lock:
            self._risk_profiles[file_path] = profile
        
        return profile
    
    def target_tests(self, 
                    changed_files: Optional[List[str]] = None,
                    risk_threshold: Optional[float] = None,
                    max_tests: Optional[int] = None) -> List[TestTarget]:
        """Target tests based on risk analysis of changed files."""
        
        # If no changed files specified, analyze high-risk areas
        if not changed_files:
            changed_files = self._identify_high_risk_files()
        
        # Analyze risk for all changed files
        risk_profiles = []
        with self._executor as executor:
            futures = []
            for file_path in changed_files:
                future = executor.submit(self.analyze_risk, file_path)
                futures.append((file_path, future))
            
            for file_path, future in futures:
                try:
                    profile = future.result(timeout=30)
                    risk_profiles.append(profile)
                except Exception as e:
                    print(f"Error analyzing risk for {file_path}: {e}")
        
        # Collect all risk factors
        all_risk_factors = []
        for profile in risk_profiles:
            all_risk_factors.extend(profile.risk_factors)
        
        # Identify tests that cover these risks
        test_targets = self._identify_test_targets(risk_profiles, all_risk_factors)
        
        # Apply risk threshold if specified
        if risk_threshold:
            test_targets = [t for t in test_targets if t.risk_coverage >= risk_threshold]
        
        # Sort by target score
        test_targets.sort(key=lambda x: x.target_score, reverse=True)
        
        # Limit number of tests if specified
        if max_tests:
            test_targets = test_targets[:max_tests]
        
        return test_targets
    
    def _analyze_complexity_risk(self, file_path: str) -> List[RiskFactor]:
        """Analyze complexity-related risks."""
        risk_factors = []
        
        try:
            result = self.complexity_analyzer.analyze(file_path)
            
            # Cyclomatic complexity risk
            complexity = result.get('total_complexity', 0)
            if complexity > 10:
                severity = min(1.0, complexity / 50.0)
                risk_factors.append(RiskFactor(
                    factor_type='complexity',
                    description=f'High cyclomatic complexity: {complexity}',
                    severity=severity,
                    confidence=0.9,
                    source='complexity_analyzer'
                ))
            
            # Cognitive complexity risk
            cognitive = result.get('cognitive_complexity', 0)
            if cognitive > 15:
                severity = min(1.0, cognitive / 100.0)
                risk_factors.append(RiskFactor(
                    factor_type='cognitive_complexity',
                    description=f'High cognitive complexity: {cognitive}',
                    severity=severity,
                    confidence=0.85,
                    source='complexity_analyzer'
                ))
            
            # Maintainability risk
            maintainability = result.get('maintainability_index', 100)
            if maintainability < 50:
                severity = (50 - maintainability) / 50.0
                risk_factors.append(RiskFactor(
                    factor_type='maintainability',
                    description=f'Low maintainability index: {maintainability:.1f}',
                    severity=severity,
                    confidence=0.8,
                    source='complexity_analyzer'
                ))
            
        except Exception as e:
            print(f"Error analyzing complexity risk for {file_path}: {e}")
        
        return risk_factors
    
    def _analyze_security_risk(self, file_path: str) -> List[RiskFactor]:
        """Analyze security-related risks."""
        risk_factors = []
        
        try:
            # Use security monitor to check for vulnerabilities
            vulnerabilities = self.security_monitor.check_file(file_path)
            
            if vulnerabilities:
                for vuln in vulnerabilities:
                    severity_map = {
                        'critical': 1.0,
                        'high': 0.8,
                        'medium': 0.5,
                        'low': 0.3,
                        'info': 0.1
                    }
                    
                    severity = severity_map.get(vuln.get('severity', 'medium'), 0.5)
                    
                    risk_factors.append(RiskFactor(
                        factor_type='security',
                        description=vuln.get('message', 'Security vulnerability detected'),
                        severity=severity,
                        confidence=vuln.get('confidence', 0.7),
                        source='security_monitor'
                    ))
            
            # Check for security patterns
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # SQL injection risk
            if 'execute(' in content or 'raw(' in content:
                risk_factors.append(RiskFactor(
                    factor_type='security',
                    description='Potential SQL injection risk',
                    severity=0.7,
                    confidence=0.6,
                    source='pattern_analysis'
                ))
            
            # Hardcoded secrets risk
            secret_patterns = ['password = os.getenv('PASSWORD')api_key = os.getenv('KEY')secret = os.getenv('SECRET')token=']
            for pattern in secret_patterns:
                if pattern in content.lower():
                    risk_factors.append(RiskFactor(
                        factor_type='security',
                        description='Potential hardcoded secrets',
                        severity=0.9,
                        confidence=0.5,
                        source='pattern_analysis'
                    ))
                    break
            
        except Exception as e:
            print(f"Error analyzing security risk for {file_path}: {e}")
        
        return risk_factors
    
    def _analyze_dependency_risk(self, file_path: str) -> List[RiskFactor]:
        """Analyze dependency-related risks."""
        risk_factors = []
        
        try:
            result = self.dependency_analyzer.analyze(file_path)
            dependencies = result.get('dependencies', [])
            
            # High coupling risk
            if len(dependencies) > 10:
                severity = min(1.0, len(dependencies) / 30.0)
                risk_factors.append(RiskFactor(
                    factor_type='dependencies',
                    description=f'High coupling: {len(dependencies)} dependencies',
                    severity=severity,
                    confidence=0.8,
                    source='dependency_analyzer'
                ))
            
            # Circular dependency risk
            circular = result.get('circular_dependencies', [])
            if circular:
                risk_factors.append(RiskFactor(
                    factor_type='dependencies',
                    description=f'Circular dependencies detected: {len(circular)}',
                    severity=0.8,
                    confidence=0.9,
                    source='dependency_analyzer'
                ))
            
            # External dependency risk
            external_deps = [d for d in dependencies if self._is_external_dependency(d)]
            if len(external_deps) > 5:
                severity = min(1.0, len(external_deps) / 20.0)
                risk_factors.append(RiskFactor(
                    factor_type='dependencies',
                    description=f'Many external dependencies: {len(external_deps)}',
                    severity=severity,
                    confidence=0.7,
                    source='dependency_analyzer'
                ))
            
        except Exception as e:
            print(f"Error analyzing dependency risk for {file_path}: {e}")
        
        return risk_factors
    
    def _analyze_change_frequency_risk(self, file_path: str) -> List[RiskFactor]:
        """Analyze risk based on change frequency."""
        risk_factors = []
        
        try:
            # Get git history for the file
            result = subprocess.run(
                ['git', 'log', '--oneline', '--', file_path],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                commit_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                
                # High change frequency risk
                if commit_count > 50:
                    severity = min(1.0, commit_count / 200.0)
                    risk_factors.append(RiskFactor(
                        factor_type='change_frequency',
                        description=f'High change frequency: {commit_count} commits',
                        severity=severity,
                        confidence=0.8,
                        source='git_history'
                    ))
                
                # Recent changes risk
                recent_result = subprocess.run(
                    ['git', 'log', '--since="30 days ago"', '--oneline', '--', file_path],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )
                
                if recent_result.returncode == 0:
                    recent_commits = len(recent_result.stdout.strip().split('\n')) if recent_result.stdout.strip() else 0
                    
                    if recent_commits > 5:
                        severity = min(1.0, recent_commits / 20.0)
                        risk_factors.append(RiskFactor(
                            factor_type='change_frequency',
                            description=f'Recent changes: {recent_commits} commits in last 30 days',
                            severity=severity,
                            confidence=0.9,
                            source='git_history'
                        ))
            
        except Exception as e:
            print(f"Error analyzing change frequency risk for {file_path}: {e}")
        
        return risk_factors
    
    def _analyze_historical_risk(self, file_path: str) -> List[RiskFactor]:
        """Analyze risk based on historical issues."""
        risk_factors = []
        
        # Check if file has historical data
        if file_path in self._change_history:
            history = self._change_history[file_path]
            
            # Bug fix frequency
            bug_fixes = [h for h in history if 'fix' in h.get('message', '').lower()]
            if len(bug_fixes) > 5:
                severity = min(1.0, len(bug_fixes) / 20.0)
                risk_factors.append(RiskFactor(
                    factor_type='history',
                    description=f'Frequent bug fixes: {len(bug_fixes)} fixes',
                    severity=severity,
                    confidence=0.7,
                    source='historical_analysis'
                ))
        
        return risk_factors
    
    def _calculate_overall_risk_score(self, risk_factors: List[RiskFactor]) -> float:
        """Calculate overall risk score from individual factors."""
        if not risk_factors:
            return 0.0
        
        # Group factors by type
        factors_by_type = defaultdict(list)
        for factor in risk_factors:
            factors_by_type[factor.factor_type].append(factor)
        
        # Calculate weighted score
        total_score = 0.0
        weights = self.config['risk_weights']
        
        for factor_type, type_factors in factors_by_type.items():
            # Get weight for this type
            weight = weights.get(factor_type, 0.1)
            
            # Calculate average severity for this type
            avg_severity = sum(f.severity * f.confidence for f in type_factors) / len(type_factors)
            
            total_score += avg_severity * weight
        
        # Normalize
        total_weight = sum(weights.get(ft, 0.1) for ft in factors_by_type.keys())
        if total_weight > 0:
            total_score /= total_weight
        
        return min(1.0, max(0.0, total_score))
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from risk score."""
        thresholds = self.config['risk_thresholds']
        
        for level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]:
            if risk_score >= thresholds[level]:
                return level
        
        return RiskLevel.MINIMAL
    
    def _identify_affected_areas(self, file_path: str) -> List[str]:
        """Identify areas affected by risks in a file."""
        affected = []
        
        try:
            # Get file dependencies
            dep_result = self.dependency_analyzer.analyze(file_path)
            dependents = dep_result.get('dependents', [])
            
            for dep in dependents:
                if isinstance(dep, dict):
                    affected.append(dep.get('path', str(dep)))
                else:
                    affected.append(str(dep))
        
        except:
            pass
        
        return affected
    
    def _recommend_tests_for_file(self, file_path: str, risk_factors: List[RiskFactor]) -> List[str]:
        """Recommend tests for a file based on its risks."""
        recommended = []
        
        # Find tests that cover this file
        file_name = Path(file_path).stem
        test_patterns = [
            f"test_{file_name}.py",
            f"{file_name}_test.py",
            f"test_*{file_name}*.py"
        ]
        
        for pattern in test_patterns:
            matching_tests = list(self.project_root.rglob(pattern))
            recommended.extend(str(t) for t in matching_tests if t.is_file())
        
        # Add tests based on risk factors
        for factor in risk_factors:
            if factor.factor_type == 'security':
                # Add security tests
                security_tests = list(self.project_root.rglob("*security*.py"))
                recommended.extend(str(t) for t in security_tests if 'test' in t.name)
            elif factor.factor_type == 'complexity':
                # Add integration tests
                integration_tests = list(self.project_root.rglob("*integration*.py"))
                recommended.extend(str(t) for t in integration_tests if 'test' in t.name)
        
        return list(set(recommended))
    
    def _identify_high_risk_files(self) -> List[str]:
        """Identify high-risk files in the project."""
        high_risk = []
        
        # Scan Python files
        for py_file in self.project_root.rglob('*.py'):
            if not any(part.startswith('.') for part in py_file.parts):
                try:
                    profile = self.analyze_risk(str(py_file), include_history=False)
                    if profile.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                        high_risk.append(str(py_file))
                except:
                    pass
        
        return high_risk
    
    def _identify_test_targets(self, risk_profiles: List[RiskProfile], 
                              risk_factors: List[RiskFactor]) -> List[TestTarget]:
        """Identify tests to target based on risk profiles."""
        test_targets = []
        
        # Discover all available tests
        all_tests = self.test_prioritizer.discover_tests()
        
        # Build test coverage map if not cached
        if not self._test_coverage_map:
            self._build_test_coverage_map(all_tests)
        
        # Score each test based on risk coverage
        for test_path in all_tests:
            test_name = Path(test_path).stem
            
            # Calculate how well this test covers the risks
            covered_risks = []
            coverage_score = 0.0
            reasons = []
            
            # Check if test covers any risky files
            for profile in risk_profiles:
                if self._test_covers_file(test_path, profile.path):
                    coverage_score += profile.risk_score
                    covered_risks.extend(profile.risk_factors)
                    reasons.append(f"Covers {Path(profile.path).name} (risk: {profile.risk_level.value})")
            
            # Check for specific risk coverage
            if 'security' in test_name.lower():
                security_risks = [r for r in risk_factors if r.factor_type == 'security']
                if security_risks:
                    coverage_score += 0.3
                    covered_risks.extend(security_risks[:3])
                    reasons.append("Security-focused test")
            
            if 'integration' in test_name.lower():
                dependency_risks = [r for r in risk_factors if r.factor_type == 'dependencies']
                if dependency_risks:
                    coverage_score += 0.2
                    covered_risks.extend(dependency_risks[:2])
                    reasons.append("Integration test for dependencies")
            
            # Calculate target score
            if coverage_score > 0:
                # Normalize coverage score
                risk_coverage = min(1.0, coverage_score / len(risk_profiles)) if risk_profiles else 0.0
                
                # Calculate estimated value
                estimated_value = risk_coverage * len(covered_risks)
                
                # Create test target
                target = TestTarget(
                    test_path=test_path,
                    test_name=test_name,
                    target_score=coverage_score,
                    risk_coverage=risk_coverage,
                    reasons=reasons,
                    covered_risks=covered_risks[:10],  # Limit to top 10
                    estimated_value=estimated_value
                )
                
                test_targets.append(target)
        
        return test_targets
    
    def _test_covers_file(self, test_path: str, file_path: str) -> bool:
        """Check if a test covers a specific file."""
        # Check coverage map
        if test_path in self._test_coverage_map:
            covered_files = self._test_coverage_map[test_path]
            if file_path in covered_files:
                return True
        
        # Check by naming convention
        test_name = Path(test_path).stem
        file_name = Path(file_path).stem
        
        if test_name.startswith('test_'):
            tested_name = test_name[5:]
            if tested_name == file_name:
                return True
        
        # Check imports in test file
        try:
            with open(test_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file is imported
            if file_name in content:
                return True
        except:
            pass
        
        return False
    
    def _build_test_coverage_map(self, test_paths: List[str]):
        """Build map of which files each test covers."""
        self._test_coverage_map.clear()
        
        for test_path in test_paths:
            covered = set()
            
            try:
                # Parse test file
                with open(test_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                # Find imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module_file = self._find_module_file(alias.name)
                            if module_file:
                                covered.add(module_file)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            module_file = self._find_module_file(node.module)
                            if module_file:
                                covered.add(module_file)
            except:
                pass
            
            self._test_coverage_map[test_path] = covered
    
    def _find_module_file(self, module_name: str) -> Optional[str]:
        """Find file path for a module name."""
        # Try direct file
        possible_paths = [
            self.project_root / f"{module_name}.py",
            self.project_root / module_name.replace('.', '/') / "__init__.py",
            self.project_root / f"{module_name.replace('.', '/')}.py"
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def _is_external_dependency(self, dep: Any) -> bool:
        """Check if a dependency is external."""
        if isinstance(dep, dict):
            module = dep.get('module', dep.get('name', ''))
        else:
            module = str(dep)
        
        # Check common external indicators
        external_patterns = ['pip', 'numpy', 'pandas', 'requests', 'django', 'flask']
        return any(pattern in module.lower() for pattern in external_patterns)
    
    def generate_risk_report(self, output_file: str = 'risk_report.json'):
        """Generate comprehensive risk report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'project': str(self.project_root),
            'risk_profiles': [],
            'summary': {
                'total_files_analyzed': 0,
                'critical_risks': 0,
                'high_risks': 0,
                'medium_risks': 0,
                'low_risks': 0,
                'minimal_risks': 0
            }
        }
        
        # Analyze all Python files
        for py_file in self.project_root.rglob('*.py'):
            if not any(part.startswith('.') for part in py_file.parts):
                try:
                    profile = self.analyze_risk(str(py_file))
                    report['risk_profiles'].append(profile.to_dict())
                    report['summary']['total_files_analyzed'] += 1
                    
                    # Update summary counts
                    level_key = f"{profile.risk_level.value}_risks"
                    if level_key in report['summary']:
                        report['summary'][level_key] += 1
                except:
                    pass
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Risk report generated: {output_file}")
        return report


def main():
    """Main function to demonstrate risk-based test targeting."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Target tests based on risk analysis')
    parser.add_argument('--project', default='.', help='Project root directory')
    parser.add_argument('--changed-files', nargs='*', help='Changed files to analyze')
    parser.add_argument('--risk-threshold', type=float, help='Minimum risk threshold for tests')
    parser.add_argument('--max-tests', type=int, help='Maximum number of tests to run')
    parser.add_argument('--report', help='Generate risk report to file')
    
    args = parser.parse_args()
    
    # Initialize targeter
    targeter = RiskBasedTestTargeter(args.project)
    
    # Generate risk report if requested
    if args.report:
        print("Generating risk report...")
        report = targeter.generate_risk_report(args.report)
        print(f"\nRisk Summary:")
        print(f"  Files analyzed: {report['summary']['total_files_analyzed']}")
        print(f"  Critical risks: {report['summary']['critical_risks']}")
        print(f"  High risks: {report['summary']['high_risks']}")
        print(f"  Medium risks: {report['summary']['medium_risks']}")
    
    # Analyze changed files or high-risk areas
    if args.changed_files:
        print(f"\nAnalyzing risks for {len(args.changed_files)} changed files...")
    else:
        print("\nIdentifying high-risk areas...")
    
    # Target tests
    targets = targeter.target_tests(
        changed_files=args.changed_files,
        risk_threshold=args.risk_threshold,
        max_tests=args.max_tests
    )
    
    print(f"\nTargeted Tests ({len(targets)} tests):")
    for i, target in enumerate(targets[:20], 1):  # Show top 20
        print(f"\n{i}. {target.test_name}")
        print(f"   Score: {target.target_score:.3f}")
        print(f"   Risk coverage: {target.risk_coverage:.1%}")
        print(f"   Value: {target.estimated_value:.2f}")
        print(f"   Reasons:")
        for reason in target.reasons[:3]:
            print(f"     - {reason}")
        if target.covered_risks:
            print(f"   Top covered risks:")
            for risk in target.covered_risks[:3]:
                print(f"     - {risk.description} (severity: {risk.severity:.2f})")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())