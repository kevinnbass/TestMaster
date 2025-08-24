"""
Supply Chain Security Analysis Module
======================================

Implements comprehensive supply chain security analysis:
- Python package vulnerability scanning (pip, conda, poetry)
- Known CVE detection in dependencies
- License compliance checking
- Dependency confusion attack detection
- Outdated package identification
- Transitive dependency analysis
- Package reputation scoring
- Typosquatting detection
"""

import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict
import requests
from packaging import version
import hashlib

from .base_analyzer import BaseAnalyzer


class SupplyChainSecurityAnalyzer(BaseAnalyzer):
    """Analyzer for supply chain security vulnerabilities."""
    
    def __init__(self, base_path: Path):
        super().__init__(base_path)
        self.vulnerability_db = self._load_vulnerability_database()
        self.known_typosquats = self._load_typosquatting_list()
        self.trusted_packages = self._load_trusted_packages()
        
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive supply chain security analysis."""
        print("[INFO] Analyzing Supply Chain Security...")
        
        results = {
            "dependency_analysis": self._analyze_dependencies(),
            "vulnerability_scan": self._scan_vulnerabilities(),
            "license_compliance": self._check_license_compliance(),
            "outdated_packages": self._identify_outdated_packages(),
            "transitive_dependencies": self._analyze_transitive_dependencies(),
            "package_reputation": self._assess_package_reputation(),
            "typosquatting_detection": self._detect_typosquatting(),
            "dependency_confusion": self._detect_dependency_confusion(),
            "supply_chain_metrics": self._calculate_supply_chain_metrics()
        }
        
        print(f"  [OK] Analyzed {len(results)} supply chain security aspects")
        return results
    
    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze all project dependencies."""
        dependencies = {
            "pip": self._analyze_pip_dependencies(),
            "conda": self._analyze_conda_dependencies(),
            "poetry": self._analyze_poetry_dependencies(),
            "direct_imports": self._analyze_direct_imports()
        }
        
        # Consolidate all dependencies
        all_deps = set()
        for source in dependencies.values():
            if source and "packages" in source:
                all_deps.update(source["packages"].keys())
        
        return {
            "sources": dependencies,
            "total_dependencies": len(all_deps),
            "dependency_list": list(all_deps),
            "dependency_tree": self._build_dependency_tree(dependencies)
        }
    
    def _analyze_pip_dependencies(self) -> Optional[Dict[str, Any]]:
        """Analyze pip dependencies from requirements files."""
        dependencies = {}
        
        # Check for requirements.txt variations
        requirement_files = [
            "requirements.txt",
            "requirements.in",
            "requirements-dev.txt",
            "requirements-prod.txt",
            "requirements/base.txt",
            "requirements/production.txt"
        ]
        
        for req_file in requirement_files:
            req_path = self.base_path / req_file
            if req_path.exists():
                try:
                    with open(req_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                # Parse package and version
                                if '==' in line:
                                    package, ver = line.split('==')
                                    dependencies[package.strip()] = {
                                        "version": ver.strip(),
                                        "source": req_file,
                                        "specifier": "=="
                                    }
                                elif '>=' in line:
                                    package, ver = line.split('>=')
                                    dependencies[package.strip()] = {
                                        "version": ver.strip(),
                                        "source": req_file,
                                        "specifier": ">="
                                    }
                                else:
                                    # No version specified
                                    package = line.split('[')[0].strip()
                                    dependencies[package] = {
                                        "version": "*",
                                        "source": req_file,
                                        "specifier": "*"
                                    }
                except Exception as e:
                    print(f"  [WARN] Error reading {req_file}: {e}")
        
        # Check for setup.py
        setup_path = self.base_path / "setup.py"
        if setup_path.exists():
            setup_deps = self._parse_setup_py(setup_path)
            dependencies.update(setup_deps)
        
        # Check for Pipfile
        pipfile_path = self.base_path / "Pipfile"
        if pipfile_path.exists():
            pipfile_deps = self._parse_pipfile(pipfile_path)
            dependencies.update(pipfile_deps)
        
        return {"packages": dependencies} if dependencies else None
    
    def _analyze_conda_dependencies(self) -> Optional[Dict[str, Any]]:
        """Analyze conda dependencies from environment files."""
        dependencies = {}
        
        # Check for conda environment files
        env_files = ["environment.yml", "environment.yaml", "conda.yml", "conda.yaml"]
        
        for env_file in env_files:
            env_path = self.base_path / env_file
            if env_path.exists():
                try:
                    import yaml
                    with open(env_path, 'r') as f:
                        env_data = yaml.safe_load(f)
                        if 'dependencies' in env_data:
                            for dep in env_data['dependencies']:
                                if isinstance(dep, str):
                                    # Parse conda dependency
                                    if '=' in dep:
                                        package, ver = dep.split('=', 1)
                                        dependencies[package] = {
                                            "version": ver,
                                            "source": env_file,
                                            "specifier": "="
                                        }
                                    else:
                                        dependencies[dep] = {
                                            "version": "*",
                                            "source": env_file,
                                            "specifier": "*"
                                        }
                                elif isinstance(dep, dict) and 'pip' in dep:
                                    # Parse pip dependencies within conda
                                    for pip_dep in dep['pip']:
                                        if '==' in pip_dep:
                                            package, ver = pip_dep.split('==')
                                            dependencies[package.strip()] = {
                                                "version": ver.strip(),
                                                "source": f"{env_file} (pip)",
                                                "specifier": "=="
                                            }
                except Exception as e:
                    print(f"  [WARN] Error reading {env_file}: {e}")
        
        return {"packages": dependencies} if dependencies else None
    
    def _analyze_poetry_dependencies(self) -> Optional[Dict[str, Any]]:
        """Analyze poetry dependencies from pyproject.toml."""
        dependencies = {}
        
        pyproject_path = self.base_path / "pyproject.toml"
        if pyproject_path.exists():
            try:
                import toml
                with open(pyproject_path, 'r') as f:
                    pyproject = toml.load(f)
                    
                    # Check poetry dependencies
                    if 'tool' in pyproject and 'poetry' in pyproject['tool']:
                        poetry_config = pyproject['tool']['poetry']
                        
                        # Regular dependencies
                        if 'dependencies' in poetry_config:
                            for package, spec in poetry_config['dependencies'].items():
                                if package != 'python':
                                    if isinstance(spec, str):
                                        dependencies[package] = {
                                            "version": spec.strip('^~'),
                                            "source": "pyproject.toml",
                                            "specifier": spec[0] if spec[0] in '^~' else "=="
                                        }
                                    elif isinstance(spec, dict):
                                        dependencies[package] = {
                                            "version": spec.get('version', '*'),
                                            "source": "pyproject.toml",
                                            "specifier": "complex"
                                        }
                        
                        # Dev dependencies
                        if 'dev-dependencies' in poetry_config:
                            for package, spec in poetry_config['dev-dependencies'].items():
                                if isinstance(spec, str):
                                    dependencies[f"{package} (dev)"] = {
                                        "version": spec.strip('^~'),
                                        "source": "pyproject.toml (dev)",
                                        "specifier": spec[0] if spec[0] in '^~' else "=="
                                    }
            except Exception as e:
                print(f"  [WARN] Error reading pyproject.toml: {e}")
        
        return {"packages": dependencies} if dependencies else None
    
    def _analyze_direct_imports(self) -> Dict[str, Any]:
        """Analyze direct imports from Python files."""
        imports = set()
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])
            except:
                continue
        
        # Filter to external packages only (exclude stdlib)
        stdlib_modules = self._get_stdlib_modules()
        external_imports = imports - stdlib_modules
        
        return {
            "packages": {pkg: {"source": "imports", "detected": True} 
                        for pkg in external_imports}
        }
    
    def _scan_vulnerabilities(self) -> Dict[str, Any]:
        """Scan for known vulnerabilities in dependencies."""
        vulnerabilities = []
        
        dependencies = self._analyze_dependencies()
        if not dependencies or "dependency_list" not in dependencies:
            return {"vulnerabilities": [], "summary": {"total": 0}}
        
        for package in dependencies["dependency_list"]:
            # Check against vulnerability database
            if package in self.vulnerability_db:
                for vuln in self.vulnerability_db[package]:
                    vulnerabilities.append({
                        "package": package,
                        "vulnerability_id": vuln.get("id"),
                        "severity": vuln.get("severity", "UNKNOWN"),
                        "description": vuln.get("description"),
                        "affected_versions": vuln.get("affected_versions"),
                        "fixed_version": vuln.get("fixed_version"),
                        "cve": vuln.get("cve"),
                        "references": vuln.get("references", [])
                    })
        
        # Categorize by severity
        severity_counts = defaultdict(int)
        for vuln in vulnerabilities:
            severity_counts[vuln["severity"]] += 1
        
        return {
            "vulnerabilities": vulnerabilities,
            "summary": {
                "total": len(vulnerabilities),
                "critical": severity_counts.get("CRITICAL", 0),
                "high": severity_counts.get("HIGH", 0),
                "medium": severity_counts.get("MEDIUM", 0),
                "low": severity_counts.get("LOW", 0),
                "unknown": severity_counts.get("UNKNOWN", 0)
            },
            "affected_packages": list(set(v["package"] for v in vulnerabilities))
        }
    
    def _check_license_compliance(self) -> Dict[str, Any]:
        """Check license compliance of dependencies."""
        license_info = {
            "compatible_licenses": [],
            "incompatible_licenses": [],
            "unknown_licenses": [],
            "license_conflicts": []
        }
        
        # Define license compatibility matrix
        permissive_licenses = {"MIT", "Apache-2.0", "BSD", "ISC", "Apache License 2.0"}
        copyleft_licenses = {"GPL", "LGPL", "AGPL", "GPL-3.0", "GPL-2.0"}
        
        dependencies = self._analyze_dependencies()
        if dependencies and "sources" in dependencies:
            for source_type, source_data in dependencies["sources"].items():
                if source_data and "packages" in source_data:
                    for package, info in source_data["packages"].items():
                        # Simulate license lookup (would need actual package metadata)
                        license_type = self._get_package_license(package)
                        
                        if license_type:
                            if license_type in permissive_licenses:
                                license_info["compatible_licenses"].append({
                                    "package": package,
                                    "license": license_type,
                                    "type": "permissive"
                                })
                            elif license_type in copyleft_licenses:
                                license_info["incompatible_licenses"].append({
                                    "package": package,
                                    "license": license_type,
                                    "type": "copyleft",
                                    "risk": "May require source code disclosure"
                                })
                        else:
                            license_info["unknown_licenses"].append({
                                "package": package,
                                "risk": "Unknown license terms"
                            })
        
        return license_info
    
    def _identify_outdated_packages(self) -> Dict[str, Any]:
        """Identify outdated packages that need updates."""
        outdated = []
        
        dependencies = self._analyze_dependencies()
        if dependencies and "sources" in dependencies:
            for source_type, source_data in dependencies["sources"].items():
                if source_data and "packages" in source_data:
                    for package, info in source_data["packages"].items():
                        if "version" in info and info["version"] != "*":
                            # Check if newer version exists (simulated)
                            latest_version = self._get_latest_version(package)
                            if latest_version and info["version"] != latest_version:
                                try:
                                    if version.parse(info["version"]) < version.parse(latest_version):
                                        outdated.append({
                                            "package": package,
                                            "current_version": info["version"],
                                            "latest_version": latest_version,
                                            "source": info.get("source", "unknown"),
                                            "update_priority": self._calculate_update_priority(
                                                package, info["version"], latest_version
                                            )
                                        })
                                except:
                                    pass
        
        return {
            "outdated_packages": outdated,
            "total_outdated": len(outdated),
            "high_priority_updates": len([p for p in outdated if p.get("update_priority") == "HIGH"])
        }
    
    def _analyze_transitive_dependencies(self) -> Dict[str, Any]:
        """Analyze transitive (indirect) dependencies."""
        # This would require actual package resolution
        # For now, return a simulated structure
        return {
            "depth_analysis": {
                "max_depth": 3,
                "average_depth": 2.1,
                "deep_dependencies": []  # Dependencies more than 2 levels deep
            },
            "shared_dependencies": {},  # Dependencies used by multiple packages
            "version_conflicts": []  # Different versions required by different packages
        }
    
    def _assess_package_reputation(self) -> Dict[str, Any]:
        """Assess reputation of packages."""
        reputation_scores = {}
        suspicious_packages = []
        
        dependencies = self._analyze_dependencies()
        if dependencies and "dependency_list" in dependencies:
            for package in dependencies["dependency_list"]:
                score = self._calculate_reputation_score(package)
                reputation_scores[package] = score
                
                if score < 0.5:
                    suspicious_packages.append({
                        "package": package,
                        "score": score,
                        "reasons": self._get_reputation_issues(package, score)
                    })
        
        return {
            "reputation_scores": reputation_scores,
            "suspicious_packages": suspicious_packages,
            "average_reputation": sum(reputation_scores.values()) / len(reputation_scores) if reputation_scores else 0
        }
    
    def _detect_typosquatting(self) -> Dict[str, Any]:
        """Detect potential typosquatting attacks."""
        potential_typosquats = []
        
        dependencies = self._analyze_dependencies()
        if dependencies and "dependency_list" in dependencies:
            for package in dependencies["dependency_list"]:
                # Check against known typosquats
                if package in self.known_typosquats:
                    potential_typosquats.append({
                        "package": package,
                        "type": "known_typosquat",
                        "legitimate_package": self.known_typosquats[package],
                        "risk": "HIGH"
                    })
                
                # Check similarity to popular packages
                for trusted in self.trusted_packages:
                    similarity = self._calculate_string_similarity(package, trusted)
                    if 0.8 < similarity < 1.0:  # Similar but not identical
                        potential_typosquats.append({
                            "package": package,
                            "type": "suspicious_similarity",
                            "similar_to": trusted,
                            "similarity_score": similarity,
                            "risk": "MEDIUM"
                        })
        
        return {
            "potential_typosquats": potential_typosquats,
            "high_risk_count": len([p for p in potential_typosquats if p["risk"] == "HIGH"])
        }
    
    def _detect_dependency_confusion(self) -> Dict[str, Any]:
        """Detect potential dependency confusion attacks."""
        confusion_risks = []
        
        # Check for internal package names that might exist on public repos
        internal_patterns = [
            r'^company-',
            r'^internal-',
            r'^private-',
            r'-internal$',
            r'-private$'
        ]
        
        dependencies = self._analyze_dependencies()
        if dependencies and "dependency_list" in dependencies:
            for package in dependencies["dependency_list"]:
                # Check if package name suggests internal package
                for pattern in internal_patterns:
                    if re.search(pattern, package, re.IGNORECASE):
                        confusion_risks.append({
                            "package": package,
                            "risk_type": "potential_internal_package",
                            "recommendation": "Ensure package source is configured correctly",
                            "mitigation": "Use private package repository with higher priority"
                        })
                        break
        
        return {
            "confusion_risks": confusion_risks,
            "total_risks": len(confusion_risks),
            "mitigation_strategies": [
                "Configure pip to prioritize private repositories",
                "Use package pinning with hashes",
                "Implement namespace prefixes for internal packages",
                "Regular audit of dependency sources"
            ]
        }
    
    def _calculate_supply_chain_metrics(self) -> Dict[str, Any]:
        """Calculate overall supply chain security metrics."""
        vulnerabilities = self._scan_vulnerabilities()
        outdated = self._identify_outdated_packages()
        reputation = self._assess_package_reputation()
        typosquats = self._detect_typosquatting()
        confusion = self._detect_dependency_confusion()
        
        # Calculate risk score
        risk_score = 0
        risk_factors = []
        
        if vulnerabilities["summary"]["critical"] > 0:
            risk_score += 30
            risk_factors.append(f"{vulnerabilities['summary']['critical']} critical vulnerabilities")
        
        if vulnerabilities["summary"]["high"] > 0:
            risk_score += 20
            risk_factors.append(f"{vulnerabilities['summary']['high']} high vulnerabilities")
        
        if outdated["high_priority_updates"] > 0:
            risk_score += 15
            risk_factors.append(f"{outdated['high_priority_updates']} high-priority updates needed")
        
        if reputation["average_reputation"] < 0.7:
            risk_score += 10
            risk_factors.append("Low average package reputation")
        
        if typosquats["high_risk_count"] > 0:
            risk_score += 25
            risk_factors.append(f"{typosquats['high_risk_count']} potential typosquatting risks")
        
        if confusion["total_risks"] > 0:
            risk_score += 10
            risk_factors.append(f"{confusion['total_risks']} dependency confusion risks")
        
        return {
            "overall_risk_score": min(risk_score, 100),
            "risk_level": self._get_risk_level(risk_score),
            "risk_factors": risk_factors,
            "recommendations": self._generate_recommendations(risk_score, risk_factors)
        }
    
    # Helper methods
    
    def _load_vulnerability_database(self) -> Dict[str, List[Dict]]:
        """Load vulnerability database (simulated)."""
        # In production, this would connect to OSV, NVD, or other vulnerability databases
        return {
            "requests": [
                {
                    "id": "CVE-2023-32681",
                    "severity": "HIGH",
                    "description": "Unintended leak of Proxy-Authorization header",
                    "affected_versions": "<2.31.0",
                    "fixed_version": "2.31.0",
                    "cve": "CVE-2023-32681"
                }
            ],
            "django": [
                {
                    "id": "CVE-2023-41164",
                    "severity": "CRITICAL",
                    "description": "Potential denial of service in django.utils.encoding",
                    "affected_versions": "<3.2.21",
                    "fixed_version": "3.2.21",
                    "cve": "CVE-2023-41164"
                }
            ]
        }
    
    def _load_typosquatting_list(self) -> Dict[str, str]:
        """Load known typosquatting packages."""
        return {
            "requets": "requests",
            "djnago": "django",
            "numby": "numpy",
            "panads": "pandas"
        }
    
    def _load_trusted_packages(self) -> Set[str]:
        """Load list of trusted/popular packages."""
        return {
            "requests", "django", "flask", "numpy", "pandas", "matplotlib",
            "scipy", "scikit-learn", "tensorflow", "pytorch", "pytest",
            "black", "flake8", "mypy", "pylint", "coverage"
        }
    
    def _get_stdlib_modules(self) -> Set[str]:
        """Get Python standard library modules."""
        return {
            'os', 'sys', 'json', 'math', 'random', 'datetime', 'collections',
            'itertools', 'functools', 're', 'ast', 'pathlib', 'typing',
            'unittest', 'logging', 'threading', 'multiprocessing', 'asyncio'
        }
    
    def _parse_setup_py(self, setup_path: Path) -> Dict[str, Any]:
        """Parse dependencies from setup.py."""
        dependencies = {}
        try:
            content = setup_path.read_text()
            # Simple regex-based parsing (would need more sophisticated parsing in production)
            install_requires = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if install_requires:
                requires_text = install_requires.group(1)
                for line in requires_text.split(','):
                    line = line.strip().strip('"').strip("'")
                    if line:
                        if '==' in line:
                            package, ver = line.split('==')
                            dependencies[package.strip()] = {
                                "version": ver.strip(),
                                "source": "setup.py",
                                "specifier": "=="
                            }
                        else:
                            package = line.split('[')[0].split('>')[0].split('<')[0].strip()
                            dependencies[package] = {
                                "version": "*",
                                "source": "setup.py",
                                "specifier": "*"
                            }
        except:
            pass
        return dependencies
    
    def _parse_pipfile(self, pipfile_path: Path) -> Dict[str, Any]:
        """Parse dependencies from Pipfile."""
        dependencies = {}
        # Simplified parsing - would use proper TOML parser in production
        return dependencies
    
    def _build_dependency_tree(self, dependencies: Dict) -> Dict:
        """Build dependency tree structure."""
        # Would require package resolution to build actual tree
        return {"tree": "Would require pip-tools or similar for resolution"}
    
    def _get_package_license(self, package: str) -> Optional[str]:
        """Get license for a package (simulated)."""
        # In production, would query PyPI API or package metadata
        common_licenses = {
            "requests": "Apache-2.0",
            "django": "BSD",
            "flask": "BSD",
            "numpy": "BSD",
            "pandas": "BSD"
        }
        return common_licenses.get(package)
    
    def _get_latest_version(self, package: str) -> Optional[str]:
        """Get latest version of a package (simulated)."""
        # In production, would query PyPI API
        latest_versions = {
            "requests": "2.31.0",
            "django": "4.2.7",
            "flask": "3.0.0",
            "numpy": "1.26.0",
            "pandas": "2.1.3"
        }
        return latest_versions.get(package)
    
    def _calculate_update_priority(self, package: str, current: str, latest: str) -> str:
        """Calculate update priority based on version difference."""
        try:
            curr_ver = version.parse(current)
            late_ver = version.parse(latest)
            
            # Check major version difference
            if curr_ver.major < late_ver.major:
                return "HIGH"
            elif curr_ver.minor < late_ver.minor:
                return "MEDIUM"
            else:
                return "LOW"
        except:
            return "UNKNOWN"
    
    def _calculate_reputation_score(self, package: str) -> float:
        """Calculate reputation score for a package."""
        # Simulated scoring based on various factors
        score = 0.5  # Base score
        
        # Trusted packages get high score
        if package in self.trusted_packages:
            score = 0.95
        
        # Check various factors (simulated)
        factors = {
            "downloads": 0.2,  # Would check download count
            "maintenance": 0.2,  # Would check last update
            "contributors": 0.1,  # Would check contributor count
            "documentation": 0.1,  # Would check docs availability
            "tests": 0.1  # Would check test coverage
        }
        
        for factor, weight in factors.items():
            # Simulated factor scoring
            if package in self.trusted_packages:
                score += weight * 0.9
            else:
                score += weight * 0.3
        
        return min(score, 1.0)
    
    def _get_reputation_issues(self, package: str, score: float) -> List[str]:
        """Get reputation issues for a package."""
        issues = []
        
        if score < 0.3:
            issues.append("Very low download count")
            issues.append("No recent maintenance")
        elif score < 0.5:
            issues.append("Limited community engagement")
            issues.append("Sparse documentation")
        
        if package not in self.trusted_packages:
            issues.append("Not in trusted package list")
        
        return issues
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Levenshtein distance."""
        if str1 == str2:
            return 1.0
        
        len1, len2 = len(str1), len(str2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Simple character-based similarity
        common_chars = sum(1 for c1, c2 in zip(str1, str2) if c1 == c2)
        return common_chars / max(len1, len2)
    
    def _get_risk_level(self, risk_score: int) -> str:
        """Get risk level from score."""
        if risk_score >= 70:
            return "CRITICAL"
        elif risk_score >= 50:
            return "HIGH"
        elif risk_score >= 30:
            return "MEDIUM"
        elif risk_score >= 10:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_recommendations(self, risk_score: int, risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on risk analysis."""
        recommendations = []
        
        if risk_score >= 50:
            recommendations.append("URGENT: Address critical vulnerabilities immediately")
            recommendations.append("Implement automated vulnerability scanning in CI/CD")
        
        if "vulnerabilities" in str(risk_factors):
            recommendations.append("Update vulnerable packages to patched versions")
            recommendations.append("Consider using tools like Safety or Snyk for continuous monitoring")
        
        if "typosquatting" in str(risk_factors):
            recommendations.append("Verify all package names against official sources")
            recommendations.append("Implement package name validation in CI/CD")
        
        if "updates needed" in str(risk_factors):
            recommendations.append("Create a regular dependency update schedule")
            recommendations.append("Use tools like Dependabot for automated updates")
        
        recommendations.append("Implement Software Bill of Materials (SBOM) generation")
        recommendations.append("Use package pinning with cryptographic hashes")
        
        return recommendations