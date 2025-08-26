"""
Dependency Security Scanner

Scans dependencies for known vulnerabilities and outdated packages.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Vulnerability:
    """Represents a dependency vulnerability."""
    package: str
    version: str
    severity: str  # critical, high, medium, low
    cve_id: Optional[str]
    description: str
    fixed_version: Optional[str]
    published_date: Optional[str]
    

@dataclass 
class Dependency:
    """Represents a project dependency."""
    name: str
    version: str
    source: str  # requirements.txt, package.json, etc.
    is_direct: bool
    vulnerabilities: List[Vulnerability]
    latest_version: Optional[str]
    

class DependencyScanner:
    """
    Scans project dependencies for security vulnerabilities.
    Supports Python, JavaScript, and other package managers.
    """
    
    def __init__(self):
        """Initialize the dependency scanner."""
        self.dependencies = []
        self.vulnerabilities = []
        self.vulnerability_db = self._load_vulnerability_db()
        logger.info("Dependency Scanner initialized")
        
    def scan_python_dependencies(self, project_path: str) -> List[Dependency]:
        """
        Scan Python dependencies.
        
        Args:
            project_path: Path to project
            
        Returns:
            List of dependencies with vulnerabilities
        """
        deps = []
        
        # Check requirements.txt
        req_file = Path(project_path) / "requirements.txt"
        if req_file.exists():
            deps.extend(self._parse_requirements(req_file))
            
        # Check Pipfile
        pipfile = Path(project_path) / "Pipfile"
        if pipfile.exists():
            deps.extend(self._parse_pipfile(pipfile))
            
        # Check setup.py
        setup_file = Path(project_path) / "setup.py"
        if setup_file.exists():
            deps.extend(self._parse_setup_py(setup_file))
            
        # Check for vulnerabilities
        for dep in deps:
            dep.vulnerabilities = self._check_vulnerabilities(dep)
            
        self.dependencies.extend(deps)
        return deps
        
    def scan_javascript_dependencies(self, project_path: str) -> List[Dependency]:
        """
        Scan JavaScript/Node.js dependencies.
        
        Args:
            project_path: Path to project
            
        Returns:
            List of dependencies with vulnerabilities
        """
        deps = []
        
        # Check package.json
        package_file = Path(project_path) / "package.json"
        if package_file.exists():
            deps.extend(self._parse_package_json(package_file))
            
        # Check package-lock.json for exact versions
        lock_file = Path(project_path) / "package-lock.json"
        if lock_file.exists():
            self._update_from_lock_file(deps, lock_file)
            
        # Check for vulnerabilities
        for dep in deps:
            dep.vulnerabilities = self._check_vulnerabilities(dep)
            
        self.dependencies.extend(deps)
        return deps
        
    def _parse_requirements(self, file_path: Path) -> List[Dependency]:
        """Parse requirements.txt file."""
        deps = []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dep = self._parse_requirement_line(line)
                        if dep:
                            dep.source = "requirements.txt"
                            deps.append(dep)
                            
        except Exception as e:
            logger.error(f"Error parsing requirements: {e}")
            
        return deps
        
    def _parse_requirement_line(self, line: str) -> Optional[Dependency]:
        """Parse a single requirement line."""
        # Handle different formats: package==1.0, package>=1.0, package~=1.0
        patterns = [
            r'^([a-zA-Z0-9\-_]+)==([0-9\.]+)',
            r'^([a-zA-Z0-9\-_]+)>=([0-9\.]+)',
            r'^([a-zA-Z0-9\-_]+)~=([0-9\.]+)',
            r'^([a-zA-Z0-9\-_]+)<([0-9\.]+)',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                return Dependency(
                    name=match.group(1),
                    version=match.group(2),
                    source="",
                    is_direct=True,
                    vulnerabilities=[],
                    latest_version=None
                )
                
        # Handle package without version
        if re.match(r'^[a-zA-Z0-9\-_]+$', line):
            return Dependency(
                name=line,
                version="*",
                source="",
                is_direct=True,
                vulnerabilities=[],
                latest_version=None
            )
            
        return None
        
    def _parse_package_json(self, file_path: Path) -> List[Dependency]:
        """Parse package.json file."""
        deps = []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Parse dependencies
            for name, version in data.get('dependencies', {}).items():
                deps.append(Dependency(
                    name=name,
                    version=self._clean_version(version),
                    source="package.json",
                    is_direct=True,
                    vulnerabilities=[],
                    latest_version=None
                ))
                
            # Parse devDependencies
            for name, version in data.get('devDependencies', {}).items():
                deps.append(Dependency(
                    name=name,
                    version=self._clean_version(version),
                    source="package.json (dev)",
                    is_direct=True,
                    vulnerabilities=[],
                    latest_version=None
                ))
                
        except Exception as e:
            logger.error(f"Error parsing package.json: {e}")
            
        return deps
        
    def _check_vulnerabilities(self, dep: Dependency) -> List[Vulnerability]:
        """Check dependency for known vulnerabilities."""
        vulns = []
        
        # Check internal database
        if dep.name in self.vulnerability_db:
            for vuln in self.vulnerability_db[dep.name]:
                if self._version_affected(dep.version, vuln):
                    vulns.append(vuln)
                    
        # Add common vulnerability checks
        vulns.extend(self._check_common_vulnerabilities(dep))
        
        return vulns
        
    def _check_common_vulnerabilities(self, dep: Dependency) -> List[Vulnerability]:
        """Check for common vulnerability patterns."""
        vulns = []
        
        # Check for known vulnerable packages
        vulnerable_packages = {
            'requests': {'<2.20.0': 'CVE-2018-18074'},
            'django': {'<2.2.10': 'CVE-2020-7471'},
            'flask': {'<1.0': 'CVE-2018-1000656'},
            'pyyaml': {'<5.4': 'CVE-2020-14343'},
            'urllib3': {'<1.26.5': 'CVE-2021-33503'},
            'pillow': {'<8.3.2': 'CVE-2021-34552'},
            'numpy': {'<1.19.0': 'CVE-2021-33430'}
        }
        
        if dep.name.lower() in vulnerable_packages:
            for affected_version, cve in vulnerable_packages[dep.name.lower()].items():
                if self._version_matches(dep.version, affected_version):
                    vulns.append(Vulnerability(
                        package=dep.name,
                        version=dep.version,
                        severity="high",
                        cve_id=cve,
                        description=f"Known vulnerability in {dep.name}",
                        fixed_version=affected_version.replace('<', ''),
                        published_date=None
                    ))
                    
        return vulns
        
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate dependency security report.
        
        Returns:
            Security report
        """
        total_deps = len(self.dependencies)
        vulnerable_deps = [d for d in self.dependencies if d.vulnerabilities]
        
        severity_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }
        
        for dep in vulnerable_deps:
            for vuln in dep.vulnerabilities:
                severity_counts[vuln.severity] += 1
                
        return {
            'scan_date': datetime.now().isoformat(),
            'total_dependencies': total_deps,
            'vulnerable_dependencies': len(vulnerable_deps),
            'vulnerabilities': {
                'total': sum(severity_counts.values()),
                'by_severity': severity_counts
            },
            'dependencies': [
                {
                    'name': dep.name,
                    'version': dep.version,
                    'source': dep.source,
                    'vulnerabilities': len(dep.vulnerabilities),
                    'severity': max((v.severity for v in dep.vulnerabilities), default='none')
                }
                for dep in vulnerable_deps
            ]
        }
        
    def suggest_updates(self) -> List[Dict[str, str]]:
        """
        Suggest dependency updates to fix vulnerabilities.
        
        Returns:
            List of update suggestions
        """
        suggestions = []
        
        for dep in self.dependencies:
            if dep.vulnerabilities:
                # Find minimum safe version
                safe_version = None
                for vuln in dep.vulnerabilities:
                    if vuln.fixed_version:
                        if not safe_version or vuln.fixed_version > safe_version:
                            safe_version = vuln.fixed_version
                            
                if safe_version:
                    suggestions.append({
                        'package': dep.name,
                        'current': dep.version,
                        'suggested': safe_version,
                        'reason': f"Fixes {len(dep.vulnerabilities)} vulnerabilities"
                    })
                    
        return suggestions
        
    def check_license_compliance(self, allowed_licenses: List[str]) -> List[Dict[str, str]]:
        """
        Check dependency license compliance.
        
        Args:
            allowed_licenses: List of allowed license types
            
        Returns:
            List of non-compliant dependencies
        """
        non_compliant = []
        
        # This would need actual license detection
        # Placeholder implementation
        for dep in self.dependencies:
            license_type = self._detect_license(dep)
            if license_type and license_type not in allowed_licenses:
                non_compliant.append({
                    'package': dep.name,
                    'version': dep.version,
                    'license': license_type
                })
                
        return non_compliant
        
    # Helper methods
    def _load_vulnerability_db(self) -> Dict[str, List[Vulnerability]]:
        """Load vulnerability database."""
        # This would load from a real vulnerability database
        # Placeholder for demonstration
        return {}
        
    def _clean_version(self, version: str) -> str:
        """Clean version string."""
        return version.strip().lstrip('^~>=<')
        
    def _version_affected(self, version: str, vuln: Vulnerability) -> bool:
        """Check if version is affected by vulnerability."""
        # Simple version comparison - would need proper semver comparison
        return version == vuln.version
        
    def _version_matches(self, version: str, pattern: str) -> bool:
        """Check if version matches pattern."""
        # Simplified version matching
        if pattern.startswith('<'):
            return version < pattern[1:]
        elif pattern.startswith('>'):
            return version > pattern[1:]
        return version == pattern
        
    def _parse_pipfile(self, file_path: Path) -> List[Dependency]:
        """Parse Pipfile."""
        return []
        
    def _parse_setup_py(self, file_path: Path) -> List[Dependency]:
        """Parse setup.py."""
        return []
        
    def _update_from_lock_file(self, deps: List[Dependency], lock_file: Path) -> None:
        """Update dependency versions from lock file."""
        pass
        
    def _detect_license(self, dep: Dependency) -> Optional[str]:
        """Detect dependency license."""
        return None