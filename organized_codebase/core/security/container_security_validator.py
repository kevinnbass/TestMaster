"""
Llama-Agents Derived Container Security Validator
Extracted from Llama-Agents container security patterns and orchestration safety
Enhanced for comprehensive container validation and runtime security
"""

import logging
import os
import json
import subprocess
import hashlib
from typing import Dict, Any, Optional, List, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from .error_handler import SecurityError, ValidationError, security_error_handler


class ContainerSecurityLevel(Enum):
    """Container security levels based on Llama-Agents patterns"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CRITICAL = "critical"


class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContainerStatus(Enum):
    """Container security validation status"""
    SECURE = "secure"
    VULNERABLE = "vulnerable"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


@dataclass
class SecurityVulnerability:
    """Container security vulnerability representation"""
    vuln_id: str
    severity: VulnerabilitySeverity
    description: str
    package: Optional[str] = None
    version: Optional[str] = None
    fixed_version: Optional[str] = None
    cve_id: Optional[str] = None
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_critical(self) -> bool:
        """Check if vulnerability is critical"""
        return self.severity in [VulnerabilitySeverity.HIGH, VulnerabilitySeverity.CRITICAL]


@dataclass
class ContainerConfiguration:
    """Container configuration based on Llama-Agents patterns"""
    base_image: str
    image_tag: str
    dockerfile_path: str
    user_context: str = "nobody"
    exposed_ports: List[int] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=list)
    security_options: List[str] = field(default_factory=list)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def runs_as_root(self) -> bool:
        """Check if container runs as root"""
        return self.user_context in ["root", "0", ""]
    
    @property
    def has_privileged_ports(self) -> bool:
        """Check if container exposes privileged ports"""
        return any(port < 1024 for port in self.exposed_ports)


@dataclass
class ContainerScanResult:
    """Container security scan result"""
    container_id: str
    image_name: str
    scan_timestamp: datetime
    security_level: ContainerSecurityLevel
    status: ContainerStatus
    vulnerabilities: List[SecurityVulnerability] = field(default_factory=list)
    configuration_issues: List[str] = field(default_factory=list)
    compliance_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def critical_vulns_count(self) -> int:
        """Count critical vulnerabilities"""
        return sum(1 for vuln in self.vulnerabilities if vuln.is_critical)
    
    @property
    def is_compliant(self) -> bool:
        """Check if container meets compliance standards"""
        return (self.compliance_score >= 80.0 and 
                self.critical_vulns_count == 0 and 
                self.status == ContainerStatus.SECURE)


class DockerfileSecurityAnalyzer:
    """Dockerfile security analysis based on Llama-Agents patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.security_rules = self._initialize_security_rules()
    
    def analyze_dockerfile(self, dockerfile_path: str) -> Dict[str, Any]:
        """Analyze Dockerfile for security issues"""
        try:
            if not Path(dockerfile_path).exists():
                raise ValidationError(f"Dockerfile not found: {dockerfile_path}")
            
            with open(dockerfile_path, 'r') as f:
                dockerfile_content = f.read()
            
            issues = []
            recommendations = []
            security_score = 100.0
            
            lines = dockerfile_content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Analyze each instruction
                instruction_issues = self._analyze_instruction(line, line_num)
                issues.extend(instruction_issues)
            
            # Calculate security score
            security_score -= len([i for i in issues if i['severity'] == 'critical']) * 20
            security_score -= len([i for i in issues if i['severity'] == 'high']) * 10
            security_score -= len([i for i in issues if i['severity'] == 'medium']) * 5
            security_score -= len([i for i in issues if i['severity'] == 'low']) * 2
            
            security_score = max(0.0, security_score)
            
            # Generate recommendations
            recommendations = self._generate_dockerfile_recommendations(issues)
            
            return {
                'dockerfile_path': dockerfile_path,
                'security_score': security_score,
                'total_issues': len(issues),
                'issues_by_severity': self._categorize_issues_by_severity(issues),
                'issues': issues,
                'recommendations': recommendations,
                'compliant': security_score >= 80.0 and 
                           len([i for i in issues if i['severity'] == 'critical']) == 0
            }
            
        except Exception as e:
            error = SecurityError(f"Dockerfile analysis failed: {str(e)}", "DOCKERFILE_ANALYSIS_001")
            security_error_handler.handle_error(error)
            return {'error': str(e)}
    
    def _analyze_instruction(self, instruction: str, line_num: int) -> List[Dict[str, Any]]:
        """Analyze individual Dockerfile instruction"""
        issues = []
        instruction_upper = instruction.upper()
        
        try:
            # Check for running as root
            if instruction_upper.startswith('USER '):
                user = instruction.split()[1] if len(instruction.split()) > 1 else ""
                if user in ['root', '0'] or not user:
                    issues.append({
                        'line': line_num,
                        'instruction': instruction,
                        'severity': 'high',
                        'issue': 'Container runs as root user',
                        'description': 'Running as root increases security risk'
                    })
            
            # Check for latest tag usage
            if instruction_upper.startswith('FROM '):
                image = instruction.split()[1] if len(instruction.split()) > 1 else ""
                if ':latest' in image or ':' not in image:
                    issues.append({
                        'line': line_num,
                        'instruction': instruction,
                        'severity': 'medium',
                        'issue': 'Using latest tag or no tag',
                        'description': 'Use specific version tags for reproducibility'
                    })
            
            # Check for ADD vs COPY
            if instruction_upper.startswith('ADD '):
                issues.append({
                    'line': line_num,
                    'instruction': instruction,
                    'severity': 'low',
                    'issue': 'Using ADD instead of COPY',
                    'description': 'COPY is preferred over ADD for simple file copying'
                })
            
            # Check for package manager cache cleanup
            if 'apt-get install' in instruction and 'rm -rf /var/lib/apt/lists/*' not in instruction:
                issues.append({
                    'line': line_num,
                    'instruction': instruction,
                    'severity': 'medium',
                    'issue': 'Package manager cache not cleaned',
                    'description': 'Clean package manager cache to reduce image size'
                })
            
            # Check for exposed privileged ports
            if instruction_upper.startswith('EXPOSE '):
                ports_str = instruction.replace('EXPOSE', '').strip()
                try:
                    ports = [int(p.strip()) for p in ports_str.split() if p.strip().isdigit()]
                    for port in ports:
                        if port < 1024:
                            issues.append({
                                'line': line_num,
                                'instruction': instruction,
                                'severity': 'medium',
                                'issue': f'Exposing privileged port {port}',
                                'description': 'Privileged ports (<1024) require root access'
                            })
                except ValueError:
                    pass
            
            # Check for hardcoded secrets
            sensitive_patterns = ['password', 'secret', 'key', 'token', 'api_key']
            instruction_lower = instruction.lower()
            for pattern in sensitive_patterns:
                if pattern in instruction_lower and '=' in instruction:
                    issues.append({
                        'line': line_num,
                        'instruction': instruction,
                        'severity': 'critical',
                        'issue': f'Potential hardcoded {pattern}',
                        'description': 'Use environment variables or secrets for sensitive data'
                    })
            
        except Exception as e:
            self.logger.error(f"Error analyzing instruction at line {line_num}: {e}")
        
        return issues
    
    def _categorize_issues_by_severity(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize issues by severity"""
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for issue in issues:
            severity = issue.get('severity', 'low')
            severity_counts[severity] += 1
        
        return severity_counts
    
    def _generate_dockerfile_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on issues"""
        recommendations = []
        
        # Group issues by type
        issue_types = set(issue['issue'] for issue in issues)
        
        recommendation_map = {
            'Container runs as root user': 'Add "USER nobody" or create a non-root user',
            'Using latest tag or no tag': 'Use specific version tags like "python:3.9-slim"',
            'Using ADD instead of COPY': 'Use COPY instead of ADD for simple file operations',
            'Package manager cache not cleaned': 'Add cache cleanup: && rm -rf /var/lib/apt/lists/*',
            'Potential hardcoded secret': 'Use ARG, ENV, or Docker secrets for sensitive data'
        }
        
        for issue_type in issue_types:
            for pattern, recommendation in recommendation_map.items():
                if pattern in issue_type:
                    recommendations.append(recommendation)
                    break
        
        # Add general recommendations
        if any('privileged port' in issue['issue'] for issue in issues):
            recommendations.append('Use non-privileged ports (>1024) when possible')
        
        return list(set(recommendations))  # Remove duplicates
    
    def _initialize_security_rules(self) -> Dict[str, Any]:
        """Initialize security analysis rules"""
        return {
            'forbidden_base_images': [
                'ubuntu:latest', 'centos:latest', 'debian:latest'
            ],
            'recommended_base_images': [
                'python:3.9-slim', 'node:16-alpine', 'golang:1.19-alpine'
            ],
            'security_scanners': [
                'trivy', 'clair', 'anchore'
            ]
        }


class ContainerRuntimeValidator:
    """Container runtime security validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.docker_available = self._check_docker_availability()
    
    def validate_container_runtime(self, container_config: ContainerConfiguration) -> ContainerScanResult:
        """Validate container runtime configuration"""
        try:
            container_id = hashlib.sha256(f"{container_config.base_image}:{container_config.image_tag}".encode()).hexdigest()[:12]
            
            scan_result = ContainerScanResult(
                container_id=container_id,
                image_name=f"{container_config.base_image}:{container_config.image_tag}",
                scan_timestamp=datetime.utcnow(),
                security_level=ContainerSecurityLevel.PRODUCTION,
                status=ContainerStatus.UNKNOWN
            )
            
            # Validate configuration
            config_issues = self._validate_configuration(container_config)
            scan_result.configuration_issues.extend(config_issues)
            
            # Generate security recommendations
            recommendations = self._generate_runtime_recommendations(container_config, config_issues)
            scan_result.recommendations.extend(recommendations)
            
            # Calculate compliance score
            compliance_score = self._calculate_compliance_score(container_config, config_issues)
            scan_result.compliance_score = compliance_score
            
            # Determine status
            if len(config_issues) == 0:
                scan_result.status = ContainerStatus.SECURE
            elif any('critical' in issue.lower() for issue in config_issues):
                scan_result.status = ContainerStatus.BLOCKED
            else:
                scan_result.status = ContainerStatus.VULNERABLE
            
            return scan_result
            
        except Exception as e:
            error = SecurityError(f"Container runtime validation failed: {str(e)}", "RUNTIME_VALIDATION_001")
            security_error_handler.handle_error(error)
            
            # Return error result
            return ContainerScanResult(
                container_id="error",
                image_name="error",
                scan_timestamp=datetime.utcnow(),
                security_level=ContainerSecurityLevel.DEVELOPMENT,
                status=ContainerStatus.UNKNOWN,
                configuration_issues=[f"Validation error: {str(e)}"]
            )
    
    def _validate_configuration(self, config: ContainerConfiguration) -> List[str]:
        """Validate container configuration for security issues"""
        issues = []
        
        # Check if running as root
        if config.runs_as_root:
            issues.append("CRITICAL: Container configured to run as root user")
        
        # Check privileged ports
        if config.has_privileged_ports:
            privileged_ports = [p for p in config.exposed_ports if p < 1024]
            issues.append(f"HIGH: Container exposes privileged ports: {privileged_ports}")
        
        # Check for overly permissive security options
        dangerous_options = ['--privileged', '--cap-add=ALL', '--security-opt=seccomp:unconfined']
        for option in config.security_options:
            if any(dangerous in option for dangerous in dangerous_options):
                issues.append(f"CRITICAL: Dangerous security option: {option}")
        
        # Check environment variables for secrets
        for key, value in config.environment_vars.items():
            if any(secret_word in key.lower() for secret_word in ['password', 'secret', 'key', 'token']):
                if len(value) > 0:  # Don't flag empty values
                    issues.append(f"HIGH: Potential secret in environment variable: {key}")
        
        # Check resource limits
        if not config.resource_limits:
            issues.append("MEDIUM: No resource limits configured")
        else:
            if 'memory' not in config.resource_limits:
                issues.append("MEDIUM: No memory limit configured")
            if 'cpus' not in config.resource_limits:
                issues.append("MEDIUM: No CPU limit configured")
        
        # Check volumes
        for volume in config.volumes:
            if volume.startswith('/'):  # Host path mount
                if any(sensitive_path in volume for sensitive_path in ['/etc', '/var/run/docker.sock', '/proc', '/sys']):
                    issues.append(f"CRITICAL: Mounting sensitive host path: {volume}")
        
        return issues
    
    def _calculate_compliance_score(self, config: ContainerConfiguration, issues: List[str]) -> float:
        """Calculate container compliance score"""
        score = 100.0
        
        # Deduct points for issues
        for issue in issues:
            if issue.startswith('CRITICAL:'):
                score -= 25.0
            elif issue.startswith('HIGH:'):
                score -= 15.0
            elif issue.startswith('MEDIUM:'):
                score -= 10.0
            elif issue.startswith('LOW:'):
                score -= 5.0
        
        return max(0.0, score)
    
    def _generate_runtime_recommendations(self, config: ContainerConfiguration, 
                                        issues: List[str]) -> List[str]:
        """Generate runtime security recommendations"""
        recommendations = []
        
        if config.runs_as_root:
            recommendations.append("Create and use a non-root user in the container")
        
        if config.has_privileged_ports:
            recommendations.append("Use non-privileged ports (>1024) or use a reverse proxy")
        
        if not config.resource_limits:
            recommendations.append("Configure memory and CPU limits to prevent resource exhaustion")
        
        if any('--privileged' in opt for opt in config.security_options):
            recommendations.append("Avoid using --privileged flag, use specific capabilities instead")
        
        # Check for health checks
        if 'healthcheck' not in str(config).lower():
            recommendations.append("Add health check configuration for container monitoring")
        
        # General security recommendations
        recommendations.extend([
            "Use multi-stage builds to minimize image size",
            "Scan images for vulnerabilities before deployment",
            "Use read-only root filesystem when possible",
            "Enable container logging and monitoring"
        ])
        
        return recommendations
    
    def _check_docker_availability(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            self.logger.warning("Docker not available for container validation")
            return False


class ContainerSecurityValidator:
    """Comprehensive container security validation system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dockerfile_analyzer = DockerfileSecurityAnalyzer()
        self.runtime_validator = ContainerRuntimeValidator()
        self.scan_history: List[ContainerScanResult] = []
        self.max_history = 1000
    
    def validate_container_security(self, dockerfile_path: str = None,
                                  container_config: ContainerConfiguration = None) -> Dict[str, Any]:
        """Comprehensive container security validation"""
        try:
            validation_result = {
                'timestamp': datetime.utcnow().isoformat(),
                'dockerfile_analysis': None,
                'runtime_validation': None,
                'overall_security_score': 0.0,
                'compliance_status': 'unknown',
                'recommendations': []
            }
            
            # Dockerfile security analysis
            if dockerfile_path:
                dockerfile_result = self.dockerfile_analyzer.analyze_dockerfile(dockerfile_path)
                validation_result['dockerfile_analysis'] = dockerfile_result
            
            # Runtime configuration validation
            if container_config:
                runtime_result = self.runtime_validator.validate_container_runtime(container_config)
                validation_result['runtime_validation'] = {
                    'container_id': runtime_result.container_id,
                    'status': runtime_result.status.value,
                    'compliance_score': runtime_result.compliance_score,
                    'configuration_issues': runtime_result.configuration_issues,
                    'recommendations': runtime_result.recommendations
                }
                
                # Add to scan history
                self._add_to_scan_history(runtime_result)
            
            # Calculate overall security score
            overall_score = self._calculate_overall_score(
                validation_result.get('dockerfile_analysis'),
                validation_result.get('runtime_validation')
            )
            validation_result['overall_security_score'] = overall_score
            
            # Determine compliance status
            validation_result['compliance_status'] = self._determine_compliance_status(overall_score)
            
            # Aggregate recommendations
            all_recommendations = []
            if dockerfile_path and validation_result['dockerfile_analysis']:
                all_recommendations.extend(validation_result['dockerfile_analysis'].get('recommendations', []))
            
            if container_config and validation_result['runtime_validation']:
                all_recommendations.extend(validation_result['runtime_validation'].get('recommendations', []))
            
            validation_result['recommendations'] = list(set(all_recommendations))
            
            return validation_result
            
        except Exception as e:
            error = SecurityError(f"Container security validation failed: {str(e)}", "CONTAINER_VALIDATION_001")
            security_error_handler.handle_error(error)
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'overall_security_score': 0.0,
                'compliance_status': 'error'
            }
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get container security statistics"""
        try:
            if not self.scan_history:
                return {'total_scans': 0}
            
            total_scans = len(self.scan_history)
            compliant_scans = sum(1 for scan in self.scan_history if scan.is_compliant)
            
            # Status distribution
            status_counts = {}
            for scan in self.scan_history:
                status = scan.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Average compliance score
            avg_compliance = sum(scan.compliance_score for scan in self.scan_history) / total_scans
            
            # Vulnerability statistics
            total_vulns = sum(len(scan.vulnerabilities) for scan in self.scan_history)
            critical_vulns = sum(scan.critical_vulns_count for scan in self.scan_history)
            
            return {
                'total_scans': total_scans,
                'compliant_containers': compliant_scans,
                'compliance_rate_pct': (compliant_scans / total_scans) * 100,
                'average_compliance_score': avg_compliance,
                'status_distribution': status_counts,
                'vulnerability_stats': {
                    'total_vulnerabilities': total_vulns,
                    'critical_vulnerabilities': critical_vulns,
                    'average_vulns_per_container': total_vulns / total_scans if total_scans > 0 else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating container security statistics: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_score(self, dockerfile_analysis: Dict[str, Any], 
                               runtime_validation: Dict[str, Any]) -> float:
        """Calculate overall security score"""
        scores = []
        
        if dockerfile_analysis and 'security_score' in dockerfile_analysis:
            scores.append(dockerfile_analysis['security_score'])
        
        if runtime_validation and 'compliance_score' in runtime_validation:
            scores.append(runtime_validation['compliance_score'])
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _determine_compliance_status(self, overall_score: float) -> str:
        """Determine compliance status based on score"""
        if overall_score >= 90:
            return 'excellent'
        elif overall_score >= 80:
            return 'compliant'
        elif overall_score >= 60:
            return 'warning'
        else:
            return 'non_compliant'
    
    def _add_to_scan_history(self, scan_result: ContainerScanResult):
        """Add scan result to history"""
        self.scan_history.append(scan_result)
        
        # Limit history size
        if len(self.scan_history) > self.max_history:
            self.scan_history = self.scan_history[-self.max_history // 2:]


# Global container security validator
container_security_validator = ContainerSecurityValidator()


# Convenience functions
def validate_dockerfile_security(dockerfile_path: str) -> Dict[str, Any]:
    """Convenience function to validate Dockerfile security"""
    return container_security_validator.validate_container_security(dockerfile_path=dockerfile_path)


def validate_container_config(base_image: str, image_tag: str, user_context: str = "nobody") -> Dict[str, Any]:
    """Convenience function to validate container configuration"""
    config = ContainerConfiguration(
        base_image=base_image,
        image_tag=image_tag,
        dockerfile_path="",
        user_context=user_context
    )
    return container_security_validator.validate_container_security(container_config=config)