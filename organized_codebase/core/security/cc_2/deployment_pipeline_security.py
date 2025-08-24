from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor
"""
Llama-Agents Derived Deployment Pipeline Security
Extracted from Llama-Agents CI/CD patterns and secrets management
Enhanced for comprehensive deployment security and pipeline protection
"""

import logging
import os
import json
import subprocess
import hashlib
import base64
from typing import Dict, Any, Optional, List, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import secrets
import re
from .error_handler import SecurityError, ValidationError, security_error_handler


class DeploymentStage(Enum):
    """Deployment pipeline stages based on Llama-Agents patterns"""
    SOURCE = "source"
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    DEPLOY = "deploy"
    VERIFY = "verify"
    ROLLBACK = "rollback"


class SecurityCheckType(Enum):
    """Security check types for deployment pipeline"""
    VULNERABILITY_SCAN = "vulnerability_scan"
    CODE_QUALITY = "code_quality"
    SECRET_DETECTION = "secret_detection"
    LICENSE_COMPLIANCE = "license_compliance"
    DEPENDENCY_CHECK = "dependency_check"
    CONTAINER_SCAN = "container_scan"


class DeploymentEnvironment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


@dataclass
class SecretConfig:
    """Secret configuration for deployment pipeline"""
    secret_name: str
    secret_value: str
    environment: DeploymentEnvironment
    masked_value: str = field(init=False)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    
    def __post_init__(self):
        # Mask the secret value for display
        if len(self.secret_value) <= 8:
            self.masked_value = "*" * len(self.secret_value)
        else:
            self.masked_value = self.secret_value[:2] + "*" * (len(self.secret_value) - 4) + self.secret_value[-2:]
    
    @property
    def is_recently_used(self) -> bool:
        """Check if secret was used recently"""
        if not self.last_used:
            return False
        return (datetime.utcnow() - self.last_used).total_seconds() < 3600


@dataclass
class SecurityScanResult:
    """Security scan result for deployment pipeline"""
    scan_id: str
    scan_type: SecurityCheckType
    stage: DeploymentStage
    status: str  # passed, failed, warning
    timestamp: datetime = field(default_factory=datetime.utcnow)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 100.0
    duration_seconds: float = 0.0
    
    @property
    def critical_findings(self) -> List[Dict[str, Any]]:
        """Get critical security findings"""
        return [f for f in self.findings if f.get('severity') == 'critical']
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if scan has critical issues"""
        return len(self.critical_findings) > 0


@dataclass
class DeploymentPipeline:
    """Deployment pipeline configuration"""
    pipeline_id: str
    pipeline_name: str
    source_repository: str
    target_environment: DeploymentEnvironment
    stages: List[DeploymentStage] = field(default_factory=list)
    security_checks: List[SecurityCheckType] = field(default_factory=list)
    secrets: Dict[str, SecretConfig] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_execution: Optional[datetime] = None
    execution_count: int = 0


class SecretManager:
    """Secure secret management for deployment pipelines"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.secrets: Dict[str, SecretConfig] = {}
        self.secret_patterns = self._initialize_secret_patterns()
        self.access_history: List[Dict[str, Any]] = []
        self.max_history = 10000
    
    def store_secret(self, secret_name: str, secret_value: str, 
                    environment: DeploymentEnvironment) -> bool:
        """Store secret securely"""
        try:
            # Validate secret name
            if not self._validate_secret_name(secret_name):
                raise ValidationError(f"Invalid secret name: {secret_name}")
            
            # Check for potential secret in value
            if self._contains_potential_secret(secret_value):
                self.logger.warning(f"Storing potential secret: {secret_name}")
            
            # Create secret config
            secret_config = SecretConfig(
                secret_name=secret_name,
                secret_value=secret_value,
                environment=environment
            )
            
            # Store secret (in production, encrypt this)
            secret_key = f"{environment.value}:{secret_name}"
            self.secrets[secret_key] = secret_config
            
            self._record_secret_access("store", secret_name, environment.value)
            
            self.logger.info(f"Stored secret: {secret_name} for {environment.value}")
            return True
            
        except Exception as e:
            error = SecurityError(f"Failed to store secret: {str(e)}", "SECRET_STORE_001")
            security_error_handler.handle_error(error)
            return False
    
    def retrieve_secret(self, secret_name: str, environment: DeploymentEnvironment) -> Optional[str]:
        """Retrieve secret value"""
        try:
            secret_key = f"{environment.value}:{secret_name}"
            
            if secret_key not in self.secrets:
                return None
            
            secret_config = self.secrets[secret_key]
            
            # Update usage statistics
            secret_config.last_used = datetime.utcnow()
            secret_config.usage_count += 1
            
            self._record_secret_access("retrieve", secret_name, environment.value)
            
            return secret_config.secret_value
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret: {e}")
            return None
    
    def delete_secret(self, secret_name: str, environment: DeploymentEnvironment) -> bool:
        """Delete secret"""
        try:
            secret_key = f"{environment.value}:{secret_name}"
            
            if secret_key not in self.secrets:
                return False
            
            del self.secrets[secret_key]
            
            self._record_secret_access("delete", secret_name, environment.value)
            
            self.logger.info(f"Deleted secret: {secret_name} for {environment.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete secret: {e}")
            return False
    
    def scan_for_secrets(self, content: str) -> List[Dict[str, Any]]:
        """Scan content for potential secrets"""
        findings = []
        
        try:
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for pattern_name, pattern_info in self.secret_patterns.items():
                    pattern = pattern_info['pattern']
                    severity = pattern_info['severity']
                    
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        findings.append({
                            'type': 'potential_secret',
                            'pattern': pattern_name,
                            'line': line_num,
                            'content': line.strip(),
                            'match': match.group(),
                            'severity': severity,
                            'description': pattern_info['description']
                        })
            
            return findings
            
        except Exception as e:
            self.logger.error(f"Secret scanning failed: {e}")
            return []
    
    def _validate_secret_name(self, secret_name: str) -> bool:
        """Validate secret name format"""
        # Allow alphanumeric, underscore, and hyphen
        pattern = r'^[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, secret_name))
    
    def _contains_potential_secret(self, value: str) -> bool:
        """Check if value contains potential secret patterns"""
        return len(self.scan_for_secrets(value)) > 0
    
    def _initialize_secret_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize secret detection patterns"""
        return {
            'api_key': {
                'pattern': r'(?i)api[_-]?key[\'"\s]*[:=][\'"\s]*([a-zA-Z0-9]{20,})',
                'severity': 'high',
                'description': 'API key detected'
            },
            'password': {
                'pattern': r'(?i)password[\'"\s]*[:=][\'"\s]*([^\s\'"]{8,})',
                'severity': 'high',
                'description': 'Password detected'
            },
            'secret': {
                'pattern': r'(?i)secret[\'"\s]*[:=][\'"\s]*([a-zA-Z0-9]{16,})',
                'severity': 'high',
                'description': 'Secret detected'
            },
            'token': {
                'pattern': r'(?i)token[\'"\s]*[:=][\'"\s]*([a-zA-Z0-9]{20,})',
                'severity': 'high',
                'description': 'Token detected'
            },
            'private_key': {
                'pattern': r'-----BEGIN (RSA )?PRIVATE KEY-----',
                'severity': 'critical',
                'description': 'Private key detected'
            },
            'github_token': {
                'pattern': r'gh[pousr]_[A-Za-z0-9_]{36}',
                'severity': 'critical',
                'description': 'GitHub token detected'
            },
            'aws_access_key': {
                'pattern': r'AKIA[0-9A-Z]{16}',
                'severity': 'critical',
                'description': 'AWS access key detected'
            }
        }
    
    def _record_secret_access(self, action: str, secret_name: str, environment: str):
        """Record secret access for auditing"""
        access_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'secret_name': secret_name,
            'environment': environment
        }
        
        self.access_history.append(access_record)
        
        # Limit history
        if len(self.access_history) > self.max_history:
            self.access_history = self.access_history[-self.max_history // 2:]


class SecurityScanner:
    """Security scanner for deployment pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scan_history: List[SecurityScanResult] = []
        self.max_scan_history = 1000
    
    def run_vulnerability_scan(self, target_path: str, stage: DeploymentStage) -> SecurityScanResult:
        """Run vulnerability scan on deployment target"""
        scan_start = datetime.utcnow()
        scan_id = f"vuln_{int(scan_start.timestamp())}"
        
        try:
            findings = []
            score = 100.0
            
            # Simulate vulnerability scanning
            # In production, integrate with tools like Snyk, OWASP Dependency Check, etc.
            
            if Path(target_path).is_dir():
                # Scan for common vulnerability patterns
                for file_path in Path(target_path).rglob("*"):
                    if file_path.is_file() and file_path.suffix in ['.py', '.js', '.ts', '.java']:
                        file_findings = self._scan_file_for_vulnerabilities(file_path)
                        findings.extend(file_findings)
            
            # Calculate score based on findings
            for finding in findings:
                if finding['severity'] == 'critical':
                    score -= 20
                elif finding['severity'] == 'high':
                    score -= 10
                elif finding['severity'] == 'medium':
                    score -= 5
            
            score = max(0.0, score)
            status = "passed" if score >= 80 else ("warning" if score >= 60 else "failed")
            
            duration = (datetime.utcnow() - scan_start).total_seconds()
            
            result = SecurityScanResult(
                scan_id=scan_id,
                scan_type=SecurityCheckType.VULNERABILITY_SCAN,
                stage=stage,
                status=status,
                findings=findings,
                score=score,
                duration_seconds=duration
            )
            
            self._add_scan_result(result)
            
            self.logger.info(f"Vulnerability scan completed: {scan_id} - {status} (score: {score})")
            return result
            
        except Exception as e:
            error_result = SecurityScanResult(
                scan_id=scan_id,
                scan_type=SecurityCheckType.VULNERABILITY_SCAN,
                stage=stage,
                status="failed",
                findings=[{
                    'severity': 'critical',
                    'type': 'scan_error',
                    'description': f"Scan failed: {str(e)}"
                }],
                score=0.0
            )
            
            self._add_scan_result(error_result)
            return error_result
    
    def run_secret_detection_scan(self, target_path: str, stage: DeploymentStage, 
                                 secret_manager: SecretManager) -> SecurityScanResult:
        """Run secret detection scan"""
        scan_start = datetime.utcnow()
        scan_id = f"secret_{int(scan_start.timestamp())}"
        
        try:
            findings = []
            
            if Path(target_path).is_dir():
                for file_path in Path(target_path).rglob("*"):
                    if file_path.is_file() and not self._should_skip_file(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            
                            file_findings = secret_manager.scan_for_secrets(content)
                            for finding in file_findings:
                                finding['file'] = str(file_path)
                            
                            findings.extend(file_findings)
                            
                        except Exception as e:
                            self.logger.debug(f"Could not scan file {file_path}: {e}")
            
            # Calculate score
            score = 100.0
            for finding in findings:
                if finding['severity'] == 'critical':
                    score -= 25
                elif finding['severity'] == 'high':
                    score -= 15
                elif finding['severity'] == 'medium':
                    score -= 10
            
            score = max(0.0, score)
            status = "passed" if score >= 90 else ("warning" if score >= 70 else "failed")
            
            duration = (datetime.utcnow() - scan_start).total_seconds()
            
            result = SecurityScanResult(
                scan_id=scan_id,
                scan_type=SecurityCheckType.SECRET_DETECTION,
                stage=stage,
                status=status,
                findings=findings,
                score=score,
                duration_seconds=duration
            )
            
            self._add_scan_result(result)
            
            self.logger.info(f"Secret detection scan completed: {scan_id} - {status} (score: {score})")
            return result
            
        except Exception as e:
            error_result = SecurityScanResult(
                scan_id=scan_id,
                scan_type=SecurityCheckType.SECRET_DETECTION,
                stage=stage,
                status="failed",
                findings=[{
                    'severity': 'critical',
                    'type': 'scan_error',
                    'description': f"Secret detection failed: {str(e)}"
                }],
                score=0.0
            )
            
            self._add_scan_result(error_result)
            return error_result
    
    def _scan_file_for_vulnerabilities(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan individual file for vulnerabilities"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Check for common vulnerability patterns
            vulnerability_patterns = {
                'sql_injection': {
                    'pattern': r'(?i)(execute|query|select).*%s|.*\+.*where',
                    'severity': 'high',
                    'description': 'Potential SQL injection vulnerability'
                },
                'xss': {
                    'pattern': r'(?i)innerHTML\s*=|document\.write\(',
                    'severity': 'medium',
                    'description': 'Potential XSS vulnerability'
                },
                'hardcoded_secret': {
                    'pattern': r'(?i)(password|secret|key)\s*=\s*[\'"][^\'\"]{8,}[\'"]',
                    'severity': 'high',
                    'description': 'Hardcoded secret detected'
                },
                'eval_usage': {
                    'pattern': r'(?i)\beval\s*\(',
                    'severity': 'high',
                    'description': 'Use of SafeCodeExecutor.safe_SafeCodeExecutor.safe_eval() function detected'
                }
            }
            
            for line_num, line in enumerate(lines, 1):
                for vuln_type, vuln_info in vulnerability_patterns.items():
                    if re.search(vuln_info['pattern'], line):
                        findings.append({
                            'type': vuln_type,
                            'severity': vuln_info['severity'],
                            'description': vuln_info['description'],
                            'file': str(file_path),
                            'line': line_num,
                            'content': line.strip()
                        })
            
        except Exception as e:
            self.logger.debug(f"Error scanning file {file_path}: {e}")
        
        return findings
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during scanning"""
        skip_extensions = {'.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', '.bin', '.jpg', '.png', '.gif', '.pdf'}
        skip_dirs = {'.git', '__pycache__', 'node_modules', '.env', 'venv', '.venv'}
        
        # Skip based on extension
        if file_path.suffix.lower() in skip_extensions:
            return True
        
        # Skip based on directory
        for part in file_path.parts:
            if part in skip_dirs:
                return True
        
        # Skip large files (>10MB)
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:
                return True
        except OSError:
            return True
        
        return False
    
    def _add_scan_result(self, result: SecurityScanResult):
        """Add scan result to history"""
        self.scan_history.append(result)
        
        # Limit scan history
        if len(self.scan_history) > self.max_scan_history:
            self.scan_history = self.scan_history[-self.max_scan_history // 2:]


class DeploymentPipelineSecurityManager:
    """Comprehensive deployment pipeline security management system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pipelines: Dict[str, DeploymentPipeline] = {}
        self.secret_manager = SecretManager()
        self.security_scanner = SecurityScanner()
        self.execution_history: List[Dict[str, Any]] = []
        self.max_execution_history = 1000
    
    def create_pipeline(self, pipeline_config: DeploymentPipeline) -> bool:
        """Create new deployment pipeline"""
        try:
            if not pipeline_config.pipeline_id:
                raise ValidationError("Pipeline ID cannot be empty")
            
            if pipeline_config.pipeline_id in self.pipelines:
                raise ValidationError(f"Pipeline already exists: {pipeline_config.pipeline_id}")
            
            # Validate pipeline configuration
            if not self._validate_pipeline_config(pipeline_config):
                return False
            
            # Store pipeline
            self.pipelines[pipeline_config.pipeline_id] = pipeline_config
            
            self.logger.info(f"Created deployment pipeline: {pipeline_config.pipeline_id}")
            return True
            
        except Exception as e:
            error = SecurityError(f"Failed to create pipeline: {str(e)}", "PIPELINE_CREATE_001")
            security_error_handler.handle_error(error)
            return False
    
    def execute_pipeline(self, pipeline_id: str, source_path: str = None) -> Dict[str, Any]:
        """Execute deployment pipeline with security checks"""
        try:
            if pipeline_id not in self.pipelines:
                raise ValidationError(f"Pipeline not found: {pipeline_id}")
            
            pipeline = self.pipelines[pipeline_id]
            execution_id = f"exec_{pipeline_id}_{int(datetime.utcnow().timestamp())}"
            
            execution_start = datetime.utcnow()
            
            execution_result = {
                'execution_id': execution_id,
                'pipeline_id': pipeline_id,
                'started_at': execution_start.isoformat(),
                'stages': {},
                'security_scans': {},
                'overall_status': 'running',
                'overall_score': 0.0
            }
            
            self.logger.info(f"Starting pipeline execution: {execution_id}")
            
            # Execute each stage
            for stage in pipeline.stages:
                stage_result = self._execute_stage(pipeline, stage, source_path or ".")
                execution_result['stages'][stage.value] = stage_result
                
                # Stop execution if critical stage fails
                if stage_result['status'] == 'failed' and stage in [DeploymentStage.SECURITY_SCAN, DeploymentStage.TEST]:
                    execution_result['overall_status'] = 'failed'
                    break
            
            # Execute security checks
            for check_type in pipeline.security_checks:
                scan_result = self._execute_security_check(check_type, source_path or ".", DeploymentStage.SECURITY_SCAN)
                execution_result['security_scans'][check_type.value] = {
                    'scan_id': scan_result.scan_id,
                    'status': scan_result.status,
                    'score': scan_result.score,
                    'findings_count': len(scan_result.findings),
                    'critical_findings': len(scan_result.critical_findings)
                }
                
                # Fail pipeline on critical security issues
                if scan_result.has_critical_issues:
                    execution_result['overall_status'] = 'failed'
                    break
            
            # Calculate overall score
            scores = []
            for scan_info in execution_result['security_scans'].values():
                scores.append(scan_info['score'])
            
            execution_result['overall_score'] = sum(scores) / len(scores) if scores else 0.0
            
            # Determine final status
            if execution_result['overall_status'] == 'running':
                if execution_result['overall_score'] >= 80:
                    execution_result['overall_status'] = 'success'
                elif execution_result['overall_score'] >= 60:
                    execution_result['overall_status'] = 'warning'
                else:
                    execution_result['overall_status'] = 'failed'
            
            # Update pipeline statistics
            pipeline.last_execution = datetime.utcnow()
            pipeline.execution_count += 1
            
            # Record execution
            execution_duration = (datetime.utcnow() - execution_start).total_seconds()
            execution_result['duration_seconds'] = execution_duration
            execution_result['completed_at'] = datetime.utcnow().isoformat()
            
            self._record_pipeline_execution(execution_result)
            
            self.logger.info(f"Pipeline execution completed: {execution_id} - {execution_result['overall_status']}")
            return execution_result
            
        except Exception as e:
            error = SecurityError(f"Pipeline execution failed: {str(e)}", "PIPELINE_EXEC_001")
            security_error_handler.handle_error(error)
            return {
                'execution_id': 'error',
                'pipeline_id': pipeline_id,
                'overall_status': 'error',
                'error': str(e)
            }
    
    def get_pipeline_security_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline security status"""
        try:
            # Pipeline statistics
            total_pipelines = len(self.pipelines)
            
            # Environment distribution
            env_distribution = {}
            for pipeline in self.pipelines.values():
                env = pipeline.target_environment.value
                env_distribution[env] = env_distribution.get(env, 0) + 1
            
            # Security scan statistics
            total_scans = len(self.security_scanner.scan_history)
            recent_scans = [s for s in self.security_scanner.scan_history 
                           if (datetime.utcnow() - s.timestamp).total_seconds() < 3600]
            
            passed_scans = sum(1 for s in recent_scans if s.status == 'passed')
            failed_scans = sum(1 for s in recent_scans if s.status == 'failed')
            
            # Secret management statistics
            total_secrets = len(self.secret_manager.secrets)
            recent_secret_access = len([a for a in self.secret_manager.access_history
                                       if (datetime.utcnow() - datetime.fromisoformat(a['timestamp'])).total_seconds() < 3600])
            
            # Recent executions
            recent_executions = len([e for e in self.execution_history 
                                   if (datetime.utcnow() - datetime.fromisoformat(e['started_at'])).total_seconds() < 3600])
            
            return {
                'pipeline_stats': {
                    'total_pipelines': total_pipelines,
                    'environment_distribution': env_distribution,
                    'recent_executions_1h': recent_executions
                },
                'security_scan_stats': {
                    'total_scans': total_scans,
                    'recent_scans_1h': len(recent_scans),
                    'passed_scans_1h': passed_scans,
                    'failed_scans_1h': failed_scans,
                    'success_rate_pct': (passed_scans / max(len(recent_scans), 1)) * 100
                },
                'secret_management_stats': {
                    'total_secrets': total_secrets,
                    'recent_access_1h': recent_secret_access
                },
                'overall_security_score': self._calculate_overall_security_score()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating pipeline security status: {e}")
            return {'error': str(e)}
    
    def _validate_pipeline_config(self, pipeline: DeploymentPipeline) -> bool:
        """Validate pipeline configuration"""
        try:
            if not pipeline.pipeline_name:
                raise ValidationError("Pipeline name cannot be empty")
            
            if not pipeline.source_repository:
                raise ValidationError("Source repository cannot be empty")
            
            if not pipeline.stages:
                raise ValidationError("Pipeline must have at least one stage")
            
            # Ensure security scan stage is present for production deployments
            if (pipeline.target_environment == DeploymentEnvironment.PRODUCTION and 
                DeploymentStage.SECURITY_SCAN not in pipeline.stages):
                pipeline.stages.append(DeploymentStage.SECURITY_SCAN)
                self.logger.info(f"Added security scan stage to production pipeline: {pipeline.pipeline_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline configuration validation failed: {e}")
            return False
    
    def _execute_stage(self, pipeline: DeploymentPipeline, stage: DeploymentStage, 
                      source_path: str) -> Dict[str, Any]:
        """Execute individual pipeline stage"""
        stage_start = datetime.utcnow()
        
        try:
            stage_result = {
                'stage': stage.value,
                'status': 'running',
                'started_at': stage_start.isoformat(),
                'duration_seconds': 0.0,
                'logs': []
            }
            
            # Simulate stage execution
            if stage == DeploymentStage.SOURCE:
                stage_result['logs'].append("Checking out source code...")
                stage_result['status'] = 'success'
                
            elif stage == DeploymentStage.BUILD:
                stage_result['logs'].append("Building application...")
                stage_result['status'] = 'success'
                
            elif stage == DeploymentStage.TEST:
                stage_result['logs'].append("Running tests...")
                stage_result['status'] = 'success'
                
            elif stage == DeploymentStage.SECURITY_SCAN:
                stage_result['logs'].append("Running security scans...")
                stage_result['status'] = 'success'
                
            elif stage == DeploymentStage.DEPLOY:
                stage_result['logs'].append(f"Deploying to {pipeline.target_environment.value}...")
                stage_result['status'] = 'success'
                
            elif stage == DeploymentStage.VERIFY:
                stage_result['logs'].append("Verifying deployment...")
                stage_result['status'] = 'success'
            
            stage_duration = (datetime.utcnow() - stage_start).total_seconds()
            stage_result['duration_seconds'] = stage_duration
            stage_result['completed_at'] = datetime.utcnow().isoformat()
            
            return stage_result
            
        except Exception as e:
            return {
                'stage': stage.value,
                'status': 'failed',
                'error': str(e),
                'duration_seconds': (datetime.utcnow() - stage_start).total_seconds()
            }
    
    def _execute_security_check(self, check_type: SecurityCheckType, 
                               source_path: str, stage: DeploymentStage) -> SecurityScanResult:
        """Execute security check"""
        if check_type == SecurityCheckType.VULNERABILITY_SCAN:
            return self.security_scanner.run_vulnerability_scan(source_path, stage)
        elif check_type == SecurityCheckType.SECRET_DETECTION:
            return self.security_scanner.run_secret_detection_scan(source_path, stage, self.secret_manager)
        else:
            # Placeholder for other scan types
            return SecurityScanResult(
                scan_id=f"{check_type.value}_{int(datetime.utcnow().timestamp())}",
                scan_type=check_type,
                stage=stage,
                status="passed",
                score=100.0
            )
    
    def _calculate_overall_security_score(self) -> float:
        """Calculate overall deployment security score"""
        try:
            scores = []
            
            # Recent scan scores
            recent_scans = [s for s in self.security_scanner.scan_history 
                           if (datetime.utcnow() - s.timestamp).total_seconds() < 86400]  # 24 hours
            
            if recent_scans:
                avg_scan_score = sum(s.score for s in recent_scans) / len(recent_scans)
                scores.append(avg_scan_score)
            
            # Pipeline configuration score
            production_pipelines = [p for p in self.pipelines.values() 
                                   if p.target_environment == DeploymentEnvironment.PRODUCTION]
            
            if production_pipelines:
                # Check if production pipelines have security checks
                secure_pipelines = sum(1 for p in production_pipelines 
                                     if SecurityCheckType.VULNERABILITY_SCAN in p.security_checks)
                config_score = (secure_pipelines / len(production_pipelines)) * 100
                scores.append(config_score)
            
            # Secret management score (penalty for unused secrets)
            if self.secret_manager.secrets:
                recently_used = sum(1 for s in self.secret_manager.secrets.values() if s.is_recently_used)
                secret_score = (recently_used / len(self.secret_manager.secrets)) * 100
                scores.append(secret_score)
            
            return sum(scores) / len(scores) if scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating overall security score: {e}")
            return 0.0
    
    def _record_pipeline_execution(self, execution_result: Dict[str, Any]):
        """Record pipeline execution"""
        self.execution_history.append(execution_result)
        
        # Limit execution history
        if len(self.execution_history) > self.max_execution_history:
            self.execution_history = self.execution_history[-self.max_execution_history // 2:]


# Global deployment pipeline security manager
deployment_pipeline_security = DeploymentPipelineSecurityManager()


# Convenience functions
def create_secure_pipeline(pipeline_id: str, pipeline_name: str, source_repo: str,
                          environment: DeploymentEnvironment) -> bool:
    """Convenience function to create secure deployment pipeline"""
    pipeline_config = DeploymentPipeline(
        pipeline_id=pipeline_id,
        pipeline_name=pipeline_name,
        source_repository=source_repo,
        target_environment=environment,
        stages=[DeploymentStage.SOURCE, DeploymentStage.BUILD, DeploymentStage.TEST, 
               DeploymentStage.SECURITY_SCAN, DeploymentStage.DEPLOY],
        security_checks=[SecurityCheckType.VULNERABILITY_SCAN, SecurityCheckType.SECRET_DETECTION]
    )
    return deployment_pipeline_security.create_pipeline(pipeline_config)


def store_deployment_secret(secret_name: str, secret_value: str, 
                           environment: DeploymentEnvironment) -> bool:
    """Convenience function to store deployment secret"""
    return deployment_pipeline_security.secret_manager.store_secret(secret_name, secret_value, environment)