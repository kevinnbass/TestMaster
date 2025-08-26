"""
Integration Configuration Module
================================

Integration and third-party service configuration settings.
Modularized from testmaster_config.py and unified_config.py.

Author: Agent E - Infrastructure Consolidation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from .data_models import ConfigBase


class IntegrationType(Enum):
    """Integration types."""
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    MESSAGE_QUEUE = "message_queue"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"


class WebhookMethod(Enum):
    """Webhook HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


@dataclass
class GitIntegrationConfig(ConfigBase):
    """Git integration configuration."""
    
    # Git Settings
    enabled: bool = True
    provider: str = "github"  # github, gitlab, bitbucket
    
    # Repository Settings
    repository_url: Optional[str] = None
    default_branch: str = "main"
    auto_commit: bool = False
    auto_push: bool = False
    
    # Authentication
    auth_method: str = "token"  # token, ssh, basic
    access_token: Optional[str] = None
    ssh_key_path: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    
    # PR/MR Settings
    create_pull_requests: bool = True
    auto_merge_passing: bool = False
    require_reviews: int = 1
    
    # Hooks
    webhook_enabled: bool = False
    webhook_secret: Optional[str] = None
    webhook_events: List[str] = field(default_factory=lambda: [
        "push", "pull_request", "issues"
    ])
    
    def validate(self) -> List[str]:
        """Validate git integration configuration."""
        errors = []
        
        if self.enabled and not self.repository_url:
            errors.append("Repository URL required when git integration enabled")
        
        if self.auth_method == "token" and not self.access_token:
            errors.append("Access token required for token authentication")
        
        if self.auth_method == "ssh" and not self.ssh_key_path:
            errors.append("SSH key path required for SSH authentication")
        
        if self.require_reviews < 0:
            errors.append("Required reviews cannot be negative")
        
        return errors


@dataclass
class CICDConfig(ConfigBase):
    """CI/CD integration configuration."""
    
    # CI/CD Settings
    enabled: bool = True
    provider: str = "jenkins"  # jenkins, github_actions, gitlab_ci, circleci
    
    # Pipeline Settings
    pipeline_file: str = ".jenkins/Jenkinsfile"
    trigger_on_commit: bool = True
    trigger_on_pr: bool = True
    trigger_on_schedule: bool = False
    schedule_cron: str = "0 0 * * *"
    
    # Build Settings
    parallel_builds: bool = True
    max_parallel_jobs: int = 5
    build_timeout_minutes: int = 60
    
    # Test Integration
    run_tests: bool = True
    test_coverage_threshold: float = 80.0
    fail_on_coverage_drop: bool = True
    
    # Deployment
    auto_deploy_on_success: bool = False
    deployment_environments: List[str] = field(default_factory=lambda: [
        "dev", "staging", "production"
    ])
    
    # Notifications
    notify_on_failure: bool = True
    notify_on_success: bool = False
    notification_channels: List[str] = field(default_factory=lambda: [
        "email", "slack"
    ])
    
    def validate(self) -> List[str]:
        """Validate CI/CD configuration."""
        errors = []
        
        if self.max_parallel_jobs <= 0:
            errors.append("Max parallel jobs must be positive")
        
        if self.build_timeout_minutes <= 0:
            errors.append("Build timeout must be positive")
        
        if self.test_coverage_threshold < 0 or self.test_coverage_threshold > 100:
            errors.append("Test coverage threshold must be between 0 and 100")
        
        return errors


@dataclass
class NotificationConfig(ConfigBase):
    """Notification integration configuration."""
    
    # Email Settings
    email_enabled: bool = False
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    email_from: str = "testmaster@example.com"
    email_recipients: List[str] = field(default_factory=list)
    
    # Slack Settings
    slack_enabled: bool = False
    slack_webhook_url: Optional[str] = None
    slack_channel: str = "#testmaster"
    slack_username: str = "TestMaster"
    
    # Teams Settings
    teams_enabled: bool = False
    teams_webhook_url: Optional[str] = None
    
    # Discord Settings
    discord_enabled: bool = False
    discord_webhook_url: Optional[str] = None
    
    # Notification Rules
    notify_on_test_failure: bool = True
    notify_on_coverage_drop: bool = True
    notify_on_performance_regression: bool = True
    notify_on_security_issue: bool = True
    
    def validate(self) -> List[str]:
        """Validate notification configuration."""
        errors = []
        
        if self.email_enabled:
            if not self.smtp_host:
                errors.append("SMTP host required for email notifications")
            
            if self.smtp_port <= 0 or self.smtp_port > 65535:
                errors.append("SMTP port must be valid")
            
            if not self.email_recipients:
                errors.append("Email recipients required for email notifications")
        
        if self.slack_enabled and not self.slack_webhook_url:
            errors.append("Slack webhook URL required for Slack notifications")
        
        if self.teams_enabled and not self.teams_webhook_url:
            errors.append("Teams webhook URL required for Teams notifications")
        
        if self.discord_enabled and not self.discord_webhook_url:
            errors.append("Discord webhook URL required for Discord notifications")
        
        return errors


@dataclass
class WebhookConfig(ConfigBase):
    """Webhook configuration."""
    
    # Webhook Settings
    enabled: bool = False
    url: Optional[str] = None
    method: WebhookMethod = WebhookMethod.POST
    
    # Authentication
    auth_type: str = "none"  # none, basic, bearer, api_key
    auth_token: Optional[str] = None
    auth_header: str = "Authorization"
    
    # Request Configuration
    headers: Dict[str, str] = field(default_factory=lambda: {
        "Content-Type": "application/json"
    })
    timeout_seconds: int = 30
    retry_count: int = 3
    retry_delay_seconds: int = 5
    
    # Payload
    include_full_report: bool = True
    include_metrics: bool = True
    include_logs: bool = False
    custom_payload: Optional[Dict[str, Any]] = None
    
    # Events
    trigger_events: List[str] = field(default_factory=lambda: [
        "test_complete", "coverage_report", "error"
    ])
    
    def validate(self) -> List[str]:
        """Validate webhook configuration."""
        errors = []
        
        if self.enabled and not self.url:
            errors.append("Webhook URL required when webhooks enabled")
        
        if self.timeout_seconds <= 0:
            errors.append("Timeout must be positive")
        
        if self.retry_count < 0:
            errors.append("Retry count cannot be negative")
        
        if self.auth_type in ["bearer", "api_key"] and not self.auth_token:
            errors.append(f"Auth token required for {self.auth_type} authentication")
        
        return errors


@dataclass
class CloudIntegrationConfig(ConfigBase):
    """Cloud provider integration configuration."""
    
    # Cloud Provider
    provider: str = "aws"  # aws, gcp, azure
    enabled: bool = False
    
    # AWS Settings
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    
    # GCP Settings
    gcp_project_id: Optional[str] = None
    gcp_credentials_path: Optional[str] = None
    
    # Azure Settings
    azure_subscription_id: Optional[str] = None
    azure_tenant_id: Optional[str] = None
    azure_client_id: Optional[str] = None
    azure_client_secret: Optional[str] = None
    
    # Services
    use_cloud_storage: bool = True
    use_cloud_logging: bool = True
    use_cloud_monitoring: bool = True
    use_cloud_secrets: bool = False
    
    def validate(self) -> List[str]:
        """Validate cloud integration configuration."""
        errors = []
        
        if self.enabled:
            if self.provider == "aws":
                if not self.aws_access_key_id or not self.aws_secret_access_key:
                    errors.append("AWS credentials required for AWS integration")
            
            elif self.provider == "gcp":
                if not self.gcp_project_id or not self.gcp_credentials_path:
                    errors.append("GCP project ID and credentials required")
            
            elif self.provider == "azure":
                if not all([self.azure_subscription_id, self.azure_tenant_id,
                           self.azure_client_id, self.azure_client_secret]):
                    errors.append("Azure credentials required for Azure integration")
        
        return errors


@dataclass
class IntegrationConfig(ConfigBase):
    """Combined integration configuration."""
    
    git: GitIntegrationConfig = field(default_factory=GitIntegrationConfig)
    cicd: CICDConfig = field(default_factory=CICDConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    webhooks: WebhookConfig = field(default_factory=WebhookConfig)
    cloud: CloudIntegrationConfig = field(default_factory=CloudIntegrationConfig)
    
    # API Integration
    external_apis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Service Mesh
    service_mesh_enabled: bool = False
    service_mesh_provider: str = "istio"
    
    # Event Streaming
    event_streaming_enabled: bool = False
    event_streaming_platform: str = "kafka"
    event_streaming_brokers: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate all integration configurations."""
        errors = []
        errors.extend(self.git.validate())
        errors.extend(self.cicd.validate())
        errors.extend(self.notifications.validate())
        errors.extend(self.webhooks.validate())
        errors.extend(self.cloud.validate())
        
        if self.event_streaming_enabled and not self.event_streaming_brokers:
            errors.append("Event streaming brokers required when streaming enabled")
        
        return errors


__all__ = [
    'IntegrationType',
    'WebhookMethod',
    'GitIntegrationConfig',
    'CICDConfig',
    'NotificationConfig',
    'WebhookConfig',
    'CloudIntegrationConfig',
    'IntegrationConfig'
]