"""
Production-Ready Documentation

Creates full-stack deployment documentation with Docker, cloud patterns,
and observability based on LLama-Agents production-ready approach.
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProductionTier(Enum):
    """Production deployment tiers."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"


class CloudProvider(Enum):
    """Supported cloud providers."""
    GOOGLE_CLOUD = "google_cloud"
    AWS = "aws"
    AZURE = "azure"
    DIGITAL_OCEAN = "digital_ocean"
    LOCAL = "local"


class ObservabilityTool(Enum):
    """Observability and monitoring tools."""
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    JAEGER = "jaeger"
    ELASTIC_STACK = "elastic_stack"
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"


@dataclass
class ProductionConfig:
    """Production deployment configuration."""
    name: str
    tier: ProductionTier
    cloud_provider: CloudProvider
    region: str = ""
    services: List[Dict[str, Any]] = field(default_factory=list)
    databases: List[Dict[str, Any]] = field(default_factory=list)
    message_queues: List[Dict[str, Any]] = field(default_factory=list)
    load_balancers: List[Dict[str, Any]] = field(default_factory=list)
    monitoring: List[ObservabilityTool] = field(default_factory=list)
    security: Dict[str, Any] = field(default_factory=dict)
    scaling: Dict[str, Any] = field(default_factory=dict)
    backup: Dict[str, Any] = field(default_factory=dict)
    disaster_recovery: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContainerImage:
    """Container image configuration."""
    name: str
    base_image: str
    dockerfile_content: str
    build_args: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    healthcheck: Dict[str, Any] = field(default_factory=dict)
    security_scan: bool = True
    multi_arch: bool = False


@dataclass
class ObservabilityStack:
    """Complete observability configuration."""
    metrics_collection: Dict[str, Any] = field(default_factory=dict)
    logging_config: Dict[str, Any] = field(default_factory=dict)
    tracing_config: Dict[str, Any] = field(default_factory=dict)
    alerting_rules: List[Dict[str, Any]] = field(default_factory=list)
    dashboards: List[Dict[str, Any]] = field(default_factory=list)
    sla_objectives: List[Dict[str, Any]] = field(default_factory=list)


class ProductionReadyDocs:
    """
    Production-ready documentation generator inspired by LLama-Agents
    full-stack deployment patterns with Docker and cloud integration.
    """
    
    def __init__(self, docs_dir: str = "production-docs"):
        """Initialize production-ready docs generator."""
        self.docs_dir = Path(docs_dir)
        self.production_configs = []
        self.container_images = []
        self.observability_stacks = []
        self.deployment_templates = self._load_deployment_templates()
        logger.info(f"Production-ready docs initialized at {docs_dir}")
        
    def create_production_config(self,
                               name: str,
                               tier: ProductionTier,
                               cloud_provider: CloudProvider,
                               **kwargs) -> ProductionConfig:
        """Create production deployment configuration."""
        config = ProductionConfig(
            name=name,
            tier=tier,
            cloud_provider=cloud_provider,
            **kwargs
        )
        
        self.production_configs.append(config)
        logger.info(f"Created production config: {name} ({tier.value})")
        return config
        
    def create_container_image(self,
                             name: str,
                             base_image: str,
                             dockerfile_content: str,
                             **kwargs) -> ContainerImage:
        """Create container image configuration."""
        image = ContainerImage(
            name=name,
            base_image=base_image,
            dockerfile_content=dockerfile_content,
            **kwargs
        )
        
        self.container_images.append(image)
        logger.info(f"Created container image: {name}")
        return image
        
    def create_observability_stack(self, **kwargs) -> ObservabilityStack:
        """Create observability stack configuration."""
        stack = ObservabilityStack(**kwargs)
        self.observability_stacks.append(stack)
        logger.info("Created observability stack")
        return stack
        
    def generate_production_overview(self) -> str:
        """Generate production deployment overview documentation."""
        overview = [
            "# Production Deployment Guide",
            "",
            "Complete guide for deploying multi-agent systems to production.",
            "",
            "## Deployment Architecture",
            "",
            "Production-ready multi-agent systems require:",
            "",
            "- **Containerized Services**: Docker containers for consistency",
            "- **Load Balancing**: High availability and traffic distribution", 
            "- **Auto Scaling**: Automatic resource adjustment",
            "- **Observability**: Comprehensive monitoring and logging",
            "- **Security**: Authentication, authorization, and encryption",
            "- **Backup & Recovery**: Data protection and disaster recovery",
            "",
            "## Supported Environments",
            ""
        ]
        
        # Group configs by cloud provider
        by_provider = {}
        for config in self.production_configs:
            if config.cloud_provider not in by_provider:
                by_provider[config.cloud_provider] = []
            by_provider[config.cloud_provider].append(config)
            
        for provider, configs in by_provider.items():
            overview.extend([
                f"### {provider.value.replace('_', ' ').title()}",
                ""
            ])
            
            for config in configs:
                overview.extend([
                    f"#### {config.name} ({config.tier.value.title()})",
                    "",
                    f"- **Region:** {config.region or 'Default'}",
                    f"- **Services:** {len(config.services)}",
                    f"- **Databases:** {len(config.databases)}",
                    f"- **Load Balancers:** {len(config.load_balancers)}",
                    ""
                ])
                
        overview.extend([
            "## Quick Start Options",
            "",
            "### Option 1: Docker Compose (Development)",
            "",
            "```bash",
            "# Clone and start with Docker Compose",
            "git clone <repository>",
            "cd <project>",
            "docker-compose up -d",
            "```",
            "",
            "### Option 2: Kubernetes (Production)",
            "",
            "```bash",
            "# Deploy to Kubernetes cluster",
            "kubectl apply -f k8s-manifests/",
            "kubectl get pods",
            "```",
            "",
            "### Option 3: Cloud Run (Serverless)",
            "",
            "```bash",
            "# Deploy to Google Cloud Run",
            "gcloud run deploy --source=.",
            "```",
            ""
        ])
        
        return "\n".join(overview)
        
    def generate_dockerfile(self, image: ContainerImage) -> str:
        """Generate Dockerfile for container image."""
        if image.dockerfile_content:
            return image.dockerfile_content
            
        # Generate default Dockerfile
        dockerfile = f"""# Multi-stage build for {image.name}
FROM {image.base_image} as builder

# Install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production image
FROM {image.base_image}
WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application
COPY . .

# Add labels
"""
        
        for key, value in image.labels.items():
            dockerfile += f'LABEL {key}="{value}"\n'
            
        dockerfile += f"""
# Health check
{self._generate_healthcheck(image.healthcheck)}

# User and permissions
RUN addgroup --gid 1000 appgroup && \\
    adduser --uid 1000 --gid 1000 --disabled-password --gecos "" appuser
USER appuser

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "main.py"]
"""
        
        return dockerfile
        
    def generate_docker_compose_production(self, config: ProductionConfig) -> str:
        """Generate production Docker Compose configuration."""
        compose = {
            "version": "3.8",
            "services": {},
            "networks": {
                "agent_network": {
                    "driver": "bridge"
                }
            },
            "volumes": {
                "postgres_data": {},
                "redis_data": {},
                "logs": {}
            }
        }
        
        # Add services
        for service in config.services:
            service_config = {
                "image": service.get("image", f"{service['name']}:latest"),
                "container_name": service["name"],
                "restart": "unless-stopped",
                "networks": ["agent_network"],
                "environment": service.get("environment", {}),
                "depends_on": service.get("depends_on", []),
                "volumes": [
                    "logs:/app/logs"
                ],
                "healthcheck": {
                    "test": service.get("healthcheck", "curl -f http://localhost:8000/health"),
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3
                }
            }
            
            if service.get("ports"):
                service_config["ports"] = service["ports"]
                
            if service.get("resources"):
                service_config["deploy"] = {
                    "resources": service["resources"]
                }
                
            compose["services"][service["name"]] = service_config
            
        # Add databases
        for db in config.databases:
            if db["type"] == "postgres":
                compose["services"]["postgres"] = {
                    "image": "postgres:14",
                    "container_name": "postgres",
                    "restart": "unless-stopped",
                    "environment": {
                        "POSTGRES_DB": db.get("database", "agentdb"),
                        "POSTGRES_USER": db.get("user", "agent"),
                        "POSTGRES_PASSWORD": db.get("password", "secure_password")
                    },
                    "volumes": [
                        "postgres_data:/var/lib/postgresql/data"
                    ],
                    "ports": ["5432:5432"],
                    "networks": ["agent_network"]
                }
                
        # Add message queues
        for mq in config.message_queues:
            if mq["type"] == "redis":
                compose["services"]["redis"] = {
                    "image": "redis:7-alpine",
                    "container_name": "redis",
                    "restart": "unless-stopped",
                    "volumes": [
                        "redis_data:/data"
                    ],
                    "ports": ["6379:6379"],
                    "networks": ["agent_network"],
                    "command": "redis-server --appendonly yes"
                }
                
        # Add load balancer if needed
        for lb in config.load_balancers:
            if lb["type"] == "nginx":
                compose["services"]["nginx"] = {
                    "image": "nginx:alpine",
                    "container_name": "nginx",
                    "restart": "unless-stopped",
                    "ports": ["80:80", "443:443"],
                    "volumes": [
                        "./nginx.conf:/etc/nginx/nginx.conf:ro",
                        "./certs:/etc/nginx/certs:ro"
                    ],
                    "networks": ["agent_network"],
                    "depends_on": [svc["name"] for svc in config.services]
                }
                
        return yaml.dump(compose, default_flow_style=False, sort_keys=False)
        
    def generate_kubernetes_manifests(self, config: ProductionConfig) -> str:
        """Generate Kubernetes deployment manifests."""
        manifests = []
        
        # Namespace
        manifests.append(f"""apiVersion: v1
kind: Namespace
metadata:
  name: {config.name}
  labels:
    tier: {config.tier.value}
---""")
        
        # ConfigMaps
        manifests.append(f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: {config.name}-config
  namespace: {config.name}
data:
  environment: {config.tier.value}
  log_level: INFO
---""")
        
        # Services
        for service in config.services:
            # Deployment
            replicas = service.get("replicas", 3 if config.tier == ProductionTier.PRODUCTION else 1)
            
            deployment = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {service["name"]}
  namespace: {config.name}
  labels:
    app: {service["name"]}
    tier: {config.tier.value}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {service["name"]}
  template:
    metadata:
      labels:
        app: {service["name"]}
    spec:
      containers:
      - name: {service["name"]}
        image: {service.get("image", f"{service['name']}:latest")}
        ports:
        - containerPort: {service.get("port", 8000)}
        env:
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: {config.name}-config
              key: environment
        resources:
          requests:
            memory: "{service.get("memory_request", "256Mi")}"
            cpu: "{service.get("cpu_request", "250m")}"
          limits:
            memory: "{service.get("memory_limit", "512Mi")}"
            cpu: "{service.get("cpu_limit", "500m")}"
        livenessProbe:
          httpGet:
            path: /health
            port: {service.get("port", 8000)}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: {service.get("port", 8000)}
          initialDelaySeconds: 5
          periodSeconds: 5
---"""
            
            manifests.append(deployment)
            
            # Service
            k8s_service = f"""apiVersion: v1
kind: Service
metadata:
  name: {service["name"]}-service
  namespace: {config.name}
  labels:
    app: {service["name"]}
spec:
  selector:
    app: {service["name"]}
  ports:
  - port: 80
    targetPort: {service.get("port", 8000)}
  type: ClusterIP
---"""
            
            manifests.append(k8s_service)
            
        # Ingress
        if config.load_balancers:
            ingress = f"""apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {config.name}-ingress
  namespace: {config.name}
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - {config.name}.example.com
    secretName: {config.name}-tls
  rules:
  - host: {config.name}.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: {config.services[0]["name"]}-service
            port:
              number: 80
---"""
            manifests.append(ingress)
            
        return "\n".join(manifests)
        
    def generate_cloud_run_deployment(self, service: Dict[str, Any], config: ProductionConfig) -> str:
        """Generate Google Cloud Run deployment configuration."""
        cloud_run_config = {
            "apiVersion": "serving.knative.dev/v1",
            "kind": "Service",
            "metadata": {
                "name": service["name"],
                "namespace": config.name,
                "labels": {
                    "cloud.googleapis.com/location": config.region or "us-central1",
                    "tier": config.tier.value
                },
                "annotations": {
                    "run.googleapis.com/ingress": "all",
                    "run.googleapis.com/execution-environment": "gen2"
                }
            },
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "autoscaling.knative.dev/minScale": str(service.get("min_instances", 0)),
                            "autoscaling.knative.dev/maxScale": str(service.get("max_instances", 100)),
                            "run.googleapis.com/cpu-throttling": "false" if config.tier == ProductionTier.PRODUCTION else "true"
                        }
                    },
                    "spec": {
                        "containerConcurrency": service.get("concurrency", 1000),
                        "timeoutSeconds": service.get("timeout", 300),
                        "containers": [{
                            "image": service.get("image", f"gcr.io/{config.name}/{service['name']}:latest"),
                            "ports": [{"containerPort": service.get("port", 8000)}],
                            "env": [{"name": k, "value": v} for k, v in service.get("environment", {}).items()],
                            "resources": {
                                "limits": {
                                    "cpu": service.get("cpu", "1000m"),
                                    "memory": service.get("memory", "512Mi")
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": service.get("port", 8000)
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 30
                            }
                        }]
                    }
                },
                "traffic": [{
                    "percent": 100,
                    "latestRevision": True
                }]
            }
        }
        
        return yaml.dump(cloud_run_config, default_flow_style=False)
        
    def generate_monitoring_stack(self, stack: ObservabilityStack) -> str:
        """Generate monitoring and observability documentation."""
        monitoring = [
            "# Monitoring and Observability",
            "",
            "Comprehensive monitoring setup for production environments.",
            "",
            "## Stack Components",
            ""
        ]
        
        if stack.metrics_collection:
            monitoring.extend([
                "### Metrics Collection (Prometheus)",
                "",
                "```yaml",
                "# prometheus.yml",
                yaml.dump(stack.metrics_collection, default_flow_style=False),
                "```",
                ""
            ])
            
        if stack.logging_config:
            monitoring.extend([
                "### Logging Configuration",
                "",
                "```yaml",
                "# logging-config.yml",
                yaml.dump(stack.logging_config, default_flow_style=False),
                "```",
                ""
            ])
            
        if stack.alerting_rules:
            monitoring.extend([
                "### Alerting Rules",
                ""
            ])
            for rule in stack.alerting_rules:
                monitoring.extend([
                    f"#### {rule.get('name', 'Alert')}",
                    "",
                    f"- **Condition:** {rule.get('condition', '')}",
                    f"- **Threshold:** {rule.get('threshold', '')}",
                    f"- **Severity:** {rule.get('severity', 'warning')}",
                    ""
                ])
                
        if stack.dashboards:
            monitoring.extend([
                "### Grafana Dashboards",
                ""
            ])
            for dashboard in stack.dashboards:
                monitoring.extend([
                    f"- **{dashboard.get('name', 'Dashboard')}:** {dashboard.get('description', '')}",
                    f"  - Panels: {dashboard.get('panel_count', 0)}",
                    f"  - Refresh: {dashboard.get('refresh_interval', '30s')}",
                    ""
                ])
                
        return "\n".join(monitoring)
        
    def create_default_production_setup(self) -> None:
        """Create default production-ready configuration."""
        # Production configuration
        prod_config = self.create_production_config(
            "multi-agent-prod",
            ProductionTier.PRODUCTION,
            CloudProvider.GOOGLE_CLOUD,
            region="us-central1",
            services=[
                {
                    "name": "orchestrator",
                    "image": "gcr.io/project/orchestrator:latest",
                    "port": 8000,
                    "replicas": 3,
                    "memory_request": "512Mi",
                    "cpu_request": "500m",
                    "memory_limit": "1Gi",
                    "cpu_limit": "1000m",
                    "environment": {
                        "ENVIRONMENT": "production",
                        "LOG_LEVEL": "INFO",
                        "REDIS_URL": "redis://redis:6379"
                    }
                },
                {
                    "name": "research-agent",
                    "image": "gcr.io/project/research-agent:latest",
                    "port": 8001,
                    "replicas": 5,
                    "memory_request": "256Mi",
                    "cpu_request": "250m",
                    "memory_limit": "512Mi",
                    "cpu_limit": "500m"
                }
            ],
            databases=[
                {
                    "type": "postgres",
                    "name": "main_db",
                    "database": "agents",
                    "user": "agent_user"
                }
            ],
            message_queues=[
                {
                    "type": "redis",
                    "name": "main_queue"
                }
            ],
            load_balancers=[
                {
                    "type": "nginx",
                    "ssl": True,
                    "compression": True
                }
            ],
            monitoring=[ObservabilityTool.PROMETHEUS, ObservabilityTool.GRAFANA],
            scaling={
                "auto_scaling": True,
                "min_instances": 2,
                "max_instances": 20,
                "cpu_threshold": 70,
                "memory_threshold": 80
            }
        )
        
        # Container images
        self.create_container_image(
            "orchestrator",
            "python:3.11-slim",
            """FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Create non-root user
RUN addgroup --gid 1000 appgroup && \\
    adduser --uid 1000 --gid 1000 --disabled-password --gecos "" appuser
USER appuser

# Expose port
EXPOSE 8000

# Start command
CMD ["python", "main.py"]""",
            labels={
                "maintainer": "team@company.com",
                "version": "1.0.0",
                "component": "orchestrator"
            }
        )
        
        # Observability stack
        self.create_observability_stack(
            metrics_collection={
                "global": {
                    "scrape_interval": "15s",
                    "evaluation_interval": "15s"
                },
                "scrape_configs": [
                    {
                        "job_name": "orchestrator",
                        "static_configs": [
                            {"targets": ["orchestrator:8000"]}
                        ]
                    },
                    {
                        "job_name": "research-agent",
                        "static_configs": [
                            {"targets": ["research-agent:8001"]}
                        ]
                    }
                ]
            },
            alerting_rules=[
                {
                    "name": "High CPU Usage",
                    "condition": "cpu_usage > 80",
                    "threshold": "80%",
                    "severity": "warning",
                    "duration": "5m"
                },
                {
                    "name": "Service Down",
                    "condition": "up == 0",
                    "threshold": "0",
                    "severity": "critical",
                    "duration": "1m"
                }
            ]
        )
        
    def _generate_healthcheck(self, healthcheck: Dict[str, Any]) -> str:
        """Generate Docker healthcheck configuration."""
        if not healthcheck:
            return "HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\\n    CMD curl -f http://localhost:8000/health || exit 1"
            
        interval = healthcheck.get("interval", "30s")
        timeout = healthcheck.get("timeout", "10s")
        start_period = healthcheck.get("start_period", "60s")
        retries = healthcheck.get("retries", 3)
        test = healthcheck.get("test", "curl -f http://localhost:8000/health || exit 1")
        
        return f"""HEALTHCHECK --interval={interval} --timeout={timeout} --start-period={start_period} --retries={retries} \\
    CMD {test}"""
    
    def _load_deployment_templates(self) -> Dict[str, str]:
        """Load deployment templates."""
        return {
            "dockerfile": """FROM {base_image}

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE {port}
CMD ["python", "main.py"]""",
            "docker_compose": """version: '3.8'
services:
  {services}
networks:
  default:
    name: {network_name}""",
            "k8s_deployment": """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      containers:
      - name: {name}
        image: {image}
        ports:
        - containerPort: {port}"""
        }
        
    def export_production_docs(self, output_dir: str) -> None:
        """Export all production documentation."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Main overview
        with open(output_path / "README.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_production_overview())
            
        # Docker configurations
        docker_dir = output_path / "docker"
        docker_dir.mkdir(exist_ok=True)
        
        for image in self.container_images:
            with open(docker_dir / f"Dockerfile.{image.name}", 'w', encoding='utf-8') as f:
                f.write(self.generate_dockerfile(image))
                
        # Deployment configurations
        for config in self.production_configs:
            config_dir = output_path / config.name
            config_dir.mkdir(exist_ok=True)
            
            # Docker Compose
            with open(config_dir / "docker-compose.yml", 'w', encoding='utf-8') as f:
                f.write(self.generate_docker_compose_production(config))
                
            # Kubernetes
            k8s_dir = config_dir / "k8s"
            k8s_dir.mkdir(exist_ok=True)
            
            with open(k8s_dir / "deployment.yaml", 'w', encoding='utf-8') as f:
                f.write(self.generate_kubernetes_manifests(config))
                
            # Cloud Run configs for each service
            cloudrun_dir = config_dir / "cloudrun"
            cloudrun_dir.mkdir(exist_ok=True)
            
            for service in config.services:
                filename = f"{service['name']}-cloudrun.yaml"
                with open(cloudrun_dir / filename, 'w', encoding='utf-8') as f:
                    f.write(self.generate_cloud_run_deployment(service, config))
                    
        # Monitoring documentation
        monitoring_dir = output_path / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        for stack in self.observability_stacks:
            with open(monitoring_dir / "README.md", 'w', encoding='utf-8') as f:
                f.write(self.generate_monitoring_stack(stack))
                
        logger.info(f"Exported production docs to {output_dir}")