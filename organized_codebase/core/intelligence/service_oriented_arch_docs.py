"""
Service-Oriented Architecture Documentation

Creates deployment-as-code documentation with YAML configuration patterns
and microservice architecture docs based on LLama-Agents approach.
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


class ServiceType(Enum):
    """Types of services in the architecture."""
    AGENT_SERVICE = "agent_service"
    ORCHESTRATOR = "orchestrator" 
    MESSAGE_QUEUE = "message_queue"
    API_GATEWAY = "api_gateway"
    DATABASE = "database"
    CONTROL_PLANE = "control_plane"
    MONITORING = "monitoring"
    UI_SERVICE = "ui_service"


class DeploymentTarget(Enum):
    """Deployment target environments."""
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    CLOUD_RUN = "cloud_run"
    AWS_ECS = "aws_ecs"
    AZURE_CONTAINER = "azure_container"


class CommunicationPattern(Enum):
    """Service communication patterns."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    EVENT_DRIVEN = "event_driven"
    REQUEST_RESPONSE = "request_response"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    MESSAGE_QUEUE = "message_queue"


@dataclass
class ServiceDefinition:
    """Definition of a service in the architecture."""
    name: str
    service_type: ServiceType
    description: str
    image: str = ""
    ports: List[int] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    health_check: Dict[str, Any] = field(default_factory=dict)
    scaling: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    volumes: List[Dict[str, str]] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeploymentConfiguration:
    """Complete deployment configuration."""
    name: str
    target: DeploymentTarget
    services: List[ServiceDefinition] = field(default_factory=list)
    networks: List[Dict[str, Any]] = field(default_factory=list)
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    secrets: List[Dict[str, Any]] = field(default_factory=list)
    config_maps: List[Dict[str, Any]] = field(default_factory=list)
    ingress: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageFlow:
    """Inter-service message flow definition."""
    name: str
    source: str
    target: str
    pattern: CommunicationPattern
    message_types: List[str] = field(default_factory=list)
    protocol: str = "HTTP"
    port: Optional[int] = None
    path: str = "/"
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 30


class ServiceOrientedArchDocs:
    """
    Service-oriented architecture documentation generator inspired by
    LLama-Agents deployment-as-code and microservice patterns.
    """
    
    def __init__(self, docs_dir: str = "soa-docs"):
        """Initialize service-oriented architecture docs."""
        self.docs_dir = Path(docs_dir)
        self.deployments = []
        self.services = []
        self.message_flows = []
        self.deployment_templates = self._load_deployment_templates()
        logger.info(f"Service-oriented arch docs initialized at {docs_dir}")
        
    def create_service_definition(self,
                                name: str,
                                service_type: ServiceType,
                                description: str,
                                **kwargs) -> ServiceDefinition:
        """Create a service definition."""
        service = ServiceDefinition(
            name=name,
            service_type=service_type,
            description=description,
            **kwargs
        )
        
        self.services.append(service)
        logger.info(f"Created service definition: {name} ({service_type.value})")
        return service
        
    def create_deployment_config(self,
                               name: str,
                               target: DeploymentTarget,
                               services: List[ServiceDefinition]) -> DeploymentConfiguration:
        """Create deployment configuration."""
        config = DeploymentConfiguration(
            name=name,
            target=target,
            services=services
        )
        
        self.deployments.append(config)
        logger.info(f"Created deployment config: {name} for {target.value}")
        return config
        
    def add_message_flow(self,
                        name: str,
                        source: str,
                        target: str,
                        pattern: CommunicationPattern,
                        **kwargs) -> MessageFlow:
        """Add message flow between services."""
        flow = MessageFlow(
            name=name,
            source=source,
            target=target,
            pattern=pattern,
            **kwargs
        )
        
        self.message_flows.append(flow)
        logger.info(f"Added message flow: {name} ({source} -> {target})")
        return flow
        
    def generate_architecture_overview(self) -> str:
        """Generate service architecture overview documentation."""
        overview = [
            "# Service-Oriented Architecture",
            "",
            "Microservice architecture for scalable multi-agent systems.",
            "",
            "## Architecture Principles",
            "",
            "- **Service Independence:** Each service is independently deployable",
            "- **Loose Coupling:** Services communicate through well-defined interfaces",
            "- **Single Responsibility:** Each service has one clear purpose", 
            "- **Configuration-Driven:** Deployment and behavior controlled via YAML",
            "- **Horizontal Scalability:** Services can be scaled independently",
            "- **Fault Isolation:** Failures in one service don't cascade",
            "",
            "## Service Types",
            ""
        ]
        
        # Group services by type
        by_type = {}
        for service in self.services:
            if service.service_type not in by_type:
                by_type[service.service_type] = []
            by_type[service.service_type].append(service)
            
        for service_type, services in by_type.items():
            overview.extend([
                f"### {service_type.value.replace('_', ' ').title()}",
                ""
            ])
            
            for service in services:
                overview.extend([
                    f"#### {service.name}",
                    "",
                    service.description,
                    ""
                ])
                
                if service.dependencies:
                    overview.extend([
                        f"**Dependencies:** {', '.join(service.dependencies)}",
                        ""
                    ])
                    
                if service.ports:
                    overview.extend([
                        f"**Ports:** {', '.join(map(str, service.ports))}",
                        ""
                    ])
                    
        overview.extend([
            "## Service Communication",
            "",
            "Services communicate using various patterns:",
            ""
        ])
        
        # Document communication patterns
        patterns_used = set(flow.pattern for flow in self.message_flows)
        for pattern in patterns_used:
            flows = [f for f in self.message_flows if f.pattern == pattern]
            overview.extend([
                f"### {pattern.value.replace('_', ' ').title()}",
                ""
            ])
            
            for flow in flows:
                overview.extend([
                    f"- **{flow.name}:** {flow.source} → {flow.target}",
                    f"  - Protocol: {flow.protocol}",
                    f"  - Timeout: {flow.timeout}s",
                    ""
                ])
                
        return "\n".join(overview)
        
    def generate_docker_compose(self, deployment: DeploymentConfiguration) -> str:
        """Generate Docker Compose configuration."""
        if deployment.target != DeploymentTarget.DOCKER:
            return "# This deployment is not configured for Docker"
            
        compose = {
            "version": "3.8",
            "services": {},
            "networks": {},
            "volumes": {}
        }
        
        # Add services
        for service in deployment.services:
            service_config = {
                "image": service.image or f"{service.name}:latest",
                "container_name": service.name,
                "restart": "unless-stopped"
            }
            
            if service.ports:
                service_config["ports"] = [f"{port}:{port}" for port in service.ports]
                
            if service.environment:
                service_config["environment"] = service.environment
                
            if service.dependencies:
                service_config["depends_on"] = service.dependencies
                
            if service.volumes:
                service_config["volumes"] = [
                    f"{vol['host']}:{vol['container']}" 
                    for vol in service.volumes
                ]
                
            if service.health_check:
                service_config["healthcheck"] = service.health_check
                
            compose["services"][service.name] = service_config
            
        # Add networks
        if deployment.networks:
            for network in deployment.networks:
                compose["networks"][network["name"]] = {
                    "driver": network.get("driver", "bridge")
                }
                
        # Add volumes
        if deployment.volumes:
            for volume in deployment.volumes:
                compose["volumes"][volume["name"]] = {}
                
        return yaml.dump(compose, default_flow_style=False, sort_keys=False)
        
    def generate_kubernetes_manifest(self, deployment: DeploymentConfiguration) -> str:
        """Generate Kubernetes deployment manifest."""
        if deployment.target != DeploymentTarget.KUBERNETES:
            return "# This deployment is not configured for Kubernetes"
            
        manifests = []
        
        # Generate namespace
        manifests.append(f"""apiVersion: v1
kind: Namespace
metadata:
  name: {deployment.name}
---""")
        
        # Generate deployments and services
        for service in deployment.services:
            # Deployment
            manifest = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {service.name}
  namespace: {deployment.name}
  labels:
    app: {service.name}
spec:
  replicas: {service.scaling.get('replicas', 1)}
  selector:
    matchLabels:
      app: {service.name}
  template:
    metadata:
      labels:
        app: {service.name}
    spec:
      containers:
      - name: {service.name}
        image: {service.image or f"{service.name}:latest"}
        ports:"""
            
            for port in service.ports:
                manifest += f"""
        - containerPort: {port}"""
                
            if service.environment:
                manifest += """
        env:"""
                for key, value in service.environment.items():
                    manifest += f"""
        - name: {key}
          value: "{value}" """
                    
            if service.resources:
                manifest += """
        resources:"""
                if "limits" in service.resources:
                    manifest += """
          limits:"""
                    for key, value in service.resources["limits"].items():
                        manifest += f"""
            {key}: {value}"""
                        
                if "requests" in service.resources:
                    manifest += """
          requests:"""
                    for key, value in service.resources["requests"].items():
                        manifest += f"""
            {key}: {value}"""
                        
            manifests.append(manifest + "\n---")
            
            # Service
            if service.ports:
                service_manifest = f"""apiVersion: v1
kind: Service
metadata:
  name: {service.name}-service
  namespace: {deployment.name}
spec:
  selector:
    app: {service.name}
  ports:"""
                
                for port in service.ports:
                    service_manifest += f"""
  - port: {port}
    targetPort: {port}"""
                    
                service_manifest += """
  type: ClusterIP"""
                manifests.append(service_manifest + "\n---")
                
        return "\n".join(manifests)
        
    def generate_cloud_run_config(self, service: ServiceDefinition) -> str:
        """Generate Google Cloud Run service configuration."""
        config = {
            "apiVersion": "serving.knative.dev/v1",
            "kind": "Service",
            "metadata": {
                "name": service.name,
                "labels": service.labels or {}
            },
            "spec": {
                "template": {
                    "metadata": {
                        "labels": service.labels or {}
                    },
                    "spec": {
                        "containers": [{
                            "image": service.image or f"gcr.io/project/{service.name}:latest",
                            "ports": [{"containerPort": service.ports[0] if service.ports else 8080}],
                            "env": [{"name": k, "value": v} for k, v in service.environment.items()],
                            "resources": service.resources or {
                                "limits": {
                                    "cpu": "1000m",
                                    "memory": "512Mi"
                                }
                            }
                        }],
                        "containerConcurrency": service.scaling.get("concurrency", 1000),
                        "timeoutSeconds": service.scaling.get("timeout", 300)
                    }
                },
                "traffic": [{
                    "percent": 100,
                    "latestRevision": True
                }]
            }
        }
        
        return yaml.dump(config, default_flow_style=False)
        
    def generate_deployment_guide(self, deployment: DeploymentConfiguration) -> str:
        """Generate deployment guide for configuration."""
        guide = [
            f"# Deployment Guide: {deployment.name}",
            "",
            f"Deploy to {deployment.target.value.replace('_', ' ').title()}",
            "",
            "## Prerequisites",
            ""
        ]
        
        # Target-specific prerequisites
        if deployment.target == DeploymentTarget.DOCKER:
            guide.extend([
                "- Docker installed and running",
                "- Docker Compose v3.8+",
                ""
            ])
        elif deployment.target == DeploymentTarget.KUBERNETES:
            guide.extend([
                "- Kubernetes cluster available",
                "- kubectl configured",
                "- Sufficient cluster resources",
                ""
            ])
        elif deployment.target == DeploymentTarget.CLOUD_RUN:
            guide.extend([
                "- Google Cloud account with billing enabled",
                "- gcloud CLI installed and authenticated",
                "- Container images pushed to Container Registry",
                ""
            ])
            
        guide.extend([
            "## Service Overview",
            ""
        ])
        
        for service in deployment.services:
            guide.extend([
                f"### {service.name}",
                "",
                service.description,
                "",
                f"- **Type:** {service.service_type.value.replace('_', ' ').title()}",
                f"- **Image:** {service.image or f'{service.name}:latest'}",
                f"- **Ports:** {', '.join(map(str, service.ports)) if service.ports else 'None'}",
                ""
            ])
            
        guide.extend([
            "## Deployment Steps",
            ""
        ])
        
        # Target-specific deployment steps
        if deployment.target == DeploymentTarget.DOCKER:
            guide.extend([
                "1. **Save Configuration**",
                "",
                "   Save the Docker Compose configuration to `docker-compose.yml`",
                "",
                "2. **Start Services**",
                "",
                "   ```bash",
                "   docker-compose up -d",
                "   ```",
                "",
                "3. **Verify Deployment**",
                "",
                "   ```bash",
                "   docker-compose ps",
                "   docker-compose logs",
                "   ```",
                ""
            ])
        elif deployment.target == DeploymentTarget.KUBERNETES:
            guide.extend([
                "1. **Apply Manifests**",
                "",
                "   ```bash",
                "   kubectl apply -f deployment.yaml",
                "   ```",
                "",
                "2. **Check Status**",
                "",
                "   ```bash",
                f"   kubectl get pods -n {deployment.name}",
                f"   kubectl get services -n {deployment.name}",
                "   ```",
                "",
                "3. **View Logs**",
                "",
                "   ```bash",
                f"   kubectl logs -n {deployment.name} -l app=SERVICE_NAME",
                "   ```",
                ""
            ])
            
        guide.extend([
            "## Configuration Options",
            "",
            "Key configuration parameters:",
            ""
        ])
        
        # Document environment variables
        all_env_vars = set()
        for service in deployment.services:
            all_env_vars.update(service.environment.keys())
            
        for env_var in sorted(all_env_vars):
            guide.append(f"- `{env_var}`: Service configuration parameter")
            
        guide.extend([
            "",
            "## Monitoring and Health Checks",
            ""
        ])
        
        for service in deployment.services:
            if service.health_check:
                guide.extend([
                    f"### {service.name}",
                    "",
                    f"- **Health Check:** {service.health_check.get('test', 'Not configured')}",
                    f"- **Interval:** {service.health_check.get('interval', '30s')}",
                    f"- **Timeout:** {service.health_check.get('timeout', '5s')}",
                    ""
                ])
                
        guide.extend([
            "## Scaling Configuration",
            ""
        ])
        
        for service in deployment.services:
            if service.scaling:
                guide.extend([
                    f"### {service.name}",
                    ""
                ])
                
                for key, value in service.scaling.items():
                    guide.append(f"- **{key.title()}:** {value}")
                    
                guide.append("")
                
        return "\n".join(guide)
        
    def generate_message_flow_diagram(self) -> str:
        """Generate Mermaid diagram for message flows."""
        diagram = [
            "# Service Communication Diagram",
            "",
            "```mermaid",
            "graph TD"
        ]
        
        # Add service nodes
        service_names = set()
        for flow in self.message_flows:
            service_names.add(flow.source)
            service_names.add(flow.target)
            
        for name in service_names:
            diagram.append(f"    {name}[{name}]")
            
        # Add message flows
        for flow in self.message_flows:
            arrow = "-->" if flow.pattern == CommunicationPattern.SYNCHRONOUS else "-..->"
            diagram.append(f"    {flow.source} {arrow} {flow.target}")
            
        diagram.extend(["```", ""])
        
        return "\n".join(diagram)
        
    def create_default_microservice_arch(self) -> None:
        """Create default microservice architecture based on LLama-Agents."""
        # Agent Service
        self.create_service_definition(
            "research-agent",
            ServiceType.AGENT_SERVICE,
            "Research agent service for information gathering",
            image="research-agent:latest",
            ports=[8001],
            environment={
                "AGENT_TYPE": "research",
                "LOG_LEVEL": "INFO",
                "MESSAGE_QUEUE_URL": "redis://redis:6379"
            },
            health_check={
                "test": "curl -f http://localhost:8001/health",
                "interval": "30s",
                "timeout": "5s",
                "retries": 3
            },
            scaling={
                "replicas": 2,
                "max_replicas": 5,
                "cpu_threshold": 70
            }
        )
        
        # Orchestrator Service
        self.create_service_definition(
            "orchestrator",
            ServiceType.ORCHESTRATOR,
            "Central orchestrator for coordinating agents",
            image="orchestrator:latest",
            ports=[8000],
            environment={
                "ORCHESTRATOR_PORT": "8000",
                "MESSAGE_QUEUE_URL": "redis://redis:6379",
                "AGENT_REGISTRY_URL": "http://registry:8080"
            },
            dependencies=["redis", "registry"],
            scaling={
                "replicas": 1,
                "max_replicas": 3
            }
        )
        
        # Message Queue Service
        self.create_service_definition(
            "redis",
            ServiceType.MESSAGE_QUEUE,
            "Redis message queue for inter-service communication",
            image="redis:7-alpine",
            ports=[6379],
            volumes=[
                {"host": "redis_data", "container": "/data"}
            ],
            health_check={
                "test": "redis-cli ping",
                "interval": "10s",
                "timeout": "3s"
            }
        )
        
        # Add message flows
        self.add_message_flow(
            "orchestrator-to-agent",
            "orchestrator",
            "research-agent",
            CommunicationPattern.ASYNCHRONOUS,
            protocol="Redis",
            message_types=["task_assignment", "status_update"]
        )
        
        self.add_message_flow(
            "agent-to-orchestrator",
            "research-agent",
            "orchestrator",
            CommunicationPattern.ASYNCHRONOUS,
            protocol="Redis",
            message_types=["task_result", "status_report"]
        )
        
    def _load_deployment_templates(self) -> Dict[str, str]:
        """Load deployment configuration templates."""
        return {
            "docker_compose": """version: '3.8'
services:
  {services}
networks:
  {networks}
volumes:
  {volumes}""",
            "kubernetes": """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  namespace: {namespace}
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
        - containerPort: {port}""",
            "cloud_run": """apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: {name}
spec:
  template:
    spec:
      containers:
      - image: {image}
        ports:
        - containerPort: {port}"""
        }
        
    def export_deployment_docs(self, output_dir: str) -> None:
        """Export all deployment documentation."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Architecture overview
        with open(output_path / "README.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_architecture_overview())
            
        # Message flow diagram
        with open(output_path / "message-flows.md", 'w', encoding='utf-8') as f:
            f.write(self.generate_message_flow_diagram())
            
        # Generate configs for each deployment
        for deployment in self.deployments:
            deployment_dir = output_path / deployment.name
            deployment_dir.mkdir(exist_ok=True)
            
            # Deployment guide
            with open(deployment_dir / "README.md", 'w', encoding='utf-8') as f:
                f.write(self.generate_deployment_guide(deployment))
                
            # Target-specific configurations
            if deployment.target == DeploymentTarget.DOCKER:
                with open(deployment_dir / "docker-compose.yml", 'w', encoding='utf-8') as f:
                    f.write(self.generate_docker_compose(deployment))
            elif deployment.target == DeploymentTarget.KUBERNETES:
                with open(deployment_dir / "deployment.yaml", 'w', encoding='utf-8') as f:
                    f.write(self.generate_kubernetes_manifest(deployment))
                    
            # Individual service configs for Cloud Run
            if deployment.target == DeploymentTarget.CLOUD_RUN:
                for service in deployment.services:
                    filename = f"{service.name}-cloudrun.yaml"
                    with open(deployment_dir / filename, 'w', encoding='utf-8') as f:
                        f.write(self.generate_cloud_run_config(service))
                        
        logger.info(f"Exported deployment docs to {output_dir}")