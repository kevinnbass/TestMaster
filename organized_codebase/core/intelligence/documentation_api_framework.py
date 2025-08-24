#!/usr/bin/env python3
"""
Documentation API & Integration Layer - Agent D Hour 6
Unified API for documentation access, generation, and real-time updates
"""

import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
from collections import defaultdict
from abc import ABC, abstractmethod
import threading
import queue

@dataclass
class DocumentationEndpoint:
    """Represents a documentation API endpoint"""
    path: str
    method: str  # GET, POST, PUT, DELETE
    description: str
    parameters: List[str] = field(default_factory=list)
    response_type: str = "json"
    authentication_required: bool = False
    rate_limit: int = 100  # requests per minute
    cache_duration: int = 300  # seconds
    integration_points: List[str] = field(default_factory=list)

@dataclass
class WebhookTrigger:
    """Represents a webhook trigger for documentation events"""
    name: str
    event_type: str  # documentation_updated, api_changed, config_modified
    url: str
    active: bool = True
    retry_count: int = 3
    timeout: int = 30
    headers: Dict[str, str] = field(default_factory=dict)
    last_triggered: Optional[datetime] = None
    
@dataclass 
class IntegrationSystem:
    """Represents an integrated documentation system"""
    name: str
    system_type: str  # internal, external, third_party
    api_endpoints: List[DocumentationEndpoint] = field(default_factory=list)
    webhooks: List[WebhookTrigger] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    last_sync: Optional[datetime] = None
    
@dataclass
class DocumentationRequest:
    """Represents an API request for documentation"""
    request_id: str
    endpoint: str
    method: str
    parameters: Dict[str, Any]
    timestamp: datetime
    user_agent: str = ""
    ip_address: str = ""
    response_time: Optional[float] = None
    status_code: Optional[int] = None

@dataclass
class DocumentationResponse:
    """Represents an API response with documentation"""
    request_id: str
    data: Any
    status_code: int
    headers: Dict[str, str]
    timestamp: datetime
    cache_hit: bool = False
    processing_time: float = 0.0

class DocumentationEventType(Enum):
    """Types of documentation events"""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    GENERATED = "generated"
    VALIDATED = "validated"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class DocumentationAPIFramework:
    """Comprehensive documentation API and integration framework"""
    
    def __init__(self, base_path: Union[str, Path] = ".", api_base_url: str = "/api/documentation"):
        self.base_path = Path(base_path)
        self.api_base_url = api_base_url
        self.endpoints: Dict[str, DocumentationEndpoint] = {}
        self.webhooks: Dict[str, WebhookTrigger] = {}
        self.integration_systems: Dict[str, IntegrationSystem] = {}
        self.request_cache: Dict[str, DocumentationResponse] = {}
        self.event_queue = queue.Queue()
        self.metrics = defaultdict(int)
        self._initialize_endpoints()
        self._initialize_webhooks()
        self._initialize_integrations()
        
    def _initialize_endpoints(self):
        """Initialize documentation API endpoints"""
        
        # Core documentation endpoints
        self.register_endpoint(DocumentationEndpoint(
            path=f"{self.api_base_url}/generate",
            method="POST",
            description="Generate documentation for specified components",
            parameters=["component_type", "format", "include_examples"],
            response_type="json",
            integration_points=["documentation_generator", "template_engine"]
        ))
        
        self.register_endpoint(DocumentationEndpoint(
            path=f"{self.api_base_url}/validate",
            method="POST",
            description="Validate documentation completeness and accuracy",
            parameters=["documentation_id", "validation_rules"],
            response_type="json",
            integration_points=["validation_framework", "quality_checker"]
        ))
        
        self.register_endpoint(DocumentationEndpoint(
            path=f"{self.api_base_url}/search",
            method="GET",
            description="Search documentation with advanced queries",
            parameters=["query", "filters", "limit", "offset"],
            response_type="json",
            cache_duration=600,
            integration_points=["search_engine", "knowledge_base"]
        ))
        
        self.register_endpoint(DocumentationEndpoint(
            path=f"{self.api_base_url}/catalog",
            method="GET",
            description="Get complete documentation catalog",
            parameters=["category", "format", "include_metadata"],
            response_type="json",
            cache_duration=1800,
            integration_points=["catalog_service", "metadata_store"]
        ))
        
        self.register_endpoint(DocumentationEndpoint(
            path=f"{self.api_base_url}/sync",
            method="POST",
            description="Synchronize documentation across systems",
            parameters=["source_system", "target_system", "sync_mode"],
            response_type="json",
            authentication_required=True,
            integration_points=["sync_service", "integration_manager"]
        ))
        
        self.register_endpoint(DocumentationEndpoint(
            path=f"{self.api_base_url}/export",
            method="POST",
            description="Export documentation in various formats",
            parameters=["format", "include_assets", "compress"],
            response_type="binary",
            integration_points=["export_service", "format_converter"]
        ))
        
        self.register_endpoint(DocumentationEndpoint(
            path=f"{self.api_base_url}/metrics",
            method="GET",
            description="Get documentation metrics and analytics",
            parameters=["time_range", "metric_types", "aggregation"],
            response_type="json",
            cache_duration=300,
            integration_points=["analytics_service", "metrics_collector"]
        ))
        
        self.register_endpoint(DocumentationEndpoint(
            path=f"{self.api_base_url}/health",
            method="GET",
            description="Check documentation system health status",
            parameters=[],
            response_type="json",
            cache_duration=60,
            integration_points=["health_monitor", "system_checker"]
        ))
    
    def _initialize_webhooks(self):
        """Initialize webhook triggers"""
        
        # Documentation update webhook
        self.register_webhook(WebhookTrigger(
            name="documentation_updates",
            event_type="documentation_updated",
            url="http://localhost:5000/webhooks/documentation",
            headers={"Content-Type": "application/json"}
        ))
        
        # API change webhook
        self.register_webhook(WebhookTrigger(
            name="api_changes",
            event_type="api_changed",
            url="http://localhost:5000/webhooks/api",
            headers={"Content-Type": "application/json"}
        ))
        
        # Configuration change webhook
        self.register_webhook(WebhookTrigger(
            name="config_changes",
            event_type="config_modified",
            url="http://localhost:5000/webhooks/config",
            headers={"Content-Type": "application/json"}
        ))
    
    def _initialize_integrations(self):
        """Initialize integration systems"""
        
        # Internal documentation systems
        self.register_integration(IntegrationSystem(
            name="knowledge_management",
            system_type="internal",
            configuration={
                "enabled": True,
                "sync_interval": 3600,
                "auto_index": True
            }
        ))
        
        self.register_integration(IntegrationSystem(
            name="api_validation",
            system_type="internal",
            configuration={
                "enabled": True,
                "validation_on_change": True,
                "openapi_generation": True
            }
        ))
        
        self.register_integration(IntegrationSystem(
            name="legacy_documentation",
            system_type="internal",
            configuration={
                "enabled": True,
                "migration_mode": "progressive",
                "preserve_history": True
            }
        ))
        
        # External documentation systems
        self.register_integration(IntegrationSystem(
            name="github_wiki",
            system_type="external",
            configuration={
                "enabled": False,
                "repo_url": "",
                "sync_bidirectional": False
            }
        ))
        
        self.register_integration(IntegrationSystem(
            name="confluence",
            system_type="third_party",
            configuration={
                "enabled": False,
                "api_url": "",
                "space_key": ""
            }
        ))
    
    def register_endpoint(self, endpoint: DocumentationEndpoint):
        """Register a documentation API endpoint"""
        self.endpoints[endpoint.path] = endpoint
        self.metrics['endpoints_registered'] += 1
        
    def register_webhook(self, webhook: WebhookTrigger):
        """Register a webhook trigger"""
        self.webhooks[webhook.name] = webhook
        self.metrics['webhooks_registered'] += 1
        
    def register_integration(self, integration: IntegrationSystem):
        """Register an integration system"""
        self.integration_systems[integration.name] = integration
        self.metrics['integrations_registered'] += 1
    
    def handle_request(self, request: DocumentationRequest) -> DocumentationResponse:
        """Handle an API request for documentation"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self.request_cache:
            cached_response = self.request_cache[cache_key]
            if self._is_cache_valid(cached_response):
                cached_response.cache_hit = True
                self.metrics['cache_hits'] += 1
                return cached_response
        
        # Process request based on endpoint
        endpoint = self.endpoints.get(request.endpoint)
        if not endpoint:
            return DocumentationResponse(
                request_id=request.request_id,
                data={"error": "Endpoint not found"},
                status_code=404,
                headers={"Content-Type": "application/json"},
                timestamp=datetime.now()
            )
        
        # Route to appropriate handler
        response_data = self._route_request(endpoint, request)
        
        # Create response
        response = DocumentationResponse(
            request_id=request.request_id,
            data=response_data,
            status_code=200,
            headers={"Content-Type": endpoint.response_type},
            timestamp=datetime.now(),
            processing_time=time.time() - start_time
        )
        
        # Cache response if applicable
        if endpoint.cache_duration > 0:
            self.request_cache[cache_key] = response
        
        # Update metrics
        self.metrics['requests_processed'] += 1
        self.metrics['total_processing_time'] += response.processing_time
        
        return response
    
    def _route_request(self, endpoint: DocumentationEndpoint, request: DocumentationRequest) -> Any:
        """Route request to appropriate handler"""
        
        # Map endpoint paths to handler methods
        handlers = {
            f"{self.api_base_url}/generate": self._handle_generate,
            f"{self.api_base_url}/validate": self._handle_validate,
            f"{self.api_base_url}/search": self._handle_search,
            f"{self.api_base_url}/catalog": self._handle_catalog,
            f"{self.api_base_url}/sync": self._handle_sync,
            f"{self.api_base_url}/export": self._handle_export,
            f"{self.api_base_url}/metrics": self._handle_metrics,
            f"{self.api_base_url}/health": self._handle_health
        }
        
        handler = handlers.get(endpoint.path)
        if handler:
            return handler(request.parameters)
        else:
            return {"error": "Handler not implemented"}
    
    def _handle_generate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle documentation generation request"""
        component_type = parameters.get('component_type', 'all')
        format_type = parameters.get('format', 'markdown')
        include_examples = parameters.get('include_examples', True)
        
        # Simulate documentation generation
        result = {
            "status": "success",
            "component_type": component_type,
            "format": format_type,
            "documentation": {
                "sections": ["overview", "api", "examples", "configuration"],
                "total_pages": 42,
                "generated_at": datetime.now().isoformat()
            },
            "examples_included": include_examples
        }
        
        # Trigger documentation generated event
        self._trigger_event(DocumentationEventType.GENERATED, result)
        
        return result
    
    def _handle_validate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle documentation validation request"""
        doc_id = parameters.get('documentation_id', '')
        rules = parameters.get('validation_rules', ['completeness', 'accuracy'])
        
        # Simulate validation
        validation_result = {
            "status": "success",
            "documentation_id": doc_id,
            "validation_results": {
                "completeness": 95.5,
                "accuracy": 98.2,
                "consistency": 97.0,
                "coverage": 92.3
            },
            "issues_found": 3,
            "recommendations": [
                "Add more code examples",
                "Update API endpoint descriptions",
                "Include error handling documentation"
            ]
        }
        
        # Trigger validation event
        self._trigger_event(DocumentationEventType.VALIDATED, validation_result)
        
        return validation_result
    
    def _handle_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle documentation search request"""
        query = parameters.get('query', '')
        filters = parameters.get('filters', {})
        limit = parameters.get('limit', 10)
        offset = parameters.get('offset', 0)
        
        # Simulate search results
        search_results = {
            "query": query,
            "total_results": 156,
            "results": [
                {
                    "id": f"doc_{i}",
                    "title": f"Documentation Item {i}",
                    "type": "api" if i % 2 == 0 else "guide",
                    "relevance_score": 0.95 - (i * 0.05),
                    "excerpt": f"This is a sample excerpt for result {i}..."
                }
                for i in range(offset, min(offset + limit, 10))
            ],
            "filters_applied": filters,
            "search_time": 0.234
        }
        
        return search_results
    
    def _handle_catalog(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle documentation catalog request"""
        category = parameters.get('category', 'all')
        format_type = parameters.get('format', 'json')
        include_metadata = parameters.get('include_metadata', True)
        
        # Simulate catalog
        catalog = {
            "category": category,
            "total_items": 1245,
            "categories": {
                "api": 342,
                "guides": 189,
                "tutorials": 267,
                "references": 447
            },
            "last_updated": datetime.now().isoformat()
        }
        
        if include_metadata:
            catalog["metadata"] = {
                "version": "2.0",
                "language": "en",
                "contributors": 23,
                "total_words": 234567
            }
        
        return catalog
    
    def _handle_sync(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle documentation synchronization request"""
        source = parameters.get('source_system', 'local')
        target = parameters.get('target_system', 'remote')
        mode = parameters.get('sync_mode', 'incremental')
        
        # Simulate sync operation
        sync_result = {
            "status": "success",
            "source": source,
            "target": target,
            "mode": mode,
            "items_synced": 47,
            "items_updated": 12,
            "items_created": 35,
            "sync_duration": 3.456,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update last sync time for integration systems
        if source in self.integration_systems:
            self.integration_systems[source].last_sync = datetime.now()
        
        return sync_result
    
    def _handle_export(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle documentation export request"""
        format_type = parameters.get('format', 'pdf')
        include_assets = parameters.get('include_assets', True)
        compress = parameters.get('compress', False)
        
        # Simulate export
        export_result = {
            "status": "success",
            "format": format_type,
            "file_size": 15678234,
            "pages": 234,
            "assets_included": include_assets,
            "compressed": compress,
            "download_url": f"/downloads/documentation.{format_type}",
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        # Trigger export event
        self._trigger_event(DocumentationEventType.PUBLISHED, export_result)
        
        return export_result
    
    def _handle_metrics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle documentation metrics request"""
        time_range = parameters.get('time_range', 'last_7_days')
        metric_types = parameters.get('metric_types', ['all'])
        aggregation = parameters.get('aggregation', 'daily')
        
        # Return actual metrics
        metrics = {
            "time_range": time_range,
            "aggregation": aggregation,
            "metrics": {
                "total_requests": self.metrics['requests_processed'],
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "average_response_time": self._calculate_avg_response_time(),
                "endpoints_registered": self.metrics['endpoints_registered'],
                "webhooks_triggered": self.metrics['webhooks_triggered'],
                "integrations_active": self._count_active_integrations()
            },
            "endpoint_usage": self._get_endpoint_usage_stats(),
            "integration_status": self._get_integration_status()
        }
        
        return metrics
    
    def _handle_health(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health check request"""
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "api": "operational",
                "cache": "operational",
                "webhooks": "operational",
                "integrations": self._check_integrations_health()
            },
            "metrics": {
                "uptime": "99.99%",
                "response_time": f"{self._calculate_avg_response_time():.3f}s",
                "error_rate": "0.01%"
            }
        }
        
        return health_status
    
    def trigger_webhook(self, event_type: str, data: Any):
        """Trigger webhooks for an event"""
        
        for webhook_name, webhook in self.webhooks.items():
            if webhook.event_type == event_type and webhook.active:
                # Simulate webhook trigger
                webhook.last_triggered = datetime.now()
                self.metrics['webhooks_triggered'] += 1
                
                # Queue webhook for async processing
                self.event_queue.put({
                    'webhook': webhook,
                    'data': data,
                    'timestamp': datetime.now()
                })
    
    def _trigger_event(self, event_type: DocumentationEventType, data: Any):
        """Trigger a documentation event"""
        
        # Map event types to webhook event types
        event_mapping = {
            DocumentationEventType.CREATED: "documentation_updated",
            DocumentationEventType.UPDATED: "documentation_updated",
            DocumentationEventType.DELETED: "documentation_updated",
            DocumentationEventType.GENERATED: "documentation_updated",
            DocumentationEventType.VALIDATED: "documentation_updated",
            DocumentationEventType.PUBLISHED: "documentation_updated",
            DocumentationEventType.ARCHIVED: "documentation_updated"
        }
        
        webhook_event = event_mapping.get(event_type)
        if webhook_event:
            self.trigger_webhook(webhook_event, data)
    
    def _generate_cache_key(self, request: DocumentationRequest) -> str:
        """Generate cache key for request"""
        key_parts = [
            request.endpoint,
            request.method,
            json.dumps(request.parameters, sort_keys=True)
        ]
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, response: DocumentationResponse) -> bool:
        """Check if cached response is still valid"""
        endpoint = self.endpoints.get(response.request_id)
        if not endpoint:
            return False
        
        age = (datetime.now() - response.timestamp).total_seconds()
        return age < endpoint.cache_duration
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.metrics['requests_processed']
        if total == 0:
            return 0.0
        return (self.metrics['cache_hits'] / total) * 100
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time"""
        total_requests = self.metrics['requests_processed']
        if total_requests == 0:
            return 0.0
        return self.metrics['total_processing_time'] / total_requests
    
    def _count_active_integrations(self) -> int:
        """Count active integration systems"""
        return sum(1 for system in self.integration_systems.values() 
                  if system.status == "active")
    
    def _get_endpoint_usage_stats(self) -> Dict[str, int]:
        """Get endpoint usage statistics"""
        # Simulate usage stats
        return {
            endpoint.path: self.metrics.get(f"endpoint_{endpoint.path}", 0)
            for endpoint in self.endpoints.values()
        }
    
    def _get_integration_status(self) -> Dict[str, str]:
        """Get integration system status"""
        return {
            name: system.status
            for name, system in self.integration_systems.items()
        }
    
    def _check_integrations_health(self) -> str:
        """Check health of all integrations"""
        unhealthy = sum(1 for system in self.integration_systems.values() 
                       if system.status != "active")
        
        if unhealthy == 0:
            return "operational"
        elif unhealthy < len(self.integration_systems) / 2:
            return "degraded"
        else:
            return "critical"
    
    def generate_integration_dashboard(self) -> str:
        """Generate HTML dashboard for documentation API"""
        
        dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Documentation API Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric-card { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 24px; font-weight: bold; color: #3498db; }
        .metric-label { color: #7f8c8d; margin-top: 5px; }
        .endpoint-list { background: white; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .endpoint-item { padding: 10px; border-bottom: 1px solid #ecf0f1; }
        .endpoint-method { display: inline-block; padding: 2px 8px; border-radius: 3px; font-weight: bold; }
        .method-get { background: #27ae60; color: white; }
        .method-post { background: #3498db; color: white; }
        .integration-status { display: inline-block; padding: 4px 10px; border-radius: 3px; }
        .status-active { background: #27ae60; color: white; }
        .status-inactive { background: #e74c3c; color: white; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Documentation API Integration Dashboard</h1>
        <p>Real-time monitoring and management of documentation systems</p>
    </div>
    
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{endpoints}</div>
            <div class="metric-label">API Endpoints</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{webhooks}</div>
            <div class="metric-label">Webhook Triggers</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{integrations}</div>
            <div class="metric-label">Integration Systems</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{cache_rate:.1f}%</div>
            <div class="metric-label">Cache Hit Rate</div>
        </div>
    </div>
    
    <div class="endpoint-list">
        <h2>API Endpoints</h2>
        {endpoint_list}
    </div>
    
    <div class="endpoint-list">
        <h2>Integration Systems</h2>
        {integration_list}
    </div>
    
    <div class="endpoint-list">
        <h2>Active Webhooks</h2>
        {webhook_list}
    </div>
</body>
</html>
"""
        
        # Generate endpoint list HTML
        endpoint_html = ""
        for endpoint in self.endpoints.values():
            method_class = f"method-{endpoint.method.lower()}"
            endpoint_html += f"""
            <div class="endpoint-item">
                <span class="endpoint-method {method_class}">{endpoint.method}</span>
                <strong>{endpoint.path}</strong> - {endpoint.description}
            </div>
            """
        
        # Generate integration list HTML
        integration_html = ""
        for name, system in self.integration_systems.items():
            status_class = "status-active" if system.status == "active" else "status-inactive"
            integration_html += f"""
            <div class="endpoint-item">
                <strong>{name}</strong>
                <span class="integration-status {status_class}">{system.status}</span>
                - Type: {system.system_type}
            </div>
            """
        
        # Generate webhook list HTML
        webhook_html = ""
        for webhook in self.webhooks.values():
            status = "Active" if webhook.active else "Inactive"
            webhook_html += f"""
            <div class="endpoint-item">
                <strong>{webhook.name}</strong> - {webhook.event_type}
                <br>URL: {webhook.url}
                <br>Status: {status}
            </div>
            """
        
        # Fill in the template
        dashboard = dashboard_html.format(
            endpoints=len(self.endpoints),
            webhooks=len(self.webhooks),
            integrations=len(self.integration_systems),
            cache_rate=self._calculate_cache_hit_rate(),
            endpoint_list=endpoint_html,
            integration_list=integration_html,
            webhook_list=webhook_html
        )
        
        return dashboard
    
    def save_integration_report(self, output_path: Optional[Path] = None) -> Path:
        """Save integration report to JSON file"""
        if not output_path:
            output_path = self.base_path / "TestMaster/docs/api_integration/integration_report.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'api_base_url': self.api_base_url,
            'summary': {
                'total_endpoints': len(self.endpoints),
                'total_webhooks': len(self.webhooks),
                'total_integrations': len(self.integration_systems),
                'active_integrations': self._count_active_integrations()
            },
            'endpoints': [
                {
                    'path': ep.path,
                    'method': ep.method,
                    'description': ep.description,
                    'parameters': ep.parameters,
                    'cache_duration': ep.cache_duration,
                    'integration_points': ep.integration_points
                }
                for ep in self.endpoints.values()
            ],
            'webhooks': [
                {
                    'name': wh.name,
                    'event_type': wh.event_type,
                    'url': wh.url,
                    'active': wh.active,
                    'last_triggered': wh.last_triggered.isoformat() if wh.last_triggered else None
                }
                for wh in self.webhooks.values()
            ],
            'integrations': [
                {
                    'name': sys.name,
                    'type': sys.system_type,
                    'status': sys.status,
                    'configuration': sys.configuration,
                    'last_sync': sys.last_sync.isoformat() if sys.last_sync else None
                }
                for sys in self.integration_systems.values()
            ],
            'metrics': {
                'requests_processed': self.metrics['requests_processed'],
                'cache_hit_rate': self._calculate_cache_hit_rate(),
                'average_response_time': self._calculate_avg_response_time(),
                'webhooks_triggered': self.metrics['webhooks_triggered']
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"Integration report saved to: {output_path}")
        return output_path