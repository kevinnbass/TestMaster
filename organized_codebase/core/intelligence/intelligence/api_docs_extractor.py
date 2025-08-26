"""
API Documentation Extractor Module
Extracts and processes API documentation patterns from various sources
"""

import json
import yaml
import re
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class APIEndpoint:
    """Represents an API endpoint with documentation"""
    path: str
    method: str
    summary: str = ""
    description: str = ""
    parameters: List[Dict[str, Any]] = None
    responses: Dict[str, Any] = None
    tags: List[str] = None
    examples: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.responses is None:
            self.responses = {}
        if self.tags is None:
            self.tags = []
        if self.examples is None:
            self.examples = []


@dataclass
class APISchema:
    """Represents an API schema/model"""
    name: str
    type: str = "object"
    properties: Dict[str, Any] = None
    required: List[str] = None
    description: str = ""
    example: Any = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.required is None:
            self.required = []


class APIDocsExtractor:
    """Extracts API documentation from various formats"""
    
    def __init__(self):
        self.supported_formats = ["openapi", "swagger", "json", "yaml"]
        self.endpoints = {}
        self.schemas = {}
        self.info = {}
    
    def extract_from_openapi(self, spec_path: Path) -> Dict[str, Any]:
        """Extract documentation from OpenAPI/Swagger specification"""
        try:
            with open(spec_path, 'r', encoding='utf-8') as f:
                if spec_path.suffix.lower() in ['.yaml', '.yml']:
                    spec = yaml.safe_load(f)
                else:
                    spec = json.load(f)
            
            return self.parse_openapi_spec(spec)
        except Exception as e:
            return {"error": f"Failed to parse OpenAPI spec: {e}"}
    
    def parse_openapi_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenAPI specification into structured format"""
        parsed_spec = {
            "info": spec.get("info", {}),
            "servers": spec.get("servers", []),
            "endpoints": [],
            "schemas": []
        }
        
        # Extract endpoints
        paths = spec.get("paths", {})
        for path, path_info in paths.items():
            for method, operation in path_info.items():
                if method.lower() in ['get', 'post', 'put', 'delete', 'patch']:
                    endpoint = self.extract_endpoint_info(
                        path, method.upper(), operation
                    )
                    parsed_spec["endpoints"].append(endpoint)
        
        # Extract schemas
        components = spec.get("components", {})
        schemas = components.get("schemas", {})
        for schema_name, schema_def in schemas.items():
            schema = self.extract_schema_info(schema_name, schema_def)
            parsed_spec["schemas"].append(schema)
        
        return parsed_spec
    
    def extract_endpoint_info(self, path: str, method: str, 
                            operation: Dict[str, Any]) -> APIEndpoint:
        """Extract endpoint information from OpenAPI operation"""
        endpoint = APIEndpoint(
            path=path,
            method=method,
            summary=operation.get("summary", ""),
            description=operation.get("description", ""),
            tags=operation.get("tags", [])
        )
        
        # Extract parameters
        parameters = operation.get("parameters", [])
        for param in parameters:
            endpoint.parameters.append({
                "name": param.get("name", ""),
                "in": param.get("in", "query"),
                "required": param.get("required", False),
                "type": param.get("schema", {}).get("type", "string"),
                "description": param.get("description", "")
            })
        
        # Extract request body
        request_body = operation.get("requestBody")
        if request_body:
            content = request_body.get("content", {})
            for media_type, schema_info in content.items():
                endpoint.parameters.append({
                    "name": "body",
                    "in": "body",
                    "required": request_body.get("required", False),
                    "schema": schema_info.get("schema", {}),
                    "description": request_body.get("description", "")
                })
        
        # Extract responses
        responses = operation.get("responses", {})
        for status_code, response_info in responses.items():
            endpoint.responses[status_code] = {
                "description": response_info.get("description", ""),
                "schema": response_info.get("content", {})
            }
        
        # Extract examples
        if "examples" in operation:
            for example_name, example_info in operation["examples"].items():
                endpoint.examples.append({
                    "name": example_name,
                    "value": example_info.get("value"),
                    "description": example_info.get("description", "")
                })
        
        return endpoint
    
    def extract_schema_info(self, name: str, schema_def: Dict[str, Any]) -> APISchema:
        """Extract schema information from OpenAPI definition"""
        return APISchema(
            name=name,
            type=schema_def.get("type", "object"),
            properties=schema_def.get("properties", {}),
            required=schema_def.get("required", []),
            description=schema_def.get("description", ""),
            example=schema_def.get("example")
        )
    
    def generate_markdown_docs(self, api_spec: Dict[str, Any]) -> str:
        """Generate markdown documentation from API specification"""
        md_content = []
        
        # API Info
        info = api_spec.get("info", {})
        if info.get("title"):
            md_content.append(f"# {info['title']}")
            md_content.append("")
        
        if info.get("description"):
            md_content.append(info["description"])
            md_content.append("")
        
        if info.get("version"):
            md_content.append(f"**Version:** {info['version']}")
            md_content.append("")
        
        # Servers
        servers = api_spec.get("servers", [])
        if servers:
            md_content.append("## Base URLs")
            md_content.append("")
            for server in servers:
                url = server.get("url", "")
                description = server.get("description", "")
                if description:
                    md_content.append(f"- `{url}` - {description}")
                else:
                    md_content.append(f"- `{url}`")
            md_content.append("")
        
        # Endpoints
        endpoints = api_spec.get("endpoints", [])
        if endpoints:
            md_content.append("## Endpoints")
            md_content.append("")
            
            # Group by tags
            grouped_endpoints = {}
            for endpoint in endpoints:
                tags = endpoint.tags or ["Default"]
                for tag in tags:
                    if tag not in grouped_endpoints:
                        grouped_endpoints[tag] = []
                    grouped_endpoints[tag].append(endpoint)
            
            for tag, tag_endpoints in grouped_endpoints.items():
                md_content.append(f"### {tag}")
                md_content.append("")
                
                for endpoint in tag_endpoints:
                    endpoint_md = self.format_endpoint_markdown(endpoint)
                    md_content.append(endpoint_md)
                    md_content.append("")
        
        # Schemas
        schemas = api_spec.get("schemas", [])
        if schemas:
            md_content.append("## Data Models")
            md_content.append("")
            
            for schema in schemas:
                schema_md = self.format_schema_markdown(schema)
                md_content.append(schema_md)
                md_content.append("")
        
        return "\n".join(md_content)
    
    def format_endpoint_markdown(self, endpoint: APIEndpoint) -> str:
        """Format endpoint as markdown"""
        md_parts = []
        
        # Endpoint header
        method_badge = f"`{endpoint.method}`"
        md_parts.append(f"#### {method_badge} {endpoint.path}")
        md_parts.append("")
        
        if endpoint.summary:
            md_parts.append(endpoint.summary)
            md_parts.append("")
        
        if endpoint.description:
            md_parts.append(endpoint.description)
            md_parts.append("")
        
        # Parameters
        if endpoint.parameters:
            md_parts.append("**Parameters:**")
            md_parts.append("")
            md_parts.append("| Name | Type | Location | Required | Description |")
            md_parts.append("|------|------|----------|----------|-------------|")
            
            for param in endpoint.parameters:
                name = param.get("name", "")
                param_type = param.get("type", "string")
                location = param.get("in", "query")
                required = "Yes" if param.get("required", False) else "No"
                description = param.get("description", "")
                
                md_parts.append(f"| {name} | {param_type} | {location} | {required} | {description} |")
            
            md_parts.append("")
        
        # Responses
        if endpoint.responses:
            md_parts.append("**Responses:**")
            md_parts.append("")
            
            for status_code, response_info in endpoint.responses.items():
                description = response_info.get("description", "")
                md_parts.append(f"- `{status_code}`: {description}")
            
            md_parts.append("")
        
        return "\n".join(md_parts)
    
    def format_schema_markdown(self, schema: APISchema) -> str:
        """Format schema as markdown"""
        md_parts = []
        
        # Schema header
        md_parts.append(f"#### {schema.name}")
        md_parts.append("")
        
        if schema.description:
            md_parts.append(schema.description)
            md_parts.append("")
        
        # Properties table
        if schema.properties:
            md_parts.append("| Property | Type | Required | Description |")
            md_parts.append("|----------|------|----------|-------------|")
            
            for prop_name, prop_info in schema.properties.items():
                prop_type = prop_info.get("type", "string")
                required = "Yes" if prop_name in schema.required else "No"
                description = prop_info.get("description", "")
                
                md_parts.append(f"| {prop_name} | {prop_type} | {required} | {description} |")
            
            md_parts.append("")
        
        # Example
        if schema.example:
            md_parts.append("**Example:**")
            md_parts.append("")
            md_parts.append("```json")
            md_parts.append(json.dumps(schema.example, indent=2))
            md_parts.append("```")
            md_parts.append("")
        
        return "\n".join(md_parts)
    
    def export_to_json(self, api_spec: Dict[str, Any], output_path: Path) -> bool:
        """Export API specification to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(api_spec, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False