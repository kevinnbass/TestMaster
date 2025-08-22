"""
API Documentation Generator - OpenAPI 3.0 specification generation

This module provides comprehensive API documentation capabilities including:
- OpenAPI 3.0.3 specification generation
- Interactive documentation
- Schema validation and examples
- Multi-format documentation export
"""

import json
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import yaml

from .api_models import APIEndpoint, HTTPMethod, AuthenticationLevel

logger = logging.getLogger(__name__)


class DocumentationFormat(Enum):
    """Documentation output formats"""
    JSON = "json"
    YAML = "yaml"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class OpenAPIInfo:
    """OpenAPI specification info section"""
    title: str
    description: str
    version: str
    contact: Optional[Dict[str, str]] = None
    license: Optional[Dict[str, str]] = None
    terms_of_service: Optional[str] = None


@dataclass
class OpenAPIServer:
    """OpenAPI server configuration"""
    url: str
    description: str
    variables: Optional[Dict[str, Any]] = None


@dataclass
class OpenAPISchema:
    """OpenAPI schema definition"""
    type: str
    properties: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None
    example: Optional[Any] = None
    description: Optional[str] = None
    format: Optional[str] = None
    enum: Optional[List[str]] = None
    items: Optional['OpenAPISchema'] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary"""
        result = {"type": self.type}
        
        if self.properties:
            result["properties"] = {
                k: v.to_dict() if isinstance(v, OpenAPISchema) else v
                for k, v in self.properties.items()
            }
        if self.required:
            result["required"] = self.required
        if self.example is not None:
            result["example"] = self.example
        if self.description:
            result["description"] = self.description
        if self.format:
            result["format"] = self.format
        if self.enum:
            result["enum"] = self.enum
        if self.items:
            result["items"] = self.items.to_dict()
        
        return result


class OpenAPIGenerator:
    """
    Generates comprehensive OpenAPI 3.0.3 specifications
    
    Provides automatic documentation generation with schema inference,
    examples, and comprehensive API coverage.
    """
    
    def __init__(
        self,
        info: Optional[OpenAPIInfo] = None,
        servers: Optional[List[OpenAPIServer]] = None
    ):
        """
        Initialize OpenAPI generator
        
        Args:
            info: API information
            servers: Server configurations
        """
        self.info = info or OpenAPIInfo(
            title="TestMaster API",
            description="Comprehensive testing framework API with AI-powered capabilities",
            version="1.0.0",
            contact={
                "name": "TestMaster Support",
                "email": "support@testmaster.dev",
                "url": "https://testmaster.dev/support"
            },
            license={
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        )
        
        self.servers = servers or [
            OpenAPIServer(
                url="https://api.testmaster.dev/v1",
                description="Production server"
            ),
            OpenAPIServer(
                url="https://staging-api.testmaster.dev/v1",
                description="Staging server"
            ),
            OpenAPIServer(
                url="http://localhost:8000/v1",
                description="Development server"
            )
        ]
        
        # OpenAPI specification components
        self.spec = {
            "openapi": "3.0.3",
            "info": asdict(self.info),
            "servers": [asdict(server) for server in self.servers],
            "components": {
                "securitySchemes": self._create_security_schemes(),
                "schemas": self._create_base_schemas(),
                "responses": self._create_common_responses(),
                "parameters": self._create_common_parameters()
            },
            "paths": {},
            "tags": []
        }
        
        self.tags_registry: Set[str] = set()
        self.endpoints_count = 0
        
        logger.info("OpenAPI Generator initialized")
    
    def add_endpoint(self, endpoint: APIEndpoint):
        """
        Add endpoint to OpenAPI specification
        
        Args:
            endpoint: API endpoint to document
        """
        if endpoint.path not in self.spec["paths"]:
            self.spec["paths"][endpoint.path] = {}
        
        method_spec = self._create_method_specification(endpoint)
        self.spec["paths"][endpoint.path][endpoint.method.value.lower()] = method_spec
        
        # Register tags
        for tag in endpoint.tags:
            if tag not in self.tags_registry:
                self.tags_registry.add(tag)
                self.spec["tags"].append({
                    "name": tag,
                    "description": f"{tag} related operations"
                })
        
        self.endpoints_count += 1
        logger.debug(f"Added endpoint to spec: {endpoint.method.value} {endpoint.path}")
    
    def _create_method_specification(self, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Create method specification for endpoint"""
        spec = {
            "summary": endpoint.description or f"{endpoint.method.value} {endpoint.path}",
            "description": self._generate_detailed_description(endpoint),
            "operationId": self._generate_operation_id(endpoint),
            "tags": endpoint.tags or ["General"],
            "parameters": self._convert_parameters(endpoint.parameters),
            "responses": self._convert_responses(endpoint.responses),
            "security": self._get_security_scheme(endpoint.auth_required)
        }
        
        # Add request body for methods that typically have bodies
        if endpoint.method in [HTTPMethod.POST, HTTPMethod.PUT, HTTPMethod.PATCH]:
            spec["requestBody"] = self._create_request_body(endpoint)
        
        # Add deprecation warning if needed
        if endpoint.deprecated:
            spec["deprecated"] = True
            spec["description"] += "\n\n**⚠️ This endpoint is deprecated.**"
        
        return spec
    
    def _generate_detailed_description(self, endpoint: APIEndpoint) -> str:
        """Generate detailed description for endpoint"""
        description = endpoint.description or f"Handles {endpoint.method.value} requests to {endpoint.path}"
        
        details = []
        
        # Add authentication info
        if endpoint.auth_required != AuthenticationLevel.NONE:
            details.append(f"**Authentication:** {endpoint.auth_required.value}")
        
        # Add rate limiting info
        if endpoint.rate_limit:
            limit, limit_type = endpoint.rate_limit
            details.append(f"**Rate Limit:** {limit} requests per {limit_type.value.replace('_', ' ')}")
        
        # Add permissions info
        if endpoint.permissions:
            details.append(f"**Required Permissions:** {', '.join(endpoint.permissions)}")
        
        # Add version info
        if endpoint.version != "1.0.0":
            details.append(f"**Version:** {endpoint.version}")
        
        # Add timeout info
        if endpoint.timeout != 30:
            details.append(f"**Timeout:** {endpoint.timeout} seconds")
        
        if details:
            description += "\n\n" + "\n\n".join(details)
        
        return description
    
    def _generate_operation_id(self, endpoint: APIEndpoint) -> str:
        """Generate unique operation ID"""
        # Convert path to camelCase operation ID
        path_parts = endpoint.path.strip('/').split('/')
        method = endpoint.method.value.lower()
        
        # Handle path parameters
        clean_parts = []
        for part in path_parts:
            if part.startswith('{') and part.endswith('}'):
                clean_parts.append(f"By{part[1:-1].title()}")
            else:
                clean_parts.append(part.title())
        
        operation_id = method + ''.join(clean_parts)
        return operation_id
    
    def _convert_parameters(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert endpoint parameters to OpenAPI format"""
        converted = []
        
        for name, param_info in parameters.items():
            if isinstance(param_info, dict):
                param_spec = {
                    "name": name,
                    "in": param_info.get("in", "query"),
                    "required": param_info.get("required", False),
                    "schema": self._create_parameter_schema(param_info),
                    "description": param_info.get("description", f"The {name} parameter")
                }
                
                # Add examples
                if "example" in param_info:
                    param_spec["example"] = param_info["example"]
                elif "examples" in param_info:
                    param_spec["examples"] = param_info["examples"]
                
                converted.append(param_spec)
        
        return converted
    
    def _create_parameter_schema(self, param_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create schema for parameter"""
        param_type = param_info.get("type", "string")
        schema = {"type": param_type}
        
        # Add format if specified
        if "format" in param_info:
            schema["format"] = param_info["format"]
        
        # Add enum values if specified
        if "enum" in param_info:
            schema["enum"] = param_info["enum"]
        
        # Add validation constraints
        if param_type in ["integer", "number"]:
            if "minimum" in param_info:
                schema["minimum"] = param_info["minimum"]
            if "maximum" in param_info:
                schema["maximum"] = param_info["maximum"]
        elif param_type == "string":
            if "minLength" in param_info:
                schema["minLength"] = param_info["minLength"]
            if "maxLength" in param_info:
                schema["maxLength"] = param_info["maxLength"]
            if "pattern" in param_info:
                schema["pattern"] = param_info["pattern"]
        elif param_type == "array":
            schema["items"] = param_info.get("items", {"type": "string"})
        
        return schema
    
    def _convert_responses(self, responses: Dict[int, str]) -> Dict[str, Dict[str, Any]]:
        """Convert endpoint responses to OpenAPI format"""
        converted = {}
        
        for status_code, description in responses.items():
            response_spec = {
                "description": description,
                "content": {
                    "application/json": {
                        "schema": self._get_response_schema(status_code),
                        "examples": self._get_response_examples(status_code)
                    }
                }
            }
            
            # Add headers for specific status codes
            if status_code == 429:  # Rate limited
                response_spec["headers"] = {
                    "X-RateLimit-Limit": {
                        "description": "Request limit per time window",
                        "schema": {"type": "integer"}
                    },
                    "X-RateLimit-Remaining": {
                        "description": "Remaining requests in current window",
                        "schema": {"type": "integer"}
                    },
                    "X-RateLimit-Reset": {
                        "description": "Time when rate limit resets",
                        "schema": {"type": "integer"}
                    }
                }
            
            converted[str(status_code)] = response_spec
        
        return converted
    
    def _get_response_schema(self, status_code: int) -> Dict[str, Any]:
        """Get response schema based on status code"""
        if 200 <= status_code < 300:
            return {
                "type": "object",
                "properties": {
                    "status_code": {"type": "integer", "example": status_code},
                    "message": {"type": "string", "example": "Success"},
                    "data": {"type": "object", "description": "Response data"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "request_id": {"type": "string", "format": "uuid"}
                },
                "required": ["status_code", "timestamp"]
            }
        else:
            return {
                "type": "object",
                "properties": {
                    "status_code": {"type": "integer", "example": status_code},
                    "message": {"type": "string", "example": "Error occurred"},
                    "errors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of error messages"
                    },
                    "timestamp": {"type": "string", "format": "date-time"},
                    "request_id": {"type": "string", "format": "uuid"}
                },
                "required": ["status_code", "errors", "timestamp"]
            }
    
    def _get_response_examples(self, status_code: int) -> Dict[str, Any]:
        """Get response examples based on status code"""
        if 200 <= status_code < 300:
            return {
                "success": {
                    "summary": "Successful response",
                    "value": {
                        "status_code": status_code,
                        "message": "Operation completed successfully",
                        "data": {"result": "example data"},
                        "timestamp": "2024-01-01T12:00:00Z",
                        "request_id": "123e4567-e89b-12d3-a456-426614174000"
                    }
                }
            }
        else:
            return {
                "error": {
                    "summary": "Error response",
                    "value": {
                        "status_code": status_code,
                        "message": "An error occurred",
                        "errors": ["Detailed error message"],
                        "timestamp": "2024-01-01T12:00:00Z",
                        "request_id": "123e4567-e89b-12d3-a456-426614174000"
                    }
                }
            }
    
    def _create_request_body(self, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Create request body specification"""
        return {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "description": f"Request body for {endpoint.path}"
                    },
                    "examples": {
                        "example1": {
                            "summary": "Example request",
                            "value": self._generate_request_example(endpoint)
                        }
                    }
                }
            }
        }
    
    def _generate_request_example(self, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Generate example request body"""
        example = {}
        
        # Extract body parameters from endpoint parameters
        for name, param_info in endpoint.parameters.items():
            if isinstance(param_info, dict) and param_info.get("in") == "body":
                if param_info.get("type") == "string":
                    example[name] = "example_string"
                elif param_info.get("type") == "integer":
                    example[name] = 123
                elif param_info.get("type") == "boolean":
                    example[name] = True
                elif param_info.get("type") == "array":
                    example[name] = ["item1", "item2"]
                else:
                    example[name] = "example_value"
        
        return example if example else {"example": "request_data"}
    
    def _create_security_schemes(self) -> Dict[str, Dict[str, Any]]:
        """Create security schemes"""
        return {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for authentication"
            },
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token authentication"
            },
            "BasicAuth": {
                "type": "http",
                "scheme": "basic",
                "description": "HTTP Basic authentication"
            },
            "OAuth2": {
                "type": "oauth2",
                "flows": {
                    "authorizationCode": {
                        "authorizationUrl": "https://api.testmaster.dev/oauth/authorize",
                        "tokenUrl": "https://api.testmaster.dev/oauth/token",
                        "scopes": {
                            "read": "Read access",
                            "write": "Write access",
                            "admin": "Administrative access"
                        }
                    }
                }
            }
        }
    
    def _create_base_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Create base schema definitions"""
        return {
            "Error": {
                "type": "object",
                "required": ["status_code", "message"],
                "properties": {
                    "status_code": {"type": "integer"},
                    "message": {"type": "string"},
                    "errors": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            "SuccessResponse": {
                "type": "object",
                "required": ["status_code", "message"],
                "properties": {
                    "status_code": {"type": "integer"},
                    "message": {"type": "string"},
                    "data": {"type": "object"}
                }
            },
            "PaginatedResponse": {
                "type": "object",
                "properties": {
                    "data": {"type": "array"},
                    "pagination": {
                        "type": "object",
                        "properties": {
                            "page": {"type": "integer"},
                            "limit": {"type": "integer"},
                            "total": {"type": "integer"},
                            "pages": {"type": "integer"}
                        }
                    }
                }
            }
        }
    
    def _create_common_responses(self) -> Dict[str, Dict[str, Any]]:
        """Create common response definitions"""
        return {
            "BadRequest": {
                "description": "Bad request - invalid input",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "Unauthorized": {
                "description": "Authentication required",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "Forbidden": {
                "description": "Insufficient permissions",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "NotFound": {
                "description": "Resource not found",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "RateLimited": {
                "description": "Rate limit exceeded",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "InternalError": {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            }
        }
    
    def _create_common_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Create common parameter definitions"""
        return {
            "PageParameter": {
                "name": "page",
                "in": "query",
                "description": "Page number for pagination",
                "schema": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1
                }
            },
            "LimitParameter": {
                "name": "limit",
                "in": "query",
                "description": "Number of items per page",
                "schema": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20
                }
            },
            "SortParameter": {
                "name": "sort",
                "in": "query",
                "description": "Sort field and order",
                "schema": {
                    "type": "string",
                    "pattern": "^[a-zA-Z_]+:(asc|desc)$",
                    "example": "created_at:desc"
                }
            }
        }
    
    def _get_security_scheme(self, auth_level: AuthenticationLevel) -> List[Dict[str, Any]]:
        """Get security scheme for authentication level"""
        if auth_level == AuthenticationLevel.NONE:
            return []
        elif auth_level == AuthenticationLevel.API_KEY:
            return [{"ApiKeyAuth": []}]
        elif auth_level == AuthenticationLevel.JWT:
            return [{"BearerAuth": []}]
        elif auth_level == AuthenticationLevel.BASIC:
            return [{"BasicAuth": []}]
        elif auth_level == AuthenticationLevel.OAUTH:
            return [{"OAuth2": ["read", "write"]}]
        else:
            return [{"ApiKeyAuth": []}]
    
    def get_spec(self) -> Dict[str, Any]:
        """Get complete OpenAPI specification"""
        return self.spec.copy()
    
    def export_spec(self, format: DocumentationFormat = DocumentationFormat.JSON) -> str:
        """
        Export specification in specified format
        
        Args:
            format: Export format
            
        Returns:
            Formatted specification
        """
        if format == DocumentationFormat.JSON:
            return json.dumps(self.spec, indent=2, sort_keys=True)
        elif format == DocumentationFormat.YAML:
            return yaml.dump(self.spec, default_flow_style=False, sort_keys=True)
        elif format == DocumentationFormat.HTML:
            return self._generate_html_docs()
        elif format == DocumentationFormat.MARKDOWN:
            return self._generate_markdown_docs()
        else:
            return json.dumps(self.spec, indent=2)
    
    def _generate_html_docs(self) -> str:
        """Generate HTML documentation"""
        # Simplified HTML generation - in production would use a proper template
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.info.title} - API Documentation</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .endpoint {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .method {{ font-weight: bold; color: #2c5aa0; }}
        .path {{ font-family: monospace; background: #f5f5f5; padding: 2px 5px; }}
    </style>
</head>
<body>
    <h1>{self.info.title}</h1>
    <p>{self.info.description}</p>
    <p><strong>Version:</strong> {self.info.version}</p>
    
    <h2>Endpoints</h2>
"""
        
        for path, methods in self.spec["paths"].items():
            for method, spec in methods.items():
                html += f"""
    <div class="endpoint">
        <h3><span class="method">{method.upper()}</span> <span class="path">{path}</span></h3>
        <p>{spec.get('description', '')}</p>
        <p><strong>Tags:</strong> {', '.join(spec.get('tags', []))}</p>
    </div>
"""
        
        html += """
</body>
</html>"""
        return html
    
    def _generate_markdown_docs(self) -> str:
        """Generate Markdown documentation"""
        md = f"""# {self.info.title}

{self.info.description}

**Version:** {self.info.version}

## Endpoints

"""
        
        for path, methods in self.spec["paths"].items():
            for method, spec in methods.items():
                md += f"""### {method.upper()} {path}

{spec.get('description', '')}

**Tags:** {', '.join(spec.get('tags', []))}

"""
        
        return md
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get documentation statistics"""
        return {
            "endpoints_documented": self.endpoints_count,
            "paths_count": len(self.spec["paths"]),
            "tags_count": len(self.tags_registry),
            "schemas_count": len(self.spec["components"]["schemas"]),
            "security_schemes": len(self.spec["components"]["securitySchemes"])
        }


# Factory function
def create_openapi_generator(
    title: str = "TestMaster API",
    description: str = "Comprehensive testing framework API",
    version: str = "1.0.0"
) -> OpenAPIGenerator:
    """
    Create and configure OpenAPI generator
    
    Args:
        title: API title
        description: API description
        version: API version
        
    Returns:
        Configured OpenAPI generator
    """
    info = OpenAPIInfo(
        title=title,
        description=description,
        version=version
    )
    return OpenAPIGenerator(info=info)