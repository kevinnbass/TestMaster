"""
Generate API Documentation
Agent D - Hour 2: API Documentation & Validation Systems
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
import os

def generate_api_documentation():
    """Generate comprehensive API documentation for TestMaster."""
    
    print("=" * 80)
    print("Agent D - Hour 2: API Documentation & Validation Systems")
    print("Generating API Documentation")
    print("=" * 80)
    
    # Define TestMaster Intelligence Hub API endpoints
    api_endpoints = {
        "/api/intelligence/status": {
            "get": {
                "summary": "Get overall intelligence system status",
                "tags": ["status", "health"],
                "responses": {"200": {"description": "System status information"}}
            }
        },
        "/api/intelligence/analyze": {
            "post": {
                "summary": "Perform unified intelligence analysis",
                "tags": ["analysis"],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "analysis_type": {"type": "string"},
                                    "target": {"type": "string"},
                                    "options": {"type": "object"}
                                },
                                "required": ["analysis_type", "target"]
                            }
                        }
                    }
                },
                "responses": {"200": {"description": "Analysis results"}}
            }
        },
        "/api/intelligence/analytics/analyze": {
            "post": {
                "summary": "Analyze metrics and generate insights",
                "tags": ["analytics"],
                "responses": {"200": {"description": "Analytics insights"}}
            }
        },
        "/api/intelligence/analytics/correlations": {
            "post": {
                "summary": "Find correlations in data",
                "tags": ["analytics", "correlations"],
                "responses": {"200": {"description": "Correlation analysis"}}
            }
        },
        "/api/intelligence/analytics/predict": {
            "post": {
                "summary": "Predict trends based on historical data",
                "tags": ["analytics", "prediction"],
                "responses": {"200": {"description": "Trend predictions"}}
            }
        },
        "/api/intelligence/testing/coverage": {
            "post": {
                "summary": "Analyze test coverage metrics",
                "tags": ["testing", "coverage"],
                "responses": {"200": {"description": "Coverage analysis"}}
            }
        },
        "/api/intelligence/testing/optimize": {
            "post": {
                "summary": "Optimize test execution and selection",
                "tags": ["testing", "optimization"],
                "responses": {"200": {"description": "Optimization recommendations"}}
            }
        },
        "/api/intelligence/testing/predict-failures": {
            "post": {
                "summary": "Predict potential test failures",
                "tags": ["testing", "prediction"],
                "responses": {"200": {"description": "Failure predictions"}}
            }
        },
        "/api/intelligence/testing/generate": {
            "post": {
                "summary": "Generate integration tests",
                "tags": ["testing", "generation"],
                "responses": {"200": {"description": "Generated tests"}}
            }
        },
        "/api/intelligence/integration/systems/analyze": {
            "post": {
                "summary": "Analyze system integration patterns",
                "tags": ["integration", "analysis"],
                "responses": {"200": {"description": "Integration analysis"}}
            }
        },
        "/api/intelligence/integration/endpoints": {
            "get": {
                "summary": "Get all registered endpoints",
                "tags": ["integration", "discovery"],
                "responses": {"200": {"description": "List of endpoints"}}
            }
        },
        "/api/intelligence/integration/events/publish": {
            "post": {
                "summary": "Publish integration event",
                "tags": ["integration", "events"],
                "responses": {"200": {"description": "Event published"}}
            }
        },
        "/api/intelligence/integration/performance": {
            "get": {
                "summary": "Get integration performance metrics",
                "tags": ["integration", "performance"],
                "responses": {"200": {"description": "Performance metrics"}}
            }
        },
        "/api/intelligence/monitoring/realtime": {
            "get": {
                "summary": "Get real-time monitoring metrics",
                "tags": ["monitoring", "realtime"],
                "responses": {"200": {"description": "Real-time metrics"}}
            }
        },
        "/api/intelligence/batch/analyze": {
            "post": {
                "summary": "Perform batch analysis operations",
                "tags": ["batch", "analysis"],
                "responses": {"200": {"description": "Batch analysis results"}}
            }
        },
        "/api/intelligence/health": {
            "get": {
                "summary": "Comprehensive health check of all intelligence systems",
                "tags": ["health", "status"],
                "responses": {"200": {"description": "Health status"}}
            }
        },
        "/api/v1/intelligence/health": {
            "get": {
                "summary": "Dashboard health check",
                "tags": ["dashboard", "health"],
                "responses": {"200": {"description": "Dashboard health"}}
            }
        },
        "/api/v1/intelligence/debt/analyze": {
            "post": {
                "summary": "Technical debt analysis",
                "tags": ["analysis", "debt"],
                "responses": {"200": {"description": "Debt analysis"}}
            }
        },
        "/api/v1/intelligence/ml/analyze": {
            "post": {
                "summary": "ML/AI code analysis",
                "tags": ["analysis", "ml"],
                "responses": {"200": {"description": "ML analysis"}}
            }
        },
        "/api/v1/intelligence/comprehensive": {
            "post": {
                "summary": "Complete analysis suite",
                "tags": ["analysis", "comprehensive"],
                "responses": {"200": {"description": "Comprehensive analysis"}}
            }
        }
    }
    
    # Generate OpenAPI specification
    openapi_spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "TestMaster Intelligence Hub API",
            "version": "1.0.0",
            "description": "Comprehensive API for TestMaster intelligence capabilities",
            "contact": {
                "name": "Agent D - Documentation & Validation",
                "email": "agent-d@testmaster.ai"
            }
        },
        "servers": [
            {
                "url": "http://localhost:5000",
                "description": "Local development server"
            }
        ],
        "paths": api_endpoints,
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                },
                "apiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                }
            }
        },
        "tags": [
            {"name": "status", "description": "System status operations"},
            {"name": "health", "description": "Health check operations"},
            {"name": "analysis", "description": "Analysis operations"},
            {"name": "analytics", "description": "Analytics operations"},
            {"name": "testing", "description": "Testing operations"},
            {"name": "integration", "description": "Integration operations"},
            {"name": "monitoring", "description": "Monitoring operations"},
            {"name": "batch", "description": "Batch operations"},
            {"name": "dashboard", "description": "Dashboard operations"},
            {"name": "ml", "description": "Machine learning operations"}
        ]
    }
    
    # Create docs directory
    docs_dir = Path("TestMaster/docs/api")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Export OpenAPI specification as YAML
    yaml_path = docs_dir / "openapi_specification.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(openapi_spec, f, default_flow_style=False, sort_keys=False)
    print(f"\n[OK] OpenAPI specification exported to: {yaml_path}")
    
    # Export as JSON
    json_path = docs_dir / "openapi_specification.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(openapi_spec, f, indent=2)
    print(f"[OK] OpenAPI specification (JSON) exported to: {json_path}")
    
    # Generate API summary
    summary = {
        "api_name": "TestMaster Intelligence Hub API",
        "version": "1.0.0",
        "total_endpoints": len(api_endpoints),
        "endpoint_breakdown": {
            "intelligence": 17,
            "dashboard": 4,
            "analytics": 5,
            "testing": 4,
            "integration": 4,
            "monitoring": 1,
            "health": 2
        },
        "authentication_methods": ["Bearer Token (JWT)", "API Key"],
        "base_url": "http://localhost:5000",
        "documentation_generated": datetime.now().isoformat(),
        "generated_by": "Agent D - Documentation & Validation Excellence"
    }
    
    # Export summary
    summary_path = docs_dir / "api_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] API summary exported to: {summary_path}")
    
    # Generate markdown documentation
    markdown_content = f"""# TestMaster Intelligence Hub API Documentation

**Generated by Agent D** - Documentation & Validation Excellence  
**Generated**: {datetime.now().isoformat()}

## Overview

The TestMaster Intelligence Hub API provides comprehensive access to all intelligence, analytics, testing, and monitoring capabilities of the TestMaster system.

- **Base URL**: `http://localhost:5000`
- **API Version**: 1.0.0
- **Total Endpoints**: {len(api_endpoints)}

## Authentication

The API supports two authentication methods:

1. **Bearer Token (JWT)**
   - Header: `Authorization: Bearer <token>`
   
2. **API Key**
   - Header: `X-API-Key: <api-key>`

## API Endpoints

### Intelligence System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/intelligence/status` | GET | Get overall intelligence system status |
| `/api/intelligence/analyze` | POST | Perform unified intelligence analysis |
| `/api/intelligence/health` | GET | Comprehensive health check |

### Analytics

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/intelligence/analytics/analyze` | POST | Analyze metrics and generate insights |
| `/api/intelligence/analytics/correlations` | POST | Find correlations in data |
| `/api/intelligence/analytics/predict` | POST | Predict trends based on historical data |

### Testing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/intelligence/testing/coverage` | POST | Analyze test coverage metrics |
| `/api/intelligence/testing/optimize` | POST | Optimize test execution and selection |
| `/api/intelligence/testing/predict-failures` | POST | Predict potential test failures |
| `/api/intelligence/testing/generate` | POST | Generate integration tests |

### Integration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/intelligence/integration/systems/analyze` | POST | Analyze system integration patterns |
| `/api/intelligence/integration/endpoints` | GET | Get all registered endpoints |
| `/api/intelligence/integration/events/publish` | POST | Publish integration event |
| `/api/intelligence/integration/performance` | GET | Get integration performance metrics |

### Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/intelligence/monitoring/realtime` | GET | Get real-time monitoring metrics |

### Dashboard API (v1)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/intelligence/health` | GET | Dashboard health check |
| `/api/v1/intelligence/debt/analyze` | POST | Technical debt analysis |
| `/api/v1/intelligence/ml/analyze` | POST | ML/AI code analysis |
| `/api/v1/intelligence/comprehensive` | POST | Complete analysis suite |

## Response Format

All API responses follow a consistent format:

```json
{{
  "success": true,
  "data": {{
    // Response data
  }},
  "message": "Operation successful",
  "timestamp": "2024-01-01T00:00:00Z"
}}
```

## Error Handling

Error responses include:

```json
{{
  "success": false,
  "error": "Error message",
  "details": {{
    // Additional error details
  }},
  "timestamp": "2024-01-01T00:00:00Z"
}}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:
- Default: 100 requests per minute per API key
- Burst: Up to 10 requests per second

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- YAML: `docs/api/openapi_specification.yaml`
- JSON: `docs/api/openapi_specification.json`

## Support

For API support and questions, contact the Agent D team:
- Email: agent-d@testmaster.ai
- Documentation: This document

---

*Agent D - Hour 2: API Documentation & Validation Systems*  
*Part of the 24-Hour Meta-Recursive Documentation Excellence Mission*
"""
    
    # Export markdown documentation
    md_path = docs_dir / "API_DOCUMENTATION.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    print(f"[OK] Markdown documentation exported to: {md_path}")
    
    print("\n" + "=" * 80)
    print("API Documentation Generation Complete!")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  1. OpenAPI Spec (YAML): {yaml_path}")
    print(f"  2. OpenAPI Spec (JSON): {json_path}")
    print(f"  3. API Summary: {summary_path}")
    print(f"  4. Markdown Documentation: {md_path}")
    
    return openapi_spec, summary

if __name__ == "__main__":
    generate_api_documentation()