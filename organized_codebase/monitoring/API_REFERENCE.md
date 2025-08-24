# TestMaster Dashboard API Reference

Complete API documentation for the TestMaster Dashboard v2.0

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Response Format](#response-format)
- [Error Handling](#error-handling)
- [API Endpoints](#api-endpoints)
  - [Health Check](#health-check)
  - [Performance API](#performance-api)
  - [Analytics API](#analytics-api)
  - [Workflow API](#workflow-api)
  - [LLM API](#llm-api)
  - [Tests API](#tests-api)
  - [Refactor API](#refactor-api)

## Overview

The TestMaster Dashboard API provides real-time monitoring, performance analytics, LLM integration, and workflow management capabilities. All endpoints return JSON responses with consistent formatting.

### Key Features
- Real-time performance monitoring (100ms collection interval)
- LLM integration with toggle functionality
- Workflow and test management
- Enhanced error handling with detailed responses
- Production-ready with Gunicorn deployment

## Base URL

```
http://localhost:5000/api
```

Production deployments will use HTTPS with appropriate domain.

## Authentication

Currently no authentication is required. Future versions may implement API keys or OAuth.

## Response Format

All API responses follow this standard format:

### Success Response
```json
{
  "status": "success",
  "data": { /* endpoint-specific data */ },
  "timestamp": "2025-08-18T13:15:30.123456"
}
```

### Error Response
```json
{
  "status": "error",
  "error_code": "ERROR_CODE",
  "message": "Human-readable error message",
  "timestamp": "2025-08-18T13:15:30.123456",
  "details": {
    "field": "parameter_name",
    "type": "ValidationError"
  },
  "request_id": "endpoint_name_1692373530.123"
}
```

## Error Handling

The API uses enhanced error handling with the following error codes:

| Error Code | HTTP Status | Description |
|------------|------------|-------------|
| `BAD_REQUEST` | 400 | Invalid request parameters |
| `VALIDATION_ERROR` | 400 | Input validation failure |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Access denied |
| `NOT_FOUND` | 404 | Resource not found |
| `METHOD_NOT_ALLOWED` | 405 | HTTP method not supported |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Unexpected server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |
| `MONITOR_ERROR` | 503 | Monitoring system unavailable |
| `CACHE_ERROR` | 503 | Cache system failure |

# API Endpoints

## Health Check

### GET /api/health

Get dashboard system health status.

#### Response
```json
{
  "status": "healthy",
  "timestamp": "2025-08-18T13:15:30.123456",
  "uptime_seconds": 3600.5,
  "monitoring_active": true,
  "version": "2.0.0"
}
```

---

## Performance API

Real-time performance monitoring endpoints for CPU, memory, and network metrics.

### GET /api/performance/realtime

Get real-time performance metrics for scrolling charts. Called every 100ms by frontend.

#### Query Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `codebase` | string | No | `/testmaster` | Specific codebase to monitor |

#### Response
```json
{
  "status": "success",
  "timeseries": {
    "cpu_usage": [23.5, 24.1, 22.8],
    "memory_usage_mb": [145.2, 146.8, 144.9],
    "network_kb_s": [5.2, 6.1, 4.8]
  },
  "timestamp": "2025-08-18T13:15:30.123456"
}
```

#### Error Responses
- `503 MONITOR_ERROR`: Performance monitor unavailable

---

### GET /api/performance/history

Get historical performance data.

#### Query Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `codebase` | string | No | `/testmaster` | Specific codebase |
| `hours` | integer | No | 1 | Hours of history (1-168) |

#### Response
```json
{
  "status": "success",
  "data": {
    "cpu_usage": [/* historical data */],
    "memory_usage_mb": [/* historical data */],
    "network_kb_s": [/* historical data */],
    "timestamps": [/* corresponding timestamps */]
  },
  "codebase": "/testmaster",
  "hours": 1,
  "timestamp": "2025-08-18T13:15:30.123456"
}
```

#### Error Responses
- `400 VALIDATION_ERROR`: Invalid hours parameter (must be 1-168)
- `503 MONITOR_ERROR`: Performance monitor unavailable

---

### GET /api/performance/summary

Get performance summary statistics.

#### Query Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `codebase` | string | No | `/testmaster` | Specific codebase |

#### Response
```json
{
  "status": "success",
  "summary": {
    "avg_cpu": 25.4,
    "peak_cpu": 67.8,
    "avg_memory_mb": 156.2,
    "peak_memory_mb": 289.1,
    "avg_network_kb_s": 7.3,
    "peak_network_kb_s": 45.2,
    "uptime_hours": 12.5
  },
  "codebase": "/testmaster",
  "timestamp": "2025-08-18T13:15:30.123456"
}
```

---

### GET /api/performance/status

Get monitoring system status.

#### Response
```json
{
  "status": "success",
  "monitoring_status": {
    "monitor_active": true,
    "cache_active": true,
    "monitored_codebases": 3,
    "collection_interval": 0.1,
    "last_update": "2025-08-18T13:15:30.123456"
  },
  "timestamp": "2025-08-18T13:15:30.123456"
}
```

---

## LLM API

LLM integration endpoints for AI-powered analysis and toggle functionality.

### GET /api/llm/status

Get current LLM status. Critical for LLM toggle button state.

#### Response
```json
{
  "status": "success",
  "llm_available": false,
  "api_enabled": false,
  "provider": "gemini",
  "demo_mode": true,
  "usage": {
    "calls_today": 0,
    "tokens_used": 0,
    "error_rate": 0.0
  },
  "message": "LLM monitor not available",
  "timestamp": "2025-08-18T13:15:30.123456"
}
```

When LLM monitor is available:
```json
{
  "status": "success",
  "llm_available": true,
  "api_enabled": true,
  "provider": "gemini",
  "demo_mode": false,
  "usage": {
    "calls_today": 42,
    "tokens_used": 15420,
    "last_call": "2025-08-18T12:45:20.123456",
    "error_rate": 0.02
  },
  "model_info": {
    "name": "gemini-2.5-pro",
    "version": "2.5",
    "max_tokens": 32000
  },
  "timestamp": "2025-08-18T13:15:30.123456"
}
```

---

### POST /api/llm/toggle-mode

Toggle LLM API mode on/off. Critical for LLM toggle button functionality.

#### Request Body
```json
{
  "enabled": true
}
```

#### Response
```json
{
  "status": "success",
  "enabled": true,
  "previous_state": false,
  "message": "LLM API enabled",
  "timestamp": "2025-08-18T13:15:30.123456"
}
```

#### Error Responses
- `400 VALIDATION_ERROR`: Missing enabled parameter

---

### GET /api/llm/metrics

Get LLM usage metrics and statistics.

#### Response
```json
{
  "status": "success",
  "metrics": {
    "total_calls": 156,
    "successful_calls": 153,
    "failed_calls": 3,
    "average_response_time": 2.3,
    "tokens_consumed": 45230,
    "cost_estimate": 0.0452
  },
  "api_enabled": true,
  "timestamp": "2025-08-18T13:15:30.123456"
}
```

---

### POST /api/llm/analyze

Perform LLM-based analysis on code modules.

#### Request Body
```json
{
  "module_path": "/path/to/module.py",
  "analysis_type": "code_review"
}
```

#### Response
```json
{
  "status": "success",
  "analysis": {
    "score": 85,
    "issues": [
      {
        "type": "complexity",
        "line": 45,
        "message": "Function complexity is high",
        "severity": "medium"
      }
    ],
    "suggestions": [
      "Consider breaking down large functions",
      "Add more comprehensive error handling"
    ],
    "strengths": [
      "Good documentation coverage",
      "Consistent naming conventions"
    ]
  },
  "module_path": "/path/to/module.py",
  "analysis_type": "code_review",
  "timestamp": "2025-08-18T13:15:30.123456"
}
```

#### Error Responses
- `400 VALIDATION_ERROR`: Missing module_path parameter
- `403 FORBIDDEN`: LLM API not enabled
- `503 SERVICE_UNAVAILABLE`: LLM monitor not available

---

### POST /api/llm/estimate-cost

Estimate cost for LLM operations.

#### Request Body
```json
{
  "operation": "analysis",
  "input_size": 5000
}
```

#### Response
```json
{
  "status": "success",
  "estimate": {
    "operation": "analysis",
    "input_size": 5000,
    "estimated_tokens": 1250,
    "estimated_cost": 0.000025,
    "currency": "USD"
  },
  "timestamp": "2025-08-18T13:15:30.123456"
}
```

---

## Analytics API

Data analytics and insights endpoints.

### GET /api/analytics/metrics

Get analytics metrics and insights.

#### Response
```json
{
  "status": "success",
  "metrics": {
    "total_modules": 156,
    "test_coverage": 78.5,
    "code_quality_score": 85.2,
    "performance_trends": {
      "cpu_trend": "stable",
      "memory_trend": "increasing",
      "network_trend": "decreasing"
    }
  },
  "timestamp": "2025-08-18T13:15:30.123456"
}
```

---

## Workflow API

Workflow management and orchestration endpoints.

### GET /api/workflow/status

Get workflow system status.

#### Response
```json
{
  "status": "success",
  "workflow_status": {
    "active_workflows": 3,
    "queued_tasks": 12,
    "completed_today": 45,
    "failed_today": 2,
    "average_completion_time": 2.5
  },
  "timestamp": "2025-08-18T13:15:30.123456"
}
```

---

## Tests API

Test management and execution endpoints.

### GET /api/tests/summary

Get test execution summary.

#### Response
```json
{
  "status": "success",
  "test_summary": {
    "total_tests": 1250,
    "passed": 1205,
    "failed": 30,
    "skipped": 15,
    "pass_rate": 96.4,
    "last_run": "2025-08-18T12:30:00.123456"
  },
  "timestamp": "2025-08-18T13:15:30.123456"
}
```

---

## Refactor API

Code refactoring and analysis endpoints.

### GET /api/refactor/recommendations

Get refactoring recommendations.

#### Response
```json
{
  "status": "success",
  "recommendations": [
    {
      "module": "/path/to/module.py",
      "type": "extract_method",
      "priority": "high",
      "description": "Extract complex method into smaller functions",
      "estimated_effort": "2 hours"
    }
  ],
  "total_recommendations": 15,
  "timestamp": "2025-08-18T13:15:30.123456"
}
```

---

## Rate Limiting

Currently no rate limiting is implemented. Future versions may implement:
- 1000 requests per hour per IP for general endpoints
- 100 requests per hour for LLM endpoints
- 10000 requests per hour for performance/realtime endpoint

## SDK and Client Libraries

No official SDKs are currently available. The API follows REST conventions and can be consumed by any HTTP client.

### Example Usage (JavaScript)

```javascript
// Get real-time performance metrics
const response = await fetch('/api/performance/realtime?codebase=/testmaster');
const data = await response.json();

if (data.status === 'success') {
  console.log('CPU Usage:', data.timeseries.cpu_usage);
  console.log('Memory Usage:', data.timeseries.memory_usage_mb);
  console.log('Network Activity:', data.timeseries.network_kb_s);
}

// Toggle LLM mode
const toggleResponse = await fetch('/api/llm/toggle-mode', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ enabled: true })
});
const toggleData = await toggleResponse.json();
console.log('LLM Mode:', toggleData.enabled ? 'Enabled' : 'Disabled');
```

### Example Usage (Python)

```python
import requests

# Get performance summary
response = requests.get('http://localhost:5000/api/performance/summary')
data = response.json()

if data['status'] == 'success':
    summary = data['summary']
    print(f"Average CPU: {summary['avg_cpu']}%")
    print(f"Peak Memory: {summary['peak_memory_mb']} MB")

# Perform LLM analysis
analysis_request = {
    'module_path': '/path/to/module.py',
    'analysis_type': 'code_review'
}
response = requests.post('http://localhost:5000/api/llm/analyze', 
                        json=analysis_request)
analysis = response.json()

if analysis['status'] == 'success':
    print(f"Code Quality Score: {analysis['analysis']['score']}")
    print(f"Issues Found: {len(analysis['analysis']['issues'])}")
```

## Changelog

### Version 2.0.0 (Current)
- Enhanced error handling with centralized error management
- Real-time performance monitoring with 100ms precision
- LLM integration with toggle functionality
- Production-ready Gunicorn deployment
- Comprehensive API documentation
- Enhanced frontend with 6-tab navigation
- Mobile responsive design
- Cross-browser compatibility

### Future Versions
- Authentication and authorization
- Rate limiting
- WebSocket support for real-time updates
- Bulk operations
- API versioning
- Official SDKs

## Support

For issues and questions:
- Check the error response details
- Review server logs at `/var/log/testmaster/dashboard.log`
- Verify monitoring system status via `/api/performance/status`
- Test connectivity with `/api/health`