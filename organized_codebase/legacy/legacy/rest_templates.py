"""
REST API Documentation Templates
"""

from typing import Dict
from ..template_engine import Template, TemplateMetadata, TemplateType, TemplateFormat
from .base import ApiContext, ApiEndpoint, ApiModel, ApiType, AuthenticationType


class RestTemplateProvider:
    """Provider for REST API documentation templates."""
    
    def __init__(self):
        self.templates: Dict[str, Template] = {}
        self._create_rest_templates()
    
    def _create_rest_templates(self):
        """Create REST API documentation templates."""
        
        # Comprehensive REST API template
        rest_comprehensive = Template(
            metadata=TemplateMetadata(
                name="rest_comprehensive",
                description="Comprehensive REST API documentation template",
                template_type=TemplateType.API_DOC,
                format=TemplateFormat.MARKDOWN,
                author="TestMaster",
                version="1.0.0",
                tags=["rest", "api", "comprehensive"],
                required_variables=["api_name", "description", "base_url", "version"],
                optional_variables=["endpoints", "authentication", "rate_limits", "models"],
                target_audience="all"
            ),
            content='''# {{api_name}} API Documentation

{{description}}

**Version:** {{version}}  
**Base URL:** `{{base_url}}`

## Table of Contents

- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Endpoints](#endpoints)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [SDKs](#sdks)

## Authentication

{{#authentication}}
{{#authentication_api_key}}
### API Key Authentication

Include your API key in the request header:

```http
Authorization: Bearer YOUR_API_KEY
```

To obtain an API key:
1. Sign up for an account at [{{base_url}}/signup]({{base_url}}/signup)
2. Navigate to your dashboard
3. Generate a new API key in the API section
{{/authentication_api_key}}

{{#authentication_oauth2}}
### OAuth 2.0 Authentication

This API uses OAuth 2.0 for authentication. Follow these steps:

1. **Authorization Request**
   ```
   GET {{base_url}}/oauth/authorize?response_type=code&client_id=YOUR_CLIENT_ID&redirect_uri=YOUR_REDIRECT_URI&scope=SCOPES
   ```

2. **Token Exchange**
   ```http
   POST {{base_url}}/oauth/token
   Content-Type: application/x-www-form-urlencoded

   grant_type=authorization_code&code=AUTHORIZATION_CODE&client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET&redirect_uri=YOUR_REDIRECT_URI
   ```

3. **Using Access Token**
   ```http
   Authorization: Bearer ACCESS_TOKEN
   ```
{{/authentication_oauth2}}

{{#authentication_jwt}}
### JWT Authentication

Include your JWT token in the Authorization header:

```http
Authorization: Bearer JWT_TOKEN
```

JWT tokens are obtained through the login endpoint and are valid for 24 hours.
{{/authentication_jwt}}
{{/authentication}}

## Rate Limiting

{{#rate_limits}}
{{rate_limits}}
{{/rate_limits}}
{{^rate_limits}}
API requests are rate limited to ensure fair usage:

- **Free tier**: 1,000 requests per hour
- **Pro tier**: 10,000 requests per hour
- **Enterprise tier**: Custom limits

Rate limit information is included in response headers:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1609459200
```

When rate limits are exceeded, you'll receive a `429 Too Many Requests` response.
{{/rate_limits}}

## Endpoints

{{#endpoints}}
{{#endpoints}}
### {{method}} {{path}}

{{description}}

{{#deprecated}}
> ⚠️ **Deprecated**: This endpoint is deprecated and will be removed in a future version.
{{/deprecated}}

**Parameters:**

{{#parameters}}
| Name | Type | Location | Required | Description |
|------|------|----------|----------|-------------|
{{#parameters}}
| `{{name}}` | {{type}} | {{location}} | {{required}} | {{description}} |
{{/parameters}}
{{/parameters}}

{{#request_body}}
**Request Body:**

```json
{{request_body}}
```
{{/request_body}}

**Example Request:**

```bash
curl -X {{method}} "{{base_url}}{{path}}" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json"
{{#request_body}}
  -d '{{request_body}}'
{{/request_body}}
```

**Response:**

{{#response_example}}
```json
{{response_example}}
```
{{/response_example}}

**Status Codes:**

{{#status_codes}}
| Code | Description |
|------|-------------|
{{#status_codes}}
| `{{code}}` | {{description}} |
{{/status_codes}}
{{/status_codes}}

{{#tags}}
**Tags:** {{#tags}}`{{.}}`{{/tags}}
{{/tags}}

---

{{/endpoints}}
{{/endpoints}}

## Data Models

{{#models}}
{{#models}}
### {{name}}

{{description}}

**Properties:**

| Property | Type | Required | Description |
|----------|------|----------|-------------|
{{#properties}}
| `{{name}}` | {{type}} | {{required}} | {{description}} |
{{/properties}}

{{#example}}
**Example:**

```json
{{example}}
```
{{/example}}

{{/models}}
{{/models}}

## Error Handling

All errors follow a consistent format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": "Additional error details",
    "timestamp": "2023-01-01T12:00:00Z"
  }
}
```

### Common Error Codes

{{#error_codes}}
| Code | HTTP Status | Description |
|------|-------------|-------------|
{{#error_codes}}
| `{{code}}` | {{status}} | {{description}} |
{{/error_codes}}
{{/error_codes}}

{{^error_codes}}
| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_API_KEY` | 401 | The provided API key is invalid |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `NOT_FOUND` | 404 | Resource not found |
| `INTERNAL_ERROR` | 500 | Internal server error |
{{/error_codes}}

## Examples

{{#examples}}
{{#examples}}
### {{title}}

{{description}}

```{{language}}
{{code}}
```

{{#response}}
**Response:**
```json
{{response}}
```
{{/response}}

{{/examples}}
{{/examples}}

{{^examples}}
### Basic Usage Example

```python
import requests

# Set up authentication
headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
}

# Make a request
response = requests.get('{{base_url}}/endpoint', headers=headers)
data = response.json()

print(data)
```

### JavaScript Example

```javascript
const apiKey = 'YOUR_API_KEY';
const baseURL = '{{base_url}}';

async function fetchData() {
    const response = await fetch(`${baseURL}/endpoint`, {
        headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        }
    });
    
    const data = await response.json();
    return data;
}
```
{{/examples}}

## SDKs and Libraries

{{#sdks}}
{{#sdks}}
### {{language}}

**Installation:**
```{{language_code}}
{{install_command}}
```

**Usage:**
```{{language_code}}
{{usage_example}}
```

{{/sdks}}
{{/sdks}}

{{^sdks}}
### Python

**Installation:**
```bash
pip install {{api_name}}-python
```

**Usage:**
```python
from {{api_name}} import Client

client = Client(api_key = os.getenv('KEY'))
result = client.get_data()
```

### JavaScript/Node.js

**Installation:**
```bash
npm install {{api_name}}-js
```

**Usage:**
```javascript
const {{ApiName}} = require('{{api_name}}-js');

const client = new {{ApiName}}('YOUR_API_KEY');
const result = await client.getData();
```
{{/sdks}}

## Testing

### Postman Collection

Import our Postman collection for easy testing:

[Download Postman Collection]({{api_name}}.postman_collection.json)

### OpenAPI Specification

View the OpenAPI specification:

[OpenAPI Spec]({{base_url}}/openapi.json)

## Support

{{#contact_info}}
- **Support Email**: {{email}}
{{#phone}}
- **Phone**: {{phone}}
{{/phone}}
{{#website}}
- **Website**: [{{website}}]({{website}})
{{/website}}
{{/contact_info}}

{{^contact_info}}
- **Documentation**: [API Docs]({{base_url}}/docs)
- **Status Page**: [Status]({{base_url}}/status)
- **Support**: support@example.com
{{/contact_info}}

## License

{{#license_info}}
{{license_info}}
{{/license_info}}
{{^license_info}}
This API documentation is licensed under the MIT License.
{{/license_info}}

---

*Generated on {{current_date}} | Version {{version}}*
''',
            examples=[
                '''# Weather API Documentation

Get current weather data and forecasts for any location worldwide.

**Version:** v1.0  
**Base URL:** `https://api.weather.example.com/v1`

## Authentication

### API Key Authentication

Include your API key in the request header:

```http
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### GET /weather/current

Get current weather for a location.

**Parameters:**

| Name | Type | Location | Required | Description |
|------|------|----------|----------|-------------|
| `lat` | number | query | Yes | Latitude |
| `lon` | number | query | Yes | Longitude |

**Response:**

```json
{
  "temperature": 22.5,
  "humidity": 65,
  "description": "Partly cloudy"
}
```
'''
            ]
        )
        
        self.templates["rest_comprehensive"] = rest_comprehensive
        
        # OpenAPI template
        openapi_template = Template(
            metadata=TemplateMetadata(
                name="rest_openapi",
                description="OpenAPI/Swagger specification template",
                template_type=TemplateType.API_DOC,
                format=TemplateFormat.SIMPLE,
                author="TestMaster",
                version="1.0.0",
                tags=["rest", "openapi", "swagger"],
                required_variables=["api_name", "description", "version", "base_url"],
                optional_variables=["endpoints", "models"],
                target_audience="all"
            ),
            content='''openapi: 3.0.0
info:
  title: {{api_name}}
  description: {{description}}
  version: "{{version}}"
  {{#contact_info}}
  contact:
    name: {{name}}
    email: {{email}}
    {{#url}}
    url: {{url}}
    {{/url}}
  {{/contact_info}}
  {{#license_info}}
  license:
    name: {{name}}
    url: {{url}}
  {{/license_info}}

servers:
  - url: {{base_url}}
    description: Production server

paths:
{{#endpoints}}
  {{path}}:
    {{method}}:
      summary: {{description}}
      {{#tags}}
      tags:
        {{#tags}}
        - {{.}}
        {{/tags}}
      {{/tags}}
      {{#parameters}}
      parameters:
        {{#parameters}}
        - name: {{name}}
          in: {{location}}
          required: {{required}}
          schema:
            type: {{type}}
          description: {{description}}
        {{/parameters}}
      {{/parameters}}
      {{#request_body}}
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/{{request_model}}'
      {{/request_body}}
      responses:
        {{#status_codes}}
        '{{code}}':
          description: {{description}}
          {{#response_model}}
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/{{response_model}}'
          {{/response_model}}
        {{/status_codes}}
{{/endpoints}}

components:
  securitySchemes:
    {{#authentication_api_key}}
    ApiKeyAuth:
      type: apiKey
      in: header
      name: Authorization
    {{/authentication_api_key}}
    {{#authentication_bearer}}
    BearerAuth:
      type: http
      scheme: bearer
    {{/authentication_bearer}}
    {{#authentication_oauth2}}
    OAuth2:
      type: oauth2
      flows:
        authorizationCode:
          authorizationUrl: {{base_url}}/oauth/authorize
          tokenUrl: {{base_url}}/oauth/token
          scopes:
            {{#scopes}}
            {{name}}: {{description}}
            {{/scopes}}
    {{/authentication_oauth2}}

  schemas:
{{#models}}
    {{name}}:
      type: object
      description: {{description}}
      {{#required_fields}}
      required:
        {{#required_fields}}
        - {{.}}
        {{/required_fields}}
      {{/required_fields}}
      properties:
        {{#properties}}
        {{name}}:
          type: {{type}}
          description: {{description}}
        {{/properties}}
{{/models}}

security:
  {{#authentication_api_key}}
  - ApiKeyAuth: []
  {{/authentication_api_key}}
  {{#authentication_bearer}}
  - BearerAuth: []
  {{/authentication_bearer}}
  {{#authentication_oauth2}}
  - OAuth2: [{{#scopes}}{{name}}{{/scopes}}]
  {{/authentication_oauth2}}
''',
            examples=[
                '''openapi: 3.0.0
info:
  title: Weather API
  description: Weather data and forecasts
  version: "1.0"
  
servers:
  - url: https://api.weather.example.com/v1
  
paths:
  /weather/current:
    get:
      summary: Get current weather
      parameters:
        - name: lat
          in: query
          required: true
          schema:
            type: number
      responses:
        '200':
          description: Current weather data
'''
            ]
        )
        
        self.templates["rest_openapi"] = openapi_template
    

    def get_templates(self) -> Dict[str, Template]:
        """Get all REST templates."""
        return self.templates
    
    def get_template(self, name: str) -> Template:
        """Get a specific REST template by name."""
        return self.templates.get(name)
