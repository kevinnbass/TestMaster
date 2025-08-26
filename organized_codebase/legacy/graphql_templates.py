"""
GRAPHQL API Documentation Templates
"""

from typing import Dict
from ..template_engine import Template, TemplateMetadata, TemplateType, TemplateFormat
from .base import ApiContext, ApiEndpoint, ApiModel, ApiType, AuthenticationType


class GraphqlTemplateProvider:
    """Provider for GRAPHQL API documentation templates."""
    
    def __init__(self):
        self.templates: Dict[str, Template] = {}
        self._create_graphql_templates()
    
    def _create_graphql_templates(self):
        """Create GraphQL API documentation templates."""
        
        graphql_template = Template(
            metadata=TemplateMetadata(
                name="graphql_comprehensive",
                description="Comprehensive GraphQL API documentation template",
                template_type=TemplateType.API_DOC,
                format=TemplateFormat.MARKDOWN,
                author="TestMaster",
                version="1.0.0",
                tags=["graphql", "api"],
                required_variables=["api_name", "description", "endpoint"],
                optional_variables=["queries", "mutations", "subscriptions", "types"],
                target_audience="all"
            ),
            content='''# {{api_name}} GraphQL API

{{description}}

**Endpoint:** `{{endpoint}}`

## Table of Contents

- [Getting Started](#getting-started)
- [Authentication](#authentication)
- [Schema Overview](#schema-overview)
- [Queries](#queries)
- [Mutations](#mutations)
- [Subscriptions](#subscriptions)
- [Types](#types)
- [Examples](#examples)

## Getting Started

### GraphQL Playground

Explore the API interactively at: [{{endpoint}}/playground]({{endpoint}}/playground)

### Schema Introspection

```graphql
query IntrospectionQuery {
  __schema {
    queryType { name }
    mutationType { name }
    subscriptionType { name }
  }
}
```

## Authentication

{{#authentication}}
{{authentication}}
{{/authentication}}
{{^authentication}}
Include your authentication token in the request headers:

```http
Authorization: Bearer YOUR_TOKEN
```
{{/authentication}}

## Schema Overview

```graphql
type Query {
  {{#queries}}
  {{name}}{{#parameters}}({{#parameters}}{{name}}: {{type}}{{/parameters}}){{/parameters}}: {{return_type}}
  {{/queries}}
}

{{#mutations}}
type Mutation {
  {{#mutations}}
  {{name}}{{#parameters}}({{#parameters}}{{name}}: {{type}}{{/parameters}}){{/parameters}}: {{return_type}}
  {{/mutations}}
}
{{/mutations}}

{{#subscriptions}}
type Subscription {
  {{#subscriptions}}
  {{name}}{{#parameters}}({{#parameters}}{{name}}: {{type}}{{/parameters}}){{/parameters}}: {{return_type}}
  {{/subscriptions}}
}
{{/subscriptions}}
```

## Queries

{{#queries}}
{{#queries}}
### {{name}}

{{description}}

**Type:** `{{return_type}}`

{{#parameters}}
**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
{{#parameters}}
| `{{name}}` | {{type}} | {{required}} | {{description}} |
{{/parameters}}
{{/parameters}}

**Example:**

```graphql
query {
  {{name}}{{#parameters}}({{#parameters}}{{name}}: {{example_value}}{{/parameters}}){{/parameters}} {
    {{#fields}}
    {{.}}
    {{/fields}}
  }
}
```

**Response:**

```json
{{response_example}}
```

---

{{/queries}}
{{/queries}}

## Mutations

{{#mutations}}
{{#mutations}}
### {{name}}

{{description}}

**Type:** `{{return_type}}`

{{#parameters}}
**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
{{#parameters}}
| `{{name}}` | {{type}} | {{required}} | {{description}} |
{{/parameters}}
{{/parameters}}

**Example:**

```graphql
mutation {
  {{name}}{{#parameters}}({{#parameters}}{{name}}: {{example_value}}{{/parameters}}){{/parameters}} {
    {{#fields}}
    {{.}}
    {{/fields}}
  }
}
```

**Response:**

```json
{{response_example}}
```

---

{{/mutations}}
{{/mutations}}

## Subscriptions

{{#subscriptions}}
{{#subscriptions}}
### {{name}}

{{description}}

**Type:** `{{return_type}}`

{{#parameters}}
**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
{{#parameters}}
| `{{name}}` | {{type}} | {{required}} | {{description}} |
{{/parameters}}
{{/parameters}}

**Example:**

```graphql
subscription {
  {{name}}{{#parameters}}({{#parameters}}{{name}}: {{example_value}}{{/parameters}}){{/parameters}} {
    {{#fields}}
    {{.}}
    {{/fields}}
  }
}
```

---

{{/subscriptions}}
{{/subscriptions}}

## Types

{{#types}}
{{#types}}
### {{name}}

{{description}}

```graphql
type {{name}} {
  {{#fields}}
  {{name}}: {{type}}{{#description}} # {{description}}{{/description}}
  {{/fields}}
}
```

{{#example}}
**Example:**

```json
{{example}}
```
{{/example}}

---

{{/types}}
{{/types}}

## Examples

### JavaScript (Apollo Client)

```javascript
import { ApolloClient, InMemoryCache, gql } from '@apollo/client';

const client = new ApolloClient({
  uri: '{{endpoint}}',
  cache: new InMemoryCache(),
  headers: {
    authorization: 'Bearer YOUR_TOKEN'
  }
});

const GET_DATA = gql`
  query GetData($id: ID!) {
    data(id: $id) {
      id
      name
      description
    }
  }
`;

const { data } = await client.query({
  query: GET_DATA,
  variables: { id: '123' }
});
```

### Python (gql)

```python
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

transport = RequestsHTTPTransport(
    url='{{endpoint}}',
    headers={'Authorization': 'Bearer YOUR_TOKEN'}
)

client = Client(transport=transport, fetch_schema_from_transport=True)

query = gql('''
    query GetData($id: ID!) {
        data(id: $id) {
            id
            name
            description
        }
    }
''')

result = client.execute(query, variable_values={'id': '123'})
```

### cURL

```bash
curl -X POST {{endpoint}} \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer YOUR_TOKEN" \\
  -d '{
    "query": "query GetData($id: ID!) { data(id: $id) { id name description } }",
    "variables": { "id": "123" }
  }'
```

## Error Handling

GraphQL errors follow this format:

```json
{
  "errors": [
    {
      "message": "Error message",
      "locations": [
        {
          "line": 2,
          "column": 3
        }
      ],
      "path": ["field", "subfield"],
      "extensions": {
        "code": "ERROR_CODE",
        "exception": {
          "stacktrace": ["Error details..."]
        }
      }
    }
  ],
  "data": null
}
```

## Rate Limiting

GraphQL queries are analyzed for complexity before execution. Queries exceeding the complexity limit will be rejected.

- **Maximum complexity**: 1000 points
- **Maximum depth**: 10 levels
- **Rate limit**: 1000 requests per hour

## Best Practices

1. **Request only needed fields** - GraphQL allows you to specify exactly which fields you need
2. **Use fragments** - Reuse common field selections
3. **Implement proper error handling** - Check both `errors` and `data` fields
4. **Cache responses** - Use tools like Apollo Client for intelligent caching
5. **Avoid over-fetching** - Don't request more data than you need

---

*Generated on {{current_date}} | Version {{version}}*
''',
            examples=[
                '''# Blog GraphQL API

A GraphQL API for managing blog posts and users.

**Endpoint:** `https://api.blog.example.com/graphql`

## Schema Overview

```graphql
type Query {
  posts(limit: Int, offset: Int): [Post!]!
  post(id: ID!): Post
}

type Mutation {
  createPost(input: PostInput!): Post!
}
```

## Queries

### posts

Get a list of blog posts.

**Example:**

```graphql
query {
  posts(limit: 10) {
    id
    title
    content
    author {
      name
    }
  }
}
```
'''
            ]
        )
        
        self.templates["graphql_comprehensive"] = graphql_template
    

    def get_templates(self) -> Dict[str, Template]:
        """Get all GRAPHQL templates."""
        return self.templates
    
    def get_template(self, name: str) -> Template:
        """Get a specific GRAPHQL template by name."""
        return self.templates.get(name)
