"""
GRPC API Documentation Templates
"""

from typing import Dict
from ..template_engine import Template, TemplateMetadata, TemplateType, TemplateFormat
from .base import ApiContext, ApiEndpoint, ApiModel, ApiType, AuthenticationType


class GrpcTemplateProvider:
    """Provider for GRPC API documentation templates."""
    
    def __init__(self):
        self.templates: Dict[str, Template] = {}
        self._create_grpc_templates()
    
    def _create_grpc_templates(self):
        """Create gRPC API documentation templates."""
        
        grpc_template = Template(
            metadata=TemplateMetadata(
                name="grpc_comprehensive",
                description="Comprehensive gRPC API documentation template",
                template_type=TemplateType.API_DOC,
                format=TemplateFormat.MARKDOWN,
                author="TestMaster",
                version="1.0.0",
                tags=["grpc", "api", "protobuf"],
                required_variables=["api_name", "description", "server_address"],
                optional_variables=["services", "messages", "authentication"],
                target_audience="all"
            ),
            content='''# {{api_name}} gRPC API

{{description}}

**Server Address:** `{{server_address}}`  
**Protocol:** gRPC over HTTP/2

## Table of Contents

- [Getting Started](#getting-started)
- [Authentication](#authentication)
- [Services](#services)
- [Messages](#messages)
- [Error Handling](#error-handling)
- [Client Examples](#client-examples)

## Getting Started

### Protocol Buffer Definition

```protobuf
syntax = "proto3";

package {{package_name}};

{{#services}}
service {{name}} {
  {{#methods}}
  rpc {{name}}({{request_type}}) returns ({{response_type}});
  {{/methods}}
}
{{/services}}

{{#messages}}
message {{name}} {
  {{#fields}}
  {{type}} {{name}} = {{number}};
  {{/fields}}
}
{{/messages}}
```

### Generating Client Code

#### Go

```bash
protoc --go_out=. --go-grpc_out=. api.proto
```

#### Python

```bash
python -m grpc_tools.protoc --python_out=. --grpc_python_out=. api.proto
```

#### Java

```bash
protoc --java_out=. --grpc-java_out=. api.proto
```

## Authentication

{{#authentication}}
{{authentication}}
{{/authentication}}
{{^authentication}}
gRPC uses metadata for authentication. Include your token in the metadata:

```
authorization: Bearer YOUR_TOKEN
```
{{/authentication}}

## Services

{{#services}}
{{#services}}
### {{name}}

{{description}}

{{#methods}}
#### {{name}}

{{description}}

**Request Type:** `{{request_type}}`  
**Response Type:** `{{response_type}}`  
**Stream Type:** {{stream_type}}

{{#request_example}}
**Request Example:**

```json
{{request_example}}
```
{{/request_example}}

{{#response_example}}
**Response Example:**

```json
{{response_example}}
```
{{/response_example}}

---

{{/methods}}

{{/services}}
{{/services}}

## Messages

{{#messages}}
{{#messages}}
### {{name}}

{{description}}

```protobuf
message {{name}} {
  {{#fields}}
  {{type}} {{name}} = {{number}}; // {{description}}
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

{{/messages}}
{{/messages}}

## Error Handling

gRPC uses status codes for error handling:

| Code | Name | Description |
|------|------|-------------|
| 0 | OK | Success |
| 1 | CANCELLED | Operation was cancelled |
| 2 | UNKNOWN | Unknown error |
| 3 | INVALID_ARGUMENT | Invalid argument |
| 4 | DEADLINE_EXCEEDED | Request timeout |
| 5 | NOT_FOUND | Resource not found |
| 6 | ALREADY_EXISTS | Resource already exists |
| 7 | PERMISSION_DENIED | Permission denied |
| 8 | RESOURCE_EXHAUSTED | Rate limited |
| 13 | INTERNAL | Internal server error |
| 14 | UNAVAILABLE | Service unavailable |
| 16 | UNAUTHENTICATED | Authentication required |

## Client Examples

### Go

```go
package main

import (
    "context"
    "log"
    
    "google.golang.org/grpc"
    "google.golang.org/grpc/metadata"
    pb "path/to/generated/proto"
)

func main() {
    conn, err := grpc.Dial("{{server_address}}", grpc.WithInsecure())
    if err != nil {
        log.Fatalf("Failed to connect: %v", err)
    }
    defer conn.Close()
    
    client := pb.New{{ServiceName}}Client(conn)
    
    // Add authentication metadata
    ctx := metadata.AppendToOutgoingContext(context.Background(), 
        "authorization", "Bearer YOUR_TOKEN")
    
    // Make request
    response, err := client.{{MethodName}}(ctx, &pb.{{RequestType}}{
        // Request fields
    })
    if err != nil {
        log.Fatalf("Request failed: %v", err)
    }
    
    log.Printf("Response: %v", response)
}
```

### Python

```python
import grpc
import api_pb2
import api_pb2_grpc

# Create channel
channel = grpc.insecure_channel('{{server_address}}')
stub = api_pb2_grpc.{{ServiceName}}Stub(channel)

# Create metadata
metadata = [('authorization', 'Bearer YOUR_TOKEN')]

# Make request
request = api_pb2.{{RequestType}}(
    # Request fields
)

try:
    response = stub.{{MethodName}}(request, metadata=metadata)
    print(f"Response: {response}")
except grpc.RpcError as e:
    print(f"Error: {e.code()} - {e.details()}")
```

### JavaScript (Node.js)

```javascript
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');

// Load proto file
const packageDefinition = protoLoader.loadSync('api.proto');
const proto = grpc.loadPackageDefinition(packageDefinition);

// Create client
const client = new proto.{{ServiceName}}('{{server_address}}', 
    grpc.credentials.createInsecure());

// Create metadata
const metadata = new grpc.Metadata();
metadata.add('authorization', 'Bearer YOUR_TOKEN');

// Make request
client.{{methodName}}({
    // Request fields
}, metadata, (error, response) => {
    if (error) {
        console.error('Error:', error);
    } else {
        console.log('Response:', response);
    }
});
```

## Streaming

### Server Streaming

```go
stream, err := client.{{StreamingMethod}}(ctx, &pb.{{RequestType}}{})
if err != nil {
    log.Fatalf("Error calling streaming method: %v", err)
}

for {
    response, err := stream.Recv()
    if err == io.EOF {
        break
    }
    if err != nil {
        log.Fatalf("Error receiving stream: %v", err)
    }
    
    log.Printf("Received: %v", response)
}
```

### Client Streaming

```go
stream, err := client.{{ClientStreamingMethod}}(ctx)
if err != nil {
    log.Fatalf("Error creating stream: %v", err)
}

// Send multiple requests
for i := 0; i < 10; i++ {
    if err := stream.Send(&pb.{{RequestType}}{}); err != nil {
        log.Fatalf("Error sending: %v", err)
    }
}

response, err := stream.CloseAndRecv()
if err != nil {
    log.Fatalf("Error receiving response: %v", err)
}

log.Printf("Response: %v", response)
```

## Tools and Testing

### grpcurl

Test the API using grpcurl:

```bash
# List services
grpcurl -plaintext {{server_address}} list

# Describe service
grpcurl -plaintext {{server_address}} describe {{ServiceName}}

# Make request
grpcurl -plaintext -d '{"field": "value"}' \\
  {{server_address}} {{ServiceName}}/{{MethodName}}
```

### BloomRPC

Use BloomRPC GUI client for interactive testing: [https://github.com/uw-labs/bloomrpc](https://github.com/uw-labs/bloomrpc)

## Performance Considerations

- **Connection Reuse**: Keep connections alive for multiple requests
- **Compression**: Enable gzip compression for large payloads
- **Streaming**: Use streaming for large datasets or real-time updates
- **Load Balancing**: Implement client-side load balancing for multiple servers

---

*Generated on {{current_date}} | Version {{version}}*
''',
            examples=[
                '''# User Service gRPC API

gRPC service for user management operations.

**Server Address:** `localhost:50051`

## Services

### UserService

User management operations.

#### GetUser

Get user by ID.

**Request Type:** `GetUserRequest`  
**Response Type:** `User`

```protobuf
service UserService {
  rpc GetUser(GetUserRequest) returns (User);
}

message GetUserRequest {
  string user_id = 1;
}

message User {
  string id = 1;
  string name = 2;
  string email = 3;
}
```
'''
            ]
        )
        
        self.templates["grpc_comprehensive"] = grpc_template
    

    def get_templates(self) -> Dict[str, Template]:
        """Get all GRPC templates."""
        return self.templates
    
    def get_template(self, name: str) -> Template:
        """Get a specific GRPC template by name."""
        return self.templates.get(name)
