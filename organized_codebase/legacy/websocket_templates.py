"""
WEBSOCKET API Documentation Templates
"""

from typing import Dict
from ..template_engine import Template, TemplateMetadata, TemplateType, TemplateFormat
from .base import ApiContext, ApiEndpoint, ApiModel, ApiType, AuthenticationType


class WebsocketTemplateProvider:
    """Provider for WEBSOCKET API documentation templates."""
    
    def __init__(self):
        self.templates: Dict[str, Template] = {}
        self._create_websocket_templates()
    
    def _create_websocket_templates(self):
        """Create WebSocket API documentation templates."""
        
        websocket_template = Template(
            metadata=TemplateMetadata(
                name="websocket_comprehensive",
                description="Comprehensive WebSocket API documentation template",
                template_type=TemplateType.API_DOC,
                format=TemplateFormat.MARKDOWN,
                author="TestMaster",
                version="1.0.0",
                tags=["websocket", "realtime", "api"],
                required_variables=["api_name", "description", "websocket_url"],
                optional_variables=["events", "authentication", "rate_limits"],
                target_audience="all"
            ),
            content='''# {{api_name}} WebSocket API

{{description}}

**WebSocket URL:** `{{websocket_url}}`

## Table of Contents

- [Connection](#connection)
- [Authentication](#authentication)
- [Message Format](#message-format)
- [Events](#events)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [Rate Limiting](#rate-limiting)

## Connection

### Establishing Connection

```javascript
const ws = new WebSocket('{{websocket_url}}');

ws.onopen = function(event) {
    console.log('Connected to WebSocket');
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
};

ws.onclose = function(event) {
    console.log('Connection closed:', event.code, event.reason);
};

ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};
```

### Connection Parameters

{{#connection_params}}
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
{{#connection_params}}
| `{{name}}` | {{type}} | {{required}} | {{description}} |
{{/connection_params}}
{{/connection_params}}

## Authentication

{{#authentication}}
{{authentication}}
{{/authentication}}
{{^authentication}}
Authentication is performed after establishing the WebSocket connection by sending an authentication message:

```json
{
  "type": "auth",
  "token": "YOUR_API_TOKEN"
}
```

Authentication response:

```json
{
  "type": "auth_response",
  "success": true,
  "user_id": "12345"
}
```
{{/authentication}}

## Message Format

All WebSocket messages follow this JSON format:

```json
{
  "type": "message_type",
  "data": {
    // Message-specific data
  },
  "timestamp": "2023-01-01T12:00:00Z",
  "id": "unique_message_id"
}
```

### Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| `auth` | Client → Server | Authentication request |
| `auth_response` | Server → Client | Authentication response |
| `subscribe` | Client → Server | Subscribe to events |
| `unsubscribe` | Client → Server | Unsubscribe from events |
| `error` | Server → Client | Error notification |

## Events

{{#events}}
{{#events}}
### {{name}}

{{description}}

**Direction:** {{direction}}

{{#subscription_required}}
**Subscription Required:** Yes

To receive this event, subscribe first:

```json
{
  "type": "subscribe",
  "event": "{{name}}"
}
```
{{/subscription_required}}

**Message Format:**

```json
{
  "type": "{{name}}",
  "data": {{data_example}}
}
```

{{#trigger_conditions}}
**Triggered By:**
{{#trigger_conditions}}
- {{.}}
{{/trigger_conditions}}
{{/trigger_conditions}}

---

{{/events}}
{{/events}}

## Error Handling

Error messages follow this format:

```json
{
  "type": "error",
  "data": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": "Additional error details"
  }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `AUTH_REQUIRED` | Authentication required |
| `INVALID_TOKEN` | Invalid authentication token |
| `RATE_LIMIT_EXCEEDED` | Too many messages |
| `INVALID_MESSAGE` | Message format invalid |
| `SUBSCRIPTION_FAILED` | Failed to subscribe to event |

## Examples

### JavaScript Client

```javascript
class {{ApiName}}Client {
    constructor(url, token) {
        this.url = url;
        this.token = token;
        this.ws = null;
        this.authenticated = false;
    }
    
    connect() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.url);
            
            this.ws.onopen = () => {
                this.authenticate().then(resolve).catch(reject);
            };
            
            this.ws.onmessage = (event) => {
                this.handleMessage(JSON.parse(event.data));
            };
            
            this.ws.onclose = (event) => {
                console.log('Connection closed:', event.code);
                this.authenticated = false;
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };
        });
    }
    
    authenticate() {
        return new Promise((resolve, reject) => {
            const authMessage = {
                type: 'auth',
                token: this.token
            };
            
            this.ws.send(JSON.stringify(authMessage));
            
            const authHandler = (message) => {
                if (message.type === 'auth_response') {
                    if (message.success) {
                        this.authenticated = true;
                        resolve();
                    } else {
                        reject(new Error('Authentication failed'));
                    }
                }
            };
            
            this.ws.addEventListener('message', authHandler, { once: true });
        });
    }
    
    subscribe(eventType) {
        if (!this.authenticated) {
            throw new Error('Not authenticated');
        }
        
        const subscribeMessage = {
            type: 'subscribe',
            event: eventType
        };
        
        this.ws.send(JSON.stringify(subscribeMessage));
    }
    
    sendMessage(type, data) {
        if (!this.authenticated) {
            throw new Error('Not authenticated');
        }
        
        const message = {
            type: type,
            data: data,
            timestamp: new Date().toISOString(),
            id: this.generateId()
        };
        
        this.ws.send(JSON.stringify(message));
    }
    
    handleMessage(message) {
        switch (message.type) {
            case 'error':
                console.error('Server error:', message.data);
                break;
            case 'user_joined':
                console.log('User joined:', message.data);
                break;
            // Handle other message types
        }
    }
    
    generateId() {
        return Math.random().toString(36).substr(2, 9);
    }
}

// Usage
const client = new {{ApiName}}Client('{{websocket_url}}', 'YOUR_TOKEN');

client.connect().then(() => {
    console.log('Connected and authenticated');
    client.subscribe('user_joined');
    client.subscribe('message_received');
}).catch((error) => {
    console.error('Connection failed:', error);
});
```

### Python Client

```python
import asyncio
import websockets
import json
from datetime import datetime

class {{ApiName}}Client:
    def __init__(self, url, token):
        self.url = url
        self.token = token
        self.ws = None
        self.authenticated = False
    
    async def connect(self):
        self.ws = await websockets.connect(self.url)
        await self.authenticate()
        
        # Start listening for messages
        asyncio.create_task(self.listen())
    
    async def authenticate(self):
        auth_message = {
            'type': 'auth',
            'token': self.token
        }
        
        await self.ws.send(json.dumps(auth_message))
        
        response = await self.ws.recv()
        message = json.loads(response)
        
        if message.get('type') == 'auth_response' and message.get('success'):
            self.authenticated = True
            print('Authenticated successfully')
        else:
            raise Exception('Authentication failed')
    
    async def subscribe(self, event_type):
        if not self.authenticated:
            raise Exception('Not authenticated')
        
        subscribe_message = {
            'type': 'subscribe',
            'event': event_type
        }
        
        await self.ws.send(json.dumps(subscribe_message))
    
    async def send_message(self, message_type, data):
        if not self.authenticated:
            raise Exception('Not authenticated')
        
        message = {
            'type': message_type,
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'id': self.generate_id()
        }
        
        await self.ws.send(json.dumps(message))
    
    async def listen(self):
        try:
            async for message in self.ws:
                data = json.loads(message)
                await self.handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            print('Connection closed')
            self.authenticated = False
    
    async def handle_message(self, message):
        message_type = message.get('type')
        
        if message_type == 'error':
            print(f"Error: {message['data']}")
        elif message_type == 'user_joined':
            print(f"User joined: {message['data']}")
        # Handle other message types
    
    def generate_id(self):
        import random
        import string
        return ''.join(random.choices(string.ascii_letters + string.digits, k=9))

# Usage
async def main():
    client = {{ApiName}}Client('{{websocket_url}}', 'YOUR_TOKEN')
    
    try:
        await client.connect()
        await client.subscribe('user_joined')
        await client.subscribe('message_received')
        
        # Keep connection alive
        await asyncio.sleep(3600)  # 1 hour
    except Exception as e:
        print(f'Error: {e}')

asyncio.run(main())
```

## Rate Limiting

{{#rate_limits}}
{{rate_limits}}
{{/rate_limits}}
{{^rate_limits}}
WebSocket connections are rate limited to prevent abuse:

- **Messages per minute**: 60
- **Subscriptions per connection**: 10
- **Concurrent connections per IP**: 5

Rate limit violations result in temporary connection suspension or termination.
{{/rate_limits}}

## Connection Management

### Heartbeat/Keep-Alive

The server sends periodic ping frames to maintain the connection:

```json
{
  "type": "ping",
  "timestamp": "2023-01-01T12:00:00Z"
}
```

Clients should respond with a pong:

```json
{
  "type": "pong",
  "timestamp": "2023-01-01T12:00:00Z"
}
```

### Reconnection

Implement automatic reconnection with exponential backoff:

```javascript
class ReconnectingWebSocket {
    constructor(url, token) {
        this.url = url;
        this.token = token;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectInterval = 1000; // Start with 1 second
    }
    
    connect() {
        // Connection logic...
        
        this.ws.onclose = (event) => {
            if (event.code !== 1000) { // Not a normal closure
                this.reconnect();
            }
        };
    }
    
    reconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            
            setTimeout(() => {
                console.log(`Reconnecting... Attempt ${this.reconnectAttempts}`);
                this.connect();
            }, this.reconnectInterval);
            
            // Exponential backoff
            this.reconnectInterval *= 2;
        } else {
            console.error('Max reconnection attempts reached');
        }
    }
}
```

---

*Generated on {{current_date}} | Version {{version}}*
''',
            examples=[
                '''# Chat WebSocket API

Real-time chat API using WebSockets.

**WebSocket URL:** `wss://api.chat.example.com/ws`

## Connection

```javascript
const ws = new WebSocket('wss://api.chat.example.com/ws');

ws.onopen = function() {
    // Authenticate
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'YOUR_TOKEN'
    }));
};
```

## Events

### message_received

Triggered when a new message is posted to a subscribed channel.

**Message Format:**

```json
{
  "type": "message_received",
  "data": {
    "channel_id": "123",
    "user_id": "456",
    "message": "Hello, world!",
    "timestamp": "2023-01-01T12:00:00Z"
  }
}
```
'''
            ]
        )
        
        self.templates["websocket_comprehensive"] = websocket_template
    

    def get_templates(self) -> Dict[str, Template]:
        """Get all WEBSOCKET templates."""
        return self.templates
    
    def get_template(self, name: str) -> Template:
        """Get a specific WEBSOCKET template by name."""
        return self.templates.get(name)
