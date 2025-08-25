
# AGENT D SECURITY INTEGRATION
try:
    from SECURITY_PATCHES.api_security_framework import APISecurityFramework
    from SECURITY_PATCHES.authentication_framework import SecurityFramework
    _security_framework = SecurityFramework()
    _api_security = APISecurityFramework()
    _SECURITY_ENABLED = True
except ImportError:
    _SECURITY_ENABLED = False
    print("Security frameworks not available - running without protection")

def apply_security_middleware():
    """Apply security middleware to requests"""
    if not _SECURITY_ENABLED:
        return True, {}
    
    from flask import request
    request_data = {
        'ip_address': request.remote_addr,
        'endpoint': request.path,
        'method': request.method,
        'user_agent': request.headers.get('User-Agent', ''),
        'body': request.get_json() if request.is_json else {},
        'query_params': dict(request.args),
        'headers': dict(request.headers)
    }
    
    return _api_security.validate_request(request_data)

"""
WEBHOOK API Documentation Templates
"""

from typing import Dict
from ..template_engine import Template, TemplateMetadata, TemplateType, TemplateFormat
from .base import ApiContext, ApiEndpoint, ApiModel, ApiType, AuthenticationType


class WebhookTemplateProvider:
    """Provider for WEBHOOK API documentation templates."""
    
    def __init__(self):
        self.templates: Dict[str, Template] = {}
        self._create_webhook_templates()
    
    def _create_webhook_templates(self):
        """Create Webhook API documentation templates."""
        
        webhook_template = Template(
            metadata=TemplateMetadata(
                name="webhook_comprehensive",
                description="Comprehensive Webhook API documentation template",
                template_type=TemplateType.API_DOC,
                format=TemplateFormat.MARKDOWN,
                author="TestMaster",
                version="1.0.0",
                tags=["webhook", "api", "callbacks"],
                required_variables=["api_name", "description"],
                optional_variables=["webhooks", "security", "examples"],
                target_audience="all"
            ),
            content='''# {{api_name}} Webhooks

{{description}}

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Security](#security)
- [Webhook Events](#webhook-events)
- [Payload Format](#payload-format)
- [Handling Webhooks](#handling-webhooks)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Overview

Webhooks allow you to receive real-time notifications when events occur in {{api_name}}. Instead of polling our API for changes, we'll send HTTP POST requests to your specified endpoints when relevant events happen.

### Benefits

- **Real-time notifications**: Receive updates immediately when events occur
- **Reduced API calls**: No need to poll for changes
- **Scalable**: Handle high volumes of events efficiently
- **Reliable**: Built-in retry mechanisms ensure delivery

## Setup

### 1. Create Webhook Endpoint

Create an endpoint in your application to receive webhook notifications:

```python
from flask import Flask, request, jsonify
import hmac
import hashlib

app = Flask(__name__)

@app.route('/webhooks/{{api_name}}', methods=['POST'])

    def get_templates(self) -> Dict[str, Template]:
        """Get all WEBHOOK templates."""
        return self.templates
    
    def get_template(self, name: str) -> Template:
        """Get a specific WEBHOOK template by name."""
        return self.templates.get(name)
