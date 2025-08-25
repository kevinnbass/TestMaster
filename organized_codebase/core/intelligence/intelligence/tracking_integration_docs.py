"""
Tracking Integration Documentation Module
Handles documentation for tracking, analytics, and observability integrations
"""

from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import re
from pathlib import Path


class TrackingProvider(Enum):
    """Supported tracking and analytics providers"""
    GOOGLE_ANALYTICS = "google_analytics"
    MIXPANEL = "mixpanel"
    AMPLITUDE = "amplitude" 
    SEGMENT = "segment"
    POSTHOG = "posthog"
    HOTJAR = "hotjar"
    FULLSTORY = "fullstory"
    LOGFLARE = "logflare"
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    SENTRY = "sentry"
    BUGSNAG = "bugsnag"
    CUSTOM = "custom"


class EventType(Enum):
    """Types of tracking events"""
    PAGE_VIEW = "page_view"
    CLICK = "click"
    FORM_SUBMIT = "form_submit"
    DOWNLOAD = "download"
    ERROR = "error"
    CUSTOM_EVENT = "custom_event"
    USER_IDENTIFICATION = "user_identification"
    CONVERSION = "conversion"
    SCROLL = "scroll"
    SESSION_START = "session_start"
    SESSION_END = "session_end"


@dataclass
class TrackingEvent:
    """Tracking event definition"""
    name: str
    event_type: EventType
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    required_properties: List[str] = field(default_factory=list)
    example_payload: Dict[str, Any] = field(default_factory=dict)
    triggers: List[str] = field(default_factory=list)
    providers: List[TrackingProvider] = field(default_factory=list)


@dataclass
class TrackingIntegration:
    """Tracking integration configuration"""
    provider: TrackingProvider
    name: str
    description: str = ""
    api_key_required: bool = True
    configuration: Dict[str, Any] = field(default_factory=dict)
    supported_events: List[EventType] = field(default_factory=list)
    implementation_code: Dict[str, str] = field(default_factory=dict)  # language -> code
    setup_instructions: List[str] = field(default_factory=list)
    privacy_considerations: List[str] = field(default_factory=list)
    gdpr_compliant: bool = True


@dataclass
class DataSchema:
    """Data schema for tracking properties"""
    property_name: str
    data_type: str  # string, number, boolean, object, array
    description: str = ""
    required: bool = False
    example_value: Any = None
    validation_rules: List[str] = field(default_factory=list)
    pii_data: bool = False  # Personally Identifiable Information


class TrackingIntegrationDocsSystem:
    """System for managing tracking integration documentation"""
    
    def __init__(self):
        self.integrations = {}
        self.events = {}
        self.schemas = {}
        self.privacy_policies = {}
        self.implementation_guides = {}
        
    def add_tracking_integration(self, integration: TrackingIntegration) -> None:
        """Add a tracking integration"""
        self.integrations[integration.provider] = integration
    
    def add_tracking_event(self, event: TrackingEvent) -> None:
        """Add a tracking event definition"""
        self.events[event.name] = event
    
    def add_data_schema(self, schema: DataSchema) -> None:
        """Add a data schema definition"""
        self.schemas[schema.property_name] = schema
    
    def generate_integration_guide(self, provider: TrackingProvider) -> str:
        """Generate integration guide for a specific provider"""
        if provider not in self.integrations:
            return f"Integration guide for {provider.value} not found"
        
        integration = self.integrations[provider]
        
        guide_sections = []
        
        # Header
        guide_sections.append(f"# {integration.name} Integration Guide")
        guide_sections.append("")
        guide_sections.append(integration.description)
        guide_sections.append("")
        
        # Overview
        guide_sections.append("## Overview")
        guide_sections.append("")
        guide_sections.append(f"This guide covers the integration of {integration.name} for tracking and analytics.")
        guide_sections.append(f"**Provider:** {provider.value}")
        guide_sections.append(f"**API Key Required:** {'Yes' if integration.api_key_required else 'No'}")
        guide_sections.append(f"**GDPR Compliant:** {'Yes' if integration.gdpr_compliant else 'No'}")
        guide_sections.append("")
        
        # Setup Instructions
        if integration.setup_instructions:
            guide_sections.append("## Setup Instructions")
            guide_sections.append("")
            for i, instruction in enumerate(integration.setup_instructions, 1):
                guide_sections.append(f"{i}. {instruction}")
            guide_sections.append("")
        
        # Configuration
        if integration.configuration:
            guide_sections.append("## Configuration")
            guide_sections.append("")
            guide_sections.append("```json")
            guide_sections.append(json.dumps(integration.configuration, indent=2))
            guide_sections.append("```")
            guide_sections.append("")
        
        # Implementation Examples
        if integration.implementation_code:
            guide_sections.append("## Implementation")
            guide_sections.append("")
            
            for language, code in integration.implementation_code.items():
                guide_sections.append(f"### {language.title()}")
                guide_sections.append("")
                guide_sections.append(f"```{language.lower()}")
                guide_sections.append(code)
                guide_sections.append("```")
                guide_sections.append("")
        
        # Supported Events
        if integration.supported_events:
            guide_sections.append("## Supported Events")
            guide_sections.append("")
            for event_type in integration.supported_events:
                guide_sections.append(f"- {event_type.value}")
            guide_sections.append("")
        
        # Privacy Considerations
        if integration.privacy_considerations:
            guide_sections.append("## Privacy Considerations")
            guide_sections.append("")
            for consideration in integration.privacy_considerations:
                guide_sections.append(f"- {consideration}")
            guide_sections.append("")
        
        return "\n".join(guide_sections)
    
    def generate_event_documentation(self, event_name: str) -> str:
        """Generate documentation for a specific tracking event"""
        if event_name not in self.events:
            return f"Event documentation for '{event_name}' not found"
        
        event = self.events[event_name]
        
        doc_sections = []
        
        # Header
        doc_sections.append(f"# {event.name} Event")
        doc_sections.append("")
        doc_sections.append(event.description)
        doc_sections.append("")
        
        # Basic Information
        doc_sections.append("## Event Details")
        doc_sections.append("")
        doc_sections.append(f"- **Event Type:** {event.event_type.value}")
        doc_sections.append(f"- **Event Name:** `{event.name}`")
        
        if event.providers:
            providers_list = [provider.value for provider in event.providers]
            doc_sections.append(f"- **Supported Providers:** {', '.join(providers_list)}")
        
        doc_sections.append("")
        
        # Triggers
        if event.triggers:
            doc_sections.append("## Triggers")
            doc_sections.append("")
            doc_sections.append("This event is triggered by:")
            for trigger in event.triggers:
                doc_sections.append(f"- {trigger}")
            doc_sections.append("")
        
        # Properties
        if event.properties:
            doc_sections.append("## Event Properties")
            doc_sections.append("")
            doc_sections.append("| Property | Type | Required | Description |")
            doc_sections.append("|----------|------|----------|-------------|")
            
            for prop_name, prop_info in event.properties.items():
                is_required = "Yes" if prop_name in event.required_properties else "No"
                prop_type = prop_info.get("type", "string")
                prop_desc = prop_info.get("description", "")
                
                doc_sections.append(f"| {prop_name} | {prop_type} | {is_required} | {prop_desc} |")
            
            doc_sections.append("")
        
        # Example Payload
        if event.example_payload:
            doc_sections.append("## Example Payload")
            doc_sections.append("")
            doc_sections.append("```json")
            doc_sections.append(json.dumps(event.example_payload, indent=2))
            doc_sections.append("```")
            doc_sections.append("")
        
        # Implementation Examples
        doc_sections.append("## Implementation Examples")
        doc_sections.append("")
        
        # JavaScript example
        doc_sections.append("### JavaScript")
        doc_sections.append("")
        doc_sections.append("```javascript")
        js_example = self._generate_js_tracking_example(event)
        doc_sections.append(js_example)
        doc_sections.append("```")
        doc_sections.append("")
        
        # Python example
        doc_sections.append("### Python")
        doc_sections.append("")
        doc_sections.append("```python")
        python_example = self._generate_python_tracking_example(event)
        doc_sections.append(python_example)
        doc_sections.append("```")
        doc_sections.append("")
        
        return "\n".join(doc_sections)
    
    def _generate_js_tracking_example(self, event: TrackingEvent) -> str:
        """Generate JavaScript tracking example"""
        properties_str = json.dumps(event.example_payload, indent=2) if event.example_payload else "{}"
        
        return f"""// Track {event.name} event
analytics.track('{event.name}', {properties_str});

// Alternative with try-catch
try {{
    analytics.track('{event.name}', {properties_str});
}} catch (error) {{
    console.error('Tracking error:', error);
}}"""
    
    def _generate_python_tracking_example(self, event: TrackingEvent) -> str:
        """Generate Python tracking example"""
        properties_str = json.dumps(event.example_payload, indent=2) if event.example_payload else "{}"
        
        return f"""import analytics

# Track {event.name} event
analytics.track(
    user_id='user123',
    event='{event.name}',
    properties={properties_str}
)

# With error handling
try:
    analytics.track(
        user_id='user123',
        event='{event.name}',
        properties={properties_str}
    )
except Exception as e:
    print(f"Tracking error: {{e}}")"""
    
    def generate_privacy_documentation(self) -> str:
        """Generate comprehensive privacy documentation"""
        doc_sections = []
        
        doc_sections.append("# Privacy and Data Protection")
        doc_sections.append("")
        doc_sections.append("This document outlines our approach to privacy and data protection in tracking integrations.")
        doc_sections.append("")
        
        # GDPR Compliance
        doc_sections.append("## GDPR Compliance")
        doc_sections.append("")
        gdpr_compliant = [i for i in self.integrations.values() if i.gdpr_compliant]
        doc_sections.append(f"**GDPR Compliant Integrations:** {len(gdpr_compliant)} out of {len(self.integrations)}")
        doc_sections.append("")
        
        for integration in gdpr_compliant:
            doc_sections.append(f"- {integration.name}")
        doc_sections.append("")
        
        # PII Data Handling
        pii_schemas = [s for s in self.schemas.values() if s.pii_data]
        if pii_schemas:
            doc_sections.append("## Personally Identifiable Information (PII)")
            doc_sections.append("")
            doc_sections.append("The following data properties contain PII and require special handling:")
            doc_sections.append("")
            
            for schema in pii_schemas:
                doc_sections.append(f"### {schema.property_name}")
                doc_sections.append(f"- **Type:** {schema.data_type}")
                doc_sections.append(f"- **Description:** {schema.description}")
                doc_sections.append("")
        
        # Data Retention
        doc_sections.append("## Data Retention")
        doc_sections.append("")
        doc_sections.append("Different providers have different data retention policies:")
        doc_sections.append("")
        
        retention_info = {
            TrackingProvider.GOOGLE_ANALYTICS: "26 months (configurable)",
            TrackingProvider.MIXPANEL: "Custom retention based on plan",
            TrackingProvider.AMPLITUDE: "Custom retention based on plan",
            TrackingProvider.SEGMENT: "Depends on destination",
        }
        
        for provider, retention in retention_info.items():
            if provider in self.integrations:
                integration = self.integrations[provider]
                doc_sections.append(f"- **{integration.name}:** {retention}")
        
        doc_sections.append("")
        
        # User Rights
        doc_sections.append("## User Rights")
        doc_sections.append("")
        doc_sections.append("Users have the following rights regarding their tracking data:")
        doc_sections.append("")
        doc_sections.append("1. **Right to Access:** Users can request access to their tracked data")
        doc_sections.append("2. **Right to Deletion:** Users can request deletion of their data")
        doc_sections.append("3. **Right to Opt-out:** Users can opt out of tracking")
        doc_sections.append("4. **Right to Portability:** Users can request data in portable format")
        doc_sections.append("")
        
        return "\n".join(doc_sections)
    
    def generate_tracking_reference(self) -> str:
        """Generate complete tracking reference documentation"""
        ref_sections = []
        
        ref_sections.append("# Tracking Reference")
        ref_sections.append("")
        ref_sections.append("Complete reference for all tracking events and integrations.")
        ref_sections.append("")
        
        # Events by Type
        events_by_type = {}
        for event in self.events.values():
            if event.event_type not in events_by_type:
                events_by_type[event.event_type] = []
            events_by_type[event.event_type].append(event)
        
        ref_sections.append("## Events by Type")
        ref_sections.append("")
        
        for event_type, events in events_by_type.items():
            ref_sections.append(f"### {event_type.value.replace('_', ' ').title()}")
            ref_sections.append("")
            
            for event in events:
                ref_sections.append(f"#### {event.name}")
                ref_sections.append(f"{event.description}")
                ref_sections.append("")
                
                if event.required_properties:
                    ref_sections.append(f"**Required Properties:** {', '.join(event.required_properties)}")
                    ref_sections.append("")
        
        # Integration Summary
        ref_sections.append("## Integration Summary")
        ref_sections.append("")
        ref_sections.append("| Provider | Events Supported | API Key Required | GDPR Compliant |")
        ref_sections.append("|----------|------------------|------------------|-----------------|")
        
        for integration in self.integrations.values():
            events_count = len(integration.supported_events)
            api_key = os.getenv('KEY') if integration.api_key_required else "No"
            gdpr = "Yes" if integration.gdpr_compliant else "No"
            
            ref_sections.append(f"| {integration.name} | {events_count} | {api_key} | {gdpr} |")
        
        ref_sections.append("")
        
        # Data Schema Reference
        if self.schemas:
            ref_sections.append("## Data Schema Reference")
            ref_sections.append("")
            ref_sections.append("| Property | Type | Required | PII | Description |")
            ref_sections.append("|----------|------|----------|-----|-------------|")
            
            for schema in self.schemas.values():
                required = "Yes" if schema.required else "No"
                pii = "Yes" if schema.pii_data else "No"
                
                ref_sections.append(f"| {schema.property_name} | {schema.data_type} | {required} | {pii} | {schema.description} |")
            
            ref_sections.append("")
        
        return "\n".join(ref_sections)
    
    def validate_tracking_implementation(self, implementation_code: str, 
                                       language: str = "javascript") -> List[str]:
        """Validate tracking implementation code"""
        issues = []
        
        if language.lower() == "javascript":
            # Check for common JavaScript tracking issues
            if "analytics.track" not in implementation_code:
                issues.append("Missing analytics.track() calls")
            
            if "try" not in implementation_code and "catch" not in implementation_code:
                issues.append("Missing error handling (try/catch blocks)")
            
            # Check for hardcoded values
            if re.search(r'["\'][a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+\.[a-zA-Z]{2,}["\']', implementation_code):
                issues.append("Hardcoded email addresses found - consider using variables")
            
            # Check for console.log (should be removed in production)
            if "console.log" in implementation_code:
                issues.append("console.log statements found - remove for production")
        
        elif language.lower() == "python":
            # Check for Python-specific issues
            if "import analytics" not in implementation_code:
                issues.append("Missing analytics import")
            
            if "except" not in implementation_code:
                issues.append("Missing exception handling")
        
        return issues
    
    def generate_implementation_checklist(self, provider: TrackingProvider) -> List[str]:
        """Generate implementation checklist for a provider"""
        if provider not in self.integrations:
            return [f"Provider {provider.value} not found"]
        
        integration = self.integrations[provider]
        checklist = []
        
        # Basic setup
        checklist.append("□ Install tracking library/SDK")
        
        if integration.api_key_required:
            checklist.append("□ Obtain API key from provider")
            checklist.append("□ Configure API key in environment variables")
        
        checklist.append("□ Initialize tracking in application")
        checklist.append("□ Set up error handling")
        
        # Event implementation
        checklist.append("□ Implement required tracking events")
        checklist.append("□ Add event properties according to schema")
        checklist.append("□ Test events in development environment")
        
        # Privacy and compliance
        if integration.gdpr_compliant:
            checklist.append("□ Implement consent management")
            checklist.append("□ Add opt-out functionality")
        
        checklist.append("□ Review privacy policy")
        checklist.append("□ Test data deletion procedures")
        
        # Production readiness
        checklist.append("□ Remove debug/console statements")
        checklist.append("□ Test in staging environment")
        checklist.append("□ Verify data appears in provider dashboard")
        checklist.append("□ Set up monitoring and alerts")
        
        return checklist
    
    def export_tracking_config(self, format: str = "json") -> str:
        """Export tracking configuration in specified format"""
        config = {
            "integrations": {
                provider.value: {
                    "name": integration.name,
                    "description": integration.description,
                    "api_key_required": integration.api_key_required,
                    "configuration": integration.configuration,
                    "supported_events": [event.value for event in integration.supported_events],
                    "gdpr_compliant": integration.gdpr_compliant
                }
                for provider, integration in self.integrations.items()
            },
            "events": {
                event.name: {
                    "type": event.event_type.value,
                    "description": event.description,
                    "properties": event.properties,
                    "required_properties": event.required_properties,
                    "example_payload": event.example_payload,
                    "triggers": event.triggers,
                    "providers": [provider.value for provider in event.providers]
                }
                for event in self.events.values()
            },
            "schemas": {
                schema.property_name: {
                    "data_type": schema.data_type,
                    "description": schema.description,
                    "required": schema.required,
                    "example_value": schema.example_value,
                    "pii_data": schema.pii_data
                }
                for schema in self.schemas.values()
            }
        }
        
        if format.lower() == "json":
            return json.dumps(config, indent=2)
        elif format.lower() == "yaml":
            import yaml
            return yaml.dump(config, default_flow_style=False)
        
        return str(config)
    
    def generate_gdpr_compliance_report(self) -> Dict[str, Any]:
        """Generate GDPR compliance report"""
        report = {
            "compliance_summary": {
                "total_integrations": len(self.integrations),
                "gdpr_compliant": 0,
                "non_compliant": 0
            },
            "compliant_integrations": [],
            "non_compliant_integrations": [],
            "pii_data_points": [],
            "recommendations": []
        }
        
        for integration in self.integrations.values():
            if integration.gdpr_compliant:
                report["compliance_summary"]["gdpr_compliant"] += 1
                report["compliant_integrations"].append(integration.name)
            else:
                report["compliance_summary"]["non_compliant"] += 1
                report["non_compliant_integrations"].append(integration.name)
        
        # PII data points
        for schema in self.schemas.values():
            if schema.pii_data:
                report["pii_data_points"].append({
                    "property": schema.property_name,
                    "type": schema.data_type,
                    "description": schema.description
                })
        
        # Recommendations
        if report["compliance_summary"]["non_compliant"] > 0:
            report["recommendations"].append("Review non-compliant integrations for GDPR compliance")
        
        if len(report["pii_data_points"]) > 0:
            report["recommendations"].append("Implement consent management for PII data collection")
            report["recommendations"].append("Add data deletion procedures for user requests")
        
        report["recommendations"].append("Regular compliance audits recommended")
        
        return report