"""
YAML Configuration Processor Module
Handles YAML-based API specifications and configuration documentation
"""

import yaml
import json
import re
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime


class YAMLType(Enum):
    """Types of YAML configurations"""
    OPENAPI_SPEC = "openapi"
    KUBERNETES_CONFIG = "kubernetes"
    DOCKER_COMPOSE = "docker-compose"
    CI_CD_CONFIG = "ci-cd"
    APPLICATION_CONFIG = "application"
    DOCUMENTATION_CONFIG = "documentation"
    UNKNOWN = "unknown"


@dataclass
class YAMLValidationIssue:
    """YAML validation issue"""
    severity: str  # error, warning, info
    message: str
    path: str = ""
    line_number: Optional[int] = None
    suggestion: str = ""


@dataclass
class YAMLMetadata:
    """Metadata extracted from YAML file"""
    yaml_type: YAMLType
    version: str = ""
    title: str = ""
    description: str = ""
    author: str = ""
    created_date: str = ""
    last_modified: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class YAMLConfigProcessor:
    """Processes and analyzes YAML configuration files"""
    
    def __init__(self):
        self.schemas = self._load_validation_schemas()
        self.processors = {
            YAMLType.OPENAPI_SPEC: self._process_openapi_spec,
            YAMLType.KUBERNETES_CONFIG: self._process_kubernetes_config,
            YAMLType.DOCKER_COMPOSE: self._process_docker_compose,
            YAMLType.CI_CD_CONFIG: self._process_ci_cd_config,
            YAMLType.APPLICATION_CONFIG: self._process_application_config,
            YAMLType.DOCUMENTATION_CONFIG: self._process_documentation_config
        }
    
    def _load_validation_schemas(self) -> Dict[YAMLType, Dict[str, Any]]:
        """Load validation schemas for different YAML types"""
        return {
            YAMLType.OPENAPI_SPEC: {
                "required_fields": ["openapi", "info", "paths"],
                "optional_fields": ["servers", "components", "security", "tags"],
                "version_patterns": [r"3\.\d+\.\d+", r"2\.\d+"]
            },
            YAMLType.KUBERNETES_CONFIG: {
                "required_fields": ["apiVersion", "kind"],
                "optional_fields": ["metadata", "spec", "data"],
                "api_versions": ["v1", "apps/v1", "networking.k8s.io/v1"]
            },
            YAMLType.DOCKER_COMPOSE: {
                "required_fields": ["version", "services"],
                "optional_fields": ["networks", "volumes", "secrets", "configs"],
                "version_patterns": [r"3\.\d+", r"2\.\d+"]
            }
        }
    
    def load_yaml_file(self, file_path: Path) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Load and parse YAML file with error handling"""
        errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Try to parse as YAML
            data = yaml.safe_load(content)
            return data, errors
            
        except yaml.YAMLError as e:
            errors.append(f"YAML parsing error: {str(e)}")
            return None, errors
        except FileNotFoundError:
            errors.append(f"File not found: {file_path}")
            return None, errors
        except Exception as e:
            errors.append(f"Unexpected error: {str(e)}")
            return None, errors
    
    def identify_yaml_type(self, data: Dict[str, Any]) -> YAMLType:
        """Identify the type of YAML configuration"""
        if not data or not isinstance(data, dict):
            return YAMLType.UNKNOWN
        
        # Check for OpenAPI specification
        if "openapi" in data or "swagger" in data:
            return YAMLType.OPENAPI_SPEC
        
        # Check for Kubernetes configuration
        if "apiVersion" in data and "kind" in data:
            return YAMLType.KUBERNETES_CONFIG
        
        # Check for Docker Compose
        if "version" in data and "services" in data:
            return YAMLType.DOCKER_COMPOSE
        
        # Check for CI/CD configurations
        ci_cd_indicators = ["stages", "jobs", "pipeline", "workflow", "steps"]
        if any(indicator in data for indicator in ci_cd_indicators):
            return YAMLType.CI_CD_CONFIG
        
        # Check for documentation configuration
        doc_indicators = ["navigation", "theme", "plugins", "markdown_extensions"]
        if any(indicator in data for indicator in doc_indicators):
            return YAMLType.DOCUMENTATION_CONFIG
        
        # Default to application config
        return YAMLType.APPLICATION_CONFIG
    
    def extract_metadata(self, data: Dict[str, Any], yaml_type: YAMLType) -> YAMLMetadata:
        """Extract metadata from YAML data"""
        metadata = YAMLMetadata(yaml_type=yaml_type)
        
        if yaml_type == YAMLType.OPENAPI_SPEC:
            info = data.get("info", {})
            metadata.version = info.get("version", "")
            metadata.title = info.get("title", "")
            metadata.description = info.get("description", "")
            
            # Extract tags
            if "tags" in data:
                metadata.tags = [tag.get("name", "") for tag in data["tags"]]
        
        elif yaml_type == YAMLType.KUBERNETES_CONFIG:
            metadata.version = data.get("apiVersion", "")
            metadata.title = data.get("kind", "")
            
            # Extract from metadata section
            k8s_metadata = data.get("metadata", {})
            metadata.title = k8s_metadata.get("name", metadata.title)
            metadata.description = k8s_metadata.get("description", "")
            
            # Extract labels as tags
            labels = k8s_metadata.get("labels", {})
            metadata.tags = list(labels.keys())
        
        elif yaml_type == YAMLType.DOCKER_COMPOSE:
            metadata.version = str(data.get("version", ""))
            
            # Extract service names as dependencies
            services = data.get("services", {})
            metadata.dependencies = list(services.keys())
            
            # Count services for title
            service_count = len(services)
            metadata.title = f"Docker Compose ({service_count} services)"
        
        return metadata
    
    def validate_yaml(self, data: Dict[str, Any], yaml_type: YAMLType) -> List[YAMLValidationIssue]:
        """Validate YAML structure and content"""
        issues = []
        
        if yaml_type not in self.schemas:
            return issues
        
        schema = self.schemas[yaml_type]
        
        # Check required fields
        for field in schema.get("required_fields", []):
            if field not in data:
                issues.append(YAMLValidationIssue(
                    severity="error",
                    message=f"Missing required field: {field}",
                    path=field,
                    suggestion=f"Add the '{field}' field to your YAML configuration"
                ))
        
        # Validate version patterns
        if "version_patterns" in schema and "version" in data:
            version = str(data["version"])
            patterns = schema["version_patterns"]
            
            if not any(re.match(pattern, version) for pattern in patterns):
                issues.append(YAMLValidationIssue(
                    severity="warning",
                    message=f"Version '{version}' doesn't match expected patterns",
                    path="version",
                    suggestion=f"Use version format matching: {', '.join(patterns)}"
                ))
        
        # Type-specific validation
        if yaml_type == YAMLType.OPENAPI_SPEC:
            issues.extend(self._validate_openapi_spec(data))
        elif yaml_type == YAMLType.KUBERNETES_CONFIG:
            issues.extend(self._validate_kubernetes_config(data))
        elif yaml_type == YAMLType.DOCKER_COMPOSE:
            issues.extend(self._validate_docker_compose(data))
        
        return issues
    
    def _validate_openapi_spec(self, data: Dict[str, Any]) -> List[YAMLValidationIssue]:
        """Validate OpenAPI specification"""
        issues = []
        
        # Validate info section
        info = data.get("info", {})
        if not info.get("title"):
            issues.append(YAMLValidationIssue(
                severity="error",
                message="Missing API title in info section",
                path="info.title"
            ))
        
        if not info.get("version"):
            issues.append(YAMLValidationIssue(
                severity="error",
                message="Missing API version in info section",
                path="info.version"
            ))
        
        # Validate paths
        paths = data.get("paths", {})
        if not paths:
            issues.append(YAMLValidationIssue(
                severity="warning",
                message="No API paths defined",
                path="paths"
            ))
        
        # Check for common HTTP methods
        for path, path_obj in paths.items():
            if not isinstance(path_obj, dict):
                continue
                
            methods = set(path_obj.keys())
            http_methods = {"get", "post", "put", "delete", "patch", "head", "options"}
            
            if not methods.intersection(http_methods):
                issues.append(YAMLValidationIssue(
                    severity="warning",
                    message=f"Path '{path}' has no HTTP method operations",
                    path=f"paths.{path}"
                ))
        
        return issues
    
    def _validate_kubernetes_config(self, data: Dict[str, Any]) -> List[YAMLValidationIssue]:
        """Validate Kubernetes configuration"""
        issues = []
        
        api_version = data.get("apiVersion", "")
        kind = data.get("kind", "")
        
        # Validate common resource requirements
        if kind in ["Deployment", "StatefulSet", "DaemonSet"]:
            spec = data.get("spec", {})
            if "selector" not in spec:
                issues.append(YAMLValidationIssue(
                    severity="error",
                    message=f"{kind} requires a selector",
                    path="spec.selector"
                ))
            
            if "template" not in spec:
                issues.append(YAMLValidationIssue(
                    severity="error",
                    message=f"{kind} requires a pod template",
                    path="spec.template"
                ))
        
        # Check metadata
        metadata = data.get("metadata", {})
        if not metadata.get("name"):
            issues.append(YAMLValidationIssue(
                severity="error",
                message="Resource must have a name",
                path="metadata.name"
            ))
        
        return issues
    
    def _validate_docker_compose(self, data: Dict[str, Any]) -> List[YAMLValidationIssue]:
        """Validate Docker Compose configuration"""
        issues = []
        
        services = data.get("services", {})
        
        for service_name, service_config in services.items():
            if not isinstance(service_config, dict):
                continue
            
            # Check for image or build
            if "image" not in service_config and "build" not in service_config:
                issues.append(YAMLValidationIssue(
                    severity="error",
                    message=f"Service '{service_name}' must specify either 'image' or 'build'",
                    path=f"services.{service_name}"
                ))
            
            # Warn about exposed ports without explicit mapping
            if "expose" in service_config and "ports" not in service_config:
                issues.append(YAMLValidationIssue(
                    severity="warning",
                    message=f"Service '{service_name}' exposes ports but doesn't map them",
                    path=f"services.{service_name}",
                    suggestion="Consider adding 'ports' mapping for external access"
                ))
        
        return issues
    
    def process_yaml_content(self, data: Dict[str, Any], yaml_type: YAMLType) -> Dict[str, Any]:
        """Process YAML content based on its type"""
        if yaml_type in self.processors:
            return self.processors[yaml_type](data)
        
        return {"processed": False, "reason": f"No processor for type {yaml_type.value}"}
    
    def _process_openapi_spec(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process OpenAPI specification"""
        result = {
            "type": "openapi",
            "version": data.get("openapi", ""),
            "info": data.get("info", {}),
            "servers": data.get("servers", []),
            "paths_count": len(data.get("paths", {})),
            "components_count": len(data.get("components", {}).get("schemas", {})),
            "security_schemes": list(data.get("components", {}).get("securitySchemes", {}).keys()),
            "tags": [tag.get("name") for tag in data.get("tags", [])],
            "endpoints": self._extract_endpoints_from_openapi(data)
        }
        
        return result
    
    def _process_kubernetes_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Kubernetes configuration"""
        result = {
            "type": "kubernetes",
            "api_version": data.get("apiVersion", ""),
            "kind": data.get("kind", ""),
            "metadata": data.get("metadata", {}),
            "namespace": data.get("metadata", {}).get("namespace", "default"),
            "labels": data.get("metadata", {}).get("labels", {}),
            "annotations": data.get("metadata", {}).get("annotations", {})
        }
        
        # Add resource-specific information
        if data.get("kind") in ["Deployment", "StatefulSet", "DaemonSet"]:
            spec = data.get("spec", {})
            result["replicas"] = spec.get("replicas", 1)
            result["selector"] = spec.get("selector", {})
            
        elif data.get("kind") == "Service":
            spec = data.get("spec", {})
            result["service_type"] = spec.get("type", "ClusterIP")
            result["ports"] = spec.get("ports", [])
        
        return result
    
    def _process_docker_compose(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Docker Compose configuration"""
        services = data.get("services", {})
        
        result = {
            "type": "docker-compose",
            "version": data.get("version", ""),
            "services_count": len(services),
            "services": {},
            "networks": list(data.get("networks", {}).keys()),
            "volumes": list(data.get("volumes", {}).keys())
        }
        
        # Process each service
        for service_name, service_config in services.items():
            result["services"][service_name] = {
                "image": service_config.get("image", ""),
                "build": service_config.get("build", ""),
                "ports": service_config.get("ports", []),
                "environment": list(service_config.get("environment", [])),
                "volumes": service_config.get("volumes", []),
                "depends_on": service_config.get("depends_on", [])
            }
        
        return result
    
    def _process_ci_cd_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process CI/CD configuration"""
        result = {
            "type": "ci-cd",
            "stages": data.get("stages", []),
            "jobs": list(data.get("jobs", {}).keys()),
            "variables": data.get("variables", {}),
            "triggers": self._extract_ci_triggers(data)
        }
        
        return result
    
    def _process_application_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process application configuration"""
        result = {
            "type": "application",
            "configuration_keys": list(data.keys()),
            "nested_sections": []
        }
        
        # Find nested configuration sections
        for key, value in data.items():
            if isinstance(value, dict):
                result["nested_sections"].append({
                    "name": key,
                    "keys": list(value.keys())
                })
        
        return result
    
    def _process_documentation_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process documentation configuration"""
        result = {
            "type": "documentation",
            "theme": data.get("theme", ""),
            "plugins": data.get("plugins", []),
            "navigation_structure": self._analyze_navigation(data.get("navigation", {})),
            "markdown_extensions": data.get("markdown_extensions", [])
        }
        
        return result
    
    def _extract_endpoints_from_openapi(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract endpoint information from OpenAPI spec"""
        endpoints = []
        
        paths = data.get("paths", {})
        for path, path_obj in paths.items():
            if not isinstance(path_obj, dict):
                continue
                
            for method, operation in path_obj.items():
                if method.lower() in ["get", "post", "put", "delete", "patch", "head", "options"]:
                    endpoints.append({
                        "path": path,
                        "method": method.upper(),
                        "summary": operation.get("summary", ""),
                        "description": operation.get("description", ""),
                        "tags": operation.get("tags", []),
                        "operation_id": operation.get("operationId", ""),
                        "parameters_count": len(operation.get("parameters", [])),
                        "responses_count": len(operation.get("responses", {}))
                    })
        
        return endpoints
    
    def _extract_ci_triggers(self, data: Dict[str, Any]) -> List[str]:
        """Extract CI/CD triggers from configuration"""
        triggers = []
        
        # Common trigger patterns
        trigger_keys = ["on", "trigger", "when", "rules"]
        
        for key in trigger_keys:
            if key in data:
                trigger_value = data[key]
                if isinstance(trigger_value, list):
                    triggers.extend([str(t) for t in trigger_value])
                elif isinstance(trigger_value, dict):
                    triggers.extend(list(trigger_value.keys()))
                else:
                    triggers.append(str(trigger_value))
        
        return triggers
    
    def _analyze_navigation(self, navigation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze navigation structure"""
        if not navigation:
            return {}
        
        def count_items(nav_item):
            if isinstance(nav_item, dict):
                if "pages" in nav_item:
                    return len(nav_item["pages"])
                elif "groups" in nav_item:
                    return sum(count_items(group) for group in nav_item["groups"])
            elif isinstance(nav_item, list):
                return sum(count_items(item) for item in nav_item)
            return 1
        
        return {
            "languages": len(navigation.get("languages", [])),
            "total_pages": count_items(navigation),
            "has_tabs": "tabs" in str(navigation).lower(),
            "has_groups": "groups" in str(navigation).lower()
        }
    
    def generate_documentation_summary(self, data: Dict[str, Any], 
                                     yaml_type: YAMLType, 
                                     metadata: YAMLMetadata,
                                     issues: List[YAMLValidationIssue]) -> str:
        """Generate a summary documentation for the YAML file"""
        lines = []
        
        lines.append(f"# {metadata.title or 'YAML Configuration'}")
        lines.append("")
        
        if metadata.description:
            lines.append(metadata.description)
            lines.append("")
        
        # Basic information
        lines.append("## Basic Information")
        lines.append("")
        lines.append(f"- **Type:** {yaml_type.value}")
        if metadata.version:
            lines.append(f"- **Version:** {metadata.version}")
        if metadata.tags:
            lines.append(f"- **Tags:** {', '.join(metadata.tags)}")
        lines.append("")
        
        # Structure summary
        processed_data = self.process_yaml_content(data, yaml_type)
        
        if yaml_type == YAMLType.OPENAPI_SPEC:
            lines.append("## API Summary")
            lines.append("")
            lines.append(f"- **Paths:** {processed_data.get('paths_count', 0)}")
            lines.append(f"- **Components:** {processed_data.get('components_count', 0)}")
            lines.append(f"- **Security Schemes:** {', '.join(processed_data.get('security_schemes', []))}")
            
            endpoints = processed_data.get('endpoints', [])
            if endpoints:
                lines.append(f"- **Endpoints:** {len(endpoints)}")
                
        elif yaml_type == YAMLType.DOCKER_COMPOSE:
            lines.append("## Docker Compose Summary")
            lines.append("")
            lines.append(f"- **Services:** {processed_data.get('services_count', 0)}")
            lines.append(f"- **Networks:** {len(processed_data.get('networks', []))}")
            lines.append(f"- **Volumes:** {len(processed_data.get('volumes', []))}")
        
        lines.append("")
        
        # Validation issues
        if issues:
            lines.append("## Validation Issues")
            lines.append("")
            
            errors = [issue for issue in issues if issue.severity == "error"]
            warnings = [issue for issue in issues if issue.severity == "warning"]
            
            if errors:
                lines.append("### Errors")
                for error in errors:
                    lines.append(f"- **{error.path}:** {error.message}")
                lines.append("")
            
            if warnings:
                lines.append("### Warnings")
                for warning in warnings:
                    lines.append(f"- **{warning.path}:** {warning.message}")
                lines.append("")
        
        return "\n".join(lines)
    
    def convert_yaml_to_json(self, yaml_data: Dict[str, Any]) -> str:
        """Convert YAML data to JSON format"""
        return json.dumps(yaml_data, indent=2, ensure_ascii=False)
    
    def convert_json_to_yaml(self, json_data: Dict[str, Any]) -> str:
        """Convert JSON data to YAML format"""
        return yaml.dump(json_data, default_flow_style=False, allow_unicode=True)
    
    def compare_yaml_versions(self, old_data: Dict[str, Any], 
                            new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two versions of YAML configuration"""
        changes = {
            "added": [],
            "removed": [],
            "modified": [],
            "summary": {}
        }
        
        def compare_recursive(old, new, path=""):
            if isinstance(old, dict) and isinstance(new, dict):
                all_keys = set(old.keys()) | set(new.keys())
                
                for key in all_keys:
                    current_path = f"{path}.{key}" if path else key
                    
                    if key not in old:
                        changes["added"].append(current_path)
                    elif key not in new:
                        changes["removed"].append(current_path)
                    elif old[key] != new[key]:
                        if isinstance(old[key], (dict, list)) and isinstance(new[key], (dict, list)):
                            compare_recursive(old[key], new[key], current_path)
                        else:
                            changes["modified"].append({
                                "path": current_path,
                                "old_value": old[key],
                                "new_value": new[key]
                            })
            
            elif isinstance(old, list) and isinstance(new, list):
                if old != new:
                    changes["modified"].append({
                        "path": path,
                        "old_value": f"List with {len(old)} items",
                        "new_value": f"List with {len(new)} items"
                    })
        
        compare_recursive(old_data, new_data)
        
        changes["summary"] = {
            "added_count": len(changes["added"]),
            "removed_count": len(changes["removed"]),
            "modified_count": len(changes["modified"])
        }
        
        return changes