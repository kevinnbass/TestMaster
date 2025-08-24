"""
JSON Documentation Processor Module
Handles JSON-based documentation formats and API specifications
"""

import json
import re
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime


class JSONDocType(Enum):
    """Types of JSON documentation"""
    OPENAPI_SPEC = "openapi"
    JSON_SCHEMA = "json_schema"
    POSTMAN_COLLECTION = "postman_collection"
    NAVIGATION_CONFIG = "navigation_config"
    LOCALIZATION_FILE = "localization"
    API_RESPONSES = "api_responses"
    CONFIGURATION = "configuration"
    DATA_DICTIONARY = "data_dictionary"
    UNKNOWN = "unknown"


@dataclass
class JSONValidationResult:
    """JSON validation result"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class JSONDocumentationItem:
    """JSON documentation item"""
    path: str
    value: Any
    data_type: str
    description: str = ""
    required: bool = False
    default_value: Any = None
    examples: List[Any] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIEndpointInfo:
    """API endpoint information extracted from JSON"""
    name: str
    method: str
    url: str
    description: str = ""
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Any] = field(default_factory=dict)
    auth_required: bool = False
    tags: List[str] = field(default_factory=list)


class JSONDocsProcessor:
    """Processor for JSON-based documentation formats"""
    
    def __init__(self):
        self.validators = {
            JSONDocType.OPENAPI_SPEC: self._validate_openapi,
            JSONDocType.JSON_SCHEMA: self._validate_json_schema,
            JSONDocType.POSTMAN_COLLECTION: self._validate_postman_collection,
            JSONDocType.NAVIGATION_CONFIG: self._validate_navigation_config
        }
        
        self.processors = {
            JSONDocType.OPENAPI_SPEC: self._process_openapi_spec,
            JSONDocType.JSON_SCHEMA: self._process_json_schema,
            JSONDocType.POSTMAN_COLLECTION: self._process_postman_collection,
            JSONDocType.NAVIGATION_CONFIG: self._process_navigation_config,
            JSONDocType.LOCALIZATION_FILE: self._process_localization_file
        }
    
    def load_json_file(self, file_path: Path) -> tuple[Optional[Dict[str, Any]], List[str]]:
        """Load and parse JSON file with error handling"""
        errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Try to parse JSON
            data = json.loads(content)
            return data, errors
            
        except json.JSONDecodeError as e:
            errors.append(f"JSON parsing error at line {e.lineno}: {e.msg}")
            return None, errors
        except FileNotFoundError:
            errors.append(f"File not found: {file_path}")
            return None, errors
        except Exception as e:
            errors.append(f"Unexpected error: {str(e)}")
            return None, errors
    
    def identify_json_type(self, data: Dict[str, Any]) -> JSONDocType:
        """Identify the type of JSON document"""
        if not data or not isinstance(data, dict):
            return JSONDocType.UNKNOWN
        
        # Check for OpenAPI specification
        if "openapi" in data or "swagger" in data:
            return JSONDocType.OPENAPI_SPEC
        
        # Check for JSON Schema
        if "$schema" in data and "json-schema" in str(data.get("$schema", "")):
            return JSONDocType.JSON_SCHEMA
        
        # Check for Postman Collection
        if "info" in data and "item" in data and "schema" in data.get("info", {}):
            schema_url = data["info"].get("schema", "")
            if "postman" in schema_url:
                return JSONDocType.POSTMAN_COLLECTION
        
        # Check for navigation configuration
        nav_indicators = ["navigation", "nav", "menu", "sidebar", "pages"]
        if any(indicator in data for indicator in nav_indicators):
            return JSONDocType.NAVIGATION_CONFIG
        
        # Check for localization file
        if self._is_localization_file(data):
            return JSONDocType.LOCALIZATION_FILE
        
        # Check for API responses
        if "data" in data and ("meta" in data or "pagination" in data):
            return JSONDocType.API_RESPONSES
        
        # Check for data dictionary
        if self._is_data_dictionary(data):
            return JSONDocType.DATA_DICTIONARY
        
        return JSONDocType.CONFIGURATION
    
    def _is_localization_file(self, data: Dict[str, Any]) -> bool:
        """Check if JSON is a localization file"""
        # Look for language codes or translation patterns
        language_patterns = [
            r'^[a-z]{2}(-[A-Z]{2})?$',  # en, en-US, pt-BR
            r'^[a-z]{2}_[A-Z]{2}$'      # en_US, pt_BR
        ]
        
        keys = list(data.keys())
        if len(keys) < 5:  # Small files might be language-specific
            return False
            
        # Check if values are mostly strings (translations)
        string_values = sum(1 for v in data.values() if isinstance(v, str))
        total_values = len(data.values())
        
        if total_values > 0 and (string_values / total_values) > 0.7:
            return True
        
        # Check for language code keys
        for key in keys[:10]:  # Check first 10 keys
            if any(re.match(pattern, key) for pattern in language_patterns):
                return True
        
        return False
    
    def _is_data_dictionary(self, data: Dict[str, Any]) -> bool:
        """Check if JSON is a data dictionary"""
        # Look for field definitions
        if "fields" in data or "properties" in data or "columns" in data:
            return True
        
        # Check if all values have dictionary-like structure with type/description
        sample_values = list(data.values())[:5]
        
        for value in sample_values:
            if isinstance(value, dict):
                dict_indicators = ["type", "description", "format", "required"]
                if any(indicator in value for indicator in dict_indicators):
                    return True
        
        return False
    
    def validate_json_document(self, data: Dict[str, Any], 
                              json_type: JSONDocType) -> JSONValidationResult:
        """Validate JSON document based on its type"""
        if json_type in self.validators:
            return self.validators[json_type](data)
        
        return JSONValidationResult(
            is_valid=True,
            warnings=[f"No specific validation available for {json_type.value}"]
        )
    
    def _validate_openapi(self, data: Dict[str, Any]) -> JSONValidationResult:
        """Validate OpenAPI specification"""
        result = JSONValidationResult(is_valid=True)
        
        # Check required fields
        required_fields = ["openapi", "info", "paths"]
        for field in required_fields:
            if field not in data:
                result.errors.append(f"Missing required field: {field}")
                result.is_valid = False
        
        # Validate info section
        if "info" in data:
            info = data["info"]
            if not info.get("title"):
                result.errors.append("Missing title in info section")
                result.is_valid = False
            
            if not info.get("version"):
                result.errors.append("Missing version in info section")
                result.is_valid = False
        
        # Validate OpenAPI version
        if "openapi" in data:
            version = data["openapi"]
            if not re.match(r"3\.\d+\.\d+", str(version)):
                result.warnings.append(f"OpenAPI version '{version}' format might be invalid")
        
        # Check paths
        paths = data.get("paths", {})
        if not paths:
            result.warnings.append("No API paths defined")
        else:
            for path, path_obj in paths.items():
                if not isinstance(path_obj, dict):
                    result.errors.append(f"Invalid path object for '{path}'")
                    result.is_valid = False
                    continue
                
                http_methods = {"get", "post", "put", "delete", "patch", "head", "options"}
                if not any(method in path_obj for method in http_methods):
                    result.warnings.append(f"Path '{path}' has no HTTP method operations")
        
        return result
    
    def _validate_json_schema(self, data: Dict[str, Any]) -> JSONValidationResult:
        """Validate JSON Schema"""
        result = JSONValidationResult(is_valid=True)
        
        # Check for schema identifier
        if "$schema" not in data:
            result.warnings.append("Missing $schema identifier")
        
        # Check for type definition
        if "type" not in data:
            result.warnings.append("Missing type definition")
        
        # Validate properties if it's an object schema
        if data.get("type") == "object":
            if "properties" not in data:
                result.warnings.append("Object schema should have properties")
            else:
                properties = data["properties"]
                for prop_name, prop_def in properties.items():
                    if not isinstance(prop_def, dict):
                        result.errors.append(f"Invalid property definition for '{prop_name}'")
                        result.is_valid = False
        
        return result
    
    def _validate_postman_collection(self, data: Dict[str, Any]) -> JSONValidationResult:
        """Validate Postman collection"""
        result = JSONValidationResult(is_valid=True)
        
        # Check required fields
        if "info" not in data:
            result.errors.append("Missing info section")
            result.is_valid = False
        
        if "item" not in data:
            result.errors.append("Missing item array")
            result.is_valid = False
        
        # Validate info section
        if "info" in data:
            info = data["info"]
            if not info.get("name"):
                result.warnings.append("Missing collection name")
            
            if not info.get("schema"):
                result.warnings.append("Missing schema URL in info")
        
        # Validate items
        if "item" in data and isinstance(data["item"], list):
            for i, item in enumerate(data["item"]):
                if not isinstance(item, dict):
                    result.errors.append(f"Invalid item at index {i}")
                    result.is_valid = False
                    continue
                
                if "request" not in item and "item" not in item:
                    result.warnings.append(f"Item at index {i} has no request or sub-items")
        
        return result
    
    def _validate_navigation_config(self, data: Dict[str, Any]) -> JSONValidationResult:
        """Validate navigation configuration"""
        result = JSONValidationResult(is_valid=True)
        
        # Check for navigation structure
        nav_fields = ["navigation", "nav", "menu", "sidebar"]
        has_nav = any(field in data for field in nav_fields)
        
        if not has_nav:
            result.warnings.append("No recognizable navigation structure found")
        
        # Validate navigation items
        for nav_field in nav_fields:
            if nav_field in data:
                nav_data = data[nav_field]
                if isinstance(nav_data, dict):
                    if "pages" in nav_data or "items" in nav_data or "links" in nav_data:
                        result.suggestions.append("Navigation structure looks valid")
                elif isinstance(nav_data, list):
                    if len(nav_data) == 0:
                        result.warnings.append(f"Empty {nav_field} array")
        
        return result
    
    def process_json_document(self, data: Dict[str, Any], 
                             json_type: JSONDocType) -> Dict[str, Any]:
        """Process JSON document based on its type"""
        if json_type in self.processors:
            return self.processors[json_type](data)
        
        return {"processed": False, "reason": f"No processor for {json_type.value}"}
    
    def _process_openapi_spec(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process OpenAPI specification"""
        result = {
            "type": "openapi",
            "version": data.get("openapi", ""),
            "info": data.get("info", {}),
            "servers": data.get("servers", []),
            "paths_count": len(data.get("paths", {})),
            "components": self._analyze_components(data.get("components", {})),
            "security": data.get("security", []),
            "tags": [tag.get("name") for tag in data.get("tags", [])],
            "endpoints": self._extract_endpoints(data)
        }
        
        return result
    
    def _process_json_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process JSON Schema"""
        result = {
            "type": "json_schema",
            "schema_version": data.get("$schema", ""),
            "title": data.get("title", ""),
            "description": data.get("description", ""),
            "data_type": data.get("type", ""),
            "properties_count": len(data.get("properties", {})) if data.get("type") == "object" else 0,
            "required_fields": data.get("required", []),
            "definitions": list(data.get("definitions", {}).keys()),
            "structure": self._analyze_schema_structure(data)
        }
        
        return result
    
    def _process_postman_collection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Postman collection"""
        result = {
            "type": "postman_collection",
            "collection_info": data.get("info", {}),
            "auth": data.get("auth", {}),
            "variables": data.get("variable", []),
            "items_count": len(data.get("item", [])),
            "requests": self._extract_postman_requests(data),
            "folders": self._extract_postman_folders(data)
        }
        
        return result
    
    def _process_navigation_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process navigation configuration"""
        result = {
            "type": "navigation_config",
            "structure": self._analyze_navigation_structure(data),
            "languages": self._extract_navigation_languages(data),
            "total_pages": self._count_navigation_pages(data),
            "has_hierarchical_structure": self._has_hierarchical_nav(data)
        }
        
        return result
    
    def _process_localization_file(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process localization file"""
        result = {
            "type": "localization",
            "total_keys": len(data),
            "nested_keys": self._count_nested_keys(data),
            "translation_coverage": self._analyze_translation_coverage(data),
            "key_categories": self._categorize_translation_keys(data)
        }
        
        return result
    
    def _analyze_components(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze OpenAPI components"""
        return {
            "schemas": len(components.get("schemas", {})),
            "responses": len(components.get("responses", {})),
            "parameters": len(components.get("parameters", {})),
            "security_schemes": len(components.get("securitySchemes", {})),
            "headers": len(components.get("headers", {}))
        }
    
    def _extract_endpoints(self, data: Dict[str, Any]) -> List[APIEndpointInfo]:
        """Extract API endpoints from OpenAPI spec"""
        endpoints = []
        
        paths = data.get("paths", {})
        for path, path_obj in paths.items():
            if not isinstance(path_obj, dict):
                continue
                
            for method, operation in path_obj.items():
                if method.lower() in ["get", "post", "put", "delete", "patch", "head", "options"]:
                    endpoint = APIEndpointInfo(
                        name=operation.get("operationId", f"{method.upper()} {path}"),
                        method=method.upper(),
                        url=path,
                        description=operation.get("description", ""),
                        parameters=operation.get("parameters", []),
                        request_body=operation.get("requestBody"),
                        responses=operation.get("responses", {}),
                        tags=operation.get("tags", [])
                    )
                    
                    # Check if authentication is required
                    if "security" in operation or "security" in data:
                        endpoint.auth_required = True
                    
                    endpoints.append(endpoint)
        
        return endpoints
    
    def _analyze_schema_structure(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze JSON schema structure"""
        structure = {
            "depth": self._calculate_schema_depth(schema),
            "complexity": self._calculate_schema_complexity(schema),
            "has_references": "$ref" in str(schema),
            "has_definitions": "definitions" in schema,
            "validation_keywords": self._extract_validation_keywords(schema)
        }
        
        return structure
    
    def _extract_postman_requests(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract requests from Postman collection"""
        requests = []
        
        def extract_from_items(items):
            for item in items:
                if "request" in item:
                    request = item["request"]
                    requests.append({
                        "name": item.get("name", ""),
                        "method": request.get("method", ""),
                        "url": self._format_postman_url(request.get("url", {})),
                        "headers": request.get("header", []),
                        "body": request.get("body", {}),
                        "auth": request.get("auth", {})
                    })
                elif "item" in item:  # Folder with sub-items
                    extract_from_items(item["item"])
        
        items = data.get("item", [])
        if isinstance(items, list):
            extract_from_items(items)
        
        return requests
    
    def _extract_postman_folders(self, data: Dict[str, Any]) -> List[str]:
        """Extract folder names from Postman collection"""
        folders = []
        
        def extract_folders(items):
            for item in items:
                if "item" in item and "request" not in item:
                    folders.append(item.get("name", ""))
                    if isinstance(item["item"], list):
                        extract_folders(item["item"])
        
        items = data.get("item", [])
        if isinstance(items, list):
            extract_folders(items)
        
        return folders
    
    def _format_postman_url(self, url_obj: Union[str, Dict[str, Any]]) -> str:
        """Format Postman URL object to string"""
        if isinstance(url_obj, str):
            return url_obj
        elif isinstance(url_obj, dict):
            if "raw" in url_obj:
                return url_obj["raw"]
            elif "host" in url_obj and "path" in url_obj:
                host = "/".join(url_obj["host"]) if isinstance(url_obj["host"], list) else str(url_obj["host"])
                path = "/".join(url_obj["path"]) if isinstance(url_obj["path"], list) else str(url_obj["path"])
                return f"{host}/{path}"
        
        return str(url_obj)
    
    def _analyze_navigation_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze navigation structure"""
        structure = {}
        
        # Look for different navigation patterns
        nav_keys = ["navigation", "nav", "menu", "sidebar", "pages"]
        
        for key in nav_keys:
            if key in data:
                nav_data = data[key]
                structure[key] = self._analyze_nav_data(nav_data)
        
        return structure
    
    def _analyze_nav_data(self, nav_data: Any) -> Dict[str, Any]:
        """Analyze navigation data structure"""
        analysis = {"type": type(nav_data).__name__}
        
        if isinstance(nav_data, dict):
            analysis.update({
                "keys": list(nav_data.keys()),
                "has_languages": "languages" in nav_data,
                "has_tabs": "tabs" in nav_data,
                "has_groups": "groups" in nav_data
            })
        elif isinstance(nav_data, list):
            analysis.update({
                "length": len(nav_data),
                "item_types": list(set(type(item).__name__ for item in nav_data))
            })
        
        return analysis
    
    def _extract_navigation_languages(self, data: Dict[str, Any]) -> List[str]:
        """Extract supported languages from navigation config"""
        languages = []
        
        # Look for language configurations
        if "navigation" in data and "languages" in data["navigation"]:
            lang_configs = data["navigation"]["languages"]
            if isinstance(lang_configs, list):
                for lang_config in lang_configs:
                    if isinstance(lang_config, dict) and "language" in lang_config:
                        languages.append(lang_config["language"])
        
        # Look for language-specific sections
        language_patterns = [r'^[a-z]{2}(-[A-Z]{2})?$', r'^[a-z]{2}_[A-Z]{2}$']
        
        for key in data.keys():
            if any(re.match(pattern, key) for pattern in language_patterns):
                languages.append(key)
        
        return list(set(languages))
    
    def _count_navigation_pages(self, data: Dict[str, Any]) -> int:
        """Count total pages in navigation"""
        def count_pages(obj):
            if isinstance(obj, dict):
                if "pages" in obj:
                    pages = obj["pages"]
                    if isinstance(pages, list):
                        return len(pages)
                
                # Recursively count in nested structures
                total = 0
                for value in obj.values():
                    total += count_pages(value)
                return total
                
            elif isinstance(obj, list):
                return sum(count_pages(item) for item in obj)
            
            return 0
        
        return count_pages(data)
    
    def _has_hierarchical_nav(self, data: Dict[str, Any]) -> bool:
        """Check if navigation has hierarchical structure"""
        def check_hierarchy(obj, level=0):
            if level > 2:  # More than 2 levels indicates hierarchy
                return True
                
            if isinstance(obj, dict):
                # Check for nested groups or folders
                if any(key in obj for key in ["groups", "folders", "children", "items"]):
                    return True
                
                for value in obj.values():
                    if check_hierarchy(value, level + 1):
                        return True
                        
            elif isinstance(obj, list):
                for item in obj:
                    if check_hierarchy(item, level + 1):
                        return True
            
            return False
        
        return check_hierarchy(data)
    
    def _count_nested_keys(self, data: Dict[str, Any]) -> int:
        """Count nested keys in JSON structure"""
        def count_keys(obj):
            if isinstance(obj, dict):
                return len(obj) + sum(count_keys(value) for value in obj.values())
            elif isinstance(obj, list):
                return sum(count_keys(item) for item in obj if isinstance(item, (dict, list)))
            return 0
        
        return count_keys(data)
    
    def _analyze_translation_coverage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze translation coverage in localization file"""
        total_values = 0
        empty_values = 0
        string_values = 0
        
        def analyze_values(obj):
            nonlocal total_values, empty_values, string_values
            
            if isinstance(obj, dict):
                for value in obj.values():
                    analyze_values(value)
            elif isinstance(obj, list):
                for item in obj:
                    analyze_values(item)
            else:
                total_values += 1
                if isinstance(obj, str):
                    string_values += 1
                    if not obj.strip():
                        empty_values += 1
                elif obj is None or obj == "":
                    empty_values += 1
        
        analyze_values(data)
        
        return {
            "total_values": total_values,
            "empty_values": empty_values,
            "string_values": string_values,
            "coverage_percentage": ((total_values - empty_values) / total_values * 100) if total_values > 0 else 0
        }
    
    def _categorize_translation_keys(self, data: Dict[str, Any]) -> Dict[str, int]:
        """Categorize translation keys by common prefixes"""
        categories = {}
        
        def categorize_keys(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_key = f"{prefix}.{key}" if prefix else key
                    
                    if isinstance(value, (dict, list)):
                        categorize_keys(value, current_key)
                    else:
                        # Extract category from key (first part before dot or underscore)
                        category = key.split('.')[0].split('_')[0]
                        categories[category] = categories.get(category, 0) + 1
        
        categorize_keys(data)
        return categories
    
    def _calculate_schema_depth(self, schema: Dict[str, Any], depth=0) -> int:
        """Calculate maximum depth of JSON schema"""
        max_depth = depth
        
        if isinstance(schema, dict):
            for value in schema.values():
                if isinstance(value, dict):
                    max_depth = max(max_depth, self._calculate_schema_depth(value, depth + 1))
        
        return max_depth
    
    def _calculate_schema_complexity(self, schema: Dict[str, Any]) -> int:
        """Calculate complexity score of JSON schema"""
        complexity = 0
        
        # Count properties
        if "properties" in schema:
            complexity += len(schema["properties"])
        
        # Add complexity for validation keywords
        validation_keywords = [
            "required", "enum", "minimum", "maximum", "pattern", "format",
            "minLength", "maxLength", "minItems", "maxItems"
        ]
        
        complexity += sum(1 for keyword in validation_keywords if keyword in schema)
        
        # Add complexity for nested schemas
        if "properties" in schema:
            for prop_schema in schema["properties"].values():
                if isinstance(prop_schema, dict):
                    complexity += self._calculate_schema_complexity(prop_schema) // 2
        
        return complexity
    
    def _extract_validation_keywords(self, schema: Dict[str, Any]) -> List[str]:
        """Extract validation keywords from JSON schema"""
        validation_keywords = [
            "type", "format", "pattern", "enum", "const",
            "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
            "minLength", "maxLength", "minItems", "maxItems",
            "uniqueItems", "required", "additionalProperties"
        ]
        
        found_keywords = []
        
        def extract_keywords(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in validation_keywords:
                        found_keywords.append(key)
                    if isinstance(value, dict):
                        extract_keywords(value)
        
        extract_keywords(schema)
        return list(set(found_keywords))
    
    def generate_json_documentation(self, data: Dict[str, Any], 
                                  json_type: JSONDocType) -> str:
        """Generate human-readable documentation from JSON"""
        doc_sections = []
        
        # Process the data
        processed = self.process_json_document(data, json_type)
        
        # Generate documentation based on type
        if json_type == JSONDocType.OPENAPI_SPEC:
            doc_sections.extend(self._generate_openapi_docs(processed))
        elif json_type == JSONDocType.JSON_SCHEMA:
            doc_sections.extend(self._generate_schema_docs(processed))
        elif json_type == JSONDocType.POSTMAN_COLLECTION:
            doc_sections.extend(self._generate_postman_docs(processed))
        elif json_type == JSONDocType.NAVIGATION_CONFIG:
            doc_sections.extend(self._generate_navigation_docs(processed))
        else:
            # Generic documentation
            doc_sections.extend(self._generate_generic_docs(data, processed))
        
        return "\n".join(doc_sections)
    
    def _generate_openapi_docs(self, processed: Dict[str, Any]) -> List[str]:
        """Generate OpenAPI documentation"""
        docs = []
        
        info = processed.get("info", {})
        docs.append(f"# {info.get('title', 'API Documentation')}")
        docs.append("")
        
        if info.get("description"):
            docs.append(info["description"])
            docs.append("")
        
        docs.append(f"**Version:** {info.get('version', 'Unknown')}")
        docs.append(f"**OpenAPI Version:** {processed.get('version', 'Unknown')}")
        docs.append("")
        
        # Server information
        servers = processed.get("servers", [])
        if servers:
            docs.append("## Servers")
            docs.append("")
            for server in servers:
                url = server.get("url", "")
                description = server.get("description", "")
                docs.append(f"- **{url}** - {description}")
            docs.append("")
        
        # Statistics
        docs.append("## API Statistics")
        docs.append("")
        docs.append(f"- **Total Paths:** {processed.get('paths_count', 0)}")
        
        components = processed.get("components", {})
        docs.append(f"- **Schemas:** {components.get('schemas', 0)}")
        docs.append(f"- **Security Schemes:** {components.get('security_schemes', 0)}")
        docs.append("")
        
        # Endpoints
        endpoints = processed.get("endpoints", [])
        if endpoints:
            docs.append("## Endpoints")
            docs.append("")
            
            # Group by tags
            by_tags = {}
            for endpoint in endpoints:
                tags = endpoint.tags or ["Default"]
                for tag in tags:
                    if tag not in by_tags:
                        by_tags[tag] = []
                    by_tags[tag].append(endpoint)
            
            for tag, tag_endpoints in by_tags.items():
                docs.append(f"### {tag}")
                docs.append("")
                
                for endpoint in tag_endpoints:
                    docs.append(f"#### {endpoint.method} {endpoint.url}")
                    if endpoint.description:
                        docs.append(endpoint.description)
                    docs.append("")
        
        return docs
    
    def _generate_schema_docs(self, processed: Dict[str, Any]) -> List[str]:
        """Generate JSON Schema documentation"""
        docs = []
        
        title = processed.get("title") or "JSON Schema"
        docs.append(f"# {title}")
        docs.append("")
        
        if processed.get("description"):
            docs.append(processed["description"])
            docs.append("")
        
        docs.append("## Schema Information")
        docs.append("")
        docs.append(f"- **Type:** {processed.get('data_type', 'Unknown')}")
        docs.append(f"- **Schema Version:** {processed.get('schema_version', 'Unknown')}")
        
        if processed.get("properties_count"):
            docs.append(f"- **Properties:** {processed['properties_count']}")
        
        structure = processed.get("structure", {})
        if structure:
            docs.append(f"- **Depth:** {structure.get('depth', 0)}")
            docs.append(f"- **Complexity Score:** {structure.get('complexity', 0)}")
        
        docs.append("")
        
        # Required fields
        required_fields = processed.get("required_fields", [])
        if required_fields:
            docs.append("## Required Fields")
            docs.append("")
            for field in required_fields:
                docs.append(f"- {field}")
            docs.append("")
        
        return docs
    
    def _generate_postman_docs(self, processed: Dict[str, Any]) -> List[str]:
        """Generate Postman collection documentation"""
        docs = []
        
        collection_info = processed.get("collection_info", {})
        docs.append(f"# {collection_info.get('name', 'Postman Collection')}")
        docs.append("")
        
        if collection_info.get("description"):
            docs.append(collection_info["description"])
            docs.append("")
        
        docs.append("## Collection Statistics")
        docs.append("")
        docs.append(f"- **Total Items:** {processed.get('items_count', 0)}")
        docs.append(f"- **Folders:** {len(processed.get('folders', []))}")
        docs.append(f"- **Requests:** {len(processed.get('requests', []))}")
        docs.append("")
        
        # Variables
        variables = processed.get("variables", [])
        if variables:
            docs.append("## Collection Variables")
            docs.append("")
            for var in variables:
                key = var.get("key", "")
                value = var.get("value", "")
                docs.append(f"- **{key}:** {value}")
            docs.append("")
        
        # Requests summary
        requests = processed.get("requests", [])
        if requests:
            docs.append("## Requests")
            docs.append("")
            
            # Group by method
            by_method = {}
            for request in requests:
                method = request.get("method", "Unknown")
                if method not in by_method:
                    by_method[method] = []
                by_method[method].append(request)
            
            for method, method_requests in by_method.items():
                docs.append(f"### {method} Requests")
                docs.append("")
                
                for request in method_requests:
                    docs.append(f"- **{request.get('name', 'Unnamed')}:** `{request.get('url', '')}`")
                
                docs.append("")
        
        return docs
    
    def _generate_navigation_docs(self, processed: Dict[str, Any]) -> List[str]:
        """Generate navigation configuration documentation"""
        docs = []
        
        docs.append("# Navigation Configuration")
        docs.append("")
        
        docs.append("## Configuration Summary")
        docs.append("")
        docs.append(f"- **Total Pages:** {processed.get('total_pages', 0)}")
        docs.append(f"- **Languages:** {len(processed.get('languages', []))}")
        docs.append(f"- **Hierarchical Structure:** {'Yes' if processed.get('has_hierarchical_structure') else 'No'}")
        docs.append("")
        
        # Languages
        languages = processed.get("languages", [])
        if languages:
            docs.append("## Supported Languages")
            docs.append("")
            for lang in languages:
                docs.append(f"- {lang}")
            docs.append("")
        
        # Structure analysis
        structure = processed.get("structure", {})
        if structure:
            docs.append("## Structure Analysis")
            docs.append("")
            for key, analysis in structure.items():
                docs.append(f"### {key}")
                if isinstance(analysis, dict):
                    for prop, value in analysis.items():
                        docs.append(f"- **{prop}:** {value}")
                docs.append("")
        
        return docs
    
    def _generate_generic_docs(self, data: Dict[str, Any], processed: Dict[str, Any]) -> List[str]:
        """Generate generic JSON documentation"""
        docs = []
        
        docs.append("# JSON Document")
        docs.append("")
        
        docs.append("## Document Information")
        docs.append("")
        docs.append(f"- **Type:** {processed.get('type', 'Unknown')}")
        docs.append(f"- **Top-level Keys:** {len(data)}")
        docs.append("")
        
        # Show structure overview
        docs.append("## Structure Overview")
        docs.append("")
        
        for key, value in data.items():
            value_type = type(value).__name__
            if isinstance(value, dict):
                docs.append(f"- **{key}:** Object with {len(value)} properties")
            elif isinstance(value, list):
                docs.append(f"- **{key}:** Array with {len(value)} items")
            else:
                docs.append(f"- **{key}:** {value_type}")
        
        docs.append("")
        
        return docs