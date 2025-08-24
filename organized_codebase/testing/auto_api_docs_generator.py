"""
Auto API Documentation Generator

Generates comprehensive API documentation with metadata extraction and 
cross-platform linking based on AutoGen's DocFX and Sphinx patterns.
"""

import os
import ast
import re
import json
import inspect
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ApiDocFormat(Enum):
    """Supported API documentation formats."""
    DOCFX = "docfx"
    SPHINX = "sphinx"
    MARKDOWN = "markdown"
    OPENAPI = "openapi"
    JSDoc = "jsdoc"


class LanguageType(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    CSHARP = "csharp"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"


@dataclass
class ApiMethod:
    """Represents an API method or function."""
    name: str
    signature: str
    description: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    returns: Dict[str, str] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    exceptions: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    decorators: List[str] = field(default_factory=list)
    access_level: str = "public"


@dataclass  
class ApiClass:
    """Represents an API class or type."""
    name: str
    namespace: str
    description: str
    methods: List[ApiMethod] = field(default_factory=list)
    properties: List[Dict[str, Any]] = field(default_factory=list)
    inheritance: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_level: str = "public"


@dataclass
class ApiNamespace:
    """Represents an API namespace or module."""
    name: str
    description: str
    classes: List[ApiClass] = field(default_factory=list)
    functions: List[ApiMethod] = field(default_factory=list)
    constants: List[Dict[str, Any]] = field(default_factory=list)
    submodules: List['ApiNamespace'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutoApiDocsGenerator:
    """
    Auto API documentation generator inspired by AutoGen's DocFX 
    and Sphinx patterns for metadata extraction and cross-platform linking.
    """
    
    def __init__(self, language: LanguageType, output_format: ApiDocFormat = ApiDocFormat.MARKDOWN):
        """Initialize auto API docs generator."""
        self.language = language
        self.output_format = output_format
        self.namespaces = {}
        self.cross_references = {}
        self.templates = self._load_templates()
        self.metadata_extractors = self._setup_extractors()
        logger.info(f"Auto API docs generator initialized for {language.value}")
        
    def extract_from_source(self, source_path: str) -> List[ApiNamespace]:
        """Extract API information from source code."""
        source_path = Path(source_path)
        namespaces = []
        
        if source_path.is_file():
            # Single file
            namespace = self._extract_from_file(source_path)
            if namespace:
                namespaces.append(namespace)
        else:
            # Directory
            namespaces = self._extract_from_directory(source_path)
            
        # Build cross-references
        self._build_cross_references(namespaces)
        
        return namespaces
        
    def _extract_from_file(self, file_path: Path) -> Optional[ApiNamespace]:
        """Extract API info from single file."""
        if self.language == LanguageType.PYTHON:
            return self._extract_python_file(file_path)
        elif self.language == LanguageType.CSHARP:
            return self._extract_csharp_file(file_path)
        elif self.language == LanguageType.JAVASCRIPT:
            return self._extract_javascript_file(file_path)
        else:
            logger.warning(f"Language {self.language.value} not yet supported")
            return None
            
    def _extract_python_file(self, file_path: Path) -> Optional[ApiNamespace]:
        """Extract Python API information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            module_name = file_path.stem
            
            # Extract module docstring
            module_doc = ast.get_docstring(tree) or ""
            
            namespace = ApiNamespace(
                name=module_name,
                description=module_doc,
                metadata={"file_path": str(file_path)}
            )
            
            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    api_class = self._extract_python_class(node, content)
                    namespace.classes.append(api_class)
                elif isinstance(node, ast.FunctionDef):
                    if not self._is_inside_class(node, tree):
                        api_function = self._extract_python_function(node, content)
                        namespace.functions.append(api_function)
                        
            return namespace
            
        except Exception as e:
            logger.error(f"Error extracting from Python file {file_path}: {e}")
            return None
            
    def _extract_python_class(self, class_node: ast.ClassDef, source_code: str) -> ApiClass:
        """Extract Python class information."""
        class_name = class_node.name
        docstring = ast.get_docstring(class_node) or ""
        
        # Extract inheritance
        inheritance = []
        for base in class_node.bases:
            if isinstance(base, ast.Name):
                inheritance.append(base.id)
            elif isinstance(base, ast.Attribute):
                inheritance.append(self._resolve_attribute_name(base))
                
        api_class = ApiClass(
            name=class_name,
            namespace="",  # Set by parent
            description=docstring,
            inheritance=inheritance,
            access_level=self._determine_access_level(class_name)
        )
        
        # Extract methods and properties
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                method = self._extract_python_function(node, source_code)
                api_class.methods.append(method)
            elif isinstance(node, ast.Assign):
                # Look for class properties/attributes
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        prop = {
                            "name": target.id,
                            "type": "Any",  # Would need type analysis
                            "description": "",
                            "access_level": self._determine_access_level(target.id)
                        }
                        api_class.properties.append(prop)
                        
        return api_class
        
    def _extract_python_function(self, func_node: ast.FunctionDef, source_code: str) -> ApiMethod:
        """Extract Python function information."""
        func_name = func_node.name
        docstring = ast.get_docstring(func_node) or ""
        
        # Build signature
        signature = self._build_python_signature(func_node)
        
        # Extract decorators
        decorators = []
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(self._resolve_attribute_name(decorator))
                
        # Parse docstring for parameters and returns
        params, returns, examples, exceptions = self._parse_python_docstring(docstring)
        
        return ApiMethod(
            name=func_name,
            signature=signature,
            description=docstring.split('\n')[0] if docstring else "",
            parameters=params,
            returns=returns,
            examples=examples,
            exceptions=exceptions,
            decorators=decorators,
            access_level=self._determine_access_level(func_name)
        )
        
    def _build_python_signature(self, func_node: ast.FunctionDef) -> str:
        """Build function signature string."""
        args = []
        
        # Handle different argument types
        for arg in func_node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    arg_str += f": {arg.annotation.id}"
                elif isinstance(arg.annotation, ast.Constant):
                    arg_str += f": {arg.annotation.value}"
            args.append(arg_str)
            
        # Handle defaults
        if func_node.args.defaults:
            n_defaults = len(func_node.args.defaults)
            for i, default in enumerate(func_node.args.defaults):
                arg_index = len(args) - n_defaults + i
                if isinstance(default, ast.Constant):
                    args[arg_index] += f" = {default.value!r}"
                    
        signature = f"{func_node.name}({', '.join(args)})"
        
        # Add return annotation
        if func_node.returns:
            if isinstance(func_node.returns, ast.Name):
                signature += f" -> {func_node.returns.id}"
                
        return signature
        
    def _parse_python_docstring(self, docstring: str) -> Tuple[List[Dict], Dict, List[str], List[Dict]]:
        """Parse Python docstring for structured information."""
        if not docstring:
            return [], {}, [], []
            
        lines = docstring.split('\n')
        params = []
        returns = {}
        examples = []
        exceptions = []
        
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            
            # Detect sections
            if line.lower().startswith(('args:', 'arguments:', 'parameters:')):
                current_section = 'params'
                continue
            elif line.lower().startswith(('returns:', 'return:')):
                current_section = 'returns'
                continue
            elif line.lower().startswith(('examples:', 'example:')):
                current_section = 'examples'
                continue
            elif line.lower().startswith(('raises:', 'exceptions:')):
                current_section = 'exceptions'
                continue
                
            # Parse content based on section
            if current_section == 'params' and line:
                # Look for parameter format: param_name (type): description
                param_match = re.match(r'(\w+)\s*\(([^)]+)\):\s*(.+)', line)
                if param_match:
                    params.append({
                        "name": param_match.group(1),
                        "type": param_match.group(2),
                        "description": param_match.group(3)
                    })
            elif current_section == 'returns' and line:
                # Parse return information
                return_match = re.match(r'([^:]+):\s*(.+)', line)
                if return_match:
                    returns = {
                        "type": return_match.group(1),
                        "description": return_match.group(2)
                    }
            elif current_section == 'examples' and line:
                if line.startswith('>>>') or line.startswith('...'):
                    examples.append(line)
            elif current_section == 'exceptions' and line:
                exc_match = re.match(r'(\w+):\s*(.+)', line)
                if exc_match:
                    exceptions.append({
                        "type": exc_match.group(1),
                        "description": exc_match.group(2)
                    })
                    
        return params, returns, examples, exceptions
        
    def generate_documentation(self, namespaces: List[ApiNamespace]) -> Dict[str, str]:
        """Generate API documentation in specified format."""
        if self.output_format == ApiDocFormat.DOCFX:
            return self._generate_docfx_docs(namespaces)
        elif self.output_format == ApiDocFormat.SPHINX:
            return self._generate_sphinx_docs(namespaces)
        elif self.output_format == ApiDocFormat.MARKDOWN:
            return self._generate_markdown_docs(namespaces)
        elif self.output_format == ApiDocFormat.OPENAPI:
            return self._generate_openapi_docs(namespaces)
        else:
            return self._generate_markdown_docs(namespaces)
            
    def _generate_markdown_docs(self, namespaces: List[ApiNamespace]) -> Dict[str, str]:
        """Generate Markdown API documentation."""
        docs = {}
        
        # Generate index
        docs["index.md"] = self._generate_api_index(namespaces)
        
        # Generate documentation for each namespace
        for namespace in namespaces:
            filename = f"{namespace.name}.md"
            docs[filename] = self._generate_namespace_markdown(namespace)
            
            # Generate class documentation
            for api_class in namespace.classes:
                class_filename = f"{namespace.name}.{api_class.name}.md"
                docs[class_filename] = self._generate_class_markdown(api_class, namespace)
                
        return docs
        
    def _generate_namespace_markdown(self, namespace: ApiNamespace) -> str:
        """Generate Markdown for namespace."""
        lines = [
            f"# {namespace.name}",
            "",
            namespace.description,
            "",
            "## Classes",
            ""
        ]
        
        # List classes
        for api_class in namespace.classes:
            lines.extend([
                f"### [{api_class.name}]({namespace.name}.{api_class.name}.md)",
                "",
                api_class.description.split('\n')[0] if api_class.description else "No description available.",
                ""
            ])
            
        # List functions
        if namespace.functions:
            lines.extend([
                "## Functions",
                ""
            ])
            
            for func in namespace.functions:
                lines.extend([
                    f"### {func.name}",
                    "",
                    f"```{self.language.value}",
                    func.signature,
                    "```",
                    "",
                    func.description,
                    ""
                ])
                
                # Add parameters
                if func.parameters:
                    lines.extend([
                        "**Parameters:**",
                        ""
                    ])
                    
                    for param in func.parameters:
                        lines.append(f"- `{param['name']}` ({param.get('type', 'Any')}): {param.get('description', '')}")
                        
                    lines.append("")
                    
                # Add return info
                if func.returns:
                    lines.extend([
                        "**Returns:**",
                        "",
                        f"- {func.returns.get('type', 'Any')}: {func.returns.get('description', '')}",
                        ""
                    ])
                    
        return "\n".join(lines)
        
    def _generate_class_markdown(self, api_class: ApiClass, namespace: ApiNamespace) -> str:
        """Generate Markdown for class."""
        lines = [
            f"# {api_class.name}",
            "",
            f"**Namespace:** {namespace.name}",
            "",
            api_class.description,
            ""
        ]
        
        # Inheritance info
        if api_class.inheritance:
            lines.extend([
                "**Inherits from:**",
                ""
            ])
            for base in api_class.inheritance:
                lines.append(f"- {base}")
            lines.append("")
            
        # Properties
        if api_class.properties:
            lines.extend([
                "## Properties",
                ""
            ])
            
            for prop in api_class.properties:
                lines.extend([
                    f"### {prop['name']}",
                    "",
                    f"**Type:** {prop.get('type', 'Any')}",
                    "",
                    prop.get('description', 'No description available.'),
                    ""
                ])
                
        # Methods
        if api_class.methods:
            lines.extend([
                "## Methods",
                ""
            ])
            
            for method in api_class.methods:
                lines.extend([
                    f"### {method.name}",
                    "",
                    f"```{self.language.value}",
                    method.signature,
                    "```",
                    "",
                    method.description,
                    ""
                ])
                
                # Parameters
                if method.parameters:
                    lines.extend([
                        "**Parameters:**",
                        ""
                    ])
                    
                    for param in method.parameters:
                        lines.append(f"- `{param['name']}` ({param.get('type', 'Any')}): {param.get('description', '')}")
                        
                    lines.append("")
                    
                # Returns
                if method.returns:
                    lines.extend([
                        "**Returns:**",
                        "",
                        f"- {method.returns.get('type', 'Any')}: {method.returns.get('description', '')}",
                        ""
                    ])
                    
        return "\n".join(lines)
        
    def _generate_api_index(self, namespaces: List[ApiNamespace]) -> str:
        """Generate API index page."""
        lines = [
            "# API Reference",
            "",
            "Complete API documentation for all modules and classes.",
            "",
            "## Namespaces",
            ""
        ]
        
        for namespace in namespaces:
            lines.extend([
                f"### [{namespace.name}]({namespace.name}.md)",
                "",
                namespace.description.split('\n')[0] if namespace.description else "No description available.",
                "",
                f"- **Classes:** {len(namespace.classes)}",
                f"- **Functions:** {len(namespace.functions)}",
                ""
            ])
            
        return "\n".join(lines)
        
    def _generate_docfx_docs(self, namespaces: List[ApiNamespace]) -> Dict[str, str]:
        """Generate DocFX format documentation."""
        docs = {}
        
        # Generate docfx.json config
        docs["docfx.json"] = json.dumps({
            "metadata": [{
                "src": [{"files": ["**/*.cs"], "exclude": ["**/bin/**", "**/obj/**"]}],
                "dest": "api",
                "disableGitFeatures": False,
                "disableDefaultFilter": False
            }],
            "build": {
                "content": [{"files": ["api/**.yml", "api/index.md"]}],
                "resource": [{"files": ["images/**"]}],
                "overwrite": [{"files": ["apidoc/**.md"], "exclude": ["obj/**", "_site/**"]}],
                "dest": "_site",
                "globalMetadataFiles": [],
                "fileMetadataFiles": [],
                "template": ["default"],
                "postProcessors": [],
                "markdownEngineName": "markdig",
                "noLangKeyword": False,
                "keepFileLink": False,
                "cleanupCacheHistory": False,
                "disableGitFeatures": False
            }
        }, indent=2)
        
        return docs
        
    def create_cross_platform_links(self, platform_mappings: Dict[str, str]) -> Dict[str, Any]:
        """Create cross-platform API documentation links."""
        cross_links = {
            "platform_mappings": platform_mappings,
            "equivalent_apis": {},
            "migration_guides": {}
        }
        
        # Build equivalent API mappings
        for platform, base_path in platform_mappings.items():
            cross_links["equivalent_apis"][platform] = {
                "base_url": base_path,
                "namespaces": {}
            }
            
        return cross_links
        
    def _extract_from_directory(self, directory: Path) -> List[ApiNamespace]:
        """Extract API info from entire directory."""
        namespaces = []
        
        if self.language == LanguageType.PYTHON:
            pattern = "**/*.py"
        elif self.language == LanguageType.CSHARP:
            pattern = "**/*.cs"
        elif self.language == LanguageType.JAVASCRIPT:
            pattern = "**/*.js"
        else:
            pattern = "**/*"
            
        for file_path in directory.rglob(pattern):
            if file_path.is_file():
                namespace = self._extract_from_file(file_path)
                if namespace:
                    namespaces.append(namespace)
                    
        return namespaces
        
    def _build_cross_references(self, namespaces: List[ApiNamespace]) -> None:
        """Build cross-references between API elements."""
        # Build lookup tables
        all_classes = {}
        all_functions = {}
        
        for namespace in namespaces:
            for api_class in namespace.classes:
                full_name = f"{namespace.name}.{api_class.name}"
                all_classes[full_name] = api_class
                
            for func in namespace.functions:
                full_name = f"{namespace.name}.{func.name}"
                all_functions[full_name] = func
                
        self.cross_references = {
            "classes": all_classes,
            "functions": all_functions
        }
        
    def _setup_extractors(self) -> Dict[LanguageType, Any]:
        """Setup language-specific extractors."""
        return {
            LanguageType.PYTHON: self._extract_python_file,
            LanguageType.CSHARP: self._extract_csharp_file,
            LanguageType.JAVASCRIPT: self._extract_javascript_file
        }
        
    def _extract_csharp_file(self, file_path: Path) -> Optional[ApiNamespace]:
        """Extract C# API information (placeholder)."""
        # Would implement C# parsing logic
        logger.warning("C# extraction not yet implemented")
        return None
        
    def _extract_javascript_file(self, file_path: Path) -> Optional[ApiNamespace]:
        """Extract JavaScript API information (placeholder)."""
        # Would implement JavaScript parsing logic  
        logger.warning("JavaScript extraction not yet implemented")
        return None
        
    def _resolve_attribute_name(self, attr_node: ast.Attribute) -> str:
        """Resolve attribute name from AST node."""
        parts = []
        current = attr_node
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
            
        if isinstance(current, ast.Name):
            parts.append(current.id)
            
        return '.'.join(reversed(parts))
        
    def _determine_access_level(self, name: str) -> str:
        """Determine access level from naming convention."""
        if name.startswith('__') and name.endswith('__'):
            return "special"
        elif name.startswith('_'):
            return "private"
        else:
            return "public"
            
    def _is_inside_class(self, func_node: ast.FunctionDef, tree: ast.Module) -> bool:
        """Check if function is inside a class."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in ast.walk(node):
                    if child is func_node:
                        return True
        return False
        
    def _load_templates(self) -> Dict[str, str]:
        """Load documentation templates."""
        return {
            "class_template": """# {class_name}

{description}

## Methods

{methods}

## Properties

{properties}
""",
            "method_template": """### {method_name}

```{language}
{signature}
```

{description}

{parameters}

{returns}
"""
        }
        
    def export_api_docs(self, namespaces: List[ApiNamespace], output_dir: str) -> None:
        """Export API documentation to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        docs = self.generate_documentation(namespaces)
        
        for filename, content in docs.items():
            file_path = output_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        logger.info(f"Exported API docs to {output_dir}")