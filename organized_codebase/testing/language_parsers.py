"""
Language Parser Registry for Universal AST Abstraction

Directly adapted from existing multi-agent frameworks to provide
universal language parsing capabilities.
"""

import ast
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass

from .universal_ast import (
    UniversalModule, UniversalFunction, UniversalClass, UniversalVariable,
    UniversalImport, UniversalParameter, CodeLocation, UniversalExpression, ASTNodeType
)


class BaseLanguageParser(ABC):
    """Base class for language-specific parsers - adapted from Agency Swarm's tool patterns."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.language = self.__class__.__name__.replace('ASTParser', '').lower()
    
    @abstractmethod
    def parse_to_universal(self, file_path: str, content: str) -> UniversalModule:
        """Parse source code to Universal Module representation."""
        pass
    
    def supports_language(self, language: str) -> bool:
        """Check if this parser supports the given language."""
        return language.lower() == self.language
    
    def _create_location(self, file_path: str, line_start: int, line_end: int = None, col_start: int = 0, col_end: int = 0) -> CodeLocation:
        """Helper to create code location."""
        return CodeLocation(
            file_path=file_path,
            line_start=line_start,
            line_end=line_end or line_start,
            column_start=col_start,
            column_end=col_end
        )


class PythonASTParser(BaseLanguageParser):
    """Python AST Parser - Enhanced from TestMaster's existing patterns."""
    
    def parse_to_universal(self, file_path: str, content: str) -> UniversalModule:
        """Parse Python code using AST module."""
        try:
            tree = ast.parse(content)
            return self._convert_python_ast(file_path, tree, content)
        except SyntaxError as e:
            print(f"Python syntax error in {file_path}: {e}")
            return self._fallback_parse(file_path, content)
    
    def _convert_python_ast(self, file_path: str, tree: ast.AST, content: str) -> UniversalModule:
        """Convert Python AST to Universal Module - adapted from TestMaster patterns."""
        functions = []
        classes = []
        imports = []
        variables = []
        expressions = []
        
        # Walk the AST and extract components
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(self._convert_python_function(node, file_path))
            elif isinstance(node, ast.AsyncFunctionDef):
                func = self._convert_python_function(node, file_path)
                func.is_async = True
                functions.append(func)
            elif isinstance(node, ast.ClassDef):
                classes.append(self._convert_python_class(node, file_path))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.extend(self._convert_python_import(node, file_path))
            elif isinstance(node, ast.Assign):
                variables.extend(self._convert_python_assignment(node, file_path))
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                expressions.append(self._convert_python_expression(node, file_path))
        
        # Calculate metrics
        lines = content.split('\n')
        lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        # Extract module docstring
        docstring = ast.get_docstring(tree) if hasattr(ast, 'get_docstring') else None
        
        return UniversalModule(
            name=Path(file_path).stem,
            file_path=file_path,
            language='python',
            functions=functions,
            classes=classes,
            imports=imports,
            variables=variables,
            expressions=expressions,
            docstring=docstring,
            lines_of_code=lines_of_code,
            location=self._create_location(file_path, 1, len(lines))
        )
    
    def _convert_python_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], file_path: str) -> UniversalFunction:
        """Convert Python function to Universal Function."""
        # Extract parameters
        parameters = []
        
        # Regular arguments
        for arg in node.args.args:
            param = UniversalParameter(
                name=arg.arg,
                type_hint=self._get_annotation_string(arg.annotation),
            )
            parameters.append(param)
        
        # Default values
        defaults = node.args.defaults
        if defaults:
            # Map defaults to parameters (from the end)
            for i, default in enumerate(defaults):
                param_idx = len(parameters) - len(defaults) + i
                if param_idx >= 0 and param_idx < len(parameters):
                    parameters[param_idx].default_value = self._get_node_string(default)
        
        # Keyword-only arguments
        for arg in node.args.kwonlyargs:
            param = UniversalParameter(
                name=arg.arg,
                type_hint=self._get_annotation_string(arg.annotation),
                is_keyword_only=True
            )
            parameters.append(param)
        
        # Variadic arguments
        if node.args.vararg:
            param = UniversalParameter(
                name=node.args.vararg.arg,
                type_hint=self._get_annotation_string(node.args.vararg.annotation),
                is_variadic=True
            )
            parameters.append(param)
        
        # Extract function calls and variable accesses
        calls_functions = []
        accesses_variables = []
        throws_exceptions = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                calls_functions.append(child.func.id)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                accesses_variables.append(child.id)
            elif isinstance(child, ast.Raise) and child.exc:
                if isinstance(child.exc, ast.Name):
                    throws_exceptions.append(child.exc.id)
                elif isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                    throws_exceptions.append(child.exc.func.id)
        
        # Calculate complexity
        complexity_score = self._calculate_complexity(node)
        cyclomatic_complexity = self._calculate_cyclomatic_complexity(node)
        
        return UniversalFunction(
            name=node.name,
            parameters=parameters,
            return_type=self._get_annotation_string(node.returns),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            decorators=[self._get_node_string(dec) for dec in node.decorator_list],
            docstring=ast.get_docstring(node),
            complexity_score=complexity_score,
            cyclomatic_complexity=cyclomatic_complexity,
            lines_of_code=node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 1,
            location=self._create_location(
                file_path,
                node.lineno,
                getattr(node, 'end_lineno', node.lineno),
                node.col_offset,
                getattr(node, 'end_col_offset', 0)
            ),
            calls_functions=list(set(calls_functions)),
            accesses_variables=list(set(accesses_variables)),
            throws_exceptions=list(set(throws_exceptions))
        )
    
    def _convert_python_class(self, node: ast.ClassDef, file_path: str) -> UniversalClass:
        """Convert Python class to Universal Class."""
        methods = []
        fields = []
        inner_classes = []
        
        # Process class body
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._convert_python_function(item, file_path))
            elif isinstance(item, ast.ClassDef):
                inner_classes.append(self._convert_python_class(item, file_path))
            elif isinstance(item, ast.Assign):
                # Class variables
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        field = UniversalVariable(
                            name=target.id,
                            location=self._create_location(file_path, item.lineno)
                        )
                        fields.append(field)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                # Annotated class variables
                field = UniversalVariable(
                    name=item.target.id,
                    type_hint=self._get_annotation_string(item.annotation),
                    location=self._create_location(file_path, item.lineno)
                )
                fields.append(field)
        
        return UniversalClass(
            name=node.name,
            base_classes=[self._get_node_string(base) for base in node.bases],
            methods=methods,
            fields=fields,
            inner_classes=inner_classes,
            decorators=[self._get_node_string(dec) for dec in node.decorator_list],
            docstring=ast.get_docstring(node),
            location=self._create_location(
                file_path,
                node.lineno,
                getattr(node, 'end_lineno', node.lineno),
                node.col_offset,
                getattr(node, 'end_col_offset', 0)
            )
        )
    
    def _convert_python_import(self, node: Union[ast.Import, ast.ImportFrom], file_path: str) -> List[UniversalImport]:
        """Convert Python import to Universal Import."""
        imports = []
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(UniversalImport(
                    module_name=alias.name,
                    alias=alias.asname,
                    location=self._create_location(file_path, node.lineno)
                ))
        elif isinstance(node, ast.ImportFrom):
            imported_items = []
            for alias in node.names:
                if alias.name == '*':
                    imports.append(UniversalImport(
                        module_name=node.module or "",
                        is_wildcard=True,
                        is_relative=node.level > 0,
                        location=self._create_location(file_path, node.lineno)
                    ))
                else:
                    imported_items.append(alias.name)
            
            if imported_items:
                imports.append(UniversalImport(
                    module_name=node.module or "",
                    imported_items=imported_items,
                    is_relative=node.level > 0,
                    location=self._create_location(file_path, node.lineno)
                ))
        
        return imports
    
    def _convert_python_assignment(self, node: ast.Assign, file_path: str) -> List[UniversalVariable]:
        """Convert Python assignment to Universal Variable."""
        variables = []
        
        for target in node.targets:
            if isinstance(target, ast.Name):
                var = UniversalVariable(
                    name=target.id,
                    default_value=self._get_node_string(node.value),
                    location=self._create_location(file_path, node.lineno)
                )
                variables.append(var)
            elif isinstance(target, ast.Tuple):
                # Handle tuple unpacking
                for i, elt in enumerate(target.elts):
                    if isinstance(elt, ast.Name):
                        var = UniversalVariable(
                            name=elt.id,
                            location=self._create_location(file_path, node.lineno)
                        )
                        variables.append(var)
        
        return variables
    
    def _convert_python_expression(self, node: ast.AST, file_path: str) -> UniversalExpression:
        """Convert Python expression/statement to Universal Expression."""
        node_type_map = {
            ast.If: ASTNodeType.CONDITIONAL,
            ast.For: ASTNodeType.LOOP,
            ast.While: ASTNodeType.LOOP,
            ast.Try: ASTNodeType.TRY_CATCH,
            ast.With: ASTNodeType.STATEMENT,
        }
        
        node_type = node_type_map.get(type(node), ASTNodeType.EXPRESSION)
        
        # Extract variables and function calls
        variables_used = []
        functions_called = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                variables_used.append(child.id)
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                functions_called.append(child.func.id)
        
        return UniversalExpression(
            node_type=node_type,
            content=self._get_node_string(node),
            location=self._create_location(file_path, node.lineno),
            variables_used=list(set(variables_used)),
            functions_called=list(set(functions_called))
        )
    
    def _get_annotation_string(self, annotation) -> Optional[str]:
        """Get string representation of type annotation."""
        if annotation is None:
            return None
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(annotation)
            else:
                return self._get_node_string(annotation)
        except:
            return str(annotation)
    
    def _get_node_string(self, node) -> str:
        """Get string representation of AST node."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                # Fallback for older Python versions
                if isinstance(node, ast.Name):
                    return node.id
                elif isinstance(node, ast.Constant):
                    return repr(node.value)
                elif isinstance(node, ast.Attribute):
                    return f"{self._get_node_string(node.value)}.{node.attr}"
                else:
                    return str(node)
        except:
            return str(node)
    
    def _calculate_complexity(self, node: ast.AST) -> float:
        """Calculate complexity score for a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return min(complexity / 10.0, 1.0)  # Normalize to 0-1
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _fallback_parse(self, file_path: str, content: str) -> UniversalModule:
        """Fallback parsing when AST parsing fails."""
        lines = content.split('\n')
        lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        return UniversalModule(
            name=Path(file_path).stem,
            file_path=file_path,
            language='python',
            lines_of_code=lines_of_code,
            location=self._create_location(file_path, 1, len(lines))
        )


class JavaScriptASTParser(BaseLanguageParser):
    """JavaScript AST Parser - Adapted from Agency Swarm's patterns with regex fallback."""
    
    def parse_to_universal(self, file_path: str, content: str) -> UniversalModule:
        """Parse JavaScript code using regex patterns."""
        lines = content.split('\n')
        lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('//') and not line.strip().startswith('/*')])
        
        functions = self._extract_js_functions(content, file_path)
        classes = self._extract_js_classes(content, file_path)
        imports = self._extract_js_imports(content, file_path)
        variables = self._extract_js_variables(content, file_path)
        
        return UniversalModule(
            name=Path(file_path).stem,
            file_path=file_path,
            language='javascript',
            functions=functions,
            classes=classes,
            imports=imports,
            variables=variables,
            lines_of_code=lines_of_code,
            location=self._create_location(file_path, 1, len(lines))
        )
    
    def _extract_js_functions(self, content: str, file_path: str) -> List[UniversalFunction]:
        """Extract JavaScript functions using enhanced regex patterns."""
        functions = []
        
        # Function declarations
        pattern = r'function\s+(\w+)\s*\(([^)]*)\)\s*\{([^}]*)\}'
        for match in re.finditer(pattern, content, re.DOTALL):
            func_name = match.group(1)
            params_str = match.group(2).strip()
            body = match.group(3)
            
            parameters = self._parse_js_parameters(params_str)
            line_num = content[:match.start()].count('\n') + 1
            
            functions.append(UniversalFunction(
                name=func_name,
                parameters=parameters,
                is_async='async' in content[max(0, match.start()-50):match.start()],
                location=self._create_location(file_path, line_num),
                calls_functions=self._extract_function_calls(body),
                accesses_variables=self._extract_variable_accesses(body)
            ))
        
        # Arrow functions
        pattern = r'const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>\s*\{?'
        for match in re.finditer(pattern, content):
            func_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            is_async = 'async' in match.group(0)
            
            functions.append(UniversalFunction(
                name=func_name,
                is_async=is_async,
                location=self._create_location(file_path, line_num)
            ))
        
        # Method definitions in classes
        pattern = r'(\w+)\s*\([^)]*\)\s*\{'
        class_context = self._find_class_contexts(content)
        
        for match in re.finditer(pattern, content):
            method_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            # Check if this is inside a class
            in_class = any(start <= match.start() <= end for start, end in class_context)
            
            if in_class and method_name not in ['if', 'for', 'while', 'switch']:
                functions.append(UniversalFunction(
                    name=method_name,
                    location=self._create_location(file_path, line_num)
                ))
        
        return functions
    
    def _extract_js_classes(self, content: str, file_path: str) -> List[UniversalClass]:
        """Extract JavaScript classes."""
        classes = []
        
        pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{([^}]*)\}'
        for match in re.finditer(pattern, content, re.DOTALL):
            class_name = match.group(1)
            base_class = match.group(2)
            class_body = match.group(3)
            line_num = content[:match.start()].count('\n') + 1
            
            base_classes = [base_class] if base_class else []
            methods = self._extract_methods_from_class_body(class_body, file_path, line_num)
            
            classes.append(UniversalClass(
                name=class_name,
                base_classes=base_classes,
                methods=methods,
                location=self._create_location(file_path, line_num)
            ))
        
        return classes
    
    def _extract_js_imports(self, content: str, file_path: str) -> List[UniversalImport]:
        """Extract JavaScript imports."""
        imports = []
        
        # ES6 imports
        patterns = [
            r'import\s+\{([^}]+)\}\s+from\s+[\'"]([^\'"]+)[\'"]',  # Named imports
            r'import\s+(\w+)\s+from\s+[\'"]([^\'"]+)[\'"]',        # Default imports
            r'import\s+\*\s+as\s+(\w+)\s+from\s+[\'"]([^\'"]+)[\'"]',  # Namespace imports
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                
                if '{' in match.group(0):  # Named imports
                    imported_items = [item.strip() for item in match.group(1).split(',')]
                    module_name = match.group(2)
                else:  # Default or namespace imports
                    imported_items = [match.group(1)]
                    module_name = match.group(2)
                
                imports.append(UniversalImport(
                    module_name=module_name,
                    imported_items=imported_items,
                    location=self._create_location(file_path, line_num)
                ))
        
        # CommonJS requires
        pattern = r'const\s+\{?([^}]+)\}?\s*=\s*require\([\'"]([^\'"]+)[\'"]\)'
        for match in re.finditer(pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            imported_item = match.group(1).strip()
            module_name = match.group(2)
            
            imports.append(UniversalImport(
                module_name=module_name,
                imported_items=[imported_item] if '{' not in match.group(0) else [item.strip() for item in imported_item.split(',')],
                location=self._create_location(file_path, line_num)
            ))
        
        return imports
    
    def _extract_js_variables(self, content: str, file_path: str) -> List[UniversalVariable]:
        """Extract JavaScript variables."""
        variables = []
        
        patterns = [
            r'const\s+(\w+)\s*=',
            r'let\s+(\w+)\s*=',
            r'var\s+(\w+)\s*='
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                var_name = match.group(1)
                line_num = content[:match.start()].count('\n') + 1
                
                is_constant = pattern.startswith('const')
                
                variables.append(UniversalVariable(
                    name=var_name,
                    is_constant=is_constant,
                    location=self._create_location(file_path, line_num)
                ))
        
        return variables
    
    def _parse_js_parameters(self, params_str: str) -> List[UniversalParameter]:
        """Parse JavaScript function parameters."""
        if not params_str.strip():
            return []
        
        parameters = []
        params = [p.strip() for p in params_str.split(',')]
        
        for param in params:
            if '=' in param:
                name, default = param.split('=', 1)
                parameters.append(UniversalParameter(
                    name=name.strip(),
                    default_value=default.strip()
                ))
            elif param.startswith('...'):
                parameters.append(UniversalParameter(
                    name=param[3:].strip(),
                    is_variadic=True
                ))
            else:
                parameters.append(UniversalParameter(name=param))
        
        return parameters
    
    def _find_class_contexts(self, content: str) -> List[Tuple[int, int]]:
        """Find start and end positions of class definitions."""
        contexts = []
        pattern = r'class\s+\w+[^{]*\{'
        
        for match in re.finditer(pattern, content):
            start = match.start()
            brace_count = 0
            pos = match.end() - 1  # Start from the opening brace
            
            while pos < len(content):
                if content[pos] == '{':
                    brace_count += 1
                elif content[pos] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        contexts.append((start, pos))
                        break
                pos += 1
        
        return contexts
    
    def _extract_methods_from_class_body(self, class_body: str, file_path: str, class_line: int) -> List[UniversalFunction]:
        """Extract methods from class body."""
        methods = []
        
        # Method pattern: methodName() { or async methodName() {
        pattern = r'(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{'
        
        for match in re.finditer(pattern, class_body):
            method_name = match.group(1)
            line_offset = class_body[:match.start()].count('\n')
            line_num = class_line + line_offset
            is_async = 'async' in match.group(0)
            
            # Skip common keywords
            if method_name not in ['if', 'for', 'while', 'switch', 'catch']:
                methods.append(UniversalFunction(
                    name=method_name,
                    is_async=is_async,
                    location=self._create_location(file_path, line_num)
                ))
        
        return methods
    
    def _extract_function_calls(self, code: str) -> List[str]:
        """Extract function calls from code."""
        calls = []
        pattern = r'(\w+)\s*\('
        
        for match in re.finditer(pattern, code):
            func_name = match.group(1)
            if func_name not in ['if', 'for', 'while', 'switch', 'catch']:
                calls.append(func_name)
        
        return list(set(calls))
    
    def _extract_variable_accesses(self, code: str) -> List[str]:
        """Extract variable accesses from code."""
        variables = []
        # Simple pattern for variable names
        pattern = r'\b([a-zA-Z_]\w*)\b'
        
        for match in re.finditer(pattern, code):
            var_name = match.group(1)
            # Filter out keywords and function calls
            if var_name not in ['const', 'let', 'var', 'function', 'class', 'if', 'else', 'for', 'while', 'return']:
                variables.append(var_name)
        
        return list(set(variables))


class TypeScriptASTParser(JavaScriptASTParser):
    """TypeScript AST Parser - Extends JavaScript parser."""
    
    def parse_to_universal(self, file_path: str, content: str) -> UniversalModule:
        """Parse TypeScript code by extending JavaScript parser."""
        module = super().parse_to_universal(file_path, content)
        module.language = 'typescript'
        
        # Add TypeScript-specific parsing
        interfaces = self._extract_ts_interfaces(content, file_path)
        module.classes.extend(interfaces)
        
        return module
    
    def _extract_ts_interfaces(self, content: str, file_path: str) -> List[UniversalClass]:
        """Extract TypeScript interfaces as Universal Classes."""
        interfaces = []
        
        pattern = r'interface\s+(\w+)(?:\s+extends\s+([\w,\s]+))?\s*\{([^}]*)\}'
        for match in re.finditer(pattern, content, re.DOTALL):
            interface_name = match.group(1)
            extends_clause = match.group(2)
            interface_body = match.group(3)
            line_num = content[:match.start()].count('\n') + 1
            
            base_interfaces = []
            if extends_clause:
                base_interfaces = [name.strip() for name in extends_clause.split(',')]
            
            # Parse interface properties as fields
            fields = self._parse_interface_properties(interface_body, file_path, line_num)
            
            interfaces.append(UniversalClass(
                name=interface_name,
                base_classes=base_interfaces,
                fields=fields,
                is_interface=True,
                location=self._create_location(file_path, line_num)
            ))
        
        return interfaces
    
    def _parse_interface_properties(self, interface_body: str, file_path: str, base_line: int) -> List[UniversalVariable]:
        """Parse TypeScript interface properties."""
        fields = []
        
        # Property pattern: propertyName: type;
        pattern = r'(\w+)\s*:\s*([^;]+);'
        for match in re.finditer(pattern, interface_body):
            prop_name = match.group(1)
            prop_type = match.group(2).strip()
            line_offset = interface_body[:match.start()].count('\n')
            line_num = base_line + line_offset
            
            fields.append(UniversalVariable(
                name=prop_name,
                type_hint=prop_type,
                location=self._create_location(file_path, line_num)
            ))
        
        return fields


class JavaASTParser(BaseLanguageParser):
    """Java AST Parser - Basic regex-based implementation."""
    
    def parse_to_universal(self, file_path: str, content: str) -> UniversalModule:
        """Parse Java code using regex patterns."""
        lines = content.split('\n')
        lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('//') and not line.strip().startswith('/*')])
        
        classes = self._extract_java_classes(content, file_path)
        imports = self._extract_java_imports(content, file_path)
        
        return UniversalModule(
            name=Path(file_path).stem,
            file_path=file_path,
            language='java',
            classes=classes,
            imports=imports,
            lines_of_code=lines_of_code,
            location=self._create_location(file_path, 1, len(lines))
        )
    
    def _extract_java_classes(self, content: str, file_path: str) -> List[UniversalClass]:
        """Extract Java classes."""
        classes = []
        
        pattern = r'(?:public\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?\s*\{'
        for match in re.finditer(pattern, content):
            class_name = match.group(1)
            base_class = match.group(2)
            interfaces = match.group(3)
            line_num = content[:match.start()].count('\n') + 1
            
            base_classes = [base_class] if base_class else []
            interface_list = [i.strip() for i in interfaces.split(',')] if interfaces else []
            
            # Extract methods
            methods = self._extract_java_methods(content, file_path, class_name)
            
            classes.append(UniversalClass(
                name=class_name,
                base_classes=base_classes,
                interfaces=interface_list,
                methods=methods,
                is_public='public' in match.group(0),
                location=self._create_location(file_path, line_num)
            ))
        
        return classes
    
    def _extract_java_methods(self, content: str, file_path: str, class_name: str) -> List[UniversalFunction]:
        """Extract Java methods."""
        methods = []
        
        # Method pattern
        pattern = r'(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*(\w+)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{'
        
        for match in re.finditer(pattern, content):
            return_type = match.group(1)
            method_name = match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            
            # Skip if this is not in our class context (basic check)
            is_static = 'static' in match.group(0)
            is_public = 'public' in match.group(0)
            is_private = 'private' in match.group(0)
            is_protected = 'protected' in match.group(0)
            
            methods.append(UniversalFunction(
                name=method_name,
                return_type=return_type,
                is_static=is_static,
                is_public=is_public,
                is_private=is_private,
                is_protected=is_protected,
                location=self._create_location(file_path, line_num)
            ))
        
        return methods
    
    def _extract_java_imports(self, content: str, file_path: str) -> List[UniversalImport]:
        """Extract Java imports."""
        imports = []
        
        pattern = r'import\s+(?:static\s+)?([\w\.]+)(?:\.\*)?;'
        for match in re.finditer(pattern, content):
            import_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            is_wildcard = '.*' in match.group(0)
            
            imports.append(UniversalImport(
                module_name=import_name,
                is_wildcard=is_wildcard,
                location=self._create_location(file_path, line_num)
            ))
        
        return imports


class CSharpASTParser(BaseLanguageParser):
    """C# AST Parser - Basic regex-based implementation."""
    
    def parse_to_universal(self, file_path: str, content: str) -> UniversalModule:
        """Parse C# code using regex patterns."""
        lines = content.split('\n')
        lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('//') and not line.strip().startswith('/*')])
        
        classes = self._extract_csharp_classes(content, file_path)
        imports = self._extract_csharp_usings(content, file_path)
        
        return UniversalModule(
            name=Path(file_path).stem,
            file_path=file_path,
            language='csharp',
            classes=classes,
            imports=imports,
            lines_of_code=lines_of_code,
            location=self._create_location(file_path, 1, len(lines))
        )
    
    def _extract_csharp_classes(self, content: str, file_path: str) -> List[UniversalClass]:
        """Extract C# classes."""
        classes = []
        
        pattern = r'(?:public\s+)?(?:abstract\s+)?(?:sealed\s+)?class\s+(\w+)(?:\s*:\s*([\w,\s]+))?\s*\{'
        for match in re.finditer(pattern, content):
            class_name = match.group(1)
            inheritance = match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            
            base_classes = []
            interfaces = []
            if inheritance:
                # In C#, first item is base class, rest are interfaces
                items = [item.strip() for item in inheritance.split(',')]
                if items:
                    base_classes = [items[0]]
                    interfaces = items[1:] if len(items) > 1 else []
            
            classes.append(UniversalClass(
                name=class_name,
                base_classes=base_classes,
                interfaces=interfaces,
                is_public='public' in match.group(0),
                location=self._create_location(file_path, line_num)
            ))
        
        return classes
    
    def _extract_csharp_usings(self, content: str, file_path: str) -> List[UniversalImport]:
        """Extract C# using statements."""
        imports = []
        
        pattern = r'using\s+([\w\.]+);'
        for match in re.finditer(pattern, content):
            using_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            imports.append(UniversalImport(
                module_name=using_name,
                location=self._create_location(file_path, line_num)
            ))
        
        return imports


class GoASTParser(BaseLanguageParser):
    """Go AST Parser - Basic regex-based implementation."""
    
    def parse_to_universal(self, file_path: str, content: str) -> UniversalModule:
        """Parse Go code using regex patterns."""
        lines = content.split('\n')
        lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('//')])
        
        functions = self._extract_go_functions(content, file_path)
        imports = self._extract_go_imports(content, file_path)
        structs = self._extract_go_structs(content, file_path)
        
        return UniversalModule(
            name=Path(file_path).stem,
            file_path=file_path,
            language='go',
            functions=functions,
            classes=structs,  # Treat structs as classes
            imports=imports,
            lines_of_code=lines_of_code,
            location=self._create_location(file_path, 1, len(lines))
        )
    
    def _extract_go_functions(self, content: str, file_path: str) -> List[UniversalFunction]:
        """Extract Go functions."""
        functions = []
        
        pattern = r'func\s+(?:\([^)]*\)\s+)?(\w+)\s*\([^)]*\)(?:\s*\([^)]*\))?\s*\{'
        for match in re.finditer(pattern, content):
            func_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            functions.append(UniversalFunction(
                name=func_name,
                location=self._create_location(file_path, line_num)
            ))
        
        return functions
    
    def _extract_go_structs(self, content: str, file_path: str) -> List[UniversalClass]:
        """Extract Go structs as classes."""
        structs = []
        
        pattern = r'type\s+(\w+)\s+struct\s*\{'
        for match in re.finditer(pattern, content):
            struct_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            structs.append(UniversalClass(
                name=struct_name,
                location=self._create_location(file_path, line_num),
                language_specific_data={'go_type': 'struct'}
            ))
        
        return structs
    
    def _extract_go_imports(self, content: str, file_path: str) -> List[UniversalImport]:
        """Extract Go imports."""
        imports = []
        
        # Single import
        pattern = r'import\s+"([^"]+)"'
        for match in re.finditer(pattern, content):
            import_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            imports.append(UniversalImport(
                module_name=import_name,
                location=self._create_location(file_path, line_num)
            ))
        
        # Multiple imports
        pattern = r'import\s*\(\s*([^)]+)\s*\)'
        for match in re.finditer(pattern, content, re.DOTALL):
            import_block = match.group(1)
            base_line = content[:match.start()].count('\n') + 1
            
            for line_offset, line in enumerate(import_block.split('\n')):
                line = line.strip()
                if line and line.startswith('"') and line.endswith('"'):
                    import_name = line[1:-1]
                    imports.append(UniversalImport(
                        module_name=import_name,
                        location=self._create_location(file_path, base_line + line_offset)
                    ))
        
        return imports


class RustASTParser(BaseLanguageParser):
    """Rust AST Parser - Basic regex-based implementation."""
    
    def parse_to_universal(self, file_path: str, content: str) -> UniversalModule:
        """Parse Rust code using regex patterns."""
        lines = content.split('\n')
        lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('//')])
        
        functions = self._extract_rust_functions(content, file_path)
        structs = self._extract_rust_structs(content, file_path)
        imports = self._extract_rust_uses(content, file_path)
        
        return UniversalModule(
            name=Path(file_path).stem,
            file_path=file_path,
            language='rust',
            functions=functions,
            classes=structs,  # Treat structs as classes
            imports=imports,
            lines_of_code=lines_of_code,
            location=self._create_location(file_path, 1, len(lines))
        )
    
    def _extract_rust_functions(self, content: str, file_path: str) -> List[UniversalFunction]:
        """Extract Rust functions."""
        functions = []
        
        pattern = r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\([^)]*\)(?:\s*->\s*[^{]+)?\s*\{'
        for match in re.finditer(pattern, content):
            func_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            is_public = 'pub' in match.group(0)
            is_async = 'async' in match.group(0)
            
            functions.append(UniversalFunction(
                name=func_name,
                is_public=is_public,
                is_async=is_async,
                location=self._create_location(file_path, line_num)
            ))
        
        return functions
    
    def _extract_rust_structs(self, content: str, file_path: str) -> List[UniversalClass]:
        """Extract Rust structs as classes."""
        structs = []
        
        pattern = r'(?:pub\s+)?struct\s+(\w+)\s*\{'
        for match in re.finditer(pattern, content):
            struct_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            is_public = 'pub' in match.group(0)
            
            structs.append(UniversalClass(
                name=struct_name,
                is_public=is_public,
                location=self._create_location(file_path, line_num),
                language_specific_data={'rust_type': 'struct'}
            ))
        
        return structs
    
    def _extract_rust_uses(self, content: str, file_path: str) -> List[UniversalImport]:
        """Extract Rust use statements."""
        imports = []
        
        pattern = r'use\s+([\w:]+)(?:::\{([^}]+)\})?;'
        for match in re.finditer(pattern, content):
            module_path = match.group(1)
            items = match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            
            if items:
                # Multiple imports from module
                imported_items = [item.strip() for item in items.split(',')]
                imports.append(UniversalImport(
                    module_name=module_path,
                    imported_items=imported_items,
                    location=self._create_location(file_path, line_num)
                ))
            else:
                # Single import
                imports.append(UniversalImport(
                    module_name=module_path,
                    location=self._create_location(file_path, line_num)
                ))
        
        return imports


class LanguageParserRegistry:
    """Registry for language parsers - Adapted from Agency Swarm's tool registry pattern."""
    
    _parsers: Dict[str, BaseLanguageParser] = {}
    
    @classmethod
    def register_parser(cls, language: str, parser: BaseLanguageParser):
        """Register a language parser."""
        cls._parsers[language.lower()] = parser
    
    @classmethod
    def get_parser(cls, language: str) -> Optional[BaseLanguageParser]:
        """Get parser for language."""
        return cls._parsers.get(language.lower())
    
    @classmethod
    def get_all_parsers(cls) -> Dict[str, BaseLanguageParser]:
        """Get all registered parsers."""
        if not cls._parsers:
            cls._initialize_default_parsers()
        return cls._parsers.copy()
    
    @classmethod
    def supports_language(cls, language: str) -> bool:
        """Check if language is supported."""
        if not cls._parsers:
            cls._initialize_default_parsers()
        return language.lower() in cls._parsers
    
    @classmethod
    def _initialize_default_parsers(cls):
        """Initialize default language parsers."""
        parsers = [
            ('python', PythonASTParser()),
            ('javascript', JavaScriptASTParser()),
            ('typescript', TypeScriptASTParser()),
            ('java', JavaASTParser()),
            ('csharp', CSharpASTParser()),
            ('go', GoASTParser()),
            ('rust', RustASTParser()),
        ]
        
        for language, parser in parsers:
            cls.register_parser(language, parser)


# Initialize default parsers
LanguageParserRegistry._initialize_default_parsers()