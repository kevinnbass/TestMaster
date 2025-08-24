"""
Metaprogramming Analysis Module
Analyzes dynamic code execution, eval/exec usage, and metaprogramming safety
"""

import ast
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import json

from testmaster.analysis.base_analyzer import BaseAnalyzer


@dataclass
class MetaprogrammingIssue:
    """Represents a metaprogramming safety issue"""
    issue_type: str
    severity: str
    location: str
    description: str
    code_snippet: str
    risk_assessment: str
    mitigation: str
    cwe_id: Optional[str]


@dataclass
class DynamicCodePattern:
    """Represents a dynamic code execution pattern"""
    pattern_type: str
    location: str
    input_source: str
    sanitization: bool
    validation: bool
    risk_level: str


class MetaprogrammingAnalyzer(BaseAnalyzer):
    """
    Analyzes metaprogramming patterns and dynamic code execution safety
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Dangerous functions to track
        self.dangerous_functions = {
            "eval": {"risk": "critical", "cwe": "CWE-95"},
            "exec": {"risk": "critical", "cwe": "CWE-95"},
            "compile": {"risk": "high", "cwe": "CWE-94"},
            "__import__": {"risk": "high", "cwe": "CWE-829"},
            "importlib.import_module": {"risk": "medium", "cwe": "CWE-829"},
            "getattr": {"risk": "medium", "cwe": "CWE-470"},
            "setattr": {"risk": "medium", "cwe": "CWE-470"},
            "delattr": {"risk": "low", "cwe": "CWE-470"},
            "globals": {"risk": "medium", "cwe": "CWE-497"},
            "locals": {"risk": "low", "cwe": "CWE-497"}
        }
        
        # Safe patterns for dynamic code
        self.safe_patterns = {
            "literal_eval": "ast.literal_eval",
            "safe_eval": "restricted eval with whitelist",
            "template_engine": "use template engines instead",
            "configuration": "use configuration files"
        }
        
        # Metaprogramming patterns
        self.meta_patterns = {
            "decorators": [],
            "metaclasses": [],
            "descriptors": [],
            "monkey_patching": [],
            "code_generation": [],
            "reflection": []
        }
        
        self.issues = []
        self.dynamic_patterns = []
        
    def analyze(self) -> Dict[str, Any]:
        """
        Perform comprehensive metaprogramming analysis
        """
        results = {
            "dynamic_execution": self._analyze_dynamic_execution(),
            "eval_exec_usage": self._analyze_eval_exec(),
            "import_security": self._analyze_dynamic_imports(),
            "reflection_usage": self._analyze_reflection(),
            "code_injection_risks": self._detect_injection_risks(),
            "metaclass_analysis": self._analyze_metaclasses(),
            "decorator_analysis": self._analyze_decorators(),
            "monkey_patching": self._detect_monkey_patching(),
            "code_generation": self._analyze_code_generation(),
            "sandbox_escapes": self._detect_sandbox_escapes(),
            "serialization_risks": self._analyze_serialization(),
            "template_injection": self._detect_template_injection(),
            "safe_alternatives": self._suggest_safe_alternatives(),
            "security_assessment": self._security_risk_assessment(),
            "summary": self._generate_metaprogramming_summary()
        }
        
        return results
    
    def _analyze_dynamic_execution(self) -> Dict[str, Any]:
        """
        Analyze all forms of dynamic code execution
        """
        dynamic_exec = {
            "exec_calls": [],
            "eval_calls": [],
            "compile_calls": [],
            "dynamic_imports": [],
            "risk_distribution": defaultdict(int),
            "input_sources": defaultdict(list)
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            # Check for dangerous function calls
                            func_name = self._get_function_name(node)
                            
                            if func_name in self.dangerous_functions:
                                issue = self._analyze_dangerous_call(node, func_name, file_path)
                                
                                if func_name == "exec":
                                    dynamic_exec["exec_calls"].append(issue)
                                elif func_name == "eval":
                                    dynamic_exec["eval_calls"].append(issue)
                                elif func_name == "compile":
                                    dynamic_exec["compile_calls"].append(issue)
                                elif "__import__" in func_name or "import_module" in func_name:
                                    dynamic_exec["dynamic_imports"].append(issue)
                                    
                                # Track risk distribution
                                risk = self.dangerous_functions[func_name]["risk"]
                                dynamic_exec["risk_distribution"][risk] += 1
                                
                                # Analyze input source
                                input_source = self._analyze_input_source(node)
                                dynamic_exec["input_sources"][input_source].append(func_name)
                                
            except Exception as e:
                self.logger.error(f"Error analyzing dynamic execution in {file_path}: {e}")
                
        return dynamic_exec
    
    def _analyze_eval_exec(self) -> Dict[str, Any]:
        """
        Deep analysis of SafeCodeExecutor.safe_eval() and exec() usage
        """
        eval_exec_analysis = {
            "eval_usage": [],
            "exec_usage": [],
            "user_input_eval": [],
            "safe_eval_patterns": [],
            "unsafe_patterns": [],
            "mitigation_suggestions": []
        }
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            func_name = self._get_function_name(node)
                            
                            if func_name == "eval":
                                eval_analysis = self._analyze_eval_call(node, file_path, content)
                                eval_exec_analysis["eval_usage"].append(eval_analysis)
                                
                                # Check if user input flows to eval
                                if eval_analysis.get("user_input_risk"):
                                    eval_exec_analysis["user_input_eval"].append(eval_analysis)
                                    
                                # Check for safe patterns
                                if self._is_safe_SafeCodeExecutor.safe_eval(node):
                                    eval_exec_analysis["safe_eval_patterns"].append({
                                        "location": str(file_path),
                                        "pattern": "Limited scope eval",
                                        "safety_measures": self._get_safety_measures(node)
                                    })
                                else:
                                    eval_exec_analysis["unsafe_patterns"].append({
                                        "location": str(file_path),
                                        "issue": "Unrestricted eval",
                                        "risk": "Code injection"
                                    })
                                    
                            elif func_name == "exec":
                                exec_analysis = self._analyze_exec_call(node, file_path, content)
                                eval_exec_analysis["exec_usage"].append(exec_analysis)
                                
                                if not self._is_safe_exec(node):
                                    eval_exec_analysis["unsafe_patterns"].append({
                                        "location": str(file_path),
                                        "issue": "Unrestricted exec",
                                        "risk": "Arbitrary code execution"
                                    })
                                    
            except Exception as e:
                self.logger.error(f"Error analyzing eval/exec in {file_path}: {e}")
                
        # Generate mitigation suggestions
        if eval_exec_analysis["unsafe_patterns"]:
            eval_exec_analysis["mitigation_suggestions"] = [
                {
                    "pattern": "eval with user input",
                    "suggestion": "Use ast.literal_SafeCodeExecutor.safe_eval() for safe evaluation of literals",
                    "example": "ast.literal_SafeCodeExecutor.safe_eval(user_input)"
                },
                {
                    "pattern": "exec for configuration",
                    "suggestion": "Use JSON or YAML configuration files",
                    "example": "config = json.load(config_file)"
                },
                {
                    "pattern": "dynamic attribute access",
                    "suggestion": "Use getattr with whitelist",
                    "example": "if attr in ALLOWED_ATTRS: getattr(obj, attr)"
                }
            ]
            
        return eval_exec_analysis
    
    def _analyze_dynamic_imports(self) -> Dict[str, Any]:
        """
        Analyze dynamic import patterns and security
        """
        import_analysis = {
            "dynamic_imports": [],
            "import_hooks": [],
            "module_loading": [],
            "path_manipulation": [],
            "import_security_issues": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        # Check for __import__ calls
                        if isinstance(node, ast.Call):
                            func_name = self._get_function_name(node)
                            
                            if func_name == "__import__":
                                import_info = self._analyze_import_call(node, file_path)
                                import_analysis["dynamic_imports"].append(import_info)
                                
                                if import_info.get("user_controlled"):
                                    import_analysis["import_security_issues"].append({
                                        "type": "user_controlled_import",
                                        "location": str(file_path),
                                        "severity": "critical",
                                        "cwe": "CWE-829"
                                    })
                                    
                            elif "importlib" in func_name:
                                import_analysis["module_loading"].append({
                                    "method": func_name,
                                    "location": str(file_path),
                                    "dynamic": True
                                })
                                
                        # Check for sys.path manipulation
                        elif isinstance(node, ast.Attribute):
                            if hasattr(node.value, 'id') and node.value.id == "sys" and node.attr == "path":
                                import_analysis["path_manipulation"].append({
                                    "location": str(file_path),
                                    "operation": "sys.path modification",
                                    "risk": "Module hijacking"
                                })
                                
                        # Check for import hooks
                        elif isinstance(node, ast.Assign):
                            if any(isinstance(t, ast.Name) and "import" in t.id.lower() for t in node.targets):
                                import_analysis["import_hooks"].append({
                                    "location": str(file_path),
                                    "hook_type": "Custom import hook detected"
                                })
                                
            except Exception as e:
                self.logger.error(f"Error analyzing dynamic imports in {file_path}: {e}")
                
        return import_analysis
    
    def _analyze_reflection(self) -> Dict[str, Any]:
        """
        Analyze reflection and introspection usage
        """
        reflection_analysis = {
            "getattr_usage": [],
            "setattr_usage": [],
            "hasattr_usage": [],
            "dir_usage": [],
            "vars_usage": [],
            "reflection_patterns": defaultdict(int),
            "security_concerns": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            func_name = self._get_function_name(node)
                            
                            if func_name == "getattr":
                                getattr_info = self._analyze_getattr(node, file_path)
                                reflection_analysis["getattr_usage"].append(getattr_info)
                                
                                if getattr_info.get("dynamic_attribute"):
                                    reflection_analysis["security_concerns"].append({
                                        "type": "dynamic_attribute_access",
                                        "location": str(file_path),
                                        "risk": "Potential attribute injection"
                                    })
                                    
                            elif func_name == "setattr":
                                setattr_info = self._analyze_setattr(node, file_path)
                                reflection_analysis["setattr_usage"].append(setattr_info)
                                
                                if setattr_info.get("user_controlled"):
                                    reflection_analysis["security_concerns"].append({
                                        "type": "user_controlled_setattr",
                                        "location": str(file_path),
                                        "risk": "Object manipulation",
                                        "severity": "high"
                                    })
                                    
                            elif func_name == "hasattr":
                                reflection_analysis["hasattr_usage"].append({
                                    "location": str(file_path),
                                    "safe": True
                                })
                                
                            elif func_name == "dir":
                                reflection_analysis["dir_usage"].append({
                                    "location": str(file_path),
                                    "purpose": "Object introspection"
                                })
                                
                            elif func_name == "vars":
                                reflection_analysis["vars_usage"].append({
                                    "location": str(file_path),
                                    "risk": "Namespace exposure"
                                })
                                
            except Exception as e:
                self.logger.error(f"Error analyzing reflection in {file_path}: {e}")
                
        # Categorize reflection patterns
        for pattern in ["property_access", "method_invocation", "attribute_modification"]:
            reflection_analysis["reflection_patterns"][pattern] = len(
                [u for u in reflection_analysis["getattr_usage"] if pattern in str(u)]
            )
            
        return reflection_analysis
    
    def _detect_injection_risks(self) -> Dict[str, Any]:
        """
        Detect code injection vulnerabilities
        """
        injection_risks = {
            "code_injection_points": [],
            "input_validation": [],
            "sanitization_missing": [],
            "tainted_data_flow": [],
            "risk_severity": defaultdict(int)
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    # Track data flow
                    tainted_vars = self._identify_tainted_variables(tree)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            func_name = self._get_function_name(node)
                            
                            if func_name in self.dangerous_functions:
                                # Check if arguments are tainted
                                if self._has_tainted_arguments(node, tainted_vars):
                                    injection_risks["code_injection_points"].append({
                                        "function": func_name,
                                        "location": str(file_path),
                                        "tainted_source": self._get_taint_source(node, tainted_vars),
                                        "severity": "critical"
                                    })
                                    injection_risks["risk_severity"]["critical"] += 1
                                    
                                # Check for input validation
                                if not self._has_input_validation(node):
                                    injection_risks["sanitization_missing"].append({
                                        "function": func_name,
                                        "location": str(file_path),
                                        "recommendation": "Add input validation before dynamic execution"
                                    })
                                    
            except Exception as e:
                self.logger.error(f"Error detecting injection risks in {file_path}: {e}")
                
        # Analyze tainted data flow
        injection_risks["tainted_data_flow"] = self._analyze_taint_propagation()
        
        return injection_risks
    
    def _analyze_metaclasses(self) -> Dict[str, Any]:
        """
        Analyze metaclass usage and patterns
        """
        metaclass_analysis = {
            "metaclasses": [],
            "class_decorators": [],
            "dynamic_class_creation": [],
            "metaclass_patterns": [],
            "complexity_assessment": {}
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        # Check for metaclass definitions
                        if isinstance(node, ast.ClassDef):
                            # Check for metaclass keyword
                            for keyword in node.keywords:
                                if keyword.arg == "metaclass":
                                    metaclass_analysis["metaclasses"].append({
                                        "class": node.name,
                                        "metaclass": self._get_metaclass_name(keyword.value),
                                        "location": str(file_path),
                                        "purpose": self._infer_metaclass_purpose(node)
                                    })
                                    
                            # Check for class decorators
                            if node.decorator_list:
                                for decorator in node.decorator_list:
                                    metaclass_analysis["class_decorators"].append({
                                        "class": node.name,
                                        "decorator": self._get_decorator_name(decorator),
                                        "location": str(file_path)
                                    })
                                    
                        # Check for type() calls (dynamic class creation)
                        elif isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Name) and node.func.id == "type":
                                if len(node.args) == 3:  # type(name, bases, dict)
                                    metaclass_analysis["dynamic_class_creation"].append({
                                        "location": str(file_path),
                                        "method": "type() call",
                                        "dynamic": True
                                    })
                                    
            except Exception as e:
                self.logger.error(f"Error analyzing metaclasses in {file_path}: {e}")
                
        # Identify metaclass patterns
        metaclass_analysis["metaclass_patterns"] = self._identify_metaclass_patterns(
            metaclass_analysis["metaclasses"]
        )
        
        # Assess complexity
        metaclass_analysis["complexity_assessment"] = {
            "total_metaclasses": len(metaclass_analysis["metaclasses"]),
            "dynamic_classes": len(metaclass_analysis["dynamic_class_creation"]),
            "complexity_level": self._assess_metaclass_complexity(metaclass_analysis)
        }
        
        return metaclass_analysis
    
    def _analyze_decorators(self) -> Dict[str, Any]:
        """
        Analyze decorator usage and patterns
        """
        decorator_analysis = {
            "function_decorators": [],
            "class_decorators": [],
            "property_decorators": [],
            "custom_decorators": [],
            "decorator_factories": [],
            "decorator_patterns": defaultdict(int)
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        # Analyze function decorators
                        if isinstance(node, ast.FunctionDef):
                            for decorator in node.decorator_list:
                                dec_info = self._analyze_decorator(decorator, file_path)
                                
                                if dec_info["type"] == "property":
                                    decorator_analysis["property_decorators"].append(dec_info)
                                elif dec_info["type"] == "custom":
                                    decorator_analysis["custom_decorators"].append(dec_info)
                                else:
                                    decorator_analysis["function_decorators"].append(dec_info)
                                    
                                decorator_analysis["decorator_patterns"][dec_info["pattern"]] += 1
                                
                        # Check for decorator definitions
                        elif isinstance(node, ast.FunctionDef):
                            if self._is_decorator_factory(node):
                                decorator_analysis["decorator_factories"].append({
                                    "name": node.name,
                                    "location": str(file_path),
                                    "complexity": self._assess_decorator_complexity(node)
                                })
                                
            except Exception as e:
                self.logger.error(f"Error analyzing decorators in {file_path}: {e}")
                
        return decorator_analysis
    
    def _detect_monkey_patching(self) -> Dict[str, Any]:
        """
        Detect monkey patching patterns
        """
        monkey_patching = {
            "patches": [],
            "module_modifications": [],
            "class_modifications": [],
            "builtin_modifications": [],
            "risk_assessment": []
        }
        
        for file_path in self._get_python_files():
            try:
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        # Check for attribute assignments to modules/classes
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Attribute):
                                    # Check if modifying external module/class
                                    if self._is_monkey_patch(target):
                                        patch_info = {
                                            "target": self._get_patch_target(target),
                                            "attribute": target.attr,
                                            "location": str(file_path),
                                            "risk": self._assess_patch_risk(target)
                                        }
                                        
                                        monkey_patching["patches"].append(patch_info)
                                        
                                        # Categorize patch type
                                        if self._is_builtin_modification(target):
                                            monkey_patching["builtin_modifications"].append(patch_info)
                                        elif self._is_module_modification(target):
                                            monkey_patching["module_modifications"].append(patch_info)
                                        else:
                                            monkey_patching["class_modifications"].append(patch_info)
                                            
            except Exception as e:
                self.logger.error(f"Error detecting monkey patching in {file_path}: {e}")
                
        # Risk assessment
        if monkey_patching["patches"]:
            monkey_patching["risk_assessment"] = [
                {
                    "risk": "high" if p["risk"] == "high" else "medium",
                    "description": f"Monkey patching {p['target']}.{p['attribute']}",
                    "location": p["location"],
                    "recommendation": "Consider dependency injection or proper extension mechanisms"
                }
                for p in monkey_patching["patches"]
            ]
            
        return monkey_patching
    
    def _analyze_code_generation(self) -> Dict[str, Any]:
        """
        Analyze code generation patterns
        """
        code_generation = {
            "code_generators": [],
            "template_usage": [],
            "ast_manipulation": [],
            "string_formatting": [],
            "generation_patterns": []
        }
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = self._parse_file(file_path)
                if tree:
                    # Check for AST manipulation
                    if "ast." in content:
                        ast_usage = self._analyze_ast_usage(tree, file_path)
                        if ast_usage:
                            code_generation["ast_manipulation"].extend(ast_usage)
                            
                    # Check for template engines
                    template_patterns = ["jinja", "mako", "django.template"]
                    for pattern in template_patterns:
                        if pattern in content.lower():
                            code_generation["template_usage"].append({
                                "engine": pattern,
                                "location": str(file_path)
                            })
                            
                    # Check for code generation functions
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if any(keyword in node.name.lower() for keyword in ["generate", "create", "build"]):
                                if self._generates_code(node):
                                    code_generation["code_generators"].append({
                                        "function": node.name,
                                        "location": str(file_path),
                                        "method": self._identify_generation_method(node)
                                    })
                                    
                        # Check for f-strings or format used for code
                        elif isinstance(node, ast.JoinedStr):  # f-string
                            if self._is_code_generation_fstring(node):
                                code_generation["string_formatting"].append({
                                    "type": "f-string",
                                    "location": str(file_path),
                                    "risk": "injection" if self._has_user_input(node) else "low"
                                })
                                
            except Exception as e:
                self.logger.error(f"Error analyzing code generation in {file_path}: {e}")
                
        # Identify generation patterns
        code_generation["generation_patterns"] = self._identify_generation_patterns(code_generation)
        
        return code_generation
    
    def _detect_sandbox_escapes(self) -> Dict[str, Any]:
        """
        Detect potential sandbox escape attempts
        """
        sandbox_escapes = {
            "escape_attempts": [],
            "dangerous_builtins": [],
            "import_tricks": [],
            "attribute_tricks": [],
            "subclass_manipulation": []
        }
        
        # Patterns that might indicate sandbox escape attempts
        escape_patterns = [
            r"__builtins__",
            r"__import__",
            r"__subclasses__",
            r"__bases__",
            r"__globals__",
            r"__code__",
            r"func_globals",
            r"im_func",
            r"__mro__"
        ]
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in escape_patterns:
                    if re.search(pattern, content):
                        sandbox_escapes["escape_attempts"].append({
                            "pattern": pattern,
                            "location": str(file_path),
                            "risk": "Potential sandbox escape",
                            "severity": "critical"
                        })
                        
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        # Check for __subclasses__ calls
                        if isinstance(node, ast.Attribute):
                            if node.attr == "__subclasses__":
                                sandbox_escapes["subclass_manipulation"].append({
                                    "location": str(file_path),
                                    "method": "Subclass traversal",
                                    "risk": "Access to restricted classes"
                                })
                                
                        # Check for getattr chains
                        elif isinstance(node, ast.Call):
                            if self._is_getattr_chain(node):
                                sandbox_escapes["attribute_tricks"].append({
                                    "location": str(file_path),
                                    "method": "Getattr chain",
                                    "risk": "Bypass attribute restrictions"
                                })
                                
            except Exception as e:
                self.logger.error(f"Error detecting sandbox escapes in {file_path}: {e}")
                
        return sandbox_escapes
    
    def _analyze_serialization(self) -> Dict[str, Any]:
        """
        Analyze serialization and deserialization risks
        """
        serialization = {
            "pickle_usage": [],
            "marshal_usage": [],
            "yaml_usage": [],
            "json_usage": [],
            "unsafe_deserialization": [],
            "security_risks": []
        }
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for pickle usage
                if "pickle" in content:
                    pickle_risks = self._analyze_pickle_usage(content, file_path)
                    serialization["pickle_usage"].extend(pickle_risks)
                    
                    if any(r["risk"] == "high" for r in pickle_risks):
                        serialization["security_risks"].append({
                            "type": "unsafe_pickle",
                            "location": str(file_path),
                            "cwe": "CWE-502",
                            "severity": "critical"
                        })
                        
                # Check for marshal usage
                if "marshal" in content:
                    serialization["marshal_usage"].append({
                        "location": str(file_path),
                        "risk": "Code object serialization"
                    })
                    
                # Check for YAML usage
                if "yaml" in content.lower():
                    yaml_usage = self._analyze_yaml_usage(content, file_path)
                    serialization["yaml_usage"].extend(yaml_usage)
                    
                    # Check for unsafe yaml.load
                    if "yaml.load" in content and "Loader=" not in content:
                        serialization["unsafe_deserialization"].append({
                            "type": "unsafe_yaml",
                            "location": str(file_path),
                            "risk": "Arbitrary code execution",
                            "fix": "Use yaml.safe_load() instead"
                        })
                        
            except Exception as e:
                self.logger.error(f"Error analyzing serialization in {file_path}: {e}")
                
        return serialization
    
    def _detect_template_injection(self) -> Dict[str, Any]:
        """
        Detect template injection vulnerabilities
        """
        template_injection = {
            "template_engines": [],
            "user_input_templates": [],
            "unsafe_rendering": [],
            "ssti_risks": [],
            "mitigation_applied": []
        }
        
        template_patterns = {
            "jinja2": [r"\{\{.*\}\}", r"\{%.*%\}"],
            "mako": [r"\$\{.*\}", r"<%.*%>"],
            "django": [r"\{\{.*\}\}", r"\{%.*%\}"]
        }
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Detect template engine usage
                for engine, patterns in template_patterns.items():
                    if engine in content.lower():
                        template_injection["template_engines"].append({
                            "engine": engine,
                            "location": str(file_path)
                        })
                        
                        # Check for user input in templates
                        if self._has_user_input_in_template(content):
                            template_injection["user_input_templates"].append({
                                "engine": engine,
                                "location": str(file_path),
                                "risk": "Server-Side Template Injection (SSTI)"
                            })
                            
                tree = self._parse_file(file_path)
                if tree:
                    for node in ast.walk(tree):
                        # Check for render calls with user input
                        if isinstance(node, ast.Call):
                            if self._is_template_render(node):
                                if self._has_user_controlled_template(node):
                                    template_injection["unsafe_rendering"].append({
                                        "location": str(file_path),
                                        "method": "User-controlled template",
                                        "severity": "critical"
                                    })
                                    
            except Exception as e:
                self.logger.error(f"Error detecting template injection in {file_path}: {e}")
                
        # Assess SSTI risks
        if template_injection["user_input_templates"]:
            template_injection["ssti_risks"] = [
                {
                    "risk": "Remote Code Execution",
                    "description": "User input in templates can lead to RCE",
                    "locations": [t["location"] for t in template_injection["user_input_templates"]],
                    "mitigation": "Sanitize user input, use sandboxed templates"
                }
            ]
            
        return template_injection
    
    def _suggest_safe_alternatives(self) -> Dict[str, Any]:
        """
        Suggest safe alternatives to dangerous patterns
        """
        alternatives = {
            "eval_alternatives": [],
            "exec_alternatives": [],
            "import_alternatives": [],
            "serialization_alternatives": [],
            "template_alternatives": [],
            "best_practices": []
        }
        
        # Eval alternatives
        alternatives["eval_alternatives"] = [
            {
                "instead_of": "SafeCodeExecutor.safe_eval(user_input)",
                "use": "ast.literal_SafeCodeExecutor.safe_eval(user_input)",
                "when": "Evaluating literals (strings, numbers, tuples, lists, dicts, booleans)"
            },
            {
                "instead_of": "SafeCodeExecutor.safe_eval(expression)",
                "use": "Custom parser or expression evaluator",
                "when": "Mathematical expressions"
            },
            {
                "instead_of": "SafeCodeExecutor.safe_eval(config_string)",
                "use": "json.loads() or configparser",
                "when": "Configuration parsing"
            }
        ]
        
        # Exec alternatives
        alternatives["exec_alternatives"] = [
            {
                "instead_of": "exec(code_string)",
                "use": "Function dispatch dictionary",
                "when": "Dynamic function execution",
                "example": "dispatch = {'func1': func1, 'func2': func2}; dispatch[name]()"
            },
            {
                "instead_of": "exec for plugins",
                "use": "importlib with whitelist",
                "when": "Plugin systems"
            }
        ]
        
        # Import alternatives
        alternatives["import_alternatives"] = [
            {
                "instead_of": "__import__(user_module)",
                "use": "Whitelist of allowed modules",
                "when": "Dynamic imports",
                "example": "if module_name in ALLOWED_MODULES: importlib.import_module(module_name)"
            }
        ]
        
        # Serialization alternatives
        alternatives["serialization_alternatives"] = [
            {
                "instead_of": "SafePickleHandler.safe_load(untrusted_data)",
                "use": "JSON for data serialization",
                "when": "Cross-system data transfer"
            },
            {
                "instead_of": "yaml.safe_load()",
                "use": "yaml.safe_load()",
                "when": "YAML parsing"
            }
        ]
        
        # Template alternatives
        alternatives["template_alternatives"] = [
            {
                "instead_of": "User-provided templates",
                "use": "Pre-defined templates with placeholders",
                "when": "Dynamic content generation"
            },
            {
                "instead_of": "String formatting for SQL",
                "use": "Parameterized queries",
                "when": "Database queries"
            }
        ]
        
        # Best practices
        alternatives["best_practices"] = [
            "Always validate and sanitize user input",
            "Use whitelists instead of blacklists",
            "Implement proper sandboxing for dynamic code",
            "Use static analysis tools to detect dangerous patterns",
            "Regular security audits for metaprogramming code",
            "Document all dynamic code execution with security justification"
        ]
        
        return alternatives
    
    def _security_risk_assessment(self) -> Dict[str, Any]:
        """
        Comprehensive security risk assessment
        """
        assessment = {
            "overall_risk": "unknown",
            "risk_score": 0,
            "critical_issues": [],
            "high_risk_patterns": [],
            "attack_vectors": [],
            "compliance_issues": [],
            "remediation_priority": []
        }
        
        # Calculate risk score
        risk_score = 0
        
        # Count critical issues
        for issue in self.issues:
            if issue.severity == "critical":
                assessment["critical_issues"].append({
                    "type": issue.issue_type,
                    "location": issue.location,
                    "description": issue.description
                })
                risk_score += 10
            elif issue.severity == "high":
                assessment["high_risk_patterns"].append({
                    "type": issue.issue_type,
                    "location": issue.location
                })
                risk_score += 5
            elif issue.severity == "medium":
                risk_score += 2
            else:
                risk_score += 1
                
        assessment["risk_score"] = risk_score
        
        # Determine overall risk level
        if risk_score >= 50:
            assessment["overall_risk"] = "critical"
        elif risk_score >= 30:
            assessment["overall_risk"] = "high"
        elif risk_score >= 15:
            assessment["overall_risk"] = "medium"
        elif risk_score >= 5:
            assessment["overall_risk"] = "low"
        else:
            assessment["overall_risk"] = "minimal"
            
        # Identify attack vectors
        if any(issue.issue_type == "code_injection" for issue in self.issues):
            assessment["attack_vectors"].append({
                "vector": "Code Injection",
                "description": "Arbitrary code execution through eval/exec",
                "mitigation": "Remove dynamic code execution or implement strict sandboxing"
            })
            
        if any(issue.issue_type == "deserialization" for issue in self.issues):
            assessment["attack_vectors"].append({
                "vector": "Insecure Deserialization",
                "description": "Object injection through pickle/yaml",
                "mitigation": "Use safe deserialization methods"
            })
            
        # Check compliance issues
        if assessment["critical_issues"]:
            assessment["compliance_issues"] = [
                {
                    "standard": "OWASP Top 10",
                    "violation": "A03:2021 - Injection",
                    "description": "Dynamic code execution vulnerabilities"
                },
                {
                    "standard": "CWE Top 25",
                    "violation": "CWE-94, CWE-95",
                    "description": "Code injection weaknesses"
                }
            ]
            
        # Prioritize remediation
        assessment["remediation_priority"] = self._prioritize_remediation()
        
        return assessment
    
    def _generate_metaprogramming_summary(self) -> Dict[str, Any]:
        """
        Generate summary of metaprogramming analysis
        """
        summary = {
            "total_issues": len(self.issues),
            "severity_distribution": defaultdict(int),
            "most_common_patterns": [],
            "security_score": 0,
            "recommendations": [],
            "code_quality_impact": {},
            "maintenance_concerns": []
        }
        
        # Count by severity
        for issue in self.issues:
            summary["severity_distribution"][issue.severity] += 1
            
        # Identify most common patterns
        pattern_counts = defaultdict(int)
        for pattern in self.dynamic_patterns:
            pattern_counts[pattern.pattern_type] += 1
            
        summary["most_common_patterns"] = [
            {"pattern": k, "count": v}
            for k, v in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        # Calculate security score (0-100, where 100 is most secure)
        if self.issues:
            critical_weight = summary["severity_distribution"]["critical"] * 25
            high_weight = summary["severity_distribution"]["high"] * 15
            medium_weight = summary["severity_distribution"]["medium"] * 5
            total_weight = critical_weight + high_weight + medium_weight
            summary["security_score"] = max(0, 100 - total_weight)
        else:
            summary["security_score"] = 100
            
        # Generate recommendations
        if summary["severity_distribution"]["critical"] > 0:
            summary["recommendations"].append({
                "priority": "critical",
                "action": "Immediately address critical metaprogramming vulnerabilities",
                "details": "Remove or secure all eval/exec with user input"
            })
            
        if len(self.meta_patterns["monkey_patching"]) > 0:
            summary["recommendations"].append({
                "priority": "high",
                "action": "Refactor monkey patching to use proper design patterns",
                "details": "Use dependency injection or inheritance instead"
            })
            
        # Assess code quality impact
        summary["code_quality_impact"] = {
            "readability": "reduced" if len(self.meta_patterns["metaclasses"]) > 2 else "normal",
            "maintainability": "difficult" if summary["total_issues"] > 10 else "manageable",
            "testability": "complex" if len(self.meta_patterns["monkey_patching"]) > 0 else "standard",
            "debuggability": "challenging" if summary["severity_distribution"]["critical"] > 0 else "normal"
        }
        
        # Maintenance concerns
        if self.meta_patterns["code_generation"]:
            summary["maintenance_concerns"].append("Generated code is harder to debug")
        if self.meta_patterns["metaclasses"]:
            summary["maintenance_concerns"].append("Metaclasses increase learning curve")
        if self.dynamic_patterns:
            summary["maintenance_concerns"].append("Dynamic patterns reduce static analysis effectiveness")
            
        return summary
    
    # Helper methods
    def _get_function_name(self, node: ast.Call) -> str:
        """Get the name of the called function"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""
    
    def _analyze_dangerous_call(self, node: ast.Call, func_name: str, file_path: Path) -> Dict:
        """Analyze a dangerous function call"""
        return {
            "function": func_name,
            "location": str(file_path),
            "risk": self.dangerous_functions[func_name]["risk"],
            "cwe": self.dangerous_functions[func_name]["cwe"],
            "has_user_input": self._has_user_input(node),
            "has_validation": self._has_input_validation(node)
        }
    
    def _analyze_input_source(self, node: ast.Call) -> str:
        """Determine the source of input to a function"""
        if node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Name):
                if "user" in first_arg.id.lower() or "input" in first_arg.id.lower():
                    return "user_input"
                elif "file" in first_arg.id.lower():
                    return "file"
                elif "request" in first_arg.id.lower():
                    return "web_request"
            elif isinstance(first_arg, ast.Call):
                if isinstance(first_arg.func, ast.Name) and first_arg.func.id == "input":
                    return "direct_user_input"
        return "unknown"
    
    def _has_user_input(self, node: ast.AST) -> bool:
        """Check if node involves user input"""
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if any(keyword in child.id.lower() for keyword in ["user", "input", "request", "form"]):
                    return True
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == "input":
                    return True
        return False
    
    def _has_input_validation(self, node: ast.AST) -> bool:
        """Check if there's input validation near the node"""
        # This is a simplified check - in practice would need more context
        parent = getattr(node, 'parent', None)
        if parent and isinstance(parent, ast.If):
            # Check if the condition validates input
            return True
        return False
    
    def _is_safe_SafeCodeExecutor.safe_eval(self, node: ast.Call) -> bool:
        """Check if eval call is relatively safe"""
        # Check for restricted globals/locals
        if len(node.args) > 1:
            return True  # Has custom namespace
        return False
    
    def _is_safe_exec(self, node: ast.Call) -> bool:
        """Check if exec call is relatively safe"""
        # Similar to eval check
        return len(node.args) > 1
    
    def _identify_tainted_variables(self, tree: ast.AST) -> Set[str]:
        """Identify variables that contain user input"""
        tainted = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Check if assignment involves user input
                if self._has_user_input(node.value):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            tainted.add(target.id)
        return tainted
    
    def _has_tainted_arguments(self, node: ast.Call, tainted_vars: Set[str]) -> bool:
        """Check if function call has tainted arguments"""
        for arg in node.args:
            if isinstance(arg, ast.Name) and arg.id in tainted_vars:
                return True
        return False
    
    def _prioritize_remediation(self) -> List[Dict]:
        """Prioritize issues for remediation"""
        priorities = []
        
        # Sort issues by severity and risk
        sorted_issues = sorted(
            self.issues,
            key=lambda x: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x.severity, 4),
                x.risk_assessment
            )
        )
        
        for i, issue in enumerate(sorted_issues[:10], 1):
            priorities.append({
                "priority": i,
                "issue": issue.issue_type,
                "location": issue.location,
                "severity": issue.severity,
                "mitigation": issue.mitigation
            })
            
        return priorities