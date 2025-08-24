"""
AutoGen Testing Patterns - AGENT B COMPREHENSIVE TESTING EXCELLENCE
====================================================================

Extracted testing patterns from autogen repository for enhanced code validation and automation testing.
Focus: Code block validation, fixup automation, task running patterns, project discovery.

AGENT B Enhancement: Phase 1.7 - AutoGen Pattern Integration
- Markdown code block syntax checking
- Automated file fixup and import correction
- Task discovery and execution patterns
- Project workspace management
- Code validation with external tools (pyright)
- Pattern substitution and file processing
"""

import asyncio
import glob
import subprocess
import tempfile
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import tomli
from unittest.mock import Mock, patch


class CodeBlockValidationPatterns:
    """
    Code block validation patterns extracted from autogen check_md_code_blocks.py
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @dataclass
    class CodeBlockInfo:
        """Information about a code block"""
        content: str
        line_number: int
        language: str = "python"
        file_path: str = ""
        has_imports: bool = False
        validation_result: Optional[Dict[str, Any]] = None
    
    @dataclass
    class ValidationResult:
        """Result of code block validation"""
        success: bool
        errors: List[str] = field(default_factory=list)
        warnings: List[str] = field(default_factory=list)
        output: str = ""
        tool_used: str = ""
        execution_time: float = 0.0
    
    def extract_python_code_blocks(self, markdown_file_path: str) -> List[CodeBlockInfo]:
        """Extract Python code blocks from a Markdown file"""
        try:
            with open(markdown_file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
        except Exception as e:
            self.logger.error(f"Failed to read file {markdown_file_path}: {e}")
            return []
        
        code_blocks = []
        in_code_block = False
        current_block = []
        current_language = "python"
        start_line = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            if line_stripped.startswith("```"):
                if not in_code_block:
                    # Starting a code block
                    in_code_block = True
                    current_block = []
                    start_line = i + 1
                    
                    # Extract language from code fence
                    if len(line_stripped) > 3:
                        current_language = line_stripped[3:].strip()
                    else:
                        current_language = "python"
                else:
                    # Ending a code block
                    in_code_block = False
                    
                    if current_language.lower() in ["python", "py"]:
                        block_content = "\n".join(current_block)
                        has_imports = self._check_for_imports(block_content)
                        
                        code_block = CodeBlockInfo(
                            content=block_content,
                            line_number=start_line,
                            language=current_language,
                            file_path=markdown_file_path,
                            has_imports=has_imports
                        )
                        code_blocks.append(code_block)
            
            elif in_code_block:
                current_block.append(line.rstrip())
        
        return code_blocks
    
    def _check_for_imports(self, code_content: str) -> bool:
        """Check if code block contains import statements"""
        import_patterns = [
            r"^\s*import\s+\w+",
            r"^\s*from\s+\w+\s+import",
            r"import\s+autogen",
            r"from\s+autogen"
        ]
        
        for pattern in import_patterns:
            if re.search(pattern, code_content, re.MULTILINE):
                return True
        
        return False
    
    def _should_skip_block(self, code_block: CodeBlockInfo, 
                          required_modules: List[str] = None) -> bool:
        """Determine if code block should be skipped"""
        if required_modules is None:
            required_modules = ["autogen_agentchat", "autogen_core", "autogen_ext"]
        
        for module in required_modules:
            import_patterns = [f"import {module}", f"from {module}"]
            for pattern in import_patterns:
                if pattern in code_block.content:
                    return False
        
        return True
    
    def validate_code_block_with_pyright(self, code_block: CodeBlockInfo) -> ValidationResult:
        """Validate code block using pyright"""
        start_time = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w', encoding='utf-8') as temp_file:
                temp_file.write(code_block.content)
                temp_file.flush()
                
                # Run pyright on the temporary file
                result = subprocess.run(
                    ["pyright", temp_file.name], 
                    capture_output=True, 
                    text=True,
                    timeout=30  # 30 second timeout
                )
                
                execution_time = (asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0) - start_time
                
                validation_result = ValidationResult(
                    success=result.returncode == 0,
                    output=result.stdout if result.stdout else result.stderr,
                    tool_used="pyright",
                    execution_time=execution_time
                )
                
                if result.returncode != 0:
                    validation_result.errors.append(f"Pyright validation failed: {result.stdout}")
                
                return validation_result
                
        except subprocess.TimeoutExpired:
            return ValidationResult(
                success=False,
                errors=["Pyright validation timed out"],
                tool_used="pyright",
                execution_time=30.0
            )
        except FileNotFoundError:
            return ValidationResult(
                success=False,
                errors=["Pyright not found - please install pyright"],
                tool_used="pyright"
            )
        except Exception as e:
            return ValidationResult(
                success=False,
                errors=[f"Validation error: {str(e)}"],
                tool_used="pyright"
            )
    
    def validate_code_block_with_syntax_check(self, code_block: CodeBlockInfo) -> ValidationResult:
        """Validate code block using Python syntax checking"""
        try:
            compile(code_block.content, f"<{code_block.file_path}:{code_block.line_number}>", "exec")
            return ValidationResult(
                success=True,
                tool_used="python_compile"
            )
        except SyntaxError as e:
            return ValidationResult(
                success=False,
                errors=[f"Syntax error: {e.msg} at line {e.lineno}"],
                tool_used="python_compile"
            )
        except Exception as e:
            return ValidationResult(
                success=False,
                errors=[f"Compilation error: {str(e)}"],
                tool_used="python_compile"
            )
    
    def check_code_blocks_in_files(self, markdown_file_paths: List[str], 
                                  use_pyright: bool = True,
                                  skip_non_autogen: bool = True) -> Dict[str, Any]:
        """Check Python code blocks in multiple Markdown files"""
        all_results = {}
        files_with_errors = []
        total_blocks = 0
        total_errors = 0
        
        for file_path in markdown_file_paths:
            self.logger.info(f"Processing file: {file_path}")
            
            # Extract code blocks
            code_blocks = self.extract_python_code_blocks(file_path)
            file_results = []
            file_errors = 0
            
            for code_block in code_blocks:
                total_blocks += 1
                
                # Skip blocks without required imports if configured
                if skip_non_autogen and self._should_skip_block(code_block):
                    validation_result = ValidationResult(
                        success=True,
                        tool_used="skipped"
                    )
                    validation_result.warnings.append("Skipped - no autogen imports")
                else:
                    # Validate the code block
                    if use_pyright:
                        validation_result = self.validate_code_block_with_pyright(code_block)
                    else:
                        validation_result = self.validate_code_block_with_syntax_check(code_block)
                
                code_block.validation_result = validation_result
                file_results.append(code_block)
                
                if not validation_result.success:
                    file_errors += 1
                    total_errors += 1
            
            all_results[file_path] = {
                'code_blocks': file_results,
                'total_blocks': len(code_blocks),
                'errors': file_errors,
                'success': file_errors == 0
            }
            
            if file_errors > 0:
                files_with_errors.append(file_path)
        
        return {
            'file_results': all_results,
            'files_with_errors': files_with_errors,
            'total_files': len(markdown_file_paths),
            'files_with_errors_count': len(files_with_errors),
            'total_code_blocks': total_blocks,
            'total_errors': total_errors,
            'success_rate': ((total_blocks - total_errors) / total_blocks) * 100 if total_blocks > 0 else 0,
            'overall_success': total_errors == 0
        }
    
    async def check_code_blocks_async(self, markdown_file_paths: List[str]) -> Dict[str, Any]:
        """Asynchronously check code blocks in multiple files"""
        tasks = []
        
        for file_path in markdown_file_paths:
            task = asyncio.create_task(self._check_file_async(file_path))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_results = {}
        files_with_errors = []
        total_blocks = 0
        total_errors = 0
        
        for i, result in enumerate(results):
            file_path = markdown_file_paths[i]
            
            if isinstance(result, Exception):
                all_results[file_path] = {
                    'error': str(result),
                    'success': False
                }
                files_with_errors.append(file_path)
            else:
                all_results[file_path] = result
                total_blocks += result['total_blocks']
                total_errors += result['errors']
                
                if result['errors'] > 0:
                    files_with_errors.append(file_path)
        
        return {
            'file_results': all_results,
            'files_with_errors': files_with_errors,
            'total_files': len(markdown_file_paths),
            'files_with_errors_count': len(files_with_errors),
            'total_code_blocks': total_blocks,
            'total_errors': total_errors,
            'success_rate': ((total_blocks - total_errors) / total_blocks) * 100 if total_blocks > 0 else 0,
            'overall_success': total_errors == 0
        }
    
    async def _check_file_async(self, file_path: str) -> Dict[str, Any]:
        """Asynchronously check a single file"""
        return await asyncio.to_thread(
            lambda: self.check_code_blocks_in_files([file_path])['file_results'][file_path]
        )


class FileFixupPatterns:
    """
    File fixup patterns extracted from autogen fixup_generated_files.py
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @dataclass
    class FixupRule:
        """Rule for fixing files"""
        pattern: str
        replacement: str
        description: str = ""
        regex: bool = False
    
    @dataclass
    class FixupResult:
        """Result of file fixup operation"""
        file_path: str
        changes_made: int = 0
        rules_applied: List[str] = field(default_factory=list)
        success: bool = True
        error: Optional[str] = None
        original_content: Optional[str] = None
        modified_content: Optional[str] = None
    
    def create_import_fixup_rules(self) -> List[FixupRule]:
        """Create standard import fixup rules"""
        return [
            FixupRule(
                pattern="\nimport agent_worker_pb2 as agent__worker__pb2\n",
                replacement="\nfrom . import agent_worker_pb2 as agent__worker__pb2\n",
                description="Fix agent_worker_pb2 import with alias"
            ),
            FixupRule(
                pattern="\nimport agent_worker_pb2\n",
                replacement="\nfrom . import agent_worker_pb2\n",
                description="Fix agent_worker_pb2 import without alias"
            ),
            FixupRule(
                pattern="\nimport cloudevent_pb2 as cloudevent__pb2\n",
                replacement="\nfrom . import cloudevent_pb2 as cloudevent__pb2\n",
                description="Fix cloudevent_pb2 import with alias"
            ),
            FixupRule(
                pattern="\nimport cloudevent_pb2\n",
                replacement="\nfrom . import cloudevent_pb2\n",
                description="Fix cloudevent_pb2 import without alias"
            )
        ]
    
    def apply_fixup_rules(self, file_path: str, rules: List[FixupRule]) -> FixupResult:
        """Apply fixup rules to a file"""
        result = FixupResult(file_path=file_path)
        
        try:
            # Read original content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            result.original_content = content
            modified_content = content
            
            # Apply each rule
            for rule in rules:
                if rule.regex:
                    import re
                    new_content = re.sub(rule.pattern, rule.replacement, modified_content)
                else:
                    new_content = modified_content.replace(rule.pattern, rule.replacement)
                
                if new_content != modified_content:
                    result.changes_made += modified_content.count(rule.pattern)
                    result.rules_applied.append(rule.description or rule.pattern)
                    modified_content = new_content
            
            result.modified_content = modified_content
            
            # Write modified content if changes were made
            if result.changes_made > 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(modified_content)
                
                self.logger.info(f"Applied {result.changes_made} fixes to {file_path}")
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            self.logger.error(f"Failed to fix file {file_path}: {e}")
        
        return result
    
    def fixup_files_batch(self, file_paths: List[str], 
                         custom_rules: List[FixupRule] = None) -> Dict[str, Any]:
        """Apply fixup rules to multiple files"""
        if custom_rules is None:
            rules = self.create_import_fixup_rules()
        else:
            rules = custom_rules
        
        results = []
        total_changes = 0
        successful_files = 0
        
        for file_path in file_paths:
            if not Path(file_path).exists():
                self.logger.warning(f"File not found: {file_path}")
                results.append(FixupResult(
                    file_path=file_path,
                    success=False,
                    error="File not found"
                ))
                continue
            
            result = self.apply_fixup_rules(file_path, rules)
            results.append(result)
            
            if result.success:
                successful_files += 1
                total_changes += result.changes_made
        
        return {
            'results': results,
            'total_files': len(file_paths),
            'successful_files': successful_files,
            'failed_files': len(file_paths) - successful_files,
            'total_changes': total_changes,
            'rules_used': len(rules),
            'success_rate': (successful_files / len(file_paths)) * 100 if file_paths else 0
        }
    
    def create_custom_fixup_rule(self, pattern: str, replacement: str, 
                                description: str = "", regex: bool = False) -> FixupRule:
        """Create a custom fixup rule"""
        return FixupRule(
            pattern=pattern,
            replacement=replacement,
            description=description,
            regex=regex
        )
    
    def validate_fixup_rules(self, rules: List[FixupRule]) -> Dict[str, Any]:
        """Validate fixup rules"""
        validation_results = []
        
        for i, rule in enumerate(rules):
            rule_result = {
                'rule_index': i,
                'pattern': rule.pattern,
                'replacement': rule.replacement,
                'valid': True,
                'warnings': []
            }
            
            # Check for empty patterns
            if not rule.pattern:
                rule_result['valid'] = False
                rule_result['warnings'].append("Empty pattern")
            
            # Check for regex syntax if regex is enabled
            if rule.regex:
                try:
                    import re
                    re.compile(rule.pattern)
                except re.error as e:
                    rule_result['valid'] = False
                    rule_result['warnings'].append(f"Invalid regex: {e}")
            
            # Check for potentially dangerous replacements
            if rule.replacement == "":
                rule_result['warnings'].append("Empty replacement - will delete matches")
            
            validation_results.append(rule_result)
        
        valid_rules = sum(1 for r in validation_results if r['valid'])
        
        return {
            'rule_validations': validation_results,
            'total_rules': len(rules),
            'valid_rules': valid_rules,
            'invalid_rules': len(rules) - valid_rules,
            'all_valid': valid_rules == len(rules)
        }


class TaskRunnerPatterns:
    """
    Task runner patterns extracted from autogen run_task_in_pkgs_if_exist.py
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @dataclass
    class ProjectInfo:
        """Information about a discovered project"""
        path: Path
        pyproject_file: Path
        tasks: Set[str] = field(default_factory=set)
        has_include: bool = False
        include_file: Optional[Path] = None
    
    @dataclass
    class TaskExecutionResult:
        """Result of task execution"""
        project_path: Path
        task_name: str
        success: bool
        return_code: Optional[int] = None
        output: str = ""
        error: str = ""
        execution_time: float = 0.0
    
    def discover_projects(self, workspace_pyproject_file: Path) -> List[ProjectInfo]:
        """Discover projects in a workspace"""
        try:
            with workspace_pyproject_file.open("rb") as f:
                data = tomli.load(f)
        except Exception as e:
            self.logger.error(f"Failed to read workspace file {workspace_pyproject_file}: {e}")
            return []
        
        # Get workspace configuration
        workspace_config = data.get("tool", {}).get("uv", {}).get("workspace", {})
        projects = workspace_config.get("members", [])
        exclude = workspace_config.get("exclude", [])
        
        all_projects: List[Path] = []
        
        # Process included projects
        for project in projects:
            if "*" in project:
                globbed = glob.glob(str(project), root_dir=workspace_pyproject_file.parent)
                globbed_paths = [Path(p) for p in globbed]
                all_projects.extend(globbed_paths)
            else:
                all_projects.append(Path(project))
        
        # Process excluded projects
        for project in exclude:
            if "*" in project:
                globbed = glob.glob(str(project), root_dir=workspace_pyproject_file.parent)
                globbed_paths = [Path(p) for p in globbed]
                all_projects = [p for p in all_projects if p not in globbed_paths]
            else:
                all_projects = [p for p in all_projects if p != Path(project)]
        
        # Create ProjectInfo objects
        project_infos = []
        for project_path in all_projects:
            full_path = workspace_pyproject_file.parent / project_path
            pyproject_file = full_path / "pyproject.toml"
            
            if pyproject_file.exists():
                project_info = ProjectInfo(
                    path=full_path,
                    pyproject_file=pyproject_file
                )
                
                # Extract tasks
                tasks = self.extract_poe_tasks(pyproject_file)
                project_info.tasks = tasks
                
                project_infos.append(project_info)
        
        return project_infos
    
    def extract_poe_tasks(self, pyproject_file: Path) -> Set[str]:
        """Extract Poe tasks from a pyproject.toml file"""
        try:
            with pyproject_file.open("rb") as f:
                data = tomli.load(f)
        except Exception as e:
            self.logger.error(f"Failed to read pyproject file {pyproject_file}: {e}")
            return set()
        
        # Get tasks from poe configuration
        tasks = set(data.get("tool", {}).get("poe", {}).get("tasks", {}).keys())
        
        # Check for included task files
        include_file_path = data.get("tool", {}).get("poe", {}).get("include", None)
        if include_file_path:
            include_file = pyproject_file.parent / include_file_path
            if include_file.exists():
                included_tasks = self.extract_poe_tasks(include_file)
                tasks = tasks.union(included_tasks)
        
        return tasks
    
    def find_projects_with_task(self, projects: List[ProjectInfo], 
                               task_name: str) -> List[ProjectInfo]:
        """Find projects that have a specific task"""
        return [project for project in projects if task_name in project.tasks]
    
    def execute_task_in_project(self, project: ProjectInfo, task_name: str, 
                               args: List[str] = None) -> TaskExecutionResult:
        """Execute a task in a specific project"""
        start_time = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        
        if task_name not in project.tasks:
            return TaskExecutionResult(
                project_path=project.path,
                task_name=task_name,
                success=False,
                error=f"Task '{task_name}' not found in project"
            )
        
        try:
            # Mock execution for testing - in real implementation would use PoeThePoet
            import subprocess
            
            cmd = ["poe", task_name]
            if args:
                cmd.extend(args)
            
            result = subprocess.run(
                cmd,
                cwd=project.path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            execution_time = (asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0) - start_time
            
            return TaskExecutionResult(
                project_path=project.path,
                task_name=task_name,
                success=result.returncode == 0,
                return_code=result.returncode,
                output=result.stdout,
                error=result.stderr,
                execution_time=execution_time
            )
            
        except subprocess.TimeoutExpired:
            return TaskExecutionResult(
                project_path=project.path,
                task_name=task_name,
                success=False,
                error="Task execution timed out",
                execution_time=300.0
            )
        except Exception as e:
            return TaskExecutionResult(
                project_path=project.path,
                task_name=task_name,
                success=False,
                error=str(e),
                execution_time=(asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0) - start_time
            )
    
    def run_task_in_all_projects(self, workspace_file: Path, task_name: str, 
                                args: List[str] = None) -> Dict[str, Any]:
        """Run a task in all projects that have it"""
        # Discover projects
        projects = self.discover_projects(workspace_file)
        
        # Find projects with the task
        projects_with_task = self.find_projects_with_task(projects, task_name)
        
        # Execute task in each project
        results = []
        successful_executions = 0
        
        for project in projects_with_task:
            result = self.execute_task_in_project(project, task_name, args)
            results.append(result)
            
            if result.success:
                successful_executions += 1
        
        return {
            'task_name': task_name,
            'total_projects': len(projects),
            'projects_with_task': len(projects_with_task),
            'executions': results,
            'successful_executions': successful_executions,
            'failed_executions': len(projects_with_task) - successful_executions,
            'success_rate': (successful_executions / len(projects_with_task)) * 100 if projects_with_task else 0,
            'overall_success': successful_executions == len(projects_with_task)
        }
    
    async def run_task_in_all_projects_async(self, workspace_file: Path, 
                                           task_name: str, args: List[str] = None) -> Dict[str, Any]:
        """Asynchronously run a task in all projects"""
        # Discover projects
        projects = await asyncio.to_thread(self.discover_projects, workspace_file)
        
        # Find projects with the task
        projects_with_task = self.find_projects_with_task(projects, task_name)
        
        # Execute tasks concurrently
        tasks = []
        for project in projects_with_task:
            task = asyncio.create_task(
                asyncio.to_thread(self.execute_task_in_project, project, task_name, args)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        execution_results = []
        successful_executions = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                execution_result = TaskExecutionResult(
                    project_path=projects_with_task[i].path,
                    task_name=task_name,
                    success=False,
                    error=str(result)
                )
            else:
                execution_result = result
                if result.success:
                    successful_executions += 1
            
            execution_results.append(execution_result)
        
        return {
            'task_name': task_name,
            'total_projects': len(projects),
            'projects_with_task': len(projects_with_task),
            'executions': execution_results,
            'successful_executions': successful_executions,
            'failed_executions': len(projects_with_task) - successful_executions,
            'success_rate': (successful_executions / len(projects_with_task)) * 100 if projects_with_task else 0,
            'overall_success': successful_executions == len(projects_with_task),
            'concurrent_execution': True
        }
    
    def get_workspace_summary(self, workspace_file: Path) -> Dict[str, Any]:
        """Get summary of workspace projects and tasks"""
        projects = self.discover_projects(workspace_file)
        
        all_tasks = set()
        task_distribution = {}
        
        for project in projects:
            for task in project.tasks:
                all_tasks.add(task)
                if task not in task_distribution:
                    task_distribution[task] = []
                task_distribution[task].append(str(project.path))
        
        return {
            'workspace_file': str(workspace_file),
            'total_projects': len(projects),
            'project_paths': [str(p.path) for p in projects],
            'total_unique_tasks': len(all_tasks),
            'all_tasks': list(all_tasks),
            'task_distribution': task_distribution,
            'projects_per_task': {
                task: len(projects) for task, projects in task_distribution.items()
            }
        }


# Export all patterns
__all__ = [
    'CodeBlockValidationPatterns',
    'FileFixupPatterns',
    'TaskRunnerPatterns'
]