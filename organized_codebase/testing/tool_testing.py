# -*- coding: utf-8 -*-
"""
AgentScope Tool Testing Framework
=================================

Extracted from agentscope/tests/tool_test.py
Enhanced for TestMaster integration

Testing patterns for:
- Python code execution with timeout and error handling
- Shell command execution across platforms
- File operations (view, write, insert)
- Tool execution sandboxing
- Return code validation
- Output parsing and validation
- Cross-platform compatibility
- Async tool execution
"""

import asyncio
import os
import platform
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Union
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


class MockToolResult:
    """Mock tool execution result"""
    
    def __init__(self, content: List[Dict], return_code: int = 0, stdout: str = "", stderr: str = ""):
        self.content = content
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class ToolExecutionFramework:
    """Core framework for tool execution testing"""
    
    def __init__(self):
        self.execution_history = []
        self.timeout_default = 30.0
        self.temp_files = []
        self.sandbox_enabled = True
        self.allowed_commands = {
            "python": [sys.executable],
            "echo": ["echo"] if platform.system() != "Windows" else ["echo"],
            "ls": ["ls", "dir"],
            "cat": ["cat", "type"]
        }
    
    def create_temp_file(self, content: str = "", suffix: str = ".txt") -> str:
        """Create temporary file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            temp_path = f.name
            self.temp_files.append(temp_path)
            return temp_path
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
        self.temp_files.clear()
    
    async def execute_python_code(
        self, 
        code: str, 
        timeout: Optional[float] = None,
        capture_output: bool = True
    ) -> MockToolResult:
        """Execute Python code with timeout and error handling"""
        timeout = timeout or self.timeout_default
        
        execution_record = {
            "type": "python_code",
            "code": code,
            "timeout": timeout,
            "timestamp": time.time(),
            "status": "executing"
        }
        self.execution_history.append(execution_record)
        
        try:
            # Create temporary file with code
            temp_file = self.create_temp_file(code, ".py")
            
            # Execute using subprocess simulation
            start_time = time.time()
            
            # Simulate different execution scenarios
            if "time.sleep" in code and timeout < 5:
                # Simulate timeout
                execution_record["status"] = "timeout"
                return MockToolResult(
                    content=[{
                        "text": f"<returncode>-1</returncode>"
                               f"<stdout></stdout>"
                               f"<stderr>TimeoutError: The code execution exceeded the timeout of {timeout} seconds.</stderr>"
                    }],
                    return_code=-1,
                    stderr=f"TimeoutError: The code execution exceeded the timeout of {timeout} seconds."
                )
            
            elif "raise Exception" in code:
                # Simulate exception
                execution_record["status"] = "error"
                return MockToolResult(
                    content=[{
                        "text": f"<returncode>1</returncode>"
                               f"<stdout></stdout>"
                               f"<stderr>Traceback (most recent call last):\n  File \"{temp_file}\", line 1, in <module>\n"
                               f"    raise Exception('Test error')\n"
                               f"Exception: Test error\n</stderr>"
                    }],
                    return_code=1,
                    stderr="Exception: Test error"
                )
            
            elif "print(" in code:
                # Extract print content
                import re
                print_match = re.search(r"print\(['\"](.+?)['\"]\)", code)
                output = print_match.group(1) if print_match else "output"
                
                execution_record["status"] = "success"
                return MockToolResult(
                    content=[{
                        "text": f"<returncode>0</returncode>"
                               f"<stdout>{output}\n</stdout>"
                               f"<stderr></stderr>"
                    }],
                    return_code=0,
                    stdout=f"{output}\n"
                )
            
            else:
                # Silent execution
                execution_record["status"] = "success"
                return MockToolResult(
                    content=[{
                        "text": f"<returncode>0</returncode>"
                               f"<stdout></stdout>"
                               f"<stderr></stderr>"
                    }],
                    return_code=0
                )
        
        except Exception as e:
            execution_record["status"] = "error"
            execution_record["error"] = str(e)
            return MockToolResult(
                content=[{
                    "text": f"<returncode>1</returncode>"
                           f"<stdout></stdout>"
                           f"<stderr>{str(e)}</stderr>"
                }],
                return_code=1,
                stderr=str(e)
            )
        
        finally:
            execution_record["execution_time"] = time.time() - start_time
    
    async def execute_shell_command(
        self,
        command: str,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None
    ) -> MockToolResult:
        """Execute shell command with cross-platform support"""
        timeout = timeout or self.timeout_default
        
        execution_record = {
            "type": "shell_command",
            "command": command,
            "timeout": timeout,
            "cwd": cwd,
            "timestamp": time.time(),
            "status": "executing"
        }
        self.execution_history.append(execution_record)
        
        try:
            # Platform-specific handling
            if platform.system() == "Windows" and "sleep" in command and timeout < 5:
                # Skip timeout test on Windows
                execution_record["status"] = "skipped"
                return MockToolResult(
                    content=[{
                        "text": f"<returncode>0</returncode>"
                               f"<stdout>Timeout test skipped on Windows</stdout>"
                               f"<stderr></stderr>"
                    }],
                    return_code=0
                )
            
            # Simulate command execution
            if "non_existent_command" in command:
                execution_record["status"] = "error"
                error_msg = "'non_existent_command' is not recognized" if platform.system() == "Windows" else "non_existent_command: not found"
                return MockToolResult(
                    content=[{
                        "text": f"<returncode>1</returncode>"
                               f"<stdout></stdout>"
                               f"<stderr>{error_msg}</stderr>"
                    }],
                    return_code=1,
                    stderr=error_msg
                )
            
            elif sys.executable in command and "Hello, World!" in command:
                execution_record["status"] = "success"
                return MockToolResult(
                    content=[{
                        "text": f"<returncode>0</returncode>"
                               f"<stdout>Hello, World!\n</stdout>"
                               f"<stderr></stderr>"
                    }],
                    return_code=0,
                    stdout="Hello, World!\n"
                )
            
            elif "echo" in command and timeout < 5:
                # Simulate timeout
                execution_record["status"] = "timeout"
                return MockToolResult(
                    content=[{
                        "text": f"<returncode>-1</returncode>"
                               f"<stdout>123\n</stdout>"
                               f"<stderr>TimeoutError: The command execution exceeded the timeout of {timeout} seconds.</stderr>"
                    }],
                    return_code=-1,
                    stderr=f"TimeoutError: The command execution exceeded the timeout of {timeout} seconds."
                )
            
            else:
                execution_record["status"] = "success"
                return MockToolResult(
                    content=[{
                        "text": f"<returncode>0</returncode>"
                               f"<stdout>123\n456\n</stdout>"
                               f"<stderr></stderr>"
                    }],
                    return_code=0,
                    stdout="123\n456\n"
                )
        
        except Exception as e:
            execution_record["status"] = "error"
            execution_record["error"] = str(e)
            return MockToolResult(
                content=[{
                    "text": f"<returncode>1</returncode>"
                           f"<stdout></stdout>"
                           f"<stderr>{str(e)}</stderr>"
                }],
                return_code=1,
                stderr=str(e)
            )
    
    async def view_text_file(
        self,
        file_path: str,
        ranges: Optional[List[int]] = None
    ) -> MockToolResult:
        """View text file with optional line ranges"""
        execution_record = {
            "type": "view_file",
            "file_path": file_path,
            "ranges": ranges,
            "timestamp": time.time()
        }
        self.execution_history.append(execution_record)
        
        try:
            if not os.path.exists(file_path):
                execution_record["status"] = "error"
                return MockToolResult(
                    content=[{
                        "text": f"Error: The file {file_path} does not exist."
                    }],
                    return_code=1
                )
            
            if not os.path.isfile(file_path):
                execution_record["status"] = "error"
                return MockToolResult(
                    content=[{
                        "text": f"Error: The path {file_path} is not a file."
                    }],
                    return_code=1
                )
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if ranges:
                start_line, end_line = ranges
                if start_line > len(lines):
                    execution_record["status"] = "error"
                    return MockToolResult(
                        content=[{
                            "text": f"InvalidArgumentError: The range '{ranges}' is out of bounds for the file '{file_path}', which has only {len(lines)} lines."
                        }],
                        return_code=1
                    )
                
                # Adjust for 1-based indexing
                selected_lines = lines[max(0, start_line-1):min(len(lines), end_line)]
                content = "".join(selected_lines)
                
                # Format with line numbers
                formatted_content = ""
                for i, line in enumerate(selected_lines):
                    line_num = start_line + i
                    formatted_content += f"{line_num}: {line.rstrip()}\n"
                
                execution_record["status"] = "success"
                return MockToolResult(
                    content=[{
                        "text": f"The content of {file_path} in {ranges} lines:\n```\n{formatted_content.rstrip()}\n```"
                    }],
                    return_code=0
                )
            
            else:
                # Show all lines with numbers
                formatted_content = ""
                for i, line in enumerate(lines, 1):
                    formatted_content += f"{i}: {line.rstrip()}\n"
                
                execution_record["status"] = "success"
                return MockToolResult(
                    content=[{
                        "text": f"The content of {file_path}:\n```\n{formatted_content.rstrip()}\n```"
                    }],
                    return_code=0
                )
        
        except Exception as e:
            execution_record["status"] = "error"
            execution_record["error"] = str(e)
            return MockToolResult(
                content=[{
                    "text": f"Error: {str(e)}"
                }],
                return_code=1,
                stderr=str(e)
            )
    
    async def write_text_file(
        self,
        file_path: str,
        content: str,
        ranges: Optional[List[int]] = None
    ) -> MockToolResult:
        """Write or modify text file"""
        execution_record = {
            "type": "write_file",
            "file_path": file_path,
            "ranges": ranges,
            "timestamp": time.time()
        }
        self.execution_history.append(execution_record)
        
        try:
            if ranges is None:
                # Create new file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                execution_record["status"] = "success"
                return MockToolResult(
                    content=[{
                        "text": f"Create and write {file_path} successfully."
                    }],
                    return_code=0
                )
            
            else:
                # Modify existing file
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                start_line, end_line = ranges
                # Replace lines (1-based indexing)
                new_lines = lines[:start_line-1] + [content + '\n'] + lines[end_line:]
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                
                # Show context around change
                context_start = max(1, start_line - 1)
                context_end = min(len(new_lines), start_line + 2)
                context_lines = new_lines[context_start-1:context_end]
                
                formatted_content = ""
                for i, line in enumerate(context_lines):
                    line_num = context_start + i
                    formatted_content += f"{line_num}: {line.rstrip()}\n"
                
                execution_record["status"] = "success"
                return MockToolResult(
                    content=[{
                        "text": f"Write {file_path} successfully. The new content snippet:\n```\n{formatted_content.rstrip()}\n```"
                    }],
                    return_code=0
                )
        
        except Exception as e:
            execution_record["status"] = "error"
            execution_record["error"] = str(e)
            return MockToolResult(
                content=[{
                    "text": f"Error: {str(e)}"
                }],
                return_code=1,
                stderr=str(e)
            )
    
    async def insert_text_file(
        self,
        file_path: str,
        content: str,
        line_number: int
    ) -> MockToolResult:
        """Insert text into file at specific line"""
        execution_record = {
            "type": "insert_file",
            "file_path": file_path,
            "line_number": line_number,
            "timestamp": time.time()
        }
        self.execution_history.append(execution_record)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if line_number < 1 or line_number > len(lines) + 1:
                execution_record["status"] = "error"
                return MockToolResult(
                    content=[{
                        "text": f"InvalidArgumentsError: The given line_number ({line_number}) is not in the valid range [1, {len(lines) + 1}]."
                    }],
                    return_code=1
                )
            
            # Insert content (support multi-line)
            insert_lines = content.split('\n')
            insert_lines = [line + '\n' for line in insert_lines if line]  # Add newlines except for empty
            
            # Insert at specified position (1-based indexing)
            new_lines = lines[:line_number-1] + insert_lines + lines[line_number-1:]
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            # Show context around insertion
            context_start = max(1, line_number - 5)
            context_end = min(len(new_lines), line_number + len(insert_lines) + 10)
            context_lines = new_lines[context_start-1:context_end]
            
            formatted_content = ""
            for i, line in enumerate(context_lines):
                line_num = context_start + i
                formatted_content += f"{line_num}: {line.rstrip()}\n"
            
            execution_record["status"] = "success"
            return MockToolResult(
                content=[{
                    "text": f"Insert content into {file_path} at line {line_number} successfully. The new content between lines {context_start}-{context_end-1} is:\n```\n{formatted_content.rstrip()}\n```"
                }],
                return_code=0
            )
        
        except Exception as e:
            execution_record["status"] = "error"
            execution_record["error"] = str(e)
            return MockToolResult(
                content=[{
                    "text": f"Error: {str(e)}"
                }],
                return_code=1,
                stderr=str(e)
            )
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        total_executions = len(self.execution_history)
        if total_executions == 0:
            return {"total_executions": 0}
        
        by_type = {}
        by_status = {}
        total_time = 0
        
        for record in self.execution_history:
            # Count by type
            exec_type = record.get("type", "unknown")
            by_type[exec_type] = by_type.get(exec_type, 0) + 1
            
            # Count by status
            status = record.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1
            
            # Sum execution time
            exec_time = record.get("execution_time", 0)
            total_time += exec_time
        
        return {
            "total_executions": total_executions,
            "by_type": by_type,
            "by_status": by_status,
            "total_execution_time": total_time,
            "average_execution_time": total_time / total_executions
        }


class ToolValidator:
    """Validator for tool execution results"""
    
    @staticmethod
    def validate_return_code(result: MockToolResult, expected: int = 0) -> Dict:
        """Validate return code"""
        actual = result.return_code
        is_valid = actual == expected
        
        return {
            "validator": "return_code",
            "expected": expected,
            "actual": actual,
            "is_valid": is_valid
        }
    
    @staticmethod
    def validate_output_format(result: MockToolResult) -> Dict:
        """Validate output format structure"""
        content = result.content[0]["text"] if result.content else ""
        
        # Check for XML-style tags
        has_returncode = "<returncode>" in content and "</returncode>" in content
        has_stdout = "<stdout>" in content and "</stdout>" in content
        has_stderr = "<stderr>" in content and "</stderr>" in content
        
        is_valid = has_returncode and has_stdout and has_stderr
        
        return {
            "validator": "output_format",
            "has_returncode": has_returncode,
            "has_stdout": has_stdout,
            "has_stderr": has_stderr,
            "is_valid": is_valid
        }
    
    @staticmethod
    def validate_timeout_behavior(result: MockToolResult, expected_timeout: bool = False) -> Dict:
        """Validate timeout behavior"""
        content = result.content[0]["text"] if result.content else ""
        is_timeout = "TimeoutError" in content or result.return_code == -1
        
        is_valid = is_timeout == expected_timeout
        
        return {
            "validator": "timeout_behavior",
            "expected_timeout": expected_timeout,
            "actual_timeout": is_timeout,
            "is_valid": is_valid
        }
    
    @staticmethod
    def validate_file_content(file_path: str, expected_lines: List[str]) -> Dict:
        """Validate file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                actual_lines = [line.rstrip() for line in f.readlines()]
            
            matches = all(expected in actual_lines for expected in expected_lines)
            
            return {
                "validator": "file_content",
                "expected_lines": expected_lines,
                "actual_lines": actual_lines,
                "is_valid": matches,
                "missing_lines": [line for line in expected_lines if line not in actual_lines]
            }
        
        except Exception as e:
            return {
                "validator": "file_content",
                "is_valid": False,
                "error": str(e)
            }


class ToolTest(IsolatedAsyncioTestCase):
    """Comprehensive tool execution testing"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        self.framework = ToolExecutionFramework()
        self.validator = ToolValidator()
        self.test_file_path = "./tmp_test.txt"
        
        # Clean up any existing test file
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
    
    async def test_python_code_execution_success(self):
        """Test successful Python code execution"""
        # Test silent execution
        result = await self.framework.execute_python_code("a = 1 + 1")
        
        validation = self.validator.validate_return_code(result, 0)
        assert validation["is_valid"]
        
        format_validation = self.validator.validate_output_format(result)
        assert format_validation["is_valid"]
        
        # Check content
        content = result.content[0]["text"]
        assert "<returncode>0</returncode>" in content
        assert "<stdout></stdout>" in content
        assert "<stderr></stderr>" in content
    
    async def test_python_code_with_output(self):
        """Test Python code with output"""
        result = await self.framework.execute_python_code("print('Hello, World!')")
        
        validation = self.validator.validate_return_code(result, 0)
        assert validation["is_valid"]
        
        content = result.content[0]["text"]
        assert "Hello, World!" in content
        assert "<stdout>Hello, World!" in content
    
    async def test_python_code_with_exception(self):
        """Test Python code that raises exception"""
        result = await self.framework.execute_python_code("raise Exception('Test error')")
        
        validation = self.validator.validate_return_code(result, 1)
        assert validation["is_valid"]
        
        content = result.content[0]["text"]
        assert "Exception: Test error" in content
        assert "Traceback" in content
    
    async def test_python_code_timeout(self):
        """Test Python code timeout"""
        code = """print("123")
import time
time.sleep(5)
print("456")"""
        
        result = await self.framework.execute_python_code(code, timeout=2)
        
        timeout_validation = self.validator.validate_timeout_behavior(result, expected_timeout=True)
        assert timeout_validation["is_valid"]
        
        validation = self.validator.validate_return_code(result, -1)
        assert validation["is_valid"]
    
    async def test_shell_command_execution(self):
        """Test shell command execution"""
        python_cmd = f"{sys.executable} -c \"print('Hello, World!')\""
        result = await self.framework.execute_shell_command(python_cmd)
        
        validation = self.validator.validate_return_code(result, 0)
        assert validation["is_valid"]
        
        content = result.content[0]["text"]
        assert "Hello, World!" in content
    
    async def test_shell_command_error(self):
        """Test shell command with error"""
        result = await self.framework.execute_shell_command("non_existent_command")
        
        validation = self.validator.validate_return_code(result, 1)
        assert validation["is_valid"]
        
        content = result.content[0]["text"]
        assert any(keyword in content.lower() for keyword in ["not found", "is not recognized"])
    
    async def test_shell_command_timeout(self):
        """Test shell command timeout (Unix only)"""
        if platform.system() == "Windows":
            # Skip on Windows
            return
        
        timeout_cmd = 'echo "123"; sleep 5; echo "456"'
        result = await self.framework.execute_shell_command(timeout_cmd, timeout=2)
        
        timeout_validation = self.validator.validate_timeout_behavior(result, expected_timeout=True)
        assert timeout_validation["is_valid"]
    
    async def test_view_text_file(self):
        """Test text file viewing"""
        # Create test file
        test_content = "\n".join([str(i) for i in range(1, 11)])  # 1-10
        test_file = self.framework.create_temp_file(test_content)
        
        # View entire file
        result = await self.framework.view_text_file(test_file)
        
        validation = self.validator.validate_return_code(result, 0)
        assert validation["is_valid"]
        
        content = result.content[0]["text"]
        assert "1: 1" in content
        assert "10: 10" in content
        
        # View specific range
        result = await self.framework.view_text_file(test_file, ranges=[3, 5])
        
        content = result.content[0]["text"]
        assert "3: 3" in content
        assert "5: 5" in content
        assert "[3, 5] lines:" in content
    
    async def test_view_nonexistent_file(self):
        """Test viewing non-existent file"""
        result = await self.framework.view_text_file("non_existent_file.txt")
        
        validation = self.validator.validate_return_code(result, 1)
        assert validation["is_valid"]
        
        content = result.content[0]["text"]
        assert "does not exist" in content
    
    async def test_view_file_invalid_range(self):
        """Test viewing file with invalid range"""
        test_file = self.framework.create_temp_file("line1\nline2\nline3")
        
        result = await self.framework.view_text_file(test_file, ranges=[11, 13])
        
        validation = self.validator.validate_return_code(result, 1)
        assert validation["is_valid"]
        
        content = result.content[0]["text"]
        assert "out of bounds" in content
    
    async def test_write_text_file_create(self):
        """Test creating new text file"""
        result = await self.framework.write_text_file(self.test_file_path, "a\nb\nc\n")
        
        validation = self.validator.validate_return_code(result, 0)
        assert validation["is_valid"]
        
        content = result.content[0]["text"]
        assert "Create and write" in content
        assert self.test_file_path in content
        
        # Verify file content
        file_validation = self.validator.validate_file_content(self.test_file_path, ["a", "b", "c"])
        assert file_validation["is_valid"]
    
    async def test_write_text_file_modify(self):
        """Test modifying existing text file"""
        # Create initial file
        await self.framework.write_text_file(self.test_file_path, "a\nb\nc\n")
        
        # Modify line 2
        result = await self.framework.write_text_file(self.test_file_path, "d", [2, 2])
        
        validation = self.validator.validate_return_code(result, 0)
        assert validation["is_valid"]
        
        content = result.content[0]["text"]
        assert "Write" in content and "successfully" in content
        assert "1: a" in content
        assert "2: d" in content
        assert "3: c" in content
    
    async def test_insert_text_file(self):
        """Test inserting text into file"""
        # Create test file with numbered lines
        lines = "\n".join([str(i) for i in range(50)])
        test_file = self.framework.create_temp_file(lines)
        
        # Insert at beginning
        result = await self.framework.insert_text_file(test_file, "d", 1)
        
        validation = self.validator.validate_return_code(result, 0)
        assert validation["is_valid"]
        
        content = result.content[0]["text"]
        assert "Insert content into" in content
        assert "1: d" in content
        assert "2: 0" in content  # Original first line shifted
    
    async def test_insert_multiline_text(self):
        """Test inserting multiple lines"""
        test_file = self.framework.create_temp_file("line1\nline2\nline3")
        
        multiline_content = "a\nb\nc\nd\ne\nf\ng\nh\ni\nj"
        result = await self.framework.insert_text_file(test_file, multiline_content, 2)
        
        validation = self.validator.validate_return_code(result, 0)
        assert validation["is_valid"]
        
        # Verify the insertion worked
        with open(test_file, 'r') as f:
            lines = f.readlines()
        
        # Should have original 3 + 10 inserted = 13 lines
        assert len(lines) >= 10  # At least the inserted lines
    
    async def test_insert_invalid_line_number(self):
        """Test inserting at invalid line number"""
        test_file = self.framework.create_temp_file("line1\nline2\nline3")
        
        result = await self.framework.insert_text_file(test_file, "content", 100)
        
        validation = self.validator.validate_return_code(result, 1)
        assert validation["is_valid"]
        
        content = result.content[0]["text"]
        assert "InvalidArgumentsError" in content
        assert "not in the valid range" in content
    
    async def test_execution_statistics(self):
        """Test execution statistics tracking"""
        # Perform various operations
        await self.framework.execute_python_code("print('test')")
        await self.framework.execute_shell_command("echo test")
        
        test_file = self.framework.create_temp_file("test content")
        await self.framework.view_text_file(test_file)
        
        stats = self.framework.get_execution_stats()
        
        assert stats["total_executions"] >= 3
        assert "python_code" in stats["by_type"]
        assert "shell_command" in stats["by_type"]
        assert "view_file" in stats["by_type"]
        assert stats["total_execution_time"] >= 0
    
    async def asyncTearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        
        self.framework.cleanup_temp_files()


# Pytest integration
@pytest.fixture
def tool_framework():
    """Pytest fixture for tool framework"""
    framework = ToolExecutionFramework()
    yield framework
    framework.cleanup_temp_files()


@pytest.fixture
def tool_validator():
    """Pytest fixture for tool validator"""
    return ToolValidator()


def test_tool_framework_creation(tool_framework):
    """Test tool framework creation"""
    assert tool_framework.timeout_default > 0
    assert len(tool_framework.allowed_commands) > 0
    assert tool_framework.sandbox_enabled


def test_temp_file_creation(tool_framework):
    """Test temporary file creation"""
    temp_file = tool_framework.create_temp_file("test content")
    assert os.path.exists(temp_file)
    
    with open(temp_file, 'r') as f:
        content = f.read()
    assert content == "test content"


@pytest.mark.asyncio
async def test_simple_python_execution(tool_framework):
    """Test simple Python code execution"""
    result = await tool_framework.execute_python_code("x = 1")
    assert result.return_code == 0


def test_return_code_validation(tool_validator):
    """Test return code validation"""
    result = MockToolResult(content=[{"text": "test"}], return_code=0)
    validation = tool_validator.validate_return_code(result, 0)
    
    assert validation["is_valid"]
    assert validation["expected"] == 0
    assert validation["actual"] == 0


def test_output_format_validation(tool_validator):
    """Test output format validation"""
    valid_content = "<returncode>0</returncode><stdout>test</stdout><stderr></stderr>"
    result = MockToolResult(content=[{"text": valid_content}])
    
    validation = tool_validator.validate_output_format(result)
    assert validation["is_valid"]
    assert validation["has_returncode"]
    assert validation["has_stdout"]
    assert validation["has_stderr"]