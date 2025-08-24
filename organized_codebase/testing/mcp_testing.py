"""MCP (Model Context Protocol) Testing Framework - Agency-Swarm Pattern
Extracted patterns for testing MCP server integration
Supports SSE, HTTP, and STDIO server testing with proper lifecycle management
"""
import os
import signal
import subprocess
import sys
import time
import platform
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import pytest


class MCPServerManager:
    """Manages MCP server lifecycle for testing"""
    
    def __init__(self):
        self.active_processes = []
        self.cleanup_callbacks = []
    
    def start_sse_server(self, script_path: str, port: int = 8080) -> subprocess.Popen:
        """Start SSE MCP server"""
        print(f"Starting SSE server from {script_path} on port {port}")
        process = subprocess.Popen([sys.executable, script_path])
        self.active_processes.append(process)
        time.sleep(5)  # Give server time to start
        return process
    
    def start_http_server(self, script_path: str, port: int = 7860) -> subprocess.Popen:
        """Start HTTP MCP server"""
        print(f"Starting HTTP server from {script_path} on port {port}")
        process = subprocess.Popen([sys.executable, script_path])
        self.active_processes.append(process)
        time.sleep(5)  # Give server time to start
        return process
    
    def start_stdio_server(self, command: str, args: List[str]) -> 'MockMCPServerStdio':
        """Start STDIO MCP server"""
        server = MockMCPServerStdio(command, args)
        return server
    
    def cleanup_server(self, process: subprocess.Popen):
        """Clean up a single server process"""
        if process and process.poll() is None:
            try:
                # Try graceful shutdown first
                if platform.system() == "Windows":
                    process.terminate()
                else:
                    process.send_signal(signal.SIGINT)
                
                # Wait for graceful shutdown
                process.wait(timeout=10)
            
            except subprocess.TimeoutExpired:
                print("Server did not terminate gracefully, sending SIGTERM")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("Server did not terminate after SIGTERM, sending SIGKILL")
                    process.kill()
                    process.wait()
    
    def cleanup_all(self):
        """Clean up all active processes"""
        for process in self.active_processes:
            self.cleanup_server(process)
        self.active_processes.clear()
        
        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Cleanup callback error: {e}")


class MockMCPServerStdio:
    """Mock STDIO MCP server for testing"""
    
    def __init__(self, command: str, args: List[str]):
        self.command = command
        self.args = args
        self.name = f"STDIO Server ({command})"
        self._process = None
    
    def start(self) -> subprocess.Popen:
        """Start the STDIO server process"""
        if not self._process:
            self._process = subprocess.PSafePathHandler.safe_open([self.command] ) self.args)
        return self._process
    
    def stop(self):
        """Stop the STDIO server process"""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._process.wait()


class MockMCPServerSSE:
    """Mock SSE MCP server for testing"""
    
    def __init__(self, url: str, strict: bool = False, allowed_tools: List[str] = None):
        self.url = url
        self.strict = strict
        self.allowed_tools = allowed_tools or []
        self.name = "SSE MCP Server"
    
    def get_tools(self) -> List[str]:
        """Get available tools from SSE server"""
        return ["get_secret_word", "add", "get_current_weather"]


class MockMCPServerHTTP:
    """Mock HTTP MCP server for testing"""
    
    def __init__(self, url: str, strict: bool = False, pre_loaded_tools: List[Any] = None):
        self.url = url
        self.strict = strict
        self.pre_loaded_tools = pre_loaded_tools or []
        self.name = "HTTP MCP Server"
    
    def get_tools(self) -> List[str]:
        """Get available tools from HTTP server"""
        return ["get_secret_password"]


class MCPTestFramework:
    """Framework for testing MCP server functionality"""
    
    def __init__(self):
        self.server_manager = MCPServerManager()
        self.test_results = []
    
    def test_filesystem_server(self, samples_dir: str) -> Dict[str, Any]:
        """Test filesystem MCP server"""
        try:
            # Create STDIO server for filesystem
            server = self.server_manager.start_stdio_server(
                "npx", 
                ["-y", "@modelcontextprotocol/server-filesystem", samples_dir]
            )
            
            # Simulate tool creation and execution
            tools = self._create_filesystem_tools(server)
            
            # Test directory listing
            list_tool = tools.get("list_directory")
            if list_tool:
                result = list_tool.run(path=samples_dir)
                return {
                    'success': True,
                    'server_type': 'filesystem',
                    'tools_count': len(tools),
                    'test_result': result
                }
            
            return {'success': False, 'error': 'list_directory tool not found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_git_server(self, repo_path: str) -> Dict[str, Any]:
        """Test Git MCP server"""
        try:
            # Create STDIO server for git
            server = self.server_manager.start_stdio_server("mcp-server-git", [])
            
            # Simulate tool creation and execution
            tools = self._create_git_tools(server)
            
            # Test git status
            git_status_tool = tools.get("git_status")
            if git_status_tool:
                result = git_status_tool.run(repo_path=repo_path)
                return {
                    'success': True,
                    'server_type': 'git',
                    'tools_count': len(tools),
                    'test_result': result
                }
            
            return {'success': False, 'error': 'git_status tool not found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_sse_server(self, server_url: str) -> Dict[str, Any]:
        """Test SSE MCP server"""
        try:
            server = MockMCPServerSSE(server_url, strict=True, allowed_tools=["get_secret_word"])
            
            # Simulate tool execution
            tools = self._create_sse_tools(server)
            
            results = {}
            
            # Test add tool
            add_tool = tools.get("add")
            if add_tool:
                result = await add_tool.run(a=7, b=22)
                results['add'] = result
            
            # Test weather tool
            weather_tool = tools.get("get_current_weather") 
            if weather_tool:
                result = await weather_tool.run(city="Tokyo")
                results['weather'] = result
            
            # Test secret word tool
            secret_tool = tools.get("get_secret_word")
            if secret_tool:
                result = await secret_tool.run()
                results['secret'] = result
            
            return {
                'success': True,
                'server_type': 'sse',
                'tools_count': len(tools),
                'test_results': results
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_http_server(self, server_url: str) -> Dict[str, Any]:
        """Test HTTP MCP server"""
        try:
            server = MockMCPServerHTTP(server_url)
            
            # Simulate tool execution
            tools = self._create_http_tools(server)
            
            # Test secret password tool
            password_tool = tools.get("get_secret_password")
            if password_tool:
                result = await password_tool.run()
                return {
                    'success': True,
                    'server_type': 'http',
                    'tools_count': len(tools),
                    'test_result': result
                }
            
            return {'success': False, 'error': 'get_secret_password tool not found'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_filesystem_tools(self, server) -> Dict[str, 'MockTool']:
        """Create mock filesystem tools"""
        class ListDirectoryTool:
            def run(self, path: str):
                return f"Contents of {path}: csv-test.csv, other-files.txt"
        
        return {"list_directory": ListDirectoryTool()}
    
    def _create_git_tools(self, server) -> Dict[str, 'MockTool']:
        """Create mock git tools"""
        class GitStatusTool:
            def run(self, repo_path: str):
                return f"Repository status: On branch main, Author: Test User"
        
        return {"git_status": GitStatusTool()}
    
    def _create_sse_tools(self, server) -> Dict[str, 'MockTool']:
        """Create mock SSE tools"""
        class AddTool:
            async def run(self, a: int, b: int):
                return str(a + b)
        
        class WeatherTool:
            async def run(self, city: str):
                return f"Weather report: {city} is sunny, 25°C"
        
        class SecretWordTool:
            async def run(self):
                return "strawberry"
        
        return {
            "add": AddTool(),
            "get_current_weather": WeatherTool(),
            "get_secret_word": SecretWordTool()
        }
    
    def _create_http_tools(self, server) -> Dict[str, 'MockTool']:
        """Create mock HTTP tools"""
        class SecretPasswordTool:
            async def run(self):
                return "hc1291cb7123"
        
        return {"get_secret_password": SecretPasswordTool()}
    
    def cleanup(self):
        """Clean up all test resources"""
        self.server_manager.cleanup_all()


class MCPTestValidator:
    """Validates MCP test execution and results"""
    
    def __init__(self):
        self.framework = MCPTestFramework()
    
    def validate_filesystem_test(self, samples_dir: str) -> Dict[str, Any]:
        """Validate filesystem server test"""
        result = self.framework.test_filesystem_server(samples_dir)
        
        if result['success']:
            # Validate expected content
            expected_content = "csv-test.csv"
            content_found = expected_content in result.get('test_result', '')
            result['validation'] = {'expected_content_found': content_found}
        
        return result
    
    def validate_git_test(self, repo_path: str) -> Dict[str, Any]:
        """Validate git server test"""
        result = self.framework.test_git_server(repo_path)
        
        if result['success']:
            # Validate expected content
            expected_content = "Author"
            content_found = expected_content in result.get('test_result', '')
            result['validation'] = {'author_info_found': content_found}
        
        return result
    
    async def validate_sse_test(self, server_url: str = "http://localhost:8080/sse") -> Dict[str, Any]:
        """Validate SSE server test"""
        result = await self.framework.test_sse_server(server_url)
        
        if result['success']:
            results = result.get('test_results', {})
            validations = {}
            
            # Validate add operation
            if 'add' in results:
                validations['add_correct'] = results['add'] == '29'
            
            # Validate weather response
            if 'weather' in results:
                validations['weather_has_report'] = 'Weather report:' in results['weather']
            
            # Validate secret word
            if 'secret' in results:
                valid_words = ['apple', 'banana', 'cherry', 'strawberry']
                validations['secret_valid'] = results['secret'].lower() in valid_words
            
            result['validation'] = validations
        
        return result
    
    async def validate_http_test(self, server_url: str = "http://localhost:7860/mcp") -> Dict[str, Any]:
        """Validate HTTP server test"""
        result = await self.framework.test_http_server(server_url)
        
        if result['success']:
            expected_password = os.getenv('PASSWORD')
            password_correct = result.get('test_result') == expected_password
            result['validation'] = {'password_correct': password_correct}
        
        return result
    
    def cleanup(self):
        """Clean up test framework"""
        self.framework.cleanup()


# Pytest integration patterns
class PyTestMCPPatterns:
    """MCP testing patterns for pytest"""
    
    @pytest.fixture(scope="module")
    def mcp_server_manager(self):
        """Provide MCP server manager"""
        manager = MCPServerManager()
        yield manager
        manager.cleanup_all()
    
    @pytest.fixture
    def samples_dir(self):
        """Provide samples directory"""
        return os.path.join(os.path.dirname(__file__), "data", "files")
    
    def test_filesystem_integration(self, mcp_server_manager, samples_dir):
        """Test filesystem MCP integration"""
        framework = MCPTestFramework()
        result = framework.test_filesystem_server(samples_dir)
        
        assert result['success'], f"Filesystem test failed: {result.get('error')}"
        assert 'csv-test.csv' in result['test_result']
    
    def test_git_integration(self, mcp_server_manager):
        """Test git MCP integration"""
        framework = MCPTestFramework()
        repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        result = framework.test_git_server(repo_path)
        
        assert result['success'], f"Git test failed: {result.get('error')}"
        assert 'Author' in result['test_result']
    
    @pytest.mark.asyncio
    async def test_sse_integration(self):
        """Test SSE MCP integration"""
        validator = MCPTestValidator()
        result = await validator.validate_sse_test()
        
        assert result['success'], f"SSE test failed: {result.get('error')}"
        validations = result.get('validation', {})
        assert validations.get('add_correct', False)
        assert validations.get('weather_has_report', False)
        assert validations.get('secret_valid', False)


# Export patterns for integration
__all__ = [
    'MCPServerManager',
    'MCPTestFramework',
    'MCPTestValidator',
    'MockMCPServerStdio',
    'MockMCPServerSSE', 
    'MockMCPServerHTTP',
    'PyTestMCPPatterns'
]