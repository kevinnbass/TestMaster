"""Tool Factory Testing Framework - Agency-Swarm Pattern
Extracted patterns for testing tool creation and integration
Supports OpenAPI, LangChain, MCP, and file-based tool generation
"""
import os
import json
import asyncio
import subprocess
import sys
import shutil
from typing import Any, Dict, List, Optional, Union, Callable
from unittest.mock import Mock, MagicMock, patch
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
import pytest


class MockToolFactory:
    """Mock tool factory for testing tool creation patterns"""
    
    @staticmethod
    def from_langchain_tool(langchain_tool_class) -> type:
        """Create tool from LangChain tool class"""
        class GeneratedTool:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.tool_class = langchain_tool_class
            
            def run(self):
                return f"Executed {langchain_tool_class.__name__} with {self.kwargs}"
        
        GeneratedTool.__name__ = langchain_tool_class.__name__
        return GeneratedTool
    
    @staticmethod
    def from_openai_schema(schema: Dict[str, Any], callback_func: Callable) -> type:
        """Create tool from OpenAI schema"""
        class GeneratedTool:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.schema = schema
                self.callback = callback_func
                self.openai_schema = schema.copy()
            
            def run(self):
                return self.callback(self.kwargs)
        
        GeneratedTool.__name__ = schema.get('name', 'GeneratedTool')
        return GeneratedTool
    
    @staticmethod
    def from_openapi_schema(schema_json: str, headers: Dict[str, str]) -> List[type]:
        """Create tools from OpenAPI schema"""
        schema = json.loads(schema_json)
        tools = []
        
        # Mock tool creation from OpenAPI paths
        paths = schema.get('paths', {})
        for path, methods in paths.items():
            for method, spec in methods.items():
                if isinstance(spec, dict):
                    operation_id = spec.get('operationId', f"{method}_{path.replace('/', '_')}")
                    
                    class GeneratedAPITool:
                        def __init__(self, **kwargs):
                            self.kwargs = kwargs
                            self.path = path
                            self.method = method
                            self.headers = headers
                            self.openai_schema = {"strict": schema.get('strict', False)}
                        
                        async def run(self):
                            # Simulate API call
                            if 'mock_response' in self.kwargs:
                                return self.kwargs['mock_response']
                            return {"result": f"API call to {self.path}"}
                    
                    GeneratedAPITool.__name__ = operation_id
                    tools.append(GeneratedAPITool)
        
        return tools
    
    @staticmethod
    def from_file(file_path: str) -> type:
        """Create tool from Python file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tool file not found: {file_path}")
        
        # Mock file-based tool creation
        filename = os.path.basename(file_path)
        tool_name = os.path.splitext(filename)[0]
        
        class FileBasedTool:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.file_path = file_path
            
            def run(self):
                if 'content' in self.kwargs:
                    return "Tool output"
                return f"Executed tool from {self.file_path}"
        
        FileBasedTool.__name__ = tool_name
        return FileBasedTool
    
    @staticmethod
    def from_mcp(mcp_server) -> List[type]:
        """Create tools from MCP server"""
        tools = []
        available_tools = getattr(mcp_server, 'get_available_tools', lambda: [])()
        
        for tool_name in available_tools:
            class MCPTool:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.server = mcp_server
                    self.tool_name = tool_name
                
                async def run(self):
                    # Simulate MCP tool execution
                    if hasattr(self.server, 'execute_tool'):
                        return await self.server.execute_tool(self.tool_name, self.kwargs)
                    return f"Executed {self.tool_name} via MCP"
            
            MCPTool.__name__ = tool_name
            tools.append(MCPTool)
        
        return tools


class MockLangChainTool:
    """Mock LangChain tool for testing"""
    
    def __init__(self, name: str = "MockTool"):
        self.__name__ = name
        self.name = name


class MockMCPServer:
    """Mock MCP server for testing"""
    
    def __init__(self, server_type: str = "stdio"):
        self.server_type = server_type
        self._process = None
    
    def get_available_tools(self) -> List[str]:
        """Get available tools from server"""
        if self.server_type == "filesystem":
            return ["read_file", "list_directory", "write_file"]
        elif self.server_type == "git":
            return ["git_status", "git_commit", "git_log"]
        elif self.server_type == "sse":
            return ["add", "get_current_weather", "get_secret_word"]
        elif self.server_type == "http":
            return ["get_secret_password"]
        return []
    
    async def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute tool via MCP server"""
        if tool_name == "read_file" and params.get("path"):
            return f"File content from {params['path']}"
        elif tool_name == "list_directory" and params.get("path"):
            return f"Directory listing for {params['path']}"
        elif tool_name == "git_status":
            return "Repository status: clean working directory"
        elif tool_name == "add" and "a" in params and "b" in params:
            return str(params["a"] + params["b"])
        elif tool_name == "get_current_weather" and params.get("city"):
            return f"Weather in {params['city']}: sunny, 25°C"
        elif tool_name == "get_secret_word":
            return "strawberry"
        elif tool_name == "get_secret_password":
            return "hc1291cb7123"
        return f"Executed {tool_name} with {params}"


class ToolFactoryTestFramework:
    """Framework for testing tool factory functionality"""
    
    def __init__(self):
        self.test_results = []
        self.mock_factory = MockToolFactory()
    
    def test_langchain_tool_creation(self, tool_class) -> Dict[str, Any]:
        """Test LangChain tool creation"""
        try:
            # Create tool from LangChain class
            generated_tool_class = self.mock_factory.from_langchain_tool(tool_class)
            
            # Instantiate and test
            tool_instance = generated_tool_class(
                destination_path="test_dest",
                source_path="test_source"
            )
            
            result = tool_instance.run()
            
            return {
                'success': True,
                'tool_name': generated_tool_class.__name__,
                'result': result,
                'creation_successful': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_openai_schema_creation(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Test OpenAI schema-based tool creation"""
        try:
            # Create tool from schema
            generated_tool_class = self.mock_factory.from_openai_schema(
                schema, lambda x: f"Processed: {x}"
            )
            
            # Test required parameters
            required_params = schema.get('parameters', {}).get('required', [])
            test_params = {param: f"test_{param}" for param in required_params}
            
            tool_instance = generated_tool_class(**test_params)
            result = tool_instance.run()
            
            # Check schema preservation
            has_strict = hasattr(tool_instance, 'openai_schema')
            strict_value = getattr(tool_instance, 'openai_schema', {}).get('strict', False)
            
            return {
                'success': True,
                'tool_name': generated_tool_class.__name__,
                'result': result,
                'schema_preserved': has_strict,
                'strict_mode': strict_value
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def test_openapi_schema_creation(self, schema_json: str, 
                                         headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Test OpenAPI schema-based tool creation"""
        try:
            headers = headers or {}
            
            # Create tools from OpenAPI schema
            generated_tools = self.mock_factory.from_openapi_schema(schema_json, headers)
            
            if not generated_tools:
                return {'success': False, 'error': 'No tools generated'}
            
            # Test first tool
            first_tool_class = generated_tools[0]
            tool_instance = first_tool_class(mock_response={"test": "data"})
            result = await tool_instance.run()
            
            return {
                'success': True,
                'tools_count': len(generated_tools),
                'first_tool_name': first_tool_class.__name__,
                'result': result,
                'has_strict_mode': hasattr(tool_instance, 'openai_schema')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_file_based_tool_creation(self, file_path: str) -> Dict[str, Any]:
        """Test file-based tool creation"""
        try:
            # Create mock file if it doesn't exist
            if not os.path.exists(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as f:
                    f.write("# Mock tool file")
            
            # Create tool from file
            generated_tool_class = self.mock_factory.from_file(file_path)
            
            # Test tool
            tool_instance = generated_tool_class(content="test content")
            result = tool_instance.run()
            
            return {
                'success': True,
                'tool_name': generated_tool_class.__name__,
                'result': result,
                'file_loaded': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def test_mcp_tool_creation(self, server_type: str = "filesystem") -> Dict[str, Any]:
        """Test MCP-based tool creation"""
        try:
            # Create mock MCP server
            mcp_server = MockMCPServer(server_type)
            
            # Create tools from MCP server
            generated_tools = self.mock_factory.from_mcp(mcp_server)
            
            if not generated_tools:
                return {'success': False, 'error': 'No MCP tools generated'}
            
            # Test specific tool based on server type
            test_results = {}
            
            for tool_class in generated_tools[:2]:  # Test first 2 tools
                try:
                    if tool_class.__name__ == "read_file":
                        tool_instance = tool_class(path="/test/file.txt")
                        result = await tool_instance.run()
                        test_results[tool_class.__name__] = result
                    
                    elif tool_class.__name__ == "list_directory":
                        tool_instance = tool_class(path="/test")
                        result = await tool_instance.run()
                        test_results[tool_class.__name__] = result
                    
                    elif tool_class.__name__ == "git_status":
                        tool_instance = tool_class(repo_path="/test/repo")
                        result = await tool_instance.run()
                        test_results[tool_class.__name__] = result
                    
                    elif tool_class.__name__ == "add":
                        tool_instance = tool_class(a=7, b=22)
                        result = await tool_instance.run()
                        test_results[tool_class.__name__] = result
                    
                    elif tool_class.__name__ == "get_secret_password":
                        tool_instance = tool_class()
                        result = await tool_instance.run()
                        test_results[tool_class.__name__] = result
                        
                except Exception as tool_error:
                    test_results[tool_class.__name__] = f"Error: {tool_error}"
            
            return {
                'success': True,
                'server_type': server_type,
                'tools_count': len(generated_tools),
                'tool_names': [t.__name__ for t in generated_tools],
                'test_results': test_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


class ComplexSchemaTestScenarios:
    """Test scenarios for complex schema handling"""
    
    @staticmethod
    def create_complex_pydantic_schema() -> Dict[str, Any]:
        """Create complex Pydantic-based schema for testing"""
        
        class FriendDetail(BaseModel):
            model_config = ConfigDict(title="FriendDetail")
            
            id: int = Field(..., description="Unique identifier for each friend.")
            name: str = Field(..., description="Name of the friend.")
            age: Optional[int] = Field(25, description="Age of the friend.")
            email: Optional[str] = Field(None, description="Email address of the friend.")
            is_active: Optional[bool] = Field(None, description="Indicates if the friend is currently active.")
        
        class RelationshipType(str, Enum):
            FAMILY = "family"
            FRIEND = "friend"
            COLLEAGUE = "colleague"
        
        class UserDetail(BaseModel):
            model_config = ConfigDict(title="UserDetail")
            
            id: int = Field(..., description="Unique identifier for each user.")
            age: int
            name: str
            friends: List[FriendDetail] = Field(
                ..., description="List of friends, each represented by a FriendDetail model."
            )
        
        # Convert to OpenAI schema format
        return {
            "name": "UserRelationships",
            "description": "Complex user relationships schema",
            "parameters": {
                "type": "object",
                "properties": {
                    "users": {
                        "type": "array",
                        "items": {
                            "type": "object", 
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string"},
                                "age": {"type": "integer"},
                                "friends": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "integer"},
                                            "name": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "relationship_type": {
                        "type": "string",
                        "enum": ["family", "friend", "colleague"]
                    }
                },
                "required": ["users", "relationship_type"]
            }
        }
    
    @staticmethod
    def create_openapi_schema() -> str:
        """Create OpenAPI schema for testing"""
        return json.dumps({
            "openapi": "3.0.0",
            "paths": {
                "/weather": {
                    "get": {
                        "operationId": "getWeather",
                        "summary": "Get weather information",
                        "parameters": [
                            {
                                "name": "city",
                                "in": "query",
                                "required": True,
                                "schema": {"type": "string"}
                            }
                        ]
                    }
                },
                "/report": {
                    "post": {
                        "operationId": "runReport",
                        "summary": "Run analytics report",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "reportType": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        })


class ToolFactoryTestValidator:
    """Validates tool factory test results"""
    
    def __init__(self):
        self.framework = ToolFactoryTestFramework()
    
    def validate_langchain_integration(self) -> Dict[str, Any]:
        """Validate LangChain tool integration"""
        mock_tool = MockLangChainTool("TestTool")
        result = self.framework.test_langchain_tool_creation(mock_tool)
        
        validation = {
            'tool_created': result.get('success', False),
            'correct_name': result.get('tool_name') == mock_tool.__name__,
            'execution_successful': 'test_dest' in result.get('result', '')
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    def validate_openai_schema_integration(self) -> Dict[str, Any]:
        """Validate OpenAI schema integration"""
        schema = ComplexSchemaTestScenarios.create_complex_pydantic_schema()
        result = self.framework.test_openai_schema_creation(schema)
        
        validation = {
            'tool_created': result.get('success', False),
            'schema_preserved': result.get('schema_preserved', False),
            'execution_successful': 'Processed:' in result.get('result', '')
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    async def validate_openapi_integration(self) -> Dict[str, Any]:
        """Validate OpenAPI schema integration"""
        schema_json = ComplexSchemaTestScenarios.create_openapi_schema()
        result = await self.framework.test_openapi_schema_creation(schema_json)
        
        validation = {
            'tools_created': result.get('success', False),
            'multiple_tools': result.get('tools_count', 0) > 1,
            'execution_successful': result.get('result') is not None
        }
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }
    
    async def validate_mcp_integration(self) -> Dict[str, Any]:
        """Validate MCP integration for different server types"""
        server_types = ["filesystem", "git", "sse", "http"]
        results = {}
        
        for server_type in server_types:
            result = await self.framework.test_mcp_tool_creation(server_type)
            
            validation = {
                'tools_created': result.get('success', False),
                'has_tools': result.get('tools_count', 0) > 0,
                'execution_successful': len(result.get('test_results', {})) > 0
            }
            
            results[server_type] = {
                'test_result': result,
                'validation': validation,
                'overall_success': all(validation.values())
            }
        
        return results
    
    def validate_file_based_integration(self) -> Dict[str, Any]:
        """Validate file-based tool integration"""
        test_file_path = os.path.join(os.getcwd(), "test_tools", "ExampleTool.py")
        
        result = self.framework.test_file_based_tool_creation(test_file_path)
        
        validation = {
            'tool_created': result.get('success', False),
            'file_loaded': result.get('file_loaded', False),
            'execution_successful': result.get('result') is not None
        }
        
        # Cleanup test file
        if os.path.exists(test_file_path):
            try:
                os.remove(test_file_path)
                os.rmdir(os.path.dirname(test_file_path))
            except:
                pass
        
        return {
            'test_result': result,
            'validation': validation,
            'overall_success': all(validation.values())
        }


# Pytest integration patterns
class PyTestToolFactoryPatterns:
    """Tool factory testing patterns for pytest"""
    
    @pytest.fixture
    def tool_factory_framework(self):
        """Provide tool factory test framework"""
        return ToolFactoryTestFramework()
    
    @pytest.fixture
    def mock_langchain_tool(self):
        """Provide mock LangChain tool"""
        return MockLangChainTool("MoveTool")
    
    def test_langchain_tool_creation(self, tool_factory_framework, mock_langchain_tool):
        """Test LangChain tool creation"""
        result = tool_factory_framework.test_langchain_tool_creation(mock_langchain_tool)
        
        assert result['success'] is True
        assert result['tool_name'] == mock_langchain_tool.__name__
        assert 'MoveTool' in result['result']
    
    def test_complex_schema_handling(self, tool_factory_framework):
        """Test complex schema handling"""
        schema = ComplexSchemaTestScenarios.create_complex_pydantic_schema()
        result = tool_factory_framework.test_openai_schema_creation(schema)
        
        assert result['success'] is True
        assert result['tool_name'] == "UserRelationships"
    
    @pytest.mark.asyncio
    async def test_mcp_filesystem_integration(self, tool_factory_framework):
        """Test MCP filesystem integration"""
        result = await tool_factory_framework.test_mcp_tool_creation("filesystem")
        
        assert result['success'] is True
        assert result['tools_count'] > 0
        assert 'read_file' in result['tool_names']
    
    def test_file_tool_import(self, tool_factory_framework):
        """Test file-based tool import"""
        # Create temporary tool file
        test_file = os.path.join(os.getcwd(), "temp_tool.py")
        with open(test_file, 'w') as f:
            f.write("# Test tool")
        
        try:
            result = tool_factory_framework.test_file_based_tool_creation(test_file)
            assert result['success'] is True
            assert result['tool_name'] == "temp_tool"
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)


# Export patterns for integration
__all__ = [
    'MockToolFactory',
    'ToolFactoryTestFramework',
    'ToolFactoryTestValidator',
    'ComplexSchemaTestScenarios',
    'MockMCPServer',
    'PyTestToolFactoryPatterns'
]