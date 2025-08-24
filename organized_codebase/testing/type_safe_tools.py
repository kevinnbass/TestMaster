"""
TestMaster Type-Safe Tool Integration
====================================

Type-safe tool framework inspired by Agency-Swarm patterns for robust agent interactions.
Provides compile-time type checking and runtime validation for all agent tools.

Author: TestMaster Team
"""

import asyncio
import inspect
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic, Union, get_type_hints
from pydantic import BaseModel, ValidationError, Field, validator
from functools import wraps
import threading
import uuid

# Type variables for generic tool definitions
T = TypeVar('T')
R = TypeVar('R')

class ToolCategory(Enum):
    """Tool category enumeration for organization"""
    TEST_EXECUTION = "test_execution"
    CODE_ANALYSIS = "code_analysis"
    COVERAGE_ANALYSIS = "coverage_analysis"
    QUALITY_ASSESSMENT = "quality_assessment"
    PERFORMANCE_MONITORING = "performance_monitoring"
    SECURITY_SCANNING = "security_scanning"
    REPORTING = "reporting"
    ORCHESTRATION = "orchestration"
    UTILITY = "utility"

class ToolStatus(Enum):
    """Tool execution status"""
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"       # Full type and constraint validation
    MODERATE = "moderate"   # Type validation with warnings
    LENIENT = "lenient"     # Basic validation only

@dataclass
class ToolMetadata:
    """Metadata for tool registration and discovery"""
    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    author: str = "TestMaster"
    requires_auth: bool = False
    timeout_seconds: float = 300.0
    max_retries: int = 3
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

class ToolInput(BaseModel):
    """Base class for type-safe tool inputs"""
    tool_id: str = Field(..., description="Unique tool identifier")
    execution_id: str = Field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:12]}")
    timeout: Optional[float] = Field(default=None, description="Execution timeout in seconds")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        extra = "allow"
        validate_assignment = True

class ToolOutput(BaseModel):
    """Base class for type-safe tool outputs"""
    tool_id: str
    execution_id: str
    status: ToolStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"
        use_enum_values = True

class TypeSafeTool(ABC, Generic[T, R]):
    """
    Abstract base class for type-safe tools with comprehensive validation.
    Inspired by Agency-Swarm patterns for robust agent interactions.
    """
    
    def __init__(self, metadata: ToolMetadata):
        self.metadata = metadata
        self.logger = logging.getLogger(f'TypeSafeTool.{metadata.name}')
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "total_execution_time": 0.0
        }
        self.lock = threading.Lock()
        
    @abstractmethod
    async def execute(self, input_data: T) -> R:
        """Execute the tool with type-safe input and output"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> T:
        """Validate and convert input to proper type"""
        pass
    
    @abstractmethod
    def validate_output(self, output_data: Any) -> R:
        """Validate and convert output to proper type"""
        pass
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get JSON schema for input validation"""
        input_type = self._get_input_type()
        if hasattr(input_type, 'schema'):
            return input_type.schema()
        return {}
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get JSON schema for output validation"""
        output_type = self._get_output_type()
        if hasattr(output_type, 'schema'):
            return output_type.schema()
        return {}
    
    def _get_input_type(self) -> type:
        """Extract input type from generic type parameters"""
        return get_type_hints(self.execute).get('input_data', Any)
    
    def _get_output_type(self) -> type:
        """Extract output type from generic type parameters"""
        return get_type_hints(self.execute).get('return', Any)
    
    async def safe_execute(self, input_data: Any, validation_level: ValidationLevel = ValidationLevel.STRICT) -> ToolOutput:
        """Execute tool with comprehensive error handling and validation"""
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        try:
            # Input validation
            validated_input = self._validate_input_with_level(input_data, validation_level)
            
            # Execute tool
            self.logger.info(f"Executing tool {self.metadata.name} (execution_id: {execution_id})")
            result = await self.execute(validated_input)
            
            # Output validation
            validated_output = self._validate_output_with_level(result, validation_level)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self._update_performance_metrics(execution_time, True)
            
            output = ToolOutput(
                tool_id=self.metadata.name,
                execution_id=execution_id,
                status=ToolStatus.COMPLETED,
                result=validated_output,
                execution_time=execution_time
            )
            
            self._record_execution(execution_id, input_data, output, None)
            return output
            
        except ValidationError as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(execution_time, False)
            
            error_msg = f"Validation error: {str(e)}"
            self.logger.error(error_msg)
            
            output = ToolOutput(
                tool_id=self.metadata.name,
                execution_id=execution_id,
                status=ToolStatus.FAILED,
                error=error_msg,
                execution_time=execution_time
            )
            
            self._record_execution(execution_id, input_data, output, e)
            return output
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(execution_time, False)
            
            error_msg = f"Execution error: {str(e)}"
            self.logger.error(error_msg)
            
            output = ToolOutput(
                tool_id=self.metadata.name,
                execution_id=execution_id,
                status=ToolStatus.FAILED,
                error=error_msg,
                execution_time=execution_time
            )
            
            self._record_execution(execution_id, input_data, output, e)
            return output
    
    def _validate_input_with_level(self, input_data: Any, level: ValidationLevel) -> T:
        """Validate input based on validation level"""
        if level == ValidationLevel.STRICT:
            return self.validate_input(input_data)
        elif level == ValidationLevel.MODERATE:
            try:
                return self.validate_input(input_data)
            except ValidationError as e:
                self.logger.warning(f"Input validation warning: {e}")
                return input_data
        else:  # LENIENT
            return input_data
    
    def _validate_output_with_level(self, output_data: Any, level: ValidationLevel) -> R:
        """Validate output based on validation level"""
        if level == ValidationLevel.STRICT:
            return self.validate_output(output_data)
        elif level == ValidationLevel.MODERATE:
            try:
                return self.validate_output(output_data)
            except ValidationError as e:
                self.logger.warning(f"Output validation warning: {e}")
                return output_data
        else:  # LENIENT
            return output_data
    
    def _update_performance_metrics(self, execution_time: float, success: bool):
        """Update tool performance metrics"""
        with self.lock:
            self.performance_metrics["total_executions"] += 1
            self.performance_metrics["total_execution_time"] += execution_time
            
            if success:
                self.performance_metrics["successful_executions"] += 1
            else:
                self.performance_metrics["failed_executions"] += 1
            
            # Update average execution time
            total_execs = self.performance_metrics["total_executions"]
            total_time = self.performance_metrics["total_execution_time"]
            self.performance_metrics["average_execution_time"] = total_time / total_execs
    
    def _record_execution(self, execution_id: str, input_data: Any, output: ToolOutput, error: Optional[Exception]):
        """Record execution in history"""
        with self.lock:
            self.execution_history.append({
                "execution_id": execution_id,
                "timestamp": datetime.now().isoformat(),
                "input_summary": self._summarize_data(input_data),
                "output_status": output.status.value,
                "execution_time": output.execution_time,
                "error": str(error) if error else None
            })
            
            # Keep only last 100 executions
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]
    
    def _summarize_data(self, data: Any) -> str:
        """Create a summary of data for logging"""
        if isinstance(data, str):
            return data[:100] + "..." if len(data) > 100 else data
        elif isinstance(data, dict):
            return f"Dict with {len(data)} keys"
        elif isinstance(data, list):
            return f"List with {len(data)} items"
        else:
            return str(type(data).__name__)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this tool"""
        with self.lock:
            metrics = self.performance_metrics.copy()
            metrics["success_rate"] = (
                metrics["successful_executions"] / metrics["total_executions"] * 100
                if metrics["total_executions"] > 0 else 0
            )
            metrics["tool_metadata"] = {
                "name": self.metadata.name,
                "category": self.metadata.category.value,
                "version": self.metadata.version
            }
            return metrics

class ToolRegistry:
    """
    Central registry for type-safe tools with discovery and validation capabilities.
    """
    
    def __init__(self):
        self.tools: Dict[str, TypeSafeTool] = {}
        self.tool_schemas: Dict[str, Dict[str, Any]] = {}
        self.categories: Dict[ToolCategory, List[str]] = {}
        self.logger = logging.getLogger('ToolRegistry')
        self.lock = threading.Lock()
        
        # Initialize category mapping
        for category in ToolCategory:
            self.categories[category] = []
    
    def register_tool(self, tool: TypeSafeTool) -> bool:
        """Register a new tool in the registry"""
        try:
            with self.lock:
                tool_name = tool.metadata.name
                
                if tool_name in self.tools:
                    self.logger.warning(f"Tool {tool_name} already registered, updating...")
                
                # Register tool
                self.tools[tool_name] = tool
                
                # Store schemas
                self.tool_schemas[tool_name] = {
                    "input_schema": tool.get_input_schema(),
                    "output_schema": tool.get_output_schema(),
                    "metadata": tool.metadata.__dict__
                }
                
                # Update category mapping
                if tool_name not in self.categories[tool.metadata.category]:
                    self.categories[tool.metadata.category].append(tool_name)
                
                self.logger.info(f"Registered tool: {tool_name} ({tool.metadata.category.value})")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to register tool {tool.metadata.name}: {e}")
            return False
    
    def get_tool(self, tool_name: str) -> Optional[TypeSafeTool]:
        """Get a tool by name"""
        return self.tools.get(tool_name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[TypeSafeTool]:
        """Get all tools in a specific category"""
        tool_names = self.categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all available tools"""
        return {
            name: {
                "metadata": tool.metadata.__dict__,
                "schemas": self.tool_schemas.get(name, {}),
                "performance": tool.get_performance_summary()
            }
            for name, tool in self.tools.items()
        }
    
    def validate_tool_compatibility(self, tool_name: str, input_data: Any) -> bool:
        """Validate if input data is compatible with tool"""
        tool = self.get_tool(tool_name)
        if not tool:
            return False
        
        try:
            tool.validate_input(input_data)
            return True
        except ValidationError:
            return False
    
    async def execute_tool_safe(
        self, 
        tool_name: str, 
        input_data: Any,
        validation_level: ValidationLevel = ValidationLevel.STRICT
    ) -> ToolOutput:
        """Execute a tool safely with comprehensive error handling"""
        tool = self.get_tool(tool_name)
        
        if not tool:
            return ToolOutput(
                tool_id=tool_name,
                execution_id=f"exec_{uuid.uuid4().hex[:12]}",
                status=ToolStatus.FAILED,
                error=f"Tool '{tool_name}' not found in registry"
            )
        
        return await tool.safe_execute(input_data, validation_level)

# Example implementations for TestMaster

class TestExecutionInput(ToolInput):
    """Input model for test execution tools"""
    test_path: str = Field(..., description="Path to test file or directory")
    test_pattern: Optional[str] = Field(default=None, description="Test pattern to match")
    coverage_enabled: bool = Field(default=True, description="Enable coverage collection")
    parallel_workers: int = Field(default=1, description="Number of parallel workers")
    
    @validator('parallel_workers')
    def validate_workers(cls, v):
        if v < 1 or v > 16:
            raise ValueError('parallel_workers must be between 1 and 16')
        return v

class TestExecutionOutput(ToolOutput):
    """Output model for test execution tools"""
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    coverage_percentage: Optional[float] = None
    test_results: List[Dict[str, Any]] = Field(default_factory=list)

class CoverageAnalysisInput(ToolInput):
    """Input model for coverage analysis tools"""
    source_path: str = Field(..., description="Path to source code")
    coverage_data_path: Optional[str] = Field(default=None, description="Path to coverage data file")
    include_patterns: List[str] = Field(default_factory=list, description="Patterns to include")
    exclude_patterns: List[str] = Field(default_factory=list, description="Patterns to exclude")

class CoverageAnalysisOutput(ToolOutput):
    """Output model for coverage analysis tools"""
    overall_coverage: float
    line_coverage: float
    branch_coverage: Optional[float] = None
    function_coverage: Optional[float] = None
    file_coverage: Dict[str, float] = Field(default_factory=dict)
    missing_lines: Dict[str, List[int]] = Field(default_factory=dict)

# Global tool registry instance
global_tool_registry = ToolRegistry()

# Decorator for automatic tool registration
def register_tool(metadata: ToolMetadata):
    """Decorator to automatically register tools"""
    def decorator(tool_class):
        def wrapper(*args, **kwargs):
            tool_instance = tool_class(metadata, *args, **kwargs)
            global_tool_registry.register_tool(tool_instance)
            return tool_instance
        return wrapper
    return decorator

# Export key components
__all__ = [
    'TypeSafeTool',
    'ToolRegistry', 
    'ToolInput',
    'ToolOutput',
    'ToolMetadata',
    'ToolCategory',
    'ToolStatus',
    'ValidationLevel',
    'TestExecutionInput',
    'TestExecutionOutput',
    'CoverageAnalysisInput',
    'CoverageAnalysisOutput',
    'global_tool_registry',
    'register_tool'
]