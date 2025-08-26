"""
Converter Base Classes and Interfaces

This module provides the base classes and interfaces for the unified
converter framework using the strategy pattern.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class ConversionStrategy(Enum):
    """Available conversion strategies"""
    ACCELERATED = "accelerated"
    BATCH = "batch"
    PARALLEL = "parallel"
    TURBO = "turbo"
    INTELLIGENT = "intelligent"
    COVERAGE = "coverage"
    SIMPLE = "simple"


class ConversionStatus(Enum):
    """Status of conversion operation"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    EXISTS = "exists"


@dataclass
class ConversionConfig:
    """Configuration for conversion operations"""
    strategy: ConversionStrategy = ConversionStrategy.INTELLIGENT
    max_workers: int = 4
    batch_size: int = 10
    timeout: int = 30
    backup_enabled: bool = True
    api_key: Optional[str] = None
    model_name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.1
    max_content_length: int = 4000
    output_dir: Path = field(default_factory=lambda: Path("tests/unit"))
    include_patterns: List[str] = field(default_factory=lambda: ["*.py"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["test_*", "__pycache__"])
    verbose: bool = False


@dataclass
class ConversionResult:
    """Result of a conversion operation"""
    file_path: Path
    status: ConversionStatus
    strategy_used: ConversionStrategy
    output_path: Optional[Path] = None
    error_message: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConversionMetrics:
    """Metrics for conversion operations"""
    total_files: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    success_rate: float = 0.0
    errors: List[str] = field(default_factory=list)


class ConverterStrategy(ABC):
    """Abstract base class for conversion strategies"""
    
    def __init__(self, config: ConversionConfig, orchestrator=None):
        self.config = config
        self.orchestrator = orchestrator  # Optional orchestrator integration
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Orchestration integration state
        self.is_orchestrated = orchestrator is not None
        self.orchestration_metrics = {
            'total_conversions': 0,
            'orchestrated_conversions': 0,
            'independent_conversions': 0
        }
        
    @abstractmethod
    async def convert(self, file_path: Path) -> ConversionResult:
        """Convert a single file"""
        pass
    
    @abstractmethod
    async def batch_convert(self, file_paths: List[Path]) -> List[ConversionResult]:
        """Convert multiple files in batch"""
        pass
    
    @abstractmethod
    def validate_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate if file can be converted"""
        pass
    
    def get_output_path(self, file_path: Path) -> Path:
        """Get output path for converted file"""
        module_name = file_path.stem
        
        # Determine test file name based on strategy
        if self.config.strategy == ConversionStrategy.INTELLIGENT:
            test_name = f"test_{module_name}_intelligent.py"
        elif self.config.strategy == ConversionStrategy.COVERAGE:
            test_name = f"test_{module_name}_coverage.py"
        else:
            test_name = f"test_{module_name}.py"
        
        return self.config.output_dir / test_name
    
    def check_exists(self, file_path: Path) -> bool:
        """Check if converted file already exists"""
        output_path = self.get_output_path(file_path)
        return output_path.exists()
    
    async def read_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content with error handling"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Truncate if too long
            if len(content) > self.config.max_content_length:
                content = content[:self.config.max_content_length] + "\n# ... truncated ..."
            
            return content
        except Exception as e:
            self.logger.error(f"Failed to read {file_path}: {e}")
            return None
    
    async def write_output(self, output_path: Path, content: str) -> bool:
        """Write converted content to output file"""
        try:
            # Create directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Backup if exists and backup enabled
            if output_path.exists() and self.config.backup_enabled:
                backup_path = output_path.with_suffix(".backup")
                output_path.rename(backup_path)
                self.logger.debug(f"Created backup: {backup_path}")
            
            # Write new content
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to write {output_path}: {e}")
            return False
    
    def build_import_path(self, file_path: Path) -> str:
        """Build Python import path for module"""
        try:
            # Try to determine relative path from project root
            if "multi_coder_analysis" in str(file_path):
                rel_path = file_path.relative_to(file_path.parent)
                while rel_path.parent.name != "multi_coder_analysis":
                    rel_path = rel_path.parent / rel_path.name
            else:
                # Default to simple module name
                return file_path.stem
            
            # Build import path
            parts = list(rel_path.parent.parts) + [file_path.stem]
            return ".".join(parts)
        except Exception:
            return file_path.stem
    
    # ========================================================================
    # ORCHESTRATION INTEGRATION METHODS
    # ========================================================================
    
    async def orchestrated_convert(self, file_path: Path) -> ConversionResult:
        """Convert file with orchestration integration if available."""
        if not self.is_orchestrated:
            # Fall back to regular conversion
            result = await self.convert(file_path)
            self.orchestration_metrics['independent_conversions'] += 1
        else:
            # Use orchestrator for enhanced conversion
            conversion_task = {
                'type': 'conversion',
                'strategy': self.config.strategy.value,
                'file_path': str(file_path),
                'config': self.config.__dict__
            }
            
            # Submit to orchestrator as a task
            task_id = self.orchestrator.submit_task(conversion_task)
            
            # Execute through orchestrator
            result = await self.orchestrator.execute_task(conversion_task)
            self.orchestration_metrics['orchestrated_conversions'] += 1
            
            # Convert orchestrator result to ConversionResult if needed
            if not isinstance(result, ConversionResult):
                result = await self.convert(file_path)
        
        self.orchestration_metrics['total_conversions'] += 1
        return result
    
    async def orchestrated_batch_convert(self, file_paths: List[Path]) -> List[ConversionResult]:
        """Convert files in batch with orchestration integration."""
        if not self.is_orchestrated:
            # Fall back to regular batch conversion
            return await self.batch_convert(file_paths)
        
        # Use orchestrator for batch management
        batch_task = {
            'type': 'batch_conversion',
            'strategy': self.config.strategy.value,
            'file_paths': [str(p) for p in file_paths],
            'batch_size': self.config.batch_size,
            'config': self.config.__dict__
        }
        
        # Execute batch through orchestrator
        result = await self.orchestrator.execute_batch([batch_task])
        
        # Convert result to list of ConversionResult if needed
        if isinstance(result, dict) and 'results' in result:
            return result['results']
        else:
            # Fall back to regular batch conversion
            return await self.batch_convert(file_paths)
    
    def set_orchestrator(self, orchestrator):
        """Set or update the orchestrator for this converter."""
        self.orchestrator = orchestrator
        self.is_orchestrated = orchestrator is not None
        
        if self.is_orchestrated:
            self.logger.info(f"Converter integrated with orchestrator: {orchestrator.name}")
        else:
            self.logger.info("Converter operating independently")
    
    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get orchestration integration metrics."""
        return {
            'is_orchestrated': self.is_orchestrated,
            'orchestrator_name': self.orchestrator.name if self.orchestrator else None,
            'metrics': self.orchestration_metrics.copy(),
            'orchestration_rate': (
                self.orchestration_metrics['orchestrated_conversions'] / 
                max(self.orchestration_metrics['total_conversions'], 1)
            ) * 100
        }
    
    def register_with_orchestrator(self) -> bool:
        """Register this converter as an agent with the orchestrator."""
        if not self.is_orchestrated:
            return False
        
        # Register converter as an agent
        agent_id = self.orchestrator.register_agent(self)
        self.logger.info(f"Converter registered with orchestrator as agent: {agent_id}")
        return True