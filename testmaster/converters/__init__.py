"""
TestMaster Test Converters

High-performance test conversion engines with parallel processing capabilities.
"""

# Base classes
from .base import (
    BaseConverter,
    ParallelConverter,
    BatchConverter,
    CachedConverter,
    ConversionConfig,
    ConversionResult,
    RateLimiter
)

# Converter implementations
from .intelligent import IntelligentConverter
from .parallel import ParallelTestConverter
from .batch import BatchTestConverter

# Legacy compatibility aliases for existing scripts
AcceleratedConverter = ParallelTestConverter
TurboConverter = ParallelTestConverter
FastConverter = IntelligentConverter
BrokenTestConverter = BatchTestConverter
SelfHealingConverter = IntelligentConverter

# Additional legacy aliases
ParallelConverterFixed = ParallelTestConverter
ParallelConverterWorking = ParallelTestConverter
Week58BatchConverter = BatchTestConverter
Week78Converter = BatchTestConverter
ParallelCoverageConverter = ParallelTestConverter
ParallelCoverageConverterFixed = ParallelTestConverter

__all__ = [
    # Base classes
    "BaseConverter",
    "ParallelConverter", 
    "BatchConverter",
    "CachedConverter",
    "ConversionConfig",
    "ConversionResult",
    "RateLimiter",
    
    # Main implementations
    "IntelligentConverter",
    "ParallelTestConverter",
    "BatchTestConverter",
    
    # Legacy aliases
    "AcceleratedConverter",
    "TurboConverter",
    "FastConverter",
    "BrokenTestConverter",
    "SelfHealingConverter",
    "ParallelConverterFixed",
    "ParallelConverterWorking", 
    "Week58BatchConverter",
    "Week78Converter",
    "ParallelCoverageConverter",
    "ParallelCoverageConverterFixed"
]