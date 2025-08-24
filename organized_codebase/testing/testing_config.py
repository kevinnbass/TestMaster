"""
Testing Configuration Module
===========================

Test generation and execution configuration settings.
Modularized from testmaster_config.py and enhanced_unified_config.py.

Author: Agent E - Infrastructure Consolidation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from .data_models import ConfigBase


@dataclass
class GenerationConfig(ConfigBase):
    """Test generation configuration."""
    
    # Generation Settings
    max_iterations: int = 5
    self_healing_enabled: bool = True
    verification_passes: int = 3
    min_quality_score: int = 70
    max_quality_score: int = 100
    
    # Parallel Processing
    parallel_workers: int = 4
    batch_size: int = 10
    async_generation: bool = True
    
    # Test Generation Options
    use_real_imports: bool = True
    generate_edge_cases: bool = True
    generate_error_tests: bool = True
    generate_performance_tests: bool = False
    generate_security_tests: bool = False
    
    # Framework Configuration
    test_framework: str = "pytest"
    coverage_target: float = 95.0
    incremental_mode: bool = True
    
    # AI Model Settings
    context_window_size: int = 8000
    temperature: float = 0.1
    max_output_tokens: int = 4000
    
    # Test Patterns
    test_patterns: List[str] = field(default_factory=lambda: [
        "test_*.py",
        "*_test.py"
    ])
    
    def validate(self) -> List[str]:
        """Validate generation configuration."""
        errors = []
        
        if self.max_iterations <= 0:
            errors.append("Max iterations must be positive")
        
        if not 0 <= self.min_quality_score <= 100:
            errors.append("Min quality score must be between 0 and 100")
        
        if not 0 <= self.max_quality_score <= 100:
            errors.append("Max quality score must be between 0 and 100")
        
        if self.min_quality_score > self.max_quality_score:
            errors.append("Min quality score cannot exceed max quality score")
        
        if self.parallel_workers <= 0:
            errors.append("Parallel workers must be positive")
        
        if self.coverage_target < 0 or self.coverage_target > 100:
            errors.append("Coverage target must be between 0 and 100")
        
        return errors


@dataclass
class ExecutionConfig(ConfigBase):
    """Test execution configuration."""
    
    # Execution Settings
    timeout_per_test: int = 30
    max_parallel_tests: int = 10
    retry_failed_tests: bool = True
    max_test_retries: int = 2
    
    # Test Discovery
    test_directories: List[str] = field(default_factory=lambda: ["tests", "test"])
    exclude_directories: List[str] = field(default_factory=lambda: [
        "__pycache__",
        ".pytest_cache",
        "venv",
        ".git"
    ])
    
    # Execution Modes
    fail_fast: bool = False
    verbose_output: bool = True
    capture_output: bool = True
    show_warnings: bool = True
    
    # Performance Settings
    use_multiprocessing: bool = True
    memory_limit_mb: int = 2048
    cpu_limit_percent: int = 80
    
    # Output Configuration
    report_format: str = "html"
    junit_xml: bool = True
    coverage_report: bool = True
    
    def validate(self) -> List[str]:
        """Validate execution configuration."""
        errors = []
        
        if self.timeout_per_test <= 0:
            errors.append("Timeout per test must be positive")
        
        if self.max_parallel_tests <= 0:
            errors.append("Max parallel tests must be positive")
        
        if self.memory_limit_mb <= 0:
            errors.append("Memory limit must be positive")
        
        if not 0 < self.cpu_limit_percent <= 100:
            errors.append("CPU limit must be between 1 and 100")
        
        return errors


@dataclass
class QualityConfig(ConfigBase):
    """Test quality configuration."""
    
    # Quality Metrics
    min_assertions_per_test: int = 1
    max_assertions_per_test: int = 10
    min_test_coverage: float = 80.0
    
    # Code Quality
    enforce_type_hints: bool = True
    enforce_docstrings: bool = True
    max_line_length: int = 120
    max_complexity: int = 10
    
    # Test Quality Checks
    check_test_isolation: bool = True
    check_test_naming: bool = True
    check_test_organization: bool = True
    detect_flaky_tests: bool = True
    
    # Quality Thresholds
    max_test_duration_seconds: int = 5
    max_setup_duration_seconds: int = 2
    max_teardown_duration_seconds: int = 2
    
    # Linting and Formatting
    use_black: bool = True
    use_isort: bool = True
    use_flake8: bool = True
    use_mypy: bool = False
    
    def validate(self) -> List[str]:
        """Validate quality configuration."""
        errors = []
        
        if self.min_assertions_per_test < 0:
            errors.append("Min assertions per test cannot be negative")
        
        if self.max_assertions_per_test < self.min_assertions_per_test:
            errors.append("Max assertions cannot be less than min assertions")
        
        if not 0 <= self.min_test_coverage <= 100:
            errors.append("Min test coverage must be between 0 and 100")
        
        if self.max_line_length <= 0:
            errors.append("Max line length must be positive")
        
        if self.max_complexity <= 0:
            errors.append("Max complexity must be positive")
        
        return errors


@dataclass
class TestingConfig(ConfigBase):
    """Combined testing configuration."""
    
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    
    # Testing Profiles
    profiles: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "quick": {
            "generation.max_iterations": 2,
            "execution.max_parallel_tests": 20,
            "quality.min_test_coverage": 70.0
        },
        "thorough": {
            "generation.max_iterations": 10,
            "generation.generate_edge_cases": True,
            "quality.min_test_coverage": 95.0
        },
        "performance": {
            "generation.generate_performance_tests": True,
            "execution.timeout_per_test": 60,
            "execution.memory_limit_mb": 4096
        }
    })
    
    def apply_profile(self, profile_name: str):
        """Apply a testing profile."""
        if profile_name in self.profiles:
            profile = self.profiles[profile_name]
            for key, value in profile.items():
                parts = key.split('.')
                if len(parts) == 2:
                    section, attr = parts
                    if hasattr(self, section):
                        section_obj = getattr(self, section)
                        if hasattr(section_obj, attr):
                            setattr(section_obj, attr, value)
    
    def validate(self) -> List[str]:
        """Validate all testing configurations."""
        errors = []
        errors.extend(self.generation.validate())
        errors.extend(self.execution.validate())
        errors.extend(self.quality.validate())
        return errors


__all__ = [
    'GenerationConfig',
    'ExecutionConfig',
    'QualityConfig',
    'TestingConfig'
]