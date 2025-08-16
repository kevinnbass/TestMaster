#!/usr/bin/env python3
"""
TestMaster Exception Classes

Custom exceptions for TestMaster operations.
"""

class TestMasterException(Exception):
    """Base exception for all TestMaster operations."""
    pass

class GenerationError(TestMasterException):
    """Exception raised during test generation."""
    pass

class VerificationError(TestMasterException):
    """Exception raised during test verification."""
    pass

class ConversionError(TestMasterException):
    """Exception raised during test conversion."""
    pass

class ConfigurationError(TestMasterException):
    """Exception raised for configuration issues."""
    pass