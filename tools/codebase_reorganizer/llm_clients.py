#!/usr/bin/env python3
"""
LLM Client Implementations
==========================

Different LLM provider client implementations for code analysis.
"""

import json
import time
from typing import Dict, Any
import urllib.request
import urllib.error


class MockLLMClient:
    """Mock LLM client for testing and development"""

    def analyze_code(self, prompt: str) -> str:
        """Return mock analysis response"""
        time.sleep(0.1)  # Simulate API delay

        # Extract basic info from prompt
        lines = prompt.split('\n')
        file_path = "unknown"
        for line in lines:
            if 'Relative path:' in line:
                file_path = line.split(':')[1].strip()
                break

        # Generate appropriate mock response based on file path
        if 'security' in file_path.lower() or 'auth' in file_path.lower():
            return self._mock_security_response()
        elif 'test' in file_path.lower():
            return self._mock_test_response()
        elif 'api' in file_path.lower():
            return self._mock_api_response()
        else:
            return self._mock_general_response()

    def _mock_security_response(self) -> str:
        return '''{
    "summary": "This module handles authentication and authorization for the system, implementing secure token-based access control with encryption.",
    "functionality": "Provides JWT token generation and validation, user authentication, role-based permissions, and secure password hashing using bcrypt.",
    "dependencies": "Uses cryptography library for encryption, JWT for tokens, and integrates with user management systems.",
    "security": "Handles sensitive authentication data, implements secure token storage, and provides protection against common attacks like CSRF and XSS.",
    "testing": "Requires comprehensive security testing, mock authentication, and penetration testing scenarios.",
    "architecture": "Security service layer providing authentication and authorization services to the application.",
    "primary_classification": "security",
    "secondary_classifications": ["authentication", "encryption"],
    "reorganization": ["Move to src/core/security/", "Group with other security modules", "Ensure access control"],
    "confidence": 0.9,
    "key_features": ["JWT tokens", "role-based access", "password hashing", "CSRF protection"],
    "integration_points": ["User management", "API endpoints", "Database layer", "Frontend authentication"],
    "complexity": "medium",
    "maintainability": "Regular security updates required, follow OWASP guidelines, consider adding rate limiting"
}'''

    def _mock_test_response(self) -> str:
        return '''{
    "summary": "This module contains unit tests and integration tests for system components, ensuring code reliability and functionality.",
    "functionality": "Provides test cases for core functionality, mock objects, test fixtures, and assertion helpers for various system components.",
    "dependencies": "Uses pytest framework, mock library, and testing utilities with minimal external dependencies.",
    "security": "No direct security implications, but tests security features of other modules.",
    "testing": "Self-testing module that defines testing patterns and practices for the codebase.",
    "architecture": "Testing infrastructure supporting the development and validation process across all system layers.",
    "primary_classification": "testing",
    "secondary_classifications": ["unit_tests", "integration_tests", "test_framework"],
    "reorganization": ["Move to tests/ directory", "Organize by component tested", "Separate unit and integration tests"],
    "confidence": 0.85,
    "key_features": ["Test fixtures", "Mock objects", "Assertion helpers", "Coverage reporting"],
    "integration_points": ["All testable components", "CI/CD pipeline", "Coverage reporting tools"],
    "complexity": "low",
    "maintainability": "Tests should be maintained alongside code changes, consider test data management"
}'''

    def _mock_api_response(self) -> str:
        return '''{
    "summary": "This module implements REST API endpoints and request handling for the system, providing external interfaces.",
    "functionality": "Defines API routes, request validation, response formatting, and error handling for HTTP endpoints using FastAPI.",
    "dependencies": "Uses FastAPI framework, integrates with business logic and data layers through dependency injection.",
    "security": "Implements input validation, authentication middleware, and secure response handling with CORS support.",
    "testing": "Requires API testing, integration testing, and load testing scenarios with tools like Postman or pytest.",
    "architecture": "Presentation layer providing external API interfaces to the system with clear separation of concerns.",
    "primary_classification": "api",
    "secondary_classifications": ["web_framework", "request_handling", "rest_api"],
    "reorganization": ["Move to src/api/", "Group related endpoints", "Separate API from business logic"],
    "confidence": 0.88,
    "key_features": ["REST endpoints", "Request validation", "Error handling", "CORS support", "API documentation"],
    "integration_points": ["Business logic", "Database layer", "Authentication", "Frontend applications"],
    "complexity": "medium",
    "maintainability": "API versioning and documentation required, consider OpenAPI specification"
}'''

    def _mock_general_response(self) -> str:
        return '''{
    "summary": "This module provides utility functions and helper classes for common operations across the system.",
    "functionality": "Contains shared utilities, helper functions, and common data structures used by multiple components.",
    "dependencies": "Minimal dependencies, primarily standard library and basic external packages.",
    "security": "No direct security implications, but should follow secure coding practices.",
    "testing": "Requires unit testing for individual functions and integration testing for usage patterns.",
    "architecture": "Utility layer providing common functionality to other system components.",
    "primary_classification": "utility",
    "secondary_classifications": ["helpers", "common"],
    "reorganization": ["Move to src/utils/", "Group similar utilities", "Consider shared library"],
    "confidence": 0.7,
    "key_features": ["Helper functions", "Common utilities", "Shared code", "Data structures"],
    "integration_points": ["Multiple system components", "Cross-cutting concerns"],
    "complexity": "low",
    "maintainability": "Should remain simple and focused on utility functions, avoid feature creep"
}'''


class OpenAIClient:
    """OpenAI LLM client"""

    def __init__(self, api_key: str, model: str):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise Exception("OpenAI library not installed")

    def analyze_code(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"


class OllamaClient:
    """Ollama local LLM client"""

    def __init__(self, model: str):
        self.model = model
        self.base_url = "http://localhost:11434"

    def analyze_code(self, prompt: str) -> str:
        try:
            # Use built-in urllib instead of external requests library
            data = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            }).encode('utf-8')

            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=data,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'Python-URLLib'
                },
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get("response", "No response")

        except urllib.error.URLError as e:
            return f"Network Error: {e}"
        except Exception as e:
            return f"Error: {e}"


class AnthropicClient:
    """Anthropic Claude LLM client (placeholder)"""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def analyze_code(self, prompt: str) -> str:
        # Placeholder implementation
        return '''{
    "summary": "Anthropic Claude analysis not yet implemented",
    "functionality": "Code analysis using Anthropic Claude model",
    "confidence": 0.5,
    "primary_classification": "unknown"
}'''


class GroqClient:
    """Groq LLM client (placeholder)"""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def analyze_code(self, prompt: str) -> str:
        # Placeholder implementation
        return '''{
    "summary": "Groq analysis not yet implemented",
    "functionality": "Code analysis using Groq model",
    "confidence": 0.5,
    "primary_classification": "unknown"
}'''


def create_llm_client(provider: str, config: Dict[str, Any]):
    """Factory function to create appropriate LLM client"""
    if provider == "mock":
        return MockLLMClient()
    elif provider == "openai":
        return OpenAIClient(config.get("api_key", ""), config.get("model", "gpt-4"))
    elif provider == "ollama":
        return OllamaClient(config.get("model", "codellama"))
    elif provider == "anthropic":
        return AnthropicClient(config.get("api_key", ""), config.get("model", "claude-3"))
    elif provider == "groq":
        return GroqClient(config.get("api_key", ""), config.get("model", "mixtral"))
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

