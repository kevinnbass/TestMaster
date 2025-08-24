#!/usr/bin/env python3
"""
Test script for the LLM Intelligence System
Demonstrates the system working with a minimal example.
"""

import os
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Import the system components
from llm_intelligence_system import LLMIntelligenceScanner
from intelligence_integration_engine import IntelligenceIntegrationEngine
from reorganization_planner import ReorganizationPlanner


def create_test_files(test_dir: Path) -> None:
    """Create some test Python files for demonstration"""

    # Create directory structure
    (test_dir / "src" / "core" / "security").mkdir(parents=True, exist_ok=True)
    (test_dir / "src" / "utils").mkdir(parents=True, exist_ok=True)
    (test_dir / "tests").mkdir(parents=True, exist_ok=True)

    # Security module
    security_code = '''
"""
Security authentication module
"""
import jwt
import bcrypt
from typing import Dict, Optional

class AuthManager:
    """Handles user authentication and token management"""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return JWT token"""
        # Verify credentials (simplified)
        if self._verify_credentials(username, password):
            return self._generate_token(username)
        return None

    def _verify_credentials(self, username: str, password: str) -> bool:
        """Verify user credentials"""
        # In real implementation, check against database
        return True

    def _generate_token(self, username: str) -> str:
        """Generate JWT token for authenticated user"""
        import datetime
        payload = {
            'user': username,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate JWT token and return payload"""
        try:
            return jwt.decode(token, self.secret_key, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
'''

    # Utility module
    utility_code = '''
"""
General utility functions
"""
import os
import sys
from typing import List, Any

def setup_logging(log_file: str = "app.log") -> None:
    """Setup basic logging configuration"""
    import logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if division by zero"""
    try:
        return a / b
    except ZeroDivisionError:
        return default

def flatten_list(nested_list: List[Any]) -> List[Any]:
    """Flatten a nested list structure"""
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0.0
'''

    # Test file
    test_code = '''
"""
Test cases for the system
"""
import unittest
from src.core.security import AuthManager
from src.utils import setup_logging

class TestAuthManager(unittest.TestCase):
    """Test cases for AuthManager"""

    def setUp(self):
        """Setup test fixtures"""
        self.auth_manager = AuthManager("test_secret_key")

    def test_authenticate_valid_user(self):
        """Test authentication with valid credentials"""
        token = self.auth_manager.authenticate_user("testuser", "testpass")
        self.assertIsNotNone(token)
        self.assertIsInstance(token, str)

    def test_authenticate_invalid_user(self):
        """Test authentication with invalid credentials"""
        token = self.auth_manager.authenticate_user("invalid", "invalid")
        self.assertIsNone(token)

    def test_token_validation(self):
        """Test JWT token validation"""
        token = self.auth_manager.authenticate_user("testuser", "testpass")
        payload = self.auth_manager.validate_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload['user'], 'testuser')

if __name__ == '__main__':
    unittest.main()
'''

    # Write test files
    (test_dir / "src" / "core" / "security" / "__init__.py").write_text('')
    (test_dir / "src" / "core" / "security" / "auth.py").write_text(security_code)
    (test_dir / "src" / "utils" / "__init__.py").write_text('')
    (test_dir / "src" / "utils" / "helpers.py").write_text(utility_code)
    (test_dir / "tests" / "__init__.py").write_text('')
    (test_dir / "tests" / "test_auth.py").write_text(test_code)


def run_test():
    """Run a complete test of the intelligence system"""

    print("🧪 Testing LLM Intelligence System")
    print("=" * 50)

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        test_root = Path(temp_dir)
        print(f"Test directory: {test_root}")

        # Create test files
        print("Creating test Python files...")
        create_test_files(test_root)

        # Count test files
        python_files = list(test_root.rglob("*.py"))
        print(f"Created {len(python_files)} test files")

        try:
            # Initialize scanner with mock LLM
            print("\nInitializing LLM Intelligence Scanner...")
            scanner = LLMIntelligenceScanner(test_root, {
                'llm_provider': 'mock',
                'enable_static_analysis': False,  # Disable for faster testing
                'max_concurrent': 1
            })

            # Run scan
            print("Running intelligence scan...")
            output_file = test_root / "test_intelligence_map.json"
            intelligence_map = scanner.scan_and_analyze(output_file, max_files=5)

            print("✅ Scan completed successfully!"            print(f"   Files scanned: {intelligence_map.total_files_scanned}")
            print(f"   Lines analyzed: {intelligence_map.total_lines_analyzed}")

            # Show sample analysis
            if intelligence_map.intelligence_entries:
                sample_entry = intelligence_map.intelligence_entries[0]
                print("
📋 Sample Analysis:"                print(f"   File: {sample_entry.relative_path}")
                print(f"   Classification: {sample_entry.primary_classification}")
                print(f"   Confidence: {sample_entry.confidence_score:.2f}")
                print(f"   Summary: {sample_entry.module_summary[:100]}...")

            # Test integration engine
            print("\n🔗 Testing Integration Engine...")
            integration_engine = IntelligenceIntegrationEngine(test_root, {
                'enable_static_analysis': False
            })

            integrated_intelligence = integration_engine.integrate_intelligence(
                intelligence_map.__dict__ if hasattr(intelligence_map, '__dict__') else intelligence_map
            )

            print(f"✅ Integration completed - {len(integrated_intelligence)} entries")

            # Test reorganization planner
            print("\n📋 Testing Reorganization Planner...")
            planner = ReorganizationPlanner(test_root, {
                'min_confidence_threshold': 0.5  # Lower for testing
            })

            reorganization_plan = planner.create_reorganization_plan(
                intelligence_map.__dict__ if hasattr(intelligence_map, '__dict__') else intelligence_map,
                integrated_intelligence
            )

            print(f"✅ Reorganization plan created - {reorganization_plan.total_batches} batches")

            # Show plan summary
            print("
📊 Plan Summary:"            print(f"   Total tasks: {reorganization_plan.total_tasks}")
            print(f"   Total batches: {reorganization_plan.total_batches}")
            print(".1f")

            for i, batch in enumerate(reorganization_plan.batches[:2], 1):
                print(f"   Batch {i}: {batch.batch_name} ({batch.risk_level}) - {len(batch.tasks)} tasks")

            # Save test results
            test_results = {
                'test_timestamp': datetime.now().isoformat(),
                'test_directory': str(test_root),
                'files_created': len(python_files),
                'scan_results': {
                    'files_scanned': intelligence_map.total_files_scanned,
                    'lines_analyzed': intelligence_map.total_lines_analyzed
                },
                'integration_results': {
                    'entries_integrated': len(integrated_intelligence)
                },
                'planning_results': {
                    'total_batches': reorganization_plan.total_batches,
                    'total_tasks': reorganization_plan.total_tasks
                }
            }

            results_file = Path("test_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)

            print("
✅ Test completed successfully!"            print(f"📄 Results saved to: {results_file}")

            return True

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main test function"""
    print("🚀 LLM Intelligence System - Test Suite")
    print("=" * 60)
    print("This test will:")
    print("1. Create sample Python files")
    print("2. Run LLM intelligence scanning")
    print("3. Test integration with static analysis")
    print("4. Generate reorganization plan")
    print("5. Verify complete pipeline")
    print("=" * 60)

    success = run_test()

    if success:
        print("\n🎉 All tests passed! The system is working correctly.")
        print("\nNext steps:")
        print("1. Review test_results.json for detailed results")
        print("2. Try the full pipeline with: python run_intelligence_system.py --full-pipeline --max-files 10")
        print("3. Experiment with different LLM providers and configurations")
    else:
        print("\n❌ Tests failed. Check the error messages above.")
        print("Troubleshooting:")
        print("1. Ensure all dependencies are installed")
        print("2. Check file permissions in the test directory")
        print("3. Verify Python path includes the reorganizer modules")


if __name__ == "__main__":
    main()

