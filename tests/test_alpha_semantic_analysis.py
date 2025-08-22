#!/usr/bin/env python3
"""
Test Agent Alpha's Enhanced Semantic Analysis
"""

import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from enhanced_intelligence_linkage import EnhancedLinkageAnalyzer
    print("SUCCESS: Successfully imported EnhancedLinkageAnalyzer")
except ImportError as e:
    print(f"ERROR: Import error: {e}")
    sys.exit(1)

def test_semantic_classification():
    """Test the semantic classification on sample code snippets."""
    
    analyzer = EnhancedLinkageAnalyzer()
    
    # Test cases for different intent categories
    test_cases = [
        # API Endpoint
        ("""
from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/api/users', methods=['GET', 'POST'])
def users_endpoint():
    if request.method == 'GET':
        return jsonify({'users': []})
    return jsonify({'status': 'created'})
""", "api_endpoint"),
        
        # Data Processing
        ("""
import pandas as pd
import json

def process_data(csv_file):
    df = pd.read_csv(csv_file)
    df = df.transform(lambda x: x.strip() if isinstance(x, str) else x)
    return df.to_json()
""", "data_processing"),
        
        # Testing
        ("""
import unittest
from unittest.mock import Mock, patch

class TestUserService(unittest.TestCase):
    def setUp(self):
        self.mock_user = Mock()
    
    def test_user_creation(self):
        self.assertIsNotNone(self.mock_user)
        self.assertTrue(self.mock_user.is_valid())
""", "testing"),
        
        # Authentication
        ("""
import jwt
from werkzeug.security import check_password_hash, generate_password_hash

def authenticate_user(username, password):
    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        token = jwt.encode({'user_id': user.id}, SECRET_KEY)
        return token
    return None
""", "authentication")
    ]
    
    print("\nTesting Semantic Classification:")
    print("=" * 50)
    
    for i, (code, expected_intent) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        result = analyzer._classify_developer_intent(code)
        
        primary_intent = result['primary_intent']
        confidence = result['confidence']
        
        status = "PASS" if primary_intent == expected_intent else "FAIL"
        print(f"Expected: {expected_intent}")
        print(f"Got: {primary_intent} (confidence: {confidence:.2f})")
        print(f"Status: {status}")
        
        # Show top 3 scoring intents
        sorted_intents = sorted(result['all_intents'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Top intents: {sorted_intents}")

def test_conceptual_extraction():
    """Test conceptual element extraction."""
    
    analyzer = EnhancedLinkageAnalyzer()
    
    sample_code = """
import os
import sys
from pathlib import Path
from typing import Dict, List

class DataProcessor:
    def __init__(self, config):
        self.config = config
    
    async def process_data(self, data: List[Dict]):
        try:
            results = []
            for item in data:
                if item.get('valid'):
                    processed = self._transform_item(item)
                    results.append(processed)
            return results
        except Exception as e:
            raise ProcessingError(f"Failed to process: {e}")
    
    def _transform_item(self, item):
        return {k: v.upper() if isinstance(v, str) else v for k, v in item.items()}

@decorators.cache
def utility_function():
    return "cached_result"
"""
    
    print("\nTesting Conceptual Extraction:")
    print("=" * 50)
    
    concepts = analyzer._extract_conceptual_elements(sample_code)
    
    print(f"Classes found: {len(concepts['classes'])}")
    for cls in concepts['classes']:
        print(f"  - {cls['name']}: {len(cls['methods'])} methods")
    
    print(f"Functions found: {len(concepts['functions'])}")
    for func in concepts['functions']:
        print(f"  - {func['name']} ({func['args']} args, async: {func['is_async']})")
    
    print(f"Imports: {concepts['imports'][:5]}...")  # Show first 5
    print(f"Architectural patterns: {concepts['architectural_patterns']}")
    print(f"Complexity indicators: {concepts['complexity_indicators']}")

if __name__ == "__main__":
    print("Agent Alpha - Semantic Analysis Test Suite")
    print("=" * 60)
    
    try:
        test_semantic_classification()
        test_conceptual_extraction()
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()