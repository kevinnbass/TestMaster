"""
Test script for Universal Language Detector

Tests the codebase-agnostic language detection system.
"""

import asyncio
import time
import tempfile
import os
from pathlib import Path
from testmaster.core.feature_flags import FeatureFlags
from testmaster.core.language_detection import UniversalLanguageDetector

class UniversalLanguageDetectorTest:
    """Test suite for Universal Language Detector."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        self.test_projects = []
        
    async def run_all_tests(self):
        """Run all language detector tests."""
        print("=" * 60)
        print("TestMaster Universal Language Detector Test")
        print("=" * 60)
        
        # Initialize feature flags
        FeatureFlags.initialize("unified_testmaster_config.yaml")
        
        # Create test detector
        detector = UniversalLanguageDetector({
            'supported_languages': ['python', 'javascript', 'java', 'csharp', 'go', 'rust'],
            'fallback_analyzers': ['ast_generic', 'text_pattern', 'ai_inference']
        })
        
        # Test different project types
        await self.test_python_project(detector)
        await self.test_javascript_project(detector)
        await self.test_mixed_language_project(detector)
        await self.test_real_testmaster_project(detector)
        
        # Display results
        self.display_results()
    
    async def test_python_project(self, detector: UniversalLanguageDetector):
        """Test detection of Python project."""
        print("\\n[*] Testing Python Project Detection...")
        
        try:
            # Create temporary Python project
            project_dir = self._create_python_test_project()
            
            # Detect codebase
            profile = detector.detect_codebase(str(project_dir))
            
            # Validate results
            success = (
                len(profile.languages) > 0 and
                profile.languages[0].name == 'python' and
                len(profile.frameworks) > 0 and
                any(fw.name == 'pytest' for fw in profile.frameworks)
            )
            
            print(f"   [+] Languages detected: {[lang.name for lang in profile.languages]}")
            print(f"   [+] Frameworks detected: {[fw.name for fw in profile.frameworks]}")
            print(f"   [+] Build systems: {[bs.name for bs in profile.build_systems]}")
            print(f"   [+] Total files: {profile.total_files}, Total lines: {profile.total_lines}")
            print(f"   [+] Analysis duration: {profile.analysis_duration:.3f}s")
            
            self.test_results['python_project'] = success
            self.test_projects.append(project_dir)
            
        except Exception as e:
            print(f"   [!] Python project test failed: {e}")
            self.test_results['python_project'] = False
    
    async def test_javascript_project(self, detector: UniversalLanguageDetector):
        """Test detection of JavaScript project."""
        print("\\n[*] Testing JavaScript Project Detection...")
        
        try:
            # Create temporary JavaScript project
            project_dir = self._create_javascript_test_project()
            
            # Detect codebase
            profile = detector.detect_codebase(str(project_dir))
            
            # Validate results
            success = (
                len(profile.languages) > 0 and
                profile.languages[0].name == 'javascript' and
                any(bs.name == 'npm' for bs in profile.build_systems)
            )
            
            print(f"   [+] Languages detected: {[lang.name for lang in profile.languages]}")
            print(f"   [+] Frameworks detected: {[fw.name for fw in profile.frameworks]}")
            print(f"   [+] Dependencies: {len(profile.dependencies)}")
            print(f"   [+] Architectural patterns: {profile.architectural_patterns}")
            
            self.test_results['javascript_project'] = success
            self.test_projects.append(project_dir)
            
        except Exception as e:
            print(f"   [!] JavaScript project test failed: {e}")
            self.test_results['javascript_project'] = False
    
    async def test_mixed_language_project(self, detector: UniversalLanguageDetector):
        """Test detection of mixed-language project."""
        print("\\n[*] Testing Mixed Language Project Detection...")
        
        try:
            # Create temporary mixed project
            project_dir = self._create_mixed_test_project()
            
            # Detect codebase
            profile = detector.detect_codebase(str(project_dir))
            
            # Validate results
            detected_languages = [lang.name for lang in profile.languages]
            success = (
                len(profile.languages) >= 2 and
                'python' in detected_languages and
                'javascript' in detected_languages
            )
            
            print(f"   [+] Languages detected: {detected_languages}")
            print(f"   [+] Language percentages: {[(lang.name, f'{lang.percentage:.1f}%') for lang in profile.languages]}")
            print(f"   [+] Complexity metrics: {profile.complexity_metrics}")
            print(f"   [+] Testing capabilities: {profile.testing_capabilities}")
            
            self.test_results['mixed_language_project'] = success
            self.test_projects.append(project_dir)
            
        except Exception as e:
            print(f"   [!] Mixed language project test failed: {e}")
            self.test_results['mixed_language_project'] = False
    
    async def test_real_testmaster_project(self, detector: UniversalLanguageDetector):
        """Test detection of real TestMaster project."""
        print("\\n[*] Testing Real TestMaster Project Detection...")
        
        try:
            # Use current TestMaster directory
            current_dir = Path.cwd()
            
            # Detect codebase
            profile = detector.detect_codebase(str(current_dir))
            
            # Validate results
            success = (
                len(profile.languages) > 0 and
                profile.total_files > 0 and
                profile.analysis_duration > 0
            )
            
            print(f"   [+] Project path: {profile.project_path}")
            print(f"   [+] Languages detected: {[(lang.name, f'{lang.percentage:.1f}%') for lang in profile.languages]}")
            print(f"   [+] Frameworks detected: {[(fw.name, fw.language) for fw in profile.frameworks]}")
            print(f"   [+] Build systems: {[bs.name for bs in profile.build_systems]}")
            print(f"   [+] Total files: {profile.total_files}, Total lines: {profile.total_lines}")
            print(f"   [+] Complexity metrics:")
            for metric, value in profile.complexity_metrics.items():
                print(f"      - {metric}: {value}")
            print(f"   [+] Architectural patterns: {profile.architectural_patterns}")
            print(f"   [+] Testing capabilities: {profile.testing_capabilities}")
            print(f"   [+] CI/CD capabilities: {profile.ci_cd_capabilities}")
            print(f"   [+] Analysis duration: {profile.analysis_duration:.3f}s")
            
            self.test_results['real_testmaster_project'] = success
            
        except Exception as e:
            print(f"   [!] Real TestMaster project test failed: {e}")
            self.test_results['real_testmaster_project'] = False
    
    def _create_python_test_project(self) -> Path:
        """Create a temporary Python test project."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_python_"))
        
        # Create Python files
        (temp_dir / "main.py").write_text("""
import requests
import json
from typing import List, Dict

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process_data(self, input_data: List[Dict]) -> Dict:
        result = {}
        for item in input_data:
            if 'id' in item:
                result[item['id']] = item
        return result
    
    async def fetch_data(self, url: str) -> Dict:
        # Async function with API call
        response = requests.get(url)
        return response.json()

if __name__ == "__main__":
    processor = DataProcessor()
    print("Data processor initialized")
""")
        
        (temp_dir / "utils.py").write_text("""
import os
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(level=logging.INFO)

def read_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)

@decorator
def validate_input(func):
    def wrapper(*args, **kwargs):
        # Input validation logic
        return func(*args, **kwargs)
    return wrapper
""")
        
        # Create test files
        test_dir = temp_dir / "tests"
        test_dir.mkdir()
        
        (test_dir / "test_main.py").write_text("""
import pytest
from unittest.mock import Mock, patch
from main import DataProcessor

class TestDataProcessor:
    def setup_method(self):
        self.processor = DataProcessor()
    
    def test_process_data(self):
        input_data = [{'id': 1, 'name': 'test'}]
        result = self.processor.process_data(input_data)
        assert result[1]['name'] == 'test'
    
    @pytest.mark.asyncio
    async def test_fetch_data(self):
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = {'status': 'ok'}
            result = await self.processor.fetch_data('http://test.com')
            assert result['status'] == 'ok'
""")
        
        # Create configuration files
        (temp_dir / "requirements.txt").write_text("""
requests==2.28.1
pytest==7.2.0
pytest-asyncio==0.20.0
""")
        
        (temp_dir / "setup.py").write_text("""
from setuptools import setup, find_packages

setup(
    name="test-python-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "pytest",
    ]
)
""")
        
        (temp_dir / "pytest.ini").write_text("""
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
""")
        
        return temp_dir
    
    def _create_javascript_test_project(self) -> Path:
        """Create a temporary JavaScript test project."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_js_"))
        
        # Create JavaScript files
        (temp_dir / "index.js").write_text("""
const express = require('express');
const axios = require('axios');

class APIServer {
    constructor(port = 3000) {
        this.app = express();
        this.port = port;
        this.setupRoutes();
    }
    
    setupRoutes() {
        this.app.get('/api/data', async (req, res) => {
            try {
                const response = await axios.get('https://api.example.com/data');
                res.json(response.data);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
        
        this.app.post('/api/process', (req, res) => {
            const data = req.body;
            const processed = this.processData(data);
            res.json(processed);
        });
    }
    
    processData(data) {
        return data.map(item => ({
            ...item,
            processed: true,
            timestamp: new Date()
        }));
    }
    
    start() {
        this.app.listen(this.port, () => {
            console.log(`Server running on port ${this.port}`);
        });
    }
}

module.exports = APIServer;
""")
        
        (temp_dir / "utils.js").write_text("""
const fs = require('fs');
const path = require('path');

function readConfig(configPath) {
    const content = fs.readFileSync(configPath, 'utf8');
    return JSON.parse(content);
}

function validateData(data) {
    if (!data || typeof data !== 'object') {
        throw new Error('Invalid data format');
    }
    return true;
}

const asyncHelper = async (callback) => {
    try {
        const result = await callback();
        return { success: true, data: result };
    } catch (error) {
        return { success: false, error: error.message };
    }
};

module.exports = {
    readConfig,
    validateData,
    asyncHelper
};
""")
        
        # Create test files
        test_dir = temp_dir / "tests"
        test_dir.mkdir()
        
        (test_dir / "api.test.js").write_text("""
const request = require('supertest');
const APIServer = require('../index');

describe('API Server', () => {
    let server;
    let app;
    
    beforeEach(() => {
        server = new APIServer(3001);
        app = server.app;
    });
    
    describe('GET /api/data', () => {
        it('should return data from external API', async () => {
            const response = await request(app)
                .get('/api/data')
                .expect(200);
            
            expect(response.body).toBeDefined();
        });
    });
    
    describe('POST /api/process', () => {
        it('should process data correctly', async () => {
            const testData = [{ id: 1, name: 'test' }];
            
            const response = await request(app)
                .post('/api/process')
                .send(testData)
                .expect(200);
            
            expect(response.body[0]).toHaveProperty('processed', true);
            expect(response.body[0]).toHaveProperty('timestamp');
        });
    });
});
""")
        
        # Create package.json
        (temp_dir / "package.json").write_text("""
{
  "name": "test-javascript-project",
  "version": "1.0.0",
  "description": "Test JavaScript project for language detection",
  "main": "index.js",
  "scripts": {
    "start": "node index.js",
    "test": "jest",
    "dev": "nodemon index.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "axios": "^1.2.0"
  },
  "devDependencies": {
    "jest": "^29.3.1",
    "supertest": "^6.3.3",
    "nodemon": "^2.0.20"
  },
  "jest": {
    "testEnvironment": "node",
    "testMatch": ["**/tests/**/*.test.js"]
  }
}
""")
        
        # Create jest config
        (temp_dir / "jest.config.js").write_text("""
module.exports = {
  testEnvironment: 'node',
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html']
};
""")
        
        return temp_dir
    
    def _create_mixed_test_project(self) -> Path:
        """Create a temporary mixed-language test project."""
        temp_dir = Path(tempfile.mkdtemp(prefix="test_mixed_"))
        
        # Create Python backend
        backend_dir = temp_dir / "backend"
        backend_dir.mkdir()
        
        (backend_dir / "app.py").write_text("""
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

@app.route('/api/users', methods=['GET'])
def get_users():
    users = User.query.all()
    return jsonify([{'id': u.id, 'username': u.username} for u in users])

@app.route('/api/users', methods=['POST'])
def create_user():
    data = request.json
    user = User(username=data['username'], email=data['email'])
    db.session.add(user)
    db.session.commit()
    return jsonify({'id': user.id})

if __name__ == '__main__':
    app.run(debug=True)
""")
        
        # Create JavaScript frontend
        frontend_dir = temp_dir / "frontend"
        frontend_dir.mkdir()
        
        (frontend_dir / "app.js").write_text("""
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const UserList = () => {
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);
    
    useEffect(() => {
        fetchUsers();
    }, []);
    
    const fetchUsers = async () => {
        try {
            const response = await axios.get('/api/users');
            setUsers(response.data);
        } catch (error) {
            console.error('Error fetching users:', error);
        } finally {
            setLoading(false);
        }
    };
    
    const createUser = async (userData) => {
        try {
            await axios.post('/api/users', userData);
            fetchUsers(); // Refresh list
        } catch (error) {
            console.error('Error creating user:', error);
        }
    };
    
    if (loading) return <div>Loading...</div>;
    
    return (
        <div className="user-list">
            <h2>Users</h2>
            {users.map(user => (
                <div key={user.id} className="user-item">
                    {user.username}
                </div>
            ))}
        </div>
    );
};

export default UserList;
""")
        
        # Create Go microservice
        services_dir = temp_dir / "services"
        services_dir.mkdir()
        
        (services_dir / "auth.go").write_text("""
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "time"
    
    "github.com/gorilla/mux"
    "github.com/golang-jwt/jwt/v4"
)

type User struct {
    ID       int    `json:"id"`
    Username string `json:"username"`
    Password string `json:"-"`
}

type AuthService struct {
    users  map[string]*User
    secret []byte
}

func NewAuthService() *AuthService {
    return &AuthService{
        users:  make(map[string]*User),
        secret: []byte("your-secret-key"),
    }
}

func (as *AuthService) Login(w http.ResponseWriter, r *http.Request) {
    var credentials struct {
        Username string `json:"username"`
        Password string `json:"password"`
    }
    
    if err := json.NewDecoder(r.Body).Decode(&credentials); err != nil {
        http.Error(w, "Invalid JSON", http.StatusBadRequest)
        return
    }
    
    user, exists := as.users[credentials.Username]
    if !exists || user.Password != credentials.Password {
        http.Error(w, "Invalid credentials", http.StatusUnauthorized)
        return
    }
    
    token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
        "user_id": user.ID,
        "exp":     time.Now().Add(time.Hour * 24).Unix(),
    })
    
    tokenString, err := token.SignedString(as.secret)
    if err != nil {
        http.Error(w, "Error generating token", http.StatusInternalServerError)
        return
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]string{"token": tokenString})
}

func main() {
    authService := NewAuthService()
    
    r := mux.NewRouter()
    r.HandleFunc("/auth/login", authService.Login).Methods("POST")
    
    fmt.Println("Auth service starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", r))
}
""")
        
        # Create project configuration files
        (temp_dir / "requirements.txt").write_text("""
Flask==2.3.2
Flask-SQLAlchemy==3.0.5
""")
        
        (frontend_dir / "package.json").write_text("""
{
  "name": "frontend",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.2.0",
    "axios": "^1.2.0"
  },
  "devDependencies": {
    "jest": "^29.3.1"
  }
}
""")
        
        (services_dir / "go.mod").write_text("""
module auth-service

go 1.19

require (
    github.com/gorilla/mux v1.8.0
    github.com/golang-jwt/jwt/v4 v4.4.3
)
""")
        
        return temp_dir
    
    def display_results(self):
        """Display test results summary."""
        print("\\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("All Universal Language Detector tests PASSED!")
        else:
            print("Some tests failed - check implementation")
        
        execution_time = time.time() - self.start_time
        print(f"Total execution time: {execution_time:.2f} seconds")
        
        # Cleanup temporary directories
        print("\\nCleaning up temporary test projects...")
        for project_dir in self.test_projects:
            try:
                import shutil
                shutil.rmtree(project_dir)
                print(f"   Cleaned up: {project_dir}")
            except Exception as e:
                print(f"   Failed to cleanup {project_dir}: {e}")

async def main():
    """Main test execution."""
    test_suite = UniversalLanguageDetectorTest()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())