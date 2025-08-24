"""
Test script for Universal AST Abstraction System

Tests the codebase-agnostic AST abstraction across multiple languages.
"""

import time
import tempfile
import os
from pathlib import Path
from testmaster.core.feature_flags import FeatureFlags
from testmaster.core.ast_abstraction import UniversalASTAbstractor, LanguageParserRegistry

class UniversalASTSystemTest:
    """Test suite for Universal AST Abstraction."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        self.test_projects = []
        
    async def run_all_tests(self):
        """Run all AST abstraction tests."""
        print("=" * 60)
        print("TestMaster Universal AST Abstraction Test")
        print("=" * 60)
        
        # Initialize feature flags
        FeatureFlags.initialize("unified_testmaster_config.yaml")
        
        # Create AST abstractor
        abstractor = UniversalASTAbstractor({
            'supported_languages': ['python', 'javascript', 'typescript', 'java', 'csharp', 'go', 'rust'],
            'semantic_analysis': True
        })
        
        # Test different language files
        await self.test_python_ast(abstractor)
        await self.test_javascript_ast(abstractor)
        await self.test_typescript_ast(abstractor)
        await self.test_java_ast(abstractor)
        await self.test_mixed_project_ast(abstractor)
        await self.test_real_testmaster_project(abstractor)
        
        # Display results
        self.display_results()
    
    async def test_python_ast(self, abstractor: UniversalASTAbstractor):
        """Test Python AST abstraction."""
        print("\n[*] Testing Python AST Abstraction...")
        
        try:
            # Create temporary Python file
            python_code = '''
"""
Sample Python module for AST testing.
"""

import os
import json
from typing import List, Dict, Optional

class DataProcessor:
    """A class for processing data."""
    
    def __init__(self, config: Dict[str, str] = None):
        self.config = config or {}
        self.data = []
    
    async def process_data(self, input_data: List[Dict]) -> Dict:
        """Process input data asynchronously."""
        result = {}
        for item in input_data:
            if 'id' in item:
                result[item['id']] = item
        return result
    
    def validate_data(self, data: Dict) -> bool:
        """Validate data structure."""
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        return True

def utility_function(param1: str, param2: int = 10) -> str:
    """A utility function."""
    if param2 > 5:
        return f"Result: {param1} - {param2}"
    return "Invalid"

# Global variable
GLOBAL_CONFIG = {"version": "1.0"}
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(python_code)
                temp_file = f.name
            
            # Create Universal AST
            universal_ast = abstractor.create_universal_ast(temp_file, 'python')
            
            # Validate results
            success = (
                universal_ast.total_functions >= 3 and  # DataProcessor methods + utility_function
                universal_ast.total_classes >= 1 and    # DataProcessor class
                len(universal_ast.modules) == 1 and
                universal_ast.modules[0].language == 'python'
            )
            
            # Display detailed results
            module = universal_ast.modules[0]
            print(f"   [+] Module: {module.name}")
            print(f"   [+] Language: {module.language}")
            print(f"   [+] Functions found: {len(module.functions)}")
            for func in module.functions:
                print(f"      - {func.name}({'async' if func.is_async else 'sync'}) - {len(func.parameters)} params")
            
            print(f"   [+] Classes found: {len(module.classes)}")
            for cls in module.classes:
                print(f"      - {cls.name} - {len(cls.methods)} methods, {len(cls.fields)} fields")
            
            print(f"   [+] Imports found: {len(module.imports)}")
            for imp in module.imports:
                items = f" ({', '.join(imp.imported_items)})" if imp.imported_items else ""
                print(f"      - {imp.module_name}{items}")
            
            print(f"   [+] Variables found: {len(module.variables)}")
            print(f"   [+] Lines of code: {module.lines_of_code}")
            print(f"   [+] Analysis duration: {universal_ast.analysis_duration:.3f}s")
            
            # Test semantic analysis
            if universal_ast.semantic_analysis:
                print(f"   [+] Semantic analysis completed")
                print(f"      - Complexity metrics: {len(universal_ast.semantic_analysis.get('complexity_analysis', {}))}")
                print(f"      - Pattern analysis: {len(universal_ast.semantic_analysis.get('pattern_analysis', {}))}")
            
            self.test_results['python_ast'] = success
            self.test_projects.append(temp_file)
            
        except Exception as e:
            print(f"   [!] Python AST test failed: {e}")
            self.test_results['python_ast'] = False
    
    async def test_javascript_ast(self, abstractor: UniversalASTAbstractor):
        """Test JavaScript AST abstraction."""
        print("\n[*] Testing JavaScript AST Abstraction...")
        
        try:
            # Create temporary JavaScript file
            js_code = '''
import { Router } from 'express';
import axios from 'axios';

/**
 * API Server class for handling requests
 */
class APIServer {
    constructor(port = 3000) {
        this.port = port;
        this.router = Router();
        this.setupRoutes();
    }
    
    async handleRequest(req, res) {
        try {
            const data = await this.processData(req.body);
            res.json(data);
        } catch (error) {
            res.status(500).json({ error: error.message });
        }
    }
    
    processData(inputData) {
        return inputData.map(item => ({
            ...item,
            processed: true,
            timestamp: new Date()
        }));
    }
    
    setupRoutes() {
        this.router.get('/api/data', this.handleRequest.bind(this));
        this.router.post('/api/process', this.handleRequest.bind(this));
    }
}

// Utility functions
const validateInput = (data) => {
    if (!data || typeof data !== 'object') {
        throw new Error('Invalid input data');
    }
    return true;
};

const asyncHelper = async (callback) => {
    try {
        const result = await callback();
        return { success: true, data: result };
    } catch (error) {
        return { success: false, error: error.message };
    }
};

// Constants
const DEFAULT_CONFIG = {
    timeout: 5000,
    retries: 3
};

export default APIServer;
export { validateInput, asyncHelper, DEFAULT_CONFIG };
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(js_code)
                temp_file = f.name
            
            # Create Universal AST
            universal_ast = abstractor.create_universal_ast(temp_file, 'javascript')
            
            # Validate results
            module = universal_ast.modules[0]
            success = (
                len(module.functions) >= 2 and  # Utility functions
                len(module.classes) >= 1 and    # APIServer class
                len(module.imports) >= 2 and    # ES6 imports
                module.language == 'javascript'
            )
            
            # Display results
            print(f"   [+] Module: {module.name}")
            print(f"   [+] Language: {module.language}")
            print(f"   [+] Functions found: {len(module.functions)}")
            print(f"   [+] Classes found: {len(module.classes)}")
            print(f"   [+] Imports found: {len(module.imports)}")
            print(f"   [+] Variables found: {len(module.variables)}")
            print(f"   [+] Lines of code: {module.lines_of_code}")
            
            self.test_results['javascript_ast'] = success
            self.test_projects.append(temp_file)
            
        except Exception as e:
            print(f"   [!] JavaScript AST test failed: {e}")
            self.test_results['javascript_ast'] = False
    
    async def test_typescript_ast(self, abstractor: UniversalASTAbstractor):
        """Test TypeScript AST abstraction."""
        print("\n[*] Testing TypeScript AST Abstraction...")
        
        try:
            # Create temporary TypeScript file
            ts_code = '''
interface User {
    id: number;
    name: string;
    email?: string;
}

interface ApiResponse<T> extends Response {
    data: T;
    success: boolean;
}

class UserService {
    private users: User[] = [];
    
    constructor(private apiUrl: string) {}
    
    async getUser(id: number): Promise<User | null> {
        const user = this.users.find(u => u.id === id);
        return user || null;
    }
    
    async createUser(userData: Partial<User>): Promise<User> {
        const newUser: User = {
            id: Date.now(),
            name: userData.name || '',
            email: userData.email
        };
        this.users.push(newUser);
        return newUser;
    }
}

type UserRole = 'admin' | 'user' | 'guest';

const createUserService = (apiUrl: string): UserService => {
    return new UserService(apiUrl);
};

export { UserService, User, ApiResponse, UserRole, createUserService };
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
                f.write(ts_code)
                temp_file = f.name
            
            # Create Universal AST
            universal_ast = abstractor.create_universal_ast(temp_file, 'typescript')
            
            # Validate results
            module = universal_ast.modules[0]
            success = (
                len(module.functions) >= 1 and  # createUserService function
                len(module.classes) >= 1 and    # UserService + interfaces (treated as classes)
                module.language == 'typescript'
            )
            
            # Display results
            print(f"   [+] Module: {module.name}")
            print(f"   [+] Language: {module.language}")
            print(f"   [+] Functions found: {len(module.functions)}")
            print(f"   [+] Classes/Interfaces found: {len(module.classes)}")
            for cls in module.classes:
                cls_type = "Interface" if cls.is_interface else "Class"
                print(f"      - {cls.name} ({cls_type}) - {len(cls.methods)} methods, {len(cls.fields)} fields")
            
            self.test_results['typescript_ast'] = success
            self.test_projects.append(temp_file)
            
        except Exception as e:
            print(f"   [!] TypeScript AST test failed: {e}")
            self.test_results['typescript_ast'] = False
    
    async def test_java_ast(self, abstractor: UniversalASTAbstractor):
        """Test Java AST abstraction."""
        print("\n[*] Testing Java AST Abstraction...")
        
        try:
            # Create temporary Java file
            java_code = '''
package com.example.testmaster;

import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;

/**
 * Data processor for handling business logic
 */
public class DataProcessor {
    private List<String> data;
    private static final String VERSION = "1.0";
    
    public DataProcessor() {
        this.data = new ArrayList<>();
    }
    
    public void addData(String item) {
        if (item != null && !item.isEmpty()) {
            data.add(item);
        }
    }
    
    public List<String> processData() throws ProcessingException {
        if (data.isEmpty()) {
            throw new ProcessingException("No data to process");
        }
        return new ArrayList<>(data);
    }
    
    public static String getVersion() {
        return VERSION;
    }
}

class ProcessingException extends Exception {
    public ProcessingException(String message) {
        super(message);
    }
}
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
                f.write(java_code)
                temp_file = f.name
            
            # Create Universal AST
            universal_ast = abstractor.create_universal_ast(temp_file, 'java')
            
            # Validate results
            module = universal_ast.modules[0]
            success = (
                len(module.classes) >= 2 and    # DataProcessor + ProcessingException
                len(module.imports) >= 2 and    # Java imports
                module.language == 'java'
            )
            
            # Display results
            print(f"   [+] Module: {module.name}")
            print(f"   [+] Language: {module.language}")
            print(f"   [+] Classes found: {len(module.classes)}")
            print(f"   [+] Imports found: {len(module.imports)}")
            print(f"   [+] Lines of code: {module.lines_of_code}")
            
            self.test_results['java_ast'] = success
            self.test_projects.append(temp_file)
            
        except Exception as e:
            print(f"   [!] Java AST test failed: {e}")
            self.test_results['java_ast'] = False
    
    async def test_mixed_project_ast(self, abstractor: UniversalASTAbstractor):
        """Test mixed-language project AST abstraction."""
        print("\n[*] Testing Mixed-Language Project AST Abstraction...")
        
        try:
            # Create temporary project directory
            import tempfile
            project_dir = Path(tempfile.mkdtemp(prefix="test_mixed_"))
            
            # Create Python file
            (project_dir / "backend.py").write_text('''
class APIController:
    def handle_request(self, request):
        return {"status": "success"}
''')
            
            # Create JavaScript file
            (project_dir / "frontend.js").write_text('''
class UIController {
    render() {
        console.log("Rendering UI");
    }
}
''')
            
            # Create TypeScript file
            (project_dir / "types.ts").write_text('''
interface Config {
    apiUrl: string;
    timeout: number;
}
''')
            
            # Create Universal AST for entire project
            universal_ast = abstractor.create_project_ast(str(project_dir))
            
            # Validate results
            languages_found = set(module.language for module in universal_ast.modules)
            success = (
                len(universal_ast.modules) >= 3 and
                len(languages_found) >= 2 and  # Multiple languages
                universal_ast.total_classes >= 2  # Classes from different languages
            )
            
            # Display results
            print(f"   [+] Project path: {universal_ast.project_path}")
            print(f"   [+] Modules analyzed: {len(universal_ast.modules)}")
            print(f"   [+] Languages detected: {sorted(languages_found)}")
            print(f"   [+] Total functions: {universal_ast.total_functions}")
            print(f"   [+] Total classes: {universal_ast.total_classes}")
            print(f"   [+] Total lines: {universal_ast.total_lines}")
            print(f"   [+] Analysis duration: {universal_ast.analysis_duration:.3f}s")
            
            # Test cross-references
            if universal_ast.function_call_graph:
                print(f"   [+] Function call graph: {len(universal_ast.function_call_graph)} entries")
            if universal_ast.class_hierarchy:
                print(f"   [+] Class hierarchy: {len(universal_ast.class_hierarchy)} entries")
            if universal_ast.dependency_graph:
                print(f"   [+] Dependency graph: {len(universal_ast.dependency_graph)} entries")
            
            self.test_results['mixed_project_ast'] = success
            self.test_projects.append(str(project_dir))
            
        except Exception as e:
            print(f"   [!] Mixed project AST test failed: {e}")
            self.test_results['mixed_project_ast'] = False
    
    async def test_real_testmaster_project(self, abstractor: UniversalASTAbstractor):
        """Test real TestMaster project AST abstraction."""
        print("\n[*] Testing Real TestMaster Project AST Abstraction...")
        
        try:
            # Use current TestMaster directory
            current_dir = Path.cwd()
            
            # Create Universal AST for TestMaster project
            universal_ast = abstractor.create_project_ast(str(current_dir))
            
            # Validate results
            success = (
                len(universal_ast.modules) > 0 and
                universal_ast.total_files > 0 and
                universal_ast.analysis_duration > 0
            )
            
            # Display comprehensive results
            print(f"   [+] Project path: {universal_ast.project_path}")
            print(f"   [+] Primary language: {universal_ast.language}")
            print(f"   [+] Modules analyzed: {len(universal_ast.modules)}")
            print(f"   [+] Total files: {universal_ast.total_files}")
            print(f"   [+] Total functions: {universal_ast.total_functions}")
            print(f"   [+] Total classes: {universal_ast.total_classes}")
            print(f"   [+] Total lines: {universal_ast.total_lines}")
            
            # Language breakdown
            language_stats = {}
            for module in universal_ast.modules:
                lang = module.language
                language_stats[lang] = language_stats.get(lang, 0) + 1
            
            print(f"   [+] Language breakdown:")
            for lang, count in sorted(language_stats.items()):
                print(f"      - {lang}: {count} modules")
            
            # Cross-reference analysis
            print(f"   [+] Function call graph: {len(universal_ast.function_call_graph)} entries")
            print(f"   [+] Class hierarchy: {len(universal_ast.class_hierarchy)} entries")
            print(f"   [+] Dependency graph: {len(universal_ast.dependency_graph)} entries")
            
            # Semantic analysis results
            if universal_ast.semantic_analysis:
                print(f"   [+] Semantic analysis completed:")
                for analysis_type, results in universal_ast.semantic_analysis.items():
                    if isinstance(results, dict):
                        print(f"      - {analysis_type}: {len(results)} metrics")
                    else:
                        print(f"      - {analysis_type}: available")
            
            print(f"   [+] Analysis duration: {universal_ast.analysis_duration:.3f}s")
            
            self.test_results['real_testmaster_ast'] = success
            
        except Exception as e:
            print(f"   [!] Real TestMaster AST test failed: {e}")
            self.test_results['real_testmaster_ast'] = False
    
    def display_results(self):
        """Display test results summary."""
        print("\n" + "=" * 60)
        print("Universal AST Abstraction Test Results")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for test_name, result in self.test_results.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("All Universal AST Abstraction tests PASSED!")
        else:
            print("Some tests failed - check implementation")
        
        execution_time = time.time() - self.start_time
        print(f"Total execution time: {execution_time:.2f} seconds")
        
        # Test parser registry
        print(f"\nLanguage Parser Registry Status:")
        print(f"  Supported languages: {len(LanguageParserRegistry.get_all_parsers())}")
        for lang, parser in LanguageParserRegistry.get_all_parsers().items():
            print(f"    - {lang}: {parser.__class__.__name__}")
        
        # Cleanup temporary files
        print("\nCleaning up temporary test files...")
        for project_path in self.test_projects:
            try:
                if os.path.isfile(project_path):
                    os.unlink(project_path)
                    print(f"   Cleaned up: {project_path}")
                elif os.path.isdir(project_path):
                    import shutil
                    shutil.rmtree(project_path)
                    print(f"   Cleaned up: {project_path}")
            except Exception as e:
                print(f"   Failed to cleanup {project_path}: {e}")

async def main():
    """Main test execution."""
    test_suite = UniversalASTSystemTest()
    await test_suite.run_all_tests()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())