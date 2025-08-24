
# AGENT D SECURITY INTEGRATION
try:
    from SECURITY_PATCHES.api_security_framework import APISecurityFramework
    from SECURITY_PATCHES.authentication_framework import SecurityFramework
    _security_framework = SecurityFramework()
    _api_security = APISecurityFramework()
    _SECURITY_ENABLED = True
except ImportError:
    _SECURITY_ENABLED = False
    print("Security frameworks not available - running without protection")

def apply_security_middleware():
    """Apply security middleware to requests"""
    if not _SECURITY_ENABLED:
        return True, {}
    
    from flask import request
    request_data = {
        'ip_address': request.remote_addr,
        'endpoint': request.path,
        'method': request.method,
        'user_agent': request.headers.get('User-Agent', ''),
        'body': request.get_json() if request.is_json else {},
        'query_params': dict(request.args),
        'headers': dict(request.headers)
    }
    
    return _api_security.validate_request(request_data)

"""
API Endpoint Validation System

Comprehensive validation of all API endpoints to ensure they are functional,
properly documented, and meet performance requirements.
"""

import os
import sys
import json
import time
import asyncio
import traceback
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from pathlib import Path
import re
import importlib.util

# Try to import requests, install if not available
try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


class APIEndpointValidator:
    """
    Comprehensive API endpoint validation system.
    
    Validates that all API endpoints:
    1. Are discoverable and documented
    2. Respond correctly to valid requests
    3. Handle invalid requests gracefully
    4. Meet performance requirements
    5. Follow REST API best practices
    6. Have proper authentication/authorization
    7. Return consistent error formats
    """
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30
        
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "base_url": base_url,
            "discovered_endpoints": [],
            "endpoint_tests": {},
            "performance_tests": {},
            "security_tests": {},
            "documentation_tests": {},
            "compliance_tests": {},
            "summary": {
                "total_endpoints": 0,
                "functional_endpoints": 0,
                "performance_compliant": 0,
                "security_compliant": 0,
                "documented_endpoints": 0,
                "overall_score": 0.0
            }
        }
        
        # API testing configuration
        self.performance_thresholds = {
            "response_time_ms": 5000,      # 5 seconds max
            "error_response_time_ms": 1000, # 1 second max for errors
            "concurrent_requests": 10       # Test with 10 concurrent requests
        }
        
        # Expected API structure patterns
        self.api_patterns = {
            "health": r"/api/health",
            "analysis": r"/api/analysis/.*",
            "security": r"/api/security/.*",
            "documentation": r"/api/documentation/.*",
            "integration": r"/api/integration/.*",
            "metrics": r"/api/metrics/.*",
            "dashboard": r"/api/dashboard/.*"
        }
        
        # Standard HTTP methods to test
        self.test_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        
        # Common error codes to test
        self.error_scenarios = [
            {"path": "/api/nonexistent", "expected_code": 404},
            {"path": "/api/health", "method": "DELETE", "expected_code": 405},
            {"path": "/api/analysis", "data": "invalid_json", "expected_code": 400}
        ]
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete API endpoint validation."""
        print("🌐 Starting Comprehensive API Endpoint Validation...")
        print("=" * 70)
        
        try:
            # Step 1: Discover all available endpoints
            await self._discover_endpoints()
            
            # Step 2: Test basic functionality
            await self._test_endpoint_functionality()
            
            # Step 3: Test performance
            await self._test_endpoint_performance()
            
            # Step 4: Test security
            await self._test_endpoint_security()
            
            # Step 5: Test documentation compliance
            await self._test_documentation_compliance()
            
            # Step 6: Test REST API compliance
            await self._test_rest_compliance()
            
            # Step 7: Generate comprehensive report
            self._generate_validation_report()
            
        except Exception as e:
            print(f"❌ Critical error during API validation: {e}")
            self.validation_results["critical_error"] = str(e)
            traceback.print_exc()
        
        return self.validation_results
    
    async def _discover_endpoints(self):
        """Discover all available API endpoints."""
        print("🔍 Discovering API Endpoints...")
        
        discovered_endpoints = []
        
        # Method 1: Check if server is running and try common endpoints
        server_running = await self._check_server_status()
        
        if server_running:
            # Try common endpoint patterns
            common_endpoints = [
                "/api/health",
                "/api/status", 
                "/api/version",
                "/api/analysis/classical",
                "/api/analysis/security",
                "/api/security/scan",
                "/api/security/compliance",
                "/api/documentation/generate",
                "/api/documentation/validate",
                "/api/integration/status",
                "/api/integration/agents",
                "/api/metrics/performance",
                "/api/metrics/system",
                "/api/dashboard/status",
                "/api/dashboard/data"
            ]
            
            for endpoint in common_endpoints:
                if await self._test_endpoint_exists(endpoint):
                    discovered_endpoints.append(endpoint)
        
        # Method 2: Scan source code for endpoint definitions
        code_endpoints = await self._scan_code_for_endpoints()
        discovered_endpoints.extend(code_endpoints)
        
        # Remove duplicates and sort
        discovered_endpoints = sorted(list(set(discovered_endpoints)))
        
        self.validation_results["discovered_endpoints"] = discovered_endpoints
        self.validation_results["summary"]["total_endpoints"] = len(discovered_endpoints)
        
        print(f"  ✅ Discovered {len(discovered_endpoints)} API endpoints")
        for endpoint in discovered_endpoints:
            print(f"    • {endpoint}")
    
    async def _check_server_status(self) -> bool:
        """Check if the API server is running."""
        try:
            response = self.session.get(f"{self.base_url}/api/health", timeout=5)
            return response.status_code < 500
        except:
            print("  ⚠️  API server not running - will test code structure only")
            return False
    
    async def _test_endpoint_exists(self, endpoint: str) -> bool:
        """Test if an endpoint exists."""
        try:
            response = self.session.get(f"{self.base_url}{endpoint}", timeout=5)
            return response.status_code != 404
        except:
            return False
    
    async def _scan_code_for_endpoints(self) -> List[str]:
        """Scan source code for API endpoint definitions."""
        endpoints = []
        
        # Look for Flask route decorators
        api_dirs = [
            "dashboard/api",
            "testmaster/dashboard/api",
            "api"
        ]
        
        for api_dir in api_dirs:
            api_path = Path(api_dir)
            if api_path.exists():
                for py_file in api_path.rglob("*.py"):
                    endpoints.extend(self._extract_endpoints_from_file(py_file))
        
        return endpoints
    
    def _extract_endpoints_from_file(self, file_path: Path) -> List[str]:
        """Extract API endpoints from a Python file."""
        endpoints = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for Flask route decorators
            route_patterns = [
                r"@.*\.route\s*\(\s*['\"]([^'\"]+)['\"]",
                r"@bp\.route\s*\(\s*['\"]([^'\"]+)['\"]",
                r"@blueprint\.route\s*\(\s*['\"]([^'\"]+)['\"]",
                r"@app\.route\s*\(\s*['\"]([^'\"]+)['\"]"
            ]
            
            for pattern in route_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                endpoints.extend(matches)
            
        except Exception as e:
            print(f"  Warning: Could not scan {file_path}: {e}")
        
        return endpoints
    
    async def _test_endpoint_functionality(self):
        """Test basic functionality of each endpoint."""
        print("🧪 Testing Endpoint Functionality...")
        
        for endpoint in self.validation_results["discovered_endpoints"]:
            test_result = {
                "endpoint": endpoint,
                "methods_supported": [],
                "response_codes": {},
                "response_times": {},
                "functional": False,
                "error": None
            }
            
            try:
                # Test common HTTP methods
                for method in ["GET", "POST", "PUT", "DELETE", "OPTIONS"]:
                    await self._test_endpoint_method(endpoint, method, test_result)
                
                # Determine if endpoint is functional
                test_result["functional"] = len(test_result["methods_supported"]) > 0
                
                if test_result["functional"]:
                    self.validation_results["summary"]["functional_endpoints"] += 1
                
            except Exception as e:
                test_result["error"] = str(e)
            
            self.validation_results["endpoint_tests"][endpoint] = test_result
            
            status = "✅" if test_result["functional"] else "❌"
            methods = ", ".join(test_result["methods_supported"])
            print(f"  {status} {endpoint} [{methods}]")
    
    async def _test_endpoint_method(self, endpoint: str, method: str, test_result: Dict):
        """Test a specific HTTP method on an endpoint."""
        try:
            start_time = time.time()
            
            if method == "GET":
                response = self.session.get(f"{self.base_url}{endpoint}")
            elif method == "POST":
                response = self.session.post(f"{self.base_url}{endpoint}", json={})
            elif method == "PUT":
                response = self.session.put(f"{self.base_url}{endpoint}", json={})
            elif method == "DELETE":
                response = self.session.delete(f"{self.base_url}{endpoint}")
            elif method == "OPTIONS":
                response = self.session.options(f"{self.base_url}{endpoint}")
            else:
                return
            
            response_time = (time.time() - start_time) * 1000
            
            # Record results
            test_result["response_codes"][method] = response.status_code
            test_result["response_times"][method] = response_time
            
            # Consider method supported if it doesn't return 404 or 405
            if response.status_code not in [404, 405]:
                test_result["methods_supported"].append(method)
            
        except Exception as e:
            test_result["response_codes"][method] = "ERROR"
            test_result["error"] = str(e)
    
    async def _test_endpoint_performance(self):
        """Test performance characteristics of endpoints."""
        print("⚡ Testing Endpoint Performance...")
        
        for endpoint in self.validation_results["discovered_endpoints"]:
            perf_result = {
                "endpoint": endpoint,
                "avg_response_time_ms": 0,
                "max_response_time_ms": 0,
                "min_response_time_ms": float('inf'),
                "concurrent_performance": {},
                "meets_thresholds": False,
                "error": None
            }
            
            try:
                # Test sequential performance
                response_times = []
                for i in range(5):  # 5 sequential requests
                    start_time = time.time()
                    try:
                        response = self.session.get(f"{self.base_url}{endpoint}", timeout=10)
                        response_time = (time.time() - start_time) * 1000
                        response_times.append(response_time)
                    except:
                        response_times.append(float('inf'))
                
                if response_times:
                    valid_times = [t for t in response_times if t != float('inf')]
                    if valid_times:
                        perf_result["avg_response_time_ms"] = sum(valid_times) / len(valid_times)
                        perf_result["max_response_time_ms"] = max(valid_times)
                        perf_result["min_response_time_ms"] = min(valid_times)
                
                # Test concurrent performance
                await self._test_concurrent_performance(endpoint, perf_result)
                
                # Check if meets thresholds
                perf_result["meets_thresholds"] = (
                    perf_result["avg_response_time_ms"] < self.performance_thresholds["response_time_ms"]
                )
                
                if perf_result["meets_thresholds"]:
                    self.validation_results["summary"]["performance_compliant"] += 1
                
            except Exception as e:
                perf_result["error"] = str(e)
            
            self.validation_results["performance_tests"][endpoint] = perf_result
            
            status = "✅" if perf_result["meets_thresholds"] else "⚠️"
            avg_time = perf_result["avg_response_time_ms"]
            print(f"  {status} {endpoint} - Avg: {avg_time:.0f}ms")
    
    async def _test_concurrent_performance(self, endpoint: str, perf_result: Dict):
        """Test concurrent request performance."""
        try:
            # Create multiple concurrent requests
            import asyncio
            import aiohttp
            
            async def make_request(session, url):
                start_time = time.time()
                try:
                    async with session.get(url, timeout=10) as response:
                        return (time.time() - start_time) * 1000
                except:
                    return float('inf')
            
            async with aiohttp.ClientSession() as session:
                tasks = [
                    make_request(session, f"{self.base_url}{endpoint}")
                    for _ in range(self.performance_thresholds["concurrent_requests"])
                ]
                
                response_times = await asyncio.gather(*tasks, return_exceptions=True)
                valid_times = [t for t in response_times if isinstance(t, (int, float)) and t != float('inf')]
                
                if valid_times:
                    perf_result["concurrent_performance"] = {
                        "concurrent_avg_ms": sum(valid_times) / len(valid_times),
                        "concurrent_max_ms": max(valid_times),
                        "successful_requests": len(valid_times),
                        "total_requests": len(tasks)
                    }
        
        except Exception as e:
            perf_result["concurrent_performance"] = {"error": str(e)}
    
    async def _test_endpoint_security(self):
        """Test security aspects of endpoints."""
        print("🔒 Testing Endpoint Security...")
        
        for endpoint in self.validation_results["discovered_endpoints"]:
            security_result = {
                "endpoint": endpoint,
                "https_support": False,
                "authentication_required": False,
                "input_validation": False,
                "rate_limiting": False,
                "security_headers": {},
                "injection_resistant": False,
                "compliant": False,
                "error": None
            }
            
            try:
                # Test HTTPS support (if available)
                security_result["https_support"] = await self._test_https_support(endpoint)
                
                # Test authentication requirements
                security_result["authentication_required"] = await self._test_authentication(endpoint)
                
                # Test input validation
                security_result["input_validation"] = await self._test_input_validation(endpoint)
                
                # Test for security headers
                security_result["security_headers"] = await self._test_security_headers(endpoint)
                
                # Test basic injection resistance
                security_result["injection_resistant"] = await self._test_injection_resistance(endpoint)
                
                # Determine overall compliance
                security_score = sum([
                    security_result["input_validation"],
                    security_result["injection_resistant"],
                    len(security_result["security_headers"]) > 0
                ])
                
                security_result["compliant"] = security_score >= 2
                
                if security_result["compliant"]:
                    self.validation_results["summary"]["security_compliant"] += 1
                
            except Exception as e:
                security_result["error"] = str(e)
            
            self.validation_results["security_tests"][endpoint] = security_result
            
            status = "✅" if security_result["compliant"] else "⚠️"
            print(f"  {status} {endpoint} - Security compliant: {security_result['compliant']}")
    
    async def _test_https_support(self, endpoint: str) -> bool:
        """Test if endpoint supports HTTPS."""
        try:
            https_url = self.base_url.replace("http://", "https://")
            response = self.session.get(f"{https_url}{endpoint}", timeout=5)
            return response.status_code < 500
        except:
            return False
    
    async def _test_authentication(self, endpoint: str) -> bool:
        """Test if endpoint requires authentication."""
        try:
            response = self.session.get(f"{self.base_url}{endpoint}")
            return response.status_code in [401, 403]
        except:
            return False
    
    async def _test_input_validation(self, endpoint: str) -> bool:
        """Test if endpoint validates input properly."""
        try:
            # Try sending malformed JSON
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                data="invalid_json",
                headers={"Content-Type": "application/json"}
            )
            # Should return 400 for bad input
            return response.status_code == 400
        except:
            return False
    
    async def _test_security_headers(self, endpoint: str) -> Dict[str, str]:
        """Test for security headers."""
        security_headers = {}
        
        try:
            response = self.session.get(f"{self.base_url}{endpoint}")
            
            # Check for common security headers
            header_checks = [
                "X-Content-Type-Options",
                "X-Frame-Options", 
                "X-XSS-Protection",
                "Strict-Transport-Security",
                "Content-Security-Policy"
            ]
            
            for header in header_checks:
                if header in response.headers:
                    security_headers[header] = response.headers[header]
        
        except:
            pass
        
        return security_headers
    
    async def _test_injection_resistance(self, endpoint: str) -> bool:
        """Test basic injection resistance."""
        try:
            # Test SQL injection patterns
            injection_payloads = [
                "'; DROP TABLE users; --",
                "<script>alert('xss')</script>",
                "../../../etc/passwd"
            ]
            
            for payload in injection_payloads:
                response = self.session.get(
                    f"{self.base_url}{endpoint}",
                    params={"test": payload}
                )
                
                # Should not reflect the payload or cause errors
                if payload in response.text or response.status_code == 500:
                    return False
            
            return True
        except:
            return False
    
    async def _test_documentation_compliance(self):
        """Test API documentation compliance."""
        print("📚 Testing Documentation Compliance...")
        
        for endpoint in self.validation_results["discovered_endpoints"]:
            doc_result = {
                "endpoint": endpoint,
                "has_docstring": False,
                "has_examples": False,
                "has_error_codes": False,
                "has_parameter_docs": False,
                "compliant": False,
                "error": None
            }
            
            try:
                # Look for endpoint documentation in source code
                doc_info = await self._find_endpoint_documentation(endpoint)
                doc_result.update(doc_info)
                
                # Determine compliance
                doc_score = sum([
                    doc_result["has_docstring"],
                    doc_result["has_examples"],
                    doc_result["has_error_codes"],
                    doc_result["has_parameter_docs"]
                ])
                
                doc_result["compliant"] = doc_score >= 2
                
                if doc_result["compliant"]:
                    self.validation_results["summary"]["documented_endpoints"] += 1
                
            except Exception as e:
                doc_result["error"] = str(e)
            
            self.validation_results["documentation_tests"][endpoint] = doc_result
            
            status = "✅" if doc_result["compliant"] else "⚠️"
            print(f"  {status} {endpoint} - Documented: {doc_result['compliant']}")
    
    async def _find_endpoint_documentation(self, endpoint: str) -> Dict[str, bool]:
        """Find documentation for a specific endpoint."""
        doc_info = {
            "has_docstring": False,
            "has_examples": False,
            "has_error_codes": False,
            "has_parameter_docs": False
        }
        
        # Look for the endpoint in source files
        api_dirs = ["dashboard/api", "testmaster/dashboard/api", "api"]
        
        for api_dir in api_dirs:
            api_path = Path(api_dir)
            if api_path.exists():
                for py_file in api_path.rglob("*.py"):
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Check if this file contains the endpoint
                        if endpoint in content:
                            # Look for docstrings
                            if '"""' in content and ("GET" in content or "POST" in content):
                                doc_info["has_docstring"] = True
                            
                            # Look for examples
                            if "example" in content.lower() or "sample" in content.lower():
                                doc_info["has_examples"] = True
                            
                            # Look for error code documentation
                            if any(code in content for code in ["400", "401", "404", "500"]):
                                doc_info["has_error_codes"] = True
                            
                            # Look for parameter documentation
                            if "param" in content.lower() or "arg" in content.lower():
                                doc_info["has_parameter_docs"] = True
                    
                    except:
                        continue
        
        return doc_info
    
    async def _test_rest_compliance(self):
        """Test REST API compliance."""
        print("🌐 Testing REST API Compliance...")
        
        compliance_result = {
            "proper_http_methods": False,
            "consistent_url_structure": False,
            "proper_status_codes": False,
            "json_responses": False,
            "resource_naming": False,
            "overall_compliant": False
        }
        
        try:
            # Test proper HTTP method usage
            compliance_result["proper_http_methods"] = self._check_http_method_usage()
            
            # Test URL structure consistency
            compliance_result["consistent_url_structure"] = self._check_url_structure()
            
            # Test status code usage
            compliance_result["proper_status_codes"] = await self._check_status_codes()
            
            # Test JSON response format
            compliance_result["json_responses"] = await self._check_json_responses()
            
            # Test resource naming conventions
            compliance_result["resource_naming"] = self._check_resource_naming()
            
            # Overall compliance
            compliance_score = sum(compliance_result.values())
            compliance_result["overall_compliant"] = compliance_score >= 3
            
        except Exception as e:
            compliance_result["error"] = str(e)
        
        self.validation_results["compliance_tests"] = compliance_result
        
        status = "✅" if compliance_result["overall_compliant"] else "⚠️"
        print(f"  {status} REST compliance: {compliance_result['overall_compliant']}")
    
    def _check_http_method_usage(self) -> bool:
        """Check if HTTP methods are used appropriately."""
        # Simple heuristic: endpoints should support GET at minimum
        endpoints_with_get = 0
        total_endpoints = len(self.validation_results["endpoint_tests"])
        
        for endpoint_test in self.validation_results["endpoint_tests"].values():
            if "GET" in endpoint_test.get("methods_supported", []):
                endpoints_with_get += 1
        
        return endpoints_with_get / total_endpoints > 0.7 if total_endpoints > 0 else False
    
    def _check_url_structure(self) -> bool:
        """Check URL structure consistency."""
        endpoints = self.validation_results["discovered_endpoints"]
        
        # Check if most endpoints follow /api/{resource} pattern
        api_pattern_count = sum(1 for ep in endpoints if ep.startswith("/api/"))
        return api_pattern_count / len(endpoints) > 0.8 if endpoints else False
    
    async def _check_status_codes(self) -> bool:
        """Check proper status code usage."""
        # Test a few endpoints for proper status codes
        proper_codes = 0
        total_tests = 0
        
        for endpoint in self.validation_results["discovered_endpoints"][:5]:  # Test first 5
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                total_tests += 1
                
                # Check for proper codes (2xx for success, 4xx for client error, etc.)
                if 200 <= response.status_code < 600:
                    proper_codes += 1
            except:
                continue
        
        return proper_codes / total_tests > 0.8 if total_tests > 0 else False
    
    async def _check_json_responses(self) -> bool:
        """Check if endpoints return proper JSON."""
        json_endpoints = 0
        total_endpoints = 0
        
        for endpoint in self.validation_results["discovered_endpoints"][:5]:  # Test first 5
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                total_endpoints += 1
                
                if response.headers.get("content-type", "").startswith("application/json"):
                    json_endpoints += 1
            except:
                continue
        
        return json_endpoints / total_endpoints > 0.7 if total_endpoints > 0 else False
    
    def _check_resource_naming(self) -> bool:
        """Check resource naming conventions."""
        endpoints = self.validation_results["discovered_endpoints"]
        
        # Check for RESTful naming (nouns, not verbs)
        good_names = 0
        for endpoint in endpoints:
            # Extract resource name from endpoint
            parts = endpoint.split("/")
            if len(parts) >= 3:  # /api/resource
                resource = parts[2]
                # Simple check: avoid common verbs
                verbs = ["get", "post", "create", "delete", "update", "fetch"]
                if not any(verb in resource.lower() for verb in verbs):
                    good_names += 1
        
        return good_names / len(endpoints) > 0.6 if endpoints else False
    
    def _generate_validation_report(self):
        """Generate comprehensive validation report."""
        print("\n" + "=" * 70)
        print("📋 API ENDPOINT VALIDATION REPORT")
        print("=" * 70)
        
        # Calculate overall score
        summary = self.validation_results["summary"]
        total_endpoints = summary["total_endpoints"]
        
        if total_endpoints > 0:
            scores = [
                summary["functional_endpoints"] / total_endpoints,
                summary["performance_compliant"] / total_endpoints,
                summary["security_compliant"] / total_endpoints,
                summary["documented_endpoints"] / total_endpoints
            ]
            overall_score = sum(scores) / len(scores)
            summary["overall_score"] = overall_score
        else:
            overall_score = 0.0
            summary["overall_score"] = 0.0
        
        # Print summary
        print(f"Overall Score: {overall_score:.2%}")
        print(f"Total Endpoints: {total_endpoints}")
        print(f"Functional: {summary['functional_endpoints']}/{total_endpoints}")
        print(f"Performance Compliant: {summary['performance_compliant']}/{total_endpoints}")
        print(f"Security Compliant: {summary['security_compliant']}/{total_endpoints}")
        print(f"Documented: {summary['documented_endpoints']}/{total_endpoints}")
        
        # REST compliance
        compliance = self.validation_results.get("compliance_tests", {})
        rest_compliant = compliance.get("overall_compliant", False)
        print(f"REST Compliant: {'✅' if rest_compliant else '❌'}")
        
        print("=" * 70)
        
        # Save detailed report
        with open("api_validation_report.json", "w") as f:
            json.dump(self.validation_results, f, indent=2)
        
        print("📄 Detailed report saved to: api_validation_report.json")


async def main():
    """Main entry point for API validation."""
    validator = APIEndpointValidator()
    results = await validator.run_comprehensive_validation()
    
    # Return appropriate exit code based on overall score
    overall_score = results["summary"]["overall_score"]
    if overall_score >= 0.8:
        return 0  # Excellent
    elif overall_score >= 0.6:
        return 1  # Good
    else:
        return 2  # Needs improvement


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)