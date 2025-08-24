"""
Interactive API Documentation

Live API explorer with testing capability and dynamic documentation.
"""

import json
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from .api_spec_builder import APISpecBuilder, APIEndpoint

logger = logging.getLogger(__name__)


@dataclass
class APITest:
    """Represents an API test."""
    endpoint: str
    method: str
    parameters: Dict[str, Any]
    expected_status: int
    response_time: float
    success: bool
    

@dataclass
class InteractiveExample:
    """Interactive code example."""
    language: str
    code: str
    description: str
    runnable: bool
    

class InteractiveDocumentation:
    """
    Interactive API documentation with live testing capabilities.
    Provides dynamic endpoint documentation and code examples.
    """
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Initialize interactive documentation.
        
        Args:
            base_url: Base URL for API testing
        """
        self.base_url = base_url.rstrip('/')
        self.api_builder = APISpecBuilder()
        self.test_history = []
        self.endpoint_status = {}
        logger.info(f"Interactive Documentation initialized for {base_url}")
        
    def generate_interactive_spec(self, project_path: str) -> Dict[str, Any]:
        """
        Generate interactive API specification.
        
        Args:
            project_path: Path to project
            
        Returns:
            Interactive API spec
        """
        # Discover endpoints
        endpoints = self.api_builder.scan_directory(project_path)
        
        # Build base spec
        spec = self.api_builder.build_openapi_spec(
            title="Interactive API Documentation",
            version="1.0.0",
            base_url=self.base_url,
            description="Live, testable API documentation"
        )
        
        # Enhance with interactive features
        interactive_spec = {
            **spec,
            'x-interactive': {
                'live_testing': True,
                'code_examples': True,
                'response_examples': True,
                'endpoint_monitoring': True
            },
            'x-endpoints': [self._enhance_endpoint(ep) for ep in endpoints]
        }
        
        return interactive_spec
        
    def test_endpoint(self, 
                     endpoint: str,
                     method: str,
                     parameters: Optional[Dict] = None,
                     headers: Optional[Dict] = None,
                     body: Optional[Dict] = None) -> APITest:
        """
        Test an API endpoint live.
        
        Args:
            endpoint: Endpoint path
            method: HTTP method
            parameters: Query parameters
            headers: Request headers
            body: Request body
            
        Returns:
            Test result
        """
        url = f"{self.base_url}{endpoint}"
        test_params = parameters or {}
        test_headers = headers or {}
        
        start_time = datetime.now()
        success = False
        status_code = 0
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=test_params, headers=test_headers, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, json=body, params=test_params, headers=test_headers, timeout=10)
            elif method.upper() == "PUT":
                response = requests.put(url, json=body, params=test_params, headers=test_headers, timeout=10)
            elif method.upper() == "DELETE":
                response = requests.delete(url, params=test_params, headers=test_headers, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")
                
            status_code = response.status_code
            success = 200 <= status_code < 400
            
        except Exception as e:
            logger.error(f"Error testing endpoint {endpoint}: {e}")
            
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds() * 1000
        
        test_result = APITest(
            endpoint=endpoint,
            method=method.upper(),
            parameters=test_params,
            expected_status=200,
            response_time=response_time,
            success=success
        )
        
        self.test_history.append(test_result)
        self._update_endpoint_status(endpoint, success, response_time)
        
        return test_result
        
    def generate_code_examples(self, endpoint: APIEndpoint) -> List[InteractiveExample]:
        """
        Generate interactive code examples for endpoint.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            List of code examples
        """
        examples = []
        
        # Python requests example
        python_code = self._generate_python_example(endpoint)
        examples.append(InteractiveExample(
            language="python",
            code=python_code,
            description=f"Python example for {endpoint.method} {endpoint.path}",
            runnable=True
        ))
        
        # JavaScript fetch example
        js_code = self._generate_javascript_example(endpoint)
        examples.append(InteractiveExample(
            language="javascript",
            code=js_code,
            description=f"JavaScript example for {endpoint.method} {endpoint.path}",
            runnable=True
        ))
        
        # cURL example
        curl_code = self._generate_curl_example(endpoint)
        examples.append(InteractiveExample(
            language="bash",
            code=curl_code,
            description=f"cURL example for {endpoint.method} {endpoint.path}",
            runnable=False
        ))
        
        return examples
        
    def get_endpoint_health(self) -> Dict[str, Any]:
        """
        Get health status of all endpoints.
        
        Returns:
            Endpoint health status
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'total_endpoints': len(self.endpoint_status),
            'healthy_endpoints': len([s for s in self.endpoint_status.values() if s['healthy']]),
            'average_response_time': self._calculate_average_response_time(),
            'endpoint_details': self.endpoint_status
        }
        
    def run_health_check(self, endpoints: List[APIEndpoint]) -> Dict[str, Any]:
        """
        Run health check on all endpoints.
        
        Args:
            endpoints: List of endpoints to check
            
        Returns:
            Health check results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'endpoints_tested': 0,
            'healthy_endpoints': 0,
            'failed_endpoints': [],
            'average_response_time': 0
        }
        
        response_times = []
        
        for endpoint in endpoints:
            if endpoint.method.upper() == "GET":  # Only test GET endpoints for health
                test_result = self.test_endpoint(
                    endpoint=endpoint.path,
                    method=endpoint.method
                )
                
                results['endpoints_tested'] += 1
                
                if test_result.success:
                    results['healthy_endpoints'] += 1
                    response_times.append(test_result.response_time)
                else:
                    results['failed_endpoints'].append({
                        'endpoint': endpoint.path,
                        'method': endpoint.method,
                        'response_time': test_result.response_time
                    })
                    
        if response_times:
            results['average_response_time'] = sum(response_times) / len(response_times)
            
        return results
        
    def generate_interactive_html(self, spec: Dict[str, Any]) -> str:
        """
        Generate interactive HTML documentation.
        
        Args:
            spec: API specification
            
        Returns:
            HTML documentation
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{spec.get('info', {}).get('title', 'API Documentation')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .endpoint {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; }}
                .method {{ font-weight: bold; color: #007bff; }}
                .test-button {{ background: #28a745; color: white; border: none; padding: 8px 16px; cursor: pointer; }}
                .test-result {{ margin-top: 10px; padding: 10px; background: #f8f9fa; }}
                .success {{ border-left: 4px solid #28a745; }}
                .error {{ border-left: 4px solid #dc3545; }}
                .code-example {{ background: #f4f4f4; padding: 10px; font-family: monospace; }}
            </style>
        </head>
        <body>
            <h1>{spec.get('info', {}).get('title', 'API Documentation')}</h1>
            <p>{spec.get('info', {}).get('description', '')}</p>
            
            <div id="endpoints">
                <!-- Endpoints will be populated by JavaScript -->
            </div>
            
            <script>
                const spec = {json.dumps(spec, indent=2)};
                
                function testEndpoint(path, method) {{
                    // Test endpoint logic would go here
                    console.log('Testing', method, path);
                }}
                
                function generateEndpoints() {{
                    const container = document.getElementById('endpoints');
                    
                    Object.keys(spec.paths || {{}}).forEach(path => {{
                        Object.keys(spec.paths[path]).forEach(method => {{
                            const endpoint = spec.paths[path][method];
                            
                            const div = document.createElement('div');
                            div.className = 'endpoint';
                            div.textContent = `
                                <div class="method">${{method.toUpperCase()}} ${{path}}</div>
                                <p>${{endpoint.summary || 'No description'}}</p>
                                <button class="test-button" onclick="testEndpoint('${{path}}', '${{method}}')">
                                    Test Endpoint
                                </button>
                                <div class="test-result" id="result-${{path}}-${{method}}" style="display: none;"></div>
                            `;
                            
                            container.appendChild(div);
                        }});
                    }});
                }}
                
                generateEndpoints();
            </script>
        </body>
        </html>
        """
        
        return html
        
    # Private methods
    def _enhance_endpoint(self, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Enhance endpoint with interactive features."""
        return {
            'path': endpoint.path,
            'method': endpoint.method,
            'summary': endpoint.summary,
            'description': endpoint.description,
            'parameters': endpoint.parameters,
            'responses': endpoint.responses,
            'x-code-examples': [asdict(ex) for ex in self.generate_code_examples(endpoint)],
            'x-testable': True,
            'x-health-status': self.endpoint_status.get(endpoint.path, {'healthy': True})
        }
        
    def _generate_python_example(self, endpoint: APIEndpoint) -> str:
        """Generate Python code example."""
        params_str = ""
        if endpoint.parameters:
            params_str = ", params=" + str({p['name']: 'value' for p in endpoint.parameters})
            
        return f"""import requests

response = requests.{endpoint.method.lower()}(
    '{self.base_url}{endpoint.path}'{params_str}
)

print(f"Status: {{response.status_code}}")
print(f"Response: {{response.json()}}")"""
        
    def _generate_javascript_example(self, endpoint: APIEndpoint) -> str:
        """Generate JavaScript code example."""
        return f"""fetch('{self.base_url}{endpoint.path}', {{
    method: '{endpoint.method.upper()}'
}})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));"""
        
    def _generate_curl_example(self, endpoint: APIEndpoint) -> str:
        """Generate cURL code example."""
        return f"""curl -X {endpoint.method.upper()} \\
  '{self.base_url}{endpoint.path}' \\
  -H 'Content-Type: application/json'"""
        
    def _update_endpoint_status(self, endpoint: str, success: bool, response_time: float) -> None:
        """Update endpoint status."""
        if endpoint not in self.endpoint_status:
            self.endpoint_status[endpoint] = {
                'healthy': True,
                'response_times': [],
                'last_test': None,
                'success_rate': 1.0
            }
            
        status = self.endpoint_status[endpoint]
        status['healthy'] = success
        status['response_times'].append(response_time)
        status['last_test'] = datetime.now().isoformat()
        
        # Keep only last 10 response times
        if len(status['response_times']) > 10:
            status['response_times'] = status['response_times'][-10:]
            
        # Calculate success rate from test history
        endpoint_tests = [t for t in self.test_history if t.endpoint == endpoint]
        if endpoint_tests:
            successes = sum(1 for t in endpoint_tests if t.success)
            status['success_rate'] = successes / len(endpoint_tests)
            
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time across all endpoints."""
        all_times = []
        for status in self.endpoint_status.values():
            all_times.extend(status.get('response_times', []))
            
        return sum(all_times) / len(all_times) if all_times else 0