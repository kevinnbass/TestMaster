"""
Test API Validation Framework
Agent D - Hour 2: API Documentation & Validation Systems
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add TestMaster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TestMaster'))

from TestMaster.core.intelligence.documentation.api_validation_framework import (
    APIValidationFramework,
    APIEndpoint,
    APIEndpointType,
    validate_intelligence_apis,
    generate_openapi_documentation
)

async def test_api_validation():
    """Test the API validation framework."""
    
    print("=" * 80)
    print("Agent D - Hour 2: API Documentation & Validation Systems")
    print("Testing API Validation Framework")
    print("=" * 80)
    
    # Initialize framework
    framework = APIValidationFramework(base_url="http://localhost:5000")
    
    # Discover endpoints
    print("\n1. Discovering Intelligence API Endpoints...")
    endpoints = framework.discover_intelligence_endpoints()
    print(f"   [OK] Discovered {len(endpoints)} endpoints")
    
    for endpoint in endpoints[:5]:  # Show first 5
        print(f"      - {endpoint.path} [{', '.join(endpoint.methods)}]")
    
    # Generate OpenAPI specification
    print("\n2. Generating OpenAPI Specification...")
    openapi_spec = framework.openapi_generator.generate_openapi_spec(endpoints)
    print(f"   [OK] OpenAPI spec generated with {len(openapi_spec['paths'])} paths")
    
    # Export OpenAPI specification
    output_path = "TestMaster/docs/api/openapi_specification.yaml"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    yaml_path = framework.export_openapi_yaml(openapi_spec, output_path)
    print(f"   [OK] OpenAPI spec exported to: {yaml_path}")
    
    # Validate endpoints
    print("\n3. Validating API Endpoints...")
    print("   Note: Some endpoints may fail if server is not running")
    
    validation_report = await framework.validator.validate_all_endpoints(endpoints)
    
    print(f"\n   Validation Summary:")
    print(f"   - Total Endpoints: {validation_report.total_endpoints}")
    print(f"   - Healthy: {validation_report.healthy_endpoints}")
    print(f"   - Warnings: {validation_report.warning_endpoints}")
    print(f"   - Errors: {validation_report.error_endpoints}")
    print(f"   - Overall Health Score: {validation_report.overall_health_score:.2%}")
    print(f"   - Average Response Time: {validation_report.average_response_time:.3f}s")
    
    # Show recommendations
    if validation_report.recommendations:
        print(f"\n   Recommendations:")
        for rec in validation_report.recommendations:
            print(f"   - {rec}")
    
    # Generate complete documentation
    print("\n4. Generating Complete API Documentation...")
    complete_docs = await framework.generate_complete_api_documentation()
    
    # Export validation report
    report_path = "TestMaster/docs/api/validation_report.json"
    json_path = framework.export_validation_report(complete_docs, report_path)
    print(f"   [OK] Validation report exported to: {json_path}")
    
    # Create HTML documentation
    print("\n5. Creating HTML API Documentation...")
    html_content = generate_html_documentation(complete_docs)
    html_path = "TestMaster/docs/api/api_documentation.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"   [OK] HTML documentation created: {html_path}")
    
    print("\n" + "=" * 80)
    print("API Validation Framework Test Complete!")
    print("=" * 80)
    
    return complete_docs

def generate_html_documentation(docs_data):
    """Generate HTML documentation from API data."""
    
    openapi = docs_data.get('openapi_specification', {})
    validation = docs_data.get('validation_report', {})
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TestMaster Intelligence Hub API Documentation</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #764ba2;
            margin-top: 30px;
        }}
        .health-score {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }}
        .health-good {{ background: #10b981; }}
        .health-warning {{ background: #f59e0b; }}
        .health-error {{ background: #ef4444; }}
        .endpoint {{
            background: #f3f4f6;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .method {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-weight: bold;
            color: white;
            font-size: 12px;
            margin-right: 10px;
        }}
        .method-get {{ background: #10b981; }}
        .method-post {{ background: #3b82f6; }}
        .method-put {{ background: #f59e0b; }}
        .method-delete {{ background: #ef4444; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .stat-label {{
            opacity: 0.9;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}
        th {{
            background: #f3f4f6;
            font-weight: 600;
        }}
        .status-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }}
        .status-healthy {{ background: #d1fae5; color: #065f46; }}
        .status-warning {{ background: #fed7aa; color: #92400e; }}
        .status-error {{ background: #fee2e2; color: #991b1b; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>TestMaster Intelligence Hub API Documentation</h1>
        <p><strong>Generated by Agent D</strong> - Documentation & Validation Excellence</p>
        <p>Timestamp: {docs_data.get('generation_timestamp', datetime.now().isoformat())}</p>
        
        <h2>API Health Summary</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{validation['summary']['total_endpoints']}</div>
                <div class="stat-label">Total Endpoints</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{validation['summary']['healthy_endpoints']}</div>
                <div class="stat-label">Healthy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{validation['summary']['warning_endpoints']}</div>
                <div class="stat-label">Warnings</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{validation['summary']['error_endpoints']}</div>
                <div class="stat-label">Errors</div>
            </div>
        </div>
        
        <p>Overall Health Score: <span class="health-score {'health-good' if validation['summary']['overall_health_score'] > 0.8 else 'health-warning' if validation['summary']['overall_health_score'] > 0.5 else 'health-error'}">{validation['summary']['overall_health_score']:.1%}</span></p>
        <p>Average Response Time: <strong>{validation['summary']['average_response_time']:.3f}s</strong></p>
        
        <h2>API Endpoints</h2>
        <table>
            <thead>
                <tr>
                    <th>Endpoint</th>
                    <th>Method</th>
                    <th>Status</th>
                    <th>Response Time</th>
                    <th>Health Score</th>
                </tr>
            </thead>
            <tbody>"""
    
    for result in validation.get('detailed_results', []):
        status = result['status']
        status_class = 'healthy' if status == 'healthy' else 'warning' if status == 'warning' else 'error'
        
        html += f"""
                <tr>
                    <td><code>{result['endpoint']}</code></td>
                    <td><span class="method method-get">GET</span></td>
                    <td><span class="status-badge status-{status_class}">{status.upper()}</span></td>
                    <td>{result['response_time']:.3f}s</td>
                    <td>{result['health_score']:.1%}</td>
                </tr>"""
    
    html += """
            </tbody>
        </table>
        
        <h2>Recommendations</h2>
        <ul>"""
    
    for rec in validation.get('recommendations', []):
        html += f"<li>{rec}</li>"
    
    html += """
        </ul>
        
        <h2>Available API Endpoints</h2>"""
    
    for path, methods in openapi.get('paths', {}).items():
        html += f'<div class="endpoint">'
        html += f'<strong>{path}</strong><br>'
        for method, details in methods.items():
            method_class = f"method-{method}"
            html += f'<span class="method {method_class}">{method.upper()}</span>'
            html += f'{details.get("summary", "")}<br>'
        html += '</div>'
    
    html += """
        <h2>API Information</h2>
        <ul>
            <li><strong>Title:</strong> """ + openapi.get('info', {}).get('title', 'TestMaster API') + """</li>
            <li><strong>Version:</strong> """ + openapi.get('info', {}).get('version', '1.0.0') + """</li>
            <li><strong>Description:</strong> """ + openapi.get('info', {}).get('description', '') + """</li>
        </ul>
        
        <hr style="margin-top: 50px;">
        <p style="text-align: center; color: #6b7280;">
            <em>Agent D - Hour 2: API Documentation & Validation Systems Complete</em><br>
            <em>Part of the 24-Hour Meta-Recursive Documentation Excellence Mission</em>
        </p>
    </div>
</body>
</html>"""
    
    return html

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_api_validation())