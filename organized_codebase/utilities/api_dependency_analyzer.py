#!/usr/bin/env python3
"""
Agent C Hour 16-18: API Dependency Mapping
Analyze REST/GraphQL/gRPC patterns and external service dependencies
"""

import os
import ast
import json
import re
from collections import defaultdict, Counter
import time

def analyze_api_dependencies():
    """Analyze API dependencies, REST/GraphQL/gRPC patterns, and external service usage"""
    start_time = time.time()
    
    api_patterns = {
        'rest_apis': defaultdict(list),
        'graphql_apis': defaultdict(list),
        'grpc_apis': defaultdict(list),
        'external_services': defaultdict(list),
        'api_clients': defaultdict(list),
        'http_methods': Counter(),
        'api_versions': defaultdict(list),
        'authentication_patterns': defaultdict(list)
    }
    
    files_analyzed = 0
    api_calls_found = 0
    external_services = set()
    
    # Common API patterns to search for
    rest_patterns = [
        r'requests\.(get|post|put|delete|patch)',
        r'http\.(get|post|put|delete|patch)',
        r'urllib\.request',
        r'aiohttp\.',
        r'httpx\.',
        r'fetch\(',
        r'axios\.'
    ]
    
    graphql_patterns = [
        r'graphql',
        r'apollo',
        r'relay',
        r'subscription',
        r'mutation',
        r'query.*\{',
        r'gql`'
    ]
    
    grpc_patterns = [
        r'grpc\.',
        r'\.proto',
        r'protobuf',
        r'_pb2',
        r'servicer',
        r'stub'
    ]
    
    # External service patterns
    service_patterns = {
        'openai': r'openai\.|gpt-|api\.openai',
        'anthropic': r'anthropic\.|claude-',
        'aws': r'boto3|aws\.|s3\.|ec2\.',
        'gcp': r'google\.cloud|gcp\.',
        'azure': r'azure\.|msrest',
        'github': r'github\.com|api\.github',
        'stripe': r'stripe\.',
        'twilio': r'twilio\.',
        'redis': r'redis\.|Redis\(',
        'mongodb': r'mongo|pymongo',
        'postgresql': r'psycopg|postgresql',
        'mysql': r'mysql|pymysql',
        'elasticsearch': r'elasticsearch\.',
        'docker': r'docker\.',
        'kubernetes': r'kubernetes\.|k8s'
    }
    
    # Walk through Python files
    for root, dirs, files in os.walk('.'):
        if any(skip in root for skip in ['__pycache__', '.git', 'node_modules']):
            continue
            
        for file in files:
            if file.endswith(('.py', '.js', '.ts', '.jsx', '.tsx')):
                file_path = os.path.join(root, file)
                files_analyzed += 1
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Check for REST API patterns
                    for pattern in rest_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            api_calls_found += 1
                            method = match.group(1) if match.groups() else 'unknown'
                            api_patterns['rest_apis'][method].append(file_path)
                            api_patterns['http_methods'][method.upper()] += 1
                    
                    # Check for GraphQL patterns
                    for pattern in graphql_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            api_patterns['graphql_apis']['query'].append(file_path)
                    
                    # Check for gRPC patterns
                    for pattern in grpc_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            api_patterns['grpc_apis']['service'].append(file_path)
                    
                    # Check for external services
                    for service, pattern in service_patterns.items():
                        if re.search(pattern, content, re.IGNORECASE):
                            api_patterns['external_services'][service].append(file_path)
                            external_services.add(service)
                    
                    # Look for API versioning patterns
                    version_patterns = [r'v\d+', r'version=', r'api_version', r'/v\d+/']
                    for pattern in version_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            api_patterns['api_versions'][match.group()].append(file_path)
                    
                    # Look for authentication patterns
                    auth_patterns = {
                        'bearer_token': r'Bearer\s+',
                        'api_key': r'api[_-]?key',
                        'oauth': r'oauth|OAuth',
                        'jwt': r'jwt|JWT',
                        'basic_auth': r'Basic\s+',
                        'session': r'session[_-]?id|sessionid'
                    }
                    
                    for auth_type, pattern in auth_patterns.items():
                        if re.search(pattern, content, re.IGNORECASE):
                            api_patterns['authentication_patterns'][auth_type].append(file_path)
                    
                except Exception as e:
                    continue
    
    analysis_time = time.time() - start_time
    
    # Consolidate results
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'analysis_metrics': {
            'files_analyzed': files_analyzed,
            'api_calls_found': api_calls_found,
            'external_services_count': len(external_services),
            'analysis_duration_seconds': round(analysis_time, 2)
        },
        'api_architecture': {
            'rest_api_usage': len(api_patterns['rest_apis']),
            'graphql_usage': len(api_patterns['graphql_apis']),
            'grpc_usage': len(api_patterns['grpc_apis']),
            'external_service_integrations': len(external_services)
        },
        'http_methods_distribution': dict(api_patterns['http_methods']),
        'external_services': {service: len(files) for service, files in api_patterns['external_services'].items()},
        'authentication_patterns': {auth: len(files) for auth, files in api_patterns['authentication_patterns'].items()},
        'api_versioning': {version: len(files) for version, files in api_patterns['api_versions'].items()},
        'detailed_patterns': {
            'rest_apis': {method: files[:10] for method, files in api_patterns['rest_apis'].items()},
            'external_services': {service: files[:5] for service, files in api_patterns['external_services'].items()},
            'auth_implementations': {auth: files[:5] for auth, files in api_patterns['authentication_patterns'].items()}
        },
        'integration_analysis': {
            'most_used_services': sorted(external_services),
            'api_complexity_score': api_calls_found / max(files_analyzed, 1),
            'service_diversity': len(external_services),
            'authentication_diversity': len(api_patterns['authentication_patterns'])
        }
    }
    
    # Save results
    with open('api_dependency_hour16.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    # Run analysis
    results = analyze_api_dependencies()
    print('API DEPENDENCY ANALYSIS COMPLETE')
    print(f'Files Analyzed: {results["analysis_metrics"]["files_analyzed"]}')
    print(f'API Calls Found: {results["analysis_metrics"]["api_calls_found"]}')
    print(f'External Services: {results["analysis_metrics"]["external_services_count"]}')
    print(f'Analysis Time: {results["analysis_metrics"]["analysis_duration_seconds"]}s')
    print('Results saved to: api_dependency_hour16.json')