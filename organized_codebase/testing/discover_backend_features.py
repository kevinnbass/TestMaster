#!/usr/bin/env python3
"""
Backend Feature Discovery for Frontend Integration
==================================================

Scans the entire TestMaster codebase to discover ALL backend capabilities
that can be exposed for frontend visualization.

Author: TestMaster Team
"""

import os
import ast
import json
from pathlib import Path
from collections import defaultdict

def discover_backend_features():
    """Discover all backend features for frontend integration."""
    print("=" * 80)
    print("BACKEND FEATURE DISCOVERY")
    print("=" * 80)
    print()
    
    testmaster_root = Path(__file__).parent.parent
    features = {
        'apis': [],
        'intelligence_agents': [],
        'test_generators': [],
        'security_features': [],
        'monitoring_systems': [],
        'data_processors': [],
        'visualization_candidates': []
    }
    
    # 1. Scan for API endpoints
    print("1. Discovering API Endpoints...")
    api_path = testmaster_root / 'testmaster'
    dashboard_api = Path(__file__).parent / 'api'
    
    for api_dir in [api_path, dashboard_api]:
        if api_dir.exists():
            for py_file in api_dir.rglob('*.py'):
                if py_file.name != '__init__.py':
                    try:
                        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if '@' in content and 'route' in content:
                                features['apis'].append({
                                    'file': str(py_file.relative_to(testmaster_root)),
                                    'type': 'api_endpoint',
                                    'category': py_file.stem
                                })
                    except:
                        pass
    
    # 2. Scan for Intelligence Agents
    print("2. Discovering Intelligence Agents...")
    intelligence_paths = [
        testmaster_root / 'testmaster' / 'intelligence',
        testmaster_root / 'testmaster' / 'agent_qa'
    ]
    
    for intel_path in intelligence_paths:
        if intel_path.exists():
            for py_file in intel_path.rglob('*.py'):
                if py_file.name != '__init__.py':
                    try:
                        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if any(word in content.lower() for word in ['agent', 'intelligence', 'consensus', 'bridge']):
                                tree = ast.parse(content)
                                for node in ast.walk(tree):
                                    if isinstance(node, ast.ClassDef):
                                        features['intelligence_agents'].append({
                                            'name': node.name,
                                            'file': str(py_file.relative_to(testmaster_root)),
                                            'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                                        })
                    except:
                        pass
    
    # 3. Scan for Test Generators
    print("3. Discovering Test Generators...")
    for py_file in testmaster_root.rglob('*test*.py'):
        if 'generator' in py_file.name.lower() or 'builder' in py_file.name.lower():
            features['test_generators'].append({
                'file': str(py_file.relative_to(testmaster_root)),
                'type': 'test_generator'
            })
    
    # 4. Scan for Security Features
    print("4. Discovering Security Features...")
    security_keywords = ['security', 'vulnerability', 'owasp', 'compliance', 'scanner']
    for py_file in testmaster_root.rglob('*.py'):
        if any(keyword in py_file.name.lower() for keyword in security_keywords):
            features['security_features'].append({
                'file': str(py_file.relative_to(testmaster_root)),
                'type': 'security_feature'
            })
    
    # 5. Scan for Monitoring Systems
    print("5. Discovering Monitoring Systems...")
    monitoring_keywords = ['monitor', 'metrics', 'telemetry', 'analytics', 'performance']
    for py_file in testmaster_root.rglob('*.py'):
        if any(keyword in py_file.name.lower() for keyword in monitoring_keywords):
            features['monitoring_systems'].append({
                'file': str(py_file.relative_to(testmaster_root)),
                'type': 'monitoring_system'
            })
    
    # 6. Identify Visualization Candidates
    print("6. Identifying Visualization Candidates...")
    viz_indicators = ['graph', 'chart', 'plot', 'visualization', 'heatmap', 'dashboard', 'dag', 'flow']
    for py_file in testmaster_root.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
                if any(indicator in content for indicator in viz_indicators):
                    features['visualization_candidates'].append({
                        'file': str(py_file.relative_to(testmaster_root)),
                        'indicators': [ind for ind in viz_indicators if ind in content]
                    })
        except:
            pass
    
    # Print discovery results
    print("\n" + "=" * 80)
    print("DISCOVERY RESULTS")
    print("=" * 80)
    
    print(f"\nAPI Endpoints Found: {len(features['apis'])}")
    for api in features['apis'][:10]:  # Show first 10
        print(f"  - {api['category']}: {api['file']}")
    if len(features['apis']) > 10:
        print(f"  ... and {len(features['apis']) - 10} more")
    
    print(f"\nIntelligence Agents Found: {len(features['intelligence_agents'])}")
    for agent in features['intelligence_agents'][:5]:
        print(f"  - {agent['name']}: {len(agent['methods'])} methods")
    if len(features['intelligence_agents']) > 5:
        print(f"  ... and {len(features['intelligence_agents']) - 5} more")
    
    print(f"\nTest Generators Found: {len(features['test_generators'])}")
    for gen in features['test_generators'][:5]:
        print(f"  - {Path(gen['file']).name}")
    
    print(f"\nSecurity Features Found: {len(features['security_features'])}")
    for sec in features['security_features'][:5]:
        print(f"  - {Path(sec['file']).name}")
    
    print(f"\nMonitoring Systems Found: {len(features['monitoring_systems'])}")
    for mon in features['monitoring_systems'][:5]:
        print(f"  - {Path(mon['file']).name}")
    
    print(f"\nVisualization Candidates Found: {len(features['visualization_candidates'])}")
    for viz in features['visualization_candidates'][:5]:
        print(f"  - {Path(viz['file']).name}: {', '.join(viz['indicators'])}")
    
    # Recommend new frontend integrations
    print("\n" + "=" * 80)
    print("RECOMMENDED NEW FRONTEND INTEGRATIONS")
    print("=" * 80)
    
    recommendations = [
        {
            'feature': 'Hybrid Intelligence Dashboard',
            'description': '16 agents + 5 bridges coordination visualization',
            'data_source': 'intelligence agents + bridge components',
            'visualization': '3D force graph with real-time coordination'
        },
        {
            'feature': 'DAG Workflow Visualization',
            'description': 'Interactive workflow orchestration diagrams',
            'data_source': 'workflow execution data + dependencies',
            'visualization': 'Interactive DAG with parallel execution paths'
        },
        {
            'feature': 'Security Compliance Dashboard',
            'description': 'OWASP/SOX/GDPR/PCI-DSS compliance tracking',
            'data_source': 'security scanner + compliance framework',
            'visualization': 'Compliance matrix with threat heatmap'
        },
        {
            'feature': 'Test Generation Pipeline',
            'description': 'Real-time test generation with self-healing',
            'data_source': 'test generators + healing iterations',
            'visualization': 'Pipeline flow with success/failure rates'
        },
        {
            'feature': 'Performance Bottleneck Map',
            'description': 'Real-time system bottleneck detection',
            'data_source': 'performance monitors + telemetry',
            'visualization': 'Flame graphs with bottleneck highlighting'
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['feature']}")
        print(f"   Description: {rec['description']}")
        print(f"   Data Source: {rec['data_source']}")
        print(f"   Visualization: {rec['visualization']}")
    
    return features

if __name__ == "__main__":
    features = discover_backend_features()
    
    # Save to JSON for frontend consumption
    output_file = Path(__file__).parent / 'backend_features.json'
    with open(output_file, 'w') as f:
        json.dump(features, f, indent=2)
    
    print(f"\nFeatures saved to: {output_file}")
    print(f"Total backend capabilities discovered: {sum(len(v) for v in features.values())}")