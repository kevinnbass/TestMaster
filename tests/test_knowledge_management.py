"""
Test Knowledge Management Framework
Agent D - Hour 4: Knowledge Management Systems
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import hashlib
from typing import List, Dict, Any

# Add TestMaster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TestMaster'))

from TestMaster.core.intelligence.documentation.knowledge_management_framework import (
    KnowledgeManagementFramework,
    KnowledgeExtractor,
    SemanticSearchEngine,
    KnowledgeGraphBuilder,
    KnowledgeItem,
    KnowledgeType,
    KnowledgeSource,
    SearchRelevance
)

def test_knowledge_management():
    """Test the knowledge management framework."""
    
    print("=" * 80)
    print("Agent D - Hour 4: Knowledge Management Systems")
    print("Testing Knowledge Management Framework")
    print("=" * 80)
    
    # Initialize framework
    framework = KnowledgeManagementFramework()
    extractor = KnowledgeExtractor()
    search_engine = SemanticSearchEngine()
    graph_builder = KnowledgeGraphBuilder()
    
    # Extract knowledge from codebase
    print("\n1. Extracting Knowledge from Codebase...")
    codebase_path = Path("TestMaster")
    
    # Extract from documentation files
    doc_knowledge = []
    doc_count = 0
    for doc_file in codebase_path.rglob("*.md"):
        if "node_modules" not in str(doc_file) and ".git" not in str(doc_file):
            try:
                knowledge = extractor.extract_from_markdown(str(doc_file))
                if knowledge:
                    doc_knowledge.extend(knowledge)
                    doc_count += 1
            except:
                pass
    
    print(f"   Extracted knowledge from {doc_count} documentation files")
    print(f"   Total knowledge items: {len(doc_knowledge)}")
    
    # Extract from Python docstrings
    print("\n2. Extracting Knowledge from Code...")
    code_knowledge = []
    py_count = 0
    for py_file in codebase_path.rglob("*.py"):
        if "node_modules" not in str(py_file) and ".git" not in str(py_file):
            try:
                knowledge = extractor.extract_from_python(str(py_file))
                if knowledge:
                    code_knowledge.extend(knowledge)
                    py_count += 1
                    if py_count >= 50:  # Limit for performance
                        break
            except:
                pass
    
    print(f"   Extracted knowledge from {py_count} Python files")
    print(f"   Total code knowledge items: {len(code_knowledge)}")
    
    # Build knowledge base
    print("\n3. Building Knowledge Base...")
    all_knowledge = doc_knowledge + code_knowledge
    
    # Add to framework
    for item in all_knowledge:
        framework.add_knowledge_item(item)
    
    knowledge_stats = framework.get_knowledge_statistics()
    print(f"   Knowledge Base Statistics:")
    print(f"   - Total Items: {knowledge_stats['total_items']}")
    print(f"   - By Type:")
    for ktype, count in knowledge_stats['by_type'].items():
        print(f"     - {ktype}: {count}")
    print(f"   - By Source:")
    for source, count in knowledge_stats['by_source'].items():
        print(f"     - {source}: {count}")
    
    # Test semantic search
    print("\n4. Testing Semantic Search...")
    
    test_queries = [
        "How to generate API documentation?",
        "What is the legacy integration framework?",
        "How does the knowledge management system work?",
        "Testing framework capabilities",
        "Archive system preservation rules"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = search_engine.search(query, all_knowledge, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"   Result {i}:")
                print(f"     - Title: {result.item.title}")
                print(f"     - Relevance: {result.relevance.value}")
                print(f"     - Score: {result.relevance_score:.3f}")
                if result.match_reasons:
                    print(f"     - Match Reasons: {', '.join(result.match_reasons[:2])}")
    
    # Build knowledge graph
    print("\n5. Building Knowledge Graph...")
    graph = graph_builder.build_graph(all_knowledge[:100])  # Limit for performance
    
    print(f"   Knowledge Graph Statistics:")
    print(f"   - Nodes: {len(graph.nodes)}")
    print(f"   - Edges: {len(graph.edges)}")
    print(f"   - Concepts: {len(graph.concepts)}")
    
    # Show top concepts
    if graph.concepts:
        top_concepts = list(graph.concepts)[:10]
        print(f"   - Top Concepts: {', '.join(top_concepts)}")
    
    # Test knowledge recommendations
    print("\n6. Testing Knowledge Recommendations...")
    
    if all_knowledge:
        sample_item = all_knowledge[0]
        recommendations = framework.get_recommendations(sample_item, limit=5)
        
        print(f"   Recommendations for: '{sample_item.title}'")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec.title} (Score: {rec.relevance_score:.3f})")
    
    # Generate knowledge reports
    print("\n7. Generating Knowledge Reports...")
    
    # Create output directory
    output_dir = Path("TestMaster/docs/knowledge")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate knowledge base report
    kb_report = {
        "timestamp": datetime.now().isoformat(),
        "analyzer": "Agent D - Knowledge Management Framework",
        "statistics": knowledge_stats,
        "total_items": len(all_knowledge),
        "sources_analyzed": {
            "documentation_files": doc_count,
            "python_files": py_count
        },
        "knowledge_types": {
            ktype.value: sum(1 for item in all_knowledge if item.knowledge_type == ktype)
            for ktype in KnowledgeType
        },
        "graph_statistics": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "concepts": len(graph.concepts)
        }
    }
    
    # Export knowledge base report
    kb_report_path = output_dir / "knowledge_base_report.json"
    with open(kb_report_path, 'w', encoding='utf-8') as f:
        json.dump(kb_report, f, indent=2)
    print(f"   Knowledge base report: {kb_report_path}")
    
    # Generate knowledge index
    knowledge_index = framework.generate_knowledge_index()
    index_path = output_dir / "knowledge_index.md"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(knowledge_index)
    print(f"   Knowledge index: {index_path}")
    
    # Generate search index
    search_index = search_engine.build_search_index(all_knowledge)
    search_index_path = output_dir / "search_index.json"
    with open(search_index_path, 'w', encoding='utf-8') as f:
        json.dump(search_index, f, indent=2)
    print(f"   Search index: {search_index_path}")
    
    # Generate knowledge graph visualization data
    graph_data = {
        "nodes": [
            {
                "id": node_id,
                "label": node_data.get("label", node_id),
                "type": node_data.get("type", "concept")
            }
            for node_id, node_data in graph.nodes.items()
        ],
        "edges": [
            {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "relationship": edge.get("relationship", "related")
            }
            for edge in graph.edges
        ]
    }
    
    graph_path = output_dir / "knowledge_graph.json"
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2)
    print(f"   Knowledge graph data: {graph_path}")
    
    print("\n" + "=" * 80)
    print("Knowledge Management Framework Test Complete!")
    print("=" * 80)
    
    return kb_report

if __name__ == "__main__":
    report = test_knowledge_management()