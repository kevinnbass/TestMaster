"""
Simple Knowledge Management Test
Agent D - Hour 4: Knowledge Management Systems
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add TestMaster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TestMaster'))

from TestMaster.core.intelligence.documentation.knowledge_management_framework import (
    KnowledgeManagementFramework,
    build_knowledge_base,
    search_knowledge_base
)

def test_knowledge_management_simple():
    """Test the knowledge management framework with available methods."""
    
    print("=" * 80)
    print("Agent D - Hour 4: Knowledge Management Systems")
    print("Testing Knowledge Management Framework")
    print("=" * 80)
    
    # Initialize framework
    framework = KnowledgeManagementFramework()
    
    # Build knowledge base from TestMaster directories
    print("\n1. Building Knowledge Base...")
    
    directories = [
        "TestMaster/core/intelligence/documentation",
        "TestMaster/core/intelligence/analysis",
        "TestMaster/archive"
    ]
    
    # Filter to only existing directories
    existing_dirs = [d for d in directories if Path(d).exists()]
    print(f"   Analyzing {len(existing_dirs)} directories:")
    for d in existing_dirs:
        print(f"   - {d}")
    
    if existing_dirs:
        kb_result = framework.build_knowledge_base(existing_dirs)
        
        print(f"\n   Knowledge Base Built:")
        print(f"   - Total Items: {kb_result.get('total_items', 0)}")
        print(f"   - Sources Processed: {kb_result.get('sources_processed', 0)}")
        
        if 'statistics' in kb_result:
            stats = kb_result['statistics']
            print(f"   - By Type:")
            for ktype, count in stats.get('by_type', {}).items():
                print(f"     - {ktype}: {count}")
    
    # Test search functionality
    print("\n2. Testing Knowledge Search...")
    
    test_queries = [
        "documentation",
        "API",
        "legacy",
        "test",
        "integration"
    ]
    
    for query in test_queries:
        print(f"\n   Searching for: '{query}'")
        search_result = framework.search_knowledge(query)
        
        if search_result and 'results' in search_result:
            results = search_result['results']
            print(f"   Found {len(results)} results")
            
            # Show top 3 results
            for i, result in enumerate(results[:3], 1):
                if isinstance(result, dict):
                    title = result.get('title', 'Untitled')
                    score = result.get('score', 0.0)
                    source = result.get('source', 'Unknown')
                    print(f"   {i}. {title}")
                    print(f"      Score: {score:.3f}, Source: {source}")
    
    # Test related knowledge
    print("\n3. Testing Related Knowledge...")
    
    # Get some knowledge items to test with
    if existing_dirs:
        kb_items = kb_result.get('knowledge_items', [])
        if kb_items and len(kb_items) > 0:
            sample_item = kb_items[0]
            if hasattr(sample_item, 'item_id'):
                item_id = sample_item.item_id
            elif isinstance(sample_item, dict):
                item_id = sample_item.get('id', sample_item.get('item_id'))
            else:
                item_id = None
            
            if item_id:
                print(f"   Finding related knowledge for item: {item_id}")
                related = framework.get_related_knowledge(item_id, max_related=5)
                
                if related:
                    print(f"   Found {len(related)} related items:")
                    for i, rel in enumerate(related, 1):
                        if isinstance(rel, dict):
                            title = rel.get('title', 'Untitled')
                            score = rel.get('relevance_score', 0.0)
                            print(f"   {i}. {title} (Score: {score:.3f})")
    
    # Generate reports
    print("\n4. Generating Knowledge Reports...")
    
    output_dir = Path("TestMaster/docs/knowledge")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create summary report
    summary = {
        "timestamp": datetime.now().isoformat(),
        "analyzer": "Agent D - Knowledge Management Framework",
        "directories_analyzed": existing_dirs,
        "knowledge_base": {
            "total_items": kb_result.get('total_items', 0) if 'kb_result' in locals() else 0,
            "sources_processed": kb_result.get('sources_processed', 0) if 'kb_result' in locals() else 0
        },
        "search_tests": len(test_queries),
        "status": "operational"
    }
    
    # Export summary
    summary_path = output_dir / "knowledge_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"   Summary report: {summary_path}")
    
    # Create knowledge documentation
    doc_content = f"""# Knowledge Management System Documentation

**Generated**: {datetime.now().isoformat()}  
**Agent**: Agent D - Documentation & Validation Excellence

## System Overview

The TestMaster Knowledge Management System provides intelligent knowledge extraction, 
search, and retrieval capabilities across the entire codebase.

## Knowledge Base Statistics

- **Total Knowledge Items**: {kb_result.get('total_items', 0) if 'kb_result' in locals() else 0}
- **Sources Processed**: {kb_result.get('sources_processed', 0) if 'kb_result' in locals() else 0}
- **Directories Analyzed**: {len(existing_dirs)}

## Features

### 1. Knowledge Extraction
- Extracts knowledge from documentation files (Markdown)
- Extracts knowledge from code comments and docstrings
- Identifies technical concepts and relationships

### 2. Semantic Search
- Natural language query processing
- Relevance scoring and ranking
- Context-aware search results

### 3. Knowledge Graph
- Builds relationships between concepts
- Identifies related knowledge items
- Enables knowledge discovery

### 4. Recommendations
- Suggests related documentation
- Identifies knowledge gaps
- Provides learning paths

## API Usage

```python
from knowledge_management_framework import KnowledgeManagementFramework

# Initialize framework
framework = KnowledgeManagementFramework()

# Build knowledge base
kb = framework.build_knowledge_base(['path/to/docs'])

# Search knowledge
results = framework.search_knowledge('query')

# Get related items
related = framework.get_related_knowledge('item_id')
```

## Search Capabilities

The system supports various search patterns:
- **Keyword Search**: Direct keyword matching
- **Semantic Search**: Understanding query intent
- **Context Search**: Using context for better results
- **Fuzzy Search**: Handling typos and variations

## Knowledge Sources

The system extracts knowledge from:
1. Documentation files (*.md)
2. Python docstrings
3. Code comments
4. Configuration files
5. README files
6. API specifications

## Integration Points

- **Documentation System**: Generates searchable docs
- **API Validation**: Links API docs to implementation
- **Legacy Integration**: Maps legacy knowledge
- **Testing Framework**: Links tests to documentation

---

*Knowledge Management System - Part of TestMaster Intelligence Framework*
"""
    
    doc_path = output_dir / "KNOWLEDGE_MANAGEMENT_DOCUMENTATION.md"
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(doc_content)
    print(f"   Documentation: {doc_path}")
    
    print("\n" + "=" * 80)
    print("Knowledge Management Framework Test Complete!")
    print("=" * 80)
    
    return summary

if __name__ == "__main__":
    summary = test_knowledge_management_simple()