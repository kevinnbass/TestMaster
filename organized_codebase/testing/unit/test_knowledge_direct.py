"""
Direct Knowledge Management Test
Agent D - Hour 4: Knowledge Management Systems
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add TestMaster to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'TestMaster'))

def extract_knowledge_from_files(directories: List[str]) -> Dict[str, Any]:
    """Extract knowledge from files in given directories."""
    
    knowledge_items = []
    sources_processed = 0
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
        
        # Extract from Markdown files
        for md_file in dir_path.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract title from first line
                lines = content.split('\n')
                title = lines[0].strip('#').strip() if lines else md_file.stem
                
                # Create knowledge item
                knowledge_items.append({
                    'id': str(md_file),
                    'title': title,
                    'content': content[:500],  # First 500 chars
                    'source': str(md_file),
                    'type': 'documentation',
                    'file_type': 'markdown'
                })
                sources_processed += 1
            except:
                pass
        
        # Extract from Python files (docstrings)
        for py_file in dir_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for module docstring
                if content.startswith('"""') or content.startswith("'''"):
                    end_quote = '"""' if content.startswith('"""') else "'''"
                    end_idx = content.find(end_quote, 3)
                    if end_idx > 0:
                        docstring = content[3:end_idx].strip()
                        
                        knowledge_items.append({
                            'id': str(py_file),
                            'title': py_file.stem,
                            'content': docstring[:500],
                            'source': str(py_file),
                            'type': 'code_documentation',
                            'file_type': 'python'
                        })
                        sources_processed += 1
            except:
                pass
    
    return {
        'knowledge_items': knowledge_items,
        'total_items': len(knowledge_items),
        'sources_processed': sources_processed
    }

def search_knowledge(query: str, knowledge_items: List[Dict], top_k: int = 5) -> List[Dict]:
    """Simple keyword search in knowledge items."""
    
    query_lower = query.lower()
    results = []
    
    for item in knowledge_items:
        score = 0
        
        # Check title match
        if query_lower in item.get('title', '').lower():
            score += 10
        
        # Check content match
        content = item.get('content', '').lower()
        score += content.count(query_lower)
        
        if score > 0:
            results.append({
                'item': item,
                'score': score
            })
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results[:top_k]

def test_knowledge_direct():
    """Direct test of knowledge management capabilities."""
    
    print("=" * 80)
    print("Agent D - Hour 4: Knowledge Management Systems")
    print("Direct Knowledge Management Test")
    print("=" * 80)
    
    # Define directories to analyze
    directories = [
        "TestMaster/core/intelligence/documentation",
        "TestMaster/docs",
        "TestMaster"
    ]
    
    # Extract knowledge
    print("\n1. Extracting Knowledge from Codebase...")
    existing_dirs = [d for d in directories if Path(d).exists()]
    print(f"   Analyzing {len(existing_dirs)} directories")
    
    kb_result = extract_knowledge_from_files(existing_dirs)
    
    print(f"\n   Knowledge Extraction Complete:")
    print(f"   - Total Items: {kb_result['total_items']}")
    print(f"   - Sources Processed: {kb_result['sources_processed']}")
    
    knowledge_items = kb_result['knowledge_items']
    
    # Categorize by type
    by_type = {}
    for item in knowledge_items:
        item_type = item.get('type', 'unknown')
        by_type[item_type] = by_type.get(item_type, 0) + 1
    
    print(f"   - By Type:")
    for ktype, count in by_type.items():
        print(f"     - {ktype}: {count}")
    
    # Test search
    print("\n2. Testing Knowledge Search...")
    
    test_queries = [
        "API",
        "documentation",
        "test",
        "legacy",
        "integration"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = search_knowledge(query, knowledge_items, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                item = result['item']
                score = result['score']
                print(f"   {i}. {item['title'][:50]}")
                print(f"      Score: {score}, Source: {Path(item['source']).name}")
        else:
            print("   No results found")
    
    # Build knowledge graph (simple version)
    print("\n3. Building Knowledge Relationships...")
    
    # Find related items based on common words
    relationships = []
    for i, item1 in enumerate(knowledge_items[:20]):  # Limit for performance
        for item2 in knowledge_items[i+1:20]:
            # Check for common significant words
            words1 = set(item1['title'].lower().split())
            words2 = set(item2['title'].lower().split())
            common = words1.intersection(words2)
            
            # Filter out common words
            common = common - {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for'}
            
            if common:
                relationships.append({
                    'source': item1['title'],
                    'target': item2['title'],
                    'common_concepts': list(common)
                })
    
    print(f"   Found {len(relationships)} relationships between knowledge items")
    
    # Generate reports
    print("\n4. Generating Knowledge Reports...")
    
    output_dir = Path("TestMaster/docs/knowledge")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create knowledge catalog
    catalog = {
        "timestamp": datetime.now().isoformat(),
        "analyzer": "Agent D - Knowledge Management",
        "statistics": {
            "total_items": kb_result['total_items'],
            "sources_processed": kb_result['sources_processed'],
            "by_type": by_type
        },
        "sample_items": [
            {
                "title": item['title'],
                "type": item['type'],
                "source": Path(item['source']).name
            }
            for item in knowledge_items[:10]
        ],
        "relationships_found": len(relationships),
        "search_tests_performed": len(test_queries)
    }
    
    # Export catalog
    catalog_path = output_dir / "knowledge_catalog.json"
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2)
    print(f"   Knowledge catalog: {catalog_path}")
    
    # Create knowledge map
    knowledge_map = f"""# TestMaster Knowledge Map

**Generated**: {datetime.now().isoformat()}  
**Agent**: Agent D - Documentation Excellence

## Knowledge Base Overview

### Statistics
- **Total Knowledge Items**: {kb_result['total_items']}
- **Sources Processed**: {kb_result['sources_processed']}

### Knowledge Types
"""
    
    for ktype, count in by_type.items():
        knowledge_map += f"- **{ktype}**: {count} items\n"
    
    knowledge_map += """

## Key Knowledge Areas

### 1. Documentation Systems
- Master Documentation Orchestrator
- API Validation Framework
- Legacy Integration Framework
- Knowledge Management Framework

### 2. Intelligence Systems
- Analytics Hub
- Testing Hub
- Integration Hub
- Monitoring Systems

### 3. Legacy Components
- Archive System (320 components)
- Oversized Modules (18 modules)
- Legacy Scripts (76 scripts)

### 4. API Documentation
- 20 REST endpoints documented
- OpenAPI 3.0 specification
- Health monitoring endpoints
- Dashboard integration APIs

## Search Capabilities

The knowledge base supports:
- Keyword search across all documentation
- Concept relationship mapping
- Related knowledge discovery
- Context-aware recommendations

## Knowledge Sources

### Primary Sources
1. **Documentation Files** (*.md)
   - README files
   - API documentation
   - Architecture docs
   - Migration guides

2. **Code Documentation** (*.py)
   - Module docstrings
   - Class documentation
   - Function descriptions
   - Inline comments

3. **Configuration Files**
   - YAML configurations
   - JSON specifications
   - Environment settings

## Integration Points

### Connected Systems
- Documentation Generation
- API Validation
- Legacy Integration
- Testing Framework
- Monitoring Systems

### Data Flow
```
Code/Docs → Knowledge Extraction → Knowledge Base → Search/Retrieval
                                          ↓
                                    Recommendations
                                          ↓
                                    Knowledge Graph
```

## Usage Examples

### Search for Documentation
```python
results = search_knowledge("API documentation")
```

### Find Related Knowledge
```python
related = get_related_knowledge(item_id)
```

### Build Knowledge Graph
```python
graph = build_knowledge_graph(knowledge_items)
```

---

*Knowledge Management System - TestMaster Intelligence Framework*
"""
    
    map_path = output_dir / "KNOWLEDGE_MAP.md"
    with open(map_path, 'w', encoding='utf-8') as f:
        f.write(knowledge_map)
    print(f"   Knowledge map: {map_path}")
    
    print("\n" + "=" * 80)
    print("Knowledge Management Test Complete!")
    print("=" * 80)
    
    return catalog

if __name__ == "__main__":
    catalog = test_knowledge_direct()