"""
Hierarchical Documentation Organizer

Creates progressive disclosure documentation systems with platform-specific
organization based on AutoGen's multi-layer documentation architecture.
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Platform(Enum):
    """Supported platforms for documentation."""
    PYTHON = "python"
    DOTNET = "dotnet"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"


class DocumentLevel(Enum):
    """Documentation complexity levels."""
    OVERVIEW = "overview"
    QUICKSTART = "quickstart"
    TUTORIAL = "tutorial"
    EXAMPLES = "examples"
    API_REFERENCE = "api_reference"
    DESIGN = "design"
    ADVANCED = "advanced"


@dataclass
class DocumentNode:
    """Represents a document in the hierarchy."""
    title: str
    path: str
    level: DocumentLevel
    platform: Optional[Platform] = None
    children: List['DocumentNode'] = field(default_factory=list)
    parent: Optional['DocumentNode'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    order: int = 0
    prerequisites: List[str] = field(default_factory=list)
    related_docs: List[str] = field(default_factory=list)


@dataclass
class NavigationStructure:
    """Represents the complete navigation structure."""
    root_nodes: List[DocumentNode] = field(default_factory=list)
    platform_trees: Dict[Platform, List[DocumentNode]] = field(default_factory=dict)
    cross_references: Dict[str, List[str]] = field(default_factory=dict)
    progression_paths: List[List[str]] = field(default_factory=list)


class HierarchicalDocsOrganizer:
    """
    Hierarchical documentation organizer inspired by AutoGen's 
    progressive disclosure and platform-specific documentation architecture.
    """
    
    def __init__(self, docs_root: str = "docs"):
        """Initialize hierarchical docs organizer."""
        self.docs_root = Path(docs_root)
        self.navigation = NavigationStructure()
        self.platform_configs = {}
        self.toc_templates = self._load_toc_templates()
        logger.info(f"Hierarchical docs organizer initialized at {docs_root}")
        
    def create_platform_structure(self, platforms: List[Platform]) -> None:
        """Create platform-specific documentation structure."""
        for platform in platforms:
            platform_dir = self.docs_root / platform.value
            platform_dir.mkdir(parents=True, exist_ok=True)
            
            # Create standard subdirectories with progressive complexity
            subdirs = {
                DocumentLevel.OVERVIEW: "overview",
                DocumentLevel.QUICKSTART: "quickstart", 
                DocumentLevel.TUTORIAL: "tutorial",
                DocumentLevel.EXAMPLES: "examples",
                DocumentLevel.API_REFERENCE: "api",
                DocumentLevel.DESIGN: "design",
                DocumentLevel.ADVANCED: "advanced"
            }
            
            for level, dirname in subdirs.items():
                subdir = platform_dir / dirname
                subdir.mkdir(exist_ok=True)
                
                # Create index file for each level
                self._create_level_index(subdir, level, platform)
                
            # Initialize platform tree
            self.navigation.platform_trees[platform] = []
            
        logger.info(f"Created platform structure for {[p.value for p in platforms]}")
        
    def add_document(self, 
                    title: str, 
                    path: str, 
                    level: DocumentLevel,
                    platform: Optional[Platform] = None,
                    parent_path: Optional[str] = None,
                    **metadata) -> DocumentNode:
        """Add document to hierarchy."""
        doc_node = DocumentNode(
            title=title,
            path=path,
            level=level,
            platform=platform,
            metadata=metadata
        )
        
        # Find and set parent
        if parent_path:
            parent = self._find_document_by_path(parent_path)
            if parent:
                doc_node.parent = parent
                parent.children.append(doc_node)
                doc_node.order = len(parent.children)
                
        # Add to appropriate tree
        if platform:
            if platform not in self.navigation.platform_trees:
                self.navigation.platform_trees[platform] = []
                
            if not parent_path:  # Root level for platform
                self.navigation.platform_trees[platform].append(doc_node)
        else:
            if not parent_path:  # Global root level
                self.navigation.root_nodes.append(doc_node)
                
        logger.info(f"Added document: {title} ({level.value})")
        return doc_node
        
    def create_progressive_navigation(self, platform: Platform) -> Dict[str, Any]:
        """Create progressive disclosure navigation for platform."""
        if platform not in self.navigation.platform_trees:
            return {}
            
        # Define progression order
        level_order = [
            DocumentLevel.OVERVIEW,
            DocumentLevel.QUICKSTART,
            DocumentLevel.TUTORIAL,
            DocumentLevel.EXAMPLES,
            DocumentLevel.API_REFERENCE,
            DocumentLevel.DESIGN,
            DocumentLevel.ADVANCED
        ]
        
        navigation = {
            "platform": platform.value,
            "progression": [],
            "quick_access": {},
            "cross_references": {}
        }
        
        # Build progressive structure
        for level in level_order:
            level_docs = self._get_documents_by_level(platform, level)
            if level_docs:
                section = {
                    "level": level.value,
                    "title": self._get_level_title(level),
                    "description": self._get_level_description(level),
                    "documents": [],
                    "estimated_time": self._estimate_reading_time(level_docs)
                }
                
                for doc in sorted(level_docs, key=lambda x: x.order):
                    doc_info = {
                        "title": doc.title,
                        "path": doc.path,
                        "prerequisites": doc.prerequisites,
                        "children": len(doc.children),
                        "metadata": doc.metadata
                    }
                    
                    section["documents"].append(doc_info)
                    
                navigation["progression"].append(section)
                
        # Add quick access links
        navigation["quick_access"] = self._create_quick_access_links(platform)
        
        # Add cross-references  
        navigation["cross_references"] = self._create_cross_references(platform)
        
        return navigation
        
    def generate_toc_yaml(self, platform: Platform) -> str:
        """Generate YAML table of contents for platform."""
        if platform not in self.navigation.platform_trees:
            return ""
            
        toc_structure = []
        
        def build_toc_node(doc_node: DocumentNode) -> Dict[str, Any]:
            node = {
                "name": doc_node.title,
                "href": doc_node.path
            }
            
            if doc_node.metadata:
                if "displayName" in doc_node.metadata:
                    node["displayName"] = doc_node.metadata["displayName"]
                if "summary" in doc_node.metadata:
                    node["summary"] = doc_node.metadata["summary"]
                    
            if doc_node.children:
                node["items"] = []
                for child in sorted(doc_node.children, key=lambda x: x.order):
                    node["items"].append(build_toc_node(child))
                    
            return node
            
        # Build TOC from platform tree
        for root_doc in sorted(self.navigation.platform_trees[platform], 
                              key=lambda x: x.order):
            toc_structure.append(build_toc_node(root_doc))
            
        return yaml.dump(toc_structure, default_flow_style=False, sort_keys=False)
        
    def create_switcher_config(self, platforms: List[Platform]) -> Dict[str, Any]:
        """Create platform/version switcher configuration."""
        switcher_config = {
            "versions": [],
            "platforms": [],
            "default_platform": platforms[0].value if platforms else None,
            "url_pattern": "/{platform}/{version}/{path}"
        }
        
        # Add platform configurations
        for platform in platforms:
            platform_config = {
                "name": platform.value,
                "display_name": platform.value.title(),
                "url": f"/{platform.value}/",
                "supported_versions": ["latest", "stable"],
                "documentation_root": f"/{platform.value}/",
                "api_reference": f"/{platform.value}/api/",
                "examples": f"/{platform.value}/examples/"
            }
            
            switcher_config["platforms"].append(platform_config)
            
        return switcher_config
        
    def analyze_documentation_gaps(self, platform: Platform) -> Dict[str, Any]:
        """Analyze gaps in documentation hierarchy."""
        if platform not in self.navigation.platform_trees:
            return {"gaps": [], "recommendations": []}
            
        analysis = {
            "gaps": [],
            "recommendations": [],
            "coverage": {},
            "depth_analysis": {}
        }
        
        # Check coverage by level
        for level in DocumentLevel:
            docs = self._get_documents_by_level(platform, level)
            coverage_percent = min(100, len(docs) * 20)  # Rough calculation
            
            analysis["coverage"][level.value] = {
                "document_count": len(docs),
                "coverage_estimate": f"{coverage_percent}%",
                "missing_topics": self._identify_missing_topics(level, docs)
            }
            
            if len(docs) == 0:
                analysis["gaps"].append({
                    "level": level.value,
                    "severity": "high",
                    "description": f"No {level.value} documentation found"
                })
                
        # Analyze progression paths
        progression_issues = self._analyze_progression_paths(platform)
        analysis["recommendations"].extend(progression_issues)
        
        # Depth analysis
        analysis["depth_analysis"] = self._analyze_documentation_depth(platform)
        
        return analysis
        
    def create_cross_platform_comparison(self, platforms: List[Platform]) -> str:
        """Create cross-platform comparison documentation."""
        comparison = [
            "# Platform Comparison",
            "",
            "Feature and capability comparison across platforms.",
            "",
            "| Feature | " + " | ".join([p.value.title() for p in platforms]) + " |",
            "|---------|" + "|".join(["-" * len(p.value) for p in platforms]) + "|"
        ]
        
        # Compare features across platforms
        features = [
            "Installation", "Quick Start", "API Reference", 
            "Examples", "Advanced Topics", "Community Support"
        ]
        
        for feature in features:
            row = [feature]
            
            for platform in platforms:
                # Check if feature exists for platform
                has_feature = self._platform_has_feature(platform, feature)
                status = "✅" if has_feature else "❌"
                row.append(status)
                
            comparison.append("| " + " | ".join(row) + " |")
            
        comparison.extend([
            "",
            "## Platform-Specific Notes",
            ""
        ])
        
        for platform in platforms:
            comparison.extend([
                f"### {platform.value.title()}",
                "",
                f"- Documentation root: `/{platform.value}/`",
                f"- API reference: `/{platform.value}/api/`",
                f"- Examples: `/{platform.value}/examples/`",
                ""
            ])
            
        return "\n".join(comparison)
        
    def _find_document_by_path(self, path: str) -> Optional[DocumentNode]:
        """Find document node by path."""
        all_docs = []
        
        # Collect all documents
        all_docs.extend(self.navigation.root_nodes)
        for platform_docs in self.navigation.platform_trees.values():
            all_docs.extend(platform_docs)
            
        # Search recursively
        def search_recursive(nodes: List[DocumentNode]) -> Optional[DocumentNode]:
            for node in nodes:
                if node.path == path:
                    return node
                result = search_recursive(node.children)
                if result:
                    return result
            return None
            
        return search_recursive(all_docs)
        
    def _get_documents_by_level(self, platform: Platform, level: DocumentLevel) -> List[DocumentNode]:
        """Get all documents for platform at specific level."""
        if platform not in self.navigation.platform_trees:
            return []
            
        def collect_level_docs(nodes: List[DocumentNode]) -> List[DocumentNode]:
            docs = []
            for node in nodes:
                if node.level == level:
                    docs.append(node)
                docs.extend(collect_level_docs(node.children))
            return docs
            
        return collect_level_docs(self.navigation.platform_trees[platform])
        
    def _create_level_index(self, directory: Path, level: DocumentLevel, platform: Platform) -> None:
        """Create index file for documentation level."""
        index_content = [
            f"# {self._get_level_title(level)}",
            "",
            self._get_level_description(level),
            "",
            f"Welcome to the {level.value} documentation for {platform.value.title()}.",
            "",
            "## Topics",
            "",
            "*Content will be populated as documentation is added.*",
            "",
            "## Navigation",
            "",
            "- [← Previous Level](../)",
            "- [Next Level →](../)",
            ""
        ]
        
        index_file = directory / "README.md"
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(index_content))
            
    def _get_level_title(self, level: DocumentLevel) -> str:
        """Get display title for documentation level."""
        titles = {
            DocumentLevel.OVERVIEW: "Overview & Getting Started",
            DocumentLevel.QUICKSTART: "Quick Start Guide", 
            DocumentLevel.TUTORIAL: "Tutorials & Learning Path",
            DocumentLevel.EXAMPLES: "Examples & Use Cases",
            DocumentLevel.API_REFERENCE: "API Reference",
            DocumentLevel.DESIGN: "Design & Architecture",
            DocumentLevel.ADVANCED: "Advanced Topics"
        }
        return titles.get(level, level.value.title())
        
    def _get_level_description(self, level: DocumentLevel) -> str:
        """Get description for documentation level."""
        descriptions = {
            DocumentLevel.OVERVIEW: "High-level introduction and conceptual overview.",
            DocumentLevel.QUICKSTART: "Get up and running quickly with minimal setup.",
            DocumentLevel.TUTORIAL: "Step-by-step guides for learning core concepts.",
            DocumentLevel.EXAMPLES: "Practical examples and real-world use cases.",
            DocumentLevel.API_REFERENCE: "Detailed API documentation and reference material.",
            DocumentLevel.DESIGN: "Architectural decisions and design principles.",
            DocumentLevel.ADVANCED: "Advanced topics and expert-level guidance."
        }
        return descriptions.get(level, "Documentation for this level.")
        
    def _estimate_reading_time(self, docs: List[DocumentNode]) -> str:
        """Estimate reading time for document level."""
        # Rough estimate based on document count and level
        base_minutes = {
            DocumentLevel.OVERVIEW: 5,
            DocumentLevel.QUICKSTART: 10,
            DocumentLevel.TUTORIAL: 20,
            DocumentLevel.EXAMPLES: 15,
            DocumentLevel.API_REFERENCE: 30,
            DocumentLevel.DESIGN: 25,
            DocumentLevel.ADVANCED: 35
        }
        
        if not docs:
            return "0 min"
            
        # Use first document's level for calculation
        level = docs[0].level
        base_time = base_minutes.get(level, 15)
        total_time = base_time * len(docs)
        
        if total_time < 60:
            return f"{total_time} min"
        else:
            hours = total_time // 60
            minutes = total_time % 60
            return f"{hours}h {minutes}min"
            
    def _create_quick_access_links(self, platform: Platform) -> Dict[str, str]:
        """Create quick access navigation links."""
        links = {}
        
        # Find key documents
        for level in DocumentLevel:
            docs = self._get_documents_by_level(platform, level)
            if docs:
                # Use first document as quick access
                links[level.value] = docs[0].path
                
        return links
        
    def _create_cross_references(self, platform: Platform) -> Dict[str, List[str]]:
        """Create cross-reference mappings."""
        cross_refs = {}
        
        # Build cross-references based on related_docs
        for tree_docs in [self.navigation.platform_trees.get(platform, [])]:
            for doc in tree_docs:
                if doc.related_docs:
                    cross_refs[doc.path] = doc.related_docs
                    
        return cross_refs
        
    def _identify_missing_topics(self, level: DocumentLevel, existing_docs: List[DocumentNode]) -> List[str]:
        """Identify missing topics for documentation level."""
        expected_topics = {
            DocumentLevel.OVERVIEW: ["Introduction", "Architecture", "Core Concepts"],
            DocumentLevel.QUICKSTART: ["Installation", "First Example", "Basic Usage"],
            DocumentLevel.TUTORIAL: ["Step-by-step Guide", "Common Patterns", "Best Practices"],
            DocumentLevel.EXAMPLES: ["Basic Examples", "Advanced Examples", "Use Cases"],
            DocumentLevel.API_REFERENCE: ["Classes", "Methods", "Configuration"],
            DocumentLevel.DESIGN: ["Architecture", "Patterns", "Decisions"],
            DocumentLevel.ADVANCED: ["Performance", "Customization", "Integration"]
        }
        
        existing_titles = [doc.title.lower() for doc in existing_docs]
        expected = expected_topics.get(level, [])
        
        missing = []
        for topic in expected:
            if not any(topic.lower() in title for title in existing_titles):
                missing.append(topic)
                
        return missing
        
    def _analyze_progression_paths(self, platform: Platform) -> List[Dict[str, Any]]:
        """Analyze learning progression paths."""
        recommendations = []
        
        # Check if progression makes sense
        level_counts = {}
        for level in DocumentLevel:
            level_counts[level] = len(self._get_documents_by_level(platform, level))
            
        # Check for gaps in progression
        if level_counts[DocumentLevel.OVERVIEW] == 0:
            recommendations.append({
                "type": "missing_foundation",
                "priority": "high",
                "description": "Missing overview documentation - users won't understand basics"
            })
            
        if level_counts[DocumentLevel.QUICKSTART] == 0 and level_counts[DocumentLevel.TUTORIAL] > 0:
            recommendations.append({
                "type": "missing_quickstart",
                "priority": "medium", 
                "description": "Users may struggle without quickstart before tutorials"
            })
            
        return recommendations
        
    def _analyze_documentation_depth(self, platform: Platform) -> Dict[str, Any]:
        """Analyze documentation depth and completeness."""
        depth_analysis = {}
        
        if platform in self.navigation.platform_trees:
            tree = self.navigation.platform_trees[platform]
            
            # Calculate tree depth
            def max_depth(nodes: List[DocumentNode]) -> int:
                if not nodes:
                    return 0
                return 1 + max((max_depth(node.children) for node in nodes), default=0)
                
            depth_analysis = {
                "max_depth": max_depth(tree),
                "total_documents": self._count_total_docs(tree),
                "average_children": self._calculate_average_children(tree),
                "leaf_documents": self._count_leaf_docs(tree)
            }
            
        return depth_analysis
        
    def _platform_has_feature(self, platform: Platform, feature: str) -> bool:
        """Check if platform has specific feature documented."""
        docs = self._get_documents_by_level(platform, DocumentLevel.OVERVIEW)
        docs.extend(self._get_documents_by_level(platform, DocumentLevel.QUICKSTART))
        
        for doc in docs:
            if feature.lower() in doc.title.lower():
                return True
                
        return False
        
    def _count_total_docs(self, nodes: List[DocumentNode]) -> int:
        """Count total documents in tree."""
        count = len(nodes)
        for node in nodes:
            count += self._count_total_docs(node.children)
        return count
        
    def _calculate_average_children(self, nodes: List[DocumentNode]) -> float:
        """Calculate average number of children per node."""
        if not nodes:
            return 0.0
            
        total_children = sum(len(node.children) for node in nodes)
        return total_children / len(nodes)
        
    def _count_leaf_docs(self, nodes: List[DocumentNode]) -> int:
        """Count leaf documents (no children)."""
        count = 0
        for node in nodes:
            if not node.children:
                count += 1
            else:
                count += self._count_leaf_docs(node.children)
        return count
        
    def _load_toc_templates(self) -> Dict[str, str]:
        """Load table of contents templates."""
        return {
            "docfx": """### YamlMime:TableOfContents
- name: {title}
  href: {href}
  items: {items}
""",
            "sphinx": """.. toctree::
   :maxdepth: {maxdepth}
   :caption: {caption}
   
   {items}
"""
        }
        
    def export_navigation_config(self, output_path: str) -> None:
        """Export complete navigation configuration."""
        config = {
            "navigation_structure": {
                "root_nodes": [self._node_to_dict(node) for node in self.navigation.root_nodes],
                "platform_trees": {
                    platform.value: [self._node_to_dict(node) for node in nodes]
                    for platform, nodes in self.navigation.platform_trees.items()
                }
            },
            "cross_references": self.navigation.cross_references,
            "progression_paths": self.navigation.progression_paths
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Exported navigation config to {output_path}")
        
    def _node_to_dict(self, node: DocumentNode) -> Dict[str, Any]:
        """Convert document node to dictionary."""
        return {
            "title": node.title,
            "path": node.path,
            "level": node.level.value,
            "platform": node.platform.value if node.platform else None,
            "order": node.order,
            "prerequisites": node.prerequisites,
            "related_docs": node.related_docs,
            "metadata": node.metadata,
            "children": [self._node_to_dict(child) for child in node.children]
        }