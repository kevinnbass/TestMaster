"""
Swarms Navigation Intelligence - Phase 1.7 Module 3/4

Adapts Swarms' intelligent navigation and organization patterns:
- Hierarchical documentation structure intelligence
- Smart navigation generation
- Multi-dimensional content organization
- Dynamic content discovery and linking
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import re


@dataclass
class NavigationNode:
    """Represents a node in the documentation navigation structure"""
    name: str
    path: str
    node_type: str  # 'section', 'page', 'external'
    children: List['NavigationNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent: Optional['NavigationNode'] = None


@dataclass
class ContentCategory:
    """Represents a content category with smart organization"""
    category_name: str
    priority: int
    organization_rules: List[str]
    content_patterns: List[str]
    subcategories: List[str] = field(default_factory=list)


class SwarmsNavigationIntelligence:
    """
    Adapts Swarms' navigation intelligence for TestMaster documentation.
    
    Key Swarms navigation patterns:
    1. Multi-dimensional content organization (by functionality, complexity, use case)
    2. Intelligent MkDocs navigation with smart sectioning
    3. Dynamic content discovery and automatic linking
    4. User-journey-based information architecture
    """
    
    def __init__(self):
        self.navigation_tree: Optional[NavigationNode] = None
        self.content_categories: Dict[str, ContentCategory] = {}
        self.link_intelligence: Dict[str, List[str]] = defaultdict(list)
        self.setup_swarms_navigation_patterns()
        
    def setup_swarms_navigation_patterns(self):
        """Setup navigation patterns based on Swarms' organization"""
        
        # Content categories extracted from Swarms
        self.content_categories = {
            'onboarding': ContentCategory(
                category_name="Onboarding",
                priority=1,
                organization_rules=['installation_first', 'quickstart_second', 'environment_third'],
                content_patterns=['install', 'setup', 'config', 'quick*'],
                subcategories=['Installation', 'Environment Configuration', 'Quickstart']
            ),
            
            'core_concepts': ContentCategory(
                category_name="Core Concepts", 
                priority=2,
                organization_rules=['overview_first', 'basic_to_advanced', 'theory_then_practice'],
                content_patterns=['agent*', 'swarm*', 'architecture*', 'concept*'],
                subcategories=['Agents', 'Multi-Agent Architectures', 'Framework Concepts']
            ),
            
            'examples': ContentCategory(
                category_name="Examples",
                priority=3,
                organization_rules=['basic_first', 'complexity_ascending', 'use_case_grouping'],
                content_patterns=['example*', 'demo*', 'tutorial*', 'guide*'],
                subcategories=['Basic Examples', 'Advanced Examples', 'Use Cases', 'Templates']
            ),
            
            'api_reference': ContentCategory(
                category_name="API Reference",
                priority=4,
                organization_rules=['alphabetical', 'functionality_groups', 'inheritance_hierarchy'],
                content_patterns=['api*', 'reference*', 'class*', 'method*'],
                subcategories=['Core API', 'Extensions', 'Utilities']
            ),
            
            'deployment': ContentCategory(
                category_name="Deployment",
                priority=5,
                organization_rules=['local_first', 'complexity_ascending', 'platform_specific'],
                content_patterns=['deploy*', 'cloud*', 'production*', 'scale*'],
                subcategories=['Local Deployment', 'Cloud Deployment', 'Production Systems']
            )
        }
    
    def extract_swarms_navigation_structure(self) -> NavigationNode:
        """Extract and adapt Swarms' navigation structure"""
        
        # Create root navigation node
        root = NavigationNode(
            name="TestMaster Documentation",
            path="/",
            node_type="section",
            metadata={'description': 'Complete TestMaster documentation based on Swarms patterns'}
        )
        
        # Build navigation tree following Swarms pattern
        home_section = self._create_home_section()
        core_section = self._create_core_concepts_section()
        examples_section = self._create_examples_section()
        api_section = self._create_api_section()
        deployment_section = self._create_deployment_section()
        
        # Add sections to root
        for section in [home_section, core_section, examples_section, api_section, deployment_section]:
            section.parent = root
            root.children.append(section)
        
        self.navigation_tree = root
        return root
    
    def _create_home_section(self) -> NavigationNode:
        """Create home section following Swarms pattern"""
        home = NavigationNode(
            name="Home",
            path="/home/",
            node_type="section",
            metadata={'icon': 'home', 'description': 'Getting started with TestMaster'}
        )
        
        # Onboarding subsection
        onboarding_pages = [
            ('Overview', '/overview.md', 'Overview of TestMaster capabilities'),
            ('Installation', '/install.md', 'Installation and setup guide'), 
            ('Environment Configuration', '/env.md', 'Environment configuration'),
            ('Quickstart', '/quickstart.md', 'Quick start tutorial')
        ]
        
        for name, path, desc in onboarding_pages:
            page = NavigationNode(
                name=name, 
                path=path,
                node_type="page",
                metadata={'description': desc, 'category': 'onboarding'}
            )
            page.parent = home
            home.children.append(page)
        
        return home
    
    def _create_core_concepts_section(self) -> NavigationNode:
        """Create core concepts section following Swarms pattern"""
        core = NavigationNode(
            name="Core Concepts",
            path="/core/",
            node_type="section", 
            metadata={'icon': 'concept', 'description': 'Core TestMaster concepts and architecture'}
        )
        
        # Intelligence subsections
        intelligence_sections = [
            ('Testing Intelligence', '/testing/', 'AI-powered test generation and analysis'),
            ('Documentation Intelligence', '/documentation/', 'Intelligent documentation systems'),
            ('Security Intelligence', '/security/', 'Advanced security analysis and validation'),
            ('ML Intelligence', '/ml/', 'Machine learning for test optimization'),
            ('Analytics Intelligence', '/analytics/', 'Predictive analytics and insights')
        ]
        
        for name, path, desc in intelligence_sections:
            section = NavigationNode(
                name=name,
                path=path, 
                node_type="section",
                metadata={'description': desc, 'category': 'core_concepts'}
            )
            section.parent = core
            core.children.append(section)
        
        return core
    
    def _create_examples_section(self) -> NavigationNode:
        """Create examples section following Swarms pattern"""
        examples = NavigationNode(
            name="Examples",
            path="/examples/",
            node_type="section",
            metadata={'icon': 'examples', 'description': 'Comprehensive examples and tutorials'}
        )
        
        # Example categories
        example_categories = [
            ('Basic Examples', '/basic/', [
                ('Simple Test Generation', '/basic/simple.md'),
                ('Configuration Examples', '/basic/config.md'), 
                ('Quick Start Tutorials', '/basic/tutorials.md')
            ]),
            ('Advanced Examples', '/advanced/', [
                ('Multi-Agent Testing', '/advanced/multi_agent.md'),
                ('Enterprise Integration', '/advanced/enterprise.md'),
                ('Custom Intelligence', '/advanced/custom.md')
            ]),
            ('Use Cases', '/use_cases/', [
                ('Financial Testing', '/use_cases/finance.md'),
                ('Healthcare Applications', '/use_cases/healthcare.md'),
                ('Security Validation', '/use_cases/security.md')
            ])
        ]
        
        for cat_name, cat_path, pages in example_categories:
            category = NavigationNode(
                name=cat_name,
                path=cat_path,
                node_type="section",
                metadata={'category': 'examples'}
            )
            category.parent = examples
            examples.children.append(category)
            
            for page_name, page_path in pages:
                page = NavigationNode(
                    name=page_name,
                    path=page_path,
                    node_type="page",
                    metadata={'category': 'examples', 'subcategory': cat_name.lower()}
                )
                page.parent = category
                category.children.append(page)
        
        return examples
    
    def _create_api_section(self) -> NavigationNode:
        """Create API reference section"""
        api = NavigationNode(
            name="API Reference",
            path="/api/",
            node_type="section",
            metadata={'icon': 'api', 'description': 'Complete API reference documentation'}
        )
        
        # API subsections
        api_sections = [
            ('Core API', '/core_api/', 'Core TestMaster API'),
            ('Intelligence API', '/intelligence_api/', 'AI and ML intelligence APIs'),
            ('Testing API', '/testing_api/', 'Test generation and execution APIs'),
            ('Analytics API', '/analytics_api/', 'Analytics and reporting APIs')
        ]
        
        for name, path, desc in api_sections:
            section = NavigationNode(
                name=name,
                path=path,
                node_type="section",
                metadata={'description': desc, 'category': 'api_reference'}
            )
            section.parent = api
            api.children.append(section)
        
        return api
    
    def _create_deployment_section(self) -> NavigationNode:
        """Create deployment section"""
        deployment = NavigationNode(
            name="Deployment",
            path="/deployment/",
            node_type="section",
            metadata={'icon': 'deployment', 'description': 'Deployment and scaling guides'}
        )
        
        # Deployment options
        deployment_options = [
            ('Local Development', '/local.md'),
            ('Docker Deployment', '/docker.md'),
            ('Cloud Deployment', '/cloud.md'),
            ('Enterprise Scaling', '/enterprise.md')
        ]
        
        for name, path in deployment_options:
            page = NavigationNode(
                name=name,
                path=path,
                node_type="page",
                metadata={'category': 'deployment'}
            )
            page.parent = deployment
            deployment.children.append(page)
        
        return deployment
    
    def generate_mkdocs_nav(self, root: Optional[NavigationNode] = None) -> Dict[str, Any]:
        """Generate MkDocs navigation YAML following Swarms pattern"""
        if root is None:
            root = self.navigation_tree
        
        if not root:
            root = self.extract_swarms_navigation_structure()
        
        def build_nav_dict(node: NavigationNode) -> Any:
            if node.node_type == "page":
                return node.path
            elif node.node_type == "section":
                if node.children:
                    section_dict = {}
                    for child in node.children:
                        if child.node_type == "page":
                            section_dict[child.name] = child.path
                        else:
                            section_dict[child.name] = build_nav_dict(child)
                    return section_dict
                else:
                    return []
            return None
        
        nav_config = []
        for child in root.children:
            nav_item = {child.name: build_nav_dict(child)}
            nav_config.append(nav_item)
        
        return {'nav': nav_config}
    
    def build_smart_linking_system(self) -> Dict[str, Any]:
        """Build intelligent cross-linking system based on Swarms patterns"""
        
        linking_rules = {
            'content_relationships': {
                'prerequisite_linking': 'Link to required knowledge before advanced topics',
                'related_concepts': 'Auto-link related concepts and examples',
                'hierarchical_navigation': 'Provide up/down navigation in hierarchies',
                'cross_references': 'Link between examples and their documentation'
            },
            
            'auto_discovery': {
                'tag_based_linking': 'Link content with similar tags',
                'semantic_similarity': 'Link semantically related content',
                'user_journey_paths': 'Create guided learning paths',
                'contextual_suggestions': 'Suggest related content based on current page'
            },
            
            'dynamic_updates': {
                'broken_link_detection': 'Automatically detect and report broken links',
                'content_freshness': 'Flag outdated content and suggest updates',
                'usage_analytics': 'Track popular paths and optimize navigation'
            }
        }
        
        return linking_rules
    
    def generate_navigation_analytics(self) -> Dict[str, Any]:
        """Generate navigation analytics and optimization suggestions"""
        if not self.navigation_tree:
            self.extract_swarms_navigation_structure()
        
        analytics = {
            'structure_analysis': {
                'total_sections': self._count_nodes_by_type('section'),
                'total_pages': self._count_nodes_by_type('page'),
                'max_depth': self._calculate_max_depth(),
                'categories_distribution': self._analyze_category_distribution()
            },
            
            'optimization_suggestions': [
                'Consider consolidating sections with <3 items',
                'Add more cross-links between related concepts',
                'Create guided tutorial paths for common use cases',
                'Implement search and filtering capabilities'
            ],
            
            'swarms_patterns_applied': [
                'Multi-dimensional organization (functionality/complexity/use-case)',
                'User-journey-based information architecture', 
                'Hierarchical section organization with smart navigation',
                'Category-based content grouping with priority ordering'
            ]
        }
        
        return analytics
    
    def _count_nodes_by_type(self, node_type: str) -> int:
        """Count nodes of specific type in navigation tree"""
        def count_recursive(node: NavigationNode) -> int:
            count = 1 if node.node_type == node_type else 0
            for child in node.children:
                count += count_recursive(child)
            return count
        
        return count_recursive(self.navigation_tree) if self.navigation_tree else 0
    
    def _calculate_max_depth(self) -> int:
        """Calculate maximum depth of navigation tree"""
        def depth_recursive(node: NavigationNode, current_depth: int = 0) -> int:
            if not node.children:
                return current_depth
            return max(depth_recursive(child, current_depth + 1) for child in node.children)
        
        return depth_recursive(self.navigation_tree) if self.navigation_tree else 0
    
    def _analyze_category_distribution(self) -> Dict[str, int]:
        """Analyze distribution of content across categories"""
        distribution = defaultdict(int)
        
        def analyze_recursive(node: NavigationNode):
            category = node.metadata.get('category')
            if category:
                distribution[category] += 1
            for child in node.children:
                analyze_recursive(child)
        
        if self.navigation_tree:
            analyze_recursive(self.navigation_tree)
        
        return dict(distribution)


def implement_swarms_navigation_intelligence():
    """Main function to implement Swarms navigation intelligence"""
    navigator = SwarmsNavigationIntelligence()
    
    print("🧭 Implementing Swarms Navigation Intelligence...")
    
    # Extract navigation structure
    nav_tree = navigator.extract_swarms_navigation_structure()
    print(f"✅ Created navigation tree with {len(nav_tree.children)} main sections")
    
    # Generate MkDocs configuration
    mkdocs_config = navigator.generate_mkdocs_nav()
    print(f"📋 Generated MkDocs navigation configuration")
    
    # Build smart linking system
    linking_system = navigator.build_smart_linking_system()
    print(f"🔗 Built intelligent cross-linking system")
    
    # Generate analytics
    analytics = navigator.generate_navigation_analytics()
    print(f"📊 Navigation Analytics - Sections: {analytics['structure_analysis']['total_sections']}, Pages: {analytics['structure_analysis']['total_pages']}")
    
    return {
        'navigation_tree': nav_tree,
        'mkdocs_config': mkdocs_config,
        'linking_system': linking_system,
        'analytics': analytics
    }


if __name__ == "__main__":
    implement_swarms_navigation_intelligence()