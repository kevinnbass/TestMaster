"""
Swarms Documentation Intelligence Extractor - Phase 1.7 Module 1/4

Extracts and adapts Swarms' advanced documentation intelligence patterns:
- AI-powered content generation
- Intelligent prompt-based documentation
- Multi-format documentation processing
- Automated YAML/Markdown generation
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DocumentationPattern:
    """Represents a Swarms documentation intelligence pattern"""
    name: str
    pattern_type: str
    description: str
    implementation_details: Dict[str, Any]
    usage_examples: List[str]
    ai_prompt_templates: List[str]


class SwarmsDocIntelligenceExtractor:
    """
    Extracts and adapts Swarms' documentation intelligence patterns for TestMaster.
    
    Key Swarms patterns identified:
    1. AI-powered documentation generation with specialized prompts
    2. YAML-based configuration documentation 
    3. Hierarchical MkDocs organization with smart navigation
    4. Auto-generated API documentation with validation
    """
    
    def __init__(self):
        self.patterns: Dict[str, DocumentationPattern] = {}
        self.ai_prompts = self._extract_swarms_prompts()
        self.doc_structures = self._extract_doc_structures()
        
    def _extract_swarms_prompts(self) -> Dict[str, str]:
        """Extract Swarms' AI documentation prompts"""
        return {
            'pytorch_style_docs': """
            Create multi-page long and explicit professional pytorch-like documentation for the {module} code below,
            provide many examples and teach the user about the code, provide examples for every function, make the documentation 10,000 words,
            provide many usage examples and note this is markdown docs, create the documentation for the code to document,
            put the arguments and methods in a table in markdown to make it visually seamless
            
            BE VERY EXPLICIT AND THOROUGH, MAKE IT DEEP AND USEFUL
            """,
            
            'auto_swarm_builder': """
            You are a specialized agent responsible for creating YAML configuration files for multi-agent swarms.
            Your role is to generate well-structured YAML that defines both individual agents and swarm architectures
            based on user requirements. Output only the yaml nothing else.
            """,
            
            'comprehensive_docs': """
            Step 1: Understand the purpose and functionality of the module or framework
            Step 2: Provide an overview and introduction
            Step 3: Provide a class or function definition
            Step 4: Explain the functionality and usage
            Step 5: Provide additional information and tips
            Step 6: Include references and resources
            """
        }
    
    def _extract_doc_structures(self) -> Dict[str, Any]:
        """Extract Swarms' documentation structure patterns"""
        return {
            'mkdocs_nav_pattern': {
                'home_section': ['Overview', 'Installation', 'Quickstart'],
                'agents_section': ['Basic Agents', 'Multi-Agent', 'Tools'],
                'examples_section': ['Basic Examples', 'Advanced Examples', 'Use Cases'],
                'api_section': ['Cloud API', 'Marketplace', 'Clients']
            },
            
            'doc_file_structure': {
                'overview': 'Brief description + architecture diagram',
                'class_reference': 'Constructor + methods with tables',
                'examples': 'Usage examples with full imports',
                'conclusion': 'Benefits and next steps'
            },
            
            'yaml_config_structure': {
                'agents': 'List of agent configurations',
                'swarm_architecture': 'Swarm coordination details',
                'validation_rules': 'Configuration validation'
            }
        }
    
    def extract_ai_generation_pattern(self) -> DocumentationPattern:
        """Extract Swarms' AI-powered documentation generation pattern"""
        return DocumentationPattern(
            name="AI Documentation Generation",
            pattern_type="ai_generation",
            description="Swarms uses specialized AI prompts to generate comprehensive, PyTorch-style documentation",
            implementation_details={
                'prompt_engineering': {
                    'style': 'pytorch_like_documentation',
                    'word_count': 10000,
                    'structure': 'explicit_and_thorough',
                    'format': 'markdown_with_tables'
                },
                'generation_steps': [
                    'analyze_module_purpose',
                    'create_overview_introduction', 
                    'generate_class_definitions',
                    'provide_usage_examples',
                    'add_tips_and_references'
                ]
            },
            usage_examples=[
                "Generate API documentation from code",
                "Create tutorial content automatically",
                "Build comprehensive class references"
            ],
            ai_prompt_templates=list(self.ai_prompts.values())
        )
    
    def extract_yaml_intelligence_pattern(self) -> DocumentationPattern:
        """Extract Swarms' YAML-based configuration intelligence"""
        return DocumentationPattern(
            name="YAML Configuration Intelligence",
            pattern_type="yaml_intelligence", 
            description="Swarms uses intelligent YAML processing for agent and swarm configuration documentation",
            implementation_details={
                'yaml_structure': {
                    'agents': 'agent_name, system_prompt, max_loops, context_length',
                    'swarm_architecture': 'name, swarm_type, description, task',
                    'validation': 'unique_names, positive_integers, specific_formats'
                },
                'intelligent_parsing': {
                    'markdown_extraction': 'regex_pattern_matching',
                    'yaml_normalization': 'spacing_and_formatting_fixes',
                    'validation_rules': 'comprehensive_rule_checking'
                }
            },
            usage_examples=[
                "Auto-generate agent configurations",
                "Validate swarm architecture specs",
                "Create configuration documentation"
            ],
            ai_prompt_templates=[self.ai_prompts['auto_swarm_builder']]
        )
    
    def extract_hierarchical_organization_pattern(self) -> DocumentationPattern:
        """Extract Swarms' hierarchical documentation organization"""
        return DocumentationPattern(
            name="Hierarchical Documentation Organization", 
            pattern_type="hierarchical_structure",
            description="Swarms uses intelligent MkDocs-based hierarchical organization with smart navigation",
            implementation_details={
                'navigation_structure': self.doc_structures['mkdocs_nav_pattern'],
                'file_organization': {
                    'by_functionality': 'agents, tools, examples, deployment',
                    'by_complexity': 'basic, advanced, enterprise',
                    'by_use_case': 'finance, medical, research, general'
                },
                'smart_features': {
                    'search_integration': 'mkdocs_search_plugin',
                    'git_integration': 'git_authors_and_committers',
                    'auto_navigation': 'yaml_based_nav_generation'
                }
            },
            usage_examples=[
                "Organize documentation by user journey",
                "Create intelligent navigation systems", 
                "Build scalable doc architectures"
            ],
            ai_prompt_templates=[self.ai_prompts['comprehensive_docs']]
        )
    
    def generate_testmaster_integration_plan(self) -> Dict[str, Any]:
        """Generate integration plan for TestMaster based on extracted patterns"""
        integration_plan = {
            'phase_1_7_implementation': {
                'ai_doc_generation': {
                    'integrate_pytorch_style_prompts': True,
                    'create_testmaster_doc_templates': True,
                    'implement_auto_api_docs': True
                },
                'yaml_intelligence': {
                    'adapt_yaml_processing': True,
                    'create_config_validation': True,
                    'build_intelligent_parsing': True
                },
                'hierarchical_organization': {
                    'implement_mkdocs_structure': True,
                    'create_smart_navigation': True,
                    'build_scalable_architecture': True
                }
            },
            'extracted_patterns': [
                self.extract_ai_generation_pattern(),
                self.extract_yaml_intelligence_pattern(), 
                self.extract_hierarchical_organization_pattern()
            ],
            'next_steps': [
                'Implement AI documentation generator',
                'Create YAML configuration processor',
                'Build hierarchical doc organizer',
                'Integrate with existing TestMaster intelligence'
            ]
        }
        
        return integration_plan
    
    def save_extracted_patterns(self, output_path: str = "swarms_patterns_extracted.json") -> str:
        """Save extracted patterns to file for Phase 2 integration"""
        patterns_data = {
            'extraction_timestamp': datetime.now().isoformat(),
            'swarms_patterns_extracted': self.generate_testmaster_integration_plan(),
            'ai_prompts': self.ai_prompts,
            'doc_structures': self.doc_structures
        }
        
        with open(output_path, 'w') as f:
            json.dump(patterns_data, f, indent=2, default=str)
            
        return f"Swarms documentation intelligence patterns extracted and saved to {output_path}"


def extract_swarms_intelligence():
    """Main extraction function for Phase 1.7"""
    extractor = SwarmsDocIntelligenceExtractor()
    
    print("🔍 Extracting Swarms Documentation Intelligence Patterns...")
    
    # Extract all patterns
    ai_pattern = extractor.extract_ai_generation_pattern()
    yaml_pattern = extractor.extract_yaml_intelligence_pattern()
    hierarchical_pattern = extractor.extract_hierarchical_organization_pattern()
    
    print(f"✅ Extracted AI Generation Pattern: {ai_pattern.name}")
    print(f"✅ Extracted YAML Intelligence Pattern: {yaml_pattern.name}")
    print(f"✅ Extracted Hierarchical Organization Pattern: {hierarchical_pattern.name}")
    
    # Generate integration plan
    integration_plan = extractor.generate_testmaster_integration_plan()
    
    # Save patterns for Phase 2
    result = extractor.save_extracted_patterns()
    print(f"💾 {result}")
    
    return integration_plan


if __name__ == "__main__":
    extract_swarms_intelligence()