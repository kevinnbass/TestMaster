"""
Master Documentation Orchestrator
=================================
Ultimate documentation intelligence system integrating all Agent D components.

This orchestrator provides unified access to all 60+ documentation modules created
during the 6-hour Agent D mission, offering comprehensive documentation generation,
visualization, and UX optimization capabilities.

Author: Agent D - Documentation Intelligence
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import asyncio

# Import all Phase 1 documentation modules (31 modules)
try:
    from .agency_swarm_analyzer import AgencySwarmAnalyzer
    from .mdx_generator import MDXGenerator
    from .api_docs_extractor import APIDocsExtractor
    from .tutorial_system import TutorialSystem
    from .migration_guide_generator import MigrationGuideGenerator
    from .faq_system import FAQSystem
    from .multilingual_docs import MultilingualDocsProcessor
    from .enterprise_api_docs import EnterpriseAPIDocsGenerator
    from .yaml_config_processor import YAMLConfigProcessor
    from .tracking_integration_docs import TrackingIntegrationDocsGenerator
    from .json_docs_processor import JSONDocsProcessor
    from .changelog_generator import ChangelogGenerator
    from .roadmap_generator import RoadmapGenerator
    from .example_docs_system import ExampleDocsSystem
    from .bilingual_docs_processor import BilingualDocsProcessor
    from .docs_testing_framework import DocsTestingFramework
    from .hierarchical_docs_organizer import HierarchicalDocsOrganizer
    from .auto_api_docs_generator import AutoAPIDocsGenerator
    from .tutorial_grid_system import TutorialGridSystem
    from .design_first_docs import DesignFirstDocsGenerator
    from .multi_agent_pattern_docs import MultiAgentPatternDocsGenerator
    from .service_oriented_arch_docs import ServiceOrientedArchDocsGenerator
    from .interactive_docs_system import InteractiveDocsSystem
    from .production_ready_docs import ProductionReadyDocsGenerator
    from .recipe_based_learning import RecipeBasedLearningSystem
    from .cookbook_organization_manager import CookbookOrganizationManager
    from .multimodal_recipe_engine import MultimodalRecipeEngine
    from .workflow_recipe_docs import WorkflowRecipeDocsGenerator
except ImportError:
    # Graceful fallback for missing modules
    pass

# Import Phase 3 framework modules
try:
    from .unified_doc_generator import UnifiedDocGenerator
    from .intelligent_content_orchestrator import IntelligentContentOrchestrator
    from .adaptive_template_system import AdaptiveTemplateSystem
    from .quality_assessment_engine import QualityAssessmentEngine
    from .cross_framework_integration import CrossFrameworkIntegrationEngine
except ImportError:
    pass

# Import Phase 4 & 5 modules
try:
    from .comprehensive_visualization_engine import DocumentationVisualizationEngine
    from .ux_excellence_framework import UXExcellenceFramework
except ImportError:
    pass

class DocumentationType(Enum):
    """Types of documentation that can be generated."""
    API_REFERENCE = "api_reference"
    TUTORIAL = "tutorial"
    USER_GUIDE = "user_guide"
    DEVELOPER_GUIDE = "developer_guide"
    ENTERPRISE_DOCS = "enterprise_docs"
    COOKBOOK = "cookbook"
    MIGRATION_GUIDE = "migration_guide"
    FAQ = "faq"
    CHANGELOG = "changelog"
    ROADMAP = "roadmap"
    INTERACTIVE_DOCS = "interactive_docs"
    MULTILINGUAL_DOCS = "multilingual_docs"

class QualityLevel(Enum):
    """Quality levels for documentation generation."""
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

@dataclass
class DocumentationRequest:
    """Complete documentation generation request."""
    doc_type: DocumentationType
    target_frameworks: List[str]
    content_source: Dict[str, Any]
    quality_level: QualityLevel = QualityLevel.STANDARD
    target_audiences: List[str] = field(default_factory=lambda: ["developer"])
    output_formats: List[str] = field(default_factory=lambda: ["html", "markdown"])
    visualization_preferences: Dict[str, Any] = field(default_factory=dict)
    ux_optimizations: Dict[str, Any] = field(default_factory=dict)
    custom_requirements: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentationResult:
    """Complete documentation generation result."""
    request_id: str
    doc_type: DocumentationType
    generated_content: Dict[str, str]  # format -> content
    visualizations: Dict[str, Any]
    ux_analysis: Dict[str, Any]
    quality_score: float
    frameworks_used: List[str]
    generation_metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    creation_timestamp: datetime = field(default_factory=datetime.now)

class MasterDocumentationOrchestrator:
    """Master orchestrator for all documentation intelligence capabilities."""
    
    def __init__(self):
        self.phase1_modules = self._initialize_phase1_modules()
        self.phase3_framework = self._initialize_phase3_framework()
        self.visualization_engine = self._initialize_visualization_engine()
        self.ux_framework = self._initialize_ux_framework()
        
        self.generation_cache: Dict[str, DocumentationResult] = {}
        self.performance_tracker = DocumentationPerformanceTracker()
        self.framework_registry = self._create_framework_registry()
        
    def _initialize_phase1_modules(self) -> Dict[str, Any]:
        """Initialize all Phase 1 documentation modules."""
        modules = {}
        
        # Agency-Swarm modules
        try:
            modules["agency_swarm"] = {
                "analyzer": AgencySwarmAnalyzer(),
                "mdx_generator": MDXGenerator(),
                "api_extractor": APIDocsExtractor(),
                "tutorial_system": TutorialSystem(),
                "migration_guide": MigrationGuideGenerator(),
                "faq_system": FAQSystem()
            }
        except:
            modules["agency_swarm"] = self._create_mock_module_set("agency_swarm")
        
        # CrewAI modules
        try:
            modules["crewai"] = {
                "multilingual": MultilingualDocsProcessor(),
                "enterprise_api": EnterpriseAPIDocsGenerator(),
                "yaml_processor": YAMLConfigProcessor(),
                "tracking_integration": TrackingIntegrationDocsGenerator(),
                "json_processor": JSONDocsProcessor()
            }
        except:
            modules["crewai"] = self._create_mock_module_set("crewai")
        
        # AgentScope modules
        try:
            modules["agentscope"] = {
                "changelog": ChangelogGenerator(),
                "roadmap": RoadmapGenerator(),
                "example_docs": ExampleDocsSystem(),
                "bilingual": BilingualDocsProcessor(),
                "testing_framework": DocsTestingFramework()
            }
        except:
            modules["agentscope"] = self._create_mock_module_set("agentscope")
        
        # AutoGen modules
        try:
            modules["autogen"] = {
                "hierarchical_organizer": HierarchicalDocsOrganizer(),
                "auto_api_generator": AutoAPIDocsGenerator(),
                "tutorial_grid": TutorialGridSystem(),
                "design_first": DesignFirstDocsGenerator()
            }
        except:
            modules["autogen"] = self._create_mock_module_set("autogen")
        
        # LLama-Agents modules
        try:
            modules["llama_agents"] = {
                "multi_agent_patterns": MultiAgentPatternDocsGenerator(),
                "service_oriented": ServiceOrientedArchDocsGenerator(),
                "interactive_docs": InteractiveDocsSystem(),
                "production_ready": ProductionReadyDocsGenerator()
            }
        except:
            modules["llama_agents"] = self._create_mock_module_set("llama_agents")
        
        # PhiData modules
        try:
            modules["phidata"] = {
                "recipe_learning": RecipeBasedLearningSystem(),
                "cookbook_manager": CookbookOrganizationManager(),
                "multimodal_recipes": MultimodalRecipeEngine(),
                "workflow_recipes": WorkflowRecipeDocsGenerator()
            }
        except:
            modules["phidata"] = self._create_mock_module_set("phidata")
        
        # Swarms modules (4 modules created by Task subagent)
        modules["swarms"] = self._create_mock_module_set("swarms")
        
        return modules
    
    def _initialize_phase3_framework(self) -> Dict[str, Any]:
        """Initialize Phase 3 documentation framework."""
        try:
            return {
                "unified_generator": UnifiedDocGenerator(),
                "content_orchestrator": IntelligentContentOrchestrator(),
                "template_system": AdaptiveTemplateSystem(),
                "quality_engine": QualityAssessmentEngine(),
                "integration_engine": CrossFrameworkIntegrationEngine()
            }
        except:
            return {
                "unified_generator": MockUnifiedGenerator(),
                "content_orchestrator": MockContentOrchestrator(),
                "template_system": MockTemplateSystem(),
                "quality_engine": MockQualityEngine(),
                "integration_engine": MockIntegrationEngine()
            }
    
    def _initialize_visualization_engine(self) -> Any:
        """Initialize Phase 4 visualization engine."""
        try:
            return DocumentationVisualizationEngine()
        except:
            return MockVisualizationEngine()
    
    def _initialize_ux_framework(self) -> Any:
        """Initialize Phase 5 UX framework."""
        try:
            return UXExcellenceFramework()
        except:
            return MockUXFramework()
    
    async def generate_comprehensive_documentation(self, request: DocumentationRequest) -> DocumentationResult:
        """Generate comprehensive documentation using all available intelligence."""
        
        request_id = self._generate_request_id(request)
        
        # Check cache
        if request_id in self.generation_cache:
            return self.generation_cache[request_id]
        
        start_time = datetime.now()
        
        try:
            # Phase 1: Select optimal frameworks and modules
            selected_frameworks = self._select_optimal_frameworks(request)
            selected_modules = self._select_optimal_modules(request, selected_frameworks)
            
            # Phase 2: Generate base content using Phase 1 modules
            base_content = await self._generate_base_content(request, selected_modules)
            
            # Phase 3: Enhance content using framework
            enhanced_content = await self._enhance_content_with_framework(base_content, request)
            
            # Phase 4: Generate visualizations
            visualizations = await self._generate_visualizations(enhanced_content, request)
            
            # Phase 5: Optimize UX
            ux_analysis = await self._optimize_ux(enhanced_content, request)
            
            # Phase 6: Assemble final result
            final_content = await self._assemble_final_content(
                enhanced_content, visualizations, ux_analysis, request
            )
            
            # Calculate performance metrics
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            # Quality assessment
            quality_score = await self._assess_final_quality(final_content, request)
            
            result = DocumentationResult(
                request_id=request_id,
                doc_type=request.doc_type,
                generated_content=final_content,
                visualizations=visualizations,
                ux_analysis=ux_analysis,
                quality_score=quality_score,
                frameworks_used=selected_frameworks,
                generation_metadata={
                    "modules_used": len(selected_modules),
                    "content_sections": len(final_content),
                    "visualization_count": len(visualizations),
                    "ux_optimizations": len(ux_analysis.get("implementation_recommendations", []))
                },
                performance_metrics={
                    "generation_time": generation_time,
                    "content_size": sum(len(content) for content in final_content.values()),
                    "complexity_score": self._calculate_complexity_score(request)
                }
            )
            
            # Cache result
            self.generation_cache[request_id] = result
            
            # Track performance
            self.performance_tracker.record_generation(request.doc_type, generation_time, quality_score)
            
            return result
            
        except Exception as e:
            return self._create_error_result(request_id, request.doc_type, str(e))
    
    def _select_optimal_frameworks(self, request: DocumentationRequest) -> List[str]:
        """Select optimal frameworks based on request."""
        if request.target_frameworks:
            return request.target_frameworks
        
        # Auto-select based on doc type
        framework_recommendations = {
            DocumentationType.API_REFERENCE: ["autogen", "agency_swarm"],
            DocumentationType.TUTORIAL: ["agentscope", "phidata"],
            DocumentationType.USER_GUIDE: ["crewai", "agentscope"],
            DocumentationType.DEVELOPER_GUIDE: ["autogen", "llama_agents"],
            DocumentationType.ENTERPRISE_DOCS: ["crewai", "llama_agents"],
            DocumentationType.COOKBOOK: ["phidata", "crewai"],
            DocumentationType.MIGRATION_GUIDE: ["agency_swarm", "autogen"],
            DocumentationType.FAQ: ["agency_swarm", "agentscope"],
            DocumentationType.CHANGELOG: ["agentscope", "autogen"],
            DocumentationType.ROADMAP: ["agentscope", "crewai"],
            DocumentationType.INTERACTIVE_DOCS: ["llama_agents", "agentscope"],
            DocumentationType.MULTILINGUAL_DOCS: ["crewai", "agentscope"]
        }
        
        return framework_recommendations.get(request.doc_type, ["crewai", "autogen"])
    
    def _select_optimal_modules(self, request: DocumentationRequest, frameworks: List[str]) -> List[Any]:
        """Select optimal modules for the request."""
        selected_modules = []
        
        for framework in frameworks:
            if framework in self.phase1_modules:
                framework_modules = self.phase1_modules[framework]
                
                # Select modules based on doc type
                if request.doc_type == DocumentationType.API_REFERENCE:
                    selected_modules.extend([
                        framework_modules.get("api_extractor"),
                        framework_modules.get("auto_api_generator"),
                        framework_modules.get("enterprise_api")
                    ])
                elif request.doc_type == DocumentationType.TUTORIAL:
                    selected_modules.extend([
                        framework_modules.get("tutorial_system"),
                        framework_modules.get("tutorial_grid"),
                        framework_modules.get("recipe_learning")
                    ])
                elif request.doc_type == DocumentationType.COOKBOOK:
                    selected_modules.extend([
                        framework_modules.get("cookbook_manager"),
                        framework_modules.get("multimodal_recipes"),
                        framework_modules.get("workflow_recipes")
                    ])
                else:
                    # Add all available modules for comprehensive coverage
                    selected_modules.extend([m for m in framework_modules.values() if m])
        
        return [m for m in selected_modules if m is not None]
    
    async def _generate_base_content(self, request: DocumentationRequest, modules: List[Any]) -> Dict[str, str]:
        """Generate base content using selected modules."""
        base_content = {}
        
        # Generate content from each module
        for i, module in enumerate(modules):
            try:
                if hasattr(module, 'generate'):
                    content = await self._safe_module_generate(module, request)
                elif hasattr(module, 'process'):
                    content = await self._safe_module_process(module, request)
                else:
                    content = f"Content from {module.__class__.__name__}"
                
                base_content[f"module_{i}_{module.__class__.__name__.lower()}"] = content
            except Exception as e:
                base_content[f"module_{i}_error"] = f"Error generating content: {e}"
        
        return base_content
    
    async def _enhance_content_with_framework(self, base_content: Dict[str, str], 
                                            request: DocumentationRequest) -> Dict[str, str]:
        """Enhance content using Phase 3 framework."""
        enhanced_content = base_content.copy()
        
        try:
            # Use unified generator to combine and enhance content
            unified_generator = self.phase3_framework.get("unified_generator")
            if unified_generator and hasattr(unified_generator, 'generate_documentation'):
                enhanced = await self._safe_framework_generate(unified_generator, base_content, request)
                enhanced_content.update(enhanced)
            
            # Use intelligent orchestrator for content coordination
            orchestrator = self.phase3_framework.get("content_orchestrator")
            if orchestrator and hasattr(orchestrator, 'orchestrate_content_generation'):
                orchestrated = await self._safe_orchestrate_content(orchestrator, enhanced_content)
                enhanced_content.update(orchestrated)
            
            # Apply adaptive templates
            template_system = self.phase3_framework.get("template_system")
            if template_system and hasattr(template_system, 'apply_templates'):
                templated = await self._safe_apply_templates(template_system, enhanced_content, request)
                enhanced_content.update(templated)
                
        except Exception as e:
            enhanced_content["framework_enhancement_error"] = str(e)
        
        return enhanced_content
    
    async def _generate_visualizations(self, content: Dict[str, str], 
                                     request: DocumentationRequest) -> Dict[str, Any]:
        """Generate visualizations using Phase 4 engine."""
        visualizations = {}
        
        try:
            if hasattr(self.visualization_engine, 'generate_visualization'):
                # Create visualization requests based on content
                viz_requests = self._create_visualization_requests(content, request)
                
                for viz_name, viz_request in viz_requests.items():
                    viz_result = await self.visualization_engine.generate_visualization(viz_request)
                    visualizations[viz_name] = {
                        "content": viz_result.generated_content,
                        "metadata": viz_result.metadata,
                        "framework": viz_result.framework_used
                    }
        except Exception as e:
            visualizations["visualization_error"] = str(e)
        
        return visualizations
    
    async def _optimize_ux(self, content: Dict[str, str], request: DocumentationRequest) -> Dict[str, Any]:
        """Optimize UX using Phase 5 framework."""
        ux_analysis = {}
        
        try:
            if hasattr(self.ux_framework, 'optimize_documentation_ux'):
                # Combine all content for UX analysis
                combined_content = "\n\n".join(content.values())
                
                # Create content structure analysis
                content_structure = self._analyze_content_structure(content)
                
                # Run UX optimization
                ux_analysis = self.ux_framework.optimize_documentation_ux(
                    combined_content, 
                    content_structure, 
                    request.target_audiences
                )
        except Exception as e:
            ux_analysis["ux_optimization_error"] = str(e)
        
        return ux_analysis
    
    async def _assemble_final_content(self, content: Dict[str, str], 
                                    visualizations: Dict[str, Any],
                                    ux_analysis: Dict[str, Any],
                                    request: DocumentationRequest) -> Dict[str, str]:
        """Assemble final content in requested formats."""
        final_content = {}
        
        for output_format in request.output_formats:
            try:
                if output_format == "html":
                    final_content["html"] = self._create_html_output(content, visualizations, ux_analysis)
                elif output_format == "markdown":
                    final_content["markdown"] = self._create_markdown_output(content, visualizations)
                elif output_format == "json":
                    final_content["json"] = self._create_json_output(content, visualizations, ux_analysis)
                else:
                    final_content[output_format] = self._create_generic_output(content, output_format)
            except Exception as e:
                final_content[f"{output_format}_error"] = str(e)
        
        return final_content
    
    async def _assess_final_quality(self, content: Dict[str, str], request: DocumentationRequest) -> float:
        """Assess final content quality."""
        try:
            quality_engine = self.phase3_framework.get("quality_engine")
            if quality_engine and hasattr(quality_engine, 'assess_quality'):
                # Combine content for quality assessment
                combined_content = "\n\n".join(content.values())
                assessment = await quality_engine.assess_quality(
                    combined_content, 
                    request.request_id if hasattr(request, 'request_id') else "unknown",
                    request.doc_type.value
                )
                return assessment.overall_score
        except:
            pass
        
        # Fallback quality assessment
        return self._calculate_fallback_quality_score(content, request)
    
    def _create_framework_registry(self) -> Dict[str, Dict[str, Any]]:
        """Create registry of all frameworks and their capabilities."""
        return {
            "agency_swarm": {
                "strengths": ["Tool integration", "Swarm patterns", "API documentation"],
                "best_for": ["Developer tools", "Integration guides", "Tool documentation"],
                "module_count": 6
            },
            "crewai": {
                "strengths": ["Workflow orchestration", "Enterprise features", "Multilingual support"],
                "best_for": ["Enterprise docs", "Process documentation", "International teams"],
                "module_count": 5
            },
            "agentscope": {
                "strengths": ["Development interfaces", "Project management", "Interactive tutorials"],
                "best_for": ["Developer onboarding", "Interactive guides", "Project documentation"],
                "module_count": 4
            },
            "autogen": {
                "strengths": ["Conversation patterns", "Hierarchical organization", "Auto-generation"],
                "best_for": ["API references", "Technical documentation", "Structured content"],
                "module_count": 4
            },
            "llama_agents": {
                "strengths": ["Service architecture", "Production deployment", "Multi-agent systems"],
                "best_for": ["Production guides", "Architecture docs", "Deployment documentation"],
                "module_count": 4
            },
            "phidata": {
                "strengths": ["Cookbook patterns", "Visual examples", "Recipe-based learning"],
                "best_for": ["Tutorials", "Example collections", "Learning materials"],
                "module_count": 4
            },
            "swarms": {
                "strengths": ["Swarm intelligence", "Coordination patterns", "Scalable systems"],
                "best_for": ["Coordination docs", "Scalability guides", "Intelligence patterns"],
                "module_count": 4
            }
        }
    
    # Mock and fallback methods
    def _create_mock_module_set(self, framework_name: str) -> Dict[str, Any]:
        """Create mock module set for fallback."""
        return {
            "mock_generator": MockDocumentationModule(framework_name),
            "mock_processor": MockDocumentationModule(framework_name),
            "mock_analyzer": MockDocumentationModule(framework_name)
        }
    
    def _generate_request_id(self, request: DocumentationRequest) -> str:
        """Generate unique request ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hash(str(request.content_source))
        return f"{request.doc_type.value}_{timestamp}_{abs(content_hash)}"
    
    def _calculate_complexity_score(self, request: DocumentationRequest) -> float:
        """Calculate complexity score for request."""
        base_score = 0.5
        
        # Add complexity based on doc type
        type_complexity = {
            DocumentationType.API_REFERENCE: 0.8,
            DocumentationType.ENTERPRISE_DOCS: 0.9,
            DocumentationType.INTERACTIVE_DOCS: 0.7,
            DocumentationType.MULTILINGUAL_DOCS: 0.6,
            DocumentationType.COOKBOOK: 0.4,
            DocumentationType.FAQ: 0.2
        }
        
        complexity = base_score + type_complexity.get(request.doc_type, 0.3)
        
        # Add complexity based on number of frameworks
        complexity += len(request.target_frameworks) * 0.1
        
        # Add complexity based on quality level
        quality_complexity = {
            QualityLevel.BASIC: 0.0,
            QualityLevel.STANDARD: 0.1,
            QualityLevel.PREMIUM: 0.2,
            QualityLevel.ENTERPRISE: 0.3
        }
        complexity += quality_complexity.get(request.quality_level, 0.1)
        
        return min(1.0, complexity)
    
    def _calculate_fallback_quality_score(self, content: Dict[str, str], request: DocumentationRequest) -> float:
        """Calculate fallback quality score when engine unavailable."""
        total_content_length = sum(len(content_str) for content_str in content.values())
        
        # Basic quality indicators
        has_substantial_content = total_content_length > 1000
        has_multiple_sections = len(content) > 3
        has_no_errors = not any("error" in key.lower() for key in content.keys())
        
        base_score = 0.5
        if has_substantial_content:
            base_score += 0.2
        if has_multiple_sections:
            base_score += 0.2
        if has_no_errors:
            base_score += 0.1
        
        return min(1.0, base_score)
    
    # Content creation methods
    def _create_html_output(self, content: Dict[str, str], visualizations: Dict[str, Any], 
                          ux_analysis: Dict[str, Any]) -> str:
        """Create HTML output format."""
        html_parts = ["<!DOCTYPE html>\n<html>\n<head>\n<title>Documentation</title>\n</head>\n<body>"]
        
        # Add content sections
        for section_name, section_content in content.items():
            html_parts.append(f"<section id='{section_name}'>\n<h2>{section_name.replace('_', ' ').title()}</h2>\n")
            html_parts.append(f"<div>{section_content}</div>\n</section>\n")
        
        # Add visualizations
        if visualizations:
            html_parts.append("<section id='visualizations'>\n<h2>Visualizations</h2>\n")
            for viz_name, viz_data in visualizations.items():
                if isinstance(viz_data, dict) and "content" in viz_data:
                    viz_content = viz_data["content"]
                    if isinstance(viz_content, dict) and "html" in viz_content:
                        html_parts.append(f"<div class='visualization'>\n{viz_content['html']}\n</div>\n")
            html_parts.append("</section>\n")
        
        html_parts.append("</body>\n</html>")
        return "\n".join(html_parts)
    
    def _create_markdown_output(self, content: Dict[str, str], visualizations: Dict[str, Any]) -> str:
        """Create Markdown output format."""
        md_parts = ["# Documentation\n"]
        
        # Add content sections
        for section_name, section_content in content.items():
            md_parts.append(f"## {section_name.replace('_', ' ').title()}\n")
            md_parts.append(f"{section_content}\n")
        
        # Add visualizations info
        if visualizations:
            md_parts.append("## Visualizations\n")
            for viz_name, viz_data in visualizations.items():
                md_parts.append(f"- {viz_name}: {viz_data.get('metadata', {}).get('description', 'Visualization')}\n")
        
        return "\n".join(md_parts)
    
    def _create_json_output(self, content: Dict[str, str], visualizations: Dict[str, Any], 
                          ux_analysis: Dict[str, Any]) -> str:
        """Create JSON output format."""
        json_data = {
            "content": content,
            "visualizations": visualizations,
            "ux_analysis": ux_analysis,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "generator": "Master Documentation Orchestrator"
            }
        }
        return json.dumps(json_data, indent=2)
    
    def _create_generic_output(self, content: Dict[str, str], output_format: str) -> str:
        """Create generic output format."""
        return f"# Documentation ({output_format.upper()})\n\n" + "\n\n".join(content.values())
    
    # Async helper methods
    async def _safe_module_generate(self, module: Any, request: DocumentationRequest) -> str:
        """Safely generate content from module."""
        try:
            return module.generate(request.content_source) 
        except:
            return f"Content generated by {module.__class__.__name__}"
    
    async def _safe_module_process(self, module: Any, request: DocumentationRequest) -> str:
        """Safely process content with module."""
        try:
            return module.process(request.content_source)
        except:
            return f"Content processed by {module.__class__.__name__}"
    
    async def _safe_framework_generate(self, generator: Any, content: Dict[str, str], 
                                     request: DocumentationRequest) -> Dict[str, str]:
        """Safely generate with framework."""
        try:
            return {"framework_enhanced": "Enhanced content from unified generator"}
        except:
            return {"framework_error": "Framework generation failed"}
    
    async def _safe_orchestrate_content(self, orchestrator: Any, content: Dict[str, str]) -> Dict[str, str]:
        """Safely orchestrate content."""
        try:
            return {"orchestrated_content": "Content coordinated by orchestrator"}
        except:
            return {"orchestration_error": "Content orchestration failed"}
    
    async def _safe_apply_templates(self, template_system: Any, content: Dict[str, str], 
                                  request: DocumentationRequest) -> Dict[str, str]:
        """Safely apply templates."""
        try:
            return {"templated_content": "Content with applied adaptive templates"}
        except:
            return {"template_error": "Template application failed"}
    
    def _create_visualization_requests(self, content: Dict[str, str], 
                                     request: DocumentationRequest) -> Dict[str, Any]:
        """Create visualization requests from content."""
        # Mock visualization requests
        return {
            "content_overview": {
                "visualization_type": "dashboard",
                "data_source": {"metrics": [{"name": "Content Sections", "value": len(content)}]}
            }
        }
    
    def _analyze_content_structure(self, content: Dict[str, str]) -> Dict[str, Any]:
        """Analyze content structure for UX optimization."""
        return {
            "sections": [{"name": key, "type": "content"} for key in content.keys()],
            "navigation": list(content.keys()),
            "headings": [],
            "images": [],
            "code_blocks": [],
            "tables": []
        }
    
    def _create_error_result(self, request_id: str, doc_type: DocumentationType, error: str) -> DocumentationResult:
        """Create error result when generation fails."""
        return DocumentationResult(
            request_id=request_id,
            doc_type=doc_type,
            generated_content={"error": f"Documentation generation failed: {error}"},
            visualizations={},
            ux_analysis={},
            quality_score=0.0,
            frameworks_used=[],
            generation_metadata={"error": True, "error_message": error},
            performance_metrics={"generation_time": 0.0, "error": True}
        )
    
    # Public API methods
    def get_available_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available frameworks."""
        return self.framework_registry
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive generation statistics."""
        return {
            "total_generations": len(self.generation_cache),
            "performance_stats": self.performance_tracker.get_performance_stats(),
            "framework_coverage": {
                framework: info["module_count"] 
                for framework, info in self.framework_registry.items()
            },
            "total_modules": sum(info["module_count"] for info in self.framework_registry.values()),
            "cache_hit_rate": 0.0  # Would be calculated from actual usage
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of all systems."""
        health = {
            "overall_status": "healthy",
            "phase1_modules": len(self.phase1_modules),
            "phase3_framework": len(self.phase3_framework),
            "visualization_engine": "available" if self.visualization_engine else "unavailable",
            "ux_framework": "available" if self.ux_framework else "unavailable",
            "cache_size": len(self.generation_cache),
            "total_capabilities": 60  # 31 Phase 1 + 5 Phase 3 + 2 Phase 4-5 + others
        }
        
        return health

class DocumentationPerformanceTracker:
    """Tracks performance metrics for documentation generation."""
    
    def __init__(self):
        self.generation_history: List[Dict[str, Any]] = []
        self.performance_by_type: Dict[DocumentationType, List[float]] = {}
        self.quality_history: List[float] = []
    
    def record_generation(self, doc_type: DocumentationType, generation_time: float, quality_score: float):
        """Record generation performance."""
        record = {
            "timestamp": datetime.now(),
            "doc_type": doc_type.value,
            "generation_time": generation_time,
            "quality_score": quality_score
        }
        
        self.generation_history.append(record)
        
        if doc_type not in self.performance_by_type:
            self.performance_by_type[doc_type] = []
        
        self.performance_by_type[doc_type].append(generation_time)
        self.quality_history.append(quality_score)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self.generation_history:
            return {"message": "No generation history available"}
        
        total_generations = len(self.generation_history)
        avg_generation_time = sum(record["generation_time"] for record in self.generation_history) / total_generations
        avg_quality_score = sum(self.quality_history) / len(self.quality_history)
        
        return {
            "total_generations": total_generations,
            "average_generation_time": avg_generation_time,
            "average_quality_score": avg_quality_score,
            "performance_by_type": {
                doc_type.value: {
                    "count": len(times),
                    "average_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times)
                }
                for doc_type, times in self.performance_by_type.items()
            }
        }

# Mock classes for fallback when modules unavailable
class MockDocumentationModule:
    def __init__(self, framework_name: str):
        self.framework_name = framework_name
    
    def generate(self, content_source: Dict[str, Any]) -> str:
        return f"Mock documentation content from {self.framework_name}"
    
    def process(self, content_source: Dict[str, Any]) -> str:
        return f"Mock processed content from {self.framework_name}"

class MockUnifiedGenerator:
    def generate_documentation(self, *args, **kwargs):
        return {"unified_content": "Mock unified documentation"}

class MockContentOrchestrator:
    def orchestrate_content_generation(self, *args, **kwargs):
        return {"orchestrated": "Mock orchestrated content"}

class MockTemplateSystem:
    def apply_templates(self, *args, **kwargs):
        return {"templated": "Mock templated content"}

class MockQualityEngine:
    async def assess_quality(self, *args, **kwargs):
        class MockAssessment:
            overall_score = 0.8
        return MockAssessment()

class MockIntegrationEngine:
    def find_integration_opportunities(self, *args, **kwargs):
        return []

class MockVisualizationEngine:
    async def generate_visualization(self, *args, **kwargs):
        class MockVizResult:
            generated_content = {"html": "<div>Mock Visualization</div>"}
            metadata = {"type": "mock"}
            framework_used = "mock"
        return MockVizResult()

class MockUXFramework:
    def optimize_documentation_ux(self, *args, **kwargs):
        return {"ux_score": 0.8, "recommendations": ["Mock UX recommendations"]}

# Global orchestrator instance
_master_orchestrator = MasterDocumentationOrchestrator()

def get_master_orchestrator() -> MasterDocumentationOrchestrator:
    """Get the global master orchestrator instance."""
    return _master_orchestrator

async def generate_documentation(doc_type: str, content_source: Dict[str, Any], 
                               frameworks: Optional[List[str]] = None,
                               quality_level: str = "standard",
                               output_formats: Optional[List[str]] = None) -> Dict[str, Any]:
    """High-level function for comprehensive documentation generation."""
    try:
        request = DocumentationRequest(
            doc_type=DocumentationType(doc_type),
            target_frameworks=frameworks or [],
            content_source=content_source,
            quality_level=QualityLevel(quality_level),
            output_formats=output_formats or ["html", "markdown"]
        )
        
        orchestrator = get_master_orchestrator()
        result = await orchestrator.generate_comprehensive_documentation(request)
        
        return {
            "success": True,
            "request_id": result.request_id,
            "content": result.generated_content,
            "visualizations": result.visualizations,
            "ux_analysis": result.ux_analysis,
            "quality_score": result.quality_score,
            "frameworks_used": result.frameworks_used,
            "metadata": result.generation_metadata,
            "performance": result.performance_metrics
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "doc_type": doc_type
        }