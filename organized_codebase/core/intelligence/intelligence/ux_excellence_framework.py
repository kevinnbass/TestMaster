"""
UX Excellence Framework
=====================
Advanced user experience optimization for documentation systems.

This framework synthesizes UX best practices from all 7 frameworks to create
exceptional documentation experiences with accessibility, responsiveness,
and intelligent user journey optimization.

Author: Agent D - Documentation Intelligence
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import re

class UXMetric(Enum):
    """User experience metrics for measurement."""
    ACCESSIBILITY = "accessibility"
    RESPONSIVENESS = "responsiveness"
    USABILITY = "usability"
    FINDABILITY = "findability"
    READABILITY = "readability"
    ENGAGEMENT = "engagement"
    EFFICIENCY = "efficiency"
    SATISFACTION = "satisfaction"

class UserType(Enum):
    """Different user types for UX optimization."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate" 
    EXPERT = "expert"
    ENTERPRISE = "enterprise"
    DEVELOPER = "developer"
    RESEARCHER = "researcher"

class DeviceType(Enum):
    """Device types for responsive design."""
    MOBILE = "mobile"
    TABLET = "tablet"
    DESKTOP = "desktop"
    LARGE_SCREEN = "large_screen"

@dataclass
class UserPersona:
    """User persona for UX optimization."""
    user_type: UserType
    primary_goals: List[str]
    pain_points: List[str]
    preferred_learning_style: str
    technical_proficiency: str
    context_usage: List[str]  # where/when they use docs
    success_metrics: List[str]

@dataclass
class UXOptimization:
    """UX optimization recommendation."""
    metric: UXMetric
    current_score: float
    target_score: float
    improvements: List[str]
    implementation_effort: str  # low, medium, high
    expected_impact: str  # low, medium, high
    user_types_benefiting: List[UserType]
    framework_patterns: List[str]  # which frameworks inspired this

@dataclass
class ResponsiveDesign:
    """Responsive design configuration."""
    breakpoints: Dict[str, int]
    layout_adaptations: Dict[DeviceType, Dict[str, Any]]
    typography_scaling: Dict[DeviceType, Dict[str, Any]]
    navigation_patterns: Dict[DeviceType, str]
    interaction_methods: Dict[DeviceType, List[str]]

class AccessibilityAnalyzer:
    """Analyzes and optimizes accessibility in documentation."""
    
    def __init__(self):
        self.wcag_guidelines = self._load_wcag_guidelines()
        self.accessibility_patterns = self._load_accessibility_patterns()
    
    def analyze_accessibility(self, content: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive accessibility analysis."""
        analysis_results = {
            "overall_score": 0.0,
            "compliance_level": "none",  # A, AA, AAA
            "violations": [],
            "recommendations": [],
            "automatic_fixes": [],
            "manual_review_needed": []
        }
        
        # Analyze different accessibility aspects
        color_analysis = self._analyze_color_contrast(content, structure)
        navigation_analysis = self._analyze_navigation_accessibility(structure)
        content_analysis = self._analyze_content_accessibility(content)
        multimedia_analysis = self._analyze_multimedia_accessibility(content)
        
        # Compile results
        all_analyses = [color_analysis, navigation_analysis, content_analysis, multimedia_analysis]
        analysis_results["overall_score"] = sum(a["score"] for a in all_analyses) / len(all_analyses)
        
        # Determine compliance level
        if analysis_results["overall_score"] >= 0.95:
            analysis_results["compliance_level"] = "AAA"
        elif analysis_results["overall_score"] >= 0.85:
            analysis_results["compliance_level"] = "AA"
        elif analysis_results["overall_score"] >= 0.75:
            analysis_results["compliance_level"] = "A"
        
        # Collect all recommendations
        for analysis in all_analyses:
            analysis_results["violations"].extend(analysis.get("violations", []))
            analysis_results["recommendations"].extend(analysis.get("recommendations", []))
            analysis_results["automatic_fixes"].extend(analysis.get("automatic_fixes", []))
            analysis_results["manual_review_needed"].extend(analysis.get("manual_review", []))
        
        return analysis_results
    
    def _analyze_color_contrast(self, content: str, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze color contrast accessibility."""
        results = {
            "score": 0.8,  # Default assuming decent contrast
            "violations": [],
            "recommendations": [],
            "automatic_fixes": []
        }
        
        # Check for color-only information
        color_only_patterns = [
            r'see.*red.*text',
            r'green.*indicates',
            r'click.*blue.*link',
        ]
        
        for pattern in color_only_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                results["violations"].append(f"Color-only information detected: {pattern}")
                results["recommendations"].append("Add text labels or icons alongside color indicators")
                results["score"] -= 0.1
        
        # Check for sufficient contrast indicators
        if "background-color" in content or "color:" in content:
            results["manual_review"].append("Manual contrast ratio verification needed for custom colors")
        
        return results
    
    def _analyze_navigation_accessibility(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze navigation accessibility."""
        results = {
            "score": 0.9,
            "violations": [],
            "recommendations": [],
            "automatic_fixes": [],
            "manual_review": []
        }
        
        # Check for proper heading hierarchy
        headings = structure.get("headings", [])
        if headings:
            heading_levels = [h.get("level", 1) for h in headings]
            if heading_levels != sorted(heading_levels):
                results["violations"].append("Improper heading hierarchy detected")
                results["recommendations"].append("Ensure headings follow logical hierarchy (h1 > h2 > h3)")
                results["automatic_fixes"].append("Restructure heading levels automatically")
                results["score"] -= 0.2
        
        # Check for skip navigation
        navigation_items = structure.get("navigation", [])
        if len(navigation_items) > 5:
            results["recommendations"].append("Add 'skip to main content' link for lengthy navigation")
        
        return results
    
    def _analyze_content_accessibility(self, content: str) -> Dict[str, Any]:
        """Analyze content accessibility."""
        results = {
            "score": 0.85,
            "violations": [],
            "recommendations": [],
            "automatic_fixes": [],
            "manual_review": []
        }
        
        # Check for alt text in images
        img_without_alt = re.findall(r'<img[^>]*(?!alt=)[^>]*>', content, re.IGNORECASE)
        if img_without_alt:
            results["violations"].append(f"Images without alt text: {len(img_without_alt)} found")
            results["recommendations"].append("Add descriptive alt text to all images")
            results["automatic_fixes"].append("Generate descriptive alt text for images")
            results["score"] -= 0.15
        
        # Check for descriptive link text
        vague_links = re.findall(r'<a[^>]*>(?:click here|read more|here|more)</a>', content, re.IGNORECASE)
        if vague_links:
            results["violations"].append(f"Vague link text found: {len(vague_links)} instances")
            results["recommendations"].append("Use descriptive link text that explains the destination")
            results["score"] -= 0.1
        
        # Check for proper list markup
        pseudo_lists = re.findall(r'^\s*[-*•]\s', content, re.MULTILINE)
        if len(pseudo_lists) > 3:
            results["recommendations"].append("Convert text-based lists to proper HTML list markup")
            results["automatic_fixes"].append("Convert pseudo-lists to proper <ul>/<ol> markup")
        
        return results
    
    def _analyze_multimedia_accessibility(self, content: str) -> Dict[str, Any]:
        """Analyze multimedia accessibility."""
        results = {
            "score": 0.9,
            "violations": [],
            "recommendations": [],
            "automatic_fixes": [],
            "manual_review": []
        }
        
        # Check for video elements
        videos = re.findall(r'<video[^>]*>', content, re.IGNORECASE)
        if videos:
            results["manual_review"].append("Videos require captions and audio descriptions")
            results["recommendations"].append("Ensure all videos have captions and transcripts")
        
        # Check for audio elements  
        audio_elements = re.findall(r'<audio[^>]*>', content, re.IGNORECASE)
        if audio_elements:
            results["manual_review"].append("Audio content requires transcripts")
            results["recommendations"].append("Provide transcripts for all audio content")
        
        return results
    
    def _load_wcag_guidelines(self) -> Dict[str, Any]:
        """Load WCAG accessibility guidelines."""
        return {
            "perceivable": [
                "Provide text alternatives for images",
                "Provide captions and alternatives for multimedia",
                "Ensure sufficient color contrast",
                "Make sure content can be presented without loss of meaning"
            ],
            "operable": [
                "Make all functionality keyboard accessible", 
                "Give users enough time to read content",
                "Don't use content that causes seizures",
                "Help users navigate and find content"
            ],
            "understandable": [
                "Make text readable and understandable",
                "Make content appear and operate predictably",
                "Help users avoid and correct mistakes"
            ],
            "robust": [
                "Maximize compatibility with assistive technologies"
            ]
        }
    
    def _load_accessibility_patterns(self) -> Dict[str, Any]:
        """Load accessibility patterns from frameworks."""
        return {
            "agency_swarm": ["Screen reader friendly tool descriptions", "Keyboard navigation for dashboards"],
            "crewai": ["High contrast workflow visualizations", "Alternative text for process diagrams"],
            "agentscope": ["Accessible development interfaces", "Voice navigation support"],
            "autogen": ["Screen reader compatible chat interfaces", "Keyboard shortcuts for actions"],
            "llama_agents": ["Accessible service documentation", "Screen reader friendly status updates"],
            "phidata": ["Alternative formats for visual recipes", "High contrast code examples"],
            "swarms": ["Accessible coordination interfaces", "Screen reader friendly swarm status"]
        }

class ResponsiveDesignOptimizer:
    """Optimizes responsive design for documentation."""
    
    def __init__(self):
        self.framework_breakpoints = self._get_framework_breakpoints()
        self.optimal_layouts = self._define_optimal_layouts()
    
    def optimize_responsive_design(self, content_structure: Dict[str, Any]) -> ResponsiveDesign:
        """Create optimized responsive design configuration."""
        
        # Analyze content complexity
        content_complexity = self._analyze_content_complexity(content_structure)
        
        # Select optimal breakpoints
        breakpoints = self._select_optimal_breakpoints(content_complexity)
        
        # Define layout adaptations
        layout_adaptations = self._create_layout_adaptations(content_structure, content_complexity)
        
        # Configure typography scaling
        typography_scaling = self._configure_typography_scaling(content_complexity)
        
        # Define navigation patterns
        navigation_patterns = self._define_navigation_patterns(content_structure)
        
        # Configure interaction methods
        interaction_methods = self._configure_interaction_methods(content_structure)
        
        return ResponsiveDesign(
            breakpoints=breakpoints,
            layout_adaptations=layout_adaptations,
            typography_scaling=typography_scaling,
            navigation_patterns=navigation_patterns,
            interaction_methods=interaction_methods
        )
    
    def _analyze_content_complexity(self, structure: Dict[str, Any]) -> str:
        """Analyze content complexity for responsive optimization."""
        complexity_score = 0
        
        # Count different content types
        complexity_score += len(structure.get("sections", [])) * 0.1
        complexity_score += len(structure.get("code_blocks", [])) * 0.2
        complexity_score += len(structure.get("images", [])) * 0.15
        complexity_score += len(structure.get("tables", [])) * 0.3
        complexity_score += len(structure.get("interactive_elements", [])) * 0.4
        
        if complexity_score < 1.0:
            return "simple"
        elif complexity_score < 3.0:
            return "moderate"
        elif complexity_score < 6.0:
            return "complex"
        else:
            return "enterprise"
    
    def _select_optimal_breakpoints(self, complexity: str) -> Dict[str, int]:
        """Select optimal breakpoints based on complexity."""
        base_breakpoints = {
            "mobile": 480,
            "tablet": 768,
            "desktop": 1024,
            "large": 1440
        }
        
        if complexity == "simple":
            return {
                "mobile": 480,
                "desktop": 768
            }
        elif complexity == "enterprise":
            return {
                "mobile": 480,
                "tablet": 768,
                "desktop": 1024,
                "large": 1440,
                "xlarge": 1920
            }
        else:
            return base_breakpoints
    
    def _create_layout_adaptations(self, structure: Dict[str, Any], complexity: str) -> Dict[DeviceType, Dict[str, Any]]:
        """Create layout adaptations for different devices."""
        adaptations = {}
        
        # Mobile adaptations
        adaptations[DeviceType.MOBILE] = {
            "layout": "single_column",
            "navigation": "hamburger_menu",
            "sidebar": "hidden_by_default",
            "code_blocks": "horizontal_scroll",
            "tables": "card_format",
            "images": "full_width_responsive"
        }
        
        # Tablet adaptations
        adaptations[DeviceType.TABLET] = {
            "layout": "flexible_two_column",
            "navigation": "condensed_horizontal",
            "sidebar": "collapsible",
            "code_blocks": "syntax_highlighting",
            "tables": "responsive_with_scroll",
            "images": "optimized_sizing"
        }
        
        # Desktop adaptations
        adaptations[DeviceType.DESKTOP] = {
            "layout": "three_column_with_sidebar",
            "navigation": "full_horizontal_menu",
            "sidebar": "persistent_visible",
            "code_blocks": "full_syntax_highlighting",
            "tables": "full_table_format",
            "images": "lightbox_gallery"
        }
        
        # Large screen adaptations
        adaptations[DeviceType.LARGE_SCREEN] = {
            "layout": "wide_three_column",
            "navigation": "extended_menu_with_search",
            "sidebar": "enhanced_with_context",
            "code_blocks": "side_by_side_examples",
            "tables": "enhanced_sorting_filtering", 
            "images": "high_resolution_gallery"
        }
        
        return adaptations
    
    def _configure_typography_scaling(self, complexity: str) -> Dict[DeviceType, Dict[str, Any]]:
        """Configure typography scaling for different devices."""
        base_scaling = {
            DeviceType.MOBILE: {
                "base_font_size": "16px",
                "heading_scale": 1.2,
                "line_height": 1.6,
                "paragraph_spacing": "1em"
            },
            DeviceType.TABLET: {
                "base_font_size": "17px", 
                "heading_scale": 1.3,
                "line_height": 1.5,
                "paragraph_spacing": "1.2em"
            },
            DeviceType.DESKTOP: {
                "base_font_size": "18px",
                "heading_scale": 1.4,
                "line_height": 1.6,
                "paragraph_spacing": "1.5em"
            },
            DeviceType.LARGE_SCREEN: {
                "base_font_size": "20px",
                "heading_scale": 1.5,
                "line_height": 1.7,
                "paragraph_spacing": "1.8em"
            }
        }
        
        # Adjust for complexity
        if complexity == "enterprise":
            for device in base_scaling:
                base_scaling[device]["code_font_size"] = "14px"
                base_scaling[device]["table_font_size"] = "15px"
        
        return base_scaling
    
    def _define_navigation_patterns(self, structure: Dict[str, Any]) -> Dict[DeviceType, str]:
        """Define navigation patterns for different devices."""
        nav_items_count = len(structure.get("navigation", []))
        
        if nav_items_count <= 5:
            return {
                DeviceType.MOBILE: "bottom_tab_bar",
                DeviceType.TABLET: "horizontal_menu",
                DeviceType.DESKTOP: "horizontal_menu",
                DeviceType.LARGE_SCREEN: "horizontal_menu"
            }
        elif nav_items_count <= 10:
            return {
                DeviceType.MOBILE: "hamburger_with_categories",
                DeviceType.TABLET: "two_level_horizontal",
                DeviceType.DESKTOP: "sidebar_with_horizontal",
                DeviceType.LARGE_SCREEN: "mega_menu"
            }
        else:
            return {
                DeviceType.MOBILE: "hierarchical_hamburger",
                DeviceType.TABLET: "accordion_sidebar",
                DeviceType.DESKTOP: "tree_sidebar_with_search",
                DeviceType.LARGE_SCREEN: "multi_level_mega_menu"
            }
    
    def _configure_interaction_methods(self, structure: Dict[str, Any]) -> Dict[DeviceType, List[str]]:
        """Configure interaction methods for different devices."""
        return {
            DeviceType.MOBILE: [
                "touch_gestures",
                "voice_commands",
                "shake_to_refresh",
                "swipe_navigation"
            ],
            DeviceType.TABLET: [
                "touch_gestures",
                "pinch_zoom",
                "two_finger_scroll",
                "keyboard_shortcuts"
            ],
            DeviceType.DESKTOP: [
                "mouse_interactions",
                "keyboard_shortcuts",
                "drag_and_drop",
                "context_menus",
                "hover_effects"
            ],
            DeviceType.LARGE_SCREEN: [
                "advanced_keyboard_shortcuts",
                "multi_monitor_support",
                "gesture_controls",
                "voice_navigation",
                "eye_tracking"
            ]
        }
    
    def _get_framework_breakpoints(self) -> Dict[str, Dict[str, int]]:
        """Get breakpoints used by different frameworks."""
        return {
            "bootstrap": {"xs": 0, "sm": 576, "md": 768, "lg": 992, "xl": 1200},
            "tailwind": {"sm": 640, "md": 768, "lg": 1024, "xl": 1280, "2xl": 1536},
            "material": {"xs": 0, "sm": 600, "md": 960, "lg": 1280, "xl": 1920},
            "foundation": {"small": 0, "medium": 640, "large": 1024, "xlarge": 1200, "xxlarge": 1440}
        }
    
    def _define_optimal_layouts(self) -> Dict[str, Dict[str, Any]]:
        """Define optimal layouts for different scenarios."""
        return {
            "simple_docs": {
                "mobile": "single_column_centered",
                "tablet": "single_column_wide",
                "desktop": "two_column_with_toc"
            },
            "api_reference": {
                "mobile": "accordion_sections",
                "tablet": "two_column_adaptive",
                "desktop": "three_column_reference"
            },
            "tutorial_series": {
                "mobile": "step_by_step_cards",
                "tablet": "side_by_side_preview",
                "desktop": "split_view_with_preview"
            }
        }

class UserJourneyOptimizer:
    """Optimizes user journeys through documentation."""
    
    def __init__(self):
        self.user_personas = self._create_user_personas()
        self.journey_patterns = self._define_journey_patterns()
    
    def optimize_user_journey(self, content_structure: Dict[str, Any], 
                            target_personas: List[UserType]) -> Dict[str, Any]:
        """Optimize user journey for target personas."""
        
        optimizations = {}
        
        for persona_type in target_personas:
            persona = self.user_personas[persona_type]
            
            journey_optimization = {
                "entry_points": self._identify_optimal_entry_points(persona, content_structure),
                "navigation_flow": self._optimize_navigation_flow(persona, content_structure),
                "content_prioritization": self._prioritize_content_for_persona(persona, content_structure),
                "interaction_optimizations": self._optimize_interactions_for_persona(persona),
                "success_metrics": persona.success_metrics,
                "pain_point_mitigations": self._create_pain_point_mitigations(persona, content_structure)
            }
            
            optimizations[persona_type.value] = journey_optimization
        
        return optimizations
    
    def _create_user_personas(self) -> Dict[UserType, UserPersona]:
        """Create detailed user personas."""
        return {
            UserType.BEGINNER: UserPersona(
                user_type=UserType.BEGINNER,
                primary_goals=["Learn basics", "Get started quickly", "Avoid mistakes"],
                pain_points=["Too much information", "Complex jargon", "Missing prerequisites"],
                preferred_learning_style="step_by_step",
                technical_proficiency="low",
                context_usage=["learning_time", "experimentation"],
                success_metrics=["time_to_first_success", "completion_rate", "error_reduction"]
            ),
            UserType.INTERMEDIATE: UserPersona(
                user_type=UserType.INTERMEDIATE,
                primary_goals=["Solve specific problems", "Learn advanced features", "Best practices"],
                pain_points=["Too basic content", "Missing advanced examples", "Poor searchability"],
                preferred_learning_style="example_driven",
                technical_proficiency="medium",
                context_usage=["problem_solving", "feature_exploration"],
                success_metrics=["problem_resolution_time", "feature_adoption", "depth_of_usage"]
            ),
            UserType.EXPERT: UserPersona(
                user_type=UserType.EXPERT,
                primary_goals=["Quick reference", "Edge cases", "Customization"],
                pain_points=["Verbose explanations", "Missing technical details", "Poor API docs"],
                preferred_learning_style="reference_oriented",
                technical_proficiency="high",
                context_usage=["quick_lookup", "troubleshooting", "integration"],
                success_metrics=["lookup_speed", "technical_accuracy", "customization_success"]
            ),
            UserType.ENTERPRISE: UserPersona(
                user_type=UserType.ENTERPRISE,
                primary_goals=["Security compliance", "Scalability", "Team adoption"],
                pain_points=["Missing security info", "No enterprise features", "Poor team docs"],
                preferred_learning_style="comprehensive",
                technical_proficiency="high",
                context_usage=["evaluation", "implementation", "team_training"],
                success_metrics=["compliance_coverage", "team_adoption_rate", "implementation_success"]
            ),
            UserType.DEVELOPER: UserPersona(
                user_type=UserType.DEVELOPER,
                primary_goals=["Integration", "Code examples", "API reference"],
                pain_points=["Outdated code", "Missing SDKs", "Poor error handling"],
                preferred_learning_style="code_first",
                technical_proficiency="high", 
                context_usage=["development", "debugging", "integration"],
                success_metrics=["integration_time", "code_quality", "error_resolution"]
            ),
            UserType.RESEARCHER: UserPersona(
                user_type=UserType.RESEARCHER,
                primary_goals=["Understanding concepts", "Comparisons", "Deep dives"],
                pain_points=["Shallow explanations", "No comparisons", "Missing theory"],
                preferred_learning_style="conceptual",
                technical_proficiency="varies",
                context_usage=["research", "comparison", "analysis"],
                success_metrics=["comprehension_depth", "comparison_quality", "research_efficiency"]
            )
        }
    
    def _identify_optimal_entry_points(self, persona: UserPersona, 
                                     content_structure: Dict[str, Any]) -> List[str]:
        """Identify optimal entry points for a persona."""
        sections = content_structure.get("sections", [])
        
        if persona.user_type == UserType.BEGINNER:
            return [
                "getting_started",
                "quick_start", 
                "tutorial",
                "basic_concepts"
            ]
        elif persona.user_type == UserType.EXPERT:
            return [
                "api_reference",
                "advanced_configuration",
                "troubleshooting",
                "changelog"
            ]
        elif persona.user_type == UserType.DEVELOPER:
            return [
                "code_examples",
                "sdk_reference",
                "integration_guide",
                "api_documentation"
            ]
        else:
            # Default balanced entry points
            return [
                "overview",
                "getting_started",
                "examples",
                "reference"
            ]
    
    def _optimize_navigation_flow(self, persona: UserPersona, 
                                content_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize navigation flow for persona."""
        if persona.preferred_learning_style == "step_by_step":
            return {
                "pattern": "linear_progression",
                "next_previous": True,
                "progress_indicator": True,
                "breadcrumbs": True,
                "related_sections": False
            }
        elif persona.preferred_learning_style == "reference_oriented":
            return {
                "pattern": "hub_and_spoke", 
                "quick_search": True,
                "categories": True,
                "bookmarks": True,
                "related_sections": True
            }
        else:
            return {
                "pattern": "flexible_exploration",
                "search": True,
                "categories": True,
                "related_content": True,
                "bookmarks": True
            }
    
    def _prioritize_content_for_persona(self, persona: UserPersona, 
                                      content_structure: Dict[str, Any]) -> Dict[str, int]:
        """Prioritize content sections for persona."""
        priorities = {}
        sections = content_structure.get("sections", [])
        
        # Base priorities for different persona types
        persona_priorities = {
            UserType.BEGINNER: {
                "getting_started": 10,
                "basic_concepts": 9,
                "tutorial": 9,
                "examples": 8,
                "faq": 7,
                "troubleshooting": 6
            },
            UserType.EXPERT: {
                "api_reference": 10,
                "advanced_configuration": 9,
                "performance_tuning": 8,
                "customization": 8,
                "changelog": 7,
                "migration_guide": 7
            },
            UserType.DEVELOPER: {
                "code_examples": 10,
                "api_documentation": 10,
                "sdk_reference": 9,
                "integration_guide": 9,
                "error_handling": 8,
                "best_practices": 7
            }
        }
        
        base_priorities = persona_priorities.get(persona.user_type, {})
        
        # Assign priorities to actual sections
        for section in sections:
            section_name = section.get("name", "").lower().replace(" ", "_")
            
            # Exact match
            if section_name in base_priorities:
                priorities[section_name] = base_priorities[section_name]
            else:
                # Fuzzy matching
                for priority_section, priority in base_priorities.items():
                    if priority_section in section_name or section_name in priority_section:
                        priorities[section_name] = priority
                        break
                else:
                    priorities[section_name] = 5  # Default medium priority
        
        return priorities
    
    def _optimize_interactions_for_persona(self, persona: UserPersona) -> Dict[str, Any]:
        """Optimize interactions for persona."""
        if persona.technical_proficiency == "low":
            return {
                "tooltips": True,
                "help_hints": True,
                "confirmation_dialogs": True,
                "undo_actions": True,
                "keyboard_shortcuts": False,
                "advanced_features": "hidden_by_default"
            }
        elif persona.technical_proficiency == "high":
            return {
                "tooltips": False,
                "help_hints": False,
                "confirmation_dialogs": False,
                "undo_actions": True,
                "keyboard_shortcuts": True,
                "advanced_features": "prominently_displayed"
            }
        else:
            return {
                "tooltips": "contextual",
                "help_hints": "progressive",
                "confirmation_dialogs": "for_destructive_actions",
                "undo_actions": True,
                "keyboard_shortcuts": True,
                "advanced_features": "discoverable"
            }
    
    def _create_pain_point_mitigations(self, persona: UserPersona, 
                                     content_structure: Dict[str, Any]) -> List[str]:
        """Create mitigations for persona pain points."""
        mitigations = []
        
        for pain_point in persona.pain_points:
            if "too much information" in pain_point.lower():
                mitigations.extend([
                    "Progressive disclosure of content",
                    "Collapsible sections for advanced topics",
                    "Summary boxes for key information"
                ])
            elif "complex jargon" in pain_point.lower():
                mitigations.extend([
                    "Glossary with definitions",
                    "Hover tooltips for technical terms",
                    "Plain language alternatives"
                ])
            elif "poor searchability" in pain_point.lower():
                mitigations.extend([
                    "Enhanced search with filters",
                    "Search result previews",
                    "Suggested searches and auto-complete"
                ])
            elif "missing" in pain_point.lower():
                mitigations.extend([
                    "Content gap analysis and filling",
                    "User feedback collection",
                    "Related content suggestions"
                ])
        
        return list(set(mitigations))  # Remove duplicates
    
    def _define_journey_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Define common user journey patterns."""
        return {
            "learning_journey": {
                "stages": ["awareness", "exploration", "understanding", "application", "mastery"],
                "content_types": ["overview", "tutorial", "examples", "practice", "advanced"],
                "success_criteria": ["comprehension", "retention", "application"]
            },
            "problem_solving_journey": {
                "stages": ["problem_identification", "solution_search", "implementation", "validation"],
                "content_types": ["troubleshooting", "how_to", "code_examples", "testing"],
                "success_criteria": ["problem_resolution", "implementation_success", "time_efficiency"]
            },
            "reference_journey": {
                "stages": ["quick_lookup", "detailed_review", "application"],
                "content_types": ["api_docs", "reference", "examples"],
                "success_criteria": ["information_accuracy", "lookup_speed", "implementation_success"]
            }
        }

class UXExcellenceFramework:
    """Main framework orchestrating all UX optimization components."""
    
    def __init__(self):
        self.accessibility_analyzer = AccessibilityAnalyzer()
        self.responsive_optimizer = ResponsiveDesignOptimizer()
        self.journey_optimizer = UserJourneyOptimizer()
        self.ux_metrics_tracker = UXMetricsTracker()
    
    def optimize_documentation_ux(self, content: str, structure: Dict[str, Any], 
                                target_personas: List[str]) -> Dict[str, Any]:
        """Comprehensive UX optimization for documentation."""
        
        # Convert string personas to enum
        persona_enums = []
        for persona in target_personas:
            try:
                persona_enums.append(UserType(persona))
            except ValueError:
                continue
        
        if not persona_enums:
            persona_enums = [UserType.INTERMEDIATE]  # Default fallback
        
        # Run all optimizations
        accessibility_analysis = self.accessibility_analyzer.analyze_accessibility(content, structure)
        responsive_design = self.responsive_optimizer.optimize_responsive_design(structure)
        journey_optimization = self.journey_optimizer.optimize_user_journey(structure, persona_enums)
        
        # Create comprehensive optimization plan
        optimization_plan = {
            "overall_ux_score": self._calculate_overall_ux_score(
                accessibility_analysis, responsive_design, journey_optimization
            ),
            "accessibility": accessibility_analysis,
            "responsive_design": {
                "breakpoints": responsive_design.breakpoints,
                "layout_adaptations": {k.value: v for k, v in responsive_design.layout_adaptations.items()},
                "typography_scaling": {k.value: v for k, v in responsive_design.typography_scaling.items()},
                "navigation_patterns": {k.value: v for k, v in responsive_design.navigation_patterns.items()}
            },
            "user_journey_optimization": journey_optimization,
            "implementation_recommendations": self._create_implementation_recommendations(
                accessibility_analysis, responsive_design, journey_optimization
            ),
            "metrics_to_track": self._define_tracking_metrics(persona_enums),
            "testing_strategy": self._create_testing_strategy(persona_enums)
        }
        
        return optimization_plan
    
    def _calculate_overall_ux_score(self, accessibility: Dict[str, Any], 
                                  responsive_design: ResponsiveDesign,
                                  journey_optimization: Dict[str, Any]) -> float:
        """Calculate overall UX score."""
        accessibility_score = accessibility["overall_score"]
        
        # Responsive design score (simplified)
        responsive_score = 0.9  # Assume good responsive design from optimization
        
        # Journey optimization score (based on number of optimizations)
        journey_score = min(1.0, len(journey_optimization) * 0.15)
        
        # Weighted average
        overall_score = (
            accessibility_score * 0.4 +
            responsive_score * 0.3 +
            journey_score * 0.3
        )
        
        return round(overall_score, 2)
    
    def _create_implementation_recommendations(self, accessibility: Dict[str, Any],
                                            responsive_design: ResponsiveDesign,
                                            journey_optimization: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create prioritized implementation recommendations."""
        recommendations = []
        
        # High priority accessibility fixes
        for fix in accessibility.get("automatic_fixes", []):
            recommendations.append({
                "priority": "high",
                "category": "accessibility",
                "action": fix,
                "effort": "low",
                "impact": "high"
            })
        
        # Medium priority responsive improvements
        if len(responsive_design.breakpoints) > 2:
            recommendations.append({
                "priority": "medium", 
                "category": "responsive",
                "action": "Implement comprehensive responsive breakpoints",
                "effort": "medium",
                "impact": "high"
            })
        
        # User journey improvements
        for persona, optimization in journey_optimization.items():
            if optimization.get("pain_point_mitigations"):
                recommendations.append({
                    "priority": "medium",
                    "category": "user_journey",
                    "action": f"Implement pain point mitigations for {persona}",
                    "effort": "medium", 
                    "impact": "medium"
                })
        
        return sorted(recommendations, key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["priority"]], reverse=True)
    
    def _define_tracking_metrics(self, personas: List[UserType]) -> Dict[str, List[str]]:
        """Define metrics to track for UX optimization."""
        base_metrics = [
            "page_load_time",
            "bounce_rate", 
            "time_on_page",
            "search_success_rate",
            "mobile_usage_percentage"
        ]
        
        persona_specific_metrics = {}
        for persona in personas:
            persona_metrics = base_metrics.copy()
            
            if persona == UserType.BEGINNER:
                persona_metrics.extend([
                    "tutorial_completion_rate",
                    "help_section_usage",
                    "error_frequency"
                ])
            elif persona == UserType.EXPERT:
                persona_metrics.extend([
                    "api_reference_usage",
                    "search_query_complexity", 
                    "advanced_feature_adoption"
                ])
            elif persona == UserType.DEVELOPER:
                persona_metrics.extend([
                    "code_copy_rate",
                    "example_success_rate",
                    "sdk_download_conversion"
                ])
            
            persona_specific_metrics[persona.value] = persona_metrics
        
        return persona_specific_metrics
    
    def _create_testing_strategy(self, personas: List[UserType]) -> Dict[str, Any]:
        """Create comprehensive UX testing strategy."""
        return {
            "automated_tests": [
                "Accessibility compliance (WCAG AA)",
                "Responsive design verification",
                "Performance benchmarking",
                "Link checking and validation"
            ],
            "user_testing": [
                f"Usability testing with {persona.value} users" for persona in personas
            ],
            "analytics_tracking": [
                "User journey mapping",
                "Heat map analysis",
                "A/B testing for key improvements"
            ],
            "continuous_monitoring": [
                "Performance monitoring",
                "Error tracking",
                "User feedback collection"
            ]
        }

class UXMetricsTracker:
    """Tracks UX metrics and improvements over time."""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_metrics(self, metrics: Dict[str, Any], timestamp: Optional[datetime] = None):
        """Record UX metrics."""
        if timestamp is None:
            timestamp = datetime.now()
        
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            
            self.metrics_history[metric_name].append({
                "timestamp": timestamp,
                "value": value
            })
    
    def get_metrics_trend(self, metric_name: str) -> Dict[str, Any]:
        """Get trend analysis for a specific metric."""
        if metric_name not in self.metrics_history:
            return {"error": f"No data for metric {metric_name}"}
        
        history = self.metrics_history[metric_name]
        if len(history) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        recent_value = history[-1]["value"]
        previous_value = history[-2]["value"]
        
        if isinstance(recent_value, (int, float)) and isinstance(previous_value, (int, float)):
            change = recent_value - previous_value
            change_percentage = (change / previous_value) * 100 if previous_value != 0 else 0
            
            return {
                "metric": metric_name,
                "current_value": recent_value,
                "previous_value": previous_value,
                "change": change,
                "change_percentage": change_percentage,
                "trend": "improving" if change > 0 else "declining" if change < 0 else "stable"
            }
        else:
            return {"message": f"Cannot calculate trend for non-numeric metric {metric_name}"}

# Global UX framework instance
_ux_framework = UXExcellenceFramework()

def get_ux_framework() -> UXExcellenceFramework:
    """Get the global UX framework instance."""
    return _ux_framework

def optimize_documentation_ux(content: str, structure: Dict[str, Any], 
                            personas: List[str] = None) -> Dict[str, Any]:
    """High-level function to optimize documentation UX."""
    if personas is None:
        personas = ["intermediate"]
    
    framework = get_ux_framework()
    return framework.optimize_documentation_ux(content, structure, personas)