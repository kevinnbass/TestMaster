"""
Adaptive Template System

Dynamic template selection and customization system that learns from usage
patterns and automatically adapts templates for optimal documentation generation.
"""

import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TemplateType(Enum):
    """Types of documentation templates."""
    API_REFERENCE = "api_reference"
    TUTORIAL = "tutorial"
    COOKBOOK_RECIPE = "cookbook_recipe"
    ARCHITECTURE_DOC = "architecture_doc"
    TROUBLESHOOTING = "troubleshooting"
    DEPLOYMENT_GUIDE = "deployment_guide"
    USER_GUIDE = "user_guide"
    CHANGELOG = "changelog"
    README = "readme"


class AdaptationStrategy(Enum):
    """Template adaptation strategies."""
    USAGE_BASED = "usage_based"
    PERFORMANCE_BASED = "performance_based"
    FEEDBACK_BASED = "feedback_based"
    CONTENT_BASED = "content_based"
    AUDIENCE_BASED = "audience_based"
    HYBRID = "hybrid"


class TemplateComplexity(Enum):
    """Template complexity levels."""
    MINIMAL = 1
    BASIC = 2
    STANDARD = 3
    COMPREHENSIVE = 4
    ENTERPRISE = 5


@dataclass
class TemplateMetrics:
    """Metrics for template performance and usage."""
    template_id: str
    usage_count: int = 0
    success_rate: float = 0.0
    average_generation_time: float = 0.0
    user_satisfaction_score: float = 0.0
    adaptation_score: float = 0.0
    last_updated: str = ""
    performance_trend: List[float] = field(default_factory=list)


@dataclass
class TemplateVariant:
    """Represents a variant of a template."""
    variant_id: str
    base_template_id: str
    modifications: List[str] = field(default_factory=list)
    target_audience: str = "general"
    complexity_level: TemplateComplexity = TemplateComplexity.STANDARD
    success_metrics: TemplateMetrics = None
    content_pattern: str = ""
    customization_points: List[str] = field(default_factory=list)


@dataclass
class AdaptationRule:
    """Rule for automatic template adaptation."""
    rule_id: str
    condition: str
    action: str
    confidence_threshold: float = 0.7
    activation_count: int = 0
    success_rate: float = 0.0
    priority: int = 3  # 1-5 scale


@dataclass
class TemplateContext:
    """Context information for template selection and adaptation."""
    content_type: TemplateType
    target_audience: str
    project_characteristics: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    performance_requirements: Dict[str, Any] = field(default_factory=dict)
    historical_usage: Dict[str, Any] = field(default_factory=dict)


class AdaptiveTemplateSystem:
    """
    Adaptive template system that learns from usage patterns and automatically
    optimizes templates for better documentation generation results.
    """
    
    def __init__(self, templates_dir: str = "adaptive_templates"):
        """Initialize adaptive template system."""
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Core template management
        self.base_templates = {}
        self.template_variants = {}
        self.template_metrics = {}
        self.adaptation_rules = {}
        
        # Learning and adaptation systems
        self.usage_patterns = {}
        self.performance_history = {}
        self.adaptation_engine = self._initialize_adaptation_engine()
        self.learning_model = self._initialize_learning_model()
        
        # Load existing templates and data
        self._load_base_templates()
        self._load_adaptation_data()
        
        logger.info("Adaptive template system initialized")
        
    def select_optimal_template(self, context: TemplateContext) -> str:
        """Select optimal template based on context and learned patterns."""
        logger.info(f"Selecting optimal template for {context.content_type.value}")
        
        # Get candidate templates
        candidates = self._get_candidate_templates(context)
        
        # Score each candidate
        scores = {}
        for template_id in candidates:
            score = self._calculate_template_score(template_id, context)
            scores[template_id] = score
            
        # Select highest scoring template
        optimal_template = max(scores.items(), key=lambda x: x[1])[0]
        
        # Record usage
        self._record_template_usage(optimal_template, context)
        
        logger.info(f"Selected template: {optimal_template} (score: {scores[optimal_template]:.3f})")
        return optimal_template
        
    def adapt_template_dynamically(self, 
                                 template_id: str, 
                                 context: TemplateContext,
                                 feedback: Dict[str, Any] = None) -> str:
        """Dynamically adapt template based on context and feedback."""
        logger.info(f"Adapting template {template_id} for context")
        
        # Get base template
        base_template = self.base_templates.get(template_id, "")
        if not base_template:
            logger.error(f"Template {template_id} not found")
            return ""
            
        # Apply context-based adaptations
        adapted_template = self._apply_context_adaptations(base_template, context)
        
        # Apply learned adaptations
        adapted_template = self._apply_learned_adaptations(adapted_template, template_id, context)
        
        # Apply feedback-based adaptations if available
        if feedback:
            adapted_template = self._apply_feedback_adaptations(adapted_template, feedback)
            
        # Create template variant if significantly different
        if self._is_significant_adaptation(base_template, adapted_template):
            variant_id = self._create_template_variant(template_id, adapted_template, context)
            logger.info(f"Created new template variant: {variant_id}")
            
        return adapted_template
        
    def learn_from_usage_patterns(self) -> Dict[str, Any]:
        """Analyze usage patterns and update adaptation rules."""
        logger.info("Learning from usage patterns")
        
        learning_results = {
            "new_rules": [],
            "updated_rules": [],
            "deprecated_rules": [],
            "insights": []
        }
        
        # Analyze usage frequency patterns
        usage_patterns = self._analyze_usage_patterns()
        
        # Analyze performance correlations
        performance_patterns = self._analyze_performance_patterns()
        
        # Generate new adaptation rules
        new_rules = self._generate_adaptation_rules(usage_patterns, performance_patterns)
        learning_results["new_rules"] = new_rules
        
        # Update existing rules based on performance
        updated_rules = self._update_adaptation_rules()
        learning_results["updated_rules"] = updated_rules
        
        # Identify underperforming rules
        deprecated_rules = self._identify_deprecated_rules()
        learning_results["deprecated_rules"] = deprecated_rules
        
        # Generate insights
        insights = self._generate_learning_insights(usage_patterns, performance_patterns)
        learning_results["insights"] = insights
        
        # Save updated adaptation data
        self._save_adaptation_data()
        
        logger.info(f"Learning complete: {len(new_rules)} new rules, {len(updated_rules)} updated")
        return learning_results
        
    def create_custom_template(self, 
                             base_type: TemplateType,
                             customizations: Dict[str, Any],
                             target_context: TemplateContext) -> str:
        """Create custom template based on requirements."""
        logger.info(f"Creating custom template for {base_type.value}")
        
        # Start with base template for type
        base_template = self._get_base_template_for_type(base_type)
        
        # Apply customizations
        custom_template = self._apply_customizations(base_template, customizations)
        
        # Optimize for target context
        optimized_template = self._optimize_for_context(custom_template, target_context)
        
        # Generate unique template ID
        template_id = f"custom_{base_type.value}_{len(self.base_templates)}"
        
        # Save custom template
        self.base_templates[template_id] = optimized_template
        self.template_metrics[template_id] = TemplateMetrics(template_id=template_id)
        
        logger.info(f"Created custom template: {template_id}")
        return template_id
        
    def analyze_template_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of all templates."""
        logger.info("Analyzing template effectiveness")
        
        effectiveness_report = {
            "overall_metrics": {},
            "template_rankings": {},
            "improvement_opportunities": [],
            "usage_trends": {},
            "performance_insights": []
        }
        
        # Calculate overall metrics
        all_metrics = list(self.template_metrics.values())
        if all_metrics:
            effectiveness_report["overall_metrics"] = {
                "average_success_rate": sum(m.success_rate for m in all_metrics) / len(all_metrics),
                "average_satisfaction": sum(m.user_satisfaction_score for m in all_metrics) / len(all_metrics),
                "total_usage": sum(m.usage_count for m in all_metrics),
                "templates_count": len(all_metrics)
            }
            
        # Rank templates by effectiveness
        template_scores = {}
        for template_id, metrics in self.template_metrics.items():
            score = self._calculate_effectiveness_score(metrics)
            template_scores[template_id] = score
            
        effectiveness_report["template_rankings"] = dict(
            sorted(template_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Identify improvement opportunities
        for template_id, metrics in self.template_metrics.items():
            if metrics.success_rate < 0.7 or metrics.user_satisfaction_score < 0.7:
                opportunity = {
                    "template_id": template_id,
                    "issues": [],
                    "recommendations": []
                }
                
                if metrics.success_rate < 0.7:
                    opportunity["issues"].append(f"Low success rate: {metrics.success_rate:.2f}")
                    opportunity["recommendations"].append("Review template structure and content")
                    
                if metrics.user_satisfaction_score < 0.7:
                    opportunity["issues"].append(f"Low satisfaction: {metrics.user_satisfaction_score:.2f}")
                    opportunity["recommendations"].append("Analyze user feedback and adapt template")
                    
                effectiveness_report["improvement_opportunities"].append(opportunity)
                
        return effectiveness_report
        
    def _get_candidate_templates(self, context: TemplateContext) -> List[str]:
        """Get candidate templates for the given context."""
        candidates = []
        
        # Find templates matching content type
        for template_id, template_content in self.base_templates.items():
            if self._matches_content_type(template_id, context.content_type):
                candidates.append(template_id)
                
        # Add relevant variants
        for variant_id, variant in self.template_variants.items():
            if (variant.base_template_id in candidates or
                self._matches_context(variant, context)):
                candidates.append(variant_id)
                
        return candidates
        
    def _calculate_template_score(self, template_id: str, context: TemplateContext) -> float:
        """Calculate score for template in given context."""
        score = 0.0
        
        # Base metrics score (40% weight)
        if template_id in self.template_metrics:
            metrics = self.template_metrics[template_id]
            metrics_score = (
                metrics.success_rate * 0.4 +
                metrics.user_satisfaction_score * 0.3 +
                (1.0 - min(metrics.average_generation_time / 60.0, 1.0)) * 0.3  # Normalize time
            )
            score += metrics_score * 0.4
            
        # Context match score (35% weight)
        context_score = self._calculate_context_match_score(template_id, context)
        score += context_score * 0.35
        
        # Usage history score (25% weight)
        usage_score = self._calculate_usage_history_score(template_id, context)
        score += usage_score * 0.25
        
        return score
        
    def _apply_context_adaptations(self, template: str, context: TemplateContext) -> str:
        """Apply context-specific adaptations to template."""
        adapted = template
        
        # Adapt for target audience
        if context.target_audience == "beginners":
            adapted = self._simplify_language(adapted)
            adapted = self._add_more_explanations(adapted)
        elif context.target_audience == "experts":
            adapted = self._make_more_technical(adapted)
            adapted = self._reduce_verbosity(adapted)
            
        # Adapt for project characteristics
        if "enterprise" in context.project_characteristics.get("type", ""):
            adapted = self._add_enterprise_sections(adapted)
            adapted = self._add_compliance_info(adapted)
            
        # Adapt for performance requirements
        if context.performance_requirements.get("fast_generation", False):
            adapted = self._optimize_for_speed(adapted)
            
        return adapted
        
    def _apply_learned_adaptations(self, template: str, template_id: str, context: TemplateContext) -> str:
        """Apply adaptations learned from historical data."""
        adapted = template
        
        # Apply successful adaptation rules
        applicable_rules = [rule for rule in self.adaptation_rules.values()
                          if self._rule_applies(rule, template_id, context)]
        
        # Sort by success rate and priority
        applicable_rules.sort(key=lambda r: (r.success_rate, r.priority), reverse=True)
        
        for rule in applicable_rules:
            if rule.success_rate >= rule.confidence_threshold:
                adapted = self._apply_adaptation_rule(adapted, rule)
                rule.activation_count += 1
                
        return adapted
        
    def _initialize_adaptation_engine(self) -> Dict[str, Any]:
        """Initialize the adaptation engine."""
        return {
            "strategies": [strategy for strategy in AdaptationStrategy],
            "active_strategy": AdaptationStrategy.HYBRID,
            "learning_rate": 0.1,
            "adaptation_threshold": 0.05,
            "max_adaptations_per_template": 10
        }
        
    def _initialize_learning_model(self) -> Dict[str, Any]:
        """Initialize the learning model."""
        return {
            "pattern_weights": {
                "usage_frequency": 0.3,
                "success_rate": 0.4,
                "user_feedback": 0.3
            },
            "confidence_threshold": 0.7,
            "learning_window": 100,  # Number of recent usages to consider
            "adaptation_cooldown": 24  # Hours between adaptations
        }
        
    def _load_base_templates(self) -> None:
        """Load base templates from framework patterns."""
        # Load templates from each framework
        base_templates = {
            "api_reference_autogen": """# {title} API Reference

## Overview
{overview}

## Installation
```bash
{installation_commands}
```

## Quick Start
{quick_start_example}

## API Documentation

{api_sections}

## Examples
{code_examples}

## Error Handling
{error_handling}
""",
            
            "tutorial_phidata": """# {title} Tutorial

## What You'll Learn
{learning_objectives}

## Prerequisites
{prerequisites}

## Step-by-Step Guide

{tutorial_steps}

## Testing Your Implementation
{testing_section}

## Next Steps
{next_steps}

## Troubleshooting
{troubleshooting}
""",
            
            "cookbook_recipe": """# {recipe_title}

{recipe_description}

**Complexity:** {"⭐" * complexity_level}
**Estimated Time:** {estimated_time}

## Ingredients (Prerequisites)
{prerequisites_list}

## Instructions
{step_by_step_instructions}

## Example Usage
```python
{code_example}
```

## Tips & Variations
{tips_and_variations}
""",
            
            "architecture_doc": """# {title} Architecture

## System Overview
{system_overview}

## Key Components
{components_section}

## Data Flow
{data_flow_diagram}

## Design Decisions
{design_decisions}

## Deployment Architecture
{deployment_section}

## Security Considerations
{security_section}
""",
            
            "deployment_guide": """# {title} Deployment Guide

## Prerequisites
{prerequisites}

## Environment Setup
{environment_setup}

## Deployment Steps
{deployment_steps}

## Configuration
{configuration_section}

## Monitoring & Maintenance
{monitoring_section}

## Troubleshooting
{troubleshooting_guide}
"""
        }
        
        self.base_templates.update(base_templates)
        
        # Initialize metrics for each template
        for template_id in base_templates:
            self.template_metrics[template_id] = TemplateMetrics(template_id=template_id)
            
    def _load_adaptation_data(self) -> None:
        """Load adaptation data from storage."""
        # Load saved metrics, rules, and patterns
        adaptation_file = self.templates_dir / "adaptation_data.json"
        
        if adaptation_file.exists():
            try:
                with open(adaptation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Load metrics
                for template_id, metrics_data in data.get("metrics", {}).items():
                    self.template_metrics[template_id] = TemplateMetrics(**metrics_data)
                    
                # Load adaptation rules
                for rule_id, rule_data in data.get("rules", {}).items():
                    self.adaptation_rules[rule_id] = AdaptationRule(**rule_data)
                    
                logger.info("Loaded adaptation data from storage")
                
            except Exception as e:
                logger.warning(f"Could not load adaptation data: {e}")
                
    def _save_adaptation_data(self) -> None:
        """Save adaptation data to storage."""
        adaptation_file = self.templates_dir / "adaptation_data.json"
        
        data = {
            "metrics": {
                template_id: {
                    "template_id": metrics.template_id,
                    "usage_count": metrics.usage_count,
                    "success_rate": metrics.success_rate,
                    "average_generation_time": metrics.average_generation_time,
                    "user_satisfaction_score": metrics.user_satisfaction_score,
                    "adaptation_score": metrics.adaptation_score,
                    "last_updated": metrics.last_updated
                }
                for template_id, metrics in self.template_metrics.items()
            },
            "rules": {
                rule_id: {
                    "rule_id": rule.rule_id,
                    "condition": rule.condition,
                    "action": rule.action,
                    "confidence_threshold": rule.confidence_threshold,
                    "activation_count": rule.activation_count,
                    "success_rate": rule.success_rate,
                    "priority": rule.priority
                }
                for rule_id, rule in self.adaptation_rules.items()
            }
        }
        
        with open(adaptation_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    def _matches_content_type(self, template_id: str, content_type: TemplateType) -> bool:
        """Check if template matches content type."""
        type_keywords = {
            TemplateType.API_REFERENCE: ["api", "reference"],
            TemplateType.TUTORIAL: ["tutorial", "guide"],
            TemplateType.COOKBOOK_RECIPE: ["recipe", "cookbook"],
            TemplateType.ARCHITECTURE_DOC: ["architecture", "design"],
            TemplateType.DEPLOYMENT_GUIDE: ["deployment", "deploy"]
        }
        
        keywords = type_keywords.get(content_type, [])
        return any(keyword in template_id.lower() for keyword in keywords)
        
    def _record_template_usage(self, template_id: str, context: TemplateContext) -> None:
        """Record template usage for learning."""
        if template_id not in self.template_metrics:
            self.template_metrics[template_id] = TemplateMetrics(template_id=template_id)
            
        metrics = self.template_metrics[template_id]
        metrics.usage_count += 1
        
        # Record usage pattern
        usage_key = f"{context.content_type.value}_{context.target_audience}"
        if usage_key not in self.usage_patterns:
            self.usage_patterns[usage_key] = {"templates": {}, "total_usage": 0}
            
        pattern = self.usage_patterns[usage_key]
        if template_id not in pattern["templates"]:
            pattern["templates"][template_id] = 0
        pattern["templates"][template_id] += 1
        pattern["total_usage"] += 1