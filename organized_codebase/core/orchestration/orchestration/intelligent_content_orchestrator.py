"""
Intelligent Content Orchestrator

AI-powered content coordination that automatically selects optimal patterns,
manages content workflows, and ensures consistency across documentation.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content to orchestrate."""
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    USER_GUIDES = "user_guides"
    API_REFERENCES = "api_references"
    TUTORIALS = "tutorials"
    TROUBLESHOOTING = "troubleshooting"
    ARCHITECTURE_DOCS = "architecture_docs"
    DEPLOYMENT_GUIDES = "deployment_guides"
    COOKBOOK_RECIPES = "cookbook_recipes"


class OrchestrationStrategy(Enum):
    """Content orchestration strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ADAPTIVE = "adaptive"
    INTELLIGENT_ROUTING = "intelligent_routing"
    DEPENDENCY_BASED = "dependency_based"


class QualityLevel(Enum):
    """Quality assurance levels."""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"


@dataclass
class ContentPiece:
    """Represents a piece of content to be orchestrated."""
    id: str
    title: str
    content_type: ContentType
    priority: int = 3  # 1-5 scale
    dependencies: List[str] = field(default_factory=list)
    target_audience: str = "developers"
    complexity_level: int = 3  # 1-5 scale
    estimated_effort: int = 60  # minutes
    quality_requirements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: QualityLevel = QualityLevel.DRAFT


@dataclass
class OrchestrationPlan:
    """Plan for orchestrating content generation."""
    plan_id: str
    strategy: OrchestrationStrategy
    content_pieces: List[ContentPiece] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    estimated_duration: int = 0  # minutes
    quality_gates: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Result of content orchestration."""
    plan_id: str
    execution_time: float
    completed_pieces: List[str] = field(default_factory=list)
    failed_pieces: List[str] = field(default_factory=list)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    generated_content: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class IntelligentContentOrchestrator:
    """
    AI-powered content orchestrator that intelligently coordinates
    documentation generation across all patterns and frameworks.
    """
    
    def __init__(self):
        """Initialize intelligent content orchestrator."""
        self.content_queue = []
        self.active_plans = {}
        self.completed_results = {}
        self.quality_assessors = self._initialize_quality_assessors()
        self.routing_intelligence = self._initialize_routing_intelligence()
        self.performance_monitor = self._initialize_performance_monitor()
        
        logger.info("Intelligent content orchestrator initialized")
        
    async def orchestrate_content_generation(self, 
                                           content_pieces: List[ContentPiece],
                                           strategy: OrchestrationStrategy = OrchestrationStrategy.ADAPTIVE) -> OrchestrationResult:
        """Orchestrate generation of multiple content pieces."""
        logger.info(f"Starting orchestration of {len(content_pieces)} content pieces")
        
        # Create orchestration plan
        plan = self._create_orchestration_plan(content_pieces, strategy)
        self.active_plans[plan.plan_id] = plan
        
        # Execute plan based on strategy
        if strategy == OrchestrationStrategy.SEQUENTIAL:
            result = await self._execute_sequential_plan(plan)
        elif strategy == OrchestrationStrategy.PARALLEL:
            result = await self._execute_parallel_plan(plan)
        elif strategy == OrchestrationStrategy.ADAPTIVE:
            result = await self._execute_adaptive_plan(plan)
        elif strategy == OrchestrationStrategy.INTELLIGENT_ROUTING:
            result = await self._execute_intelligent_routing_plan(plan)
        else:
            result = await self._execute_dependency_based_plan(plan)
            
        # Store result
        self.completed_results[plan.plan_id] = result
        
        logger.info(f"Orchestration complete: {len(result.completed_pieces)} successful, {len(result.failed_pieces)} failed")
        return result
        
    def analyze_content_requirements(self, project_context: Dict[str, Any]) -> List[ContentPiece]:
        """Analyze project and automatically generate content requirements."""
        logger.info("Analyzing content requirements from project context")
        
        content_pieces = []
        
        # Analyze project type
        project_type = project_context.get("type", "general")
        complexity = project_context.get("complexity_level", 3)
        target_users = project_context.get("target_users", ["developers"])
        
        # Generate required documentation based on analysis
        required_docs = self._determine_required_documentation(project_type, complexity, target_users)
        
        for doc_type, requirements in required_docs.items():
            content_piece = ContentPiece(
                id=f"{doc_type}_{len(content_pieces)}",
                title=requirements["title"],
                content_type=ContentType(doc_type),
                priority=requirements["priority"],
                complexity_level=complexity,
                estimated_effort=requirements["effort"],
                quality_requirements=requirements["quality_requirements"],
                target_audience=requirements["audience"]
            )
            content_pieces.append(content_piece)
            
        logger.info(f"Generated {len(content_pieces)} content requirements")
        return content_pieces
        
    def optimize_content_flow(self, content_pieces: List[ContentPiece]) -> OrchestrationPlan:
        """Optimize content generation flow using AI analysis."""
        logger.info("Optimizing content generation flow")
        
        # Analyze dependencies
        dependency_graph = self._build_dependency_graph(content_pieces)
        
        # Calculate optimal execution order
        optimal_order = self._calculate_optimal_order(dependency_graph, content_pieces)
        
        # Identify parallel opportunities
        parallel_groups = self._identify_parallel_groups(optimal_order, dependency_graph)
        
        # Estimate duration
        total_duration = self._estimate_total_duration(content_pieces, parallel_groups)
        
        # Create optimized plan
        plan = OrchestrationPlan(
            plan_id=f"optimized_{len(self.active_plans)}",
            strategy=OrchestrationStrategy.ADAPTIVE,
            content_pieces=content_pieces,
            execution_order=optimal_order,
            parallel_groups=parallel_groups,
            estimated_duration=total_duration,
            quality_gates=self._generate_quality_gates(content_pieces)
        )
        
        logger.info(f"Optimized plan created: {len(parallel_groups)} parallel groups, {total_duration} min duration")
        return plan
        
    async def monitor_quality_continuously(self, plan_id: str) -> Dict[str, Any]:
        """Continuously monitor quality during content generation."""
        if plan_id not in self.active_plans:
            return {"error": "Plan not found"}
            
        plan = self.active_plans[plan_id]
        quality_report = {
            "plan_id": plan_id,
            "overall_quality": 0.0,
            "piece_qualities": {},
            "quality_gates_passed": 0,
            "recommendations": []
        }
        
        # Monitor each content piece
        for piece in plan.content_pieces:
            piece_quality = await self._assess_content_quality(piece)
            quality_report["piece_qualities"][piece.id] = piece_quality
            
            # Generate recommendations if quality is low
            if piece_quality["score"] < 0.7:
                recommendations = self._generate_quality_recommendations(piece, piece_quality)
                quality_report["recommendations"].extend(recommendations)
                
        # Calculate overall quality
        quality_scores = [q["score"] for q in quality_report["piece_qualities"].values()]
        quality_report["overall_quality"] = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Check quality gates
        passed_gates = sum(1 for gate in plan.quality_gates 
                          if self._check_quality_gate(gate, quality_report))
        quality_report["quality_gates_passed"] = passed_gates
        
        return quality_report
        
    def suggest_content_improvements(self, content_id: str, current_content: str) -> List[Dict[str, Any]]:
        """AI-powered suggestions for content improvements."""
        logger.info(f"Generating improvement suggestions for content: {content_id}")
        
        suggestions = []
        
        # Analyze content structure
        structure_issues = self._analyze_content_structure(current_content)
        for issue in structure_issues:
            suggestions.append({
                "type": "structure",
                "severity": issue["severity"],
                "description": issue["description"],
                "suggestion": issue["fix"],
                "confidence": issue["confidence"]
            })
            
        # Analyze content clarity
        clarity_issues = self._analyze_content_clarity(current_content)
        for issue in clarity_issues:
            suggestions.append({
                "type": "clarity",
                "severity": issue["severity"],
                "description": issue["description"],
                "suggestion": issue["fix"],
                "confidence": issue["confidence"]
            })
            
        # Analyze completeness
        completeness_issues = self._analyze_content_completeness(current_content, content_id)
        for issue in completeness_issues:
            suggestions.append({
                "type": "completeness",
                "severity": issue["severity"],
                "description": issue["description"],
                "suggestion": issue["fix"],
                "confidence": issue["confidence"]
            })
            
        # Sort by priority and confidence
        suggestions.sort(key=lambda x: (x["severity"] == "high", x["confidence"]), reverse=True)
        
        logger.info(f"Generated {len(suggestions)} improvement suggestions")
        return suggestions
        
    async def _execute_sequential_plan(self, plan: OrchestrationPlan) -> OrchestrationResult:
        """Execute plan sequentially."""
        import time
        start_time = time.time()
        
        result = OrchestrationResult(
            plan_id=plan.plan_id,
            execution_time=0
        )
        
        for piece_id in plan.execution_order:
            piece = next(p for p in plan.content_pieces if p.id == piece_id)
            try:
                # Generate content for piece
                generated_content = await self._generate_content_piece(piece)
                result.generated_content[piece_id] = generated_content
                result.completed_pieces.append(piece_id)
                
                # Assess quality
                quality = await self._assess_content_quality(piece)
                result.quality_scores[piece_id] = quality["score"]
                
            except Exception as e:
                logger.error(f"Failed to generate content for {piece_id}: {e}")
                result.failed_pieces.append(piece_id)
                
        result.execution_time = time.time() - start_time
        return result
        
    async def _execute_parallel_plan(self, plan: OrchestrationPlan) -> OrchestrationResult:
        """Execute plan with maximum parallelization."""
        import time
        start_time = time.time()
        
        result = OrchestrationResult(
            plan_id=plan.plan_id,
            execution_time=0
        )
        
        # Create tasks for all pieces
        tasks = []
        for piece in plan.content_pieces:
            task = asyncio.create_task(self._generate_and_assess_content(piece))
            tasks.append((piece.id, task))
            
        # Execute all tasks in parallel
        for piece_id, task in tasks:
            try:
                generated_content, quality_score = await task
                result.generated_content[piece_id] = generated_content
                result.quality_scores[piece_id] = quality_score
                result.completed_pieces.append(piece_id)
                
            except Exception as e:
                logger.error(f"Failed to generate content for {piece_id}: {e}")
                result.failed_pieces.append(piece_id)
                
        result.execution_time = time.time() - start_time
        return result
        
    async def _execute_adaptive_plan(self, plan: OrchestrationPlan) -> OrchestrationResult:
        """Execute plan with adaptive strategy based on real-time analysis."""
        import time
        start_time = time.time()
        
        result = OrchestrationResult(
            plan_id=plan.plan_id,
            execution_time=0
        )
        
        remaining_pieces = list(plan.content_pieces)
        completed_ids = set()
        
        while remaining_pieces:
            # Analyze current situation and adapt strategy
            ready_pieces = [p for p in remaining_pieces 
                          if all(dep in completed_ids for dep in p.dependencies)]
            
            if not ready_pieces:
                # Handle deadlock - force execution of highest priority piece
                ready_pieces = [max(remaining_pieces, key=lambda x: x.priority)]
                
            # Decide whether to execute in parallel or sequential
            if len(ready_pieces) > 1 and self._should_parallelize(ready_pieces):
                # Parallel execution
                tasks = [(p.id, asyncio.create_task(self._generate_and_assess_content(p))) 
                        for p in ready_pieces]
                
                for piece_id, task in tasks:
                    try:
                        generated_content, quality_score = await task
                        result.generated_content[piece_id] = generated_content
                        result.quality_scores[piece_id] = quality_score
                        result.completed_pieces.append(piece_id)
                        completed_ids.add(piece_id)
                        
                    except Exception as e:
                        logger.error(f"Failed to generate content for {piece_id}: {e}")
                        result.failed_pieces.append(piece_id)
                        
            else:
                # Sequential execution of highest priority piece
                piece = max(ready_pieces, key=lambda x: x.priority)
                try:
                    generated_content, quality_score = await self._generate_and_assess_content(piece)
                    result.generated_content[piece.id] = generated_content
                    result.quality_scores[piece.id] = quality_score
                    result.completed_pieces.append(piece.id)
                    completed_ids.add(piece.id)
                    
                except Exception as e:
                    logger.error(f"Failed to generate content for {piece.id}: {e}")
                    result.failed_pieces.append(piece.id)
                    
            # Remove completed pieces
            remaining_pieces = [p for p in remaining_pieces if p.id not in completed_ids]
            
        result.execution_time = time.time() - start_time
        return result
        
    async def _execute_intelligent_routing_plan(self, plan: OrchestrationPlan) -> OrchestrationResult:
        """Execute plan using intelligent routing of content pieces."""
        # Route each piece to optimal generator based on AI analysis
        routing_decisions = {}
        
        for piece in plan.content_pieces:
            optimal_route = self._determine_optimal_route(piece)
            routing_decisions[piece.id] = optimal_route
            
        # Execute with intelligent routing
        return await self._execute_routed_plan(plan, routing_decisions)
        
    async def _generate_content_piece(self, piece: ContentPiece) -> str:
        """Generate content for a specific piece."""
        # Mock content generation - would use actual generators
        content = f"""# {piece.title}

Generated {piece.content_type.value} documentation.

**Target Audience:** {piece.target_audience}
**Complexity Level:** {piece.complexity_level}/5
**Priority:** {piece.priority}/5

## Content

This is generated content for {piece.title}.
It follows the patterns determined by the intelligent orchestrator.

## Quality Requirements

{chr(10).join(f"- {req}" for req in piece.quality_requirements)}
"""
        
        # Simulate generation time
        await asyncio.sleep(0.1)
        
        return content
        
    async def _generate_and_assess_content(self, piece: ContentPiece) -> Tuple[str, float]:
        """Generate content and assess its quality."""
        content = await self._generate_content_piece(piece)
        quality_assessment = await self._assess_content_quality(piece)
        return content, quality_assessment["score"]
        
    async def _assess_content_quality(self, piece: ContentPiece) -> Dict[str, Any]:
        """Assess the quality of generated content."""
        # Mock quality assessment - would use actual AI assessment
        base_score = 0.8
        
        # Adjust based on complexity
        complexity_adjustment = (6 - piece.complexity_level) * 0.02
        
        # Adjust based on requirements met
        requirements_met = len(piece.quality_requirements) * 0.02
        
        final_score = min(1.0, base_score + complexity_adjustment + requirements_met)
        
        return {
            "score": final_score,
            "components": {
                "structure": final_score * 0.9,
                "clarity": final_score * 0.95,
                "completeness": final_score * 0.85,
                "accuracy": final_score * 0.92
            },
            "recommendations": []
        }
        
    def _create_orchestration_plan(self, content_pieces: List[ContentPiece], strategy: OrchestrationStrategy) -> OrchestrationPlan:
        """Create orchestration plan from content pieces."""
        plan_id = f"plan_{len(self.active_plans)}"
        
        # Calculate execution order based on dependencies and priorities
        execution_order = self._calculate_execution_order(content_pieces)
        
        # Identify parallel groups
        parallel_groups = self._identify_parallel_groups(execution_order, self._build_dependency_graph(content_pieces))
        
        # Estimate duration
        estimated_duration = sum(piece.estimated_effort for piece in content_pieces)
        if parallel_groups:
            # Adjust for parallelization
            estimated_duration = int(estimated_duration * 0.7)  # 30% efficiency gain
            
        return OrchestrationPlan(
            plan_id=plan_id,
            strategy=strategy,
            content_pieces=content_pieces,
            execution_order=execution_order,
            parallel_groups=parallel_groups,
            estimated_duration=estimated_duration,
            quality_gates=[f"Quality score > 0.7 for all pieces", 
                          f"All dependencies satisfied",
                          f"Target audience requirements met"]
        )
        
    def _determine_required_documentation(self, project_type: str, complexity: int, target_users: List[str]) -> Dict[str, Dict[str, Any]]:
        """Determine required documentation based on project analysis."""
        base_requirements = {
            "api_references": {
                "title": "API Reference Documentation",
                "priority": 5,
                "effort": 120,
                "quality_requirements": ["Complete API coverage", "Code examples", "Error handling"],
                "audience": "developers"
            },
            "user_guides": {
                "title": "User Guide",
                "priority": 4,
                "effort": 90,
                "quality_requirements": ["Step-by-step instructions", "Screenshots", "Troubleshooting"],
                "audience": "end_users"
            },
            "tutorials": {
                "title": "Getting Started Tutorial",
                "priority": 4,
                "effort": 60,
                "quality_requirements": ["Progressive complexity", "Working examples", "Clear explanations"],
                "audience": "beginners"
            }
        }
        
        # Adjust based on complexity
        if complexity >= 4:
            base_requirements["architecture_docs"] = {
                "title": "Architecture Documentation",
                "priority": 5,
                "effort": 180,
                "quality_requirements": ["System diagrams", "Component descriptions", "Integration details"],
                "audience": "architects"
            }
            
        # Adjust based on target users
        if "operators" in target_users:
            base_requirements["deployment_guides"] = {
                "title": "Deployment Guide",
                "priority": 4,
                "effort": 150,
                "quality_requirements": ["Production readiness", "Monitoring setup", "Troubleshooting"],
                "audience": "operators"
            }
            
        return base_requirements
        
    def _calculate_execution_order(self, content_pieces: List[ContentPiece]) -> List[str]:
        """Calculate optimal execution order based on dependencies and priorities."""
        # Simple topological sort with priority weighting
        remaining = {piece.id: piece for piece in content_pieces}
        ordered = []
        
        while remaining:
            # Find pieces with satisfied dependencies
            ready = [piece for piece in remaining.values() 
                    if all(dep in ordered for dep in piece.dependencies)]
            
            if not ready:
                # Handle circular dependencies - take highest priority
                ready = [max(remaining.values(), key=lambda x: x.priority)]
                
            # Sort by priority and take the highest
            next_piece = max(ready, key=lambda x: x.priority)
            ordered.append(next_piece.id)
            del remaining[next_piece.id]
            
        return ordered
        
    def _build_dependency_graph(self, content_pieces: List[ContentPiece]) -> Dict[str, List[str]]:
        """Build dependency graph from content pieces."""
        graph = {}
        for piece in content_pieces:
            graph[piece.id] = piece.dependencies
        return graph
        
    def _identify_parallel_groups(self, execution_order: List[str], dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Identify groups that can be executed in parallel."""
        parallel_groups = []
        processed = set()
        
        for piece_id in execution_order:
            if piece_id in processed:
                continue
                
            # Find all pieces that can run in parallel with this one
            group = [piece_id]
            dependencies = set(dependency_graph[piece_id])
            
            for other_id in execution_order:
                if other_id != piece_id and other_id not in processed:
                    other_deps = set(dependency_graph[other_id])
                    
                    # Can run in parallel if no dependency conflicts
                    if not dependencies.intersection({other_id}) and not other_deps.intersection({piece_id}):
                        group.append(other_id)
                        dependencies.update(other_deps)
                        
            parallel_groups.append(group)
            processed.update(group)
            
        return parallel_groups
        
    def _initialize_quality_assessors(self) -> Dict[str, Callable]:
        """Initialize quality assessment functions."""
        return {
            "structure": self._assess_structure_quality,
            "clarity": self._assess_clarity_quality,
            "completeness": self._assess_completeness_quality,
            "accuracy": self._assess_accuracy_quality
        }
        
    def _initialize_routing_intelligence(self) -> Dict[str, Any]:
        """Initialize intelligent routing system."""
        return {
            "api_references": "auto_api_docs_generator",
            "tutorials": "recipe_based_learning", 
            "cookbooks": "cookbook_organization_manager",
            "architecture": "design_first_docs"
        }
        
    def _initialize_performance_monitor(self) -> Dict[str, Any]:
        """Initialize performance monitoring system."""
        return {
            "metrics": {},
            "thresholds": {
                "max_generation_time": 300,  # 5 minutes
                "min_quality_score": 0.7,
                "max_parallel_tasks": 10
            }
        }
        
    def _estimate_total_duration(self, content_pieces: List[ContentPiece], parallel_groups: List[List[str]]) -> int:
        """Estimate total duration considering parallelization."""
        if not parallel_groups:
            return sum(piece.estimated_effort for piece in content_pieces)
            
        total_duration = 0
        piece_lookup = {piece.id: piece for piece in content_pieces}
        
        for group in parallel_groups:
            # Duration of group is the maximum duration of pieces in the group
            group_duration = max(piece_lookup[piece_id].estimated_effort for piece_id in group)
            total_duration += group_duration
            
        return total_duration
        
    def _generate_quality_gates(self, content_pieces: List[ContentPiece]) -> List[str]:
        """Generate quality gates for content pieces."""
        gates = [
            "All content pieces completed successfully",
            "Minimum quality score of 0.7 achieved",
            "All dependencies properly satisfied",
            "Target audience requirements met"
        ]
        
        # Add specific gates based on content types
        content_types = {piece.content_type for piece in content_pieces}
        
        if ContentType.API_REFERENCES in content_types:
            gates.append("API documentation completeness verified")
            
        if ContentType.TUTORIALS in content_types:
            gates.append("Tutorial progression validated")
            
        return gates