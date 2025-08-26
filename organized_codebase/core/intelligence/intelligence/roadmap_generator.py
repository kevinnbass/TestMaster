"""
Roadmap Generator Module
Creates comprehensive product roadmaps with long-term and short-term goals
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json


class GoalType(Enum):
    """Types of roadmap goals"""
    LONG_TERM = "long_term"        # Strategic, vision-oriented goals
    SHORT_TERM = "short_term"      # Tactical, execution-oriented goals
    MILESTONE = "milestone"        # Specific deliverables
    FEATURE = "feature"           # Product features
    IMPROVEMENT = "improvement"    # Enhancements to existing features
    TECHNICAL = "technical"        # Technical debt, infrastructure
    RESEARCH = "research"          # R&D initiatives


class Priority(Enum):
    """Priority levels for roadmap items"""
    CRITICAL = "critical"    # Must have, blocks other work
    HIGH = "high"           # Important for success
    MEDIUM = "medium"       # Nice to have, good impact
    LOW = "low"            # Future consideration


class Status(Enum):
    """Status of roadmap items"""
    PLANNED = "planned"         # Future work
    IN_PROGRESS = "in_progress" # Currently being worked on
    COMPLETED = "completed"     # Done
    CANCELLED = "cancelled"     # Cancelled/deprioritized
    ON_HOLD = "on_hold"        # Temporarily paused


@dataclass
class RoadmapItem:
    """Individual roadmap item"""
    id: str
    title: str
    description: str
    goal_type: GoalType
    priority: Priority = Priority.MEDIUM
    status: Status = Status.PLANNED
    category: str = ""
    estimated_effort: str = ""  # e.g., "2 weeks", "1 month"
    target_date: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # IDs of dependent items
    deliverables: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    assigned_to: str = ""
    tags: List[str] = field(default_factory=list)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat()[:10])


@dataclass
class RoadmapVersion:
    """Version of the roadmap"""
    version: str
    release_date: str
    title: str = ""
    description: str = ""
    goals: List[RoadmapItem] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    focus_areas: List[str] = field(default_factory=list)


class RoadmapGenerator:
    """Generates comprehensive product roadmaps"""
    
    def __init__(self, product_name: str = "Product"):
        self.product_name = product_name
        self.versions = {}
        self.items = {}
        self.categories = {
            "features": "✨ New Features",
            "tools": "🛠️ Tool/MCP Integration",
            "memory": "💾 Memory Management",
            "agents": "🤖 Agent System",
            "development": "👨‍💻 Development",
            "evaluation": "🔍 Evaluation",
            "deployment": "🏗️ Deployment",
            "ui": "🎨 User Interface",
            "docs": "📖 Documentation",
            "performance": "⚡ Performance",
            "security": "🔒 Security",
            "infrastructure": "🏢 Infrastructure"
        }
        
        self.status_icons = {
            Status.PLANNED: "📋",
            Status.IN_PROGRESS: "🚧", 
            Status.COMPLETED: "✅",
            Status.CANCELLED: "❌",
            Status.ON_HOLD: "⏸️"
        }
        
        self.priority_icons = {
            Priority.CRITICAL: "🔴",
            Priority.HIGH: "🟠",
            Priority.MEDIUM: "🟡",
            Priority.LOW: "🟢"
        }
    
    def add_roadmap_version(self, version: RoadmapVersion) -> None:
        """Add a roadmap version"""
        self.versions[version.version] = version
        
        # Add all items to the global items dict
        for item in version.goals:
            self.items[item.id] = item
    
    def add_item(self, version: str, item: RoadmapItem) -> None:
        """Add an item to a specific roadmap version"""
        if version not in self.versions:
            self.versions[version] = RoadmapVersion(
                version=version,
                release_date=datetime.now().isoformat()[:10]
            )
        
        self.versions[version].goals.append(item)
        self.items[item.id] = item
    
    def create_long_term_goal(self, title: str, description: str, **kwargs) -> RoadmapItem:
        """Create a long-term strategic goal"""
        item_id = self._generate_item_id(title)
        return RoadmapItem(
            id=item_id,
            title=title,
            description=description,
            goal_type=GoalType.LONG_TERM,
            priority=Priority.HIGH,
            **kwargs
        )
    
    def create_short_term_goal(self, title: str, description: str, **kwargs) -> RoadmapItem:
        """Create a short-term tactical goal"""
        item_id = self._generate_item_id(title)
        return RoadmapItem(
            id=item_id,
            title=title,
            description=description,
            goal_type=GoalType.SHORT_TERM,
            **kwargs
        )
    
    def create_feature_roadmap(self, features: List[Dict[str, Any]], version: str) -> RoadmapVersion:
        """Create a feature-focused roadmap version"""
        roadmap_version = RoadmapVersion(
            version=version,
            release_date=datetime.now().isoformat()[:10],
            title=f"{self.product_name} {version} Roadmap",
            focus_areas=["Feature Development", "User Experience", "Performance"]
        )
        
        for feature_data in features:
            item = RoadmapItem(
                id=self._generate_item_id(feature_data["title"]),
                title=feature_data["title"],
                description=feature_data.get("description", ""),
                goal_type=GoalType.FEATURE,
                category=feature_data.get("category", "features"),
                priority=Priority(feature_data.get("priority", "medium")),
                estimated_effort=feature_data.get("effort", ""),
                target_date=feature_data.get("target_date"),
                deliverables=feature_data.get("deliverables", []),
                success_metrics=feature_data.get("metrics", [])
            )
            roadmap_version.goals.append(item)
        
        return roadmap_version
    
    def generate_markdown_roadmap(self, version: str = None) -> str:
        """Generate markdown roadmap document"""
        lines = []
        
        # Header
        lines.append(f"# {self.product_name} Roadmap")
        lines.append("")
        
        if version and version in self.versions:
            # Single version roadmap
            roadmap_version = self.versions[version]
            lines.extend(self._generate_version_section(roadmap_version))
        else:
            # All versions roadmap
            # Sort versions by release date
            sorted_versions = sorted(
                self.versions.values(),
                key=lambda v: v.release_date,
                reverse=True
            )
            
            for roadmap_version in sorted_versions:
                lines.extend(self._generate_version_section(roadmap_version))
                lines.append("")
        
        return "\n".join(lines)
    
    def _generate_version_section(self, roadmap_version: RoadmapVersion) -> List[str]:
        """Generate markdown section for a roadmap version"""
        lines = []
        
        # Version header
        lines.append(f"## {roadmap_version.title or roadmap_version.version}")
        lines.append("")
        
        if roadmap_version.description:
            lines.append(roadmap_version.description)
            lines.append("")
        
        # Long-term Goals
        long_term_items = [item for item in roadmap_version.goals if item.goal_type == GoalType.LONG_TERM]
        if long_term_items:
            lines.append("## Long-term Goals")
            lines.append("")
            for item in long_term_items:
                lines.append(item.description)
                lines.append("")
        
        # Short-term Goals with detailed breakdown
        short_term_items = [item for item in roadmap_version.goals if item.goal_type == GoalType.SHORT_TERM]
        if short_term_items:
            lines.append("## Short-term Goals")
            lines.append("")
            lines.extend(self._generate_detailed_items_section(short_term_items))
        
        # Group other items by category
        other_items = [item for item in roadmap_version.goals 
                      if item.goal_type not in [GoalType.LONG_TERM, GoalType.SHORT_TERM]]
        
        if other_items:
            lines.extend(self._generate_categorized_items(other_items))
        
        # Focus areas
        if roadmap_version.focus_areas:
            lines.append("## 🎯 Focus Areas")
            lines.append("")
            for area in roadmap_version.focus_areas:
                lines.append(f"- {area}")
            lines.append("")
        
        return lines
    
    def _generate_detailed_items_section(self, items: List[RoadmapItem]) -> List[str]:
        """Generate detailed section for roadmap items"""
        lines = []
        
        for item in items:
            # Item header with status and priority
            status_icon = self.status_icons.get(item.status, "📋")
            priority_icon = self.priority_icons.get(item.priority, "🟡")
            
            lines.append(f"### {status_icon} {item.title}")
            lines.append("")
            lines.append(item.description)
            lines.append("")
            
            # Item details
            details = []
            if item.priority != Priority.MEDIUM:
                details.append(f"**Priority:** {priority_icon} {item.priority.value.title()}")
            
            if item.estimated_effort:
                details.append(f"**Estimated Effort:** {item.estimated_effort}")
            
            if item.target_date:
                details.append(f"**Target Date:** {item.target_date}")
            
            if item.assigned_to:
                details.append(f"**Assigned To:** {item.assigned_to}")
            
            if details:
                lines.extend(details)
                lines.append("")
            
            # Deliverables
            if item.deliverables:
                lines.append("**Deliverables:**")
                for deliverable in item.deliverables:
                    lines.append(f"- {deliverable}")
                lines.append("")
            
            # Success metrics
            if item.success_metrics:
                lines.append("**Success Metrics:**")
                for metric in item.success_metrics:
                    lines.append(f"- {metric}")
                lines.append("")
            
            # Dependencies
            if item.dependencies:
                lines.append("**Dependencies:**")
                for dep_id in item.dependencies:
                    dep_item = self.items.get(dep_id)
                    if dep_item:
                        lines.append(f"- {dep_item.title} ({dep_id})")
                    else:
                        lines.append(f"- {dep_id}")
                lines.append("")
        
        return lines
    
    def _generate_categorized_items(self, items: List[RoadmapItem]) -> List[str]:
        """Generate items grouped by category"""
        lines = []
        
        # Group by category
        by_category = {}
        for item in items:
            category = item.category or "general"
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(item)
        
        # Generate sections for each category
        for category, category_items in by_category.items():
            category_title = self.categories.get(category, category.title())
            lines.append(f"## {category_title}")
            lines.append("")
            
            for item in category_items:
                status_icon = self.status_icons.get(item.status, "📋")
                priority_icon = self.priority_icons.get(item.priority, "🟡")
                
                line = f"- {status_icon} **{item.title}**"
                if item.priority in [Priority.CRITICAL, Priority.HIGH]:
                    line += f" {priority_icon}"
                
                if item.estimated_effort:
                    line += f" *(~{item.estimated_effort})*"
                
                lines.append(line)
                
                if item.description:
                    lines.append(f"  - {item.description}")
            
            lines.append("")
        
        return lines
    
    def generate_timeline_view(self, start_date: str, end_date: str) -> str:
        """Generate timeline view of roadmap items"""
        lines = []
        
        lines.append("# Roadmap Timeline")
        lines.append("")
        lines.append(f"**Timeline:** {start_date} to {end_date}")
        lines.append("")
        
        # Group items by target date
        items_with_dates = [item for item in self.items.values() if item.target_date]
        items_with_dates.sort(key=lambda x: x.target_date or "9999-12-31")
        
        current_month = ""
        for item in items_with_dates:
            if item.target_date:
                item_month = item.target_date[:7]  # YYYY-MM
                
                if item_month != current_month:
                    current_month = item_month
                    lines.append(f"## {current_month}")
                    lines.append("")
                
                status_icon = self.status_icons.get(item.status, "📋")
                priority_icon = self.priority_icons.get(item.priority, "🟡")
                
                line = f"- {status_icon} **{item.title}**"
                if item.priority in [Priority.CRITICAL, Priority.HIGH]:
                    line += f" {priority_icon}"
                
                lines.append(line)
                
                if item.description:
                    lines.append(f"  - {item.description}")
        
        return "\n".join(lines)
    
    def generate_status_report(self) -> str:
        """Generate status report of roadmap progress"""
        lines = []
        
        lines.append("# Roadmap Status Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        
        # Overall statistics
        total_items = len(self.items)
        completed_items = len([item for item in self.items.values() if item.status == Status.COMPLETED])
        in_progress_items = len([item for item in self.items.values() if item.status == Status.IN_PROGRESS])
        
        completion_rate = (completed_items / total_items * 100) if total_items > 0 else 0
        
        lines.append("## 📊 Overall Progress")
        lines.append("")
        lines.append(f"- **Total Items:** {total_items}")
        lines.append(f"- **Completed:** {completed_items} ({completion_rate:.1f}%)")
        lines.append(f"- **In Progress:** {in_progress_items}")
        lines.append(f"- **Planned:** {total_items - completed_items - in_progress_items}")
        lines.append("")
        
        # Progress by category
        by_category = {}
        for item in self.items.values():
            category = item.category or "general"
            if category not in by_category:
                by_category[category] = {"total": 0, "completed": 0}
            
            by_category[category]["total"] += 1
            if item.status == Status.COMPLETED:
                by_category[category]["completed"] += 1
        
        lines.append("## 📈 Progress by Category")
        lines.append("")
        
        for category, stats in by_category.items():
            category_title = self.categories.get(category, category.title())
            completion = (stats["completed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            lines.append(f"- **{category_title}:** {stats['completed']}/{stats['total']} ({completion:.0f}%)")
        
        lines.append("")
        
        # Overdue items
        today = datetime.now().strftime('%Y-%m-%d')
        overdue_items = [
            item for item in self.items.values() 
            if item.target_date and item.target_date < today and item.status not in [Status.COMPLETED, Status.CANCELLED]
        ]
        
        if overdue_items:
            lines.append("## ⚠️ Overdue Items")
            lines.append("")
            for item in overdue_items:
                lines.append(f"- **{item.title}** (due: {item.target_date})")
            lines.append("")
        
        return "\n".join(lines)
    
    def export_to_json(self, version: str = None) -> str:
        """Export roadmap to JSON format"""
        if version and version in self.versions:
            roadmap_data = {
                "product": self.product_name,
                "version": version,
                "roadmap": self._version_to_dict(self.versions[version])
            }
        else:
            roadmap_data = {
                "product": self.product_name,
                "versions": [self._version_to_dict(v) for v in self.versions.values()],
                "total_items": len(self.items)
            }
        
        return json.dumps(roadmap_data, indent=2)
    
    def _version_to_dict(self, version: RoadmapVersion) -> Dict[str, Any]:
        """Convert roadmap version to dictionary"""
        return {
            "version": version.version,
            "release_date": version.release_date,
            "title": version.title,
            "description": version.description,
            "themes": version.themes,
            "focus_areas": version.focus_areas,
            "goals": [self._item_to_dict(item) for item in version.goals]
        }
    
    def _item_to_dict(self, item: RoadmapItem) -> Dict[str, Any]:
        """Convert roadmap item to dictionary"""
        return {
            "id": item.id,
            "title": item.title,
            "description": item.description,
            "goal_type": item.goal_type.value,
            "priority": item.priority.value,
            "status": item.status.value,
            "category": item.category,
            "estimated_effort": item.estimated_effort,
            "target_date": item.target_date,
            "dependencies": item.dependencies,
            "deliverables": item.deliverables,
            "success_metrics": item.success_metrics,
            "assigned_to": item.assigned_to,
            "tags": item.tags,
            "created_date": item.created_date
        }
    
    def _generate_item_id(self, title: str) -> str:
        """Generate unique ID from title"""
        import re
        clean_title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
        base_id = clean_title.lower().replace(' ', '_')[:20]
        
        # Ensure uniqueness
        counter = 1
        item_id = base_id
        while item_id in self.items:
            item_id = f"{base_id}_{counter}"
            counter += 1
        
        return item_id
    
    def analyze_dependencies(self) -> Dict[str, List[str]]:
        """Analyze and return dependency chains"""
        dependency_chains = {}
        
        for item in self.items.values():
            if item.dependencies:
                chains = []
                self._find_dependency_chains(item.id, [], chains)
                if chains:
                    dependency_chains[item.id] = chains
        
        return dependency_chains
    
    def _find_dependency_chains(self, item_id: str, current_chain: List[str], 
                               all_chains: List[List[str]]) -> None:
        """Recursively find dependency chains"""
        if item_id in current_chain:  # Circular dependency
            return
        
        item = self.items.get(item_id)
        if not item or not item.dependencies:
            if current_chain:
                all_chains.append(current_chain + [item_id])
            return
        
        for dep_id in item.dependencies:
            self._find_dependency_chains(dep_id, current_chain + [item_id], all_chains)