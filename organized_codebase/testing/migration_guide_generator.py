"""
Migration Guide Generator Module
Creates comprehensive migration guides and upgrade documentation
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import json
from pathlib import Path
from datetime import datetime


class MigrationType(Enum):
    """Types of migrations supported"""
    VERSION_UPGRADE = "version_upgrade"
    FRAMEWORK_MIGRATION = "framework_migration"
    API_BREAKING_CHANGE = "api_breaking_change"
    DEPENDENCY_UPDATE = "dependency_update"
    CONFIGURATION_CHANGE = "configuration_change"


class ChangeType(Enum):
    """Types of changes in migration"""
    BREAKING = "breaking"
    DEPRECATED = "deprecated"
    NEW_FEATURE = "new_feature"
    IMPROVEMENT = "improvement"
    BUG_FIX = "bug_fix"
    SECURITY = "security"


@dataclass
class ChangeItem:
    """Represents a single change item"""
    id: str
    type: ChangeType
    title: str
    description: str
    impact: str = "medium"  # low, medium, high, critical
    affected_components: List[str] = field(default_factory=list)
    before_code: str = ""
    after_code: str = ""
    migration_steps: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class MigrationGuide:
    """Represents a complete migration guide"""
    id: str
    title: str
    description: str
    migration_type: MigrationType
    source_version: str
    target_version: str
    estimated_time: str = "1-2 hours"
    difficulty: str = "intermediate"
    changes: List[ChangeItem] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    post_migration_steps: List[str] = field(default_factory=list)
    troubleshooting: List[Dict[str, str]] = field(default_factory=list)
    resources: List[Dict[str, str]] = field(default_factory=list)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


class MigrationGuideGenerator:
    """Generates migration guides from changelog and version data"""
    
    def __init__(self):
        self.guides = {}
        self.templates = self.load_templates()
    
    def load_templates(self) -> Dict[str, str]:
        """Load migration guide templates"""
        return {
            "version_upgrade": """
# Migration Guide: {title}

## Overview
{description}

**Migration Path:** {source_version} → {target_version}
**Estimated Time:** {estimated_time}
**Difficulty:** {difficulty}

## Prerequisites
{prerequisites_list}

## Breaking Changes
{breaking_changes}

## Migration Steps
{migration_steps}

## Post-Migration Verification
{post_migration_steps}

## Troubleshooting
{troubleshooting_section}

## Additional Resources
{resources_section}
            """.strip(),
            
            "api_changes": """
### {change_title}

**Impact:** {impact}
**Components Affected:** {components}

{description}

#### Before (v{source_version})
```{language}
{before_code}
```

#### After (v{target_version})
```{language}
{after_code}
```

#### Migration Steps
{migration_steps_list}

{notes_section}
            """.strip()
        }
    
    def create_migration_guide(self, source_version: str, target_version: str,
                             changes: List[Dict[str, Any]], 
                             migration_type: MigrationType = MigrationType.VERSION_UPGRADE) -> MigrationGuide:
        """Create a migration guide from version changes"""
        guide_id = f"migration_{source_version}_to_{target_version}".replace(".", "_")
        
        guide = MigrationGuide(
            id=guide_id,
            title=f"Migration Guide: {source_version} to {target_version}",
            description=f"Complete guide for migrating from version {source_version} to {target_version}",
            migration_type=migration_type,
            source_version=source_version,
            target_version=target_version
        )
        
        # Process changes
        for change_data in changes:
            change_item = self.create_change_item(change_data)
            guide.changes.append(change_item)
        
        # Analyze changes and set guide properties
        self.analyze_guide_complexity(guide)
        self.generate_prerequisites(guide)
        self.generate_post_migration_steps(guide)
        self.generate_troubleshooting_section(guide)
        
        self.guides[guide_id] = guide
        return guide
    
    def create_change_item(self, change_data: Dict[str, Any]) -> ChangeItem:
        """Create a change item from data"""
        change_type = self.determine_change_type(change_data)
        
        change_item = ChangeItem(
            id=change_data.get("id", self.generate_change_id(change_data.get("title", ""))),
            type=change_type,
            title=change_data.get("title", ""),
            description=change_data.get("description", ""),
            impact=change_data.get("impact", self.assess_impact(change_data, change_type)),
            affected_components=change_data.get("components", []),
            before_code=change_data.get("before_code", ""),
            after_code=change_data.get("after_code", ""),
            migration_steps=change_data.get("migration_steps", []),
            notes=change_data.get("notes", [])
        )
        
        # Generate migration steps if not provided
        if not change_item.migration_steps:
            change_item.migration_steps = self.generate_migration_steps(change_item)
        
        return change_item
    
    def determine_change_type(self, change_data: Dict[str, Any]) -> ChangeType:
        """Determine the type of change based on data"""
        title = change_data.get("title", "").lower()
        description = change_data.get("description", "").lower()
        content = f"{title} {description}"
        
        # Check for breaking changes
        breaking_keywords = ["breaking", "removed", "deprecated", "incompatible", "changed signature"]
        if any(keyword in content for keyword in breaking_keywords):
            if "deprecated" in content:
                return ChangeType.DEPRECATED
            return ChangeType.BREAKING
        
        # Check for security changes
        security_keywords = ["security", "vulnerability", "cve", "exploit"]
        if any(keyword in content for keyword in security_keywords):
            return ChangeType.SECURITY
        
        # Check for new features
        feature_keywords = ["added", "new", "introduced", "feature", "support for"]
        if any(keyword in content for keyword in feature_keywords):
            return ChangeType.NEW_FEATURE
        
        # Check for bug fixes
        bug_keywords = ["fixed", "bug", "issue", "error", "problem"]
        if any(keyword in content for keyword in bug_keywords):
            return ChangeType.BUG_FIX
        
        return ChangeType.IMPROVEMENT
    
    def assess_impact(self, change_data: Dict[str, Any], change_type: ChangeType) -> str:
        """Assess the impact level of a change"""
        if change_type == ChangeType.BREAKING:
            return "high"
        elif change_type == ChangeType.SECURITY:
            return "critical"
        elif change_type == ChangeType.DEPRECATED:
            return "medium"
        elif change_type == ChangeType.NEW_FEATURE:
            return "low"
        
        # Check for impact indicators in description
        description = change_data.get("description", "").lower()
        if any(word in description for word in ["major", "significant", "important"]):
            return "high"
        elif any(word in description for word in ["minor", "small", "slight"]):
            return "low"
        
        return "medium"
    
    def generate_migration_steps(self, change_item: ChangeItem) -> List[str]:
        """Generate migration steps for a change item"""
        steps = []
        
        if change_item.type == ChangeType.BREAKING:
            if change_item.before_code and change_item.after_code:
                steps.append(f"Update {change_item.title.lower()} usage")
                steps.append("Replace old syntax with new syntax")
                steps.append("Test the updated code")
            else:
                steps.append(f"Review and update {change_item.title.lower()}")
                steps.append("Update imports and references")
                steps.append("Verify functionality after changes")
        
        elif change_item.type == ChangeType.DEPRECATED:
            steps.append(f"Identify usage of deprecated {change_item.title.lower()}")
            steps.append("Plan replacement with recommended alternative")
            steps.append("Update code to use new approach")
            steps.append("Remove deprecated usage")
        
        elif change_item.type == ChangeType.NEW_FEATURE:
            steps.append(f"Review new {change_item.title.lower()} feature")
            steps.append("Determine if feature benefits your use case")
            steps.append("Optionally integrate new feature")
        
        else:
            steps.append(f"Review changes to {change_item.title.lower()}")
            steps.append("Update code if necessary")
            steps.append("Test updated functionality")
        
        return steps
    
    def analyze_guide_complexity(self, guide: MigrationGuide) -> None:
        """Analyze and set guide complexity properties"""
        breaking_changes = sum(1 for change in guide.changes if change.type == ChangeType.BREAKING)
        high_impact_changes = sum(1 for change in guide.changes if change.impact == "high")
        total_changes = len(guide.changes)
        
        # Determine difficulty
        if breaking_changes > 5 or high_impact_changes > 3:
            guide.difficulty = "advanced"
            guide.estimated_time = "4-6 hours"
        elif breaking_changes > 2 or high_impact_changes > 1:
            guide.difficulty = "intermediate"
            guide.estimated_time = "2-4 hours"
        else:
            guide.difficulty = "beginner"
            guide.estimated_time = "30 minutes - 1 hour"
    
    def generate_prerequisites(self, guide: MigrationGuide) -> None:
        """Generate prerequisites for migration"""
        guide.prerequisites = [
            f"Current version: {guide.source_version}",
            "Backup your project before starting migration",
            "Read through this entire guide before beginning",
            "Ensure you have sufficient time allocated for migration"
        ]
        
        # Add specific prerequisites based on changes
        breaking_changes = [c for c in guide.changes if c.type == ChangeType.BREAKING]
        if breaking_changes:
            guide.prerequisites.append("Review all breaking changes and affected code")
        
        security_changes = [c for c in guide.changes if c.type == ChangeType.SECURITY]
        if security_changes:
            guide.prerequisites.append("Plan for immediate deployment after migration")
    
    def generate_post_migration_steps(self, guide: MigrationGuide) -> None:
        """Generate post-migration verification steps"""
        guide.post_migration_steps = [
            "Run all tests to ensure functionality",
            "Check that all features work as expected",
            "Review logs for any errors or warnings",
            "Update documentation to reflect changes",
            "Deploy to staging environment for testing"
        ]
        
        # Add specific steps based on changes
        api_changes = any(
            "api" in change.title.lower() or "endpoint" in change.description.lower()
            for change in guide.changes
        )
        if api_changes:
            guide.post_migration_steps.insert(1, "Test all API endpoints")
    
    def generate_troubleshooting_section(self, guide: MigrationGuide) -> None:
        """Generate troubleshooting section"""
        common_issues = [
            {
                "problem": "Import errors after migration",
                "solution": "Check that all import statements are updated to new module paths"
            },
            {
                "problem": "Configuration not recognized",
                "solution": "Verify configuration format matches new version requirements"
            },
            {
                "problem": "Tests failing after migration", 
                "solution": "Update test assertions to match new API behavior"
            }
        ]
        
        # Add specific troubleshooting based on changes
        for change in guide.changes:
            if change.type == ChangeType.BREAKING and change.notes:
                for note in change.notes:
                    if "issue" in note.lower() or "problem" in note.lower():
                        common_issues.append({
                            "problem": f"Issue with {change.title}",
                            "solution": note
                        })
        
        guide.troubleshooting = common_issues
    
    def generate_change_id(self, title: str) -> str:
        """Generate ID from change title"""
        clean_title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
        return clean_title.lower().replace(' ', '_')
    
    def render_migration_guide(self, guide_id: str, format: str = "markdown") -> Optional[str]:
        """Render migration guide in specified format"""
        if guide_id not in self.guides:
            return None
        
        guide = self.guides[guide_id]
        
        if format.lower() == "markdown":
            return self.render_markdown(guide)
        elif format.lower() == "json":
            return json.dumps(self.guide_to_dict(guide), indent=2)
        
        return None
    
    def render_markdown(self, guide: MigrationGuide) -> str:
        """Render guide as markdown"""
        sections = []
        
        # Header
        sections.append(f"# {guide.title}")
        sections.append("")
        sections.append(guide.description)
        sections.append("")
        
        # Migration info
        sections.append("## Migration Information")
        sections.append("")
        sections.append(f"- **Source Version:** {guide.source_version}")
        sections.append(f"- **Target Version:** {guide.target_version}")
        sections.append(f"- **Estimated Time:** {guide.estimated_time}")
        sections.append(f"- **Difficulty:** {guide.difficulty}")
        sections.append("")
        
        # Prerequisites
        if guide.prerequisites:
            sections.append("## Prerequisites")
            sections.append("")
            for prereq in guide.prerequisites:
                sections.append(f"- {prereq}")
            sections.append("")
        
        # Changes by type
        change_types = {}
        for change in guide.changes:
            if change.type not in change_types:
                change_types[change.type] = []
            change_types[change.type].append(change)
        
        for change_type, changes in change_types.items():
            section_title = {
                ChangeType.BREAKING: "Breaking Changes",
                ChangeType.DEPRECATED: "Deprecated Features", 
                ChangeType.NEW_FEATURE: "New Features",
                ChangeType.IMPROVEMENT: "Improvements",
                ChangeType.BUG_FIX: "Bug Fixes",
                ChangeType.SECURITY: "Security Updates"
            }
            
            sections.append(f"## {section_title.get(change_type, 'Other Changes')}")
            sections.append("")
            
            for change in changes:
                sections.append(self.render_change_item(change))
                sections.append("")
        
        # Post-migration steps
        if guide.post_migration_steps:
            sections.append("## Post-Migration Verification")
            sections.append("")
            for i, step in enumerate(guide.post_migration_steps, 1):
                sections.append(f"{i}. {step}")
            sections.append("")
        
        # Troubleshooting
        if guide.troubleshooting:
            sections.append("## Troubleshooting")
            sections.append("")
            for issue in guide.troubleshooting:
                sections.append(f"**Problem:** {issue['problem']}")
                sections.append(f"**Solution:** {issue['solution']}")
                sections.append("")
        
        return "\n".join(sections)
    
    def render_change_item(self, change: ChangeItem) -> str:
        """Render a single change item"""
        sections = []
        
        # Change header
        sections.append(f"### {change.title}")
        sections.append("")
        sections.append(f"**Impact:** {change.impact.title()}")
        
        if change.affected_components:
            sections.append(f"**Affected Components:** {', '.join(change.affected_components)}")
        sections.append("")
        
        sections.append(change.description)
        sections.append("")
        
        # Code examples
        if change.before_code and change.after_code:
            sections.append("#### Before")
            sections.append("```")
            sections.append(change.before_code)
            sections.append("```")
            sections.append("")
            
            sections.append("#### After")
            sections.append("```")
            sections.append(change.after_code)
            sections.append("```")
            sections.append("")
        
        # Migration steps
        if change.migration_steps:
            sections.append("#### Migration Steps")
            sections.append("")
            for i, step in enumerate(change.migration_steps, 1):
                sections.append(f"{i}. {step}")
            sections.append("")
        
        # Notes
        if change.notes:
            sections.append("#### Notes")
            sections.append("")
            for note in change.notes:
                sections.append(f"- {note}")
            sections.append("")
        
        return "\n".join(sections)
    
    def guide_to_dict(self, guide: MigrationGuide) -> Dict[str, Any]:
        """Convert guide to dictionary"""
        return {
            "id": guide.id,
            "title": guide.title,
            "description": guide.description,
            "migration_type": guide.migration_type.value,
            "source_version": guide.source_version,
            "target_version": guide.target_version,
            "estimated_time": guide.estimated_time,
            "difficulty": guide.difficulty,
            "prerequisites": guide.prerequisites,
            "changes": [
                {
                    "id": change.id,
                    "type": change.type.value,
                    "title": change.title,
                    "description": change.description,
                    "impact": change.impact,
                    "affected_components": change.affected_components,
                    "before_code": change.before_code,
                    "after_code": change.after_code,
                    "migration_steps": change.migration_steps,
                    "notes": change.notes
                }
                for change in guide.changes
            ],
            "post_migration_steps": guide.post_migration_steps,
            "troubleshooting": guide.troubleshooting,
            "resources": guide.resources,
            "created_date": guide.created_date
        }