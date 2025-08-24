"""
Changelog Generator

Automatically generates changelogs from git commits and semantic versioning.
"""

import re
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of changes for semantic versioning."""
    BREAKING = "breaking"  # Major version
    FEATURE = "feature"    # Minor version
    FIX = "fix"           # Patch version
    DOCS = "docs"         # Documentation
    STYLE = "style"       # Formatting
    REFACTOR = "refactor" # Code refactoring
    PERF = "perf"         # Performance
    TEST = "test"         # Testing
    CHORE = "chore"       # Maintenance
    

@dataclass
class Commit:
    """Represents a git commit."""
    hash: str
    date: datetime
    author: str
    message: str
    type: ChangeType
    scope: Optional[str]
    breaking: bool
    

@dataclass
class Release:
    """Represents a release version."""
    version: str
    date: datetime
    commits: List[Commit]
    breaking_changes: List[str]
    features: List[str]
    fixes: List[str]
    

class ChangelogGenerator:
    """
    Generates changelogs from git history with semantic versioning.
    Supports conventional commits and automatic version bumping.
    """
    
    def __init__(self):
        """Initialize the changelog generator."""
        self.commits = []
        self.releases = []
        self.commit_pattern = re.compile(
            r'^(?P<type>\w+)(?:\((?P<scope>[^)]+)\))?(?P<breaking>!)?: (?P<message>.+)$'
        )
        logger.info("Changelog Generator initialized")
        
    def parse_commits(self, since: Optional[str] = None, until: Optional[str] = "HEAD") -> List[Commit]:
        """
        Parse git commits.
        
        Args:
            since: Start commit/tag
            until: End commit/tag
            
        Returns:
            List of parsed commits
        """
        try:
            # Get git log
            cmd = ["git", "log", "--pretty=format:%H|%ai|%an|%s"]
            if since:
                cmd.append(f"{since}..{until}")
            else:
                cmd.append(until)
                
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Git log failed: {result.stderr}")
                return []
                
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|', 3)
                    if len(parts) == 4:
                        commit = self._parse_commit_line(parts)
                        commits.append(commit)
                        
            self.commits = commits
            return commits
            
        except Exception as e:
            logger.error(f"Error parsing commits: {e}")
            return []
            
    def _parse_commit_line(self, parts: List[str]) -> Commit:
        """Parse a single commit line."""
        hash_val, date_str, author, message = parts
        
        # Parse conventional commit format
        match = self.commit_pattern.match(message)
        
        if match:
            type_str = match.group('type')
            scope = match.group('scope')
            breaking = bool(match.group('breaking'))
            msg = match.group('message')
            
            # Map to ChangeType
            type_map = {
                'feat': ChangeType.FEATURE,
                'fix': ChangeType.FIX,
                'docs': ChangeType.DOCS,
                'style': ChangeType.STYLE,
                'refactor': ChangeType.REFACTOR,
                'perf': ChangeType.PERF,
                'test': ChangeType.TEST,
                'chore': ChangeType.CHORE,
                'breaking': ChangeType.BREAKING
            }
            change_type = type_map.get(type_str, ChangeType.CHORE)
            
        else:
            # Fallback parsing
            change_type = self._detect_change_type(message)
            scope = None
            breaking = False
            msg = message
            
        return Commit(
            hash=hash_val[:7],
            date=datetime.fromisoformat(date_str.split()[0]),
            author=author,
            message=msg,
            type=change_type,
            scope=scope,
            breaking=breaking
        )
        
    def _detect_change_type(self, message: str) -> ChangeType:
        """Detect change type from commit message."""
        msg_lower = message.lower()
        
        if 'breaking' in msg_lower or 'major' in msg_lower:
            return ChangeType.BREAKING
        elif 'feat' in msg_lower or 'add' in msg_lower or 'new' in msg_lower:
            return ChangeType.FEATURE
        elif 'fix' in msg_lower or 'bug' in msg_lower or 'patch' in msg_lower:
            return ChangeType.FIX
        elif 'doc' in msg_lower:
            return ChangeType.DOCS
        elif 'perf' in msg_lower or 'optim' in msg_lower:
            return ChangeType.PERF
        elif 'test' in msg_lower:
            return ChangeType.TEST
        elif 'refactor' in msg_lower:
            return ChangeType.REFACTOR
        else:
            return ChangeType.CHORE
            
    def generate_version(self, current: str, commits: List[Commit]) -> str:
        """
        Generate next version based on commits.
        
        Args:
            current: Current version (e.g., "1.2.3")
            commits: List of commits since last version
            
        Returns:
            Next version string
        """
        major, minor, patch = map(int, current.split('.'))
        
        # Check for breaking changes
        if any(c.breaking or c.type == ChangeType.BREAKING for c in commits):
            return f"{major + 1}.0.0"
            
        # Check for features
        if any(c.type == ChangeType.FEATURE for c in commits):
            return f"{major}.{minor + 1}.0"
            
        # Check for fixes
        if any(c.type == ChangeType.FIX for c in commits):
            return f"{major}.{minor}.{patch + 1}"
            
        # No version bump for other changes
        return current
        
    def group_commits(self, commits: List[Commit]) -> Dict[ChangeType, List[Commit]]:
        """
        Group commits by change type.
        
        Args:
            commits: List of commits
            
        Returns:
            Grouped commits
        """
        grouped = {}
        
        for commit in commits:
            if commit.type not in grouped:
                grouped[commit.type] = []
            grouped[commit.type].append(commit)
            
        return grouped
        
    def generate_changelog(self, 
                          format: str = "markdown",
                          include_author: bool = False,
                          include_hash: bool = True) -> str:
        """
        Generate changelog document.
        
        Args:
            format: Output format (markdown, json, html)
            include_author: Include commit authors
            include_hash: Include commit hashes
            
        Returns:
            Formatted changelog
        """
        if format == "markdown":
            return self._generate_markdown_changelog(include_author, include_hash)
        elif format == "json":
            return self._generate_json_changelog()
        else:
            return self._generate_markdown_changelog(include_author, include_hash)
            
    def _generate_markdown_changelog(self, include_author: bool, include_hash: bool) -> str:
        """Generate markdown format changelog."""
        lines = [
            "# Changelog",
            "",
            "All notable changes to this project will be documented in this file.",
            "",
            "The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),",
            "and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).",
            ""
        ]
        
        # Group commits by type
        grouped = self.group_commits(self.commits)
        
        # Add unreleased section
        lines.extend([
            "## [Unreleased]",
            ""
        ])
        
        # Section order and titles
        sections = [
            (ChangeType.BREAKING, "### ⚠ BREAKING CHANGES"),
            (ChangeType.FEATURE, "### 🚀 Features"),
            (ChangeType.FIX, "### 🐛 Bug Fixes"),
            (ChangeType.PERF, "### ⚡ Performance"),
            (ChangeType.DOCS, "### 📝 Documentation"),
            (ChangeType.REFACTOR, "### ♻️ Refactoring"),
            (ChangeType.TEST, "### ✅ Tests"),
            (ChangeType.CHORE, "### 🔧 Chores")
        ]
        
        for change_type, title in sections:
            if change_type in grouped:
                lines.extend([title, ""])
                
                for commit in grouped[change_type]:
                    line = f"- "
                    
                    if commit.scope:
                        line += f"**{commit.scope}:** "
                        
                    line += commit.message
                    
                    if include_hash:
                        line += f" ({commit.hash})"
                        
                    if include_author:
                        line += f" - @{commit.author}"
                        
                    lines.append(line)
                    
                lines.append("")
                
        return "\n".join(lines)
        
    def _generate_json_changelog(self) -> str:
        """Generate JSON format changelog."""
        import json
        
        changelog = {
            "changelog": {
                "unreleased": []
            }
        }
        
        for commit in self.commits:
            changelog["changelog"]["unreleased"].append({
                "hash": commit.hash,
                "date": commit.date.isoformat(),
                "author": commit.author,
                "message": commit.message,
                "type": commit.type.value,
                "scope": commit.scope,
                "breaking": commit.breaking
            })
            
        return json.dumps(changelog, indent=2)
        
    def generate_release_notes(self, version: str, commits: List[Commit]) -> str:
        """
        Generate release notes for a version.
        
        Args:
            version: Version number
            commits: Commits in this release
            
        Returns:
            Release notes markdown
        """
        grouped = self.group_commits(commits)
        
        notes = [
            f"# Release {version}",
            f"*{datetime.now().strftime('%Y-%m-%d')}*",
            ""
        ]
        
        # Highlights
        if ChangeType.BREAKING in grouped:
            notes.extend([
                "## ⚠️ Breaking Changes",
                ""
            ])
            for commit in grouped[ChangeType.BREAKING]:
                notes.append(f"- {commit.message}")
            notes.append("")
            
        if ChangeType.FEATURE in grouped:
            notes.extend([
                "## ✨ New Features",
                ""
            ])
            for commit in grouped[ChangeType.FEATURE]:
                notes.append(f"- {commit.message}")
            notes.append("")
            
        if ChangeType.FIX in grouped:
            notes.extend([
                "## 🐛 Bug Fixes",
                ""
            ])
            for commit in grouped[ChangeType.FIX]:
                notes.append(f"- {commit.message}")
            notes.append("")
            
        # Statistics
        notes.extend([
            "## 📊 Statistics",
            f"- Total commits: {len(commits)}",
            f"- Contributors: {len(set(c.author for c in commits))}",
            ""
        ])
        
        return "\n".join(notes)
        
    def export_changelog(self, output_path: str, content: str) -> None:
        """
        Export changelog to file.
        
        Args:
            output_path: Output file path
            content: Changelog content
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Exported changelog to {output_path}")