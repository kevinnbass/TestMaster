"""
Git Integration for Enhanced Commit Messages

Intelligent git commit message enhancement using code analysis insights
to generate more informative and structured commit messages.
"""

import re
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from ..core.context_builder import AnalysisContextBuilder
from ..core.llm_integration import LLMIntegration

logger = logging.getLogger(__name__)


@dataclass
class GitChange:
    """Represents a single git change."""
    file_path: str
    change_type: str  # "added", "modified", "deleted", "renamed"
    lines_added: int
    lines_removed: int
    old_path: Optional[str] = None  # For renames


@dataclass
class CommitContext:
    """Context information for commit message generation."""
    changes: List[GitChange]
    branch_name: str
    staged_files: List[str]
    commit_type: str  # "feature", "fix", "refactor", "docs", etc.
    scope: Optional[str]
    breaking_changes: bool
    analysis_insights: List[str]


class GitCommitEnhancer:
    """
    Intelligent git commit message enhancement.
    
    Features:
    - Analyzes staged changes and their impact
    - Generates conventional commit messages
    - Includes analysis insights in commit context
    - Detects breaking changes automatically
    - Suggests appropriate commit types and scopes
    """
    
    def __init__(self, 
                 llm_integration: LLMIntegration,
                 context_builder: AnalysisContextBuilder):
        """
        Initialize the git commit enhancer.
        
        Args:
            llm_integration: LLM integration for message generation
            context_builder: Analysis context builder
        """
        self.llm_integration = llm_integration
        self.context_builder = context_builder
        
        # Conventional commit types
        self.commit_types = {
            "feat": "New feature",
            "fix": "Bug fix", 
            "docs": "Documentation changes",
            "style": "Code style changes",
            "refactor": "Code refactoring",
            "perf": "Performance improvements",
            "test": "Test changes",
            "chore": "Maintenance tasks",
            "ci": "CI/CD changes",
            "build": "Build system changes"
        }
        
    def enhance_commit_message(self, repo_path: str = ".") -> str:
        """
        Generate enhanced commit message for staged changes.
        
        Args:
            repo_path: Path to git repository
            
        Returns:
            str: Enhanced commit message
        """
        logger.info("Generating enhanced commit message")
        
        # Analyze staged changes
        commit_context = self._analyze_staged_changes(repo_path)
        
        if not commit_context.changes:
            return "chore: no staged changes found"
        
        # Generate commit message using LLM
        enhanced_message = await self._generate_commit_message(commit_context)
        
        logger.info(f"Generated commit message: {enhanced_message[:50]}...")
        return enhanced_message
        
    def suggest_commit_message(self, files: List[str], repo_path: str = ".") -> str:
        """
        Suggest commit message for specific files.
        
        Args:
            files: List of file paths to analyze
            repo_path: Path to git repository
            
        Returns:
            str: Suggested commit message
        """
        logger.info(f"Suggesting commit message for {len(files)} files")
        
        # Analyze specific files
        commit_context = self._analyze_file_changes(files, repo_path)
        
        # Generate commit message
        suggested_message = await self._generate_commit_message(commit_context)
        
        return suggested_message
        
    def _analyze_staged_changes(self, repo_path: str) -> CommitContext:
        """Analyze staged changes in git repository."""
        try:
            # Get staged files
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-status"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            staged_files = []
            changes = []
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) >= 2:
                    status = parts[0]
                    file_path = parts[1]
                    staged_files.append(file_path)
                    
                    # Determine change type
                    if status.startswith('A'):
                        change_type = "added"
                    elif status.startswith('M'):
                        change_type = "modified"
                    elif status.startswith('D'):
                        change_type = "deleted"
                    elif status.startswith('R'):
                        change_type = "renamed"
                    else:
                        change_type = "modified"
                    
                    # Get line counts
                    lines_added, lines_removed = self._get_line_counts(file_path, repo_path)
                    
                    changes.append(GitChange(
                        file_path=file_path,
                        change_type=change_type,
                        lines_added=lines_added,
                        lines_removed=lines_removed
                    ))
            
            # Get branch name
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            branch_name = branch_result.stdout.strip()
            
            # Analyze changes to determine commit type and scope
            commit_type, scope, breaking_changes = self._analyze_change_patterns(changes)
            
            # Get analysis insights for changed files
            analysis_insights = self._get_analysis_insights(staged_files, repo_path)
            
            return CommitContext(
                changes=changes,
                branch_name=branch_name,
                staged_files=staged_files,
                commit_type=commit_type,
                scope=scope,
                breaking_changes=breaking_changes,
                analysis_insights=analysis_insights
            )
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
            return CommitContext(
                changes=[],
                branch_name="unknown",
                staged_files=[],
                commit_type="chore",
                scope=None,
                breaking_changes=False,
                analysis_insights=[]
            )
            
    def _analyze_file_changes(self, files: List[str], repo_path: str) -> CommitContext:
        """Analyze changes for specific files."""
        changes = []
        
        for file_path in files:
            # Determine if file is new, modified, or deleted
            if Path(repo_path) / file_path.exists():
                try:
                    # Check if file is tracked
                    subprocess.run(
                        ["git", "ls-files", "--error-unmatch", file_path],
                        cwd=repo_path,
                        capture_output=True,
                        check=True
                    )
                    change_type = "modified"
                except subprocess.CalledProcessError:
                    change_type = "added"
            else:
                change_type = "deleted"
            
            lines_added, lines_removed = self._get_line_counts(file_path, repo_path)
            
            changes.append(GitChange(
                file_path=file_path,
                change_type=change_type,
                lines_added=lines_added,
                lines_removed=lines_removed
            ))
        
        # Analyze patterns
        commit_type, scope, breaking_changes = self._analyze_change_patterns(changes)
        
        # Get analysis insights
        analysis_insights = self._get_analysis_insights(files, repo_path)
        
        return CommitContext(
            changes=changes,
            branch_name="unknown",
            staged_files=files,
            commit_type=commit_type,
            scope=scope,
            breaking_changes=breaking_changes,
            analysis_insights=analysis_insights
        )
        
    def _get_line_counts(self, file_path: str, repo_path: str) -> Tuple[int, int]:
        """Get lines added and removed for a file."""
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--numstat", file_path],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout.strip():
                parts = result.stdout.strip().split('\t')
                if len(parts) >= 2:
                    added = int(parts[0]) if parts[0] != '-' else 0
                    removed = int(parts[1]) if parts[1] != '-' else 0
                    return added, removed
                    
        except (subprocess.CalledProcessError, ValueError):
            pass
            
        return 0, 0
        
    def _analyze_change_patterns(self, changes: List[GitChange]) -> Tuple[str, Optional[str], bool]:
        """Analyze change patterns to determine commit type, scope, and breaking changes."""
        if not changes:
            return "chore", None, False
        
        # Count change types
        added_files = sum(1 for c in changes if c.change_type == "added")
        modified_files = sum(1 for c in changes if c.change_type == "modified")
        deleted_files = sum(1 for c in changes if c.change_type == "deleted")
        
        # Analyze file patterns
        file_patterns = {
            "test": 0,
            "docs": 0,
            "config": 0,
            "src": 0,
            "api": 0
        }
        
        for change in changes:
            file_lower = change.file_path.lower()
            
            if any(pattern in file_lower for pattern in ['test', 'spec', '__test__']):
                file_patterns["test"] += 1
            elif any(pattern in file_lower for pattern in ['readme', 'doc', '.md', '.rst']):
                file_patterns["docs"] += 1
            elif any(pattern in file_lower for pattern in ['config', 'setup', 'requirements', 'package']):
                file_patterns["config"] += 1
            elif any(pattern in file_lower for pattern in ['api', 'endpoint', 'route']):
                file_patterns["api"] += 1
            else:
                file_patterns["src"] += 1
        
        # Determine commit type
        if file_patterns["test"] > file_patterns["src"]:
            commit_type = "test"
        elif file_patterns["docs"] > 0 and file_patterns["src"] == 0:
            commit_type = "docs"
        elif file_patterns["config"] > file_patterns["src"]:
            commit_type = "chore"
        elif added_files > 0 and modified_files == 0:
            commit_type = "feat"
        elif "fix" in " ".join([c.file_path for c in changes]).lower():
            commit_type = "fix"
        elif any("refactor" in c.file_path.lower() for c in changes):
            commit_type = "refactor"
        else:
            commit_type = "feat" if added_files > 0 else "fix"
        
        # Determine scope
        scope = None
        if file_patterns["api"] > 0:
            scope = "api"
        elif file_patterns["test"] > 0:
            scope = "test"
        elif file_patterns["docs"] > 0:
            scope = "docs"
        
        # Check for breaking changes
        breaking_changes = False
        total_changes = sum(c.lines_added + c.lines_removed for c in changes)
        if total_changes > 100 or deleted_files > 0:
            breaking_changes = True
        
        return commit_type, scope, breaking_changes
        
    def _get_analysis_insights(self, files: List[str], repo_path: str) -> List[str]:
        """Get analysis insights for changed files."""
        insights = []
        
        for file_path in files:
            if file_path.endswith('.py'):
                try:
                    full_path = str(Path(repo_path) / file_path)
                    if Path(full_path).exists():
                        # Build context for this file
                        module_context = self.context_builder.build_module_context(full_path)
                        
                        # Extract key insights
                        for insight in module_context.insights[:2]:  # Top 2 insights
                            insights.append(f"{file_path}: {insight.description}")
                            
                except Exception as e:
                    logger.debug(f"Could not analyze {file_path}: {e}")
        
        return insights
        
    async def _generate_commit_message(self, context: CommitContext) -> str:
        """Generate enhanced commit message using LLM."""
        # Build context for LLM
        context_str = self._build_commit_context(context)
        
        prompt = f"""
Generate a conventional commit message for the following changes:

COMMIT CONTEXT:
{context_str}

REQUIREMENTS:
1. Use conventional commit format: type(scope): description
2. Type should be one of: feat, fix, docs, style, refactor, perf, test, chore, ci, build
3. Include scope if applicable
4. Keep description under 50 characters
5. Add body with bullet points for significant changes
6. Add footer with breaking change notice if applicable
7. Include relevant analysis insights

EXAMPLE FORMAT:
feat(api): add user authentication endpoint

- Implement JWT token generation
- Add password hashing with bcrypt
- Include rate limiting for auth attempts
- Security: Added input validation for credentials

BREAKING CHANGE: Authentication now required for all API endpoints

Generate the commit message:
"""
        
        response = await self.llm_integration.generate_documentation(
            doc_type="git_commit",
            context=prompt,
            code="",
            style="conventional"
        )
        
        # Post-process the commit message
        commit_message = self._post_process_commit_message(response.content, context)
        
        return commit_message
        
    def _build_commit_context(self, context: CommitContext) -> str:
        """Build context string for LLM prompt."""
        context_parts = [
            f"Branch: {context.branch_name}",
            f"Files changed: {len(context.changes)}",
            f"Commit type: {context.commit_type}",
        ]
        
        if context.scope:
            context_parts.append(f"Scope: {context.scope}")
        
        if context.breaking_changes:
            context_parts.append("Breaking changes: YES")
        
        context_parts.append("\nChanges:")
        for change in context.changes:
            context_parts.append(f"- {change.change_type}: {change.file_path} (+{change.lines_added}/-{change.lines_removed})")
        
        if context.analysis_insights:
            context_parts.append("\nAnalysis insights:")
            for insight in context.analysis_insights[:3]:  # Top 3 insights
                context_parts.append(f"- {insight}")
        
        return "\n".join(context_parts)
        
    def _post_process_commit_message(self, raw_message: str, context: CommitContext) -> str:
        """Post-process the generated commit message."""
        lines = raw_message.strip().split('\n')
        processed_lines = []
        
        # Ensure first line follows conventional commit format
        first_line = lines[0] if lines else ""
        if not re.match(r'^(feat|fix|docs|style|refactor|perf|test|chore|ci|build)(\(.+\))?: .+', first_line):
            # Fix the format
            if context.scope:
                first_line = f"{context.commit_type}({context.scope}): {first_line}"
            else:
                first_line = f"{context.commit_type}: {first_line}"
        
        # Limit first line to 50 characters
        if len(first_line) > 50:
            first_line = first_line[:47] + "..."
        
        processed_lines.append(first_line)
        
        # Add remaining lines
        if len(lines) > 1:
            processed_lines.extend(lines[1:])
        
        return '\n'.join(processed_lines)
        
    def create_commit_template(self, repo_path: str = ".") -> str:
        """
        Create a commit message template file.
        
        Args:
            repo_path: Path to git repository
            
        Returns:
            str: Path to created template file
        """
        template_content = """# <type>(<scope>): <subject>
#
# <body>
#
# <footer>

# Type should be one of:
# feat: A new feature
# fix: A bug fix
# docs: Documentation only changes
# style: Changes that do not affect the meaning of the code
# refactor: A code change that neither fixes a bug nor adds a feature
# perf: A code change that improves performance
# test: Adding missing tests or correcting existing tests
# chore: Changes to the build process or auxiliary tools

# Scope (optional): Component or module affected

# Subject: Imperative mood, no period, max 50 chars

# Body (optional): Explain what and why, not how
# Wrap at 72 characters

# Footer (optional): Reference issues, breaking changes
# BREAKING CHANGE: Description of breaking change
"""
        
        template_path = Path(repo_path) / ".gitmessage"
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        # Configure git to use the template
        try:
            subprocess.run(
                ["git", "config", "commit.template", str(template_path)],
                cwd=repo_path,
                check=True
            )
            logger.info(f"Created commit template: {template_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to configure git template: {e}")
        
        return str(template_path)
        
    def install_commit_hook(self, repo_path: str = ".") -> str:
        """
        Install git commit hook for automatic message enhancement.
        
        Args:
            repo_path: Path to git repository
            
        Returns:
            str: Path to installed hook
        """
        hook_content = f"""#!/bin/sh
# Git commit message enhancement hook
# Generated by TestMaster Documentation System

python -c "
import sys
sys.path.append('{Path(__file__).parent}')
from git_integration import GitCommitEnhancer
from ..core.llm_integration import create_default_llm_integration
from ..core.context_builder import AnalysisContextBuilder

try:
    llm = create_default_llm_integration()
    context_builder = AnalysisContextBuilder()
    enhancer = GitCommitEnhancer(llm, context_builder)
    
    # Read current commit message
    with open(sys.argv[1], 'r') as f:
        current_message = f.read().strip()
    
    # Skip if message is already enhanced or is a merge
    if current_message.startswith(('feat', 'fix', 'docs', 'style', 'refactor', 'perf', 'test', 'chore', 'ci', 'build')):
        sys.exit(0)
    if 'Merge' in current_message:
        sys.exit(0)
    
    # Generate enhanced message
    enhanced = enhancer.enhance_commit_message()
    
    # Write enhanced message
    with open(sys.argv[1], 'w') as f:
        f.write(enhanced)
        
except Exception as e:
    # Fail silently to not block commits
    pass
" "$1"
"""
        
        hooks_dir = Path(repo_path) / ".git" / "hooks"
        hook_path = hooks_dir / "prepare-commit-msg"
        
        # Create hooks directory if it doesn't exist
        hooks_dir.mkdir(parents=True, exist_ok=True)
        
        # Write hook
        with open(hook_path, 'w', encoding='utf-8') as f:
            f.write(hook_content)
        
        # Make executable
        hook_path.chmod(0o755)
        
        logger.info(f"Installed commit hook: {hook_path}")
        return str(hook_path)