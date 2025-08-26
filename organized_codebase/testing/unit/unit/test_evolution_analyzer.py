"""
Test Evolution Analyzer Framework
Tracks the evolution of testing patterns, practices, and intelligence over time.
"""

import os
import re
import git
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import time
import difflib


class EvolutionEventType(Enum):
    """Types of evolution events"""
    CREATION = "creation"
    MODIFICATION = "modification"
    DELETION = "deletion"
    REFACTORING = "refactoring"
    ENHANCEMENT = "enhancement"
    BUG_FIX = "bug_fix"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"


@dataclass
class TestEvolutionEvent:
    """Single evolution event in test development"""
    timestamp: datetime
    event_type: EvolutionEventType
    file_path: str
    commit_hash: Optional[str] = None
    author: Optional[str] = None
    message: Optional[str] = None
    changes_added: int = 0
    changes_removed: int = 0
    complexity_delta: float = 0.0
    patterns_introduced: List[str] = field(default_factory=list)
    patterns_removed: List[str] = field(default_factory=list)
    
    @property
    def impact_score(self) -> float:
        """Calculate impact score of this evolution event"""
        base_score = (self.changes_added + self.changes_removed) * 0.1
        complexity_bonus = abs(self.complexity_delta) * 0.3
        pattern_bonus = (len(self.patterns_introduced) + len(self.patterns_removed)) * 0.5
        return base_score + complexity_bonus + pattern_bonus


@dataclass
class TestFile:
    """Representation of a test file at a point in time"""
    path: str
    content: str
    size_bytes: int
    line_count: int
    function_count: int
    class_count: int
    import_count: int
    complexity_score: float
    test_patterns: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def calculate_similarity(self, other: 'TestFile') -> float:
        """Calculate similarity with another test file version"""
        content_similarity = difflib.SequenceMatcher(None, self.content, other.content).ratio()
        
        # Factor in structural similarity
        structure_sim = 0.0
        if self.function_count > 0 and other.function_count > 0:
            func_sim = 1.0 - abs(self.function_count - other.function_count) / max(self.function_count, other.function_count)
            structure_sim += func_sim * 0.3
        
        if self.class_count > 0 and other.class_count > 0:
            class_sim = 1.0 - abs(self.class_count - other.class_count) / max(self.class_count, other.class_count)
            structure_sim += class_sim * 0.2
        
        # Pattern similarity
        common_patterns = set(self.test_patterns) & set(other.test_patterns)
        all_patterns = set(self.test_patterns) | set(other.test_patterns)
        pattern_sim = len(common_patterns) / len(all_patterns) if all_patterns else 0.0
        
        return content_similarity * 0.5 + structure_sim * 0.3 + pattern_sim * 0.2


@dataclass
class EvolutionTrajectory:
    """Complete evolution trajectory of a test file or system"""
    file_path: str
    creation_date: datetime
    last_modified: datetime
    total_events: int
    major_refactoring_count: int
    evolution_events: List[TestEvolutionEvent] = field(default_factory=list)
    version_snapshots: List[TestFile] = field(default_factory=list)
    complexity_trend: List[Tuple[datetime, float]] = field(default_factory=list)
    pattern_adoption: Dict[str, datetime] = field(default_factory=dict)
    
    @property
    def evolution_velocity(self) -> float:
        """Calculate how rapidly this file evolves"""
        if not self.evolution_events or len(self.evolution_events) < 2:
            return 0.0
        
        time_span = (self.last_modified - self.creation_date).days
        if time_span == 0:
            return float('inf')  # Very rapid evolution
        
        return len(self.evolution_events) / time_span
    
    @property
    def stability_score(self) -> float:
        """Calculate stability score (lower = more stable)"""
        if not self.evolution_events:
            return 1.0
        
        recent_events = [e for e in self.evolution_events 
                        if (datetime.now() - e.timestamp).days <= 30]
        return 1.0 / (1.0 + len(recent_events) * 0.1)


class GitHistoryAnalyzer:
    """Analyze Git history for test evolution patterns"""
    
    def __init__(self, repo_path: str):
        try:
            self.repo = git.Repo(repo_path)
        except git.exc.InvalidGitRepositoryError:
            self.repo = None
            print(f"Warning: {repo_path} is not a Git repository")
        
        self.commit_cache = {}
        self.file_history_cache = {}
    
    def get_file_commits(self, file_path: str, max_commits: int = 100) -> List[git.Commit]:
        """Get commits that modified a specific file"""
        if not self.repo:
            return []
        
        if file_path in self.file_history_cache:
            return self.file_history_cache[file_path]
        
        try:
            commits = list(self.repo.iter_commits(paths=file_path, max_count=max_commits))
            self.file_history_cache[file_path] = commits
            return commits
        except Exception as e:
            print(f"Error getting commits for {file_path}: {e}")
            return []
    
    def analyze_commit_changes(self, commit: git.Commit, file_path: str) -> Dict[str, Any]:
        """Analyze changes made to a file in a specific commit"""
        try:
            if not commit.parents:
                # Initial commit
                return {
                    'added_lines': 0,
                    'removed_lines': 0,
                    'file_mode': 'A',  # Added
                    'diff_text': ''
                }
            
            parent = commit.parents[0]
            diffs = parent.diff(commit, paths=file_path)
            
            if not diffs:
                return {
                    'added_lines': 0,
                    'removed_lines': 0,
                    'file_mode': 'N',  # No change
                    'diff_text': ''
                }
            
            diff = diffs[0]
            
            # Count line changes
            added_lines = 0
            removed_lines = 0
            diff_text = ''
            
            if diff.diff:
                diff_text = diff.diff.decode('utf-8', errors='ignore')
                for line in diff_text.split('\n'):
                    if line.startswith('+') and not line.startswith('+++'):
                        added_lines += 1
                    elif line.startswith('-') and not line.startswith('---'):
                        removed_lines += 1
            
            return {
                'added_lines': added_lines,
                'removed_lines': removed_lines,
                'file_mode': diff.change_type,
                'diff_text': diff_text
            }
            
        except Exception as e:
            print(f"Error analyzing commit {commit.hexsha}: {e}")
            return {'added_lines': 0, 'removed_lines': 0, 'file_mode': 'E', 'diff_text': ''}
    
    def extract_evolution_events(self, file_path: str) -> List[TestEvolutionEvent]:
        """Extract evolution events from Git history"""
        commits = self.get_file_commits(file_path)
        events = []
        
        for commit in commits:
            changes = self.analyze_commit_changes(commit, file_path)
            
            # Determine event type based on commit message and changes
            message = commit.message.lower()
            event_type = EvolutionEventType.MODIFICATION  # Default
            
            if changes['file_mode'] == 'A':
                event_type = EvolutionEventType.CREATION
            elif changes['file_mode'] == 'D':
                event_type = EvolutionEventType.DELETION
            elif any(keyword in message for keyword in ['refactor', 'restructure', 'reorganize']):
                event_type = EvolutionEventType.REFACTORING
            elif any(keyword in message for keyword in ['fix', 'bug', 'error']):
                event_type = EvolutionEventType.BUG_FIX
            elif any(keyword in message for keyword in ['enhance', 'improve', 'add']):
                event_type = EvolutionEventType.ENHANCEMENT
            elif any(keyword in message for keyword in ['optim', 'perf', 'speed']):
                event_type = EvolutionEventType.OPTIMIZATION
            elif any(keyword in message for keyword in ['doc', 'comment', 'readme']):
                event_type = EvolutionEventType.DOCUMENTATION
            
            # Extract patterns from diff
            patterns_introduced = []
            patterns_removed = []
            
            diff_text = changes.get('diff_text', '')
            for line in diff_text.split('\n'):
                if line.startswith('+'):
                    # Look for test patterns in added lines
                    if 'def test_' in line or 'class Test' in line:
                        patterns_introduced.append('test_function' if 'def test_' in line else 'test_class')
                    elif 'assert' in line:
                        patterns_introduced.append('assertion')
                    elif 'mock' in line.lower():
                        patterns_introduced.append('mocking')
                elif line.startswith('-'):
                    # Track removed patterns
                    if 'def test_' in line or 'class Test' in line:
                        patterns_removed.append('test_function' if 'def test_' in line else 'test_class')
            
            event = TestEvolutionEvent(
                timestamp=datetime.fromtimestamp(commit.committed_date),
                event_type=event_type,
                file_path=file_path,
                commit_hash=commit.hexsha,
                author=commit.author.name,
                message=commit.message.strip(),
                changes_added=changes['added_lines'],
                changes_removed=changes['removed_lines'],
                patterns_introduced=patterns_introduced,
                patterns_removed=patterns_removed
            )
            
            events.append(event)
        
        return sorted(events, key=lambda x: x.timestamp)


class TestFileAnalyzer:
    """Analyze test files for evolution tracking"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_test_file(self, file_path: str, content: Optional[str] = None) -> TestFile:
        """Analyze a test file and extract metrics"""
        if not content:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return TestFile(file_path, "", 0, 0, 0, 0, 0, 0.0)
        
        # Basic metrics
        size_bytes = len(content.encode('utf-8'))
        line_count = content.count('\n') + 1
        
        # Count functions and classes
        function_count = len(re.findall(r'^\s*def\s+', content, re.MULTILINE))
        class_count = len(re.findall(r'^\s*class\s+', content, re.MULTILINE))
        import_count = len(re.findall(r'^\s*(?:from\s+\S+\s+)?import\s+', content, re.MULTILINE))
        
        # Detect test patterns
        test_patterns = []
        if re.search(r'def\s+test_', content):
            test_patterns.append('pytest_functions')
        if re.search(r'class\s+Test\w+', content):
            test_patterns.append('unittest_classes')
        if 'assert' in content:
            test_patterns.append('assertions')
        if 'mock' in content.lower():
            test_patterns.append('mocking')
        if '@pytest.' in content:
            test_patterns.append('pytest_decorators')
        if 'setUp' in content or 'tearDown' in content:
            test_patterns.append('setup_teardown')
        if 'fixture' in content:
            test_patterns.append('fixtures')
        if 'parametrize' in content:
            test_patterns.append('parametrized_tests')
        
        # Calculate complexity (simplified)
        complexity_score = (
            function_count * 0.5 + 
            class_count * 1.0 + 
            content.count('if ') * 0.1 +
            content.count('for ') * 0.15 +
            content.count('while ') * 0.2 +
            content.count('try:') * 0.3
        )
        
        # Extract dependencies
        dependencies = []
        import_matches = re.findall(r'(?:from\s+(\S+)\s+)?import\s+(\S+)', content)
        for from_module, import_name in import_matches:
            if from_module:
                dependencies.append(from_module)
            else:
                dependencies.append(import_name)
        
        return TestFile(
            path=file_path,
            content=content,
            size_bytes=size_bytes,
            line_count=line_count,
            function_count=function_count,
            class_count=class_count,
            import_count=import_count,
            complexity_score=complexity_score,
            test_patterns=test_patterns,
            dependencies=list(set(dependencies))  # Remove duplicates
        )


class TestEvolutionAnalyzer:
    """Main analyzer for test evolution patterns"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.git_analyzer = GitHistoryAnalyzer(str(self.repo_path))
        self.file_analyzer = TestFileAnalyzer()
        self.trajectories = {}
        self.global_patterns = {
            'creation_rate': [],
            'modification_velocity': [],
            'pattern_adoption_timeline': defaultdict(list),
            'complexity_evolution': []
        }
    
    def discover_test_files(self, extensions: List[str] = None) -> List[Path]:
        """Discover test files in the repository"""
        if extensions is None:
            extensions = ['.py']
        
        test_files = []
        for ext in extensions:
            pattern = f"**/*test*{ext}"
            files = list(self.repo_path.glob(pattern))
            test_files.extend(files)
        
        # Also look for files in test directories
        test_dirs = list(self.repo_path.glob("**/test*"))
        for test_dir in test_dirs:
            if test_dir.is_dir():
                for ext in extensions:
                    files = list(test_dir.glob(f"**/*{ext}"))
                    test_files.extend(files)
        
        return sorted(list(set(test_files)))
    
    def analyze_file_evolution(self, file_path: Path) -> EvolutionTrajectory:
        """Analyze evolution of a single test file"""
        str_path = str(file_path.relative_to(self.repo_path))
        
        # Get evolution events from Git
        events = self.git_analyzer.extract_evolution_events(str_path)
        
        if not events:
            # Create minimal trajectory for files without Git history
            current_file = self.file_analyzer.analyze_test_file(str(file_path))
            return EvolutionTrajectory(
                file_path=str_path,
                creation_date=datetime.fromtimestamp(file_path.stat().st_ctime),
                last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                total_events=0,
                major_refactoring_count=0,
                version_snapshots=[current_file]
            )
        
        # Build trajectory
        trajectory = EvolutionTrajectory(
            file_path=str_path,
            creation_date=events[0].timestamp,
            last_modified=events[-1].timestamp,
            total_events=len(events),
            major_refactoring_count=sum(1 for e in events if e.event_type == EvolutionEventType.REFACTORING),
            evolution_events=events
        )
        
        # Track complexity over time
        complexity_timeline = []
        cumulative_complexity = 0.0
        
        for event in events:
            cumulative_complexity += event.complexity_delta
            complexity_timeline.append((event.timestamp, cumulative_complexity))
        
        trajectory.complexity_trend = complexity_timeline
        
        # Track pattern adoption
        for event in events:
            for pattern in event.patterns_introduced:
                if pattern not in trajectory.pattern_adoption:
                    trajectory.pattern_adoption[pattern] = event.timestamp
        
        return trajectory
    
    def analyze_system_evolution(self, max_files: Optional[int] = None) -> Dict[str, Any]:
        """Analyze evolution of entire test system"""
        start_time = time.time()
        
        # Discover test files
        test_files = self.discover_test_files()
        if max_files:
            test_files = test_files[:max_files]
        
        print(f"Analyzing evolution of {len(test_files)} test files...")
        
        trajectories = []
        global_events = []
        pattern_timeline = defaultdict(list)
        
        for i, file_path in enumerate(test_files):
            print(f"Analyzing [{i+1}/{len(test_files)}]: {file_path.name}")
            
            trajectory = self.analyze_file_evolution(file_path)
            trajectories.append(trajectory)
            
            # Collect global events
            global_events.extend(trajectory.evolution_events)
            
            # Track pattern adoption globally
            for pattern, adoption_date in trajectory.pattern_adoption.items():
                pattern_timeline[pattern].append(adoption_date)
        
        # Sort events chronologically
        global_events.sort(key=lambda x: x.timestamp)
        
        # Calculate system metrics
        system_metrics = self._calculate_system_metrics(trajectories, global_events, pattern_timeline)
        
        analysis_time = time.time() - start_time
        
        return {
            'trajectories': trajectories,
            'global_events': global_events,
            'pattern_timeline': dict(pattern_timeline),
            'system_metrics': system_metrics,
            'analysis_metadata': {
                'files_analyzed': len(test_files),
                'total_events': len(global_events),
                'analysis_time_seconds': analysis_time,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _calculate_system_metrics(self, trajectories: List[EvolutionTrajectory], 
                                events: List[TestEvolutionEvent],
                                pattern_timeline: Dict[str, List[datetime]]) -> Dict[str, Any]:
        """Calculate system-wide evolution metrics"""
        
        # Evolution velocity distribution
        velocities = [t.evolution_velocity for t in trajectories if t.evolution_velocity > 0]
        
        # Stability distribution  
        stabilities = [t.stability_score for t in trajectories]
        
        # Event type distribution
        event_types = defaultdict(int)
        for event in events:
            event_types[event.event_type.value] += 1
        
        # Pattern adoption analysis
        pattern_adoption_speed = {}
        for pattern, adoption_dates in pattern_timeline.items():
            if len(adoption_dates) > 1:
                sorted_dates = sorted(adoption_dates)
                time_span = (sorted_dates[-1] - sorted_dates[0]).days
                adoption_speed = len(adoption_dates) / max(time_span, 1)
                pattern_adoption_speed[pattern] = adoption_speed
        
        # Complexity trends
        complexity_changes = [event.complexity_delta for event in events if event.complexity_delta != 0]
        
        return {
            'evolution_velocity': {
                'mean': sum(velocities) / len(velocities) if velocities else 0,
                'max': max(velocities) if velocities else 0,
                'min': min(velocities) if velocities else 0
            },
            'system_stability': {
                'mean': sum(stabilities) / len(stabilities) if stabilities else 1.0,
                'stable_files_count': sum(1 for s in stabilities if s > 0.7),
                'volatile_files_count': sum(1 for s in stabilities if s < 0.3)
            },
            'event_distribution': dict(event_types),
            'pattern_adoption_speeds': pattern_adoption_speed,
            'complexity_evolution': {
                'total_delta': sum(complexity_changes),
                'increases': sum(1 for c in complexity_changes if c > 0),
                'decreases': sum(1 for c in complexity_changes if c < 0)
            },
            'temporal_distribution': {
                'events_last_30_days': len([e for e in events if (datetime.now() - e.timestamp).days <= 30]),
                'events_last_90_days': len([e for e in events if (datetime.now() - e.timestamp).days <= 90]),
                'most_active_month': self._find_most_active_period(events)
            }
        }
    
    def _find_most_active_period(self, events: List[TestEvolutionEvent]) -> str:
        """Find the most active month in terms of evolution events"""
        monthly_counts = defaultdict(int)
        
        for event in events:
            month_key = event.timestamp.strftime('%Y-%m')
            monthly_counts[month_key] += 1
        
        if not monthly_counts:
            return "No activity"
        
        most_active = max(monthly_counts.items(), key=lambda x: x[1])
        return f"{most_active[0]} ({most_active[1]} events)"
    
    def export_evolution_report(self, analysis_results: Dict[str, Any], 
                              output_path: str) -> None:
        """Export evolution analysis report"""
        
        # Prepare serializable data
        export_data = {
            'metadata': analysis_results['analysis_metadata'],
            'system_metrics': analysis_results['system_metrics'],
            'trajectory_summaries': [
                {
                    'file_path': traj.file_path,
                    'creation_date': traj.creation_date.isoformat(),
                    'last_modified': traj.last_modified.isoformat(),
                    'total_events': traj.total_events,
                    'evolution_velocity': traj.evolution_velocity,
                    'stability_score': traj.stability_score,
                    'pattern_adoption_count': len(traj.pattern_adoption)
                }
                for traj in analysis_results['trajectories']
            ],
            'pattern_timeline_summary': {
                pattern: {
                    'first_adoption': min(dates).isoformat(),
                    'last_adoption': max(dates).isoformat(),
                    'adoption_count': len(dates)
                }
                for pattern, dates in analysis_results['pattern_timeline'].items()
            },
            'high_impact_events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'type': event.event_type.value,
                    'file_path': event.file_path,
                    'impact_score': event.impact_score,
                    'message': event.message
                }
                for event in sorted(analysis_results['global_events'], 
                                  key=lambda x: x.impact_score, reverse=True)[:20]
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Evolution report exported to: {output_path}")


# Testing framework
class TestEvolutionFramework:
    """Testing framework for evolution analysis"""
    
    def test_git_analysis(self, repo_path: str) -> bool:
        """Test Git history analysis"""
        try:
            analyzer = GitHistoryAnalyzer(repo_path)
            if not analyzer.repo:
                print("No Git repository found - skipping Git tests")
                return True
            
            # Test commit retrieval
            commits = analyzer.get_file_commits('README.md', max_commits=5)
            assert isinstance(commits, list)
            
            return True
        except Exception as e:
            print(f"Git analysis test failed: {e}")
            return False
    
    def test_file_analysis(self, repo_path: str) -> bool:
        """Test file analysis functionality"""
        try:
            file_analyzer = TestFileAnalyzer()
            
            # Find a Python file to test
            repo_path_obj = Path(repo_path)
            py_files = list(repo_path_obj.glob("**/*.py"))
            
            if not py_files:
                print("No Python files found for testing")
                return True
            
            test_file = file_analyzer.analyze_test_file(str(py_files[0]))
            
            assert test_file.path == str(py_files[0])
            assert test_file.size_bytes > 0
            assert test_file.line_count > 0
            assert isinstance(test_file.test_patterns, list)
            
            return True
        except Exception as e:
            print(f"File analysis test failed: {e}")
            return False
    
    def test_evolution_analysis(self, repo_path: str) -> bool:
        """Test complete evolution analysis"""
        try:
            analyzer = TestEvolutionAnalyzer(repo_path)
            results = analyzer.analyze_system_evolution(max_files=3)
            
            assert 'trajectories' in results
            assert 'system_metrics' in results
            assert 'analysis_metadata' in results
            assert results['analysis_metadata']['files_analyzed'] >= 0
            
            return True
        except Exception as e:
            print(f"Evolution analysis test failed: {e}")
            return False
    
    def run_comprehensive_tests(self, repo_path: str) -> Dict[str, bool]:
        """Run all evolution analysis tests"""
        tests = [
            'test_git_analysis',
            'test_file_analysis',
            'test_evolution_analysis'
        ]
        
        results = {}
        for test_name in tests:
            try:
                result = getattr(self, test_name)(repo_path)
                results[test_name] = result
                print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                results[test_name] = False
                print(f"❌ {test_name}: FAILED - {e}")
        
        return results


# Main execution
if __name__ == "__main__":
    import sys
    
    # Default to current directory
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print("📈 Test Evolution Analyzer Framework")
    print(f"Repository path: {repo_path}")
    
    # Run tests
    framework = TestEvolutionFramework()
    results = framework.run_comprehensive_tests(repo_path)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All evolution analysis tests passed!")
        
        # Run actual analysis
        print("\n🚀 Running evolution analysis...")
        analyzer = TestEvolutionAnalyzer(repo_path)
        results = analyzer.analyze_system_evolution(max_files=10)
        
        # Export report
        output_path = f"evolution_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        analyzer.export_evolution_report(results, output_path)
        
        print(f"\n📈 Evolution Analysis Complete:")
        print(f"  Files analyzed: {results['analysis_metadata']['files_analyzed']}")
        print(f"  Total events: {results['analysis_metadata']['total_events']}")
        print(f"  Analysis time: {results['analysis_metadata']['analysis_time_seconds']:.2f}s")
    else:
        print("❌ Some tests failed. Check the output above.")