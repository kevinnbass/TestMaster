#!/usr/bin/env python3
"""
🏗️ MODULE: Personal Coding Behavior Analyzer - Habit & Workflow Analysis
==================================================================

📋 PURPOSE:
    Analyzes personal coding behaviors, habits, and workflow patterns to provide
    insights into development practices, productivity patterns, and workflow
    optimization opportunities.

🎯 CORE FUNCTIONALITY:
    • Coding session pattern analysis
    • Preference detection (tools, patterns, styles)
    • Workflow analysis and optimization
    • Productivity metrics tracking
    • Behavioral insight generation

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 [2025-08-23 06:00:00] | Agent B | 🆕 FEATURE
   └─ Goal: Create personal coding behavior analysis system
   └─ Changes: Initial implementation with session and workflow analysis
   └─ Impact: Enables personal habit tracking and workflow optimization

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent B
🔧 Language: Python
📦 Dependencies: git, datetime, statistics, json, subprocess
🎯 Integration Points: Trend analyzer, pattern detector
⚡ Performance Notes: Git history analysis may be slow on large repos
🔒 Security Notes: No sensitive data exposure

Author: Agent B - Behavior Analysis Specialist
"""

import os
import json
import subprocess
import statistics
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from pathlib import Path
import re
import ast

@dataclass
class CodingSession:
    """Represents a coding session"""
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    files_modified: List[str]
    lines_added: int
    lines_deleted: int
    commit_count: int
    is_productive: bool  # Based on output metrics

@dataclass
class CodingPreference:
    """Represents a detected coding preference"""
    preference_type: str  # 'naming', 'structure', 'tooling', 'pattern'
    preference_name: str
    frequency: float
    consistency_score: float  # 0-1
    examples: List[str]
    confidence: float

@dataclass
class WorkflowPattern:
    """Represents a workflow pattern"""
    pattern_name: str
    description: str
    frequency: int
    typical_sequence: List[str]
    productivity_impact: str  # 'positive', 'negative', 'neutral'
    optimization_suggestion: str

@dataclass
class ProductivityMetric:
    """Productivity measurement"""
    metric_name: str
    value: float
    unit: str
    trend: str  # 'increasing', 'decreasing', 'stable'
    percentile: float  # Compared to historical data

@dataclass
class BehaviorInsight:
    """Behavioral insight derived from analysis"""
    insight_type: str
    title: str
    description: str
    impact_level: str  # 'high', 'medium', 'low'
    evidence: List[str]
    recommendation: str
    potential_improvement: str

@dataclass
class BehaviorAnalysisReport:
    """Comprehensive behavior analysis report"""
    analysis_period: Tuple[datetime, datetime]
    coding_sessions: List[CodingSession]
    preferences: List[CodingPreference]
    workflow_patterns: List[WorkflowPattern]
    productivity_metrics: List[ProductivityMetric]
    behavioral_insights: List[BehaviorInsight]
    peak_productivity_times: List[Tuple[int, int]]  # Hour ranges
    workflow_recommendations: List[str]

class PersonalCodingBehaviorAnalyzer:
    """
    Analyzes personal coding behaviors and habits
    Integrates with git history and file analysis
    """
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.sessions = []
        self.commit_history = []
        self.file_modifications = defaultdict(list)
        self.time_patterns = defaultdict(int)
        self.preference_data = defaultdict(list)
        
    def analyze_coding_behavior(self, days_back: int = 30) -> BehaviorAnalysisReport:
        """
        Comprehensive analysis of personal coding behavior
        """
        print(f"Analyzing coding behavior for last {days_back} days...")
        
        # Analyze git history
        self._analyze_git_history(days_back)
        
        # Extract coding sessions
        coding_sessions = self._extract_coding_sessions()
        
        # Detect preferences
        preferences = self._detect_coding_preferences()
        
        # Analyze workflow patterns
        workflow_patterns = self._analyze_workflow_patterns()
        
        # Calculate productivity metrics
        productivity_metrics = self._calculate_productivity_metrics(coding_sessions)
        
        # Generate behavioral insights
        behavioral_insights = self._generate_behavioral_insights(
            coding_sessions, preferences, workflow_patterns, productivity_metrics
        )
        
        # Identify peak productivity times
        peak_times = self._identify_peak_productivity_times(coding_sessions)
        
        # Generate workflow recommendations
        recommendations = self._generate_workflow_recommendations(
            workflow_patterns, productivity_metrics, behavioral_insights
        )
        
        return BehaviorAnalysisReport(
            analysis_period=(
                datetime.now() - timedelta(days=days_back),
                datetime.now()
            ),
            coding_sessions=coding_sessions,
            preferences=preferences,
            workflow_patterns=workflow_patterns,
            productivity_metrics=productivity_metrics,
            behavioral_insights=behavioral_insights,
            peak_productivity_times=peak_times,
            workflow_recommendations=recommendations
        )
    
    def _analyze_git_history(self, days_back: int):
        """Analyze git commit history"""
        try:
            # Get git log for the specified period
            since_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Get commit history with stats
            cmd = [
                'git', 'log',
                f'--since={since_date}',
                '--pretty=format:%H|%ai|%s',
                '--numstat'
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self._parse_git_log(result.stdout)
            else:
                print(f"Git analysis skipped: {result.stderr}")
                
        except Exception as e:
            print(f"Git analysis failed: {e}")
            # Continue with file-based analysis
    
    def _parse_git_log(self, log_output: str):
        """Parse git log output"""
        lines = log_output.strip().split('\n')
        current_commit = None
        
        for line in lines:
            if '|' in line and not '\t' in line:
                # Commit line
                parts = line.split('|')
                if len(parts) >= 3:
                    commit_hash = parts[0]
                    timestamp_str = parts[1]
                    message = parts[2]
                    
                    # Parse timestamp
                    timestamp = datetime.strptime(
                        timestamp_str.split(' ')[0] + ' ' + timestamp_str.split(' ')[1],
                        '%Y-%m-%d %H:%M:%S'
                    )
                    
                    current_commit = {
                        'hash': commit_hash,
                        'timestamp': timestamp,
                        'message': message,
                        'files': [],
                        'additions': 0,
                        'deletions': 0
                    }
                    self.commit_history.append(current_commit)
                    
            elif '\t' in line and current_commit:
                # File stat line
                parts = line.split('\t')
                if len(parts) == 3:
                    additions = int(parts[0]) if parts[0] != '-' else 0
                    deletions = int(parts[1]) if parts[1] != '-' else 0
                    filename = parts[2]
                    
                    current_commit['files'].append(filename)
                    current_commit['additions'] += additions
                    current_commit['deletions'] += deletions
                    
                    # Track file modifications
                    self.file_modifications[filename].append({
                        'timestamp': current_commit['timestamp'],
                        'additions': additions,
                        'deletions': deletions
                    })
    
    def _extract_coding_sessions(self) -> List[CodingSession]:
        """Extract coding sessions from commit history"""
        sessions = []
        
        if not self.commit_history:
            # No git history, create synthetic session
            return [CodingSession(
                start_time=datetime.now() - timedelta(hours=1),
                end_time=datetime.now(),
                duration_minutes=60,
                files_modified=[],
                lines_added=0,
                lines_deleted=0,
                commit_count=0,
                is_productive=False
            )]
        
        # Group commits into sessions (commits within 2 hours of each other)
        session_threshold = timedelta(hours=2)
        current_session_commits = []
        
        for i, commit in enumerate(self.commit_history):
            if not current_session_commits:
                current_session_commits.append(commit)
            else:
                time_diff = commit['timestamp'] - current_session_commits[-1]['timestamp']
                
                if time_diff <= session_threshold:
                    current_session_commits.append(commit)
                else:
                    # End current session and start new one
                    session = self._create_session_from_commits(current_session_commits)
                    sessions.append(session)
                    current_session_commits = [commit]
        
        # Add last session
        if current_session_commits:
            session = self._create_session_from_commits(current_session_commits)
            sessions.append(session)
        
        return sessions
    
    def _create_session_from_commits(self, commits: List[Dict]) -> CodingSession:
        """Create a coding session from a group of commits"""
        start_time = min(c['timestamp'] for c in commits)
        end_time = max(c['timestamp'] for c in commits)
        duration = (end_time - start_time).total_seconds() / 60
        
        # Aggregate metrics
        all_files = set()
        total_additions = 0
        total_deletions = 0
        
        for commit in commits:
            all_files.update(commit['files'])
            total_additions += commit['additions']
            total_deletions += commit['deletions']
        
        # Determine if session was productive
        is_productive = (
            len(commits) >= 2 or
            total_additions > 50 or
            len(all_files) >= 3
        )
        
        return CodingSession(
            start_time=start_time,
            end_time=end_time,
            duration_minutes=max(30, duration),  # Minimum 30 min session
            files_modified=list(all_files),
            lines_added=total_additions,
            lines_deleted=total_deletions,
            commit_count=len(commits),
            is_productive=is_productive
        )
    
    def _detect_coding_preferences(self) -> List[CodingPreference]:
        """Detect coding preferences from patterns"""
        preferences = []
        
        # Analyze file naming preferences
        if self.file_modifications:
            naming_patterns = self._analyze_naming_preferences()
            preferences.extend(naming_patterns)
        
        # Analyze commit message patterns
        if self.commit_history:
            commit_patterns = self._analyze_commit_patterns()
            preferences.extend(commit_patterns)
        
        # Analyze code structure preferences (from actual files)
        structure_prefs = self._analyze_structure_preferences()
        preferences.extend(structure_prefs)
        
        return preferences
    
    def _analyze_naming_preferences(self) -> List[CodingPreference]:
        """Analyze file and variable naming preferences"""
        preferences = []
        
        # Analyze file naming
        file_names = list(self.file_modifications.keys())
        
        # Check for snake_case vs camelCase
        snake_count = sum(1 for f in file_names if '_' in f and f.endswith('.py'))
        camel_count = sum(1 for f in file_names if re.search(r'[a-z][A-Z]', f))
        
        if snake_count > camel_count:
            preferences.append(CodingPreference(
                preference_type='naming',
                preference_name='snake_case_files',
                frequency=snake_count / max(1, len(file_names)),
                consistency_score=snake_count / max(1, snake_count + camel_count),
                examples=file_names[:3],
                confidence=0.8
            ))
        
        return preferences
    
    def _analyze_commit_patterns(self) -> List[CodingPreference]:
        """Analyze commit message patterns"""
        preferences = []
        
        # Extract commit messages
        messages = [c['message'] for c in self.commit_history]
        
        # Check for conventional commits
        conventional_count = sum(1 for m in messages if re.match(r'^(feat|fix|docs|style|refactor|test|chore):', m))
        
        if conventional_count > len(messages) * 0.3:
            preferences.append(CodingPreference(
                preference_type='tooling',
                preference_name='conventional_commits',
                frequency=conventional_count / max(1, len(messages)),
                consistency_score=conventional_count / max(1, len(messages)),
                examples=messages[:3],
                confidence=0.9
            ))
        
        # Check for emoji usage
        emoji_count = sum(1 for m in messages if re.search(r'[🔧🐛✨📝🚀💡⚡🔥]', m))
        
        if emoji_count > 0:
            preferences.append(CodingPreference(
                preference_type='tooling',
                preference_name='emoji_commits',
                frequency=emoji_count / max(1, len(messages)),
                consistency_score=emoji_count / max(1, len(messages)),
                examples=[m for m in messages if '🔧' in m or '✨' in m][:3],
                confidence=0.7
            ))
        
        return preferences
    
    def _analyze_structure_preferences(self) -> List[CodingPreference]:
        """Analyze code structure preferences"""
        preferences = []
        
        # Sample Python files for structure analysis
        py_files = list(self.repo_path.rglob("*.py"))[:20]  # Sample first 20
        
        class_count = 0
        function_count = 0
        async_count = 0
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_count += 1
                    elif isinstance(node, ast.FunctionDef):
                        function_count += 1
                    elif isinstance(node, ast.AsyncFunctionDef):
                        async_count += 1
                        
            except:
                continue
        
        # Determine preferences
        if class_count > function_count * 0.3:
            preferences.append(CodingPreference(
                preference_type='structure',
                preference_name='object_oriented',
                frequency=class_count / max(1, class_count + function_count),
                consistency_score=0.7,
                examples=['Heavy use of classes'],
                confidence=0.8
            ))
        
        if async_count > 0:
            preferences.append(CodingPreference(
                preference_type='pattern',
                preference_name='async_programming',
                frequency=async_count / max(1, function_count),
                consistency_score=0.6,
                examples=['Async/await patterns used'],
                confidence=0.7
            ))
        
        return preferences
    
    def _analyze_workflow_patterns(self) -> List[WorkflowPattern]:
        """Analyze workflow patterns from behavior"""
        patterns = []
        
        # Analyze commit frequency patterns
        if self.commit_history:
            commit_pattern = self._analyze_commit_frequency()
            patterns.append(commit_pattern)
        
        # Analyze file modification patterns
        if self.file_modifications:
            mod_pattern = self._analyze_modification_patterns()
            patterns.append(mod_pattern)
        
        # Analyze refactoring patterns
        refactor_pattern = self._detect_refactoring_patterns()
        if refactor_pattern:
            patterns.append(refactor_pattern)
        
        return patterns
    
    def _analyze_commit_frequency(self) -> WorkflowPattern:
        """Analyze commit frequency patterns"""
        if not self.commit_history:
            return WorkflowPattern(
                pattern_name='unknown_commit_pattern',
                description='Insufficient data for commit pattern analysis',
                frequency=0,
                typical_sequence=[],
                productivity_impact='neutral',
                optimization_suggestion='Start tracking commits for better insights'
            )
        
        # Calculate average commits per session
        sessions = self._extract_coding_sessions()
        avg_commits = statistics.mean([s.commit_count for s in sessions]) if sessions else 0
        
        if avg_commits < 2:
            return WorkflowPattern(
                pattern_name='large_commits',
                description='Tendency to make large, infrequent commits',
                frequency=len(sessions),
                typical_sequence=['Code for extended period', 'Single large commit'],
                productivity_impact='negative',
                optimization_suggestion='Consider more frequent, atomic commits for better history'
            )
        elif avg_commits > 5:
            return WorkflowPattern(
                pattern_name='frequent_commits',
                description='High frequency of small commits',
                frequency=len(sessions),
                typical_sequence=['Small change', 'Commit', 'Repeat'],
                productivity_impact='positive',
                optimization_suggestion='Good commit hygiene - keep it up!'
            )
        else:
            return WorkflowPattern(
                pattern_name='balanced_commits',
                description='Balanced commit frequency',
                frequency=len(sessions),
                typical_sequence=['Feature work', '2-5 commits per session'],
                productivity_impact='positive',
                optimization_suggestion='Commit pattern is well-balanced'
            )
    
    def _analyze_modification_patterns(self) -> WorkflowPattern:
        """Analyze file modification patterns"""
        # Find frequently modified files
        hot_files = sorted(
            self.file_modifications.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]
        
        if hot_files and len(hot_files[0][1]) > 10:
            return WorkflowPattern(
                pattern_name='hotspot_focus',
                description='Frequent modifications to specific files',
                frequency=len(hot_files[0][1]),
                typical_sequence=[f for f, _ in hot_files[:3]],
                productivity_impact='negative',
                optimization_suggestion='Consider refactoring frequently modified files'
            )
        else:
            return WorkflowPattern(
                pattern_name='distributed_changes',
                description='Changes distributed across codebase',
                frequency=len(self.file_modifications),
                typical_sequence=['Various files modified'],
                productivity_impact='positive',
                optimization_suggestion='Good distribution of changes'
            )
    
    def _detect_refactoring_patterns(self) -> Optional[WorkflowPattern]:
        """Detect refactoring patterns from commits"""
        if not self.commit_history:
            return None
        
        refactor_keywords = ['refactor', 'cleanup', 'reorganize', 'restructure', 'optimize']
        refactor_commits = [
            c for c in self.commit_history
            if any(keyword in c['message'].lower() for keyword in refactor_keywords)
        ]
        
        if len(refactor_commits) > len(self.commit_history) * 0.1:
            return WorkflowPattern(
                pattern_name='regular_refactoring',
                description='Regular refactoring and code improvement',
                frequency=len(refactor_commits),
                typical_sequence=['Feature development', 'Refactoring', 'Testing'],
                productivity_impact='positive',
                optimization_suggestion='Excellent practice - regular refactoring maintains code quality'
            )
        
        return None
    
    def _calculate_productivity_metrics(self, sessions: List[CodingSession]) -> List[ProductivityMetric]:
        """Calculate productivity metrics"""
        metrics = []
        
        if not sessions:
            return metrics
        
        # Lines per hour
        total_lines = sum(s.lines_added for s in sessions)
        total_hours = sum(s.duration_minutes for s in sessions) / 60
        
        if total_hours > 0:
            lines_per_hour = total_lines / total_hours
            metrics.append(ProductivityMetric(
                metric_name='lines_per_hour',
                value=round(lines_per_hour, 1),
                unit='lines/hour',
                trend=self._calculate_trend([s.lines_added for s in sessions]),
                percentile=self._calculate_percentile(lines_per_hour, [50, 100, 150, 200])
            ))
        
        # Commits per session
        avg_commits = statistics.mean([s.commit_count for s in sessions])
        metrics.append(ProductivityMetric(
            metric_name='commits_per_session',
            value=round(avg_commits, 1),
            unit='commits',
            trend=self._calculate_trend([s.commit_count for s in sessions]),
            percentile=self._calculate_percentile(avg_commits, [1, 3, 5, 10])
        ))
        
        # Session duration
        avg_duration = statistics.mean([s.duration_minutes for s in sessions])
        metrics.append(ProductivityMetric(
            metric_name='average_session_duration',
            value=round(avg_duration, 0),
            unit='minutes',
            trend='stable',
            percentile=self._calculate_percentile(avg_duration, [30, 60, 120, 240])
        ))
        
        # Productivity rate
        productive_sessions = sum(1 for s in sessions if s.is_productive)
        productivity_rate = productive_sessions / len(sessions) if sessions else 0
        
        metrics.append(ProductivityMetric(
            metric_name='productivity_rate',
            value=round(productivity_rate * 100, 1),
            unit='%',
            trend=self._calculate_trend([1 if s.is_productive else 0 for s in sessions]),
            percentile=productivity_rate * 100
        ))
        
        return metrics
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from values"""
        if len(values) < 3:
            return 'stable'
        
        # Compare first half to second half
        mid = len(values) // 2
        first_half_avg = statistics.mean(values[:mid])
        second_half_avg = statistics.mean(values[mid:])
        
        change = (second_half_avg - first_half_avg) / max(1, first_half_avg)
        
        if change > 0.1:
            return 'increasing'
        elif change < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_percentile(self, value: float, thresholds: List[float]) -> float:
        """Calculate percentile based on thresholds"""
        percentile = 0
        for threshold in thresholds:
            if value >= threshold:
                percentile += 25
        return min(100, percentile)
    
    def _generate_behavioral_insights(self, sessions: List[CodingSession],
                                     preferences: List[CodingPreference],
                                     patterns: List[WorkflowPattern],
                                     metrics: List[ProductivityMetric]) -> List[BehaviorInsight]:
        """Generate behavioral insights"""
        insights = []
        
        # Session timing insights
        if sessions:
            peak_hours = self._identify_peak_productivity_times(sessions)
            if peak_hours:
                insights.append(BehaviorInsight(
                    insight_type='productivity_timing',
                    title='Peak Productivity Hours Identified',
                    description=f'Most productive during {peak_hours[0][0]}:00-{peak_hours[0][1]}:00',
                    impact_level='high',
                    evidence=[f'Based on {len(sessions)} coding sessions'],
                    recommendation='Schedule complex tasks during peak hours',
                    potential_improvement='15-25% productivity gain'
                ))
        
        # Workflow insights
        for pattern in patterns:
            if pattern.productivity_impact == 'negative':
                insights.append(BehaviorInsight(
                    insight_type='workflow_optimization',
                    title=f'Workflow Pattern: {pattern.pattern_name}',
                    description=pattern.description,
                    impact_level='medium',
                    evidence=pattern.typical_sequence,
                    recommendation=pattern.optimization_suggestion,
                    potential_improvement='10-20% efficiency gain'
                ))
        
        # Consistency insights
        if preferences:
            consistency_scores = [p.consistency_score for p in preferences]
            avg_consistency = statistics.mean(consistency_scores)
            
            if avg_consistency > 0.7:
                insights.append(BehaviorInsight(
                    insight_type='coding_consistency',
                    title='High Coding Consistency',
                    description='Consistent coding patterns and preferences detected',
                    impact_level='low',
                    evidence=[f'{p.preference_name}: {p.consistency_score:.1%}' for p in preferences[:3]],
                    recommendation='Maintain current consistency levels',
                    potential_improvement='Already optimized'
                ))
        
        # Productivity insights
        for metric in metrics:
            if metric.metric_name == 'productivity_rate' and metric.value < 50:
                insights.append(BehaviorInsight(
                    insight_type='productivity_improvement',
                    title='Productivity Enhancement Opportunity',
                    description=f'Current productivity rate: {metric.value}%',
                    impact_level='high',
                    evidence=['Many sessions with low output'],
                    recommendation='Focus on completing tasks within sessions',
                    potential_improvement='30-50% productivity increase possible'
                ))
        
        return insights
    
    def _identify_peak_productivity_times(self, sessions: List[CodingSession]) -> List[Tuple[int, int]]:
        """Identify peak productivity hours"""
        if not sessions:
            return []
        
        # Track productivity by hour of day
        hour_productivity = defaultdict(lambda: {'lines': 0, 'sessions': 0})
        
        for session in sessions:
            hour = session.start_time.hour
            hour_productivity[hour]['lines'] += session.lines_added
            hour_productivity[hour]['sessions'] += 1
        
        # Calculate average lines per session per hour
        hour_scores = {}
        for hour, data in hour_productivity.items():
            if data['sessions'] > 0:
                hour_scores[hour] = data['lines'] / data['sessions']
        
        if not hour_scores:
            return []
        
        # Find top hours
        sorted_hours = sorted(hour_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Group consecutive hours
        peak_ranges = []
        current_range = [sorted_hours[0][0]]
        
        for hour, _ in sorted_hours[1:3]:  # Top 3 hours
            if hour == current_range[-1] + 1:
                current_range.append(hour)
            else:
                if len(current_range) > 1:
                    peak_ranges.append((current_range[0], current_range[-1]))
                current_range = [hour]
        
        if len(current_range) > 1:
            peak_ranges.append((current_range[0], current_range[-1]))
        elif current_range:
            peak_ranges.append((current_range[0], current_range[0] + 1))
        
        return peak_ranges
    
    def _generate_workflow_recommendations(self, patterns: List[WorkflowPattern],
                                          metrics: List[ProductivityMetric],
                                          insights: List[BehaviorInsight]) -> List[str]:
        """Generate workflow improvement recommendations"""
        recommendations = []
        
        # Based on patterns
        for pattern in patterns:
            if pattern.optimization_suggestion and pattern.productivity_impact != 'positive':
                recommendations.append(pattern.optimization_suggestion)
        
        # Based on metrics
        for metric in metrics:
            if metric.metric_name == 'lines_per_hour' and metric.value < 50:
                recommendations.append('Consider using code generation tools or snippets for boilerplate')
            elif metric.metric_name == 'average_session_duration' and metric.value < 30:
                recommendations.append('Try longer, focused coding sessions for better flow state')
            elif metric.metric_name == 'commits_per_session' and metric.value < 2:
                recommendations.append('Make more frequent commits to track progress better')
        
        # Based on insights
        high_impact_insights = [i for i in insights if i.impact_level == 'high']
        for insight in high_impact_insights[:2]:
            if insight.recommendation not in recommendations:
                recommendations.append(insight.recommendation)
        
        # General recommendations
        if not recommendations:
            recommendations.append('Continue current practices - workflow is well-optimized')
        
        return recommendations[:5]  # Top 5 recommendations
    
    def export_behavior_analysis(self, report: BehaviorAnalysisReport, output_file: str):
        """Export behavior analysis to JSON"""
        export_data = {
            'analysis_period': {
                'start': report.analysis_period[0].isoformat(),
                'end': report.analysis_period[1].isoformat()
            },
            'summary': {
                'total_sessions': len(report.coding_sessions),
                'total_lines_written': sum(s.lines_added for s in report.coding_sessions),
                'average_session_duration': statistics.mean([s.duration_minutes for s in report.coding_sessions]) if report.coding_sessions else 0,
                'productivity_rate': sum(1 for s in report.coding_sessions if s.is_productive) / max(1, len(report.coding_sessions)) * 100
            },
            'coding_sessions': [asdict(s) for s in report.coding_sessions],
            'preferences': [asdict(p) for p in report.preferences],
            'workflow_patterns': [asdict(w) for w in report.workflow_patterns],
            'productivity_metrics': [asdict(m) for m in report.productivity_metrics],
            'behavioral_insights': [asdict(i) for i in report.behavioral_insights],
            'peak_productivity_times': report.peak_productivity_times,
            'workflow_recommendations': report.workflow_recommendations,
            'metadata': {
                'analyzer': 'PersonalCodingBehaviorAnalyzer',
                'version': '1.0',
                'generated_at': datetime.now().isoformat()
            }
        }
        
        # Convert datetime objects
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=serialize_datetime)

def main():
    """Run personal coding behavior analysis"""
    print("=" * 60)
    print("Personal Coding Behavior Analyzer")
    print("Agent B - Phase 1 Hour 33 Implementation")
    print("=" * 60)
    
    # Analyze current repository
    analyzer = PersonalCodingBehaviorAnalyzer(r"C:\Users\kbass\OneDrive\Documents\testmaster")
    
    print("\nAnalyzing coding behavior patterns...")
    report = analyzer.analyze_coding_behavior(days_back=30)
    
    # Display results
    print("\nCODING SESSIONS SUMMARY")
    print("-" * 40)
    print(f"Total sessions: {len(report.coding_sessions)}")
    if report.coding_sessions:
        total_lines = sum(s.lines_added for s in report.coding_sessions)
        print(f"Total lines written: {total_lines:,}")
        avg_duration = statistics.mean([s.duration_minutes for s in report.coding_sessions])
        print(f"Average session duration: {avg_duration:.0f} minutes")
    
    print("\nCODING PREFERENCES")
    print("-" * 40)
    for pref in report.preferences[:5]:
        print(f"{pref.preference_name}: {pref.frequency:.1%} frequency")
    
    print("\nWORKFLOW PATTERNS")
    print("-" * 40)
    for pattern in report.workflow_patterns:
        print(f"{pattern.pattern_name}: {pattern.description}")
        print(f"  Impact: {pattern.productivity_impact}")
    
    print("\nPRODUCTIVITY METRICS")
    print("-" * 40)
    for metric in report.productivity_metrics:
        print(f"{metric.metric_name}: {metric.value} {metric.unit} ({metric.trend})")
    
    print("\nBEHAVIORAL INSIGHTS")
    print("-" * 40)
    for insight in report.behavioral_insights:
        print(f"[{insight.impact_level.upper()}] {insight.title}")
        print(f"  {insight.description}")
    
    print("\nWORKFLOW RECOMMENDATIONS")
    print("-" * 40)
    for i, rec in enumerate(report.workflow_recommendations, 1):
        print(f"{i}. {rec}")
    
    # Export to JSON
    output_file = "personal_coding_behavior.json"
    analyzer.export_behavior_analysis(report, output_file)
    print(f"\nBehavior analysis exported to: {output_file}")
    
    print("\nPersonal Coding Behavior Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()