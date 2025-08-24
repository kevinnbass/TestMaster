#!/usr/bin/env python3
"""
🏗️ MODULE: Personal Development Trend Analyzer - Advanced Trend Detection
==================================================================

📋 PURPOSE:
    Analyzes personal development trends across projects by tracking skill progression,
    technology adoption patterns, and coding practice evolution over time. Integrates
    with Agent B's streaming intelligence platform for real-time insights.

🎯 CORE FUNCTIONALITY:
    • Skill progression tracking and analysis
    • Technology adoption pattern detection
    • Code quality trend analysis over time
    • Project complexity evolution tracking
    • Personal productivity metrics and predictions

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 [2025-08-23 05:00:00] | Agent B | 🆕 FEATURE
   └─ Goal: Create personal development trend analysis system
   └─ Changes: Initial implementation with trend detection
   └─ Impact: Enables personal growth tracking and predictions

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent B
🔧 Language: Python
📦 Dependencies: ast, datetime, statistics, json, dataclasses
🎯 Integration Points: Streaming intelligence, pattern analyzer
⚡ Performance Notes: Optimized for incremental analysis
🔒 Security Notes: No sensitive data handling

Author: Agent B - Development Trend Analysis Specialist
"""

import os
import json
import ast
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from pathlib import Path
import subprocess
import re

@dataclass
class SkillMetric:
    """Represents a measurable skill metric"""
    skill_name: str
    proficiency_level: float  # 0-100
    growth_rate: float  # % per period
    practice_frequency: int
    last_used: datetime
    confidence: float  # 0-1

@dataclass
class TechnologyAdoption:
    """Tracks technology adoption patterns"""
    technology: str
    first_used: datetime
    usage_frequency: int
    proficiency_trend: List[float]
    project_count: int
    lines_written: int

@dataclass
class ProjectComplexity:
    """Measures project complexity over time"""
    project_name: str
    timestamp: datetime
    file_count: int
    total_lines: int
    average_complexity: float
    dependency_count: int
    pattern_maturity: float  # 0-1

@dataclass
class DevelopmentTrend:
    """Represents a development trend"""
    trend_type: str  # 'skill', 'technology', 'complexity', 'quality'
    metric_name: str
    direction: str  # 'improving', 'declining', 'stable'
    velocity: float  # rate of change
    confidence: float
    prediction: Dict[str, Any]
    evidence: List[str]

@dataclass
class PersonalGrowthReport:
    """Comprehensive personal growth analysis"""
    analysis_period: Tuple[datetime, datetime]
    skill_progression: List[SkillMetric]
    technology_adoption: List[TechnologyAdoption]
    complexity_evolution: List[ProjectComplexity]
    development_trends: List[DevelopmentTrend]
    growth_predictions: Dict[str, Any]
    recommendations: List[str]
    achievements: List[str]

class PersonalDevelopmentTrendAnalyzer:
    """
    Analyzes personal development trends across projects
    Integrates with streaming intelligence for real-time insights
    """
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.projects_analyzed = []
        self.skill_metrics = defaultdict(list)
        self.technology_usage = defaultdict(list)
        self.complexity_history = []
        self.quality_metrics = defaultdict(list)
        self.git_history = []
        
    def analyze_development_trends(self, projects: List[str]) -> PersonalGrowthReport:
        """
        Analyze development trends across multiple projects
        """
        print("Analyzing personal development trends...")
        
        # Collect data from all projects
        for project_path in projects:
            self._analyze_project(project_path)
        
        # Analyze skill progression
        skill_progression = self._analyze_skill_progression()
        
        # Track technology adoption
        technology_adoption = self._analyze_technology_adoption()
        
        # Analyze complexity evolution
        complexity_evolution = self._analyze_complexity_evolution()
        
        # Detect development trends
        development_trends = self._detect_development_trends()
        
        # Generate growth predictions
        growth_predictions = self._generate_growth_predictions(
            skill_progression, technology_adoption, complexity_evolution
        )
        
        # Create recommendations
        recommendations = self._generate_recommendations(
            development_trends, growth_predictions
        )
        
        # Identify achievements
        achievements = self._identify_achievements(
            skill_progression, technology_adoption
        )
        
        return PersonalGrowthReport(
            analysis_period=(datetime.now() - timedelta(days=365), datetime.now()),
            skill_progression=skill_progression,
            technology_adoption=technology_adoption,
            complexity_evolution=complexity_evolution,
            development_trends=development_trends,
            growth_predictions=growth_predictions,
            recommendations=recommendations,
            achievements=achievements
        )
    
    def _analyze_project(self, project_path: str):
        """Analyze a single project for development metrics"""
        project_path = Path(project_path)
        if not project_path.exists():
            return
        
        # Analyze Python files
        py_files = list(project_path.rglob("*.py"))
        
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                self._extract_skill_indicators(tree, py_file)
                self._extract_technology_usage(content, py_file)
                self._measure_code_complexity(tree, py_file)
                
            except Exception as e:
                continue
        
        # Track project complexity
        self.complexity_history.append(ProjectComplexity(
            project_name=project_path.name,
            timestamp=datetime.now(),
            file_count=len(py_files),
            total_lines=sum(self._count_lines(f) for f in py_files),
            average_complexity=self._calculate_average_complexity(py_files),
            dependency_count=self._count_dependencies(project_path),
            pattern_maturity=self._assess_pattern_maturity(py_files)
        ))
    
    def _extract_skill_indicators(self, tree: ast.AST, filepath: Path):
        """Extract indicators of coding skills from AST"""
        skills = {
            'error_handling': 0,
            'async_programming': 0,
            'decorators': 0,
            'generators': 0,
            'context_managers': 0,
            'type_hints': 0,
            'comprehensions': 0,
            'class_design': 0
        }
        
        for node in ast.walk(tree):
            # Error handling
            if isinstance(node, ast.Try):
                skills['error_handling'] += 1
            
            # Async programming
            elif isinstance(node, (ast.AsyncFunctionDef, ast.AsyncWith, ast.AsyncFor)):
                skills['async_programming'] += 1
            
            # Decorators
            elif isinstance(node, ast.FunctionDef) and node.decorator_list:
                skills['decorators'] += len(node.decorator_list)
            
            # Generators
            elif isinstance(node, (ast.Yield, ast.YieldFrom)):
                skills['generators'] += 1
            
            # Context managers
            elif isinstance(node, ast.With):
                skills['context_managers'] += 1
            
            # Type hints
            elif isinstance(node, ast.FunctionDef):
                if node.returns or any(arg.annotation for arg in node.args.args):
                    skills['type_hints'] += 1
            
            # Comprehensions
            elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                skills['comprehensions'] += 1
            
            # Class design
            elif isinstance(node, ast.ClassDef):
                skills['class_design'] += 1
        
        # Update skill metrics
        for skill, count in skills.items():
            if count > 0:
                self.skill_metrics[skill].append({
                    'file': str(filepath),
                    'count': count,
                    'timestamp': datetime.now()
                })
    
    def _extract_technology_usage(self, content: str, filepath: Path):
        """Extract technology/library usage from imports"""
        import_pattern = r'^(?:from\s+(\S+)|import\s+(\S+))'
        
        for line in content.split('\n'):
            match = re.match(import_pattern, line.strip())
            if match:
                module = match.group(1) or match.group(2)
                if module:
                    base_module = module.split('.')[0]
                    self.technology_usage[base_module].append({
                        'file': str(filepath),
                        'timestamp': datetime.now()
                    })
    
    def _measure_code_complexity(self, tree: ast.AST, filepath: Path):
        """Measure code complexity metrics"""
        complexity_score = 0
        
        # Count decision points (McCabe complexity)
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try,
                                ast.With, ast.ExceptHandler, ast.Lambda)):
                complexity_score += 1
        
        self.quality_metrics['complexity'].append({
            'file': str(filepath),
            'score': complexity_score,
            'timestamp': datetime.now()
        })
    
    def _analyze_skill_progression(self) -> List[SkillMetric]:
        """Analyze how skills have progressed over time"""
        skill_progression = []
        
        for skill, occurrences in self.skill_metrics.items():
            if not occurrences:
                continue
            
            # Calculate growth rate
            if len(occurrences) > 1:
                early_avg = statistics.mean([o['count'] for o in occurrences[:len(occurrences)//2]])
                late_avg = statistics.mean([o['count'] for o in occurrences[len(occurrences)//2:]])
                growth_rate = ((late_avg - early_avg) / (early_avg + 1)) * 100
            else:
                growth_rate = 0
            
            # Calculate proficiency level (0-100)
            total_usage = sum(o['count'] for o in occurrences)
            proficiency = min(100, (total_usage / 10) * 10)  # Scale to 0-100
            
            skill_progression.append(SkillMetric(
                skill_name=skill,
                proficiency_level=proficiency,
                growth_rate=growth_rate,
                practice_frequency=len(occurrences),
                last_used=max(o['timestamp'] for o in occurrences),
                confidence=min(1.0, len(occurrences) / 10)
            ))
        
        return sorted(skill_progression, key=lambda x: x.proficiency_level, reverse=True)
    
    def _analyze_technology_adoption(self) -> List[TechnologyAdoption]:
        """Analyze technology adoption patterns"""
        adoption_patterns = []
        
        for tech, usages in self.technology_usage.items():
            if not usages or tech in ['os', 'sys', 'json', 're']:  # Skip standard libs
                continue
            
            first_used = min(u['timestamp'] for u in usages)
            usage_count = len(usages)
            
            # Calculate proficiency trend
            proficiency_trend = []
            for i in range(0, len(usages), max(1, len(usages) // 5)):
                subset = usages[i:i + len(usages) // 5]
                proficiency_trend.append(len(subset))
            
            adoption_patterns.append(TechnologyAdoption(
                technology=tech,
                first_used=first_used,
                usage_frequency=usage_count,
                proficiency_trend=proficiency_trend,
                project_count=len(set(u['file'].split('/')[0] for u in usages)),
                lines_written=usage_count * 10  # Estimate
            ))
        
        return sorted(adoption_patterns, key=lambda x: x.usage_frequency, reverse=True)[:20]
    
    def _analyze_complexity_evolution(self) -> List[ProjectComplexity]:
        """Analyze how project complexity has evolved"""
        return sorted(self.complexity_history, key=lambda x: x.timestamp)
    
    def _detect_development_trends(self) -> List[DevelopmentTrend]:
        """Detect trends in development patterns"""
        trends = []
        
        # Skill trends
        for skill, occurrences in self.skill_metrics.items():
            if len(occurrences) >= 3:
                counts = [o['count'] for o in occurrences]
                trend_direction = self._calculate_trend_direction(counts)
                velocity = self._calculate_velocity(counts)
                
                trends.append(DevelopmentTrend(
                    trend_type='skill',
                    metric_name=skill,
                    direction=trend_direction,
                    velocity=velocity,
                    confidence=min(1.0, len(occurrences) / 10),
                    prediction={'next_month': counts[-1] * (1 + velocity/100)},
                    evidence=[f"{len(occurrences)} data points analyzed"]
                ))
        
        # Complexity trends
        if len(self.complexity_history) >= 2:
            complexities = [p.average_complexity for p in self.complexity_history]
            trend_direction = self._calculate_trend_direction(complexities)
            velocity = self._calculate_velocity(complexities)
            
            trends.append(DevelopmentTrend(
                trend_type='complexity',
                metric_name='project_complexity',
                direction=trend_direction,
                velocity=velocity,
                confidence=min(1.0, len(self.complexity_history) / 5),
                prediction={'next_project': complexities[-1] * (1 + velocity/100)},
                evidence=[f"Based on {len(self.complexity_history)} projects"]
            ))
        
        return trends
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 'stable'
        
        slope = numerator / denominator
        
        if slope > 0.1:
            return 'improving'
        elif slope < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_velocity(self, values: List[float]) -> float:
        """Calculate rate of change"""
        if len(values) < 2:
            return 0.0
        
        changes = [values[i] - values[i-1] for i in range(1, len(values))]
        avg_change = statistics.mean(changes) if changes else 0
        
        # Return as percentage
        return (avg_change / (values[0] + 1)) * 100
    
    def _generate_growth_predictions(self, skills: List[SkillMetric],
                                    tech: List[TechnologyAdoption],
                                    complexity: List[ProjectComplexity]) -> Dict[str, Any]:
        """Generate predictions for future growth"""
        predictions = {
            'skill_forecast': {},
            'technology_forecast': {},
            'complexity_forecast': {},
            'timeline': '3_months'
        }
        
        # Skill predictions
        for skill in skills[:5]:  # Top 5 skills
            if skill.growth_rate > 0:
                predicted_level = min(100, skill.proficiency_level * (1 + skill.growth_rate/100))
                predictions['skill_forecast'][skill.skill_name] = {
                    'current': skill.proficiency_level,
                    'predicted': predicted_level,
                    'confidence': skill.confidence
                }
        
        # Technology predictions
        for technology in tech[:5]:  # Top 5 technologies
            if technology.proficiency_trend:
                trend = technology.proficiency_trend
                predicted_usage = trend[-1] * 1.2 if len(trend) > 1 else trend[0]
                predictions['technology_forecast'][technology.technology] = {
                    'current_frequency': technology.usage_frequency,
                    'predicted_frequency': int(predicted_usage),
                    'adoption_stage': self._get_adoption_stage(technology)
                }
        
        # Complexity predictions
        if complexity:
            recent_complexity = [p.average_complexity for p in complexity[-3:]]
            if recent_complexity:
                predictions['complexity_forecast'] = {
                    'current_average': statistics.mean(recent_complexity),
                    'predicted_average': statistics.mean(recent_complexity) * 1.1,
                    'recommendation': 'Consider refactoring if complexity continues to increase'
                }
        
        return predictions
    
    def _get_adoption_stage(self, tech: TechnologyAdoption) -> str:
        """Determine technology adoption stage"""
        if tech.usage_frequency < 5:
            return 'experimental'
        elif tech.usage_frequency < 20:
            return 'learning'
        elif tech.usage_frequency < 50:
            return 'proficient'
        else:
            return 'expert'
    
    def _generate_recommendations(self, trends: List[DevelopmentTrend],
                                 predictions: Dict[str, Any]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Skill recommendations
        improving_skills = [t for t in trends if t.trend_type == 'skill' and t.direction == 'improving']
        declining_skills = [t for t in trends if t.trend_type == 'skill' and t.direction == 'declining']
        
        if improving_skills:
            top_skill = improving_skills[0].metric_name
            recommendations.append(f"Continue focusing on {top_skill} - showing strong improvement")
        
        if declining_skills:
            skill = declining_skills[0].metric_name
            recommendations.append(f"Consider practicing {skill} more - usage declining")
        
        # Complexity recommendations
        complexity_trends = [t for t in trends if t.trend_type == 'complexity']
        if complexity_trends and complexity_trends[0].direction == 'improving':
            recommendations.append("Code complexity increasing - consider refactoring sessions")
        
        # Technology recommendations
        if 'technology_forecast' in predictions:
            emerging_tech = [t for t, p in predictions['technology_forecast'].items()
                           if p['adoption_stage'] == 'learning']
            if emerging_tech:
                recommendations.append(f"Good progress with {emerging_tech[0]} - consider deepening expertise")
        
        # Growth recommendations
        if 'skill_forecast' in predictions:
            high_growth = [(s, p) for s, p in predictions['skill_forecast'].items()
                          if p['predicted'] - p['current'] > 10]
            if high_growth:
                skill, _ = high_growth[0]
                recommendations.append(f"Excellent trajectory with {skill} - on track for mastery")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _identify_achievements(self, skills: List[SkillMetric],
                              tech: List[TechnologyAdoption]) -> List[str]:
        """Identify notable achievements"""
        achievements = []
        
        # Skill achievements
        expert_skills = [s for s in skills if s.proficiency_level >= 80]
        if expert_skills:
            achievements.append(f"Achieved expertise in {expert_skills[0].skill_name}")
        
        high_growth = [s for s in skills if s.growth_rate > 50]
        if high_growth:
            achievements.append(f"Rapid improvement in {high_growth[0].skill_name} (+{high_growth[0].growth_rate:.0f}%)")
        
        # Technology achievements
        adopted_tech = [t for t in tech if t.usage_frequency > 30]
        if adopted_tech:
            achievements.append(f"Successfully adopted {len(adopted_tech)} new technologies")
        
        # Consistency achievements
        consistent_skills = [s for s in skills if s.confidence > 0.8]
        if len(consistent_skills) >= 3:
            achievements.append(f"Maintained consistent practice across {len(consistent_skills)} skill areas")
        
        return achievements
    
    def _count_lines(self, filepath: Path) -> int:
        """Count lines in a file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return len(f.readlines())
        except:
            return 0
    
    def _calculate_average_complexity(self, files: List[Path]) -> float:
        """Calculate average complexity across files"""
        if not self.quality_metrics['complexity']:
            return 0.0
        
        scores = [m['score'] for m in self.quality_metrics['complexity']]
        return statistics.mean(scores) if scores else 0.0
    
    def _count_dependencies(self, project_path: Path) -> int:
        """Count project dependencies"""
        # Check for requirements.txt or package.json
        req_file = project_path / "requirements.txt"
        if req_file.exists():
            with open(req_file, 'r') as f:
                return len([l for l in f.readlines() if l.strip() and not l.startswith('#')])
        return 0
    
    def _assess_pattern_maturity(self, files: List[Path]) -> float:
        """Assess design pattern maturity (0-1)"""
        pattern_indicators = 0
        total_files = len(files)
        
        if total_files == 0:
            return 0.0
        
        for filepath in files[:10]:  # Sample first 10 files
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for pattern indicators
                if '__init__' in content:
                    pattern_indicators += 0.1
                if 'class' in content:
                    pattern_indicators += 0.1
                if '@property' in content:
                    pattern_indicators += 0.2
                if 'def __' in content:  # Magic methods
                    pattern_indicators += 0.2
                if 'super()' in content:
                    pattern_indicators += 0.2
                if 'abc.ABC' in content or 'abstractmethod' in content:
                    pattern_indicators += 0.2
            except:
                continue
        
        return min(1.0, pattern_indicators / min(10, total_files))
    
    def export_trends_to_json(self, report: PersonalGrowthReport, output_file: str):
        """Export trend analysis to JSON"""
        export_data = {
            'analysis_period': {
                'start': report.analysis_period[0].isoformat(),
                'end': report.analysis_period[1].isoformat()
            },
            'skill_progression': [asdict(s) for s in report.skill_progression],
            'technology_adoption': [asdict(t) for t in report.technology_adoption],
            'complexity_evolution': [asdict(c) for c in report.complexity_evolution],
            'development_trends': [asdict(t) for t in report.development_trends],
            'growth_predictions': report.growth_predictions,
            'recommendations': report.recommendations,
            'achievements': report.achievements,
            'metadata': {
                'analyzer': 'PersonalDevelopmentTrendAnalyzer',
                'version': '1.0',
                'generated_at': datetime.now().isoformat()
            }
        }
        
        # Convert datetime objects to strings
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=serialize_datetime)

def main():
    """Run personal development trend analysis"""
    print("=" * 60)
    print("Personal Development Trend Analyzer")
    print("Agent B - Phase 1 Advanced Analysis Engine")
    print("=" * 60)
    
    # Analyze current testmaster directory
    analyzer = PersonalDevelopmentTrendAnalyzer(r"C:\Users\kbass\OneDrive\Documents\testmaster")
    
    # Analyze analyzers directory as a focused example
    projects = [r"C:\Users\kbass\OneDrive\Documents\testmaster\analyzers"]
    
    print("\nAnalyzing development trends...")
    report = analyzer.analyze_development_trends(projects)
    
    # Display results
    print("\nSKILL PROGRESSION")
    print("-" * 40)
    for skill in report.skill_progression[:5]:
        print(f"{skill.skill_name}: Level {skill.proficiency_level:.0f} (Growth: {skill.growth_rate:+.1f}%)")
    
    print("\nTECHNOLOGY ADOPTION")
    print("-" * 40)
    for tech in report.technology_adoption[:5]:
        print(f"{tech.technology}: {tech.usage_frequency} uses across {tech.project_count} projects")
    
    print("\nDEVELOPMENT TRENDS")
    print("-" * 40)
    for trend in report.development_trends[:5]:
        print(f"{trend.metric_name}: {trend.direction} (Velocity: {trend.velocity:+.1f}%)")
    
    print("\nRECOMMENDATIONS")
    print("-" * 40)
    for i, rec in enumerate(report.recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\nACHIEVEMENTS")
    print("-" * 40)
    for achievement in report.achievements:
        print(f"* {achievement}")
    
    # Export to JSON
    output_file = "personal_development_trends.json"
    analyzer.export_trends_to_json(report, output_file)
    print(f"\nTrend analysis exported to: {output_file}")
    
    print("\nPersonal Development Trend Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()