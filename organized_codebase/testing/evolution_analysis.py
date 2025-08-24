"""
Evolution Analysis Module
=========================

Implements comprehensive code evolution analysis:
- Git history analysis and change patterns
- File age and growth pattern analysis  
- Refactoring detection and change hotspots
- Developer pattern analysis
- Temporal coupling and stability metrics
"""

import ast
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
import statistics

from .base_analyzer import BaseAnalyzer


class EvolutionAnalyzer(BaseAnalyzer):
    """Analyzer for code evolution patterns."""
    
    def __init__(self, base_path: Path):
        super().__init__(base_path)
        self.has_git = self._check_git_availability()
    
    def _check_git_availability(self) -> bool:
        """Check if git is available and this is a git repository."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive evolution analysis."""
        print("[INFO] Analyzing Code Evolution...")
        
        if not self.has_git:
            print("  [WARNING] Git not available - using file system analysis only")
            return self._analyze_without_git()
        
        results = {
            "file_ages": self._analyze_file_ages(),
            "growth_patterns": self._analyze_growth_patterns(),
            "change_hotspots": self._identify_change_hotspots(),
            "refactoring_patterns": self._detect_refactoring_patterns(),
            "developer_patterns": self._analyze_developer_patterns(),
            "temporal_coupling": self._analyze_temporal_coupling(),
            "stability_metrics": self._calculate_stability_metrics(),
            "evolution_trends": self._analyze_evolution_trends()
        }
        
        print(f"  [OK] Analyzed {len(results)} evolution categories")
        return results
    
    def _analyze_without_git(self) -> Dict[str, Any]:
        """Fallback analysis without git history."""
        file_stats = {}
        
        for py_file in self._get_python_files():
            try:
                stat = py_file.stat()
                file_key = str(py_file.relative_to(self.base_path))
                
                file_stats[file_key] = {
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'created': datetime.fromtimestamp(stat.st_ctime)
                }
            except Exception:
                continue
        
        # Calculate basic metrics without git
        now = datetime.now()
        ages = [(now - info['created']).days for info in file_stats.values()]
        sizes = [info['size'] for info in file_stats.values()]
        
        return {
            "file_ages": {
                "average_age_days": statistics.mean(ages) if ages else 0,
                "oldest_file_days": max(ages) if ages else 0,
                "newest_file_days": min(ages) if ages else 0,
                "file_count": len(file_stats)
            },
            "growth_patterns": {
                "total_size_bytes": sum(sizes),
                "average_file_size": statistics.mean(sizes) if sizes else 0,
                "size_distribution": self._calculate_size_distribution(sizes)
            },
            "git_available": False,
            "analysis_method": "filesystem_only"
        }
    
    def _analyze_file_ages(self) -> Dict[str, Any]:
        """Analyze file ages using git history."""
        if not self.has_git:
            return {"error": "Git not available"}
        
        file_ages = {}
        creation_dates = []
        
        for py_file in self._get_python_files():
            try:
                file_key = str(py_file.relative_to(self.base_path))
                
                # Get first commit for this file
                cmd = ['git', 'log', '--follow', '--format=%ct', '--', str(py_file)]
                result = subprocess.run(cmd, cwd=self.base_path, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and result.stdout.strip():
                    timestamps = result.stdout.strip().split('\n')
                    if timestamps:
                        # Get the oldest timestamp (last in the list)
                        oldest_timestamp = int(timestamps[-1])
                        creation_date = datetime.fromtimestamp(oldest_timestamp)
                        
                        age_days = (datetime.now() - creation_date).days
                        file_ages[file_key] = {
                            'creation_date': creation_date.isoformat(),
                            'age_days': age_days,
                            'age_category': self._categorize_age(age_days)
                        }
                        creation_dates.append(creation_date)
                        
            except Exception:
                continue
        
        # Calculate summary statistics
        if creation_dates:
            ages = [(datetime.now() - date).days for date in creation_dates]
            
            return {
                "per_file": file_ages,
                "summary": {
                    "total_files": len(file_ages),
                    "average_age_days": statistics.mean(ages),
                    "median_age_days": statistics.median(ages),
                    "oldest_file_days": max(ages),
                    "newest_file_days": min(ages),
                    "age_distribution": self._calculate_age_distribution(ages),
                    "project_start_date": min(creation_dates).isoformat(),
                    "development_duration_days": (datetime.now() - min(creation_dates)).days
                }
            }
        else:
            return {"per_file": {}, "summary": {"total_files": 0}}
    
    def _categorize_age(self, age_days: int) -> str:
        """Categorize file age."""
        if age_days < 30:
            return "very_new"
        elif age_days < 90:
            return "new"
        elif age_days < 365:
            return "mature"
        elif age_days < 730:
            return "old"
        else:
            return "legacy"
    
    def _calculate_age_distribution(self, ages: List[int]) -> Dict[str, int]:
        """Calculate age distribution."""
        distribution = {"very_new": 0, "new": 0, "mature": 0, "old": 0, "legacy": 0}
        
        for age in ages:
            category = self._categorize_age(age)
            distribution[category] += 1
        
        return distribution
    
    def _analyze_growth_patterns(self) -> Dict[str, Any]:
        """Analyze codebase growth patterns over time."""
        if not self.has_git:
            return {"error": "Git not available"}
        
        try:
            # Get commit history with stats
            cmd = ['git', 'log', '--oneline', '--shortstat', '--since=1 year ago']
            result = subprocess.run(cmd, cwd=self.base_path, capture_output=True, text=True, timeout=15)
            
            if result.returncode != 0:
                return {"error": "Failed to get git log"}
            
            commits = result.stdout.strip().split('\n\n')
            growth_data = []
            
            for commit_block in commits:
                if not commit_block.strip():
                    continue
                    
                lines = commit_block.strip().split('\n')
                if len(lines) >= 2:
                    commit_line = lines[0]
                    stats_line = lines[1] if len(lines) > 1 else ""
                    
                    # Parse stats line (e.g., "2 files changed, 15 insertions(+), 3 deletions(-)")
                    insertions = 0
                    deletions = 0
                    files_changed = 0
                    
                    import re
                    files_match = re.search(r'(\d+) files? changed', stats_line)
                    if files_match:
                        files_changed = int(files_match.group(1))
                    
                    insertions_match = re.search(r'(\d+) insertions?', stats_line)
                    if insertions_match:
                        insertions = int(insertions_match.group(1))
                    
                    deletions_match = re.search(r'(\d+) deletions?', stats_line)
                    if deletions_match:
                        deletions = int(deletions_match.group(1))
                    
                    growth_data.append({
                        'files_changed': files_changed,
                        'insertions': insertions,
                        'deletions': deletions,
                        'net_lines': insertions - deletions
                    })
            
            # Calculate growth metrics
            if growth_data:
                total_insertions = sum(item['insertions'] for item in growth_data)
                total_deletions = sum(item['deletions'] for item in growth_data)
                total_net = sum(item['net_lines'] for item in growth_data)
                avg_files_per_commit = statistics.mean([item['files_changed'] for item in growth_data])
                
                return {
                    "commits_analyzed": len(growth_data),
                    "total_insertions": total_insertions,
                    "total_deletions": total_deletions,
                    "net_growth": total_net,
                    "average_files_per_commit": avg_files_per_commit,
                    "growth_rate": total_net / max(len(growth_data), 1),  # Net lines per commit
                    "churn_ratio": total_deletions / max(total_insertions, 1),
                    "commit_activity": len(growth_data),
                    "development_velocity": len(growth_data) / 52  # Commits per week (assuming 1 year)
                }
            else:
                return {"commits_analyzed": 0}
                
        except Exception as e:
            return {"error": f"Growth analysis failed: {str(e)}"}
    
    def _identify_change_hotspots(self) -> List[Dict[str, Any]]:
        """Identify files that change frequently (hotspots)."""
        if not self.has_git:
            return []
        
        hotspots = []
        file_change_counts = defaultdict(int)
        
        try:
            # Get list of changed files in recent commits
            cmd = ['git', 'log', '--name-only', '--since=6 months ago', '--pretty=format:']
            result = subprocess.run(cmd, cwd=self.base_path, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                files = [f.strip() for f in result.stdout.split('\n') if f.strip() and f.strip().endswith('.py')]
                
                for file_path in files:
                    file_change_counts[file_path] += 1
                
                # Sort by change frequency
                sorted_files = sorted(file_change_counts.items(), key=lambda x: x[1], reverse=True)
                
                # Take top hotspots
                for i, (file_path, change_count) in enumerate(sorted_files[:20]):
                    if change_count >= 5:  # Threshold for hotspot
                        hotspots.append({
                            'rank': i + 1,
                            'file': file_path,
                            'change_count': change_count,
                            'hotness_level': self._categorize_hotness(change_count),
                            'risk_level': 'HIGH' if change_count > 20 else 'MEDIUM' if change_count > 10 else 'LOW'
                        })
            
        except Exception:
            pass
        
        return hotspots
    
    def _categorize_hotness(self, change_count: int) -> str:
        """Categorize file hotness level."""
        if change_count > 50:
            return "extremely_hot"
        elif change_count > 20:
            return "very_hot"
        elif change_count > 10:
            return "hot"
        elif change_count > 5:
            return "warm"
        else:
            return "cool"
    
    def _detect_refactoring_patterns(self) -> List[Dict[str, Any]]:
        """Detect refactoring patterns in commit history."""
        if not self.has_git:
            return []
        
        refactoring_patterns = []
        
        try:
            # Get commit messages to identify refactoring
            cmd = ['git', 'log', '--oneline', '--since=6 months ago']
            result = subprocess.run(cmd, cwd=self.base_path, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                commits = result.stdout.strip().split('\n')
                
                refactoring_keywords = [
                    'refactor', 'refactoring', 'cleanup', 'clean up', 'reorganize',
                    'restructure', 'rename', 'move', 'extract', 'simplify',
                    'optimize', 'improve', 'consolidate', 'merge'
                ]
                
                pattern_id = 1
                for commit_line in commits:
                    if not commit_line.strip():
                        continue
                    
                    parts = commit_line.split(' ', 1)
                    if len(parts) >= 2:
                        commit_hash = parts[0]
                        message = parts[1].lower()
                        
                        matched_patterns = [kw for kw in refactoring_keywords if kw in message]
                        
                        if matched_patterns:
                            refactoring_patterns.append({
                                'pattern_id': pattern_id,
                                'commit_hash': commit_hash,
                                'message': parts[1],
                                'matched_keywords': matched_patterns,
                                'refactoring_type': self._classify_refactoring_type(message),
                                'intensity': len(matched_patterns)
                            })
                            pattern_id += 1
            
        except Exception:
            pass
        
        return refactoring_patterns
    
    def _classify_refactoring_type(self, message: str) -> str:
        """Classify the type of refactoring based on commit message."""
        if any(word in message for word in ['rename', 'renaming']):
            return 'rename'
        elif any(word in message for word in ['extract', 'split']):
            return 'extract_method'
        elif any(word in message for word in ['move', 'relocate']):
            return 'move_method'
        elif any(word in message for word in ['merge', 'consolidate', 'combine']):
            return 'merge'
        elif any(word in message for word in ['simplify', 'cleanup', 'clean']):
            return 'simplification'
        elif any(word in message for word in ['optimize', 'performance']):
            return 'optimization'
        else:
            return 'general_refactoring'
    
    def _analyze_developer_patterns(self) -> Dict[str, Any]:
        """Analyze developer contribution patterns."""
        if not self.has_git:
            return {"error": "Git not available"}
        
        try:
            # Get author statistics
            cmd = ['git', 'shortlog', '-sn', '--since=1 year ago']
            result = subprocess.run(cmd, cwd=self.base_path, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return {"error": "Failed to get author statistics"}
            
            authors = []
            total_commits = 0
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        commit_count = int(parts[0])
                        author_name = parts[1]
                        
                        authors.append({
                            'name': author_name,
                            'commits': commit_count
                        })
                        total_commits += commit_count
            
            # Calculate developer metrics
            if authors and total_commits > 0:
                # Sort by contribution
                authors.sort(key=lambda x: x['commits'], reverse=True)
                
                # Calculate ownership distribution
                ownership_distribution = []
                for author in authors:
                    ownership_percentage = (author['commits'] / total_commits) * 100
                    ownership_distribution.append({
                        **author,
                        'ownership_percentage': ownership_percentage
                    })
                
                # Calculate bus factor (number of developers with > 50% of knowledge)
                cumulative_percentage = 0
                bus_factor = 0
                for author in ownership_distribution:
                    cumulative_percentage += author['ownership_percentage']
                    bus_factor += 1
                    if cumulative_percentage >= 50:
                        break
                
                return {
                    "active_developers": len(authors),
                    "total_commits": total_commits,
                    "ownership_distribution": ownership_distribution,
                    "bus_factor": bus_factor,
                    "top_contributor_percentage": ownership_distribution[0]['ownership_percentage'] if authors else 0,
                    "collaboration_score": 1 - (ownership_distribution[0]['ownership_percentage'] / 100) if authors else 0,
                    "development_model": self._classify_development_model(ownership_distribution)
                }
            else:
                return {"active_developers": 0, "total_commits": 0}
                
        except Exception as e:
            return {"error": f"Developer analysis failed: {str(e)}"}
    
    def _classify_development_model(self, ownership_distribution: List[Dict]) -> str:
        """Classify the development model based on ownership distribution."""
        if not ownership_distribution:
            return "unknown"
        
        top_contributor = ownership_distribution[0]['ownership_percentage']
        
        if top_contributor > 80:
            return "single_maintainer"
        elif top_contributor > 60:
            return "lead_developer"
        elif top_contributor > 40:
            return "core_team"
        else:
            return "collaborative"
    
    def _analyze_temporal_coupling(self) -> Dict[str, Any]:
        """Analyze temporal coupling between files."""
        if not self.has_git:
            return {"error": "Git not available"}
        
        coupling_data = defaultdict(lambda: defaultdict(int))
        
        try:
            # Get commits with changed files
            cmd = ['git', 'log', '--name-only', '--pretty=format:%H', '--since=6 months ago']
            result = subprocess.run(cmd, cwd=self.base_path, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                commits = result.stdout.strip().split('\n\n')
                
                for commit_block in commits:
                    if not commit_block.strip():
                        continue
                    
                    lines = commit_block.strip().split('\n')
                    if len(lines) > 1:
                        files = [f.strip() for f in lines[1:] if f.strip().endswith('.py')]
                        
                        # Calculate coupling between all pairs of files in this commit
                        for i in range(len(files)):
                            for j in range(i + 1, len(files)):
                                file1, file2 = sorted([files[i], files[j]])
                                coupling_data[file1][file2] += 1
            
            # Convert to list of coupled files
            coupled_files = []
            for file1, couplings in coupling_data.items():
                for file2, strength in couplings.items():
                    if strength >= 3:  # Threshold for significant coupling
                        coupled_files.append({
                            'file1': file1,
                            'file2': file2,
                            'coupling_strength': strength,
                            'coupling_level': self._categorize_coupling_strength(strength)
                        })
            
            # Sort by coupling strength
            coupled_files.sort(key=lambda x: x['coupling_strength'], reverse=True)
            
            return {
                "coupled_file_pairs": len(coupled_files),
                "strong_couplings": len([c for c in coupled_files if c['coupling_strength'] > 10]),
                "moderate_couplings": len([c for c in coupled_files if 5 <= c['coupling_strength'] <= 10]),
                "weak_couplings": len([c for c in coupled_files if 3 <= c['coupling_strength'] < 5]),
                "top_coupled_pairs": coupled_files[:10],
                "coupling_density": len(coupled_files) / max(len(self._get_python_files()), 1)
            }
            
        except Exception as e:
            return {"error": f"Temporal coupling analysis failed: {str(e)}"}
    
    def _categorize_coupling_strength(self, strength: int) -> str:
        """Categorize temporal coupling strength."""
        if strength > 15:
            return "very_strong"
        elif strength > 10:
            return "strong"
        elif strength > 5:
            return "moderate"
        else:
            return "weak"
    
    def _calculate_stability_metrics(self) -> Dict[str, Any]:
        """Calculate code stability metrics."""
        if not self.has_git:
            return {"error": "Git not available"}
        
        try:
            # Get recent change frequency for each file
            file_changes = defaultdict(int)
            
            cmd = ['git', 'log', '--name-only', '--since=3 months ago', '--pretty=format:']
            result = subprocess.run(cmd, cwd=self.base_path, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                files = [f.strip() for f in result.stdout.split('\n') if f.strip().endswith('.py')]
                
                for file_path in files:
                    file_changes[file_path] += 1
            
            # Calculate stability metrics
            all_python_files = [str(f.relative_to(self.base_path)) for f in self._get_python_files()]
            total_files = len(all_python_files)
            
            if total_files == 0:
                return {"total_files": 0}
            
            stable_files = len([f for f in all_python_files if file_changes[f] <= 2])
            volatile_files = len([f for f in all_python_files if file_changes[f] > 10])
            moderate_files = total_files - stable_files - volatile_files
            
            stability_index = stable_files / total_files
            volatility_index = volatile_files / total_files
            
            return {
                "total_files": total_files,
                "stable_files": stable_files,
                "moderate_files": moderate_files,
                "volatile_files": volatile_files,
                "stability_index": stability_index,
                "volatility_index": volatility_index,
                "stability_grade": self._grade_stability(stability_index),
                "most_volatile_files": [
                    {"file": file, "changes": changes} 
                    for file, changes in sorted(file_changes.items(), key=lambda x: x[1], reverse=True)[:5]
                    if changes > 5
                ]
            }
            
        except Exception as e:
            return {"error": f"Stability analysis failed: {str(e)}"}
    
    def _grade_stability(self, stability_index: float) -> str:
        """Grade stability based on index."""
        if stability_index > 0.8:
            return "A"
        elif stability_index > 0.6:
            return "B"
        elif stability_index > 0.4:
            return "C"
        elif stability_index > 0.2:
            return "D"
        else:
            return "F"
    
    def _analyze_evolution_trends(self) -> Dict[str, Any]:
        """Analyze overall evolution trends."""
        if not self.has_git:
            return {"error": "Git not available"}
        
        try:
            # Get commit frequency over time
            cmd = ['git', 'rev-list', '--count', '--since=1 year ago', 'HEAD']
            result = subprocess.run(cmd, cwd=self.base_path, capture_output=True, text=True, timeout=10)
            
            commits_last_year = 0
            if result.returncode == 0 and result.stdout.strip():
                commits_last_year = int(result.stdout.strip())
            
            # Get commit frequency over last 6 months
            cmd = ['git', 'rev-list', '--count', '--since=6 months ago', 'HEAD']
            result = subprocess.run(cmd, cwd=self.base_path, capture_output=True, text=True, timeout=10)
            
            commits_last_six_months = 0
            if result.returncode == 0 and result.stdout.strip():
                commits_last_six_months = int(result.stdout.strip())
            
            # Calculate trend
            annual_rate = commits_last_year / 12  # Commits per month
            recent_rate = commits_last_six_months / 6  # Commits per month
            
            trend_direction = "stable"
            if recent_rate > annual_rate * 1.2:
                trend_direction = "increasing"
            elif recent_rate < annual_rate * 0.8:
                trend_direction = "decreasing"
            
            # Get repository age
            cmd = ['git', 'log', '--reverse', '--format=%ct', '-1']
            result = subprocess.run(cmd, cwd=self.base_path, capture_output=True, text=True, timeout=10)
            
            repo_age_days = 0
            if result.returncode == 0 and result.stdout.strip():
                first_commit_timestamp = int(result.stdout.strip())
                first_commit_date = datetime.fromtimestamp(first_commit_timestamp)
                repo_age_days = (datetime.now() - first_commit_date).days
            
            return {
                "commits_last_year": commits_last_year,
                "commits_last_six_months": commits_last_six_months,
                "annual_commit_rate": annual_rate,
                "recent_commit_rate": recent_rate,
                "trend_direction": trend_direction,
                "trend_strength": abs(recent_rate - annual_rate) / max(annual_rate, 1),
                "repository_age_days": repo_age_days,
                "development_maturity": self._assess_maturity(repo_age_days, commits_last_year),
                "activity_level": self._classify_activity_level(recent_rate)
            }
            
        except Exception as e:
            return {"error": f"Evolution trends analysis failed: {str(e)}"}
    
    def _assess_maturity(self, age_days: int, commits_per_year: int) -> str:
        """Assess repository maturity."""
        if age_days > 1095 and commits_per_year > 100:  # > 3 years, active
            return "mature_active"
        elif age_days > 1095:  # > 3 years
            return "mature"
        elif age_days > 365 and commits_per_year > 50:  # > 1 year, active
            return "developing"
        elif age_days > 365:  # > 1 year
            return "established"
        else:
            return "young"
    
    def _classify_activity_level(self, commits_per_month: float) -> str:
        """Classify activity level."""
        if commits_per_month > 20:
            return "very_active"
        elif commits_per_month > 10:
            return "active"
        elif commits_per_month > 5:
            return "moderate"
        elif commits_per_month > 1:
            return "low"
        else:
            return "inactive"
    
    def _calculate_size_distribution(self, sizes: List[int]) -> Dict[str, int]:
        """Calculate file size distribution."""
        distribution = {
            "small": 0,     # < 1KB
            "medium": 0,    # 1KB - 10KB
            "large": 0,     # 10KB - 100KB
            "very_large": 0 # > 100KB
        }
        
        for size in sizes:
            if size < 1024:
                distribution["small"] += 1
            elif size < 10240:
                distribution["medium"] += 1
            elif size < 102400:
                distribution["large"] += 1
            else:
                distribution["very_large"] += 1
        
        return distribution