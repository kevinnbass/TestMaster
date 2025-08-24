#!/usr/bin/env python3
"""
Markdown Content Consolidation Tool - Agent C Hours 54-56
Analyzes and consolidates markdown files, documentation, and README files.
Identifies content overlap, consolidation opportunities, and organization improvements.
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any, Optional, Tuple
import hashlib

class MarkdownConsolidator:
    """Analyzes and consolidates markdown content across the codebase."""
    
    def __init__(self):
        self.markdown_files = []
        self.content_analysis = {}
        self.duplicate_content = []
        self.topic_clusters = defaultdict(list)
        self.consolidation_opportunities = []
        self.documentation_structure = {}
        
    def analyze_directory(self, root_path: Path) -> Dict[str, Any]:
        """Analyze directory for markdown files and documentation."""
        results = {
            'files_analyzed': 0,
            'markdown_files': 0,
            'total_size': 0,
            'content_types': defaultdict(int),
            'duplicate_sections': 0
        }
        
        print("=== Agent C Hours 54-56: Markdown Content Consolidation ===")
        print(f"Scanning directory: {root_path}")
        
        # Find all markdown and documentation files
        markdown_patterns = ['*.md', '*.markdown', '*.txt', '*.rst']
        all_files = []
        
        for pattern in markdown_patterns:
            all_files.extend(root_path.rglob(pattern))
        
        # Filter for documentation-like files
        doc_files = self._filter_documentation_files(all_files)
        
        print(f"Found {len(doc_files)} documentation files")
        
        for file_path in doc_files:
            if file_path.is_file():
                results['files_analyzed'] += 1
                
                analysis = self._analyze_markdown_file(file_path)
                if analysis:
                    self.markdown_files.append(analysis)
                    results['markdown_files'] += 1
                    results['total_size'] += analysis['size']
                    results['content_types'][analysis['content_type']] += 1
        
        print(f"Analysis complete:")
        print(f"  Markdown files: {results['markdown_files']}")
        print(f"  Total size: {self._format_size(results['total_size'])}")
        print(f"  Content types: {dict(results['content_types'])}")
        
        # Analyze for duplicates and consolidation opportunities
        self._detect_duplicate_content()
        self._cluster_by_topics()
        self._identify_consolidation_opportunities()
        
        results['duplicate_sections'] = len(self.duplicate_content)
        
        return results
    
    def _filter_documentation_files(self, files: List[Path]) -> List[Path]:
        """Filter files to include only documentation-related files."""
        doc_files = []
        
        for file_path in files:
            filename = file_path.name.lower()
            path_str = str(file_path).lower()
            
            # Include documentation files
            doc_indicators = [
                'readme', 'documentation', 'docs', 'manual', 'guide',
                'tutorial', 'instructions', 'notes', 'changelog',
                'history', 'todo', 'roadmap', 'architecture', 'design'
            ]
            
            # Exclude non-documentation files
            exclude_patterns = [
                'node_modules', '.git', '__pycache__', '.pytest_cache',
                'dist', 'build', '.venv', 'venv', '.tox'
            ]
            
            # Check if file should be excluded
            if any(pattern in path_str for pattern in exclude_patterns):
                continue
            
            # Check if file is documentation
            if (file_path.suffix.lower() in ['.md', '.markdown'] or
                any(indicator in filename for indicator in doc_indicators) or
                any(indicator in path_str for indicator in ['doc', 'readme'])):
                doc_files.append(file_path)
        
        return doc_files
    
    def _analyze_markdown_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a single markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            analysis = {
                'path': str(file_path),
                'name': file_path.name,
                'size': len(content.encode('utf-8')),
                'line_count': len(content.split('\n')),
                'content_type': self._determine_content_type(file_path, content),
                'content': content,
                'sections': self._extract_sections(content),
                'headers': self._extract_headers(content),
                'links': self._extract_links(content),
                'images': self._extract_images(content),
                'code_blocks': self._extract_code_blocks(content),
                'tables': self._extract_tables(content),
                'word_count': len(content.split()),
                'topics': self._extract_topics(content),
                'quality_score': self._assess_content_quality(content),
                'last_modified': self._get_last_modified(file_path)
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    def _determine_content_type(self, file_path: Path, content: str) -> str:
        """Determine the type of markdown content."""
        filename = file_path.name.lower()
        content_lower = content.lower()
        
        if filename.startswith('readme'):
            return 'readme'
        elif filename in ['changelog.md', 'history.md', 'changes.md']:
            return 'changelog'
        elif filename in ['todo.md', 'todos.md']:
            return 'todo'
        elif filename in ['architecture.md', 'design.md']:
            return 'architecture'
        elif any(word in filename for word in ['api', 'reference']):
            return 'api_documentation'
        elif any(word in filename for word in ['tutorial', 'guide', 'howto']):
            return 'tutorial'
        elif any(word in filename for word in ['install', 'setup']):
            return 'installation'
        elif any(word in content_lower for word in ['installation', 'getting started']):
            return 'getting_started'
        elif '# api' in content_lower or '## api' in content_lower:
            return 'api_documentation'
        elif len(re.findall(r'```', content)) > 4:  # Lots of code blocks
            return 'technical_documentation'
        else:
            return 'general_documentation'
    
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections from markdown content."""
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)', line)
            
            if header_match:
                # Save previous section
                if current_section:
                    sections.append({
                        'level': current_section['level'],
                        'title': current_section['title'],
                        'content': '\n'.join(current_content),
                        'hash': hashlib.md5('\n'.join(current_content).encode()).hexdigest()
                    })
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = {'level': level, 'title': title}
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Add final section
        if current_section:
            sections.append({
                'level': current_section['level'],
                'title': current_section['title'],
                'content': '\n'.join(current_content),
                'hash': hashlib.md5('\n'.join(current_content).encode()).hexdigest()
            })
        
        return sections
    
    def _extract_headers(self, content: str) -> List[Dict[str, Any]]:
        """Extract headers from markdown content."""
        headers = []
        for match in re.finditer(r'^(#{1,6})\s+(.+)', content, re.MULTILINE):
            headers.append({
                'level': len(match.group(1)),
                'text': match.group(2).strip(),
                'line': content[:match.start()].count('\n') + 1
            })
        return headers
    
    def _extract_links(self, content: str) -> List[str]:
        """Extract links from markdown content."""
        # Match both [text](url) and [text][ref] patterns
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        return [match.group(2) for match in re.finditer(link_pattern, content)]
    
    def _extract_images(self, content: str) -> List[str]:
        """Extract image references from markdown content."""
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        return [match.group(2) for match in re.finditer(image_pattern, content)]
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks from markdown content."""
        code_blocks = []
        
        # Fenced code blocks
        fenced_pattern = r'```(\w+)?\n(.*?)\n```'
        for match in re.finditer(fenced_pattern, content, re.DOTALL):
            code_blocks.append({
                'language': match.group(1) or 'unknown',
                'code': match.group(2),
                'type': 'fenced'
            })
        
        # Indented code blocks (4+ spaces)
        indented_pattern = r'^(    .+)(?:\n    .+)*'
        for match in re.finditer(indented_pattern, content, re.MULTILINE):
            code_blocks.append({
                'language': 'unknown',
                'code': match.group(0),
                'type': 'indented'
            })
        
        return code_blocks
    
    def _extract_tables(self, content: str) -> int:
        """Count tables in markdown content."""
        # Simple table detection (lines with | characters)
        table_lines = [line for line in content.split('\n') if '|' in line and line.strip()]
        
        # Group consecutive table lines
        table_count = 0
        in_table = False
        
        for line in content.split('\n'):
            if '|' in line and line.strip():
                if not in_table:
                    table_count += 1
                    in_table = True
            else:
                in_table = False
        
        return table_count
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract main topics from content."""
        topics = []
        
        # Extract from headers
        headers = self._extract_headers(content)
        for header in headers:
            if header['level'] <= 2:  # Only major headers
                topics.append(header['text'].lower())
        
        # Extract common technical topics
        content_lower = content.lower()
        technical_topics = [
            'installation', 'configuration', 'api', 'tutorial', 'examples',
            'testing', 'deployment', 'architecture', 'security', 'performance',
            'troubleshooting', 'changelog', 'contributing', 'license'
        ]
        
        for topic in technical_topics:
            if topic in content_lower:
                topics.append(topic)
        
        return list(set(topics))  # Remove duplicates
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess the quality of markdown content."""
        score = 0.0
        
        # Length indicators
        word_count = len(content.split())
        if 100 <= word_count <= 5000:
            score += 0.2
        elif word_count > 5000:
            score += 0.1
        
        # Structure indicators
        headers = len(re.findall(r'^#{1,6}\s+', content, re.MULTILINE))
        if headers >= 2:
            score += 0.2
        
        # Code examples
        code_blocks = len(re.findall(r'```', content)) // 2
        if code_blocks > 0:
            score += 0.2
        
        # Links and references
        links = len(re.findall(r'\[([^\]]+)\]\([^)]+\)', content))
        if links > 0:
            score += 0.1
        
        # Table of contents or organized structure
        if 'table of contents' in content.lower() or '## contents' in content.lower():
            score += 0.1
        
        # Completeness indicators
        if any(word in content.lower() for word in ['example', 'usage', 'how to']):
            score += 0.1
        
        # Deduct for issues
        if len(content.split()) < 50:  # Too short
            score -= 0.2
        
        if content.count('TODO') + content.count('FIXME') > 3:  # Too many TODOs
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _get_last_modified(self, file_path: Path) -> str:
        """Get last modified timestamp."""
        try:
            timestamp = file_path.stat().st_mtime
            from datetime import datetime
            return datetime.fromtimestamp(timestamp).isoformat()
        except:
            return 'unknown'
    
    def _detect_duplicate_content(self):
        """Detect duplicate content across markdown files."""
        print("\nDetecting duplicate content...")
        
        section_hashes = defaultdict(list)
        
        # Group sections by content hash
        for file_info in self.markdown_files:
            for section in file_info['sections']:
                content_hash = section['hash']
                section_hashes[content_hash].append({
                    'file': file_info['path'],
                    'title': section['title'],
                    'content': section['content']
                })
        
        # Find duplicates
        for content_hash, sections in section_hashes.items():
            if len(sections) > 1:
                # Only consider substantial content (not empty sections)
                if len(sections[0]['content'].strip()) > 100:
                    self.duplicate_content.append({
                        'hash': content_hash,
                        'sections': sections,
                        'duplicate_count': len(sections),
                        'content_preview': sections[0]['content'][:200] + '...'
                    })
        
        print(f"Found {len(self.duplicate_content)} duplicate content sections")
    
    def _cluster_by_topics(self):
        """Cluster markdown files by topics."""
        print("Clustering files by topics...")
        
        for file_info in self.markdown_files:
            for topic in file_info['topics']:
                self.topic_clusters[topic].append({
                    'file': file_info['path'],
                    'name': file_info['name'],
                    'content_type': file_info['content_type'],
                    'quality_score': file_info['quality_score']
                })
        
        # Sort clusters by file count
        sorted_clusters = sorted(self.topic_clusters.items(), 
                               key=lambda x: len(x[1]), reverse=True)
        
        print(f"Identified {len(sorted_clusters)} topic clusters")
        for topic, files in sorted_clusters[:10]:  # Top 10
            print(f"  {topic}: {len(files)} files")
    
    def _identify_consolidation_opportunities(self):
        """Identify opportunities for content consolidation."""
        print("Identifying consolidation opportunities...")
        
        # Multiple README files
        readme_files = [f for f in self.markdown_files if f['content_type'] == 'readme']
        if len(readme_files) > 1:
            self.consolidation_opportunities.append({
                'type': 'multiple_readmes',
                'description': f"Found {len(readme_files)} README files",
                'files': [f['path'] for f in readme_files],
                'recommendation': 'Consolidate into single main README',
                'priority': 'high'
            })
        
        # Duplicate API documentation
        api_docs = [f for f in self.markdown_files if f['content_type'] == 'api_documentation']
        if len(api_docs) > 2:
            self.consolidation_opportunities.append({
                'type': 'scattered_api_docs',
                'description': f"Found {len(api_docs)} API documentation files",
                'files': [f['path'] for f in api_docs],
                'recommendation': 'Consolidate API documentation',
                'priority': 'medium'
            })
        
        # Topic clusters with many files
        for topic, files in self.topic_clusters.items():
            if len(files) > 5:
                self.consolidation_opportunities.append({
                    'type': 'topic_scatter',
                    'description': f"Topic '{topic}' appears in {len(files)} files",
                    'files': [f['file'] for f in files],
                    'recommendation': f'Consolidate {topic} documentation',
                    'priority': 'low' if len(files) < 10 else 'medium'
                })
        
        # Low quality documentation
        low_quality = [f for f in self.markdown_files if f['quality_score'] < 0.3]
        if len(low_quality) > 0:
            self.consolidation_opportunities.append({
                'type': 'low_quality_docs',
                'description': f"Found {len(low_quality)} low quality documentation files",
                'files': [f['path'] for f in low_quality],
                'recommendation': 'Review and improve or consolidate low quality docs',
                'priority': 'medium'
            })
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def generate_consolidation_strategy(self) -> Dict[str, Any]:
        """Generate comprehensive consolidation strategy."""
        print("\nGenerating consolidation strategy...")
        
        strategy = {
            'documentation_audit': self._create_documentation_audit(),
            'consolidation_plan': self._create_consolidation_plan(),
            'organization_structure': self._design_documentation_structure(),
            'quality_improvements': self._identify_quality_improvements(),
            'automation_recommendations': self._recommend_automation(),
            'implementation_timeline': self._create_implementation_timeline()
        }
        
        return strategy
    
    def _create_documentation_audit(self) -> Dict[str, Any]:
        """Create documentation audit summary."""
        content_types = defaultdict(int)
        quality_distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for file_info in self.markdown_files:
            content_types[file_info['content_type']] += 1
            
            if file_info['quality_score'] >= 0.7:
                quality_distribution['high'] += 1
            elif file_info['quality_score'] >= 0.4:
                quality_distribution['medium'] += 1
            else:
                quality_distribution['low'] += 1
        
        return {
            'total_files': len(self.markdown_files),
            'content_types': dict(content_types),
            'quality_distribution': dict(quality_distribution),
            'duplicate_sections': len(self.duplicate_content),
            'consolidation_opportunities': len(self.consolidation_opportunities),
            'average_quality_score': sum(f['quality_score'] for f in self.markdown_files) / len(self.markdown_files) if self.markdown_files else 0
        }
    
    def _create_consolidation_plan(self) -> List[Dict[str, Any]]:
        """Create detailed consolidation plan."""
        plan = []
        
        # Sort opportunities by priority
        high_priority = [op for op in self.consolidation_opportunities if op['priority'] == 'high']
        medium_priority = [op for op in self.consolidation_opportunities if op['priority'] == 'medium']
        low_priority = [op for op in self.consolidation_opportunities if op['priority'] == 'low']
        
        for opportunities in [high_priority, medium_priority, low_priority]:
            for opportunity in opportunities:
                plan.append({
                    'type': opportunity['type'],
                    'description': opportunity['description'],
                    'action': opportunity['recommendation'],
                    'priority': opportunity['priority'],
                    'affected_files': len(opportunity['files']),
                    'estimated_effort': self._estimate_consolidation_effort(opportunity)
                })
        
        return plan
    
    def _design_documentation_structure(self) -> Dict[str, Any]:
        """Design improved documentation structure."""
        return {
            'root_docs': {
                'README.md': 'Main project overview and quick start',
                'CHANGELOG.md': 'Version history and changes',
                'CONTRIBUTING.md': 'Contribution guidelines',
                'LICENSE.md': 'License information'
            },
            'docs_directory': {
                'getting-started/': 'Installation and basic setup',
                'user-guide/': 'User documentation and tutorials',
                'api-reference/': 'API documentation',
                'developer-guide/': 'Development and architecture docs',
                'examples/': 'Code examples and demos',
                'troubleshooting/': 'Common issues and solutions'
            },
            'consolidation_rules': [
                'Single README.md at project root',
                'API documentation in dedicated api-reference/ directory',
                'Tutorials and guides in user-guide/ directory',
                'Architecture and design docs in developer-guide/ directory',
                'Remove duplicate content and cross-reference instead'
            ]
        }
    
    def _identify_quality_improvements(self) -> List[Dict[str, Any]]:
        """Identify quality improvement opportunities."""
        improvements = []
        
        # Files with low quality scores
        low_quality = [f for f in self.markdown_files if f['quality_score'] < 0.4]
        if low_quality:
            improvements.append({
                'type': 'quality_improvement',
                'description': f"{len(low_quality)} files need quality improvements",
                'action': 'Review structure, add examples, improve formatting',
                'files': [f['path'] for f in low_quality]
            })
        
        # Files without code examples
        no_examples = [f for f in self.markdown_files 
                      if len(f['code_blocks']) == 0 and f['content_type'] in ['tutorial', 'api_documentation']]
        if no_examples:
            improvements.append({
                'type': 'add_examples',
                'description': f"{len(no_examples)} documentation files lack code examples",
                'action': 'Add relevant code examples and usage demonstrations',
                'files': [f['path'] for f in no_examples]
            })
        
        # Files with broken links (would need network check)
        improvements.append({
            'type': 'link_validation',
            'description': 'Validate all external and internal links',
            'action': 'Run link checker and fix broken references',
            'files': 'all_with_links'
        })
        
        return improvements
    
    def _recommend_automation(self) -> List[Dict[str, Any]]:
        """Recommend automation for documentation maintenance."""
        return [
            {
                'automation': 'link_checker',
                'description': 'Automated link validation in CI/CD',
                'impact': 'high',
                'implementation': 'GitHub Actions or similar'
            },
            {
                'automation': 'documentation_linting',
                'description': 'Markdown linting and style checking',
                'impact': 'medium',
                'implementation': 'markdownlint or similar tools'
            },
            {
                'automation': 'duplicate_detection',
                'description': 'Automated duplicate content detection',
                'impact': 'medium',
                'implementation': 'Custom script or existing tools'
            },
            {
                'automation': 'api_doc_generation',
                'description': 'Automated API documentation from code',
                'impact': 'high',
                'implementation': 'Sphinx, JSDoc, or similar'
            }
        ]
    
    def _create_implementation_timeline(self) -> List[Dict[str, Any]]:
        """Create implementation timeline for consolidation."""
        return [
            {
                'phase': 'Phase 1: Critical Consolidation',
                'duration': '1-2 weeks',
                'actions': [
                    'Consolidate multiple README files',
                    'Remove obvious duplicate content',
                    'Fix critical documentation gaps'
                ]
            },
            {
                'phase': 'Phase 2: Structure Organization',
                'duration': '2-3 weeks',
                'actions': [
                    'Implement new documentation structure',
                    'Move files to appropriate directories',
                    'Update cross-references and links'
                ]
            },
            {
                'phase': 'Phase 3: Quality Enhancement',
                'duration': '1-2 weeks',
                'actions': [
                    'Improve low-quality documentation',
                    'Add missing code examples',
                    'Validate and fix links'
                ]
            },
            {
                'phase': 'Phase 4: Automation Setup',
                'duration': '1 week',
                'actions': [
                    'Set up documentation linting',
                    'Implement link checking',
                    'Create maintenance procedures'
                ]
            }
        ]
    
    def _estimate_consolidation_effort(self, opportunity: Dict[str, Any]) -> str:
        """Estimate effort required for consolidation."""
        file_count = len(opportunity['files'])
        
        if opportunity['type'] == 'multiple_readmes':
            return 'medium'  # Requires careful content merging
        elif opportunity['type'] == 'low_quality_docs':
            return 'high'  # Requires content rewriting
        elif file_count < 5:
            return 'low'
        elif file_count < 15:
            return 'medium'
        else:
            return 'high'
    
    def generate_summary(self, analysis_results: Dict[str, Any], consolidation_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary."""
        return {
            'analysis_metadata': {
                'tool': 'markdown_consolidator',
                'version': '1.0',
                'agent': 'Agent_C',
                'hours': '54-56',
                'phase': 'Debug_Markdown_Stowage'
            },
            'analysis_results': analysis_results,
            'consolidation_strategy': consolidation_strategy,
            'summary_statistics': {
                'total_markdown_files': len(self.markdown_files),
                'total_content_size': analysis_results['total_size'],
                'duplicate_content_sections': len(self.duplicate_content),
                'topic_clusters': len(self.topic_clusters),
                'consolidation_opportunities': len(self.consolidation_opportunities),
                'average_quality_score': consolidation_strategy['documentation_audit']['average_quality_score']
            },
            'recommendations': self._generate_final_recommendations(consolidation_strategy),
            'implementation_readiness': self._assess_implementation_readiness(consolidation_strategy)
        }
    
    def _generate_final_recommendations(self, strategy: Dict[str, Any]) -> List[str]:
        """Generate final recommendations."""
        recommendations = []
        
        audit = strategy['documentation_audit']
        
        if audit['duplicate_sections'] > 5:
            recommendations.append("High duplicate content - prioritize consolidation")
        
        if audit['quality_distribution']['low'] > audit['quality_distribution']['high']:
            recommendations.append("Focus on improving documentation quality")
        
        if audit['consolidation_opportunities'] > 10:
            recommendations.append("Implement systematic consolidation strategy")
        
        if len(self.markdown_files) > 50:
            recommendations.append("Consider automated documentation management tools")
        
        return recommendations
    
    def _assess_implementation_readiness(self, strategy: Dict[str, Any]) -> str:
        """Assess readiness for implementation."""
        high_priority_ops = len([op for op in self.consolidation_opportunities if op['priority'] == 'high'])
        
        if high_priority_ops > 3:
            return 'needs_planning'
        elif high_priority_ops > 0:
            return 'ready_with_prioritization'
        else:
            return 'ready'


def main():
    parser = argparse.ArgumentParser(description='Markdown Content Consolidation Tool')
    parser.add_argument('--root', type=str, required=True, help='Root directory to analyze')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    consolidator = MarkdownConsolidator()
    root_path = Path(args.root)
    
    # Analyze directory
    analysis_results = consolidator.analyze_directory(root_path)
    
    # Generate consolidation strategy
    consolidation_strategy = consolidator.generate_consolidation_strategy()
    
    # Generate summary
    summary = consolidator.generate_summary(analysis_results, consolidation_strategy)
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n=== MARKDOWN CONSOLIDATION COMPLETE ===")
    print(f"Markdown files analyzed: {analysis_results['markdown_files']}")
    print(f"Total content size: {consolidator._format_size(analysis_results['total_size'])}")
    print(f"Duplicate sections found: {len(consolidator.duplicate_content)}")
    print(f"Topic clusters: {len(consolidator.topic_clusters)}")
    print(f"Consolidation opportunities: {len(consolidator.consolidation_opportunities)}")
    print(f"Implementation readiness: {summary['implementation_readiness']}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()