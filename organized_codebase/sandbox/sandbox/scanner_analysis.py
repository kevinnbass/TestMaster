#!/usr/bin/env python3
"""
LLM Intelligence Scanner Analysis Module
========================================

Core analysis functionality for the LLM intelligence scanner.
Handles prompt generation, response parsing, and analysis logic.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from scanner_models import LLMIntelligenceEntry, LLMProvider


class LLMAnalysisEngine:
    """Handles LLM-based code analysis"""

    def __init__(self, llm_client, logger):
        """Initialize the analysis engine"""
        self.llm_client = llm_client
        self.logger = logger

    def analyze_file_with_llm(self, file_path: Path, root_dir: Path) -> Optional[LLMIntelligenceEntry]:
        """Analyze a single file with LLM"""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Generate analysis prompt
            prompt = self._generate_analysis_prompt(content, file_path, root_dir)

            # Get LLM analysis
            llm_response = self.llm_client.analyze_code(prompt)

            # Parse response
            parsed_data = self._parse_llm_response(llm_response)

            # Create intelligence entry
            return self._create_intelligence_entry(file_path, root_dir, content, parsed_data)

        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return None

    def _generate_analysis_prompt(self, content: str, file_path: Path, root_dir: Path) -> str:
        """Generate a comprehensive analysis prompt for the LLM"""
        relative_path = file_path.relative_to(root_dir)

        prompt = f"""
You are an expert Python code analyst tasked with deeply understanding a Python module for reorganization purposes.

FILE TO ANALYZE:
- Full path: {file_path}
- Relative path: {relative_path}
- File size: {len(content.encode('utf-8'))} bytes
- Lines of code: {len(content.split('\n'))}

CODE CONTENT:
```
{content[:4000]}...  # Truncated for brevity
```

Please provide a comprehensive analysis in the following JSON format:

{{
    "summary": "A concise but exhaustive 2-3 sentence description of what this module does and its primary purpose",
    "functionality": "Detailed breakdown of the module's functionality, key classes, functions, and their purposes",
    "dependencies": "Analysis of imports, external dependencies, and coupling relationships",
    "security": "Security implications, authentication, encryption, or sensitive data handling",
    "testing": "Testing requirements, mock objects needed, or test coverage considerations",
    "architecture": "Architectural role - is this a service, utility, model, controller, etc.?",
    "primary_classification": "Primary category: security/intelligence/frontend_dashboard/documentation/testing/utility/api/database/automation/orchestration",
    "secondary_classifications": ["List", "of", "secondary", "categories"],
    "reorganization": ["Specific reorganization recommendations", "where this should be moved", "related modules it should be grouped with"],
    "confidence": 0.0-1.0,
    "key_features": ["List", "of", "key", "functionality", "features"],
    "integration_points": ["How", "this", "integrates", "with", "other", "modules"],
    "complexity": "Assessment of code complexity and maintainability challenges",
    "maintainability": "Notes on maintainability, potential refactoring needs, or code quality issues"
}}

Focus on:
- Deep semantic understanding of what the code DOES, not just what it CONTAINS
- How this module fits into the larger system architecture
- Security and operational considerations
- How this should be reorganized for better cohesion and coupling
- Practical insights for developers who need to understand and maintain this code

Be specific and actionable in your analysis. If you're uncertain about any aspect, note that in your confidence score.
"""

        return prompt

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data"""
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback: create structured data from text response
                return self._fallback_parse_response(response)

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return self._fallback_parse_response(response)

    def _fallback_parse_response(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for non-JSON LLM responses"""
        # Extract what we can from the text response
        return {
            'summary': response[:500] + '...' if len(response) > 500 else response,
            'functionality': 'Analysis provided in summary',
            'dependencies': 'Dependencies mentioned in analysis',
            'security': 'Security aspects mentioned in analysis',
            'testing': 'Testing aspects mentioned in analysis',
            'architecture': 'Architecture mentioned in analysis',
            'primary_classification': 'uncategorized',
            'secondary_classifications': [],
            'reorganization': ['Manual review required'],
            'confidence': 0.3,
            'key_features': [],
            'integration_points': [],
            'complexity': 'Unknown complexity',
            'maintainability': 'Manual review required'
        }

    def _create_intelligence_entry(self, file_path: Path, root_dir: Path, content: str,
                                 parsed_data: Dict[str, Any]) -> LLMIntelligenceEntry:
        """Create an intelligence entry from parsed data"""
        # Calculate file statistics
        lines = content.split('\n')
        file_stats = self._calculate_file_statistics(content)

        # Create entry
        return LLMIntelligenceEntry(
            full_path=str(file_path),
            relative_path=str(file_path.relative_to(root_dir)),
            file_hash=self._calculate_file_hash(file_path),
            analysis_timestamp=datetime.now().isoformat(),
            module_summary=parsed_data.get('summary', 'No summary available'),
            functionality_details=parsed_data.get('functionality', 'No functionality details'),
            dependencies_analysis=parsed_data.get('dependencies', 'No dependency analysis'),
            security_implications=parsed_data.get('security', 'No security analysis'),
            testing_requirements=parsed_data.get('testing', 'No testing analysis'),
            architectural_role=parsed_data.get('architecture', 'No architectural role'),
            primary_classification=parsed_data.get('primary_classification', 'uncategorized'),
            secondary_classifications=parsed_data.get('secondary_classifications', []),
            reorganization_recommendations=parsed_data.get('reorganization', []),
            confidence_score=float(parsed_data.get('confidence', 0.5)),
            key_features=parsed_data.get('key_features', []),
            integration_points=parsed_data.get('integration_points', []),
            complexity_assessment=parsed_data.get('complexity', 'Unknown'),
            maintainability_notes=parsed_data.get('maintainability', 'No notes'),
            file_size=len(content.encode('utf-8')),
            line_count=len(lines),
            class_count=file_stats['class_count'],
            function_count=file_stats['function_count']
        )

    def _calculate_file_statistics(self, content: str) -> Dict[str, int]:
        """Calculate basic file statistics"""
        import re

        class_count = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))
        function_count = len(re.findall(r'^def\s+\w+', content, re.MULTILINE))

        return {
            'class_count': class_count,
            'function_count': function_count
        }

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate file hash for change detection"""
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class InsightsGenerator:
    """Generates reorganization insights from intelligence entries"""

    def create_classification_summary(self, entries: List[LLMIntelligenceEntry]) -> Dict[str, int]:
        """Create a summary of classifications across all entries"""
        summary = {}
        for entry in entries:
            primary = entry.primary_classification
            summary[primary] = summary.get(primary, 0) + 1

            for secondary in entry.secondary_classifications:
                summary[secondary] = summary.get(secondary, 0) + 1

        return summary

    def generate_reorganization_insights(self, entries: List[LLMIntelligenceEntry],
                                       classification_summary: Dict[str, int]) -> Dict[str, Any]:
        """Generate high-level reorganization insights"""
        insights = {
            'classification_distribution': classification_summary,
            'recommended_structure': self._generate_recommended_structure(entries),
            'problematic_modules': self._identify_problematic_modules(entries),
            'integration_opportunities': self._identify_integration_opportunities(entries),
            'security_concerns': self._analyze_security_concerns(entries),
            'testing_gaps': self._analyze_testing_gaps(entries)
        }

        return insights

    def _generate_recommended_structure(self, entries: List[LLMIntelligenceEntry]) -> Dict[str, Any]:
        """Generate recommended directory structure based on analysis"""
        # Group entries by primary classification
        by_classification = {}
        for entry in entries:
            primary = entry.primary_classification
            if primary not in by_classification:
                by_classification[primary] = []
            by_classification[primary].append(entry.relative_path)

        return {
            'core_directories': list(by_classification.keys()),
            'directory_contents': by_classification,
            'estimated_reduction': self._calculate_organization_benefit(entries)
        }

    def _identify_problematic_modules(self, entries: List[LLMIntelligenceEntry]) -> List[Dict[str, Any]]:
        """Identify modules that need special attention"""
        problematic = []

        for entry in entries:
            issues = []

            if entry.confidence_score < 0.6:
                issues.append("Low analysis confidence")

            if "security" in entry.security_implications.lower() and "none" not in entry.security_implications.lower():
                issues.append("Security concerns identified")

            if entry.complexity_assessment.lower() in ['high', 'very high', 'complex']:
                issues.append("High complexity")

            if issues:
                problematic.append({
                    'path': entry.relative_path,
                    'issues': issues,
                    'confidence': entry.confidence_score
                })

        return problematic

    def _identify_integration_opportunities(self, entries: List[LLMIntelligenceEntry]) -> List[Dict[str, Any]]:
        """Identify opportunities for better module integration"""
        opportunities = []

        # Look for modules with similar classifications that could be grouped
        by_classification = {}
        for entry in entries:
            primary = entry.primary_classification
            if primary not in by_classification:
                by_classification[primary] = []
            by_classification[primary].append(entry)

        for classification, class_entries in by_classification.items():
            if len(class_entries) > 1:
                opportunities.append({
                    'type': 'similar_classification',
                    'classification': classification,
                    'modules': [entry.relative_path for entry in class_entries],
                    'recommendation': f'Consider grouping {len(class_entries)} {classification} modules together'
                })

        return opportunities

    def _analyze_security_concerns(self, entries: List[LLMIntelligenceEntry]) -> List[Dict[str, Any]]:
        """Analyze security concerns across modules"""
        concerns = []

        for entry in entries:
            if "security" in entry.primary_classification or "security" in entry.security_implications.lower():
                concerns.append({
                    'module': entry.relative_path,
                    'security_notes': entry.security_implications,
                    'confidence': entry.confidence_score
                })

        return concerns

    def _analyze_testing_gaps(self, entries: List[LLMIntelligenceEntry]) -> List[Dict[str, Any]]:
        """Analyze testing gaps and requirements"""
        gaps = []

        for entry in entries:
            if entry.testing_requirements and "none" not in entry.testing_requirements.lower():
                gaps.append({
                    'module': entry.relative_path,
                    'testing_needs': entry.testing_requirements,
                    'complexity': entry.complexity_assessment
                })

        return gaps

    def _calculate_organization_benefit(self, entries: List[LLMIntelligenceEntry]) -> float:
        """Calculate estimated benefit from reorganization"""
        # Simple heuristic: more classified modules = better organization potential
        classified_entries = [e for e in entries if e.primary_classification != 'uncategorized']
        classification_ratio = len(classified_entries) / len(entries) if entries else 0

        return round(classification_ratio * 100, 1)  # Return as percentage
