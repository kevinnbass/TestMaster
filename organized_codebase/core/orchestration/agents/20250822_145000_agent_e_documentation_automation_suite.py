#!/usr/bin/env python3
"""
Agent E Documentation Automation Suite
Advanced automation tools to support Agent E's extensive documentation efforts
"""

import os
import re
import json
import sqlite3
import datetime
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import markdown
from collections import defaultdict
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade


class DocumentType(Enum):
    ANALYSIS = "analysis"
    TECHNICAL = "technical"
    MISSION_REPORT = "mission_report"
    EXECUTIVE_SUMMARY = "executive_summary"
    HANDOFF = "handoff"
    FRAMEWORK = "framework"


class QualityLevel(Enum):
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 80-89
    ADEQUATE = "adequate"   # 70-79
    NEEDS_IMPROVEMENT = "needs_improvement"  # <70


@dataclass
class DocumentMetrics:
    word_count: int
    section_count: int
    subsection_count: int
    code_block_count: int
    table_count: int
    list_count: int
    link_count: int
    reading_level: float
    readability_score: float
    quality_score: float
    completion_percentage: float


@dataclass
class DocumentationTask:
    task_id: str
    title: str
    doc_type: DocumentType
    priority: str
    status: str
    assigned_agent: str
    created_at: datetime.datetime
    estimated_completion: datetime.datetime
    actual_completion: Optional[datetime.datetime]
    deliverables: List[str]
    dependencies: List[str]
    quality_requirements: Dict[str, Any]


class AgentEDocumentationAutomationSuite:
    """
    Comprehensive automation suite to support Agent E's extensive documentation efforts
    Provides intelligent analysis, quality assessment, cross-referencing, and automation
    """
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.db_path = self.base_path / "documentation_automation.db"
        self.analysis_cache = {}
        self.quality_standards = self._init_quality_standards()
        self._init_database()
        
    def _init_quality_standards(self) -> Dict[str, Any]:
        """Initialize comprehensive quality standards for documentation"""
        return {
            "minimum_word_count": {
                DocumentType.ANALYSIS: 2000,
                DocumentType.TECHNICAL: 1500,
                DocumentType.MISSION_REPORT: 3000,
                DocumentType.EXECUTIVE_SUMMARY: 1000,
                DocumentType.HANDOFF: 1200,
                DocumentType.FRAMEWORK: 2500
            },
            "minimum_sections": {
                DocumentType.ANALYSIS: 6,
                DocumentType.TECHNICAL: 5,
                DocumentType.MISSION_REPORT: 8,
                DocumentType.EXECUTIVE_SUMMARY: 4,
                DocumentType.HANDOFF: 5,
                DocumentType.FRAMEWORK: 7
            },
            "required_elements": {
                "executive_summary": True,
                "detailed_analysis": True,
                "metrics_and_data": True,
                "examples_or_code": True,
                "conclusions": True
            },
            "quality_thresholds": {
                "excellent": 90,
                "good": 80,
                "adequate": 70,
                "minimum_acceptable": 60
            },
            "readability_targets": {
                "max_reading_level": 16,  # College graduate level
                "min_readability_score": 40  # Fairly difficult but acceptable
            }
        }
        
    def _init_database(self):
        """Initialize documentation automation database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Documentation analysis table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_analysis (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                doc_type TEXT,
                agent_author TEXT,
                analysis_date TIMESTAMP,
                word_count INTEGER,
                section_count INTEGER,
                quality_score REAL,
                readability_score REAL,
                completion_status TEXT,
                recommendations TEXT,
                cross_references TEXT,
                integration_status TEXT
            )
        """)
        
        # Documentation tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documentation_tasks (
                task_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                doc_type TEXT,
                priority TEXT,
                status TEXT,
                assigned_agent TEXT,
                created_at TIMESTAMP,
                estimated_completion TIMESTAMP,
                actual_completion TIMESTAMP,
                deliverables TEXT,
                dependencies TEXT,
                quality_requirements TEXT
            )
        """)
        
        # Cross-reference tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cross_references (
                ref_id TEXT PRIMARY KEY,
                source_doc TEXT NOT NULL,
                target_doc TEXT NOT NULL,
                reference_type TEXT,
                link_text TEXT,
                validation_status TEXT,
                last_validated TIMESTAMP,
                validation_notes TEXT
            )
        """)
        
        # Quality metrics tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_metrics (
                metric_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                metric_type TEXT,
                metric_value REAL,
                benchmark_value REAL,
                measurement_date TIMESTAMP,
                improvement_suggestions TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
    def analyze_document_comprehensive(self, file_path: str, doc_type: DocumentType = None) -> Dict[str, Any]:
        """Perform comprehensive analysis of a documentation file"""
        file_path = Path(file_path)
        if not file_path.exists():
            return {"error": f"File {file_path} not found"}
            
        # Read and parse document
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            return {"error": f"Failed to read file: {str(e)}"}
            
        # Detect document type if not provided
        if doc_type is None:
            doc_type = self._detect_document_type(content, file_path.name)
            
        # Comprehensive content analysis
        metrics = self._analyze_content_metrics(content)
        structure = self._analyze_document_structure(content)
        quality = self._assess_quality_comprehensive(content, doc_type, metrics, structure)
        cross_refs = self._analyze_cross_references(content, file_path)
        integration = self._analyze_integration_points(content)
        
        # Generate improvement recommendations
        recommendations = self._generate_improvement_recommendations(content, doc_type, quality, metrics)
        
        doc_id = hashlib.md5(f"{file_path}_{datetime.datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        analysis_result = {
            "doc_id": doc_id,
            "filename": str(file_path),
            "doc_type": doc_type.value if doc_type else "unknown",
            "agent_author": self._detect_agent_author(content),
            "analysis_timestamp": datetime.datetime.now().isoformat(),
            "metrics": metrics.__dict__,
            "structure_analysis": structure,
            "quality_assessment": quality,
            "cross_references": cross_refs,
            "integration_analysis": integration,
            "recommendations": recommendations,
            "overall_score": quality.get("overall_score", 0),
            "completion_status": self._determine_completion_status(quality, metrics)
        }
        
        # Save to database
        self._save_document_analysis(analysis_result)
        
        return analysis_result
        
    def _analyze_content_metrics(self, content: str) -> DocumentMetrics:
        """Analyze detailed content metrics"""
        word_count = len(content.split())
        section_count = content.count('\n#')
        subsection_count = content.count('\n##') + content.count('\n###')
        code_block_count = content.count('```')
        table_count = content.count('|')
        list_count = content.count('\n-') + content.count('\n*') + content.count('\n1.')
        link_count = content.count('[') + content.count('http')
        
        # Readability analysis
        try:
            reading_level = flesch_kincaid_grade(content)
            readability_score = flesch_reading_ease(content)
        except:
            reading_level = 12.0  # Default to college level
            readability_score = 50.0  # Default to average
            
        # Calculate quality score based on content richness
        quality_score = min(100, (
            min(word_count / 20, 50) +  # Word count factor (up to 50 points)
            min(section_count * 5, 25) +  # Structure factor (up to 25 points)
            min(code_block_count * 3, 15) +  # Code examples factor (up to 15 points)
            (10 if table_count > 0 else 0)  # Tables factor (10 points)
        ))
        
        completion_percentage = min(100, (word_count / 2000) * 100)  # Based on 2000 word target
        
        return DocumentMetrics(
            word_count=word_count,
            section_count=section_count,
            subsection_count=subsection_count,
            code_block_count=code_block_count,
            table_count=table_count,
            list_count=list_count,
            link_count=link_count,
            reading_level=reading_level,
            readability_score=readability_score,
            quality_score=quality_score,
            completion_percentage=completion_percentage
        )
        
    def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure and organization"""
        lines = content.split('\n')
        
        structure = {
            "has_title": bool(re.search(r'^#\s+', content, re.MULTILINE)),
            "has_toc": "table of contents" in content.lower() or "## " in content,
            "has_executive_summary": any(word in content.lower() for word in ["executive summary", "overview", "summary"]),
            "has_conclusion": any(word in content.lower() for word in ["conclusion", "summary", "final"]),
            "header_hierarchy": self._analyze_header_hierarchy(content),
            "section_distribution": self._analyze_section_distribution(content),
            "formatting_consistency": self._check_formatting_consistency(content),
            "metadata_present": self._check_metadata_presence(content)
        }
        
        return structure
        
    def _assess_quality_comprehensive(self, content: str, doc_type: DocumentType, 
                                    metrics: DocumentMetrics, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive quality assessment based on multiple factors"""
        if not doc_type:
            doc_type = DocumentType.ANALYSIS  # Default
            
        standards = self.quality_standards
        
        # Content quality scoring
        content_score = 0
        
        # Word count assessment
        min_words = standards["minimum_word_count"].get(doc_type, 1000)
        content_score += min(30, (metrics.word_count / min_words) * 30)
        
        # Structure assessment
        min_sections = standards["minimum_sections"].get(doc_type, 5)
        content_score += min(25, (metrics.section_count / min_sections) * 25)
        
        # Content richness assessment
        if metrics.code_block_count > 0:
            content_score += 10
        if metrics.table_count > 0:
            content_score += 10
        if metrics.list_count > 3:
            content_score += 10
            
        # Readability assessment
        if 30 <= metrics.readability_score <= 70:  # Good readability range
            content_score += 10
        if metrics.reading_level <= 16:  # College level or below
            content_score += 5
            
        # Structure quality assessment
        structure_score = 0
        if structure["has_title"]:
            structure_score += 10
        if structure["has_executive_summary"]:
            structure_score += 15
        if structure["has_conclusion"]:
            structure_score += 10
        if structure["formatting_consistency"]["score"] > 0.8:
            structure_score += 15
            
        overall_score = min(100, content_score + structure_score)
        
        # Determine quality level
        if overall_score >= 90:
            quality_level = QualityLevel.EXCELLENT
        elif overall_score >= 80:
            quality_level = QualityLevel.GOOD
        elif overall_score >= 70:
            quality_level = QualityLevel.ADEQUATE
        else:
            quality_level = QualityLevel.NEEDS_IMPROVEMENT
            
        return {
            "overall_score": round(overall_score, 1),
            "content_score": round(content_score, 1),
            "structure_score": round(structure_score, 1),
            "quality_level": quality_level.value,
            "meets_standards": overall_score >= standards["quality_thresholds"]["minimum_acceptable"],
            "assessment_details": {
                "word_count_adequacy": metrics.word_count >= min_words,
                "structure_adequacy": metrics.section_count >= min_sections,
                "readability_score": metrics.readability_score,
                "content_richness": {
                    "has_code_examples": metrics.code_block_count > 0,
                    "has_tables": metrics.table_count > 0,
                    "has_lists": metrics.list_count > 0
                }
            }
        }
        
    def _analyze_cross_references(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Analyze cross-references and links within the document"""
        # Find markdown links
        markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        
        # Find direct file references
        file_refs = re.findall(r'`([^`]*\.(py|md|json|txt|yaml|yml))`', content)
        
        # Find agent references
        agent_refs = re.findall(r'Agent [A-Z]', content)
        
        # Validate internal links
        valid_links = []
        broken_links = []
        
        for link_text, link_url in markdown_links:
            if link_url.startswith('#'):  # Internal anchor
                # Check if anchor exists in document
                anchor = link_url[1:].lower().replace('-', ' ')
                if anchor in content.lower():
                    valid_links.append((link_text, link_url))
                else:
                    broken_links.append((link_text, link_url))
            elif link_url.startswith('http'):  # External link
                valid_links.append((link_text, link_url))  # Assume valid for now
            else:  # File reference
                ref_path = file_path.parent / link_url
                if ref_path.exists():
                    valid_links.append((link_text, link_url))
                else:
                    broken_links.append((link_text, link_url))
                    
        return {
            "total_links": len(markdown_links),
            "valid_links": len(valid_links),
            "broken_links": len(broken_links),
            "file_references": [ref[0] for ref in file_refs],
            "agent_references": list(set(agent_refs)),
            "link_validation": {
                "valid": valid_links,
                "broken": broken_links
            }
        }
        
    def _analyze_integration_points(self, content: str) -> Dict[str, Any]:
        """Analyze integration points and dependencies mentioned in the document"""
        integration_keywords = [
            "integration", "coordinate", "handoff", "dependency", "collaborate",
            "interface", "api", "framework", "system", "component", "module"
        ]
        
        integration_mentions = {}
        for keyword in integration_keywords:
            pattern = rf'\b{keyword}\b'
            matches = re.findall(pattern, content, re.IGNORECASE)
            integration_mentions[keyword] = len(matches)
            
        # Find specific agent collaboration mentions
        collaboration_patterns = [
            r'Agent [A-Z] (?:and|with|collaborates|coordinates|integrates) Agent [A-Z]',
            r'handoff (?:from|to|between) Agent [A-Z]',
            r'(?:supports?|assists?) Agent [A-Z]'
        ]
        
        collaborations = []
        for pattern in collaboration_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            collaborations.extend(matches)
            
        return {
            "integration_density": sum(integration_mentions.values()),
            "integration_keywords": integration_mentions,
            "collaboration_mentions": collaborations,
            "integration_score": min(100, sum(integration_mentions.values()) * 2)
        }
        
    def _generate_improvement_recommendations(self, content: str, doc_type: DocumentType,
                                            quality: Dict[str, Any], metrics: DocumentMetrics) -> List[str]:
        """Generate specific improvement recommendations for the document"""
        recommendations = []
        
        # Word count recommendations
        if doc_type in self.quality_standards["minimum_word_count"]:
            min_words = self.quality_standards["minimum_word_count"][doc_type]
            if metrics.word_count < min_words:
                recommendations.append(f"Expand content to reach minimum {min_words} words (currently {metrics.word_count})")
                
        # Structure recommendations
        if metrics.section_count < 3:
            recommendations.append("Add more sections to improve document structure")
            
        # Content richness recommendations
        if metrics.code_block_count == 0 and doc_type in [DocumentType.TECHNICAL, DocumentType.FRAMEWORK]:
            recommendations.append("Add code examples or implementation details")
            
        if metrics.table_count == 0 and doc_type in [DocumentType.ANALYSIS, DocumentType.MISSION_REPORT]:
            recommendations.append("Consider adding tables for metrics or comparative data")
            
        # Readability recommendations
        if metrics.readability_score < 30:
            recommendations.append("Improve readability by using shorter sentences and simpler language")
        elif metrics.readability_score > 80:
            recommendations.append("Content may be too simple; consider adding more technical detail")
            
        # Quality-based recommendations
        if quality["overall_score"] < 80:
            recommendations.append("Overall document quality needs improvement")
            
        if not any("summary" in content.lower() for content in [content]):
            recommendations.append("Add an executive summary or overview section")
            
        return recommendations
        
    def create_documentation_task(self, title: str, doc_type: DocumentType, 
                                 priority: str = "medium", assigned_agent: str = "Agent_E",
                                 deliverables: List[str] = None, dependencies: List[str] = None) -> str:
        """Create a new documentation task for Agent E"""
        task_id = hashlib.md5(f"{title}_{datetime.datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        task = DocumentationTask(
            task_id=task_id,
            title=title,
            doc_type=doc_type,
            priority=priority,
            status="pending",
            assigned_agent=assigned_agent,
            created_at=datetime.datetime.now(),
            estimated_completion=datetime.datetime.now() + datetime.timedelta(days=7),
            actual_completion=None,
            deliverables=deliverables or [],
            dependencies=dependencies or [],
            quality_requirements=self.quality_standards["minimum_word_count"].get(doc_type, 1000)
        )
        
        self._save_documentation_task(task)
        return task_id
        
    def get_agent_e_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard for Agent E's documentation work"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all Agent E documents
        cursor.execute("""
            SELECT doc_type, COUNT(*), AVG(quality_score), SUM(word_count)
            FROM document_analysis 
            WHERE agent_author = 'Agent_E' OR agent_author LIKE '%Agent E%'
            GROUP BY doc_type
        """)
        doc_stats = cursor.fetchall()
        
        # Get task statistics
        cursor.execute("""
            SELECT status, COUNT(*) 
            FROM documentation_tasks 
            WHERE assigned_agent = 'Agent_E'
            GROUP BY status
        """)
        task_stats = cursor.fetchall()
        
        # Get quality trends
        cursor.execute("""
            SELECT AVG(quality_score), COUNT(*)
            FROM document_analysis 
            WHERE agent_author = 'Agent_E' OR agent_author LIKE '%Agent E%'
        """)
        quality_avg, total_docs = cursor.fetchone()
        
        conn.close()
        
        return {
            "dashboard_timestamp": datetime.datetime.now().isoformat(),
            "agent_status": "EXTENSIVELY_DOCUMENTING",
            "documentation_statistics": {
                "total_documents": total_docs or 0,
                "average_quality": round(quality_avg or 0, 1),
                "documents_by_type": {doc_type: {"count": count, "avg_quality": round(avg_qual, 1), "total_words": total_words} 
                                    for doc_type, count, avg_qual, total_words in doc_stats},
                "total_word_count": sum(stats[3] for stats in doc_stats) if doc_stats else 0
            },
            "task_management": {
                "tasks_by_status": {status: count for status, count in task_stats},
                "documentation_workflow": "ACTIVE_EXTENSIVE_DOCUMENTATION"
            },
            "quality_metrics": {
                "overall_quality": round(quality_avg or 0, 1),
                "quality_rating": self._get_quality_rating(quality_avg or 0),
                "improvement_focus": self._get_improvement_focus()
            },
            "support_recommendations": [
                "Continue extensive documentation efforts with automated quality checking",
                "Utilize cross-reference validation for integration documentation", 
                "Implement automated formatting consistency checks",
                "Use task management system for coordination with Agent D handoffs"
            ]
        }
        
    def validate_cross_references_batch(self, doc_paths: List[str]) -> Dict[str, Any]:
        """Validate cross-references across multiple documents"""
        all_documents = {}
        all_references = []
        validation_results = {}
        
        # Read all documents and extract references
        for doc_path in doc_paths:
            if Path(doc_path).exists():
                content = Path(doc_path).read_text(encoding='utf-8', errors='ignore')
                all_documents[doc_path] = content
                refs = self._analyze_cross_references(content, Path(doc_path))
                all_references.extend(refs["link_validation"]["valid"])
                all_references.extend(refs["link_validation"]["broken"])
                
        # Validate references across all documents
        for doc_path in doc_paths:
            if doc_path in all_documents:
                validation_results[doc_path] = self._validate_document_references(
                    all_documents[doc_path], doc_path, all_documents
                )
                
        return {
            "total_documents_processed": len(doc_paths),
            "total_references_found": len(all_references),
            "validation_results": validation_results,
            "overall_reference_health": self._calculate_reference_health(validation_results)
        }
        
    # Helper methods
    def _detect_document_type(self, content: str, filename: str) -> DocumentType:
        """Detect document type based on content and filename"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        if "final_mission_report" in filename_lower or "mission report" in content_lower:
            return DocumentType.MISSION_REPORT
        elif "executive_summary" in filename_lower or "executive summary" in content_lower:
            return DocumentType.EXECUTIVE_SUMMARY
        elif "handoff" in filename_lower or "handoff" in content_lower:
            return DocumentType.HANDOFF
        elif "framework" in filename_lower or "framework" in content_lower:
            return DocumentType.FRAMEWORK
        elif any(word in filename_lower for word in ["technical", "implementation", ".py"]):
            return DocumentType.TECHNICAL
        else:
            return DocumentType.ANALYSIS
            
    def _detect_agent_author(self, content: str) -> str:
        """Detect which agent authored the document"""
        if "agent e" in content.lower():
            return "Agent_E"
        elif "agent d" in content.lower():
            return "Agent_D"
        elif "agent c" in content.lower():
            return "Agent_C"
        elif "agent b" in content.lower():
            return "Agent_B"
        elif "agent a" in content.lower():
            return "Agent_A"
        else:
            return "Unknown"
            
    def _analyze_header_hierarchy(self, content: str) -> Dict[str, int]:
        """Analyze header hierarchy structure"""
        h1_count = content.count('\n# ')
        h2_count = content.count('\n## ')
        h3_count = content.count('\n### ')
        h4_count = content.count('\n#### ')
        
        return {
            "h1_count": h1_count,
            "h2_count": h2_count, 
            "h3_count": h3_count,
            "h4_count": h4_count,
            "proper_hierarchy": h1_count >= 1 and h2_count >= h1_count
        }
        
    def _analyze_section_distribution(self, content: str) -> Dict[str, Any]:
        """Analyze how content is distributed across sections"""
        sections = re.split(r'\n#+\s+', content)
        section_lengths = [len(section.split()) for section in sections]
        
        return {
            "section_count": len(sections) - 1,  # Minus 1 for content before first header
            "avg_section_length": sum(section_lengths) / len(section_lengths) if section_lengths else 0,
            "section_length_distribution": {
                "short": len([l for l in section_lengths if l < 100]),
                "medium": len([l for l in section_lengths if 100 <= l < 500]),
                "long": len([l for l in section_lengths if l >= 500])
            }
        }
        
    def _check_formatting_consistency(self, content: str) -> Dict[str, Any]:
        """Check formatting consistency across the document"""
        # Check consistent header formatting
        headers = re.findall(r'\n(#+)\s+(.+)', content)
        consistent_spacing = all(' ' in header[0] for header in headers) if headers else True
        
        # Check list formatting consistency
        list_items = re.findall(r'\n([-*]|\d+\.)\s+', content)
        consistent_lists = len(set(item.strip() for item in list_items)) <= 2 if list_items else True
        
        # Check code block formatting
        code_blocks = re.findall(r'```(\w*)\n', content)
        consistent_code = len(set(code_blocks)) <= 3 if code_blocks else True
        
        consistency_score = sum([consistent_spacing, consistent_lists, consistent_code]) / 3
        
        return {
            "score": consistency_score,
            "consistent_headers": consistent_spacing,
            "consistent_lists": consistent_lists, 
            "consistent_code_blocks": consistent_code
        }
        
    def _check_metadata_presence(self, content: str) -> Dict[str, bool]:
        """Check for presence of metadata elements"""
        return {
            "has_title": bool(re.search(r'^#\s+', content, re.MULTILINE)),
            "has_author": "agent" in content.lower(),
            "has_date": bool(re.search(r'\d{4}-\d{2}-\d{2}', content)),
            "has_version": "version" in content.lower(),
            "has_status": "status" in content.lower()
        }
        
    def _determine_completion_status(self, quality: Dict[str, Any], metrics: DocumentMetrics) -> str:
        """Determine completion status based on quality and metrics"""
        if quality.get("overall_score", 0) >= 90 and metrics.completion_percentage >= 90:
            return "COMPLETE_EXCELLENT"
        elif quality.get("overall_score", 0) >= 80 and metrics.completion_percentage >= 80:
            return "COMPLETE_GOOD"
        elif quality.get("overall_score", 0) >= 70 and metrics.completion_percentage >= 70:
            return "COMPLETE_ADEQUATE"
        elif metrics.completion_percentage >= 50:
            return "IN_PROGRESS"
        else:
            return "DRAFT"
            
    def _save_document_analysis(self, analysis: Dict[str, Any]):
        """Save document analysis to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO document_analysis 
            (doc_id, filename, doc_type, agent_author, analysis_date, word_count, 
             section_count, quality_score, readability_score, completion_status, 
             recommendations, cross_references, integration_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis["doc_id"], analysis["filename"], analysis["doc_type"],
            analysis["agent_author"], analysis["analysis_timestamp"],
            analysis["metrics"]["word_count"], analysis["metrics"]["section_count"],
            analysis["overall_score"], analysis["metrics"]["readability_score"],
            analysis["completion_status"], json.dumps(analysis["recommendations"]),
            json.dumps(analysis["cross_references"]), 
            json.dumps(analysis["integration_analysis"])
        ))
        
        conn.commit()
        conn.close()
        
    def _save_documentation_task(self, task: DocumentationTask):
        """Save documentation task to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO documentation_tasks VALUES 
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id, task.title, task.doc_type.value, task.priority,
            task.status, task.assigned_agent, task.created_at,
            task.estimated_completion, task.actual_completion,
            json.dumps(task.deliverables), json.dumps(task.dependencies),
            json.dumps(task.quality_requirements)
        ))
        
        conn.commit()
        conn.close()
        
    def _get_quality_rating(self, score: float) -> str:
        """Get quality rating from score"""
        if score >= 90:
            return "EXCELLENT"
        elif score >= 80:
            return "GOOD"
        elif score >= 70:
            return "ADEQUATE"
        else:
            return "NEEDS_IMPROVEMENT"
            
    def _get_improvement_focus(self) -> List[str]:
        """Get improvement focus areas"""
        return [
            "Maintain comprehensive documentation standards",
            "Ensure consistent cross-referencing and integration",
            "Continue extensive technical analysis and insights",
            "Optimize documentation workflow and automation"
        ]
        
    def _validate_document_references(self, content: str, doc_path: str, all_docs: Dict[str, str]) -> Dict[str, Any]:
        """Validate references within a single document against all available documents"""
        markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        valid_refs = []
        broken_refs = []
        
        for link_text, link_url in markdown_links:
            if link_url.startswith('#'):
                # Internal anchor
                if link_url[1:].lower().replace('-', ' ') in content.lower():
                    valid_refs.append((link_text, link_url))
                else:
                    broken_refs.append((link_text, link_url))
            elif not link_url.startswith('http'):
                # File reference - check if exists in available docs
                if any(link_url in doc for doc in all_docs.keys()):
                    valid_refs.append((link_text, link_url))
                else:
                    broken_refs.append((link_text, link_url))
                    
        return {
            "total_references": len(markdown_links),
            "valid_references": len(valid_refs),
            "broken_references": len(broken_refs),
            "validation_score": len(valid_refs) / len(markdown_links) * 100 if markdown_links else 100
        }
        
    def _calculate_reference_health(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall reference health across all documents"""
        total_refs = sum(result["total_references"] for result in validation_results.values())
        total_valid = sum(result["valid_references"] for result in validation_results.values())
        total_broken = sum(result["broken_references"] for result in validation_results.values())
        
        return {
            "total_references": total_refs,
            "total_valid": total_valid,
            "total_broken": total_broken,
            "health_score": (total_valid / total_refs * 100) if total_refs > 0 else 100,
            "health_status": "EXCELLENT" if total_broken == 0 else "GOOD" if total_broken < 5 else "NEEDS_ATTENTION"
        }


def main():
    """Demonstration of Agent E documentation automation capabilities"""
    print("[AUTOMATION] Agent E Documentation Automation Suite - Initializing...")
    
    automation_suite = AgentEDocumentationAutomationSuite()
    
    # Analyze existing Agent E documentation
    print("\n[ANALYSIS] Analyzing Agent E documentation quality...")
    agent_e_docs = [
        "AGENT_E_FINAL_MISSION_REPORT.md",
        "AGENT_E_COMPREHENSIVE_FINDINGS.md",
        "AGENT_E_100_HOUR_EXECUTIVE_SUMMARY.md"
    ]
    
    analysis_results = []
    for doc in agent_e_docs:
        if Path(doc).exists():
            result = automation_suite.analyze_document_comprehensive(doc, DocumentType.MISSION_REPORT)
            analysis_results.append(result)
            print(f"[QUALITY] {doc}: {result['overall_score']}/100")
    
    # Create documentation tasks for Agent E
    print("\n[TASKS] Creating documentation tasks for Agent E...")
    task_ids = []
    
    task_id = automation_suite.create_documentation_task(
        "Agent D Integration Documentation",
        DocumentType.TECHNICAL,
        priority="high",
        deliverables=["Integration analysis", "Handoff procedures", "Validation framework"]
    )
    task_ids.append(task_id)
    
    task_id = automation_suite.create_documentation_task(
        "Comprehensive Handoff Documentation Framework",
        DocumentType.FRAMEWORK,
        priority="high",
        deliverables=["Framework templates", "Quality standards", "Automation tools"]
    )
    task_ids.append(task_id)
    
    print(f"[SUCCESS] Created {len(task_ids)} documentation tasks")
    
    # Generate Agent E dashboard
    print("\n[DASHBOARD] Generating Agent E documentation dashboard...")
    dashboard = automation_suite.get_agent_e_dashboard()
    print(json.dumps(dashboard, indent=2))
    
    # Validate cross-references
    if analysis_results:
        print("\n[VALIDATION] Validating cross-references across documents...")
        ref_validation = automation_suite.validate_cross_references_batch(agent_e_docs)
        print(f"[REFERENCES] Health Score: {ref_validation['overall_reference_health']['health_score']:.1f}%")
    
    print("\n[READY] Agent E Documentation Automation Suite - Ready for Extensive Documentation Support")


if __name__ == "__main__":
    main()