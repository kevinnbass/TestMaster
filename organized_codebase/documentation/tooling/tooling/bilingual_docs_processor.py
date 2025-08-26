"""
Bilingual Documentation Processor

Handles bilingual documentation creation, translation, and synchronization
based on AgentScope's en/zh_CN documentation structure.
"""

import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    CHINESE_SIMPLIFIED = "zh_CN"
    JAPANESE = "ja"
    KOREAN = "ko"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    RUSSIAN = "ru"


class DocumentType(Enum):
    """Types of documentation."""
    README = "readme"
    API = "api"
    TUTORIAL = "tutorial"
    GUIDE = "guide"
    REFERENCE = "reference"
    CHANGELOG = "changelog"
    FAQ = "faq"


@dataclass
class TranslationEntry:
    """Represents a translation entry."""
    key: str
    source_text: str
    target_text: str
    source_lang: Language
    target_lang: Language
    context: str = ""
    last_updated: str = ""
    reviewer: str = ""
    status: str = "pending"  # pending, reviewed, approved


@dataclass
class DocumentSection:
    """Represents a section of documentation."""
    id: str
    title: str
    content: str
    subsections: List['DocumentSection'] = field(default_factory=list)
    translations: Dict[Language, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BilingualDocsProcessor:
    """
    Bilingual documentation processor inspired by AgentScope's
    comprehensive en/zh_CN documentation structure.
    """
    
    def __init__(self, docs_root: str = "docs"):
        """Initialize bilingual docs processor."""
        self.docs_root = Path(docs_root)
        self.translations = {}
        self.documents = {}
        self.language_configs = self._load_language_configs()
        self.translation_memory = {}
        logger.info(f"Bilingual docs processor initialized at {docs_root}")
        
    def create_language_structure(self, languages: List[Language]) -> None:
        """Create directory structure for multiple languages."""
        for lang in languages:
            lang_dir = self.docs_root / lang.value
            lang_dir.mkdir(parents=True, exist_ok=True)
            
            # Create standard subdirectories
            for subdir in ["api", "tutorial", "examples", "guides"]:
                (lang_dir / subdir).mkdir(exist_ok=True)
                
        logger.info(f"Created language structure for {[l.value for l in languages]}")
        
    def parse_document(self, file_path: str, doc_type: DocumentType) -> DocumentSection:
        """Parse a document into sections."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Detect language from path or content
        lang = self._detect_language(file_path, content)
        
        # Parse sections based on markdown headers
        sections = self._parse_markdown_sections(content)
        
        # Create document structure
        doc = DocumentSection(
            id=Path(file_path).stem,
            title=self._extract_title(content),
            content=content,
            subsections=sections,
            metadata={
                "language": lang,
                "type": doc_type,
                "file_path": file_path
            }
        )
        
        self.documents[doc.id] = doc
        return doc
        
    def extract_translatable_strings(self, document: DocumentSection) -> List[str]:
        """Extract strings that need translation."""
        strings = []
        
        # Extract from content
        content_strings = self._extract_content_strings(document.content)
        strings.extend(content_strings)
        
        # Extract from subsections
        for section in document.subsections:
            section_strings = self._extract_content_strings(section.content)
            strings.extend(section_strings)
            
        return list(set(strings))  # Remove duplicates
        
    def create_translation_template(self, 
                                  source_lang: Language, 
                                  target_lang: Language,
                                  document: DocumentSection) -> Dict[str, Any]:
        """Create translation template for a document."""
        strings = self.extract_translatable_strings(document)
        
        template = {
            "metadata": {
                "source_language": source_lang.value,
                "target_language": target_lang.value,
                "document_id": document.id,
                "created_date": self._get_current_timestamp(),
                "total_strings": len(strings)
            },
            "translations": {}
        }
        
        # Create translation entries
        for i, string in enumerate(strings):
            key = f"string_{i:04d}"
            template["translations"][key] = {
                "source": string,
                "target": "",
                "context": self._extract_context(document.content, string),
                "status": "pending"
            }
            
        return template
        
    def apply_translations(self, 
                          document: DocumentSection,
                          translations: Dict[str, str],
                          target_lang: Language) -> str:
        """Apply translations to document content."""
        content = document.content
        
        # Apply translations
        for source, target in translations.items():
            if target:  # Only apply non-empty translations
                content = content.replace(source, target)
                
        # Update language-specific formatting
        content = self._apply_language_formatting(content, target_lang)
        
        return content
        
    def sync_bilingual_docs(self, 
                           primary_lang: Language, 
                           secondary_lang: Language) -> Dict[str, Any]:
        """Synchronize bilingual documentation."""
        sync_report = {
            "added": [],
            "modified": [],
            "deleted": [],
            "conflicts": []
        }
        
        primary_dir = self.docs_root / primary_lang.value
        secondary_dir = self.docs_root / secondary_lang.value
        
        # Find all documentation files in primary language
        primary_files = self._find_doc_files(primary_dir)
        secondary_files = self._find_doc_files(secondary_dir)
        
        # Check for new files
        for file_path in primary_files:
            rel_path = file_path.relative_to(primary_dir)
            secondary_file = secondary_dir / rel_path
            
            if not secondary_file.exists():
                sync_report["added"].append(str(rel_path))
                # Create placeholder file
                self._create_translation_placeholder(file_path, secondary_file, secondary_lang)
                
        # Check for modified files
        for file_path in primary_files:
            rel_path = file_path.relative_to(primary_dir)
            secondary_file = secondary_dir / rel_path
            
            if secondary_file.exists():
                primary_mtime = file_path.stat().st_mtime
                secondary_mtime = secondary_file.stat().st_mtime
                
                if primary_mtime > secondary_mtime:
                    sync_report["modified"].append(str(rel_path))
                    
        # Check for orphaned translations
        for file_path in secondary_files:
            rel_path = file_path.relative_to(secondary_dir)
            primary_file = primary_dir / rel_path
            
            if not primary_file.exists():
                sync_report["deleted"].append(str(rel_path))
                
        return sync_report
        
    def generate_translation_status(self) -> Dict[str, Any]:
        """Generate translation status report."""
        status = {
            "summary": {},
            "by_language": {},
            "by_document": {}
        }
        
        total_documents = len(self.documents)
        total_translations = 0
        completed_translations = 0
        
        # Calculate overall statistics
        for doc_id, document in self.documents.items():
            for lang, translation in document.translations.items():
                total_translations += 1
                if translation and translation.strip():
                    completed_translations += 1
                    
        completion_rate = (completed_translations / total_translations * 100) if total_translations > 0 else 0
        
        status["summary"] = {
            "total_documents": total_documents,
            "total_translations": total_translations,
            "completed_translations": completed_translations,
            "completion_rate": f"{completion_rate:.1f}%"
        }
        
        return status
        
    def validate_translations(self, document: DocumentSection) -> List[Dict[str, Any]]:
        """Validate translations for consistency."""
        issues = []
        
        # Check for missing translations
        for lang in Language:
            if lang not in document.translations or not document.translations[lang]:
                issues.append({
                    "type": "missing_translation",
                    "language": lang.value,
                    "severity": "warning"
                })
                
        # Check for format consistency
        source_format = self._analyze_format(document.content)
        
        for lang, translation in document.translations.items():
            if translation:
                target_format = self._analyze_format(translation)
                
                # Check header consistency
                if source_format["headers"] != target_format["headers"]:
                    issues.append({
                        "type": "header_mismatch",
                        "language": lang.value,
                        "severity": "error",
                        "details": "Header structure doesn't match source"
                    })
                    
                # Check link consistency
                if len(source_format["links"]) != len(target_format["links"]):
                    issues.append({
                        "type": "link_count_mismatch",
                        "language": lang.value,
                        "severity": "warning",
                        "details": "Number of links doesn't match source"
                    })
                    
        return issues
        
    def create_glossary(self, terms: Dict[str, Dict[Language, str]]) -> str:
        """Create multilingual glossary."""
        glossary_lines = [
            "# Multilingual Glossary",
            "",
            "Technical terms and their translations across languages.",
            "",
            "| Term (EN) | 中文 | 日本語 | 한국어 |",
            "|-----------|------|--------|--------|"
        ]
        
        for en_term, translations in terms.items():
            row = [en_term]
            for lang in [Language.CHINESE_SIMPLIFIED, Language.JAPANESE, Language.KOREAN]:
                translation = translations.get(lang, "—")
                row.append(translation)
                
            glossary_lines.append("| " + " | ".join(row) + " |")
            
        return "\n".join(glossary_lines)
        
    def _load_language_configs(self) -> Dict[Language, Dict[str, Any]]:
        """Load language-specific configuration."""
        return {
            Language.ENGLISH: {
                "direction": "ltr",
                "date_format": "%Y-%m-%d",
                "number_format": "1,234.56"
            },
            Language.CHINESE_SIMPLIFIED: {
                "direction": "ltr",
                "date_format": "%Y年%m月%d日",
                "number_format": "1,234.56"
            },
            Language.JAPANESE: {
                "direction": "ltr",
                "date_format": "%Y年%m月%d日",
                "number_format": "1,234.56"
            }
        }
        
    def _detect_language(self, file_path: str, content: str) -> Language:
        """Detect language from file path or content."""
        path_parts = Path(file_path).parts
        
        # Check if language is in path
        for lang in Language:
            if lang.value in path_parts:
                return lang
                
        # Detect from content
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', content)
        japanese_chars = re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', content)
        
        if len(chinese_chars) > 10:
            return Language.CHINESE_SIMPLIFIED
        elif len(japanese_chars) > 10:
            return Language.JAPANESE
        else:
            return Language.ENGLISH
            
    def _parse_markdown_sections(self, content: str) -> List[DocumentSection]:
        """Parse markdown content into sections."""
        sections = []
        
        # Find headers and their content
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        current_section = None
        current_content = []
        
        for line in lines:
            match = re.match(header_pattern, line)
            
            if match:
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(current_content)
                    sections.append(current_section)
                    
                # Start new section
                level = len(match.group(1))
                title = match.group(2)
                section_id = re.sub(r'[^a-zA-Z0-9]', '_', title.lower())
                
                current_section = DocumentSection(
                    id=section_id,
                    title=title,
                    content="",
                    metadata={"level": level}
                )
                current_content = [line]
            else:
                if current_content is not None:
                    current_content.append(line)
                    
        # Add final section
        if current_section:
            current_section.content = '\n'.join(current_content)
            sections.append(current_section)
            
        return sections
        
    def _extract_title(self, content: str) -> str:
        """Extract title from document content."""
        lines = content.split('\n')
        
        for line in lines:
            if line.startswith('# '):
                return line[2:].strip()
                
        return "Untitled Document"
        
    def _extract_content_strings(self, content: str) -> List[str]:
        """Extract translatable strings from content."""
        strings = []
        
        # Extract headers
        header_matches = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        strings.extend(header_matches)
        
        # Extract paragraphs (skip code blocks)
        in_code_block = False
        lines = content.split('\n')
        
        for line in lines:
            if line.startswith('```'):
                in_code_block = not in_code_block
                continue
                
            if not in_code_block and line.strip() and not line.startswith('#'):
                # Skip links, code, and other non-translatable content
                clean_line = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', line)  # Extract link text
                clean_line = re.sub(r'`[^`]+`', '', clean_line)  # Remove inline code
                
                if clean_line.strip():
                    strings.append(clean_line.strip())
                    
        return strings
        
    def _extract_context(self, content: str, string: str) -> str:
        """Extract context for a string."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if string in line:
                # Get surrounding lines for context
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                context_lines = lines[start:end]
                return ' | '.join(context_lines)
                
        return ""
        
    def _apply_language_formatting(self, content: str, target_lang: Language) -> str:
        """Apply language-specific formatting."""
        config = self.language_configs.get(target_lang, {})
        
        # Apply date format changes
        if target_lang == Language.CHINESE_SIMPLIFIED:
            # Convert English dates to Chinese format
            content = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\1年\2月\3日', content)
            
        return content
        
    def _find_doc_files(self, directory: Path) -> List[Path]:
        """Find all documentation files in directory."""
        doc_files = []
        
        for ext in ['.md', '.rst', '.txt']:
            doc_files.extend(directory.rglob(f'*{ext}'))
            
        return doc_files
        
    def _create_translation_placeholder(self, source_file: Path, target_file: Path, target_lang: Language) -> None:
        """Create placeholder file for translation."""
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        placeholder_content = f"""# Translation Needed

This document needs to be translated to {target_lang.value}.

Source: {source_file.relative_to(self.docs_root)}

---

*This is a placeholder. Please translate the content from the source file.*
"""
        
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(placeholder_content)
            
    def _analyze_format(self, content: str) -> Dict[str, Any]:
        """Analyze document format structure."""
        format_info = {
            "headers": len(re.findall(r'^#+\s', content, re.MULTILINE)),
            "links": len(re.findall(r'\[([^\]]+)\]\([^\)]+\)', content)),
            "code_blocks": len(re.findall(r'```', content)) // 2,
            "lists": len(re.findall(r'^\s*[-*+]\s', content, re.MULTILINE))
        }
        
        return format_info
        
    def _get_current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
        
    def export_translation_template(self, document_id: str, source_lang: Language, target_lang: Language, output_path: str) -> None:
        """Export translation template to file."""
        if document_id not in self.documents:
            logger.error(f"Document {document_id} not found")
            return
            
        document = self.documents[document_id]
        template = self.create_translation_template(source_lang, target_lang, document)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Exported translation template to {output_path}")
        
    def import_completed_translations(self, template_path: str) -> None:
        """Import completed translations from template file."""
        with open(template_path, 'r', encoding='utf-8') as f:
            template = json.load(f)
            
        document_id = template["metadata"]["document_id"]
        target_lang = Language(template["metadata"]["target_language"])
        
        if document_id in self.documents:
            translations = {}
            for entry in template["translations"].values():
                if entry["target"]:
                    translations[entry["source"]] = entry["target"]
                    
            translated_content = self.apply_translations(
                self.documents[document_id], 
                translations, 
                target_lang
            )
            
            self.documents[document_id].translations[target_lang] = translated_content
            logger.info(f"Imported translations for {document_id} ({target_lang.value})")