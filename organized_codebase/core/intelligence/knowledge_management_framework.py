"""
Knowledge Management Framework
============================
Intelligent knowledge management and information retrieval system for TestMaster.

This module provides comprehensive knowledge base consolidation, semantic search,
knowledge graphs, context-aware documentation, and automated knowledge extraction.

Author: Agent D - Documentation & Validation Excellence  
Phase: Hour 4 - Knowledge Management Systems
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import re
import hashlib
import math
from collections import defaultdict, Counter
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class KnowledgeType(Enum):
    """Types of knowledge in the system."""
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    API_REFERENCE = "api_reference"
    TUTORIAL = "tutorial"
    TROUBLESHOOTING = "troubleshooting"
    BEST_PRACTICES = "best_practices"
    ARCHITECTURAL_DECISIONS = "architectural_decisions"
    CODE_EXAMPLES = "code_examples"
    CONFIGURATION = "configuration"
    LEGACY_KNOWLEDGE = "legacy_knowledge"
    OPERATIONAL_PROCEDURES = "operational_procedures"

class KnowledgeSource(Enum):
    """Sources of knowledge content."""
    DOCUMENTATION_FILES = "documentation_files"
    CODE_COMMENTS = "code_comments"
    DOCSTRINGS = "docstrings"
    README_FILES = "readme_files"
    LEGACY_ARCHIVES = "legacy_archives"
    API_SPECIFICATIONS = "api_specifications"
    TEST_DOCUMENTATION = "test_documentation"
    CONFIGURATION_FILES = "configuration_files"
    COMMIT_MESSAGES = "commit_messages"
    ISSUE_TRACKING = "issue_tracking"

class SearchRelevance(Enum):
    """Relevance levels for search results."""
    EXACT_MATCH = "exact_match"
    HIGH_RELEVANCE = "high_relevance"
    MEDIUM_RELEVANCE = "medium_relevance"
    LOW_RELEVANCE = "low_relevance"
    TANGENTIAL = "tangential"

@dataclass
class KnowledgeItem:
    """Individual knowledge item in the system."""
    item_id: str
    title: str
    content: str
    knowledge_type: KnowledgeType
    source: KnowledgeSource
    source_path: str
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    related_items: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    access_count: int = 0
    relevance_score: float = 0.0
    embedding_vector: Optional[List[float]] = None

@dataclass 
class KnowledgeGraph:
    """Knowledge graph representing relationships between concepts."""
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    concepts: Set[str] = field(default_factory=set)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class SearchResult:
    """Search result with relevance information."""
    item: KnowledgeItem
    relevance: SearchRelevance
    relevance_score: float
    match_reasons: List[str]
    context_snippets: List[str]
    search_query: str

@dataclass
class KnowledgeContext:
    """Context for knowledge delivery and recommendations."""
    user_role: str = "developer"
    current_task: Optional[str] = None
    recent_queries: List[str] = field(default_factory=list)
    preferred_formats: List[str] = field(default_factory=lambda: ["markdown", "code"])
    expertise_level: str = "intermediate"
    active_projects: List[str] = field(default_factory=list)

class SemanticSearchEngine:
    """Advanced semantic search engine for knowledge retrieval."""
    
    def __init__(self):
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        self.concept_index: Dict[str, Set[str]] = defaultdict(set)
        self.phrase_index: Dict[str, Set[str]] = defaultdict(set)
        self.search_history: List[Tuple[str, datetime]] = []
        
    def index_knowledge_item(self, item: KnowledgeItem):
        """Index a knowledge item for search."""
        self.knowledge_items[item.item_id] = item
        
        # Build inverted index
        words = self._extract_searchable_words(item.content + " " + item.title)
        for word in words:
            self.inverted_index[word.lower()].add(item.item_id)
        
        # Index tags and keywords
        for tag in item.tags + item.keywords:
            self.concept_index[tag.lower()].add(item.item_id)
        
        # Index phrases
        phrases = self._extract_phrases(item.content + " " + item.title)
        for phrase in phrases:
            self.phrase_index[phrase.lower()].add(item.item_id)
    
    def search(self, query: str, context: KnowledgeContext = None, 
              max_results: int = 20) -> List[SearchResult]:
        """Perform intelligent semantic search."""
        
        # Record search
        self.search_history.append((query, datetime.now()))
        
        # Parse and expand query
        expanded_query = self._expand_query(query, context)
        
        # Find candidate items
        candidates = self._find_candidate_items(expanded_query)
        
        # Score and rank candidates
        scored_results = []
        for item_id in candidates:
            item = self.knowledge_items[item_id]
            score, reasons, snippets = self._calculate_relevance_score(item, expanded_query, context)
            
            if score > 0.1:  # Minimum relevance threshold
                relevance = self._determine_relevance_level(score)
                result = SearchResult(
                    item=item,
                    relevance=relevance,
                    relevance_score=score,
                    match_reasons=reasons,
                    context_snippets=snippets,
                    search_query=query
                )
                scored_results.append(result)
        
        # Sort by relevance score
        scored_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return scored_results[:max_results]
    
    def suggest_related_queries(self, query: str) -> List[str]:
        """Suggest related queries based on content and search history."""
        suggestions = set()
        
        # Extract key terms from query
        key_terms = self._extract_key_terms(query)
        
        # Find related concepts
        for term in key_terms:
            # Find items containing this term
            if term in self.inverted_index:
                for item_id in list(self.inverted_index[term])[:5]:  # Limit to avoid overload
                    item = self.knowledge_items[item_id]
                    # Add tags as suggestions
                    for tag in item.tags:
                        if tag.lower() != term.lower():
                            suggestions.add(f"{query} {tag}")
            
            # Add concept-based suggestions
            if term in self.concept_index:
                suggestions.add(f"best practices for {term}")
                suggestions.add(f"examples of {term}")
                suggestions.add(f"troubleshooting {term}")
        
        return list(suggestions)[:10]
    
    def _extract_searchable_words(self, text: str) -> List[str]:
        """Extract searchable words from text."""
        # Remove code blocks and special characters
        text = re.sub(r'```[\s\S]*?```', ' ', text)
        text = re.sub(r'`[^`]*`', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Extract words (minimum 3 characters)
        words = [word for word in text.split() if len(word) >= 3]
        return words
    
    def _extract_phrases(self, text: str, max_phrase_length: int = 3) -> List[str]:
        """Extract meaningful phrases from text."""
        words = self._extract_searchable_words(text)
        phrases = []
        
        # Create n-grams
        for n in range(2, max_phrase_length + 1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i+n])
                if len(phrase) >= 6:  # Minimum phrase length
                    phrases.append(phrase)
        
        return phrases
    
    def _expand_query(self, query: str, context: KnowledgeContext) -> Dict[str, float]:
        """Expand query with synonyms and context."""
        expanded = {}
        
        # Original query terms
        query_terms = self._extract_key_terms(query)
        for term in query_terms:
            expanded[term] = 1.0
        
        # Add synonyms and related terms
        synonyms = self._get_synonyms(query_terms)
        for synonym in synonyms:
            expanded[synonym] = 0.7
        
        # Context-based expansion
        if context:
            if context.current_task:
                task_terms = self._extract_key_terms(context.current_task)
                for term in task_terms:
                    expanded[term] = 0.5
            
            # Recent query terms
            for recent_query in context.recent_queries[-3:]:  # Last 3 queries
                recent_terms = self._extract_key_terms(recent_query)
                for term in recent_terms:
                    expanded[term] = 0.3
        
        return expanded
    
    def _find_candidate_items(self, expanded_query: Dict[str, float]) -> Set[str]:
        """Find candidate items matching query terms."""
        candidates = set()
        
        for term, weight in expanded_query.items():
            term_lower = term.lower()
            
            # Exact word matches
            if term_lower in self.inverted_index:
                candidates.update(self.inverted_index[term_lower])
            
            # Concept matches
            if term_lower in self.concept_index:
                candidates.update(self.concept_index[term_lower])
            
            # Phrase matches
            if term_lower in self.phrase_index:
                candidates.update(self.phrase_index[term_lower])
            
            # Partial matches in phrases
            for phrase in self.phrase_index:
                if term_lower in phrase:
                    candidates.update(self.phrase_index[phrase])
        
        return candidates
    
    def _calculate_relevance_score(self, item: KnowledgeItem, 
                                 expanded_query: Dict[str, float],
                                 context: KnowledgeContext = None) -> Tuple[float, List[str], List[str]]:
        """Calculate relevance score for an item."""
        score = 0.0
        reasons = []
        snippets = []
        
        text = (item.title + " " + item.content + " " + " ".join(item.tags)).lower()
        
        # Term frequency scoring
        for term, weight in expanded_query.items():
            term_lower = term.lower()
            
            # Count occurrences
            title_count = item.title.lower().count(term_lower)
            content_count = item.content.lower().count(term_lower)
            tag_count = sum(1 for tag in item.tags if term_lower in tag.lower())
            
            # Title matches are more important
            if title_count > 0:
                score += title_count * weight * 3.0
                reasons.append(f"Title contains '{term}'")
            
            # Content matches
            if content_count > 0:
                score += min(content_count, 5) * weight * 1.0  # Cap at 5 occurrences
                reasons.append(f"Content contains '{term}' ({content_count}x)")
                
                # Extract snippets
                snippet = self._extract_snippet(item.content, term)
                if snippet:
                    snippets.append(snippet)
            
            # Tag matches are highly relevant
            if tag_count > 0:
                score += tag_count * weight * 2.0
                reasons.append(f"Tagged with '{term}'")
        
        # Knowledge type bonuses based on context
        if context:
            if context.current_task:
                if "api" in context.current_task.lower() and item.knowledge_type == KnowledgeType.API_REFERENCE:
                    score *= 1.5
                    reasons.append("API reference for API-related task")
                elif "troubleshoot" in context.current_task.lower() and item.knowledge_type == KnowledgeType.TROUBLESHOOTING:
                    score *= 1.5
                    reasons.append("Troubleshooting guide for problem-solving task")
        
        # Recency bonus
        days_old = (datetime.now() - item.last_updated).days
        if days_old < 30:
            score *= 1.2
            reasons.append("Recently updated")
        elif days_old > 365:
            score *= 0.8
            reasons.append("Older content")
        
        # Popularity bonus
        if item.access_count > 10:
            score *= 1.1
            reasons.append("Frequently accessed")
        
        return score, reasons, snippets
    
    def _extract_snippet(self, content: str, term: str, snippet_length: int = 150) -> str:
        """Extract relevant snippet containing the term."""
        term_lower = term.lower()
        content_lower = content.lower()
        
        # Find term position
        pos = content_lower.find(term_lower)
        if pos == -1:
            return ""
        
        # Extract snippet around term
        start = max(0, pos - snippet_length // 2)
        end = min(len(content), pos + snippet_length // 2)
        
        snippet = content[start:end]
        
        # Clean up snippet
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    def _determine_relevance_level(self, score: float) -> SearchRelevance:
        """Determine relevance level from score."""
        if score >= 10.0:
            return SearchRelevance.EXACT_MATCH
        elif score >= 5.0:
            return SearchRelevance.HIGH_RELEVANCE
        elif score >= 2.0:
            return SearchRelevance.MEDIUM_RELEVANCE
        elif score >= 0.5:
            return SearchRelevance.LOW_RELEVANCE
        else:
            return SearchRelevance.TANGENTIAL
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Remove common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        
        words = self._extract_searchable_words(text)
        key_terms = [word for word in words if word.lower() not in stop_words]
        
        return key_terms
    
    def _get_synonyms(self, terms: List[str]) -> List[str]:
        """Get synonyms for terms (simple rule-based for now)."""
        synonym_map = {
            'test': ['testing', 'spec', 'verification'],
            'api': ['interface', 'endpoint', 'service'],
            'config': ['configuration', 'settings', 'options'],
            'doc': ['documentation', 'docs', 'guide'],
            'error': ['bug', 'issue', 'problem', 'exception'],
            'install': ['setup', 'installation', 'deployment'],
            'run': ['execute', 'start', 'launch'],
            'debug': ['troubleshoot', 'diagnose', 'fix']
        }
        
        synonyms = []
        for term in terms:
            term_lower = term.lower()
            if term_lower in synonym_map:
                synonyms.extend(synonym_map[term_lower])
        
        return synonyms

class KnowledgeGraphBuilder:
    """Builds and maintains knowledge graphs from content."""
    
    def __init__(self):
        self.graph = KnowledgeGraph()
        self.concept_patterns = self._initialize_concept_patterns()
    
    def build_graph_from_knowledge_items(self, items: List[KnowledgeItem]) -> KnowledgeGraph:
        """Build knowledge graph from knowledge items."""
        
        # Extract concepts from all items
        all_concepts = set()
        item_concepts = {}
        
        for item in items:
            concepts = self._extract_concepts(item.content + " " + item.title)
            all_concepts.update(concepts)
            item_concepts[item.item_id] = concepts
            
            # Add item as node
            self.graph.nodes[item.item_id] = {
                'type': 'knowledge_item',
                'title': item.title,
                'knowledge_type': item.knowledge_type.value,
                'concepts': list(concepts)
            }
        
        # Add concept nodes
        for concept in all_concepts:
            concept_id = f"concept_{hashlib.md5(concept.encode()).hexdigest()[:8]}"
            self.graph.nodes[concept_id] = {
                'type': 'concept',
                'name': concept,
                'related_items': []
            }
            self.graph.concepts.add(concept)
        
        # Create edges between items and concepts
        for item_id, concepts in item_concepts.items():
            for concept in concepts:
                concept_id = f"concept_{hashlib.md5(concept.encode()).hexdigest()[:8]}"
                
                # Item -> Concept edge
                self.graph.edges.append({
                    'source': item_id,
                    'target': concept_id,
                    'type': 'contains_concept',
                    'weight': 1.0
                })
                
                # Track relationships
                if concept not in self.graph.relationships:
                    self.graph.relationships[concept] = []
                self.graph.relationships[concept].append(item_id)
        
        # Create inter-concept relationships
        self._create_concept_relationships(all_concepts)
        
        # Create inter-item relationships based on shared concepts
        self._create_item_relationships(item_concepts)
        
        self.graph.last_updated = datetime.now()
        return self.graph
    
    def _extract_concepts(self, text: str) -> Set[str]:
        """Extract concepts from text."""
        concepts = set()
        text_lower = text.lower()
        
        # Pattern-based concept extraction
        for pattern, concept_type in self.concept_patterns.items():
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    concepts.add(match[0])
                else:
                    concepts.add(match)
        
        # Extract technical terms (CamelCase, snake_case)
        tech_terms = re.findall(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+', text)  # CamelCase
        tech_terms.extend(re.findall(r'\b[a-z]+_[a-z_]+\b', text))  # snake_case
        
        for term in tech_terms:
            if len(term) > 3:
                concepts.add(term.lower())
        
        return concepts
    
    def _initialize_concept_patterns(self) -> Dict[str, str]:
        """Initialize patterns for concept extraction."""
        return {
            r'\b(test|testing|spec|specification)\b': 'testing',
            r'\b(api|endpoint|service|interface)\b': 'api',
            r'\b(config|configuration|settings)\b': 'configuration',
            r'\b(error|exception|bug|issue)\b': 'error_handling',
            r'\b(database|db|sql|query)\b': 'database',
            r'\b(security|auth|authentication|authorization)\b': 'security',
            r'\b(performance|optimization|speed|latency)\b': 'performance',
            r'\b(deployment|deploy|release|production)\b': 'deployment',
            r'\b(monitoring|logging|metrics|observability)\b': 'monitoring',
            r'\b(integration|webhook|event|message)\b': 'integration'
        }
    
    def _create_concept_relationships(self, concepts: Set[str]):
        """Create relationships between related concepts."""
        concept_relationships = {
            'testing': ['api', 'configuration', 'error_handling'],
            'api': ['security', 'performance', 'integration'],
            'database': ['performance', 'security'],
            'deployment': ['configuration', 'monitoring', 'security'],
            'monitoring': ['performance', 'error_handling'],
            'integration': ['api', 'security', 'error_handling']
        }
        
        for concept1 in concepts:
            for concept2, related in concept_relationships.items():
                if concept1 in related and concept2 in concepts:
                    concept1_id = f"concept_{hashlib.md5(concept1.encode()).hexdigest()[:8]}"
                    concept2_id = f"concept_{hashlib.md5(concept2.encode()).hexdigest()[:8]}"
                    
                    self.graph.edges.append({
                        'source': concept1_id,
                        'target': concept2_id,
                        'type': 'related_concept',
                        'weight': 0.7
                    })
    
    def _create_item_relationships(self, item_concepts: Dict[str, Set[str]]):
        """Create relationships between items based on shared concepts."""
        items = list(item_concepts.keys())
        
        for i, item1 in enumerate(items):
            for item2 in items[i+1:]:
                concepts1 = item_concepts[item1]
                concepts2 = item_concepts[item2]
                
                shared_concepts = concepts1.intersection(concepts2)
                if shared_concepts:
                    # Calculate relationship strength
                    strength = len(shared_concepts) / len(concepts1.union(concepts2))
                    
                    if strength > 0.2:  # Minimum relationship threshold
                        self.graph.edges.append({
                            'source': item1,
                            'target': item2,
                            'type': 'related_content',
                            'weight': strength,
                            'shared_concepts': list(shared_concepts)
                        })

class KnowledgeExtractor:
    """Extracts knowledge from various sources automatically."""
    
    def __init__(self):
        self.extraction_patterns = self._initialize_extraction_patterns()
        self.source_handlers = self._initialize_source_handlers()
    
    def extract_from_file(self, file_path: Path) -> List[KnowledgeItem]:
        """Extract knowledge from a single file."""
        if not file_path.exists():
            return []
        
        # Determine source type
        source_type = self._determine_source_type(file_path)
        
        # Get appropriate handler
        handler = self.source_handlers.get(source_type)
        if not handler:
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return handler(content, str(file_path))
            
        except Exception as e:
            logger.error(f"Error extracting from {file_path}: {e}")
            return []
    
    def extract_from_directory(self, directory_path: Path, 
                             recursive: bool = True) -> List[KnowledgeItem]:
        """Extract knowledge from all files in directory."""
        knowledge_items = []
        
        if not directory_path.exists():
            return knowledge_items
        
        # Define file patterns to process
        patterns = ['*.py', '*.md', '*.rst', '*.txt', '*.json', '*.yaml', '*.yml']
        
        for pattern in patterns:
            if recursive:
                files = directory_path.rglob(pattern)
            else:
                files = directory_path.glob(pattern)
            
            for file_path in files:
                items = self.extract_from_file(file_path)
                knowledge_items.extend(items)
        
        return knowledge_items
    
    def _determine_source_type(self, file_path: Path) -> KnowledgeSource:
        """Determine the knowledge source type from file path."""
        name_lower = file_path.name.lower()
        path_lower = str(file_path).lower()
        
        if name_lower == 'readme.md' or name_lower == 'readme.rst':
            return KnowledgeSource.README_FILES
        elif file_path.suffix.lower() in ['.md', '.rst']:
            return KnowledgeSource.DOCUMENTATION_FILES
        elif file_path.suffix.lower() == '.py':
            return KnowledgeSource.DOCSTRINGS
        elif file_path.suffix.lower() in ['.json', '.yaml', '.yml']:
            return KnowledgeSource.CONFIGURATION_FILES
        elif 'archive' in path_lower:
            return KnowledgeSource.LEGACY_ARCHIVES
        elif 'test' in path_lower:
            return KnowledgeSource.TEST_DOCUMENTATION
        else:
            return KnowledgeSource.DOCUMENTATION_FILES
    
    def _initialize_source_handlers(self) -> Dict[KnowledgeSource, Any]:
        """Initialize handlers for different source types."""
        return {
            KnowledgeSource.DOCUMENTATION_FILES: self._extract_from_markdown,
            KnowledgeSource.README_FILES: self._extract_from_readme,
            KnowledgeSource.DOCSTRINGS: self._extract_from_python,
            KnowledgeSource.CONFIGURATION_FILES: self._extract_from_config,
            KnowledgeSource.LEGACY_ARCHIVES: self._extract_from_legacy
        }
    
    def _extract_from_markdown(self, content: str, file_path: str) -> List[KnowledgeItem]:
        """Extract knowledge from Markdown files."""
        items = []
        
        # Split by headers
        sections = re.split(r'^#+\s+(.+)$', content, flags=re.MULTILINE)
        
        current_title = Path(file_path).stem
        current_content = ""
        
        for i, section in enumerate(sections):
            if i == 0:  # Content before first header
                current_content = section.strip()
                continue
            
            if i % 2 == 1:  # Header
                # Save previous section if it has content
                if current_content.strip():
                    item = self._create_knowledge_item(
                        title=current_title,
                        content=current_content,
                        knowledge_type=self._determine_knowledge_type(current_title, current_content),
                        source=KnowledgeSource.DOCUMENTATION_FILES,
                        source_path=file_path
                    )
                    items.append(item)
                
                # Start new section
                current_title = section.strip()
                current_content = ""
            else:  # Content
                current_content = section.strip()
        
        # Don't forget the last section
        if current_content.strip():
            item = self._create_knowledge_item(
                title=current_title,
                content=current_content,
                knowledge_type=self._determine_knowledge_type(current_title, current_content),
                source=KnowledgeSource.DOCUMENTATION_FILES,
                source_path=file_path
            )
            items.append(item)
        
        return items
    
    def _extract_from_readme(self, content: str, file_path: str) -> List[KnowledgeItem]:
        """Extract knowledge from README files."""
        # README files are treated specially as they contain overview information
        item = self._create_knowledge_item(
            title=f"Overview - {Path(file_path).parent.name}",
            content=content,
            knowledge_type=KnowledgeType.TECHNICAL_DOCUMENTATION,
            source=KnowledgeSource.README_FILES,
            source_path=file_path
        )
        return [item]
    
    def _extract_from_python(self, content: str, file_path: str) -> List[KnowledgeItem]:
        """Extract knowledge from Python docstrings and comments."""
        items = []
        
        # Extract module docstring
        module_docstring = re.search(r'^"""(.*?)"""', content, re.DOTALL | re.MULTILINE)
        if module_docstring:
            item = self._create_knowledge_item(
                title=f"Module: {Path(file_path).stem}",
                content=module_docstring.group(1).strip(),
                knowledge_type=KnowledgeType.TECHNICAL_DOCUMENTATION,
                source=KnowledgeSource.DOCSTRINGS,
                source_path=file_path
            )
            items.append(item)
        
        # Extract function/class docstrings
        docstring_pattern = r'(?:def|class)\s+(\w+).*?:\s*"""(.*?)"""'
        docstrings = re.findall(docstring_pattern, content, re.DOTALL)
        
        for name, docstring in docstrings:
            if len(docstring.strip()) > 20:  # Meaningful docstrings only
                item = self._create_knowledge_item(
                    title=f"{name} - {Path(file_path).stem}",
                    content=docstring.strip(),
                    knowledge_type=KnowledgeType.CODE_EXAMPLES,
                    source=KnowledgeSource.DOCSTRINGS,
                    source_path=file_path
                )
                items.append(item)
        
        return items
    
    def _extract_from_config(self, content: str, file_path: str) -> List[KnowledgeItem]:
        """Extract knowledge from configuration files."""
        try:
            if file_path.endswith('.json'):
                config_data = json.loads(content)
            else:  # YAML
                import yaml
                config_data = yaml.safe_load(content)
            
            # Extract configuration documentation
            if isinstance(config_data, dict):
                config_content = self._format_config_documentation(config_data)
                
                item = self._create_knowledge_item(
                    title=f"Configuration: {Path(file_path).stem}",
                    content=config_content,
                    knowledge_type=KnowledgeType.CONFIGURATION,
                    source=KnowledgeSource.CONFIGURATION_FILES,
                    source_path=file_path
                )
                return [item]
        
        except Exception as e:
            logger.debug(f"Could not parse config file {file_path}: {e}")
        
        return []
    
    def _extract_from_legacy(self, content: str, file_path: str) -> List[KnowledgeItem]:
        """Extract knowledge from legacy archive files."""
        # Legacy files are treated as historical knowledge
        item = self._create_knowledge_item(
            title=f"Legacy: {Path(file_path).stem}",
            content=content[:1000] + "..." if len(content) > 1000 else content,  # Truncate for brevity
            knowledge_type=KnowledgeType.LEGACY_KNOWLEDGE,
            source=KnowledgeSource.LEGACY_ARCHIVES,
            source_path=file_path
        )
        return [item]
    
    def _create_knowledge_item(self, title: str, content: str, 
                             knowledge_type: KnowledgeType,
                             source: KnowledgeSource, source_path: str) -> KnowledgeItem:
        """Create a knowledge item with extracted information."""
        
        # Generate unique ID
        item_id = hashlib.md5((title + source_path).encode()).hexdigest()
        
        # Extract tags and keywords
        tags = self._extract_tags(title + " " + content)
        keywords = self._extract_keywords(content)
        
        return KnowledgeItem(
            item_id=item_id,
            title=title,
            content=content,
            knowledge_type=knowledge_type,
            source=source,
            source_path=source_path,
            tags=tags,
            keywords=keywords,
            last_updated=datetime.now()
        )
    
    def _determine_knowledge_type(self, title: str, content: str) -> KnowledgeType:
        """Determine knowledge type from title and content."""
        title_lower = title.lower()
        content_lower = content.lower()
        
        if 'api' in title_lower or 'endpoint' in title_lower:
            return KnowledgeType.API_REFERENCE
        elif 'tutorial' in title_lower or 'how to' in title_lower:
            return KnowledgeType.TUTORIAL
        elif 'troubleshoot' in title_lower or 'error' in title_lower or 'problem' in title_lower:
            return KnowledgeType.TROUBLESHOOTING
        elif 'best practice' in title_lower or 'guideline' in title_lower:
            return KnowledgeType.BEST_PRACTICES
        elif 'config' in title_lower or 'setting' in title_lower:
            return KnowledgeType.CONFIGURATION
        elif 'example' in title_lower or 'def ' in content_lower or 'class ' in content_lower:
            return KnowledgeType.CODE_EXAMPLES
        else:
            return KnowledgeType.TECHNICAL_DOCUMENTATION
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract tags from text."""
        tags = []
        
        # Technical terms
        tech_patterns = [
            r'\b(api|rest|graphql|webhook)\b',
            r'\b(database|sql|nosql|mongodb)\b',
            r'\b(security|auth|oauth|jwt)\b',
            r'\b(test|testing|unit|integration)\b',
            r'\b(performance|optimization|cache)\b',
            r'\b(deployment|docker|kubernetes)\b',
            r'\b(monitoring|logging|metrics)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text.lower())
            tags.extend(matches)
        
        return list(set(tags))  # Remove duplicates
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content."""
        # Simple keyword extraction based on frequency
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        word_freq = Counter(words)
        
        # Get most common words (excluding common words)
        stop_words = {'this', 'that', 'with', 'from', 'they', 'have', 'will', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'also', 'back', 'after', 'work', 'well', 'way', 'only', 'new', 'may', 'years', 'come', 'its', 'like', 'make', 'him', 'has', 'two', 'how', 'when', 'where', 'much', 'your', 'them', 'some', 'all', 'would', 'there', 'use'}
        
        keywords = [word for word, freq in word_freq.most_common(10) 
                   if word not in stop_words and len(word) > 3]
        
        return keywords[:5]  # Top 5 keywords
    
    def _format_config_documentation(self, config_data: Dict) -> str:
        """Format configuration data as documentation."""
        doc_lines = []
        
        def format_dict(d, indent=0):
            for key, value in d.items():
                prefix = "  " * indent
                if isinstance(value, dict):
                    doc_lines.append(f"{prefix}**{key}**:")
                    format_dict(value, indent + 1)
                elif isinstance(value, list):
                    doc_lines.append(f"{prefix}**{key}**: {len(value)} items")
                    if value and len(str(value[0])) < 50:
                        doc_lines.append(f"{prefix}  Example: {value[0]}")
                else:
                    doc_lines.append(f"{prefix}**{key}**: {value}")
        
        format_dict(config_data)
        return "\n".join(doc_lines)
    
    def _initialize_extraction_patterns(self) -> Dict[str, str]:
        """Initialize patterns for knowledge extraction."""
        return {
            'api_endpoint': r'(GET|POST|PUT|DELETE)\s+(/\S+)',
            'code_function': r'def\s+(\w+)\s*\(',
            'code_class': r'class\s+(\w+)\s*[\(:]',
            'config_key': r'"(\w+)"\s*:',
            'error_pattern': r'(Error|Exception|Failed):\s*(.+)',
            'command_usage': r'Usage:\s*(.+)'
        }

class KnowledgeManagementFramework:
    """Main framework coordinating all knowledge management capabilities."""
    
    def __init__(self):
        self.search_engine = SemanticSearchEngine()
        self.graph_builder = KnowledgeGraphBuilder()
        self.extractor = KnowledgeExtractor()
        self.knowledge_base: Dict[str, KnowledgeItem] = {}
        self.knowledge_graph: Optional[KnowledgeGraph] = None
        
    def build_knowledge_base(self, source_directories: List[str]) -> Dict[str, Any]:
        """Build comprehensive knowledge base from source directories."""
        
        logger.info(f"Building knowledge base from {len(source_directories)} directories")
        
        all_items = []
        
        # Extract knowledge from all sources
        for directory in source_directories:
            dir_path = Path(directory)
            if dir_path.exists():
                items = self.extractor.extract_from_directory(dir_path)
                all_items.extend(items)
                logger.info(f"Extracted {len(items)} items from {directory}")
        
        # Index all items in search engine
        for item in all_items:
            self.knowledge_base[item.item_id] = item
            self.search_engine.index_knowledge_item(item)
        
        # Build knowledge graph
        self.knowledge_graph = self.graph_builder.build_graph_from_knowledge_items(all_items)
        
        # Generate statistics
        stats = self._generate_knowledge_base_stats(all_items)
        
        return {
            "total_items": len(all_items),
            "knowledge_types": stats["knowledge_types"],
            "knowledge_sources": stats["knowledge_sources"], 
            "graph_nodes": len(self.knowledge_graph.nodes),
            "graph_edges": len(self.knowledge_graph.edges),
            "concepts_identified": len(self.knowledge_graph.concepts),
            "build_timestamp": datetime.now().isoformat()
        }
    
    def search_knowledge(self, query: str, context: KnowledgeContext = None, 
                        max_results: int = 10) -> Dict[str, Any]:
        """Search knowledge base with context awareness."""
        
        # Perform search
        results = self.search_engine.search(query, context, max_results)
        
        # Get suggestions
        suggestions = self.search_engine.suggest_related_queries(query)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.item.title,
                "content": result.item.content[:200] + "..." if len(result.item.content) > 200 else result.item.content,
                "type": result.item.knowledge_type.value,
                "source": result.item.source.value,
                "relevance": result.relevance.value,
                "score": result.relevance_score,
                "reasons": result.match_reasons,
                "snippets": result.context_snippets,
                "path": result.item.source_path
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "total_results": len(results),
            "suggestions": suggestions,
            "search_timestamp": datetime.now().isoformat()
        }
    
    def get_related_knowledge(self, item_id: str, max_related: int = 5) -> List[Dict[str, Any]]:
        """Get knowledge items related to a specific item."""
        if not self.knowledge_graph or item_id not in self.knowledge_base:
            return []
        
        related_items = []
        
        # Find related items through graph edges
        for edge in self.knowledge_graph.edges:
            if edge['source'] == item_id and edge['type'] == 'related_content':
                target_item = self.knowledge_base.get(edge['target'])
                if target_item:
                    related_items.append({
                        "title": target_item.title,
                        "type": target_item.knowledge_type.value,
                        "relationship_strength": edge['weight'],
                        "shared_concepts": edge.get('shared_concepts', []),
                        "item_id": target_item.item_id
                    })
        
        # Sort by relationship strength
        related_items.sort(key=lambda x: x['relationship_strength'], reverse=True)
        
        return related_items[:max_related]
    
    def _generate_knowledge_base_stats(self, items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Generate statistics about the knowledge base."""
        
        knowledge_types = Counter(item.knowledge_type.value for item in items)
        knowledge_sources = Counter(item.source.value for item in items)
        
        return {
            "knowledge_types": dict(knowledge_types),
            "knowledge_sources": dict(knowledge_sources),
            "avg_content_length": sum(len(item.content) for item in items) / len(items) if items else 0,
            "total_tags": sum(len(item.tags) for item in items),
            "unique_sources": len(set(item.source_path for item in items))
        }

# Global knowledge management framework instance
_knowledge_framework = KnowledgeManagementFramework()

def get_knowledge_management_framework() -> KnowledgeManagementFramework:
    """Get the global knowledge management framework instance."""
    return _knowledge_framework

def build_knowledge_base(directories: List[str]) -> Dict[str, Any]:
    """High-level function to build knowledge base."""
    framework = get_knowledge_management_framework()
    return framework.build_knowledge_base(directories)

def search_knowledge_base(query: str, context_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    """High-level function to search knowledge base."""
    framework = get_knowledge_management_framework()
    
    context = None
    if context_dict:
        context = KnowledgeContext(**context_dict)
    
    return framework.search_knowledge(query, context)