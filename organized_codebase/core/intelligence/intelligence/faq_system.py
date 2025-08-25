"""
FAQ System Module
Creates and manages FAQ systems with intelligent categorization and search
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import json
from datetime import datetime
from pathlib import Path


class FAQCategory(Enum):
    """FAQ category types"""
    GETTING_STARTED = "getting_started"
    INSTALLATION = "installation"
    CONFIGURATION = "configuration"
    TROUBLESHOOTING = "troubleshooting"
    API_USAGE = "api_usage"
    BEST_PRACTICES = "best_practices"
    INTEGRATIONS = "integrations"
    DEPLOYMENT = "deployment"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class FAQItem:
    """Represents a single FAQ item"""
    id: str
    question: str
    answer: str
    category: FAQCategory
    tags: List[str] = field(default_factory=list)
    code_examples: List[Dict[str, str]] = field(default_factory=list)
    related_links: List[Dict[str, str]] = field(default_factory=list)
    difficulty: str = "beginner"  # beginner, intermediate, advanced
    votes: int = 0
    views: int = 0
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FAQSection:
    """Represents a section of related FAQs"""
    id: str
    title: str
    description: str
    icon: str = ""
    faqs: List[str] = field(default_factory=list)  # FAQ IDs
    order: int = 0


class FAQSystem:
    """Manages FAQ creation, organization, and search"""
    
    def __init__(self):
        self.faqs = {}
        self.sections = {}
        self.categories = {category.value: category for category in FAQCategory}
        self.search_index = {}
        self.auto_suggestions = []
    
    def create_faq_from_content(self, content: str, category: FAQCategory) -> List[FAQItem]:
        """Extract FAQ items from content"""
        faqs = []
        
        # Parse different FAQ formats
        accordion_faqs = self.parse_accordion_format(content, category)
        faqs.extend(accordion_faqs)
        
        qa_faqs = self.parse_qa_format(content, category)
        faqs.extend(qa_faqs)
        
        # Add to system
        for faq in faqs:
            self.add_faq(faq)
        
        return faqs
    
    def parse_accordion_format(self, content: str, category: FAQCategory) -> List[FAQItem]:
        """Parse accordion-style FAQ format"""
        faqs = []
        
        # Look for Accordion components
        accordion_pattern = r'<Accordion\s+title="([^"]+)"[^>]*>(.*?)</Accordion>'
        matches = re.findall(accordion_pattern, content, re.DOTALL)
        
        for i, (title, answer_content) in enumerate(matches):
            faq_id = self.generate_faq_id(title)
            
            # Clean up answer content
            answer = self.clean_faq_content(answer_content)
            
            # Extract code examples
            code_examples = self.extract_code_examples(answer_content)
            
            # Extract tags from question
            tags = self.extract_tags(title + " " + answer)
            
            faq = FAQItem(
                id=faq_id,
                question=title.strip(),
                answer=answer,
                category=category,
                tags=tags,
                code_examples=code_examples,
                difficulty=self.determine_difficulty(title, answer)
            )
            
            faqs.append(faq)
        
        return faqs
    
    def parse_qa_format(self, content: str, category: FAQCategory) -> List[FAQItem]:
        """Parse Q&A format FAQs"""
        faqs = []
        lines = content.split('\n')
        
        current_question = ""
        current_answer = []
        in_answer = False
        
        for line in lines:
            line = line.strip()
            
            # Check for question indicators
            if self.is_question_line(line):
                # Save previous FAQ if exists
                if current_question and current_answer:
                    faq = self.create_faq_from_qa(
                        current_question, '\n'.join(current_answer), category
                    )
                    faqs.append(faq)
                
                current_question = self.clean_question(line)
                current_answer = []
                in_answer = True
                
            elif in_answer and line:
                # Check if this starts a new section
                if line.startswith('#') or self.is_question_line(line):
                    in_answer = False
                    continue
                
                current_answer.append(line)
        
        # Add final FAQ
        if current_question and current_answer:
            faq = self.create_faq_from_qa(
                current_question, '\n'.join(current_answer), category
            )
            faqs.append(faq)
        
        return faqs
    
    def is_question_line(self, line: str) -> bool:
        """Check if line looks like a question"""
        question_indicators = [
            line.endswith('?'),
            line.lower().startswith('how'),
            line.lower().startswith('what'),
            line.lower().startswith('why'),
            line.lower().startswith('when'),
            line.lower().startswith('where'),
            line.lower().startswith('can i'),
            line.lower().startswith('do i'),
            'question:' in line.lower()
        ]
        
        return any(question_indicators)
    
    def clean_question(self, line: str) -> str:
        """Clean question text"""
        # Remove common prefixes
        prefixes = ['q:', 'question:', 'faq:', '**', '*']
        clean_line = line
        
        for prefix in prefixes:
            if clean_line.lower().startswith(prefix.lower()):
                clean_line = clean_line[len(prefix):].strip()
        
        # Remove markdown formatting
        clean_line = re.sub(r'[*_`]', '', clean_line)
        
        return clean_line
    
    def create_faq_from_qa(self, question: str, answer: str, category: FAQCategory) -> FAQItem:
        """Create FAQ from question and answer text"""
        faq_id = self.generate_faq_id(question)
        
        return FAQItem(
            id=faq_id,
            question=question,
            answer=self.clean_faq_content(answer),
            category=category,
            tags=self.extract_tags(question + " " + answer),
            code_examples=self.extract_code_examples(answer),
            difficulty=self.determine_difficulty(question, answer)
        )
    
    def clean_faq_content(self, content: str) -> str:
        """Clean FAQ content"""
        # Remove markdown artifacts
        clean_content = re.sub(r'```[\w]*\n(.*?)\n```', r'```\n\1\n```', content, flags=re.DOTALL)
        
        # Clean up whitespace
        lines = [line.strip() for line in content.split('\n')]
        clean_content = '\n'.join(line for line in lines if line)
        
        return clean_content.strip()
    
    def extract_code_examples(self, content: str) -> List[Dict[str, str]]:
        """Extract code examples from content"""
        examples = []
        
        # Find code blocks
        code_pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.findall(code_pattern, content, re.DOTALL)
        
        for language, code in matches:
            examples.append({
                "language": language or "text",
                "code": code.strip(),
                "title": f"{language.title()} Example" if language else "Code Example"
            })
        
        # Find inline code
        inline_pattern = r'`([^`]+)`'
        inline_matches = re.findall(inline_pattern, content)
        
        for inline_code in inline_matches:
            if len(inline_code) > 20:  # Longer inline code might be an example
                examples.append({
                    "language": "text",
                    "code": inline_code,
                    "title": "Code Snippet"
                })
        
        return examples
    
    def extract_tags(self, content: str) -> List[str]:
        """Extract relevant tags from content"""
        content_lower = content.lower()
        
        # Common technical tags
        tag_patterns = {
            "api": ["api", "endpoint", "request", "response"],
            "authentication": ["auth", "token", "key", "login", "oauth"],
            "configuration": ["config", "setup", "install", "configure"],
            "database": ["database", "db", "sql", "query"],
            "error": ["error", "exception", "bug", "issue", "problem"],
            "integration": ["integrate", "connect", "webhook", "plugin"],
            "performance": ["performance", "slow", "optimize", "cache"],
            "security": ["security", "secure", "ssl", "https", "encrypt"]
        }
        
        tags = []
        for tag, keywords in tag_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                tags.append(tag)
        
        # Extract version numbers
        version_pattern = r'v?\d+\.\d+(?:\.\d+)?'
        versions = re.findall(version_pattern, content)
        tags.extend(f"v{v}" for v in versions)
        
        return tags[:5]  # Limit to 5 tags
    
    def determine_difficulty(self, question: str, answer: str) -> str:
        """Determine difficulty level based on content"""
        content = f"{question} {answer}".lower()
        
        # Advanced indicators
        advanced_keywords = [
            "advanced", "complex", "enterprise", "production", "scale",
            "architecture", "optimization", "custom", "extend"
        ]
        
        # Beginner indicators
        beginner_keywords = [
            "getting started", "how to", "basic", "simple", "first",
            "install", "setup", "quick", "easy"
        ]
        
        # Intermediate indicators
        intermediate_keywords = [
            "configure", "integrate", "customize", "deploy", "troubleshoot"
        ]
        
        if any(keyword in content for keyword in advanced_keywords):
            return "advanced"
        elif any(keyword in content for keyword in beginner_keywords):
            return "beginner"
        elif any(keyword in content for keyword in intermediate_keywords):
            return "intermediate"
        
        # Default based on answer length and code examples
        if len(answer) > 500 or "```" in answer:
            return "intermediate"
        
        return "beginner"
    
    def add_faq(self, faq: FAQItem) -> None:
        """Add FAQ to system"""
        self.faqs[faq.id] = faq
        self.update_search_index(faq)
    
    def update_search_index(self, faq: FAQItem) -> None:
        """Update search index with FAQ content"""
        # Create searchable text
        searchable_text = f"{faq.question} {faq.answer} {' '.join(faq.tags)}"
        words = re.findall(r'\w+', searchable_text.lower())
        
        for word in words:
            if len(word) > 2:  # Ignore very short words
                if word not in self.search_index:
                    self.search_index[word] = set()
                self.search_index[word].add(faq.id)
    
    def search_faqs(self, query: str, category: Optional[FAQCategory] = None,
                   limit: int = 10) -> List[FAQItem]:
        """Search FAQs by query"""
        query_words = re.findall(r'\w+', query.lower())
        
        # Find matching FAQ IDs
        matching_ids = set()
        for word in query_words:
            if word in self.search_index:
                if not matching_ids:
                    matching_ids = self.search_index[word].copy()
                else:
                    matching_ids &= self.search_index[word]
        
        # If no exact matches, try partial matches
        if not matching_ids:
            for word in query_words:
                for index_word, faq_ids in self.search_index.items():
                    if word in index_word or index_word in word:
                        matching_ids.update(faq_ids)
        
        # Filter by category if specified
        results = []
        for faq_id in matching_ids:
            if faq_id in self.faqs:
                faq = self.faqs[faq_id]
                if category is None or faq.category == category:
                    results.append(faq)
        
        # Sort by relevance (basic scoring)
        def relevance_score(faq: FAQItem) -> float:
            score = 0
            question_lower = faq.question.lower()
            answer_lower = faq.answer.lower()
            
            for word in query_words:
                if word in question_lower:
                    score += 3  # Question matches are more important
                if word in answer_lower:
                    score += 1
                if word in [tag.lower() for tag in faq.tags]:
                    score += 2
            
            # Boost popular FAQs
            score += faq.votes * 0.1
            
            return score
        
        results.sort(key=relevance_score, reverse=True)
        return results[:limit]
    
    def create_faq_section(self, section_id: str, title: str, description: str,
                          faq_ids: List[str], icon: str = "", order: int = 0) -> FAQSection:
        """Create an FAQ section"""
        section = FAQSection(
            id=section_id,
            title=title,
            description=description,
            icon=icon,
            faqs=faq_ids,
            order=order
        )
        
        self.sections[section_id] = section
        return section
    
    def auto_categorize_faqs(self) -> None:
        """Automatically categorize FAQs based on content"""
        category_keywords = {
            FAQCategory.GETTING_STARTED: ["start", "begin", "first", "new", "intro"],
            FAQCategory.INSTALLATION: ["install", "setup", "download", "requirements"],
            FAQCategory.CONFIGURATION: ["config", "setting", "option", "parameter"],
            FAQCategory.TROUBLESHOOTING: ["error", "problem", "issue", "fix", "debug"],
            FAQCategory.API_USAGE: ["api", "endpoint", "request", "response", "curl"],
            FAQCategory.DEPLOYMENT: ["deploy", "production", "server", "hosting"],
            FAQCategory.SECURITY: ["security", "auth", "token", "permission", "ssl"],
            FAQCategory.PERFORMANCE: ["slow", "performance", "optimize", "cache"]
        }
        
        for faq in self.faqs.values():
            content = f"{faq.question} {faq.answer}".lower()
            
            best_category = faq.category
            best_score = 0
            
            for category, keywords in category_keywords.items():
                score = sum(1 for keyword in keywords if keyword in content)
                if score > best_score:
                    best_score = score
                    best_category = category
            
            faq.category = best_category
    
    def generate_faq_page(self, format: str = "markdown") -> str:
        """Generate complete FAQ page"""
        if format.lower() == "markdown":
            return self.generate_markdown_faq()
        elif format.lower() == "html":
            return self.generate_html_faq()
        elif format.lower() == "json":
            return self.generate_json_faq()
        
        return ""
    
    def generate_markdown_faq(self) -> str:
        """Generate markdown FAQ page"""
        content = []
        
        content.append("# Frequently Asked Questions")
        content.append("")
        content.append("Find answers to common questions below.")
        content.append("")
        
        # Group FAQs by category
        by_category = {}
        for faq in self.faqs.values():
            if faq.category not in by_category:
                by_category[faq.category] = []
            by_category[faq.category].append(faq)
        
        # Generate accordion groups for each category
        for category, category_faqs in by_category.items():
            category_title = category.value.replace('_', ' ').title()
            content.append(f"## {category_title}")
            content.append("")
            content.append("<AccordionGroup defaultOpen={true}>")
            content.append("")
            
            for faq in category_faqs:
                content.append(f'<Accordion title="{faq.question}" icon="question">')
                content.append(faq.answer)
                
                # Add code examples
                for example in faq.code_examples:
                    content.append("")
                    content.append(f"```{example['language']}")
                    content.append(example['code'])
                    content.append("```")
                
                content.append("</Accordion>")
                content.append("")
            
            content.append("</AccordionGroup>")
            content.append("")
        
        return '\n'.join(content)
    
    def generate_html_faq(self) -> str:
        """Generate HTML FAQ page"""
        # This would generate a full HTML FAQ page
        # For brevity, returning a simple structure
        return "<div class='faq-container'>FAQ content would go here</div>"
    
    def generate_json_faq(self) -> str:
        """Generate JSON representation of FAQs"""
        faq_data = {
            "faqs": [],
            "sections": [],
            "categories": list(self.categories.keys())
        }
        
        for faq in self.faqs.values():
            faq_data["faqs"].append({
                "id": faq.id,
                "question": faq.question,
                "answer": faq.answer,
                "category": faq.category.value,
                "tags": faq.tags,
                "code_examples": faq.code_examples,
                "difficulty": faq.difficulty,
                "votes": faq.votes,
                "views": faq.views
            })
        
        for section in self.sections.values():
            faq_data["sections"].append({
                "id": section.id,
                "title": section.title,
                "description": section.description,
                "icon": section.icon,
                "faqs": section.faqs,
                "order": section.order
            })
        
        return json.dumps(faq_data, indent=2)
    
    def generate_faq_id(self, question: str) -> str:
        """Generate unique ID for FAQ"""
        clean_question = re.sub(r'[^a-zA-Z0-9\s]', '', question)
        base_id = clean_question.lower().replace(' ', '_')[:50]
        
        # Ensure uniqueness
        counter = 1
        faq_id = base_id
        while faq_id in self.faqs:
            faq_id = f"{base_id}_{counter}"
            counter += 1
        
        return faq_id