"""
Rich Markdown Generator

Generates beautiful, feature-rich markdown documentation with advanced formatting.
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarkdownSection:
    """Represents a markdown document section."""
    level: int  # 1-6 for heading levels
    title: str
    content: str
    subsections: List['MarkdownSection'] = None
    

class MarkdownGenerator:
    """
    Advanced markdown generation with rich formatting capabilities.
    Supports tables, code blocks, diagrams, badges, and more.
    """
    
    def __init__(self):
        """Initialize the markdown generator."""
        self.sections = []
        self.toc_enabled = True
        self.badges = []
        logger.info("Markdown Generator initialized")
        
    def add_badge(self, label: str, message: str, color: str = "blue") -> str:
        """
        Add a shield.io style badge.
        
        Args:
            label: Badge label
            message: Badge message
            color: Badge color
            
        Returns:
            Badge markdown
        """
        badge = f"![{label}](https://img.shields.io/badge/{label}-{message}-{color})"
        self.badges.append(badge)
        return badge
        
    def generate_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """
        Generate a markdown table.
        
        Args:
            headers: Table headers
            rows: Table data rows
            
        Returns:
            Formatted markdown table
        """
        table = []
        
        # Headers
        table.append("| " + " | ".join(headers) + " |")
        table.append("|" + "|".join([" --- " for _ in headers]) + "|")
        
        # Rows
        for row in rows:
            table.append("| " + " | ".join(str(cell) for cell in row) + " |")
            
        return "\n".join(table)
        
    def generate_code_block(self, code: str, language: str = "python") -> str:
        """
        Generate a syntax-highlighted code block.
        
        Args:
            code: Code content
            language: Programming language
            
        Returns:
            Formatted code block
        """
        return f"```{language}\n{code}\n```"
        
    def generate_mermaid_diagram(self, diagram_type: str, content: str) -> str:
        """
        Generate a Mermaid diagram block.
        
        Args:
            diagram_type: Type of diagram (graph, sequence, etc.)
            content: Diagram content
            
        Returns:
            Mermaid diagram markdown
        """
        return f"```mermaid\n{diagram_type}\n{content}\n```"
        
    def generate_toc(self, sections: List[MarkdownSection]) -> str:
        """
        Generate table of contents.
        
        Args:
            sections: Document sections
            
        Returns:
            TOC markdown
        """
        toc = ["## Table of Contents\n"]
        
        def add_section(section: MarkdownSection, indent: int = 0):
            link = section.title.lower().replace(" ", "-")
            toc.append(f"{'  ' * indent}- [{section.title}](#{link})")
            if section.subsections:
                for subsection in section.subsections:
                    add_section(subsection, indent + 1)
                    
        for section in sections:
            add_section(section)
            
        return "\n".join(toc)
        
    def generate_api_doc(self, 
                        endpoint: str,
                        method: str,
                        description: str,
                        params: Dict[str, str],
                        response: Dict[str, Any]) -> str:
        """
        Generate API endpoint documentation.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            description: Endpoint description
            params: Parameters
            response: Response example
            
        Returns:
            API documentation markdown
        """
        doc = [
            f"### {method} {endpoint}",
            "",
            description,
            ""
        ]
        
        if params:
            doc.extend([
                "**Parameters:**",
                "",
                self.generate_table(
                    ["Name", "Type", "Required", "Description"],
                    [[k, v.get('type', 'string'), v.get('required', 'No'), v.get('description', '')] 
                     for k, v in params.items()]
                ),
                ""
            ])
            
        if response:
            doc.extend([
                "**Response:**",
                "",
                self.generate_code_block(str(response), "json"),
                ""
            ])
            
        return "\n".join(doc)
        
    def generate_class_doc(self, 
                          class_name: str,
                          description: str,
                          methods: List[Dict[str, str]],
                          attributes: List[Dict[str, str]]) -> str:
        """
        Generate class documentation.
        
        Args:
            class_name: Name of class
            description: Class description
            methods: List of methods
            attributes: List of attributes
            
        Returns:
            Class documentation markdown
        """
        doc = [
            f"## Class: `{class_name}`",
            "",
            description,
            ""
        ]
        
        if attributes:
            doc.extend([
                "### Attributes",
                "",
                self.generate_table(
                    ["Name", "Type", "Description"],
                    [[a['name'], a.get('type', 'Any'), a.get('description', '')] 
                     for a in attributes]
                ),
                ""
            ])
            
        if methods:
            doc.extend(["### Methods", ""])
            for method in methods:
                doc.extend([
                    f"#### `{method['name']}({method.get('params', '')})`",
                    "",
                    method.get('description', ''),
                    ""
                ])
                
        return "\n".join(doc)
        
    def add_section(self, title: str, content: str, level: int = 2) -> None:
        """
        Add a section to the document.
        
        Args:
            title: Section title
            content: Section content
            level: Heading level (1-6)
        """
        section = MarkdownSection(level, title, content)
        self.sections.append(section)
        
    def generate_document(self, 
                         title: str,
                         description: str = "",
                         include_toc: bool = True) -> str:
        """
        Generate complete markdown document.
        
        Args:
            title: Document title
            description: Document description
            include_toc: Include table of contents
            
        Returns:
            Complete markdown document
        """
        doc = [f"# {title}", ""]
        
        # Add badges
        if self.badges:
            doc.append(" ".join(self.badges))
            doc.append("")
            
        # Add description
        if description:
            doc.extend([description, ""])
            
        # Add TOC
        if include_toc and self.toc_enabled and self.sections:
            doc.extend([self.generate_toc(self.sections), ""])
            
        # Add sections
        for section in self.sections:
            doc.extend([
                "#" * section.level + " " + section.title,
                "",
                section.content,
                ""
            ])
            
        # Add footer
        doc.extend([
            "---",
            f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ])
        
        return "\n".join(doc)
        
    def generate_readme(self, 
                       project_name: str,
                       description: str,
                       installation: str,
                       usage: str,
                       features: List[str]) -> str:
        """
        Generate a README.md file.
        
        Args:
            project_name: Name of project
            description: Project description
            installation: Installation instructions
            usage: Usage examples
            features: List of features
            
        Returns:
            README markdown
        """
        self.add_badge("Python", "3.8+", "blue")
        self.add_badge("License", "MIT", "green")
        self.add_badge("Build", "Passing", "brightgreen")
        
        self.add_section("Features", "\n".join(f"- {f}" for f in features))
        self.add_section("Installation", self.generate_code_block(installation, "bash"))
        self.add_section("Usage", usage)
        self.add_section("Contributing", "Contributions are welcome! Please read our contributing guidelines.")
        self.add_section("License", "This project is licensed under the MIT License.")
        
        return self.generate_document(project_name, description)
        
    def export_to_file(self, content: str, file_path: str) -> None:
        """
        Export markdown to file.
        
        Args:
            content: Markdown content
            file_path: Output file path
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Exported markdown to {file_path}")