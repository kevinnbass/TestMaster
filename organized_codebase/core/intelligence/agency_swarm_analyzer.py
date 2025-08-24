"""
Agency-Swarm Documentation Analysis Module
Extracts and analyzes documentation patterns from Agency-Swarm framework
"""

import os
import re
import json
import textstat
from typing import Dict, List, Any, Optional
from pathlib import Path


class AgencySwarmAnalyzer:
    """Analyzes Agency-Swarm documentation structure and patterns"""
    
    def __init__(self, docs_path: str):
        self.docs_path = Path(docs_path)
        self.readability_threshold = 15
    
    def process_inline_code(self, code: str) -> str:
        """Process inline code for readability analysis"""
        if re.search(r'\w+\.\w+', code):
            code = code.replace('.', ' ')
        if '[' in code or ']' in code:
            code = code.replace('[', ' ').replace(']', ' ')
        return code
    
    def clean_markdown(self, text: str) -> str:
        """Clean markdown content for analysis"""
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # Process inline code
        def inline_replacer(match):
            return self.process_inline_code(match.group(1))
        text = re.sub(r'`([^`]+)`', inline_replacer, text)
        
        # Clean markdown formatting
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'[*_~]', '', text)
        
        return text
    
    def analyze_readability(self, content: str) -> Dict[str, float]:
        """Compute readability metrics for content"""
        plain_text = self.clean_markdown(content)
        return {
            "flesch_kincaid": textstat.flesch_kincaid_grade(plain_text),
            "smog_index": textstat.smog_index(plain_text),
            "ari_index": textstat.automated_readability_index(plain_text),
            "coleman_liau": textstat.coleman_liau_index(plain_text)
        }
    
    def extract_navigation_structure(self, docs_json: Dict) -> Dict[str, Any]:
        """Extract navigation patterns from docs.json"""
        nav_structure = {
            "theme": docs_json.get("theme"),
            "colors": docs_json.get("colors", {}),
            "tabs": [],
            "groups": [],
            "anchors": []
        }
        
        navigation = docs_json.get("navigation", {})
        if "tabs" in navigation:
            for tab in navigation["tabs"]:
                tab_info = {
                    "name": tab.get("tab"),
                    "groups": []
                }
                
                if "groups" in tab:
                    for group in tab["groups"]:
                        group_info = {
                            "name": group.get("group"),
                            "icon": group.get("icon"),
                            "pages": group.get("pages", [])
                        }
                        tab_info["groups"].append(group_info)
                
                nav_structure["tabs"].append(tab_info)
        
        return nav_structure
    
    def analyze_mdx_patterns(self, file_path: Path) -> Dict[str, Any]:
        """Analyze MDX file patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            patterns = {
                "frontmatter": self.extract_frontmatter(content),
                "components": self.extract_components(content),
                "code_blocks": self.extract_code_blocks(content),
                "links": self.extract_links(content),
                "readability": self.analyze_readability(content)
            }
            
            return patterns
        except Exception as e:
            return {"error": str(e)}
    
    def extract_frontmatter(self, content: str) -> Dict[str, Any]:
        """Extract YAML frontmatter from MDX"""
        match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        if match:
            frontmatter = {}
            for line in match.group(1).split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    frontmatter[key.strip()] = value.strip().strip('"')
            return frontmatter
        return {}
    
    def extract_components(self, content: str) -> List[str]:
        """Extract React components used in MDX"""
        components = set()
        
        # Find JSX tags
        jsx_pattern = r'<([A-Z][a-zA-Z]*)'
        matches = re.findall(jsx_pattern, content)
        components.update(matches)
        
        return list(components)
    
    def extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks with language info"""
        code_blocks = []
        pattern = r'```(\w+)?\n(.*?)\n```'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2)
            code_blocks.append({
                "language": language,
                "code": code.strip(),
                "length": len(code.strip().split('\n'))
            })
        
        return code_blocks
    
    def extract_links(self, content: str) -> List[Dict[str, str]]:
        """Extract markdown links"""
        links = []
        pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        
        for match in re.finditer(pattern, content):
            links.append({
                "text": match.group(1),
                "url": match.group(2),
                "is_internal": not match.group(2).startswith(('http', 'mailto'))
            })
        
        return links
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        report = {
            "structure_analysis": {},
            "readability_analysis": {},
            "component_usage": {},
            "content_patterns": {}
        }
        
        # Analyze docs.json if exists
        docs_json_path = self.docs_path / "docs.json"
        if docs_json_path.exists():
            with open(docs_json_path, 'r') as f:
                docs_config = json.load(f)
                report["structure_analysis"] = self.extract_navigation_structure(docs_config)
        
        # Analyze all MDX files
        mdx_files = list(self.docs_path.glob("**/*.mdx"))
        for file_path in mdx_files:
            relative_path = str(file_path.relative_to(self.docs_path))
            analysis = self.analyze_mdx_patterns(file_path)
            report["content_patterns"][relative_path] = analysis
        
        return report