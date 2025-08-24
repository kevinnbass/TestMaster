"""
MDX Documentation Generator Module
Creates MDX files with React components and advanced formatting
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MDXComponent:
    """Represents an MDX component configuration"""
    name: str
    props: Dict[str, Any]
    content: str = ""
    self_closing: bool = False


class MDXGenerator:
    """Generates MDX documentation with React components"""
    
    def __init__(self):
        self.components = {
            "Accordion": {"props": ["title", "icon"], "container": True},
            "AccordionGroup": {"props": ["defaultOpen"], "container": True},
            "Card": {"props": ["title", "icon", "href"], "container": True},
            "CardGroup": {"props": ["cols"], "container": True},
            "Tabs": {"props": ["defaultValue"], "container": True},
            "Tab": {"props": ["title"], "container": True},
            "Warning": {"props": ["title"], "container": True},
            "Info": {"props": ["title"], "container": True},
            "Tip": {"props": ["title"], "container": True}
        }
    
    def create_frontmatter(self, title: str, description: str = "", 
                          icon: str = "", additional_props: Dict = None) -> str:
        """Generate YAML frontmatter for MDX"""
        frontmatter = ["---"]
        frontmatter.append(f'title: "{title}"')
        
        if description:
            frontmatter.append(f'description: "{description}"')
        
        if icon:
            frontmatter.append(f'icon: "{icon}"')
        
        if additional_props:
            for key, value in additional_props.items():
                if isinstance(value, str):
                    frontmatter.append(f'{key}: "{value}"')
                else:
                    frontmatter.append(f'{key}: {value}')
        
        frontmatter.append("---")
        frontmatter.append("")
        
        return "\n".join(frontmatter)
    
    def create_component(self, component: MDXComponent) -> str:
        """Generate MDX component markup"""
        if component.name not in self.components:
            raise ValueError(f"Unknown component: {component.name}")
        
        if component.self_closing:
            props_str = self.format_props(component.props)
            return f"<{component.name}{props_str} />"
        
        props_str = self.format_props(component.props)
        opening_tag = f"<{component.name}{props_str}>"
        closing_tag = f"</{component.name}>"
        
        return f"{opening_tag}\n{component.content}\n{closing_tag}"
    
    def format_props(self, props: Dict[str, Any]) -> str:
        """Format component properties"""
        if not props:
            return ""
        
        prop_strings = []
        for key, value in props.items():
            if isinstance(value, str):
                prop_strings.append(f'{key}="{value}"')
            elif isinstance(value, bool):
                if value:
                    prop_strings.append(f'{key}={{true}}')
                else:
                    prop_strings.append(f'{key}={{false}}')
            else:
                prop_strings.append(f'{key}={{{value}}}')
        
        return " " + " ".join(prop_strings) if prop_strings else ""
    
    def create_accordion_group(self, accordions: List[Dict[str, str]], 
                              default_open: bool = True) -> str:
        """Create an accordion group with multiple accordions"""
        accordion_content = []
        
        for accordion in accordions:
            accordion_comp = MDXComponent(
                name="Accordion",
                props={
                    "title": accordion["title"],
                    "icon": accordion.get("icon", "")
                },
                content=accordion["content"]
            )
            accordion_content.append(self.create_component(accordion_comp))
        
        group = MDXComponent(
            name="AccordionGroup",
            props={"defaultOpen": default_open},
            content="\n\n".join(accordion_content)
        )
        
        return self.create_component(group)
    
    def create_card_group(self, cards: List[Dict[str, str]], cols: int = 2) -> str:
        """Create a card group with multiple cards"""
        card_content = []
        
        for card in cards:
            card_comp = MDXComponent(
                name="Card",
                props={
                    "title": card["title"],
                    "icon": card.get("icon", ""),
                    "href": card.get("href", "")
                },
                content=card.get("content", "")
            )
            card_content.append(self.create_component(card_comp))
        
        group = MDXComponent(
            name="CardGroup",
            props={"cols": cols},
            content="\n".join(card_content)
        )
        
        return self.create_component(group)
    
    def create_code_block(self, code: str, language: str = "python", 
                         title: str = "") -> str:
        """Create a formatted code block"""
        if title:
            return f"```{language} {title}\n{code}\n```"
        return f"```{language}\n{code}\n```"
    
    def create_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Create a markdown table"""
        if not headers or not rows:
            return ""
        
        # Header row
        header_row = "| " + " | ".join(headers) + " |"
        
        # Separator row
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        
        # Data rows
        data_rows = []
        for row in rows:
            if len(row) != len(headers):
                row.extend([""] * (len(headers) - len(row)))
            data_rows.append("| " + " | ".join(row) + " |")
        
        return "\n".join([header_row, separator] + data_rows)
    
    def create_callout(self, content: str, type: str = "info", 
                      title: str = "") -> str:
        """Create a callout/alert component"""
        component_map = {
            "info": "Info",
            "warning": "Warning", 
            "tip": "Tip"
        }
        
        component_name = component_map.get(type, "Info")
        props = {"title": title} if title else {}
        
        component = MDXComponent(
            name=component_name,
            props=props,
            content=content
        )
        
        return self.create_component(component)
    
    def generate_mdx_file(self, title: str, content_sections: List[str],
                         description: str = "", icon: str = "",
                         additional_frontmatter: Dict = None) -> str:
        """Generate complete MDX file content"""
        mdx_content = []
        
        # Add frontmatter
        frontmatter = self.create_frontmatter(
            title, description, icon, additional_frontmatter
        )
        mdx_content.append(frontmatter)
        
        # Add content sections
        for section in content_sections:
            mdx_content.append(section)
            mdx_content.append("")  # Add spacing
        
        return "\n".join(mdx_content)
    
    def save_mdx_file(self, content: str, file_path: Path) -> bool:
        """Save MDX content to file"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error saving MDX file: {e}")
            return False