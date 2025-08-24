"""
AGENT X STEELCLAD: Atomic Component Registry
Central registry for all atomized dashboard components
"""

from pathlib import Path
from typing import Dict, List, Optional
import json

class AtomicComponentRegistry:
    """
    Central registry for all atomic dashboard components.
    Manages CSS, JS, and Python atomic modules.
    """
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent  # web/dashboard_modules
        self.atoms_path = self.base_path / "specialized" / "atoms"
        self.templates_path = self.base_path / "templates"
        
        # Component registry
        self.registry = {
            "css": {},
            "js": {},
            "python": {},
            "html_cores": {}
        }
        
        self._scan_components()
    
    def _scan_components(self):
        """Scan and register all atomic components"""
        
        # CSS Components
        css_path = self.templates_path / "css"
        if css_path.exists():
            for css_file in css_path.glob("*.css"):
                self.registry["css"][css_file.stem] = {
                    "path": str(css_file),
                    "url": f"/static/css/{css_file.name}",
                    "type": "stylesheet",
                    "atomized": True
                }
        
        # JavaScript Components
        js_path = self.templates_path / "js"
        if js_path.exists():
            for js_file in js_path.glob("*.js"):
                self.registry["js"][js_file.stem] = {
                    "path": str(js_file),
                    "url": f"/static/js/{js_file.name}",
                    "type": "script",
                    "atomized": True
                }
        
        # Python Atomic Modules
        for py_file in self.atoms_path.glob("*.py"):
            if py_file.stem != "__init__" and py_file.stem != "atomic_registry":
                self.registry["python"][py_file.stem] = {
                    "path": str(py_file),
                    "module": f"specialized.atoms.{py_file.stem}",
                    "type": "python_module",
                    "atomized": True
                }
        
        # HTML Core Templates
        for html_file in self.templates_path.glob("*_core.html"):
            self.registry["html_cores"][html_file.stem] = {
                "path": str(html_file),
                "template": html_file.name,
                "type": "html_core",
                "atomized": True
            }
    
    def get_component(self, component_type: str, name: str) -> Optional[Dict]:
        """Get a specific atomic component"""
        return self.registry.get(component_type, {}).get(name)
    
    def get_dashboard_assets(self, dashboard_name: str) -> Dict[str, List]:
        """Get all assets needed for a specific dashboard"""
        assets = {
            "stylesheets": [],
            "scripts": [],
            "modules": []
        }
        
        # Map dashboard names to their atomic components
        dashboard_map = {
            "dashboard": ["dashboard_styles", "dashboard_scripts"],
            "unified_gamma": ["unified_gamma_styles", "unified_gamma_scripts"],
            "charts": ["charts_styles", "charts_scripts"],
            "unified_template": ["unified_template_styles", "unified_template_scripts"]
        }
        
        components = dashboard_map.get(dashboard_name, [])
        
        for component in components:
            if "_styles" in component:
                css = self.get_component("css", component)
                if css:
                    assets["stylesheets"].append(css["url"])
            elif "_scripts" in component:
                js = self.get_component("js", component)
                if js:
                    assets["scripts"].append(js["url"])
        
        return assets
    
    def get_atomic_stats(self) -> Dict:
        """Get statistics about atomized components"""
        return {
            "total_css_atoms": len(self.registry["css"]),
            "total_js_atoms": len(self.registry["js"]),
            "total_python_atoms": len(self.registry["python"]),
            "total_html_cores": len(self.registry["html_cores"]),
            "total_components": sum(len(v) for v in self.registry.values()),
            "atomization_complete": True
        }
    
    def generate_import_manifest(self) -> str:
        """Generate import manifest for all atomic components"""
        manifest = {
            "version": "1.0.0",
            "created_by": "AGENT_X_STEELCLAD",
            "atomic_components": self.registry,
            "stats": self.get_atomic_stats()
        }
        return json.dumps(manifest, indent=2)


class AtomicDashboardBuilder:
    """
    Builder class to compose dashboards from atomic components
    """
    
    def __init__(self):
        self.registry = AtomicComponentRegistry()
        self.components = []
    
    def add_css(self, *names):
        """Add CSS atomic components"""
        for name in names:
            component = self.registry.get_component("css", name)
            if component:
                self.components.append(component)
        return self
    
    def add_js(self, *names):
        """Add JavaScript atomic components"""
        for name in names:
            component = self.registry.get_component("js", name)
            if component:
                self.components.append(component)
        return self
    
    def add_python_module(self, *names):
        """Add Python atomic modules"""
        for name in names:
            component = self.registry.get_component("python", name)
            if component:
                self.components.append(component)
        return self
    
    def build_html_imports(self) -> str:
        """Build HTML import statements for all components"""
        imports = []
        
        for component in self.components:
            if component["type"] == "stylesheet":
                imports.append(f'<link rel="stylesheet" href="{component["url"]}">')
            elif component["type"] == "script":
                imports.append(f'<script src="{component["url"]}"></script>')
        
        return "\n".join(imports)
    
    def build_python_imports(self) -> str:
        """Build Python import statements for all modules"""
        imports = []
        
        for component in self.components:
            if component["type"] == "python_module":
                module_path = component["module"]
                module_name = module_path.split(".")[-1]
                imports.append(f"from {module_path} import *")
        
        return "\n".join(imports)


# Atomic component loader for Flask integration
def load_atomic_assets(app, dashboard_type: str = "unified"):
    """
    Load atomic assets into Flask app
    
    Args:
        app: Flask application instance
        dashboard_type: Type of dashboard to load assets for
    """
    registry = AtomicComponentRegistry()
    assets = registry.get_dashboard_assets(dashboard_type)
    
    # Store in app config for template access
    app.config['ATOMIC_STYLESHEETS'] = assets.get('stylesheets', [])
    app.config['ATOMIC_SCRIPTS'] = assets.get('scripts', [])
    
    return assets


# Export atomic stats for monitoring
def get_atomization_report():
    """Generate atomization report"""
    registry = AtomicComponentRegistry()
    stats = registry.get_atomic_stats()
    
    report = [
        "=" * 60,
        "AGENT X STEELCLAD: ATOMIZATION REPORT",
        "=" * 60,
        f"CSS Atoms: {stats['total_css_atoms']}",
        f"JavaScript Atoms: {stats['total_js_atoms']}",
        f"Python Atoms: {stats['total_python_atoms']}",
        f"HTML Cores: {stats['total_html_cores']}",
        "-" * 60,
        f"TOTAL ATOMIC COMPONENTS: {stats['total_components']}",
        f"ATOMIZATION STATUS: {'✅ COMPLETE' if stats['atomization_complete'] else '⏳ IN PROGRESS'}",
        "=" * 60
    ]
    
    return "\n".join(report)


if __name__ == "__main__":
    # Test the registry
    print(get_atomization_report())
    
    # Test the builder
    builder = AtomicDashboardBuilder()
    builder.add_css("dashboard_styles", "charts_styles")
    builder.add_js("dashboard_scripts", "charts_scripts")
    
    print("\nHTML Imports:")
    print(builder.build_html_imports())
    
    print("\nPython Imports:")
    builder.add_python_module("dashboard_analytics", "viz_engine")
    print(builder.build_python_imports())