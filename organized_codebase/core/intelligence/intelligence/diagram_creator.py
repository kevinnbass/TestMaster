"""
Architecture Diagram Creator

Generates architecture diagrams and visualizations from code analysis.
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Component:
    """Represents a system component."""
    name: str
    type: str  # module, class, function, service
    file_path: str
    dependencies: List[str]
    exports: List[str]
    metrics: Dict[str, Any]
    

@dataclass
class Relationship:
    """Represents relationship between components."""
    source: str
    target: str
    type: str  # imports, calls, inherits, implements
    strength: int  # 1-10
    

class DiagramCreator:
    """
    Creates architecture diagrams and visualizations.
    Generates Mermaid, PlantUML, and GraphViz formats.
    """
    
    def __init__(self):
        """Initialize the diagram creator."""
        self.components = {}
        self.relationships = []
        self.layers = defaultdict(list)
        logger.info("Diagram Creator initialized")
        
    def analyze_architecture(self, root_path: str) -> Dict[str, Any]:
        """
        Analyze system architecture from codebase.
        
        Args:
            root_path: Root directory of codebase
            
        Returns:
            Architecture analysis results
        """
        root = Path(root_path)
        
        # Scan all Python files
        for py_file in root.rglob("*.py"):
            if '__pycache__' not in str(py_file):
                self._analyze_file(py_file)
                
        # Detect layers
        self._detect_layers()
        
        return {
            'components': len(self.components),
            'relationships': len(self.relationships),
            'layers': dict(self.layers)
        }
        
    def generate_mermaid_diagram(self) -> str:
        """
        Generate Mermaid diagram code.
        
        Returns:
            Mermaid diagram definition
        """
        lines = ["graph TD"]
        
        # Add components
        for comp_id, comp in self.components.items():
            shape = self._get_mermaid_shape(comp.type)
            lines.append(f"    {comp_id}{shape[0]}{comp.name}{shape[1]}")
            
        # Add relationships
        for rel in self.relationships:
            arrow = self._get_mermaid_arrow(rel.type)
            lines.append(f"    {rel.source} {arrow} {rel.target}")
            
        # Add subgraphs for layers
        for layer, components in self.layers.items():
            lines.append(f"    subgraph {layer}")
            for comp in components:
                lines.append(f"        {comp}")
            lines.append("    end")
            
        return "\n".join(lines)
        
    def generate_plantuml_diagram(self) -> str:
        """
        Generate PlantUML diagram code.
        
        Returns:
            PlantUML diagram definition
        """
        lines = ["@startuml", "!theme plain"]
        
        # Add packages/layers
        for layer, components in self.layers.items():
            lines.append(f"package {layer} {{")
            for comp_id in components:
                if comp_id in self.components:
                    comp = self.components[comp_id]
                    lines.append(f"  {comp.type} {comp.name}")
            lines.append("}")
            
        # Add relationships
        for rel in self.relationships:
            arrow = self._get_plantuml_arrow(rel.type)
            lines.append(f"{rel.source} {arrow} {rel.target}")
            
        lines.append("@enduml")
        return "\n".join(lines)
        
    def generate_dependency_matrix(self) -> Dict[str, List[str]]:
        """
        Generate dependency matrix.
        
        Returns:
            Dependency matrix as dict
        """
        matrix = defaultdict(list)
        
        for rel in self.relationships:
            if rel.type in ['imports', 'calls']:
                matrix[rel.source].append(rel.target)
                
        return dict(matrix)
        
    def generate_component_graph(self) -> Dict[str, Any]:
        """
        Generate component graph data structure.
        
        Returns:
            Graph representation for visualization tools
        """
        graph = {
            'nodes': [],
            'edges': [],
            'clusters': []
        }
        
        # Add nodes
        for comp_id, comp in self.components.items():
            graph['nodes'].append({
                'id': comp_id,
                'label': comp.name,
                'type': comp.type,
                'metrics': comp.metrics
            })
            
        # Add edges
        for rel in self.relationships:
            graph['edges'].append({
                'source': rel.source,
                'target': rel.target,
                'type': rel.type,
                'weight': rel.strength
            })
            
        # Add clusters (layers)
        for layer, components in self.layers.items():
            graph['clusters'].append({
                'id': layer,
                'nodes': components
            })
            
        return graph
        
    def export_diagram(self, output_path: str, format: str = "mermaid") -> None:
        """
        Export diagram to file.
        
        Args:
            output_path: Output file path
            format: Diagram format (mermaid, plantuml, json)
        """
        content = ""
        
        if format == "mermaid":
            content = self.generate_mermaid_diagram()
        elif format == "plantuml":
            content = self.generate_plantuml_diagram()
        elif format == "json":
            content = json.dumps(self.generate_component_graph(), indent=2)
            
        with open(output_path, 'w') as f:
            f.write(content)
            
        logger.info(f"Exported diagram to {output_path}")
        
    def get_complexity_metrics(self) -> Dict[str, Any]:
        """
        Calculate architecture complexity metrics.
        
        Returns:
            Complexity metrics
        """
        metrics = {
            'total_components': len(self.components),
            'total_relationships': len(self.relationships),
            'coupling_score': 0,
            'cohesion_score': 0,
            'layer_violations': 0
        }
        
        # Calculate coupling (external dependencies)
        external_deps = defaultdict(int)
        for rel in self.relationships:
            external_deps[rel.source] += 1
            
        metrics['coupling_score'] = sum(external_deps.values()) / max(len(self.components), 1)
        
        # Calculate cohesion (internal connections within layers)
        for layer, components in self.layers.items():
            internal = 0
            for rel in self.relationships:
                if rel.source in components and rel.target in components:
                    internal += 1
            metrics['cohesion_score'] += internal
            
        # Detect layer violations
        layer_order = ['presentation', 'business', 'data']
        for rel in self.relationships:
            src_layer = self._get_component_layer(rel.source)
            tgt_layer = self._get_component_layer(rel.target)
            if src_layer and tgt_layer:
                if layer_order.index(src_layer) > layer_order.index(tgt_layer):
                    metrics['layer_violations'] += 1
                    
        return metrics
        
    # Helper methods
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a Python file for components and relationships."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
                
            comp_id = file_path.stem
            
            # Create component
            comp = Component(
                name=file_path.stem,
                type='module',
                file_path=str(file_path),
                dependencies=[],
                exports=[],
                metrics={'lines': len(open(file_path).readlines())}
            )
            
            # Find imports (dependencies)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        comp.dependencies.append(alias.name)
                        self.relationships.append(Relationship(
                            source=comp_id,
                            target=alias.name,
                            type='imports',
                            strength=5
                        ))
                        
                elif isinstance(node, ast.ImportFrom) and node.module:
                    comp.dependencies.append(node.module)
                    
            self.components[comp_id] = comp
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            
    def _detect_layers(self) -> None:
        """Detect architectural layers from component paths."""
        for comp_id, comp in self.components.items():
            path_parts = Path(comp.file_path).parts
            
            if 'api' in path_parts or 'routes' in path_parts:
                self.layers['presentation'].append(comp_id)
            elif 'models' in path_parts or 'db' in path_parts:
                self.layers['data'].append(comp_id)
            else:
                self.layers['business'].append(comp_id)
                
    def _get_component_layer(self, comp_id: str) -> Optional[str]:
        """Get layer for a component."""
        for layer, components in self.layers.items():
            if comp_id in components:
                return layer
        return None
        
    def _get_mermaid_shape(self, comp_type: str) -> Tuple[str, str]:
        """Get Mermaid shape for component type."""
        shapes = {
            'module': ('[', ']'),
            'class': ('(', ')'),
            'function': ('{{', '}}'),
            'service': ('[(', ')]')
        }
        return shapes.get(comp_type, ('[', ']'))
        
    def _get_mermaid_arrow(self, rel_type: str) -> str:
        """Get Mermaid arrow for relationship type."""
        arrows = {
            'imports': '-->',
            'calls': '->',
            'inherits': '--|>',
            'implements': '..|>'
        }
        return arrows.get(rel_type, '-->')
        
    def _get_plantuml_arrow(self, rel_type: str) -> str:
        """Get PlantUML arrow for relationship type."""
        arrows = {
            'imports': '-->',
            'calls': '->',
            'inherits': '--|>',
            'implements': '..|>'
        }
        return arrows.get(rel_type, '-->')