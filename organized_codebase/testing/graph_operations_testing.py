# -*- coding: utf-8 -*-
"""
FalkorDB Graph Operations Testing Framework
==========================================

Extracted from falkordb-py tests
Enhanced for TestMaster integration

Testing patterns for:
- Core graph CRUD operations (Create, Read, Update, Delete)
- Node and edge lifecycle management
- Graph path traversal and validation
- Vector operations and embeddings
- Parameterized queries and data type handling
- Array and collection operations
- Graph relationship management
- Query result validation and comparison
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


class MockNode:
    """Mock node for graph testing"""
    
    def __init__(
        self, 
        node_id: Optional[int] = None,
        alias: Optional[str] = None,
        labels: Union[str, List[str], None] = None,
        properties: Optional[Dict[str, Any]] = None
    ):
        self.node_id = node_id
        self.alias = alias
        self.labels = self._normalize_labels(labels)
        self.properties = properties or {}
    
    def _normalize_labels(self, labels: Union[str, List[str], None]) -> List[str]:
        """Normalize labels to list format"""
        if labels is None:
            return []
        elif isinstance(labels, str):
            return [labels]
        else:
            return labels
    
    def to_string(self) -> str:
        """Get properties string representation"""
        if not self.properties:
            return ""
        
        prop_strs = []
        for key, value in self.properties.items():
            if isinstance(value, str):
                prop_strs.append(f'{key}:"{value}"')
            else:
                prop_strs.append(f'{key}:{value}')
        
        return "{" + ",".join(prop_strs) + "}"
    
    def __str__(self) -> str:
        """String representation for Cypher queries"""
        parts = []
        
        if self.alias:
            parts.append(self.alias)
        
        if self.labels:
            label_str = ":".join(self.labels)
            parts.append(f":{label_str}")
        
        if self.properties:
            parts.append(self.to_string())
        
        content = "".join(parts)
        return f"({content})"
    
    def __eq__(self, other) -> bool:
        """Equality comparison for testing"""
        if not isinstance(other, MockNode):
            return False
        
        # If both have node_id, compare by ID (ignores alias)
        if self.node_id is not None and other.node_id is not None:
            return (
                self.node_id == other.node_id and
                self.labels == other.labels and
                self.properties == other.properties
            )
        
        # Otherwise compare all attributes
        return (
            self.node_id == other.node_id and
            self.alias == other.alias and
            self.labels == other.labels and
            self.properties == other.properties
        )


class MockEdge:
    """Mock edge for graph testing"""
    
    def __init__(
        self,
        src_node: MockNode,
        relation: Optional[str],
        dest_node: MockNode,
        edge_id: Optional[int] = None,
        alias: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ):
        # Validate nodes are provided
        assert src_node is not None, "Source node cannot be None"
        assert dest_node is not None, "Destination node cannot be None"
        
        self.src_node = src_node
        self.relation = relation
        self.dest_node = dest_node
        self.edge_id = edge_id
        self.alias = alias
        self.properties = properties or {}
    
    def to_string(self) -> str:
        """Get properties string representation"""
        if not self.properties:
            return ""
        
        prop_strs = []
        for key, value in self.properties.items():
            if isinstance(value, str):
                prop_strs.append(f'{key}:"{value}"')
            else:
                prop_strs.append(f'{key}:{value}')
        
        return "{" + ",".join(prop_strs) + "}"
    
    def __str__(self) -> str:
        """String representation for Cypher queries"""
        # Build relationship part
        rel_parts = []
        
        if self.alias:
            rel_parts.append(self.alias)
        
        if self.relation:
            if rel_parts:
                rel_parts.append(f":{self.relation}")
            else:
                rel_parts.append(f":{self.relation}")
        
        if self.properties:
            rel_parts.append(self.to_string())
        
        if rel_parts:
            rel_content = "".join(rel_parts)
        else:
            rel_content = ""
        
        # Format: (src)-[rel]->(dest)
        return f"({self.src_node.alias or ''})-[{rel_content}]->({self.dest_node.alias or ''})"
    
    def __eq__(self, other) -> bool:
        """Equality comparison for testing"""
        if not isinstance(other, MockEdge):
            return False
        
        return (
            self.src_node == other.src_node and
            self.relation == other.relation and
            self.dest_node == other.dest_node and
            self.edge_id == other.edge_id and
            self.properties == other.properties
        )


class MockPath:
    """Mock path for graph testing"""
    
    def __init__(self, nodes: List[MockNode], edges: List[MockEdge]):
        self.nodes = nodes
        self.edges = edges
    
    def __eq__(self, other) -> bool:
        """Equality comparison for testing"""
        if not isinstance(other, MockPath):
            return False
        
        return self.nodes == other.nodes and self.edges == other.edges
    
    def __str__(self) -> str:
        """String representation"""
        return f"Path(nodes={len(self.nodes)}, edges={len(self.edges)})"


class MockQueryResult:
    """Mock query result"""
    
    def __init__(self, result_set: List[List[Any]], statistics: Optional[Dict] = None):
        self.result_set = result_set
        self.statistics = statistics or {}
        self.nodes_created = statistics.get('nodes_created', 0) if statistics else 0
        self.relationships_created = statistics.get('relationships_created', 0) if statistics else 0
        self.properties_set = statistics.get('properties_set', 0) if statistics else 0
    
    def __bool__(self) -> bool:
        """Boolean evaluation"""
        return len(self.result_set) > 0


class MockGraph:
    """Mock graph database for testing"""
    
    def __init__(self, name: str):
        self.name = name
        self.nodes = {}
        self.edges = {}
        self.queries_executed = []
        self.node_counter = 0
        self.edge_counter = 0
        self.deleted = False
    
    async def query(self, query: str, params: Optional[Dict] = None) -> MockQueryResult:
        """Execute mock query"""
        params = params or {}
        
        self.queries_executed.append({
            'query': query,
            'params': params,
            'timestamp': time.time()
        })
        
        # Handle different query types
        if query.startswith('CREATE'):
            return self._handle_create_query(query, params)
        elif query.startswith('MATCH'):
            return self._handle_match_query(query, params)
        elif query.startswith('RETURN'):
            return self._handle_return_query(query, params)
        else:
            # Default empty result
            return MockQueryResult([])
    
    def _handle_create_query(self, query: str, params: Dict) -> MockQueryResult:
        """Handle CREATE query simulation"""
        # Simple simulation - count potential creations
        nodes_created = query.count('Node(') + query.count('(')
        relationships_created = query.count('-[') + query.count('Edge(')
        
        statistics = {
            'nodes_created': nodes_created,
            'relationships_created': relationships_created,
            'properties_set': query.count('{')
        }
        
        # For CREATE...RETURN queries, simulate return values
        if 'RETURN' in query:
            # Extract return variables and create mock results
            return_part = query.split('RETURN')[1].strip()
            variables = [v.strip() for v in return_part.split(',')]
            
            # Create mock return values
            mock_results = []
            for var in variables:
                if 'p' in var:  # Person node
                    mock_results.append(MockNode(
                        alias='p',
                        labels='person',
                        properties={'name': 'John Doe', 'age': 33}
                    ))
                elif 'v' in var:  # Visited edge
                    mock_results.append(MockEdge(
                        MockNode(alias='p'),
                        'visited',
                        MockNode(alias='c'),
                        properties={'purpose': 'pleasure'}
                    ))
                elif 'c' in var:  # Country node
                    mock_results.append(MockNode(
                        alias='c',
                        labels='country',
                        properties={'name': 'Japan'}
                    ))
            
            result_set = [mock_results] if mock_results else []
        else:
            result_set = []
        
        return MockQueryResult(result_set, statistics)
    
    def _handle_match_query(self, query: str, params: Dict) -> MockQueryResult:
        """Handle MATCH query simulation"""
        # Simulate path matching
        if 'p=' in query and 'RETURN p' in query:
            # Return mock path
            node0 = MockNode(node_id=0, labels=['L1'])
            node1 = MockNode(node_id=1, labels=['L1'])
            edge01 = MockEdge(node0, 'R1', node1, edge_id=0, properties={'value': 1})
            path = MockPath([node0, node1], [edge01])
            
            return MockQueryResult([[path]])
        
        # Simulate collect function
        if 'collect(' in query:
            mock_node = MockNode(
                node_id=0,
                labels=['person'],
                properties={'name': 'a', 'age': 32, 'array': [0, 1, 2]}
            )
            return MockQueryResult([[mock_node]])
        
        return MockQueryResult([])
    
    def _handle_return_query(self, query: str, params: Dict) -> MockQueryResult:
        """Handle RETURN query simulation"""
        query = query.strip()
        
        # Handle parameter substitution
        if '$param' in query and 'param' in params:
            return MockQueryResult([[params['param']]])
        
        # Handle vector operations
        if 'vecf32(' in query:
            # Extract vector values
            import re
            vector_match = re.search(r'vecf32\(\[([^\]]+)\]', query)
            if vector_match:
                vector_str = vector_match.group(1)
                values = [float(x.strip()) for x in vector_str.split(',')]
                return MockQueryResult([values])
        
        # Handle array literals
        if query.startswith('RETURN [') and query.endswith(']'):
            # Parse array literal
            array_content = query[8:-1]  # Remove 'RETURN [' and ']'
            
            if not array_content.strip():
                return MockQueryResult([[]])
            
            # Parse mixed types
            elements = []
            for item in array_content.split(','):
                item = item.strip()
                if item == 'true':
                    elements.append(True)
                elif item == 'false':
                    elements.append(False)
                elif item == 'null':
                    elements.append(None)
                elif item.startswith('"') and item.endswith('"'):
                    elements.append(item[1:-1])  # Remove quotes
                elif '.' in item:
                    elements.append(float(item))
                else:
                    try:
                        elements.append(int(item))
                    except ValueError:
                        elements.append(item)
            
            return MockQueryResult([elements])
        
        return MockQueryResult([])
    
    def delete(self):
        """Delete the graph"""
        self.deleted = True
        self.nodes.clear()
        self.edges.clear()
        self.queries_executed.clear()
    
    def reset(self):
        """Reset graph state for testing"""
        self.delete()
        self.deleted = False


class GraphOperationsTestFramework:
    """Core framework for graph operations testing"""
    
    def __init__(self):
        self.graphs = {}
        self.nodes = {}
        self.edges = {}
        self.test_data = {}
    
    def create_mock_graph(self, name: str) -> MockGraph:
        """Create mock graph for testing"""
        graph = MockGraph(name)
        self.graphs[name] = graph
        return graph
    
    def create_test_node(
        self,
        name: str,
        node_id: Optional[int] = None,
        alias: Optional[str] = None,
        labels: Union[str, List[str], None] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> MockNode:
        """Create test node"""
        node = MockNode(node_id, alias, labels, properties)
        self.nodes[name] = node
        return node
    
    def create_test_edge(
        self,
        name: str,
        src_node: MockNode,
        relation: Optional[str],
        dest_node: MockNode,
        edge_id: Optional[int] = None,
        alias: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> MockEdge:
        """Create test edge"""
        edge = MockEdge(src_node, relation, dest_node, edge_id, alias, properties)
        self.edges[name] = edge
        return edge
    
    async def execute_graph_operation_test(
        self,
        graph_name: str,
        operation_type: str,
        query: str,
        params: Optional[Dict] = None,
        expected_result: Any = None
    ) -> Dict[str, Any]:
        """Execute graph operation test"""
        if graph_name not in self.graphs:
            raise ValueError(f"Graph '{graph_name}' not found")
        
        graph = self.graphs[graph_name]
        start_time = time.time()
        
        try:
            result = await graph.query(query, params)
            execution_time = time.time() - start_time
            
            test_result = {
                'operation_type': operation_type,
                'query': query,
                'params': params,
                'result': result,
                'execution_time': execution_time,
                'success': True,
                'nodes_created': result.nodes_created,
                'relationships_created': result.relationships_created,
                'result_count': len(result.result_set),
                'query_executed': len(graph.queries_executed)
            }
            
            # Validate expected result if provided
            if expected_result is not None:
                if hasattr(result, 'result_set') and result.result_set:
                    actual_result = result.result_set[0] if result.result_set else None
                    test_result['expected_match'] = actual_result == expected_result
                else:
                    test_result['expected_match'] = result == expected_result
            
            return test_result
        
        except Exception as e:
            return {
                'operation_type': operation_type,
                'query': query,
                'params': params,
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def create_standard_test_data(self):
        """Create standard test data for common scenarios"""
        # Standard person node
        self.test_data['john'] = self.create_test_node(
            'john',
            alias='p',
            labels='person',
            properties={
                'name': 'John Doe',
                'age': 33,
                'gender': 'male',
                'status': 'single'
            }
        )
        
        # Standard country node
        self.test_data['japan'] = self.create_test_node(
            'japan',
            alias='c',
            labels='country',
            properties={'name': 'Japan'}
        )
        
        # Standard relationship
        self.test_data['visited'] = self.create_test_edge(
            'visited',
            self.test_data['john'],
            'visited',
            self.test_data['japan'],
            alias='v',
            properties={'purpose': 'pleasure'}
        )
        
        # Array test node
        self.test_data['array_node'] = self.create_test_node(
            'array_node',
            node_id=0,
            labels='person',
            properties={
                'name': 'a',
                'age': 32,
                'array': [0, 1, 2]
            }
        )
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get comprehensive test summary"""
        return {
            'total_graphs': len(self.graphs),
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'test_data_items': len(self.test_data),
            'graph_names': list(self.graphs.keys()),
            'node_names': list(self.nodes.keys()),
            'edge_names': list(self.edges.keys())
        }


class GraphValidator:
    """Validator for graph operations"""
    
    @staticmethod
    def validate_node_structure(node: MockNode, expected_properties: Dict = None) -> Dict[str, Any]:
        """Validate node structure"""
        validations = {
            'has_properties': hasattr(node, 'properties'),
            'has_labels': hasattr(node, 'labels'),
            'has_node_id': hasattr(node, 'node_id'),
            'has_alias': hasattr(node, 'alias')
        }
        
        if expected_properties:
            property_validations = {}
            for key, expected_value in expected_properties.items():
                actual_value = node.properties.get(key)
                property_validations[key] = {
                    'expected': expected_value,
                    'actual': actual_value,
                    'matches': actual_value == expected_value
                }
            
            validations['property_validations'] = property_validations
            validations['all_properties_match'] = all(
                v['matches'] for v in property_validations.values()
            )
        
        return {
            'is_valid': all(v if isinstance(v, bool) else v.get('all_properties_match', True) 
                          for v in validations.values()),
            'validations': validations
        }
    
    @staticmethod
    def validate_edge_structure(edge: MockEdge, expected_relation: str = None) -> Dict[str, Any]:
        """Validate edge structure"""
        validations = {
            'has_src_node': edge.src_node is not None,
            'has_dest_node': edge.dest_node is not None,
            'has_relation': hasattr(edge, 'relation'),
            'has_properties': hasattr(edge, 'properties')
        }
        
        if expected_relation:
            validations['relation_matches'] = edge.relation == expected_relation
        
        return {
            'is_valid': all(validations.values()),
            'validations': validations
        }
    
    @staticmethod
    def validate_query_result(
        result: MockQueryResult,
        expected_count: int = None,
        expected_statistics: Dict = None
    ) -> Dict[str, Any]:
        """Validate query result"""
        validations = {
            'has_result_set': hasattr(result, 'result_set'),
            'has_statistics': hasattr(result, 'statistics')
        }
        
        if expected_count is not None:
            validations['count_matches'] = len(result.result_set) == expected_count
        
        if expected_statistics:
            stat_validations = {}
            for key, expected_value in expected_statistics.items():
                actual_value = getattr(result, key, 0)
                stat_validations[key] = {
                    'expected': expected_value,
                    'actual': actual_value,
                    'matches': actual_value == expected_value
                }
            
            validations['statistics_validations'] = stat_validations
        
        return {
            'is_valid': all(v if isinstance(v, bool) else all(sv['matches'] for sv in v.values()) 
                          for v in validations.values()),
            'validations': validations
        }
    
    @staticmethod
    def validate_path_structure(path: MockPath, expected_node_count: int = None, expected_edge_count: int = None) -> Dict[str, Any]:
        """Validate path structure"""
        validations = {
            'has_nodes': hasattr(path, 'nodes'),
            'has_edges': hasattr(path, 'edges'),
            'nodes_is_list': isinstance(path.nodes, list),
            'edges_is_list': isinstance(path.edges, list)
        }
        
        if expected_node_count is not None:
            validations['node_count_matches'] = len(path.nodes) == expected_node_count
        
        if expected_edge_count is not None:
            validations['edge_count_matches'] = len(path.edges) == expected_edge_count
        
        return {
            'is_valid': all(validations.values()),
            'validations': validations
        }


class GraphOperationsTest(IsolatedAsyncioTestCase):
    """Comprehensive graph operations testing"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        self.framework = GraphOperationsTestFramework()
        self.validator = GraphValidator()
        
        # Create standard test data
        self.framework.create_standard_test_data()
        
        # Create test graph
        self.graph = self.framework.create_mock_graph("test_graph")
    
    async def test_basic_graph_creation(self):
        """Test basic graph creation operations"""
        # Create nodes and edge
        john = self.framework.test_data['john']
        japan = self.framework.test_data['japan']
        visited = self.framework.test_data['visited']
        
        # Build CREATE query
        query = f"CREATE {john}, {japan}, {visited} RETURN p,v,c"
        
        # Execute test
        result = await self.framework.execute_graph_operation_test(
            "test_graph",
            "create",
            query
        )
        
        assert result['success'] == True
        assert result['nodes_created'] > 0
        assert result['result_count'] > 0
        
        # Validate returned entities
        if result['result'] and result['result'].result_set:
            returned_person = result['result'].result_set[0][0]
            returned_visit = result['result'].result_set[0][1]
            returned_country = result['result'].result_set[0][2]
            
            # Validate person
            person_validation = self.validator.validate_node_structure(
                returned_person,
                {'name': 'John Doe', 'age': 33}
            )
            assert person_validation['is_valid']
            
            # Validate edge
            edge_validation = self.validator.validate_edge_structure(
                returned_visit,
                'visited'
            )
            assert edge_validation['is_valid']
    
    async def test_array_operations(self):
        """Test array and collection operations"""
        # Test basic array return
        result = await self.framework.execute_graph_operation_test(
            "test_graph",
            "array",
            "RETURN [0,1,2]"
        )
        
        assert result['success'] == True
        assert result['result'].result_set[0] == [0, 1, 2]
        
        # Test mixed type array
        result = await self.framework.execute_graph_operation_test(
            "test_graph",
            "mixed_array",
            'RETURN [1, 2.3, "4", true, false, null]'
        )
        
        assert result['success'] == True
        expected_array = [1, 2.3, "4", True, False, None]
        assert result['result'].result_set[0] == expected_array
        
        # Test collect function
        array_node = self.framework.test_data['array_node']
        await self.graph.query(f"CREATE {array_node}")
        
        result = await self.framework.execute_graph_operation_test(
            "test_graph",
            "collect",
            "MATCH(n) return collect(n)"
        )
        
        assert result['success'] == True
        assert len(result['result'].result_set) > 0
    
    async def test_path_operations(self):
        """Test graph path operations"""
        # Create path test data
        node0 = MockNode(alias="node0", node_id=0, labels=["L1"])
        node1 = MockNode(alias="node1", node_id=1, labels=["L1"])
        edge01 = MockEdge(node0, "R1", node1, edge_id=0, properties={"value": 1})
        
        # Create the path in graph
        await self.graph.query(f"CREATE {node0}, {node1}, {edge01}")
        
        # Query path
        result = await self.framework.execute_graph_operation_test(
            "test_graph",
            "path",
            "MATCH p=(:L1)-[:R1]->(:L1) RETURN p"
        )
        
        assert result['success'] == True
        
        # Validate path structure
        if result['result'].result_set:
            returned_path = result['result'].result_set[0][0]
            
            path_validation = self.validator.validate_path_structure(
                returned_path,
                expected_node_count=2,
                expected_edge_count=1
            )
            assert path_validation['is_valid']
    
    async def test_vector_operations(self):
        """Test vector operations"""
        # Test vector function
        result = await self.framework.execute_graph_operation_test(
            "test_graph",
            "vector",
            "RETURN vecf32([1.2, 2.3, -1.2, 0.1])"
        )
        
        assert result['success'] == True
        
        # Validate vector result (allowing for floating point precision)
        if result['result'].result_set:
            vector_result = result['result'].result_set[0]
            expected_vector = [1.2, 2.3, -1.2, 0.1]
            
            # Round for comparison
            actual_rounded = [round(x, 3) for x in vector_result]
            expected_rounded = [round(x, 3) for x in expected_vector]
            
            assert actual_rounded == expected_rounded
    
    async def test_parameterized_queries(self):
        """Test parameterized queries"""
        test_params = [
            1,
            2.3,
            "str",
            True,
            False,
            None,
            [0, 1, 2],
            r"\" RETURN 1337 //"  # SQL injection test
        ]
        
        for i, param in enumerate(test_params):
            result = await self.framework.execute_graph_operation_test(
                "test_graph",
                f"param_test_{i}",
                "RETURN $param",
                {"param": param}
            )
            
            assert result['success'] == True
            assert result['result'].result_set[0][0] == param
    
    async def test_node_comparison_and_equality(self):
        """Test node comparison and equality operations"""
        # Test different node configurations
        no_args = MockNode(alias="n")
        no_props = MockNode(node_id=1, alias="n", labels="l")
        no_label = MockNode(node_id=1, alias="n", properties={"a": "a"})
        props_only = MockNode(alias="n", properties={"a": "a", "b": 10})
        multi_label = MockNode(node_id=1, alias="n", labels=["l", "ll"])
        
        # Test string representations
        assert no_args.to_string() == ""
        assert no_props.to_string() == ""
        assert no_label.to_string() == '{a:"a"}'
        assert props_only.to_string() == '{a:"a",b:10}'
        assert multi_label.to_string() == ""
        
        # Test full string representation
        assert str(no_args) == "(n)"
        assert str(no_props) == "(n:l)"
        assert str(no_label) == '(n{a:"a"})'
        assert str(props_only) == '(n{a:"a",b:10})'
        assert str(multi_label) == "(n:l:ll)"
        
        # Test equality comparisons
        assert MockNode() != MockNode(properties={"a": 10})
        assert MockNode() == MockNode()
        assert MockNode(node_id=1) == MockNode(node_id=1)
        assert MockNode(node_id=1) != MockNode(node_id=2)
        assert MockNode(node_id=1, alias="a") == MockNode(node_id=1, alias="b")  # ID takes precedence
    
    async def test_edge_comparison_and_validation(self):
        """Test edge comparison and validation"""
        node1 = MockNode(node_id=1)
        node2 = MockNode(node_id=2)
        node3 = MockNode(node_id=3)
        
        # Test edge creation validation
        with pytest.raises(AssertionError):
            MockEdge(None, None, None)
        
        # Test valid edge creation
        edge1 = MockEdge(node1, None, node2)
        assert isinstance(edge1, MockEdge)
        
        # Test edge string representations
        john = MockNode(
            alias="a",
            labels="person",
            properties={"name": 'John Doe', "age": 33, "someArray": [1, 2, 3]}
        )
        japan = MockNode(alias="b", labels="country", properties={"name": 'Japan'})
        
        edge_with_relation = MockEdge(
            john, "visited", japan, properties={"purpose": "pleasure"}
        )
        assert str(edge_with_relation) == "(a)-[:visited{purpose:\"pleasure\"}]->(b)"
        
        edge_no_relation = MockEdge(japan, "", john)
        assert str(edge_no_relation) == "(b)-[]->(a)"
        
        # Test edge equality
        assert edge1 == MockEdge(node1, None, node2)
        assert edge1 != MockEdge(node1, "bla", node2)
        assert edge1 != MockEdge(node1, None, node3)
    
    async def test_query_result_validation(self):
        """Test query result validation"""
        # Test successful query
        result = await self.framework.execute_graph_operation_test(
            "test_graph",
            "validation_test",
            "RETURN [1, 2, 3]"
        )
        
        # Validate result structure
        result_validation = self.validator.validate_query_result(
            result['result'],
            expected_count=1
        )
        assert result_validation['is_valid']
        
        # Test query with statistics
        john = self.framework.test_data['john']
        create_result = await self.framework.execute_graph_operation_test(
            "test_graph",
            "stats_test",
            f"CREATE {john}"
        )
        
        stats_validation = self.validator.validate_query_result(
            create_result['result'],
            expected_statistics={'nodes_created': 1}
        )
        assert stats_validation['is_valid']
    
    async def test_graph_lifecycle_management(self):
        """Test graph lifecycle management"""
        graph = self.framework.create_mock_graph("lifecycle_test")
        
        # Test initial state
        assert graph.name == "lifecycle_test"
        assert not graph.deleted
        assert len(graph.queries_executed) == 0
        
        # Execute some operations
        await graph.query("CREATE (n:Person {name: 'Test'})")
        assert len(graph.queries_executed) == 1
        
        # Test reset
        graph.reset()
        assert len(graph.queries_executed) == 0
        assert not graph.deleted
        
        # Test deletion
        graph.delete()
        assert graph.deleted
    
    async def test_framework_comprehensive_functionality(self):
        """Test framework's comprehensive functionality"""
        # Create additional test data
        self.framework.create_test_node(
            'test_node',
            node_id=100,
            labels=['Test', 'Node'],
            properties={'test': True}
        )
        
        self.framework.create_test_edge(
            'test_edge',
            self.framework.nodes['test_node'],
            'TESTS',
            self.framework.test_data['john'],
            properties={'confidence': 0.95}
        )
        
        # Get comprehensive summary
        summary = self.framework.get_test_summary()
        
        assert summary['total_graphs'] >= 1
        assert summary['total_nodes'] >= 4  # john, japan, array_node, test_node
        assert summary['total_edges'] >= 2  # visited, test_edge
        assert 'test_node' in summary['node_names']
        assert 'test_edge' in summary['edge_names']
    
    async def asyncTearDown(self):
        """Clean up test environment"""
        # Clean up all graphs
        for graph in self.framework.graphs.values():
            graph.delete()


# Pytest integration
@pytest.fixture
def graph_framework():
    """Pytest fixture for graph framework"""
    framework = GraphOperationsTestFramework()
    framework.create_standard_test_data()
    yield framework


@pytest.fixture
def graph_validator():
    """Pytest fixture for graph validator"""
    return GraphValidator()


def test_graph_framework_creation(graph_framework):
    """Test graph framework creation"""
    graph = graph_framework.create_mock_graph("test")
    assert graph.name == "test"
    assert "test" in graph_framework.graphs


def test_mock_node_creation():
    """Test mock node creation and operations"""
    node = MockNode(
        node_id=1,
        alias="n",
        labels="person",
        properties={"name": "Test"}
    )
    
    assert node.node_id == 1
    assert node.alias == "n"
    assert node.labels == ["person"]
    assert node.properties["name"] == "Test"
    
    # Test string representations
    assert "Test" in node.to_string()
    assert "n:person" in str(node)


def test_mock_edge_creation():
    """Test mock edge creation and validation"""
    node1 = MockNode(node_id=1, alias="a")
    node2 = MockNode(node_id=2, alias="b")
    
    edge = MockEdge(
        node1, "KNOWS", node2,
        properties={"since": 2020}
    )
    
    assert edge.src_node == node1
    assert edge.dest_node == node2
    assert edge.relation == "KNOWS"
    assert edge.properties["since"] == 2020


def test_node_structure_validation(graph_validator):
    """Test node structure validation"""
    node = MockNode(
        labels="person",
        properties={"name": "Test", "age": 30}
    )
    
    validation = graph_validator.validate_node_structure(
        node,
        expected_properties={"name": "Test", "age": 30}
    )
    
    assert validation["is_valid"]


def test_edge_structure_validation(graph_validator):
    """Test edge structure validation"""
    node1 = MockNode(alias="a")
    node2 = MockNode(alias="b")
    edge = MockEdge(node1, "KNOWS", node2)
    
    validation = graph_validator.validate_edge_structure(edge, "KNOWS")
    assert validation["is_valid"]


@pytest.mark.asyncio
async def test_simple_graph_operations(graph_framework):
    """Test simple graph operations"""
    graph = graph_framework.create_mock_graph("simple_test")
    
    # Test basic query
    result = await graph.query("RETURN [1, 2, 3]")
    assert len(result.result_set) == 1
    assert result.result_set[0] == [1, 2, 3]


def test_query_result_validation(graph_validator):
    """Test query result validation"""
    result = MockQueryResult([[1, 2, 3]], {"nodes_created": 1})
    
    validation = graph_validator.validate_query_result(
        result,
        expected_count=1,
        expected_statistics={"nodes_created": 1}
    )
    
    assert validation["is_valid"]