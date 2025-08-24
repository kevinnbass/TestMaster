"""
Graph Vector Testing Framework
Extracted from FalkorDB vector operations and similarity testing patterns.
"""

import pytest
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass, field
from collections import defaultdict
import math
import random


@dataclass
class MockVector:
    """Mock vector object for testing vector operations"""
    data: List[float]
    dimension: int = field(init=False)
    
    def __post_init__(self):
        self.dimension = len(self.data)
    
    def normalize(self) -> 'MockVector':
        """Normalize vector to unit length"""
        magnitude = math.sqrt(sum(x * x for x in self.data))
        if magnitude == 0:
            return MockVector([0.0] * self.dimension)
        return MockVector([x / magnitude for x in self.data])
    
    def dot_product(self, other: 'MockVector') -> float:
        """Calculate dot product with another vector"""
        if self.dimension != other.dimension:
            raise ValueError("Vector dimensions must match")
        return sum(a * b for a, b in zip(self.data, other.data))
    
    def euclidean_distance(self, other: 'MockVector') -> float:
        """Calculate Euclidean distance to another vector"""
        if self.dimension != other.dimension:
            raise ValueError("Vector dimensions must match")
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.data, other.data)))
    
    def cosine_similarity(self, other: 'MockVector') -> float:
        """Calculate cosine similarity with another vector"""
        norm_self = self.normalize()
        norm_other = other.normalize()
        return norm_self.dot_product(norm_other)


@dataclass
class MockVectorIndex:
    """Mock vector index for testing vector indexing operations"""
    label: str
    properties: List[str]
    dimension: int
    similarity_function: str = "cosine"
    entity_type: str = "NODE"
    vectors: Dict[str, MockVector] = field(default_factory=dict)
    
    def add_vector(self, entity_id: str, vector_data: List[float]) -> bool:
        """Add vector to index"""
        if len(vector_data) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector_data)} doesn't match index dimension {self.dimension}")
        
        self.vectors[entity_id] = MockVector(vector_data)
        return True
    
    def search_similar(self, query_vector: List[float], k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        if len(query_vector) != self.dimension:
            raise ValueError("Query vector dimension doesn't match index dimension")
        
        query_vec = MockVector(query_vector)
        results = []
        
        for entity_id, stored_vector in self.vectors.items():
            if self.similarity_function == "cosine":
                score = query_vec.cosine_similarity(stored_vector)
            elif self.similarity_function == "euclidean":
                score = -query_vec.euclidean_distance(stored_vector)  # Negative for similarity ranking
            else:
                score = query_vec.dot_product(stored_vector)
            
            results.append((entity_id, score))
        
        # Sort by score descending and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


class MockVectorDatabase:
    """Mock vector-enabled graph database for testing"""
    
    def __init__(self):
        self.vector_indices = {}
        self.node_vectors = defaultdict(dict)  # {label: {property: {entity_id: vector}}}
        self.edge_vectors = defaultdict(dict)
        self.embedding_cache = {}
    
    def create_node_vector_index(self, label: str, property: str, dim: int = 128, 
                               similarity_function: str = "cosine") -> Mock:
        """Create vector index for node property"""
        index_key = f"NODE_{label}_{property}"
        
        if index_key in self.vector_indices:
            raise Exception("Vector index already exists")
        
        self.vector_indices[index_key] = MockVectorIndex(
            label=label,
            properties=[property],
            dimension=dim,
            similarity_function=similarity_function,
            entity_type="NODE"
        )
        
        result = Mock()
        result.indices_created = 1
        return result
    
    def create_edge_vector_index(self, relation: str, property: str, dim: int = 128,
                               similarity_function: str = "cosine") -> Mock:
        """Create vector index for edge property"""
        index_key = f"EDGE_{relation}_{property}"
        
        if index_key in self.vector_indices:
            raise Exception("Vector index already exists")
        
        self.vector_indices[index_key] = MockVectorIndex(
            label=relation,
            properties=[property],
            dimension=dim,
            similarity_function=similarity_function,
            entity_type="EDGE"
        )
        
        result = Mock()
        result.indices_created = 1
        return result
    
    def drop_node_vector_index(self, label: str, property: str) -> Mock:
        """Drop vector index for node property"""
        index_key = f"NODE_{label}_{property}"
        
        if index_key not in self.vector_indices:
            raise Exception("Vector index does not exist")
        
        del self.vector_indices[index_key]
        
        result = Mock()
        result.indices_deleted = 1
        return result
    
    def drop_edge_vector_index(self, relation: str, property: str) -> Mock:
        """Drop vector index for edge property"""
        index_key = f"EDGE_{relation}_{property}"
        
        if index_key not in self.vector_indices:
            raise Exception("Vector index does not exist")
        
        del self.vector_indices[index_key]
        
        result = Mock()
        result.indices_deleted = 1
        return result
    
    def add_node_vector(self, label: str, property: str, entity_id: str, vector_data: List[float]) -> bool:
        """Add vector data to node"""
        index_key = f"NODE_{label}_{property}"
        if index_key in self.vector_indices:
            return self.vector_indices[index_key].add_vector(entity_id, vector_data)
        return False
    
    def add_edge_vector(self, relation: str, property: str, entity_id: str, vector_data: List[float]) -> bool:
        """Add vector data to edge"""
        index_key = f"EDGE_{relation}_{property}"
        if index_key in self.vector_indices:
            return self.vector_indices[index_key].add_vector(entity_id, vector_data)
        return False
    
    def vector_similarity_search(self, label: str, property: str, query_vector: List[float], 
                                k: int = 10, entity_type: str = "NODE") -> List[Tuple[str, float]]:
        """Perform vector similarity search"""
        index_key = f"{entity_type}_{label}_{property}"
        if index_key not in self.vector_indices:
            raise Exception(f"Vector index {index_key} does not exist")
        
        return self.vector_indices[index_key].search_similar(query_vector, k)
    
    def parse_vecf32(self, vector_string: str) -> List[float]:
        """Parse vecf32 string representation to float list"""
        # Simulate parsing "vecf32([1, -2, 3.14])" format
        if "vecf32" in vector_string and "[" in vector_string and "]" in vector_string:
            # Extract vector data from string
            start = vector_string.find('[') + 1
            end = vector_string.find(']')
            vector_str = vector_string[start:end]
            return [float(x.strip()) for x in vector_str.split(',')]
        return []


class VectorSimilarityCalculator:
    """Utility class for vector similarity calculations"""
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same dimension")
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
        """Calculate Euclidean distance between two vectors"""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same dimension")
        
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
    
    @staticmethod
    def manhattan_distance(vec1: List[float], vec2: List[float]) -> float:
        """Calculate Manhattan distance between two vectors"""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same dimension")
        
        return sum(abs(a - b) for a, b in zip(vec1, vec2))
    
    @staticmethod
    def generate_random_vector(dimension: int, min_val: float = -1.0, max_val: float = 1.0) -> List[float]:
        """Generate random vector for testing"""
        return [random.uniform(min_val, max_val) for _ in range(dimension)]
    
    @staticmethod
    def normalize_vector(vector: List[float]) -> List[float]:
        """Normalize vector to unit length"""
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude == 0:
            return [0.0] * len(vector)
        return [x / magnitude for x in vector]


class GraphVectorTestFramework:
    """Comprehensive test framework for graph vector operations"""
    
    def __init__(self):
        self.vector_db = MockVectorDatabase()
        self.similarity_calc = VectorSimilarityCalculator()
        self.test_vectors = {}
    
    def setup_test_vectors(self) -> None:
        """Setup test vectors for various scenarios"""
        # Create test vectors with different dimensions
        self.test_vectors = {
            'embedding_128': self.similarity_calc.generate_random_vector(128),
            'embedding_256': self.similarity_calc.generate_random_vector(256),
            'embedding_512': self.similarity_calc.generate_random_vector(512),
            'small_vec': [1.0, -2.0, 3.14],
            'unit_vec': [1.0, 0.0, 0.0],
            'zero_vec': [0.0, 0.0, 0.0]
        }
    
    def test_vector_index_creation(self) -> bool:
        """Test vector index creation patterns"""
        try:
            # Test node vector index creation
            result = self.vector_db.create_node_vector_index("Document", "embedding", 
                                                           dim=128, similarity_function="cosine")
            assert result.indices_created == 1
            
            result = self.vector_db.create_node_vector_index("Image", "features", 
                                                           dim=256, similarity_function="euclidean")
            assert result.indices_created == 1
            
            # Test edge vector index creation
            result = self.vector_db.create_edge_vector_index("SIMILAR_TO", "similarity_vec", 
                                                           dim=64, similarity_function="cosine")
            assert result.indices_created == 1
            
            return True
        except Exception as e:
            pytest.fail(f"Vector index creation test failed: {e}")
    
    def test_vector_index_operations(self) -> bool:
        """Test vector index CRUD operations"""
        try:
            # Create index
            self.vector_db.create_node_vector_index("Product", "description_vec", 
                                                   dim=128, similarity_function="cosine")
            
            # Add vectors
            test_vectors = [
                ("product_1", [1.0] * 128),
                ("product_2", [0.5] * 128),
                ("product_3", [-1.0] * 128)
            ]
            
            for product_id, vector_data in test_vectors:
                success = self.vector_db.add_node_vector("Product", "description_vec", 
                                                       product_id, vector_data)
                assert success
            
            # Search similar vectors
            query_vector = [0.8] * 128
            results = self.vector_db.vector_similarity_search("Product", "description_vec", 
                                                            query_vector, k=2)
            
            assert len(results) == 2
            assert all(isinstance(score, float) for _, score in results)
            
            # Drop index
            result = self.vector_db.drop_node_vector_index("Product", "description_vec")
            assert result.indices_deleted == 1
            
            return True
        except Exception as e:
            pytest.fail(f"Vector index operations test failed: {e}")
    
    def test_similarity_calculations(self) -> bool:
        """Test vector similarity calculations"""
        try:
            vec1 = [1.0, 0.0, 0.0]
            vec2 = [0.0, 1.0, 0.0]
            vec3 = [1.0, 0.0, 0.0]  # Same as vec1
            
            # Test cosine similarity
            sim_orthogonal = self.similarity_calc.cosine_similarity(vec1, vec2)
            sim_identical = self.similarity_calc.cosine_similarity(vec1, vec3)
            
            assert abs(sim_orthogonal - 0.0) < 1e-10  # Orthogonal vectors
            assert abs(sim_identical - 1.0) < 1e-10   # Identical vectors
            
            # Test Euclidean distance
            dist_orthogonal = self.similarity_calc.euclidean_distance(vec1, vec2)
            dist_identical = self.similarity_calc.euclidean_distance(vec1, vec3)
            
            assert abs(dist_orthogonal - math.sqrt(2)) < 1e-10
            assert dist_identical == 0.0
            
            return True
        except Exception as e:
            pytest.fail(f"Similarity calculations test failed: {e}")
    
    def test_vecf32_parsing(self) -> bool:
        """Test vecf32 format parsing"""
        try:
            test_cases = [
                ("vecf32([1, -2, 3.14])", [1.0, -2.0, 3.14]),
                ("vecf32([0.5, 0.25, -0.75])", [0.5, 0.25, -0.75]),
                ("vecf32([100, -50, 75.5])", [100.0, -50.0, 75.5])
            ]
            
            for vecf32_str, expected in test_cases:
                parsed = self.vector_db.parse_vecf32(vecf32_str)
                assert len(parsed) == len(expected)
                for a, b in zip(parsed, expected):
                    assert abs(a - b) < 1e-10
            
            return True
        except Exception as e:
            pytest.fail(f"vecf32 parsing test failed: {e}")
    
    def test_vector_normalization(self) -> bool:
        """Test vector normalization"""
        try:
            test_vector = [3.0, 4.0, 0.0]  # Length = 5
            normalized = self.similarity_calc.normalize_vector(test_vector)
            
            # Check unit length
            length = math.sqrt(sum(x * x for x in normalized))
            assert abs(length - 1.0) < 1e-10
            
            # Check direction preservation
            assert abs(normalized[0] - 0.6) < 1e-10  # 3/5
            assert abs(normalized[1] - 0.8) < 1e-10  # 4/5
            assert abs(normalized[2] - 0.0) < 1e-10
            
            return True
        except Exception as e:
            pytest.fail(f"Vector normalization test failed: {e}")
    
    def test_high_dimensional_vectors(self) -> bool:
        """Test operations with high-dimensional vectors"""
        try:
            # Create high-dimensional index
            self.vector_db.create_node_vector_index("HighDim", "embedding", 
                                                   dim=1024, similarity_function="cosine")
            
            # Generate and add high-dimensional vectors
            for i in range(5):
                vector_data = self.similarity_calc.generate_random_vector(1024)
                success = self.vector_db.add_node_vector("HighDim", "embedding", 
                                                       f"item_{i}", vector_data)
                assert success
            
            # Perform similarity search
            query_vector = self.similarity_calc.generate_random_vector(1024)
            results = self.vector_db.vector_similarity_search("HighDim", "embedding", 
                                                            query_vector, k=3)
            
            assert len(results) == 3
            assert all(isinstance(score, float) for _, score in results)
            
            return True
        except Exception as e:
            pytest.fail(f"High dimensional vectors test failed: {e}")
    
    def test_multiple_similarity_functions(self) -> bool:
        """Test different similarity functions"""
        try:
            # Test cosine similarity index
            self.vector_db.create_node_vector_index("CosineTest", "vec", 
                                                   dim=3, similarity_function="cosine")
            
            # Test euclidean similarity index
            self.vector_db.create_node_vector_index("EuclideanTest", "vec", 
                                                   dim=3, similarity_function="euclidean")
            
            test_vec = [1.0, 0.0, 0.0]
            
            # Add same vector to both indices
            self.vector_db.add_node_vector("CosineTest", "vec", "item1", test_vec)
            self.vector_db.add_node_vector("EuclideanTest", "vec", "item1", test_vec)
            
            # Search with identical query (should return perfect match)
            query_vec = [1.0, 0.0, 0.0]
            
            cosine_results = self.vector_db.vector_similarity_search("CosineTest", "vec", 
                                                                   query_vec, k=1)
            euclidean_results = self.vector_db.vector_similarity_search("EuclideanTest", "vec", 
                                                                       query_vec, k=1)
            
            assert len(cosine_results) == 1
            assert len(euclidean_results) == 1
            assert abs(cosine_results[0][1] - 1.0) < 1e-10  # Perfect cosine similarity
            assert euclidean_results[0][1] == 0.0  # Zero distance (perfect match)
            
            return True
        except Exception as e:
            pytest.fail(f"Multiple similarity functions test failed: {e}")
    
    def run_comprehensive_tests(self) -> Dict[str, bool]:
        """Run all vector operation tests"""
        results = {}
        
        test_methods = [
            'test_vector_index_creation',
            'test_vector_index_operations',
            'test_similarity_calculations',
            'test_vecf32_parsing',
            'test_vector_normalization',
            'test_high_dimensional_vectors',
            'test_multiple_similarity_functions'
        ]
        
        for test_method in test_methods:
            try:
                # Reset for each test
                self.vector_db = MockVectorDatabase()
                self.setup_test_vectors()
                
                results[test_method] = getattr(self, test_method)()
            except Exception as e:
                results[test_method] = False
                print(f"{test_method} failed: {e}")
        
        return results


# Pytest integration patterns
class TestGraphVectorOperations:
    """Pytest test class for graph vector operations"""
    
    @pytest.fixture
    def framework(self):
        return GraphVectorTestFramework()
    
    def test_vector_lifecycle(self, framework):
        """Test complete vector index lifecycle"""
        # Create
        result = framework.vector_db.create_node_vector_index("Test", "embedding", 
                                                            dim=128, similarity_function="cosine")
        assert result.indices_created == 1
        
        # Add data
        test_vector = [1.0] * 128
        success = framework.vector_db.add_node_vector("Test", "embedding", "test_item", test_vector)
        assert success
        
        # Search
        results = framework.vector_db.vector_similarity_search("Test", "embedding", 
                                                             test_vector, k=1)
        assert len(results) == 1
        
        # Drop
        result = framework.vector_db.drop_node_vector_index("Test", "embedding")
        assert result.indices_deleted == 1
    
    def test_similarity_metrics(self, framework):
        """Test different similarity metrics"""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        
        cosine_sim = framework.similarity_calc.cosine_similarity(vec1, vec2)
        euclidean_dist = framework.similarity_calc.euclidean_distance(vec1, vec2)
        manhattan_dist = framework.similarity_calc.manhattan_distance(vec1, vec2)
        
        assert cosine_sim == 0.0  # Orthogonal
        assert abs(euclidean_dist - math.sqrt(2)) < 1e-10
        assert manhattan_dist == 2.0
    
    def test_vector_search_ranking(self, framework):
        """Test vector search result ranking"""
        framework.vector_db.create_node_vector_index("Ranking", "vec", 
                                                    dim=2, similarity_function="cosine")
        
        # Add vectors with known similarities to query
        framework.vector_db.add_node_vector("Ranking", "vec", "identical", [1.0, 0.0])
        framework.vector_db.add_node_vector("Ranking", "vec", "similar", [0.8, 0.6])
        framework.vector_db.add_node_vector("Ranking", "vec", "orthogonal", [0.0, 1.0])
        
        query = [1.0, 0.0]
        results = framework.vector_db.vector_similarity_search("Ranking", "vec", query, k=3)
        
        # Results should be ordered by similarity (highest first)
        assert results[0][0] == "identical"  # Perfect match first
        assert results[0][1] == 1.0  # Perfect cosine similarity
        assert results[1][1] > results[2][1]  # Similar > orthogonal