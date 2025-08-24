"""
Intelligent Fuzzing Engine for TestMaster
Generates random inputs to discover bugs and edge cases
"""

import random
import string
import struct
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json


class FuzzStrategy(Enum):
    """Fuzzing strategies"""
    RANDOM = "random"
    BOUNDARY = "boundary"
    MUTATION = "mutation"
    GRAMMAR = "grammar"
    SMART = "smart"


@dataclass
class FuzzTarget:
    """Target for fuzzing"""
    name: str
    input_type: str
    constraints: Dict[str, Any]
    known_crashes: List[Any] = None
    coverage_info: Dict[str, Any] = None


@dataclass 
class FuzzResult:
    """Result from fuzzing run"""
    input_data: Any
    output: Any
    crashed: bool
    error: Optional[str]
    execution_time: float
    memory_usage: Optional[int]
    coverage_increase: float = 0.0


class FuzzGenerators:
    """Collection of fuzz input generators"""
    
    @staticmethod
    def random_bytes(size: int = 100) -> bytes:
        """Generate random bytes"""
        return bytes(random.randint(0, 255) for _ in range(size))
    
    @staticmethod
    def random_string(size: int = 100, charset: str = None) -> str:
        """Generate random string"""
        if charset is None:
            charset = string.printable
        return ''.join(random.choice(charset) for _ in range(size))
    
    @staticmethod
    def random_int(min_val: int = -2**31, max_val: int = 2**31-1) -> int:
        """Generate random integer"""
        return random.randint(min_val, max_val)
    
    @staticmethod
    def random_float(min_val: float = -1e10, max_val: float = 1e10) -> float:
        """Generate random float"""
        return random.uniform(min_val, max_val)
    
    @staticmethod
    def boundary_values(dtype: str) -> List[Any]:
        """Generate boundary test values"""
        if dtype == "int":
            return [
                0, 1, -1, 
                2**31-1, -2**31,  # 32-bit boundaries
                2**63-1, -2**63,  # 64-bit boundaries
                255, -128, 127,  # Byte boundaries
                65535, -32768, 32767  # Short boundaries
            ]
        elif dtype == "float":
            return [
                0.0, 1.0, -1.0,
                float('inf'), float('-inf'), float('nan'),
                1e-308, 1e308,  # Near limits
                3.14159, 2.71828  # Common values
            ]
        elif dtype == "string":
            return [
                "", " ", "\n", "\t", "\r",
                "A" * 1000,  # Long string
                "\x00", "\xff",  # Null and max byte
                "'; DROP TABLE;", "<script>",  # Injection attempts
                "\\", "/", ".", "..",  # Path traversal
                "%s%s%s", "${jndi:ldap://}"  # Format string, JNDI
            ]
        return []
    
    @staticmethod
    def mutate_bytes(data: bytes, num_mutations: int = 5) -> bytes:
        """Mutate byte array"""
        if not data:
            return FuzzGenerators.random_bytes(10)
            
        result = bytearray(data)
        for _ in range(num_mutations):
            mutation_type = random.choice([
                'flip_bit', 'flip_byte', 'insert', 'delete', 'shuffle'
            ])
            
            if mutation_type == 'flip_bit' and len(result) > 0:
                pos = random.randint(0, len(result) - 1)
                bit = random.randint(0, 7)
                result[pos] ^= (1 << bit)
                
            elif mutation_type == 'flip_byte' and len(result) > 0:
                pos = random.randint(0, len(result) - 1)
                result[pos] = random.randint(0, 255)
                
            elif mutation_type == 'insert':
                pos = random.randint(0, len(result))
                result.insert(pos, random.randint(0, 255))
                
            elif mutation_type == 'delete' and len(result) > 1:
                pos = random.randint(0, len(result) - 1)
                del result[pos]
                
            elif mutation_type == 'shuffle' and len(result) > 2:
                start = random.randint(0, len(result) - 2)
                end = random.randint(start + 1, min(start + 10, len(result)))
                chunk = result[start:end]
                random.shuffle(chunk)
                result[start:end] = chunk
                
        return bytes(result)


class IntelligentFuzzer:
    """Main fuzzing engine with intelligent features"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.timeout = self.config.get('timeout', 1.0)
        self.generators = FuzzGenerators()
        self.corpus = []  # Interesting inputs
        self.crashes = []  # Inputs that caused crashes
        self.coverage_map = {}
        
    def fuzz(self, target_func: Callable, target: FuzzTarget,
            strategy: FuzzStrategy = FuzzStrategy.SMART) -> List[FuzzResult]:
        """Run fuzzing campaign"""
        results = []
        
        if strategy == FuzzStrategy.RANDOM:
            results = self._random_fuzz(target_func, target)
        elif strategy == FuzzStrategy.BOUNDARY:
            results = self._boundary_fuzz(target_func, target)
        elif strategy == FuzzStrategy.MUTATION:
            results = self._mutation_fuzz(target_func, target)
        elif strategy == FuzzStrategy.SMART:
            results = self._smart_fuzz(target_func, target)
            
        return results
    
    def _random_fuzz(self, func: Callable, target: FuzzTarget) -> List[FuzzResult]:
        """Pure random fuzzing"""
        results = []
        
        for _ in range(self.max_iterations):
            input_data = self._generate_input(target.input_type)
            result = self._execute_target(func, input_data)
            results.append(result)
            
            if result.crashed:
                self.crashes.append(input_data)
                
        return results
    
    def _boundary_fuzz(self, func: Callable, target: FuzzTarget) -> List[FuzzResult]:
        """Boundary value fuzzing"""
        results = []
        boundary_values = self.generators.boundary_values(target.input_type)
        
        for value in boundary_values:
            result = self._execute_target(func, value)
            results.append(result)
            
            if result.crashed:
                self.crashes.append(value)
                
        # Continue with random after boundaries
        remaining = self.max_iterations - len(boundary_values)
        if remaining > 0:
            for _ in range(remaining):
                input_data = self._generate_input(target.input_type)
                result = self._execute_target(func, input_data)
                results.append(result)
                
        return results
    
    def _mutation_fuzz(self, func: Callable, target: FuzzTarget) -> List[FuzzResult]:
        """Mutation-based fuzzing"""
        results = []
        
        # Start with some seed inputs
        if not self.corpus:
            self.corpus = [
                self._generate_input(target.input_type) 
                for _ in range(10)
            ]
        
        for _ in range(self.max_iterations):
            # Pick random input from corpus
            base_input = random.choice(self.corpus)
            
            # Mutate it
            if isinstance(base_input, bytes):
                mutated = self.generators.mutate_bytes(base_input)
            elif isinstance(base_input, str):
                mutated = self._mutate_string(base_input)
            elif isinstance(base_input, (int, float)):
                mutated = self._mutate_number(base_input)
            else:
                mutated = base_input
                
            result = self._execute_target(func, mutated)
            results.append(result)
            
            # Add interesting inputs to corpus
            if result.crashed:
                self.crashes.append(mutated)
            elif result.coverage_increase > 0:
                self.corpus.append(mutated)
                
        return results
    
    def _smart_fuzz(self, func: Callable, target: FuzzTarget) -> List[FuzzResult]:
        """Smart fuzzing with coverage guidance"""
        results = []
        
        # Phase 1: Boundary testing
        boundary_results = self._boundary_fuzz(func, target)
        results.extend(boundary_results[:50])
        
        # Phase 2: Grammar-based if applicable
        if target.input_type in ["json", "xml", "sql"]:
            grammar_results = self._grammar_fuzz(func, target)
            results.extend(grammar_results[:50])
        
        # Phase 3: Coverage-guided mutation
        mutation_results = self._mutation_fuzz(func, target)
        results.extend(mutation_results)
        
        return results[:self.max_iterations]
    
    def _grammar_fuzz(self, func: Callable, target: FuzzTarget) -> List[FuzzResult]:
        """Grammar-based fuzzing for structured inputs"""
        results = []
        
        if target.input_type == "json":
            for _ in range(50):
                json_data = self._generate_json()
                result = self._execute_target(func, json.dumps(json_data))
                results.append(result)
                
        return results
    
    def _generate_input(self, input_type: str) -> Any:
        """Generate input based on type"""
        if input_type == "bytes":
            return self.generators.random_bytes()
        elif input_type == "string":
            return self.generators.random_string()
        elif input_type == "int":
            return self.generators.random_int()
        elif input_type == "float":
            return self.generators.random_float()
        else:
            return None
    
    def _mutate_string(self, s: str) -> str:
        """Mutate string input"""
        if not s:
            return self.generators.random_string(10)
            
        mutations = []
        # Character mutations
        if len(s) > 0:
            pos = random.randint(0, len(s) - 1)
            mutations.append(s[:pos] + chr(random.randint(0, 127)) + s[pos+1:])
        # Insertions
        mutations.append(s + self.generators.random_string(5))
        # Deletions
        if len(s) > 1:
            mutations.append(s[:-1])
        # Repetitions
        mutations.append(s * 2)
        
        return random.choice(mutations)
    
    def _mutate_number(self, n: Union[int, float]) -> Union[int, float]:
        """Mutate numeric input"""
        mutations = [
            n + 1, n - 1, n * 2, n // 2 if n != 0 else 1,
            -n, n + random.randint(-100, 100)
        ]
        return random.choice(mutations)
    
    def _generate_json(self) -> Dict:
        """Generate random JSON structure"""
        return {
            "field1": random.choice([None, True, False]),
            "field2": random.randint(-1000, 1000),
            "field3": self.generators.random_string(20),
            "nested": {
                "data": [random.random() for _ in range(5)]
            }
        }
    
    def _execute_target(self, func: Callable, input_data: Any) -> FuzzResult:
        """Execute target function with input"""
        import time
        
        crashed = False
        error = None
        output = None
        start_time = time.time()
        
        try:
            output = func(input_data)
        except Exception as e:
            crashed = True
            error = str(e)
            
        execution_time = time.time() - start_time
        
        return FuzzResult(
            input_data=input_data,
            output=output,
            crashed=crashed,
            error=error,
            execution_time=execution_time,
            memory_usage=None
        )