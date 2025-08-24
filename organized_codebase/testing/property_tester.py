"""
Property-Based Testing Engine for TestMaster
Generates and executes property-based tests to find edge cases
"""

import ast
import random
import string
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class PropertyType(Enum):
    """Types of properties to test"""
    INVARIANT = "invariant"  # Always true
    IDEMPOTENT = "idempotent"  # f(f(x)) = f(x)
    COMMUTATIVE = "commutative"  # f(a,b) = f(b,a)
    ASSOCIATIVE = "associative"  # f(f(a,b),c) = f(a,f(b,c))
    INVERSE = "inverse"  # f(f_inv(x)) = x
    MONOTONIC = "monotonic"  # x <= y => f(x) <= f(y)
    BOUNDED = "bounded"  # min <= f(x) <= max


@dataclass
class Property:
    """Represents a property to test"""
    name: str
    type: PropertyType
    predicate: Callable
    description: str
    examples: List[Any] = None
    counterexamples: List[Any] = None


@dataclass
class TestStrategy:
    """Strategy for generating test inputs"""
    name: str
    generator: Callable
    shrinker: Optional[Callable] = None
    
    def generate(self, size: int = 10) -> Any:
        """Generate test input"""
        return self.generator(size)
    
    def shrink(self, value: Any) -> List[Any]:
        """Shrink failing input to minimal case"""
        if self.shrinker:
            return self.shrinker(value)
        return []


class PropertyStrategies:
    """Collection of input generation strategies"""
    
    @staticmethod
    def integers(min_val: int = -1000, max_val: int = 1000) -> TestStrategy:
        """Generate random integers"""
        def gen(size):
            return random.randint(min_val, max_val)
        
        def shrink(val):
            candidates = []
            if val > 0:
                candidates.extend([0, val // 2, val - 1])
            elif val < 0:
                candidates.extend([0, val // 2, val + 1])
            return candidates
            
        return TestStrategy("integers", gen, shrink)
    
    @staticmethod
    def floats(min_val: float = -1000.0, max_val: float = 1000.0) -> TestStrategy:
        """Generate random floats"""
        def gen(size):
            return random.uniform(min_val, max_val)
        
        def shrink(val):
            return [0.0, val / 2, round(val)]
            
        return TestStrategy("floats", gen, shrink)
    
    @staticmethod
    def strings(alphabet: str = string.printable, max_size: int = 100) -> TestStrategy:
        """Generate random strings"""
        def gen(size):
            length = random.randint(0, min(size, max_size))
            return ''.join(random.choice(alphabet) for _ in range(length))
        
        def shrink(val):
            if not val:
                return []
            return ['', val[:len(val)//2], val[1:], val[:-1]]
            
        return TestStrategy("strings", gen, shrink)
    
    @staticmethod
    def lists(element_strategy: TestStrategy, max_size: int = 100) -> TestStrategy:
        """Generate lists of elements"""
        def gen(size):
            length = random.randint(0, min(size, max_size))
            return [element_strategy.generate(size) for _ in range(length)]
        
        def shrink(val):
            if not val:
                return []
            candidates = [[], val[:len(val)//2], val[1:], val[:-1]]
            # Also try shrinking individual elements
            for i, elem in enumerate(val):
                for shrunk in element_strategy.shrink(elem):
                    candidate = val.copy()
                    candidate[i] = shrunk
                    candidates.append(candidate)
            return candidates
            
        return TestStrategy("lists", gen, shrink)
    
    @staticmethod
    def dictionaries(key_strategy: TestStrategy, value_strategy: TestStrategy) -> TestStrategy:
        """Generate dictionaries"""
        def gen(size):
            length = random.randint(0, size)
            return {
                key_strategy.generate(size): value_strategy.generate(size)
                for _ in range(length)
            }
        
        def shrink(val):
            if not val:
                return []
            candidates = [{}]
            # Remove keys
            for key in val:
                candidate = val.copy()
                del candidate[key]
                candidates.append(candidate)
            return candidates
            
        return TestStrategy("dictionaries", gen, shrink)


class PropertyTester:
    """Main property-based testing engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_examples = self.config.get('max_examples', 100)
        self.max_shrink_steps = self.config.get('max_shrink_steps', 100)
        self.strategies = PropertyStrategies()
        
    def test_property(self, prop: Property, strategy: TestStrategy, 
                     target_func: Callable) -> Dict[str, Any]:
        """Test a property with generated inputs"""
        passed = 0
        failed = 0
        errors = []
        
        for _ in range(self.max_examples):
            try:
                test_input = strategy.generate()
                
                if prop.type == PropertyType.INVARIANT:
                    result = self._test_invariant(prop, target_func, test_input)
                elif prop.type == PropertyType.IDEMPOTENT:
                    result = self._test_idempotent(prop, target_func, test_input)
                elif prop.type == PropertyType.COMMUTATIVE:
                    result = self._test_commutative(prop, target_func, test_input)
                elif prop.type == PropertyType.INVERSE:
                    result = self._test_inverse(prop, target_func, test_input)
                elif prop.type == PropertyType.MONOTONIC:
                    result = self._test_monotonic(prop, target_func, test_input)
                else:
                    result = prop.predicate(target_func(test_input))
                
                if result:
                    passed += 1
                else:
                    failed += 1
                    # Try to shrink the failing input
                    minimal = self._shrink_failure(
                        prop, strategy, target_func, test_input
                    )
                    errors.append({
                        'input': test_input,
                        'minimal': minimal,
                        'property': prop.name
                    })
                    
            except Exception as e:
                errors.append({
                    'input': test_input,
                    'error': str(e),
                    'property': prop.name
                })
                failed += 1
        
        return {
            'property': prop.name,
            'type': prop.type.value,
            'passed': passed,
            'failed': failed,
            'total': passed + failed,
            'success_rate': (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 0,
            'errors': errors[:10]  # Limit error reporting
        }
    
    def _test_invariant(self, prop: Property, func: Callable, input_val: Any) -> bool:
        """Test invariant property"""
        result = func(input_val)
        return prop.predicate(result)
    
    def _test_idempotent(self, prop: Property, func: Callable, input_val: Any) -> bool:
        """Test idempotent property: f(f(x)) = f(x)"""
        result1 = func(input_val)
        result2 = func(result1)
        return result1 == result2
    
    def _test_commutative(self, prop: Property, func: Callable, inputs: Tuple) -> bool:
        """Test commutative property: f(a,b) = f(b,a)"""
        if len(inputs) < 2:
            return True
        a, b = inputs[:2]
        return func(a, b) == func(b, a)
    
    def _test_inverse(self, prop: Property, func: Callable, input_val: Any) -> bool:
        """Test inverse property with provided inverse function"""
        # Assumes prop.predicate is the inverse function
        result = func(input_val)
        inversed = prop.predicate(result)
        return inversed == input_val
    
    def _test_monotonic(self, prop: Property, func: Callable, inputs: List) -> bool:
        """Test monotonic property"""
        if len(inputs) < 2:
            return True
        sorted_inputs = sorted(inputs)
        results = [func(x) for x in sorted_inputs]
        return all(results[i] <= results[i+1] for i in range(len(results)-1))
    
    def _shrink_failure(self, prop: Property, strategy: TestStrategy,
                       func: Callable, failing_input: Any) -> Any:
        """Shrink failing input to minimal case"""
        current = failing_input
        
        for _ in range(self.max_shrink_steps):
            candidates = strategy.shrink(current)
            found_smaller = False
            
            for candidate in candidates:
                try:
                    # Test if candidate still fails
                    if prop.type == PropertyType.INVARIANT:
                        result = not prop.predicate(func(candidate))
                    else:
                        result = not self._test_property_single(
                            prop, func, candidate
                        )
                    
                    if result:  # Still fails
                        current = candidate
                        found_smaller = True
                        break
                except:
                    continue
                    
            if not found_smaller:
                break
                
        return current
    
    def _test_property_single(self, prop: Property, func: Callable, 
                              input_val: Any) -> bool:
        """Test property with single input"""
        if prop.type == PropertyType.IDEMPOTENT:
            return self._test_idempotent(prop, func, input_val)
        elif prop.type == PropertyType.INVARIANT:
            return self._test_invariant(prop, func, input_val)
        else:
            return prop.predicate(func(input_val))
    
    def discover_properties(self, func: Callable, 
                           sample_inputs: List[Any]) -> List[Property]:
        """Automatically discover properties from function behavior"""
        properties = []
        
        # Test for idempotency
        idempotent = True
        for inp in sample_inputs[:10]:
            try:
                r1 = func(inp)
                r2 = func(r1)
                if r1 != r2:
                    idempotent = False
                    break
            except:
                pass
        
        if idempotent:
            properties.append(Property(
                name=f"{func.__name__}_idempotent",
                type=PropertyType.IDEMPOTENT,
                predicate=lambda x: True,
                description=f"{func.__name__} is idempotent"
            ))
        
        return properties