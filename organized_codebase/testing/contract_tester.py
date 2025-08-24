"""
Contract and Invariant Testing Framework for TestMaster
Verifies contracts, invariants, and design-by-contract principles
"""

import ast
import inspect
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import functools


class ContractType(Enum):
    """Types of contracts"""
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"
    ASSERTION = "assertion"
    CONSTRAINT = "constraint"


@dataclass
class Contract:
    """Represents a contract specification"""
    name: str
    type: ContractType
    predicate: Callable
    description: str
    error_message: Optional[str] = None
    severity: str = "error"  # error, warning, info


@dataclass
class InvariantViolation:
    """Records an invariant violation"""
    contract_name: str
    violation_type: str
    input_values: Dict[str, Any]
    output_value: Any
    error_message: str
    stack_trace: List[str]


class ContractValidator:
    """Validates contracts at runtime"""
    
    def __init__(self):
        self.contracts: Dict[str, List[Contract]] = {}
        self.violations: List[InvariantViolation] = []
        self.enabled = True
        
    def precondition(self, predicate: Callable, message: str = None):
        """Decorator for preconditions"""
        def decorator(func):
            contract = Contract(
                name=f"{func.__name__}_precondition",
                type=ContractType.PRECONDITION,
                predicate=predicate,
                description=f"Precondition for {func.__name__}",
                error_message=message
            )
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if self.enabled:
                    if not predicate(*args, **kwargs):
                        self._handle_violation(contract, args, kwargs, None)
                return func(*args, **kwargs)
            
            self._register_contract(func.__name__, contract)
            return wrapper
        return decorator
    
    def postcondition(self, predicate: Callable, message: str = None):
        """Decorator for postconditions"""
        def decorator(func):
            contract = Contract(
                name=f"{func.__name__}_postcondition",
                type=ContractType.POSTCONDITION,
                predicate=predicate,
                description=f"Postcondition for {func.__name__}",
                error_message=message
            )
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                if self.enabled:
                    if not predicate(result, *args, **kwargs):
                        self._handle_violation(contract, args, kwargs, result)
                return result
            
            self._register_contract(func.__name__, contract)
            return wrapper
        return decorator
    
    def invariant(self, predicate: Callable, message: str = None):
        """Decorator for class invariants"""
        def decorator(cls):
            contract = Contract(
                name=f"{cls.__name__}_invariant",
                type=ContractType.INVARIANT,
                predicate=predicate,
                description=f"Invariant for {cls.__name__}",
                error_message=message
            )
            
            # Wrap all public methods
            for name, method in inspect.getmembers(cls, inspect.isfunction):
                if not name.startswith('_'):
                    setattr(cls, name, self._wrap_with_invariant(method, contract))
            
            self._register_contract(cls.__name__, contract)
            return cls
        return decorator
    
    def _wrap_with_invariant(self, method: Callable, contract: Contract):
        """Wrap method with invariant checking"""
        @functools.wraps(method)
        def wrapper(self_obj, *args, **kwargs):
            # Check invariant before
            if self.enabled and not contract.predicate(self_obj):
                self._handle_violation(contract, args, kwargs, None)
            
            result = method(self_obj, *args, **kwargs)
            
            # Check invariant after
            if self.enabled and not contract.predicate(self_obj):
                self._handle_violation(contract, args, kwargs, result)
            
            return result
        return wrapper
    
    def _register_contract(self, name: str, contract: Contract):
        """Register a contract"""
        if name not in self.contracts:
            self.contracts[name] = []
        self.contracts[name].append(contract)
    
    def _handle_violation(self, contract: Contract, args, kwargs, result):
        """Handle contract violation"""
        import traceback
        
        violation = InvariantViolation(
            contract_name=contract.name,
            violation_type=contract.type.value,
            input_values={'args': args, 'kwargs': kwargs},
            output_value=result,
            error_message=contract.error_message or f"Contract {contract.name} violated",
            stack_trace=traceback.format_stack()
        )
        
        self.violations.append(violation)
        
        if contract.severity == "error":
            raise AssertionError(violation.error_message)
        elif contract.severity == "warning":
            print(f"WARNING: {violation.error_message}")


class ContractTester:
    """Main contract testing framework"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validator = ContractValidator()
        self.discovered_contracts = []
        
    def discover_contracts(self, source_code: str) -> List[Contract]:
        """Discover implicit contracts from code"""
        contracts = []
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                # Find assertions
                if isinstance(node, ast.Assert):
                    contracts.append(self._extract_assertion_contract(node))
                
                # Find type hints that imply contracts
                elif isinstance(node, ast.FunctionDef):
                    contracts.extend(self._extract_type_contracts(node))
                
                # Find docstring contracts
                if hasattr(node, 'body') and node.body:
                    first = node.body[0]
                    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                        if isinstance(first.value.value, str):
                            contracts.extend(self._extract_docstring_contracts(first.value.value))
                            
        except SyntaxError:
            pass
        
        self.discovered_contracts = contracts
        return contracts
    
    def _extract_assertion_contract(self, node: ast.Assert) -> Contract:
        """Extract contract from assertion"""
        return Contract(
            name=f"assertion_line_{node.lineno}",
            type=ContractType.ASSERTION,
            predicate=lambda: True,  # Placeholder
            description=ast.unparse(node.test) if hasattr(ast, 'unparse') else "Assertion",
            error_message=ast.unparse(node.msg) if node.msg else None
        )
    
    def _extract_type_contracts(self, node: ast.FunctionDef) -> List[Contract]:
        """Extract contracts from type hints"""
        contracts = []
        
        # Parameter type contracts
        for arg in node.args.args:
            if arg.annotation:
                contract = Contract(
                    name=f"{node.name}_{arg.arg}_type",
                    type=ContractType.PRECONDITION,
                    predicate=lambda x: True,  # Type check placeholder
                    description=f"Type contract for {arg.arg}"
                )
                contracts.append(contract)
        
        # Return type contract
        if node.returns:
            contract = Contract(
                name=f"{node.name}_return_type",
                type=ContractType.POSTCONDITION,
                predicate=lambda x: True,  # Type check placeholder
                description=f"Return type contract for {node.name}"
            )
            contracts.append(contract)
            
        return contracts
    
    def _extract_docstring_contracts(self, docstring: str) -> List[Contract]:
        """Extract contracts from docstring"""
        contracts = []
        
        # Look for contract specifications in docstring
        import re
        
        # Preconditions
        pre_pattern = r'(?:Requires?|Pre(?:condition)?s?):\s*(.+?)(?:\n\n|\Z)'
        pre_matches = re.findall(pre_pattern, docstring, re.IGNORECASE | re.DOTALL)
        for match in pre_matches:
            contracts.append(Contract(
                name=f"docstring_precondition",
                type=ContractType.PRECONDITION,
                predicate=lambda: True,
                description=match.strip()
            ))
        
        # Postconditions
        post_pattern = r'(?:Ensures?|Post(?:condition)?s?):\s*(.+?)(?:\n\n|\Z)'
        post_matches = re.findall(post_pattern, docstring, re.IGNORECASE | re.DOTALL)
        for match in post_matches:
            contracts.append(Contract(
                name=f"docstring_postcondition",
                type=ContractType.POSTCONDITION,
                predicate=lambda: True,
                description=match.strip()
            ))
        
        # Invariants
        inv_pattern = r'(?:Invariants?):\s*(.+?)(?:\n\n|\Z)'
        inv_matches = re.findall(inv_pattern, docstring, re.IGNORECASE | re.DOTALL)
        for match in inv_matches:
            contracts.append(Contract(
                name=f"docstring_invariant",
                type=ContractType.INVARIANT,
                predicate=lambda: True,
                description=match.strip()
            ))
            
        return contracts
    
    def generate_contract_tests(self, contracts: List[Contract]) -> str:
        """Generate test code for contracts"""
        test_code = []
        test_code.append("import unittest")
        test_code.append("")
        test_code.append("class ContractTests(unittest.TestCase):")
        
        for i, contract in enumerate(contracts):
            test_code.append(f"    def test_contract_{i}_{contract.type.value}(self):")
            test_code.append(f'        """Test {contract.description}"""')
            test_code.append(f"        # Contract: {contract.name}")
            test_code.append(f"        # Type: {contract.type.value}")
            
            if contract.type == ContractType.PRECONDITION:
                test_code.append(f"        # Test invalid inputs should raise")
                test_code.append(f"        with self.assertRaises(AssertionError):")
                test_code.append(f"            pass  # Add invalid input test")
                
            elif contract.type == ContractType.POSTCONDITION:
                test_code.append(f"        # Test output satisfies condition")
                test_code.append(f"        result = None  # Call function")
                test_code.append(f"        self.assertTrue(True)  # Check postcondition")
                
            elif contract.type == ContractType.INVARIANT:
                test_code.append(f"        # Test invariant holds")
                test_code.append(f"        obj = None  # Create object")
                test_code.append(f"        self.assertTrue(True)  # Check invariant")
                
            test_code.append("")
        
        return "\n".join(test_code)
    
    def verify_contracts(self, func: Callable, test_inputs: List[Tuple]) -> Dict[str, Any]:
        """Verify contracts with test inputs"""
        results = {
            'total_tests': len(test_inputs),
            'passed': 0,
            'failed': 0,
            'violations': []
        }
        
        for inputs in test_inputs:
            try:
                if isinstance(inputs, tuple):
                    func(*inputs)
                else:
                    func(inputs)
                results['passed'] += 1
            except AssertionError as e:
                results['failed'] += 1
                results['violations'].append({
                    'input': inputs,
                    'error': str(e)
                })
        
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate contract testing report"""
        return {
            'discovered_contracts': len(self.discovered_contracts),
            'violations': len(self.validator.violations),
            'contract_types': {
                ctype.value: sum(1 for c in self.discovered_contracts if c.type == ctype)
                for ctype in ContractType
            },
            'violation_details': [
                {
                    'contract': v.contract_name,
                    'type': v.violation_type,
                    'message': v.error_message
                }
                for v in self.validator.violations[:10]
            ]
        }