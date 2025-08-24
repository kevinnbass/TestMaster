"""
Mutation Testing Engine for TestMaster
Generates code mutations and evaluates test suite effectiveness
"""

import ast
import copy
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class MutationType(Enum):
    """Types of mutations that can be applied"""
    ARITHMETIC = "arithmetic"
    COMPARISON = "comparison"
    LOGICAL = "logical"
    CONSTANT = "constant"
    STRING = "string"
    RETURN = "return"
    DELETION = "deletion"


@dataclass
class Mutation:
    """Represents a single mutation"""
    type: MutationType
    location: Tuple[int, int]  # (line, column)
    original: str
    mutated: str
    description: str
    killed: bool = False
    test_that_killed: Optional[str] = None


class MutationOperators:
    """Collection of mutation operators"""
    
    ARITHMETIC_OPS = {
        ast.Add: [ast.Sub, ast.Mult, ast.Div],
        ast.Sub: [ast.Add, ast.Mult, ast.Div],
        ast.Mult: [ast.Add, ast.Sub, ast.Div],
        ast.Div: [ast.Add, ast.Sub, ast.Mult],
        ast.Mod: [ast.Add, ast.Sub, ast.Mult],
        ast.Pow: [ast.Mult, ast.Div]
    }
    
    COMPARISON_OPS = {
        ast.Eq: [ast.NotEq, ast.Lt, ast.Gt],
        ast.NotEq: [ast.Eq, ast.Lt, ast.Gt],
        ast.Lt: [ast.LtE, ast.Gt, ast.GtE, ast.Eq],
        ast.LtE: [ast.Lt, ast.Gt, ast.GtE],
        ast.Gt: [ast.GtE, ast.Lt, ast.LtE, ast.Eq],
        ast.GtE: [ast.Gt, ast.Lt, ast.LtE]
    }
    
    LOGICAL_OPS = {
        ast.And: [ast.Or],
        ast.Or: [ast.And],
        ast.Not: []  # Special handling
    }


class MutationEngine:
    """Main mutation testing engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_mutations = self.config.get('max_mutations', 100)
        self.mutation_types = self.config.get('mutation_types', list(MutationType))
        self.operators = MutationOperators()
        
    def generate_mutations(self, source_code: str) -> List[Mutation]:
        """Generate mutations for source code"""
        try:
            tree = ast.parse(source_code)
            mutations = []
            
            for node in ast.walk(tree):
                if len(mutations) >= self.max_mutations:
                    break
                    
                # Arithmetic mutations
                if isinstance(node, ast.BinOp) and MutationType.ARITHMETIC in self.mutation_types:
                    mutations.extend(self._mutate_arithmetic(node, source_code))
                    
                # Comparison mutations
                elif isinstance(node, ast.Compare) and MutationType.COMPARISON in self.mutation_types:
                    mutations.extend(self._mutate_comparison(node, source_code))
                    
                # Logical mutations
                elif isinstance(node, ast.BoolOp) and MutationType.LOGICAL in self.mutation_types:
                    mutations.extend(self._mutate_logical(node, source_code))
                    
                # Constant mutations
                elif isinstance(node, ast.Constant) and MutationType.CONSTANT in self.mutation_types:
                    mutations.extend(self._mutate_constant(node, source_code))
                    
                # Return mutations
                elif isinstance(node, ast.Return) and MutationType.RETURN in self.mutation_types:
                    mutations.extend(self._mutate_return(node, source_code))
                    
            return mutations[:self.max_mutations]
            
        except SyntaxError:
            return []
    
    def _mutate_arithmetic(self, node: ast.BinOp, source: str) -> List[Mutation]:
        """Generate arithmetic operation mutations"""
        mutations = []
        op_class = type(node.op)
        
        if op_class in self.operators.ARITHMETIC_OPS:
            for new_op_class in self.operators.ARITHMETIC_OPS[op_class]:
                mutation = Mutation(
                    type=MutationType.ARITHMETIC,
                    location=(node.lineno, node.col_offset),
                    original=ast.unparse(node),
                    mutated=self._apply_operator_mutation(node, new_op_class()),
                    description=f"Replace {op_class.__name__} with {new_op_class.__name__}"
                )
                mutations.append(mutation)
                
        return mutations
    
    def _mutate_comparison(self, node: ast.Compare, source: str) -> List[Mutation]:
        """Generate comparison operation mutations"""
        mutations = []
        
        for op in node.ops:
            op_class = type(op)
            if op_class in self.operators.COMPARISON_OPS:
                for new_op_class in self.operators.COMPARISON_OPS[op_class]:
                    mutation = Mutation(
                        type=MutationType.COMPARISON,
                        location=(node.lineno, node.col_offset),
                        original=ast.unparse(node),
                        mutated=self._apply_comparison_mutation(node, new_op_class()),
                        description=f"Replace {op_class.__name__} with {new_op_class.__name__}"
                    )
                    mutations.append(mutation)
                    
        return mutations
    
    def _mutate_logical(self, node: ast.BoolOp, source: str) -> List[Mutation]:
        """Generate logical operation mutations"""
        mutations = []
        op_class = type(node.op)
        
        if op_class in self.operators.LOGICAL_OPS:
            for new_op_class in self.operators.LOGICAL_OPS[op_class]:
                mutation = Mutation(
                    type=MutationType.LOGICAL,
                    location=(node.lineno, node.col_offset),
                    original=ast.unparse(node),
                    mutated=self._apply_logical_mutation(node, new_op_class()),
                    description=f"Replace {op_class.__name__} with {new_op_class.__name__}"
                )
                mutations.append(mutation)
                
        return mutations
    
    def _mutate_constant(self, node: ast.Constant, source: str) -> List[Mutation]:
        """Generate constant value mutations"""
        mutations = []
        
        if isinstance(node.value, (int, float)):
            # Numeric mutations
            mutated_values = [
                node.value + 1,
                node.value - 1,
                node.value * 2,
                0 if node.value != 0 else 1,
                -node.value
            ]
            
            for val in mutated_values[:2]:  # Limit mutations
                mutation = Mutation(
                    type=MutationType.CONSTANT,
                    location=(node.lineno, node.col_offset),
                    original=str(node.value),
                    mutated=str(val),
                    description=f"Replace {node.value} with {val}"
                )
                mutations.append(mutation)
                
        elif isinstance(node.value, str) and node.value:
            # String mutations
            mutation = Mutation(
                type=MutationType.STRING,
                location=(node.lineno, node.col_offset),
                original=repr(node.value),
                mutated='""',
                description=f"Replace string with empty"
            )
            mutations.append(mutation)
            
        elif isinstance(node.value, bool):
            # Boolean mutations
            mutation = Mutation(
                type=MutationType.CONSTANT,
                location=(node.lineno, node.col_offset),
                original=str(node.value),
                mutated=str(not node.value),
                description=f"Negate boolean value"
            )
            mutations.append(mutation)
            
        return mutations
    
    def _mutate_return(self, node: ast.Return, source: str) -> List[Mutation]:
        """Generate return statement mutations"""
        mutations = []
        
        if node.value:
            # Replace with None
            mutation = Mutation(
                type=MutationType.RETURN,
                location=(node.lineno, node.col_offset),
                original=ast.unparse(node),
                mutated="return None",
                description="Replace return value with None"
            )
            mutations.append(mutation)
            
        return mutations
    
    def _apply_operator_mutation(self, node: ast.BinOp, new_op: ast.operator) -> str:
        """Apply operator mutation to binary operation"""
        mutated = copy.deepcopy(node)
        mutated.op = new_op
        return ast.unparse(mutated)
    
    def _apply_comparison_mutation(self, node: ast.Compare, new_op: ast.cmpop) -> str:
        """Apply comparison mutation"""
        mutated = copy.deepcopy(node)
        mutated.ops = [new_op]
        return ast.unparse(mutated)
    
    def _apply_logical_mutation(self, node: ast.BoolOp, new_op: ast.boolop) -> str:
        """Apply logical operation mutation"""
        mutated = copy.deepcopy(node)
        mutated.op = new_op
        return ast.unparse(mutated)
    
    def calculate_mutation_score(self, mutations: List[Mutation]) -> float:
        """Calculate mutation score (killed/total)"""
        if not mutations:
            return 0.0
        killed = sum(1 for m in mutations if m.killed)
        return (killed / len(mutations)) * 100
    
    def generate_report(self, mutations: List[Mutation]) -> Dict[str, Any]:
        """Generate mutation testing report"""
        killed = [m for m in mutations if m.killed]
        survived = [m for m in mutations if not m.killed]
        
        by_type = {}
        for mt in MutationType:
            type_mutations = [m for m in mutations if m.type == mt]
            if type_mutations:
                by_type[mt.value] = {
                    'total': len(type_mutations),
                    'killed': sum(1 for m in type_mutations if m.killed),
                    'survived': sum(1 for m in type_mutations if not m.killed)
                }
        
        return {
            'total_mutations': len(mutations),
            'killed': len(killed),
            'survived': len(survived),
            'mutation_score': self.calculate_mutation_score(mutations),
            'by_type': by_type,
            'survived_mutations': [
                {
                    'type': m.type.value,
                    'location': m.location,
                    'original': m.original,
                    'mutated': m.mutated,
                    'description': m.description
                }
                for m in survived[:10]  # Top 10 survived
            ]
        }