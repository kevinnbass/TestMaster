#!/usr/bin/env python3
"""
Simple test to verify linting integration works
"""

def test_simple_function() -> bool:
    """Simple test function under 60 lines"""
    assert True is True
    assert False is False
    assert 1 == 1
    return True

def test_with_bounds_checking() -> bool:
    """Test function with proper bounds checking"""
    items = [1, 2, 3, 4, 5]
    max_items = 10

    # Simple for loop with bounds checking
    result = []
    for i in range(len(items)):
        if i >= max_items:
            break
        result.append(items[i] * 2)

    assert len(result) <= max_items
    return True

if __name__ == "__main__":
    print("Running linting test...")
    print(f"Simple function: {test_simple_function()}")
    print(f"Bounds checking: {test_with_bounds_checking()}")
    print("Linting test completed successfully!")
