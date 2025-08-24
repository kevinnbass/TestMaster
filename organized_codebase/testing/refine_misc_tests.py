#!/usr/bin/env python3
"""
Refine the oversized test_misc.py file into better categories
"""

import os
import re
from pathlib import Path

def analyze_misc_file():
    """Analyze the misc file and create better categorization"""
    misc_file = "C:/Users/kbass/OneDrive/Documents/testmaster/TestMaster/tests/modularized/test_misc.py"
    
    with open(misc_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all class names in the misc file
    class_pattern = r'class (Test\w+_Tests):'
    classes = re.findall(class_pattern, content)
    
    print(f"Found {len(classes)} classes in misc file:")
    
    # Better categorization based on class names
    refined_categories = {
        'converters': [],
        'parsers': [],  
        'validators': [],
        'frameworks': [],
        'layers': [],
        'utilities': [],
        'integrations': [],
        'processors': [],
        'managers': [],
        'engines': [],
        'systems': [],
        'builders': [],
        'trackers': [],
        'handlers': [],
        'uncategorized': []
    }
    
    for class_name in classes:
        # Remove Test prefix and _Tests suffix, convert to lowercase
        core_name = class_name.replace('Test', '').replace('_Tests', '').lower()
        
        categorized = False
        
        # Converter patterns
        if any(word in core_name for word in ['converter', 'convert', 'transform']):
            refined_categories['converters'].append(class_name)
            categorized = True
        # Parser patterns    
        elif any(word in core_name for word in ['parser', 'parse', 'syntax', 'lexer']):
            refined_categories['parsers'].append(class_name)
            categorized = True
        # Validator patterns
        elif any(word in core_name for word in ['validator', 'validate', 'verification', 'checker', 'compliance']):
            refined_categories['validators'].append(class_name)
            categorized = True
        # Framework patterns
        elif any(word in core_name for word in ['framework', 'adapter', 'bridge', 'wrapper']):
            refined_categories['frameworks'].append(class_name)
            categorized = True
        # Layer patterns
        elif any(word in core_name for word in ['layer', 'level', 'tier']):
            refined_categories['layers'].append(class_name)
            categorized = True
        # Manager patterns
        elif any(word in core_name for word in ['manager', 'supervisor', 'coordinator']):
            refined_categories['managers'].append(class_name)
            categorized = True
        # Engine patterns
        elif any(word in core_name for word in ['engine', 'processor', 'executor']):
            refined_categories['engines'].append(class_name)
            categorized = True
        # System patterns
        elif any(word in core_name for word in ['system', 'service', 'platform']):
            refined_categories['systems'].append(class_name)
            categorized = True
        # Builder patterns
        elif any(word in core_name for word in ['builder', 'factory', 'creator', 'generator']):
            refined_categories['builders'].append(class_name)
            categorized = True
        # Tracker patterns
        elif any(word in core_name for word in ['tracker', 'monitor', 'watcher', 'observer']):
            refined_categories['trackers'].append(class_name)
            categorized = True
        # Handler patterns
        elif any(word in core_name for word in ['handler', 'processor', 'dispatcher']):
            refined_categories['handlers'].append(class_name)
            categorized = True
        # Integration patterns  
        elif any(word in core_name for word in ['integration', 'connector', 'interface']):
            refined_categories['integrations'].append(class_name)
            categorized = True
        # Utility patterns
        elif any(word in core_name for word in ['utility', 'util', 'helper', 'tool']):
            refined_categories['utilities'].append(class_name)
            categorized = True
        
        if not categorized:
            refined_categories['uncategorized'].append(class_name)
    
    # Print categorization results
    print("\nRefined categorization:")
    total_distributed = 0
    for category, class_list in refined_categories.items():
        if class_list:
            print(f"  {category}: {len(class_list)} classes")
            total_distributed += len(class_list)
            for cls in class_list[:3]:  # Show first 3
                print(f"    - {cls}")
            if len(class_list) > 3:
                print(f"    - ... and {len(class_list) - 3} more")
    
    print(f"\nTotal classes distributed: {total_distributed}")
    print(f"Average per category: {total_distributed // len([c for c in refined_categories.values() if c])}")
    
    return refined_categories, content

def split_misc_file():
    """Split the misc file into refined categories"""
    refined_categories, content = analyze_misc_file()
    
    output_dir = Path("C:/Users/kbass/OneDrive/Documents/testmaster/TestMaster/tests/modularized/misc_split")
    output_dir.mkdir(exist_ok=True)
    
    # Split file content by class
    lines = content.split('\n')
    class_sections = {}
    current_class = None
    current_section = []
    
    for line in lines:
        if line.strip().startswith('class Test') and line.strip().endswith('_Tests:'):
            # Save previous class if exists
            if current_class:
                class_sections[current_class] = '\n'.join(current_section)
            
            # Start new class
            current_class = line.strip().replace('class ', '').replace(':', '')
            current_section = [line]
        else:
            if current_class:
                current_section.append(line)
    
    # Save last class
    if current_class:
        class_sections[current_class] = '\n'.join(current_section)
    
    # Create refined category files
    files_created = 0
    for category, class_list in refined_categories.items():
        if not class_list:
            continue
            
        output_file = output_dir / f"test_{category}.py"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f'''"""
TestMaster {category.title()} Tests
Generated from test_misc.py refinement
Category: {category}
"""

import pytest

''')
            
            for class_name in class_list:
                if class_name in class_sections:
                    f.write(class_sections[class_name])
                    f.write('\n\n')
        
        files_created += 1
        print(f"Created {output_file} with {len(class_list)} classes")
    
    print(f"\nRefinement complete! Created {files_created} refined test files")
    
    # Archive the original misc file
    archive_path = "C:/Users/kbass/OneDrive/Documents/testmaster/TestMaster/archive/test_misc_original_6141_lines.py"
    os.rename(
        "C:/Users/kbass/OneDrive/Documents/testmaster/TestMaster/tests/modularized/test_misc.py",
        archive_path
    )
    print(f"Archived original misc file to: {archive_path}")

if __name__ == "__main__":
    split_misc_file()