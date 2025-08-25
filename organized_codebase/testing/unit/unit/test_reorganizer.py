#!/usr/bin/env python3
"""
Test Script for Codebase Reorganizer
===================================

This script demonstrates and tests the codebase reorganizer functionality.
"""

import sys
import os
from pathlib import Path

def test_reorganizer() -> None:
    """Test the reorganizer with preview mode"""
    print("Testing Codebase Reorganizer")
    print("=" * 30)

    # Find the reorganizer
    current_dir = Path(__file__).parent
    reorganizer_path = current_dir / "reorganizer.py"

    if not reorganizer_path.exists():
        print(f"ERROR: Reorganizer not found at {reorganizer_path}")
        return False

    # Find TestMaster root (go up directories until we find indicators)
    testmaster_root = current_dir.parent.parent  # Assume we're in tools/codebase_reorganizer

    indicators = ['TestMaster', 'PRODUCTION_PACKAGES', 'AGENT_D_HOUR_8-10_PREDICTIVE_INTELLIGENCE_BREAKTHROUGH.md']
    found_indicator = False

    for indicator in indicators:
        if (testmaster_root / indicator).exists():
            found_indicator = True
            break

    if not found_indicator:
        print(f"WARNING: Could not verify TestMaster root at {testmaster_root}")
        print("This might be expected if running in a different environment")

    print(f"TestMaster root: {testmaster_root}")
    print(f"Reorganizer path: {reorganizer_path}")

    # Test with preview mode
    print("\nRunning preview mode...")
    cmd_args = [
        sys.executable,
        str(reorganizer_path),
        "--preview",
        "--root", str(testmaster_root)
    ]

    print(f"Command: {' '.join(cmd_args)}")

    # Change to reorganizer directory
    os.chdir(current_dir)

    # Run the command
    result = os.system(' '.join(cmd_args))

    if result == 0:
        print("\n✅ Preview completed successfully!")
        return True
    else:
        print(f"\n❌ Preview failed with exit code: {result}")
        return False

def show_usage_examples() -> None:
    """Show usage examples"""
    print("\nUsage Examples:")
    print("=" * 20)
    print("1. Preview what would be reorganized:")
    print("   python reorganize_codebase.py --preview")
    print()
    print("2. Use symlinks (safest option):")
    print("   python reorganize_codebase.py --symlinks")
    print()
    print("3. Interactive mode (ask before each change):")
    print("   python reorganize_codebase.py --interactive")
    print()
    print("4. Full automation:")
    print("   python reorganize_codebase.py --automatic")
    print()
    print("5. With Aider:")
    print("   aider")
    print("   /run python reorganize_codebase.py --preview")
    print()
    print("6. With OpenDevin:")
    print("   python reorganize_codebase.py --automatic --move")

if __name__ == "__main__":
    try:
        success = test_reorganizer()
        show_usage_examples()

        if success:
            print("\n🎉 Test completed successfully!")
        else:
            print("\n⚠️  Test had issues - check the output above")

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

