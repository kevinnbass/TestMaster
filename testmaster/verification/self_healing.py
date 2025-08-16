#!/usr/bin/env python3
"""
Self-Healing Test Verifier for TestMaster

Consolidates functionality from:
- enhanced_self_healing_verifier.py
- self_healing_converter.py (verification parts)

Provides comprehensive self-healing test verification with iterative improvement.
"""

import os
import sys
import json
import ast
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import SelfHealingVerifier, VerificationResult, VerificationConfig

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class SelfHealingTestVerifier(SelfHealingVerifier):
    """
    Self-healing test verifier with iterative improvement and AI assistance.
    
    Process:
    1. Analyze test structure and quality
    2. Identify syntax and logical issues
    3. Apply self-healing with multiple passes
    4. Verify improvements and generate reports
    """
    
    def __init__(self, 
                 mode: str = "auto",
                 model: str = None,
                 api_key: Optional[str] = None,
                 config: Optional[VerificationConfig] = None):
        """
        Initialize self-healing verifier.
        
        Args:
            mode: AI mode ("provider", "sdk", "template", "auto")
            model: AI model to use
            api_key: API key for AI services
            config: Verification configuration
        """
        super().__init__(config)
        self.mode = mode
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.provider = None
        self.client = None
        
        # Initialize AI provider
        if mode == "auto":
            self._auto_initialize()
        elif mode == "provider":
            self._init_provider_mode()
        elif mode == "sdk":
            self._init_sdk_mode()
        elif mode == "template":
            self._init_template_mode()
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def _auto_initialize(self):
        """Auto-detect best available AI provider."""
        print("Auto-detecting verification mode...")
        
        # Try provider mode first
        try:
            self._init_provider_mode()
            print("SUCCESS: Using provider mode for verification")
            self.mode = "provider"
            return
        except Exception as e:
            print(f"Provider mode failed: {str(e)[:50]}")
        
        # Try SDK mode
        try:
            self._init_sdk_mode()
            print("SUCCESS: Using SDK mode for verification")
            self.mode = "sdk"
            return
        except Exception as e:
            print(f"SDK mode failed: {str(e)[:50]}")
        
        # Fallback to template mode
        print("Using template mode (no AI assistance)")
        self.mode = "template"
        self._init_template_mode()
    
    def _init_provider_mode(self):
        """Initialize using GeminiProvider."""
        try:
            from multi_coder_analysis.llm_providers.gemini_provider import GeminiProvider
            
            if self.model is None:
                self.model = "models/gemini-2.5-pro"
            
            self.provider = GeminiProvider(model=self.model, api_key=self.api_key)
            print(f"Initialized provider mode with {self.model}")
            
        except ImportError:
            raise ImportError("GeminiProvider not available")
    
    def _init_sdk_mode(self):
        """Initialize using direct Google GenAI SDK."""
        try:
            import google.generativeai as genai
            
            if self.model is None:
                self.model = "models/gemini-2.5-pro"
            
            genai.configure(api_key=self.api_key)
            self.client = genai
            print(f"Initialized SDK mode with {self.model}")
            
        except ImportError:
            raise ImportError("google.generativeai not available")
    
    def _init_template_mode(self):
        """Initialize template-based mode."""
        print("Initialized template mode (no AI assistance)")
    
    def verify_test(self, test_file: Path, module_file: Optional[Path] = None) -> VerificationResult:
        """Verify and potentially heal a test file."""
        start_time = time.time()
        
        try:
            print(f"\n{'='*60}")
            print(f"Self-Healing Verification: {test_file.name}")
            print('='*60)
            
            # Read test file
            if not test_file.exists():
                return VerificationResult(
                    success=False,
                    test_file=str(test_file),
                    error_message="Test file not found",
                    execution_time=time.time() - start_time
                )
            
            test_code = test_file.read_text(encoding='utf-8')
            original_code = test_code
            
            # Find module file if not provided
            if module_file is None:
                module_file = self.find_module_for_test(test_file)
            
            print(f"Step 1: Initial analysis...")
            
            # Initial structural analysis
            structure = self.analyze_test_structure(test_code)
            initial_quality = self.calculate_quality_score(structure, test_code)
            
            print(f"  Initial quality score: {initial_quality:.1f}")
            print(f"  Test methods: {structure.get('test_methods', 0)}")
            print(f"  Assertions: {structure.get('assertions', 0)}")
            
            # Step 2: Syntax healing if needed
            print(f"\nStep 2: Syntax healing...")
            healed_code = self.heal_syntax_errors(test_code)
            
            if healed_code is None:
                return VerificationResult(
                    success=False,
                    test_file=str(test_file),
                    module_file=str(module_file) if module_file else None,
                    quality_score=initial_quality,
                    test_count=structure.get('test_methods', 0),
                    assertion_count=structure.get('assertions', 0),
                    issues=["Could not heal syntax errors"],
                    execution_time=time.time() - start_time,
                    error_message="Syntax healing failed"
                )
            
            if healed_code != test_code:
                print("  Syntax errors found and healed")
                test_code = healed_code
            else:
                print("  No syntax errors found")
            
            # Step 3: Iterative quality improvement
            print(f"\nStep 3: Quality improvement...")
            improved_code = self._iterative_improvement(test_code, module_file)
            
            if improved_code:
                test_code = improved_code
                print("  Quality improvements applied")
            else:
                print("  No improvements possible or needed")
            
            # Step 4: Final analysis
            print(f"\nStep 4: Final analysis...")
            final_structure = self.analyze_test_structure(test_code)
            final_quality = self.calculate_quality_score(final_structure, test_code)
            
            # Identify remaining issues
            issues = self._identify_issues(final_structure, test_code)
            suggestions = self._generate_suggestions(final_structure, test_code, module_file)
            
            # Save improved test if it's better
            if test_code != original_code and final_quality > initial_quality:
                backup_file = test_file.with_suffix('.py.backup')
                backup_file.write_text(original_code, encoding='utf-8')
                test_file.write_text(test_code, encoding='utf-8')
                print(f"  Improved test saved (backup: {backup_file.name})")
            
            execution_time = time.time() - start_time
            
            print(f"\nVerification complete:")
            print(f"  Final quality score: {final_quality:.1f} (improvement: {final_quality - initial_quality:+.1f})")
            print(f"  Issues found: {len(issues)}")
            print(f"  Execution time: {execution_time:.1f}s")
            
            result = VerificationResult(
                success=True,
                test_file=str(test_file),
                module_file=str(module_file) if module_file else None,
                quality_score=final_quality,
                test_count=final_structure.get('test_methods', 0),
                assertion_count=final_structure.get('assertions', 0),
                issues=issues,
                suggestions=suggestions,
                execution_time=execution_time
            )
            
            self.update_stats(result)
            return result
            
        except Exception as e:
            error_result = VerificationResult(
                success=False,
                test_file=str(test_file),
                module_file=str(module_file) if module_file else None,
                error_message=f"Verification error: {e}",
                execution_time=time.time() - start_time
            )
            self.update_stats(error_result)
            return error_result
    
    def _heal_with_ai(self, test_code: str, error_message: str) -> Optional[str]:
        """Heal syntax errors using AI."""
        if self.mode == "template":
            return None
        
        try:
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            
            prompt = f"""Fix the syntax errors in this Python test code:

SYNTAX ERROR: {error_message}

CODE:
```python
{test_code}
```

Requirements:
1. Fix all syntax errors
2. Maintain the original test logic and structure
3. Ensure all imports are correct
4. Keep all test methods and assertions
5. Return only the corrected Python code
6. No explanations or markdown

Return the fixed code:"""

            # Make AI call
            if self.mode == "provider":
                response = self.provider(prompt, temperature=0.0, max_output_tokens=4000)
                if isinstance(response, list):
                    response_text = response[0] if response else ""
                else:
                    response_text = str(response)
            else:  # SDK mode
                model = self.client.GenerativeModel(self.model)
                response = model.generate_content(
                    prompt,
                    generation_config=self.client.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=4000
                    )
                )
                response_text = response.text if response.text else ""
            
            # Clean response
            healed_code = response_text.strip()
            if healed_code.startswith("```python"):
                healed_code = healed_code[9:]
            if healed_code.startswith("```"):
                healed_code = healed_code[3:]
            if healed_code.endswith("```"):
                healed_code = healed_code[:-3]
            
            return healed_code.strip()
            
        except Exception as e:
            print(f"AI healing failed: {e}")
            return None
    
    def _iterative_improvement(self, test_code: str, module_file: Optional[Path] = None) -> Optional[str]:
        """Apply iterative improvements to test quality."""
        if self.mode == "template" or not self.config.use_ai_analysis:
            return None
        
        try:
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Read module code if available
            module_code = ""
            if module_file and module_file.exists():
                module_code = module_file.read_text(encoding='utf-8')[:4000]  # Limit size
            
            prompt = f"""Improve this Python test code for better quality and coverage:

MODULE CODE (if available):
```python
{module_code}
```

CURRENT TEST CODE:
```python
{test_code}
```

Improvements needed:
1. Add missing test methods for uncovered functions/classes
2. Add more comprehensive assertions
3. Add edge case testing
4. Add error condition testing
5. Improve test method docstrings
6. Add setup/teardown if needed

Requirements:
- Keep all existing test logic
- Only add improvements, don't remove tests
- Ensure all tests are realistic and executable
- Use pytest framework conventions
- Return only the improved Python code

Return the improved test code:"""

            # Make AI call
            if self.mode == "provider":
                response = self.provider(prompt, temperature=0.1, max_output_tokens=4000)
                if isinstance(response, list):
                    response_text = response[0] if response else ""
                else:
                    response_text = str(response)
            else:  # SDK mode
                model = self.client.GenerativeModel(self.model)
                response = model.generate_content(
                    prompt,
                    generation_config=self.client.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=4000
                    )
                )
                response_text = response.text if response.text else ""
            
            # Clean response
            improved_code = response_text.strip()
            if improved_code.startswith("```python"):
                improved_code = improved_code[9:]
            if improved_code.startswith("```"):
                improved_code = improved_code[3:]
            if improved_code.endswith("```"):
                improved_code = improved_code[:-3]
            
            improved_code = improved_code.strip()
            
            # Validate improved code
            try:
                ast.parse(improved_code)
                
                # Check if it's actually better
                original_structure = self.analyze_test_structure(test_code)
                improved_structure = self.analyze_test_structure(improved_code)
                
                original_quality = self.calculate_quality_score(original_structure, test_code)
                improved_quality = self.calculate_quality_score(improved_structure, improved_code)
                
                if improved_quality > original_quality:
                    return improved_code
                else:
                    print(f"  AI improvements didn't increase quality ({improved_quality:.1f} vs {original_quality:.1f})")
                    return None
                    
            except SyntaxError:
                print("  AI improvements introduced syntax errors")
                return None
            
        except Exception as e:
            print(f"Iterative improvement failed: {e}")
            return None
    
    def verify_batch(self, test_files: List[Path]) -> List[VerificationResult]:
        """Verify multiple test files."""
        print(f"\n{'='*60}")
        print(f"BATCH SELF-HEALING VERIFICATION")
        print('='*60)
        print(f"Verifying {len(test_files)} test files...")
        
        results = []
        
        for i, test_file in enumerate(test_files):
            print(f"\n[{i+1}/{len(test_files)}] Processing {test_file.name}...")
            
            result = self.verify_test(test_file)
            results.append(result)
            
            status = "✅" if result.success else "❌"
            quality = result.quality_score
            print(f"{status} {test_file.name} - Quality: {quality:.1f}")
        
        # Print batch summary
        self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results: List[VerificationResult]):
        """Print summary of batch verification."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        if successful:
            avg_quality = sum(r.quality_score for r in successful) / len(successful)
            total_tests = sum(r.test_count for r in successful)
            total_assertions = sum(r.assertion_count for r in successful)
        else:
            avg_quality = 0
            total_tests = 0
            total_assertions = 0
        
        print(f"\n{'='*60}")
        print("BATCH VERIFICATION SUMMARY")
        print('='*60)
        print(f"Total files: {len(results)}")
        print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
        print(f"Average quality score: {avg_quality:.1f}")
        print(f"Total test methods: {total_tests}")
        print(f"Total assertions: {total_assertions}")
        
        if failed:
            print(f"\nFailed files:")
            for result in failed[:5]:  # Show first 5
                file_name = Path(result.test_file).name
                print(f"  ❌ {file_name}: {result.error_message}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")


def main():
    """Test the self-healing verifier."""
    print("="*60)
    print("TestMaster Self-Healing Verifier")
    print("="*60)
    
    # Create verifier
    config = VerificationConfig(
        quality_threshold=70.0,
        healing_iterations=3,
        use_ai_analysis=True
    )
    
    verifier = SelfHealingTestVerifier(mode="auto", config=config)
    
    # Find test files
    test_dir = Path("tests/unit")
    if not test_dir.exists():
        test_dir = Path("tests")
    
    if test_dir.exists():
        test_files = list(test_dir.glob("test_*.py"))[:5]  # Test first 5
        if test_files:
            print(f"Testing on {len(test_files)} test files...")
            results = verifier.verify_batch(test_files)
        else:
            print("No test files found to verify")
            return 1
    else:
        print("No test directory found")
        return 1
    
    # Print final stats
    verifier.print_stats()
    
    success_count = sum(1 for r in results if r.success)
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())