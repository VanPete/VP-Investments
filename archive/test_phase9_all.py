"""
Phase 9: Master Test Runner
Runs all Phase 9 tests: unit tests, integration tests, and validation
"""

import sys
import subprocess
from typing import List, Tuple


class TestRunner:
    """Master test runner for Phase 9"""
    
    def __init__(self):
        self.results = []
    
    def run_test_file(self, test_file: str, test_name: str) -> Tuple[bool, str]:
        """Run a test file and capture results"""
        print(f"\n{'='*80}")
        print(f"Running: {test_name}")
        print(f"File: {test_file}")
        print(f"{'='*80}\n")
        
        try:
            result = subprocess.run(
                ['python', test_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Print output
            if result.stdout:
                print(result.stdout)
            
            if result.stderr and result.returncode != 0:
                print(f"STDERR:\n{result.stderr}")
            
            # Check if passed
            passed = result.returncode == 0
            
            # Extract test counts from output
            test_summary = self._extract_summary(result.stdout)
            
            self.results.append({
                'test_name': test_name,
                'passed': passed,
                'summary': test_summary,
                'returncode': result.returncode
            })
            
            return passed, test_summary
            
        except subprocess.TimeoutExpired:
            print(f"❌ Test timed out after 30 seconds")
            self.results.append({
                'test_name': test_name,
                'passed': False,
                'summary': 'TIMEOUT',
                'returncode': -1
            })
            return False, 'TIMEOUT'
        except Exception as e:
            print(f"❌ Error running test: {e}")
            self.results.append({
                'test_name': test_name,
                'passed': False,
                'summary': f'ERROR: {e}',
                'returncode': -1
            })
            return False, f'ERROR: {e}'
    
    def _extract_summary(self, output: str) -> str:
        """Extract test summary from output"""
        if not output:
            return 'NO OUTPUT'
        
        lines = output.split('\n')
        for line in lines:
            if 'Tests Passed:' in line:
                return line.strip()
            if 'ALL' in line and 'PASSED' in line:
                return line.strip()
        
        return 'COMPLETED'
    
    def print_final_summary(self):
        """Print final test summary"""
        print("\n" + "="*80)
        print("PHASE 9 MASTER TEST SUMMARY")
        print("="*80)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"\nTest Suites Run: {total_tests}")
        print(f"✅ Suites Passed: {passed_tests}")
        print(f"❌ Suites Failed: {failed_tests}")
        
        print("\n" + "-"*80)
        print("Detailed Results:")
        print("-"*80)
        
        for result in self.results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status}: {result['test_name']}")
            print(f"       {result['summary']}")
        
        if failed_tests == 0:
            print("\n" + "="*80)
            print("🎉 ALL PHASE 9 TESTS PASSED!")
            print("="*80)
            print("\nPhase 9 Testing Complete:")
            print("  ✓ Unit Tests: Trade Classification (8 tests)")
            print("  ✓ Unit Tests: Risk Scoring (8 tests)")
            print("  ✓ Integration Tests (8 tests)")
            print("\nTotal: 24 tests across 3 test suites")
            print("\nReady for Phase 10: Documentation")
        else:
            print("\n" + "="*80)
            print(f"⚠️  {failed_tests} TEST SUITE(S) FAILED")
            print("="*80)
            print("\nReview errors above and re-run failed tests")
        
        return failed_tests == 0


def main():
    """Run all Phase 9 tests"""
    print("="*80)
    print("PHASE 9: TESTING & VALIDATION")
    print("="*80)
    print("\nRunning comprehensive test suite...")
    
    runner = TestRunner()
    
    # Define test suite
    test_suites = [
        ('test_trade_classification.py', 'Unit Tests: Trade Classification'),
        ('test_risk_scoring.py', 'Unit Tests: Risk Scoring'),
        ('test_integration.py', 'Integration Tests'),
    ]
    
    # Run all tests
    all_passed = True
    for test_file, test_name in test_suites:
        passed, summary = runner.run_test_file(test_file, test_name)
        if not passed:
            all_passed = False
    
    # Print final summary
    success = runner.print_final_summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
