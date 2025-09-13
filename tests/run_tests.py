#!/usr/bin/env python3
"""
Test runner script for Small Object Detection Enhancement System

This script runs comprehensive tests for the small object detection enhancements,
including background independence validation, adaptive thresholds, and API endpoints.

Usage:
    python tests/run_tests.py [options]
    
Options:
    --fast          Run only fast tests (skip slow/integration tests)
    --integration   Run only integration tests
    --api          Run only API tests
    --performance  Run performance benchmarks
    --coverage     Generate coverage report
    --verbose      Verbose output
    --help         Show this help message
"""

import sys
import subprocess
import argparse
from pathlib import Path
import time

def run_command(cmd, capture_output=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=capture_output, 
            text=True, 
            check=False
        )
        return result
    except FileNotFoundError:
        print(f"Error: Command not found: {cmd[0]}")
        return None

def check_dependencies():
    """Check if required testing dependencies are installed."""
    required_packages = [
        'pytest',
        'pytest-cov',
        'pytest-asyncio',
        'httpx',  # For FastAPI testing
        'fastapi[all]'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.split('[')[0].replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def run_background_independence_tests(verbose=False):
    """Run background independence validation tests."""
    print("\n=== Running Background Independence Tests ===")
    
    cmd = [
        'python', '-m', 'pytest', 
        'tests/test_small_object_detection.py::TestBackgroundIndependentDetector',
        '-v' if verbose else '-q'
    ]
    
    result = run_command(cmd, capture_output=not verbose)
    
    if result and result.returncode == 0:
        print("✅ Background independence tests PASSED")
        return True
    else:
        print("❌ Background independence tests FAILED")
        if result and result.stderr:
            print(f"Error: {result.stderr}")
        return False

def run_adaptive_threshold_tests(verbose=False):
    """Run adaptive threshold system tests."""
    print("\n=== Running Adaptive Threshold Tests ===")
    
    cmd = [
        'python', '-m', 'pytest', 
        'tests/test_small_object_detection.py::TestAdaptiveThresholdSystem',
        '-v' if verbose else '-q'
    ]
    
    result = run_command(cmd, capture_output=not verbose)
    
    if result and result.returncode == 0:
        print("✅ Adaptive threshold tests PASSED")
        return True
    else:
        print("❌ Adaptive threshold tests FAILED")
        if result and result.stderr:
            print(f"Error: {result.stderr}")
        return False

def run_small_object_detector_tests(verbose=False):
    """Run small object detector tests."""
    print("\n=== Running Small Object Detector Tests ===")
    
    cmd = [
        'python', '-m', 'pytest', 
        'tests/test_small_object_detection.py::TestSmallObjectDetector',
        '-v' if verbose else '-q'
    ]
    
    result = run_command(cmd, capture_output=not verbose)
    
    if result and result.returncode == 0:
        print("✅ Small object detector tests PASSED")
        return True
    else:
        print("❌ Small object detector tests FAILED")
        if result and result.stderr:
            print(f"Error: {result.stderr}")
        return False

def run_api_tests(verbose=False):
    """Run API endpoint tests."""
    print("\n=== Running API Tests ===")
    
    cmd = [
        'python', '-m', 'pytest', 
        'tests/test_api_endpoints.py',
        '-v' if verbose else '-q'
    ]
    
    result = run_command(cmd, capture_output=not verbose)
    
    if result and result.returncode == 0:
        print("✅ API tests PASSED")
        return True
    else:
        print("❌ API tests FAILED")
        if result and result.stderr:
            print(f"Error: {result.stderr}")
        return False

def run_integration_tests(verbose=False):
    """Run integration tests."""
    print("\n=== Running Integration Tests ===")
    
    cmd = [
        'python', '-m', 'pytest', 
        'tests/test_small_object_detection.py::TestIntegration',
        '-v' if verbose else '-q'
    ]
    
    result = run_command(cmd, capture_output=not verbose)
    
    if result and result.returncode == 0:
        print("✅ Integration tests PASSED")
        return True
    else:
        print("❌ Integration tests FAILED")
        if result and result.stderr:
            print(f"Error: {result.stderr}")
        return False

def run_performance_tests(verbose=False):
    """Run performance benchmark tests."""
    print("\n=== Running Performance Tests ===")
    
    cmd = [
        'python', '-m', 'pytest', 
        'tests/',
        '-m', 'performance',
        '-v' if verbose else '-q'
    ]
    
    result = run_command(cmd, capture_output=not verbose)
    
    if result and result.returncode == 0:
        print("✅ Performance tests PASSED")
        return True
    else:
        print("❌ Performance tests FAILED")
        if result and result.stderr:
            print(f"Error: {result.stderr}")
        return False

def run_all_tests(verbose=False, coverage=False):
    """Run all tests."""
    print("\n=== Running All Tests ===")
    
    cmd = ['python', '-m', 'pytest', 'tests/']
    
    if coverage:
        cmd.extend(['--cov=src', '--cov-report=html', '--cov-report=term'])
    
    if verbose:
        cmd.append('-v')
    else:
        cmd.append('-q')
    
    result = run_command(cmd, capture_output=not verbose)
    
    if result and result.returncode == 0:
        print("✅ All tests PASSED")
        if coverage:
            print("📊 Coverage report generated in htmlcov/")
        return True
    else:
        print("❌ Some tests FAILED")
        if result and result.stderr:
            print(f"Error: {result.stderr}")
        return False

def validate_background_independence_success_rate():
    """Validate that background independence achieves 85%+ success rate."""
    print("\n=== Validating Background Independence Success Rate ===")
    
    # This would run specific validation tests
    cmd = [
        'python', '-m', 'pytest', 
        'tests/test_small_object_detection.py::TestBackgroundIndependentDetector::test_background_independence_success_rate',
        '-v'
    ]
    
    result = run_command(cmd)
    
    if result and result.returncode == 0:
        print("✅ Background independence success rate validation PASSED (≥85%)")
        return True
    else:
        print("❌ Background independence success rate validation FAILED (<85%)")
        return False

def generate_test_report():
    """Generate a comprehensive test report."""
    print("\n=== Generating Test Report ===")
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tests': {}
    }
    
    # Run each test suite and collect results
    test_suites = [
        ('Background Independence', run_background_independence_tests),
        ('Adaptive Thresholds', run_adaptive_threshold_tests),
        ('Small Object Detector', run_small_object_detector_tests),
        ('API Endpoints', run_api_tests),
        ('Integration', run_integration_tests)
    ]
    
    all_passed = True
    
    for suite_name, test_func in test_suites:
        passed = test_func(verbose=False)
        report['tests'][suite_name] = 'PASSED' if passed else 'FAILED'
        if not passed:
            all_passed = False
    
    # Validate success rate
    success_rate_passed = validate_background_independence_success_rate()
    report['background_independence_success_rate'] = 'PASSED (≥85%)' if success_rate_passed else 'FAILED (<85%)'
    
    if not success_rate_passed:
        all_passed = False
    
    # Print report
    print("\n" + "="*60)
    print("SMALL OBJECT DETECTION ENHANCEMENT TEST REPORT")
    print("="*60)
    print(f"Generated: {report['timestamp']}")
    print()
    
    for test_name, status in report['tests'].items():
        status_icon = "✅" if status == 'PASSED' else "❌"
        print(f"{status_icon} {test_name}: {status}")
    
    print(f"\n🎯 Background Independence Success Rate: {report['background_independence_success_rate']}")
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Small Object Detection Enhancement Ready!")
    else:
        print("⚠️  SOME TESTS FAILED - Review and fix issues before deployment")
    print("="*60)
    
    return all_passed

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description='Test runner for Small Object Detection Enhancement System'
    )
    
    parser.add_argument('--fast', action='store_true', 
                       help='Run only fast tests (skip slow/integration tests)')
    parser.add_argument('--integration', action='store_true', 
                       help='Run only integration tests')
    parser.add_argument('--api', action='store_true', 
                       help='Run only API tests')
    parser.add_argument('--performance', action='store_true', 
                       help='Run performance benchmarks')
    parser.add_argument('--coverage', action='store_true', 
                       help='Generate coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    parser.add_argument('--report', action='store_true', 
                       help='Generate comprehensive test report')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    success = True
    
    try:
        if args.report:
            success = generate_test_report()
        elif args.fast:
            print("Running fast tests only...")
            cmd = ['python', '-m', 'pytest', 'tests/', '-m', 'not slow and not integration']
            if args.verbose:
                cmd.append('-v')
            result = run_command(cmd, capture_output=not args.verbose)
            success = result and result.returncode == 0
        elif args.integration:
            success = run_integration_tests(args.verbose)
        elif args.api:
            success = run_api_tests(args.verbose)
        elif args.performance:
            success = run_performance_tests(args.verbose)
        else:
            success = run_all_tests(args.verbose, args.coverage)
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return 1
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())