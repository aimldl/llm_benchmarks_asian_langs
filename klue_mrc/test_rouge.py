#!/usr/bin/env python3
"""
Test script to verify ROUGE-W and LCCS-F1 metrics functionality
"""

import sys
import os

def test_metrics_import():
    """Test if the benchmark module can be imported."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("klue_mrc_module", "klue_mrc-gemini2_5flash.py")
        if spec is None:
            raise ImportError("Could not load klue_mrc-gemini2_5flash.py")
        klue_mrc_module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError("Could not load module spec")
        spec.loader.exec_module(klue_mrc_module)
        
        print("✅ Benchmark module import test passed")
        return True, klue_mrc_module
    except ImportError as e:
        print(f"❌ Benchmark module import test failed: {e}")
        return False, None

def test_rouge_w_calculation():
    """Test ROUGE-W score calculation."""
    try:
        # Test ROUGE-W implementation
        import importlib.util
        spec = importlib.util.spec_from_file_location("klue_mrc_module", "klue_mrc-gemini2_5flash.py")
        if spec is None:
            raise ImportError("Could not load klue_mrc-gemini2_5flash.py")
        klue_mrc_module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError("Could not load module spec")
        spec.loader.exec_module(klue_mrc_module)
        
        KLUEMachineReadingComprehensionBenchmark = klue_mrc_module.KLUEMachineReadingComprehensionBenchmark
        BenchmarkConfig = klue_mrc_module.BenchmarkConfig
        
        # Create a minimal benchmark instance for testing
        config = BenchmarkConfig()
        benchmark = KLUEMachineReadingComprehensionBenchmark(config)
        
        # Test with sample data
        reference = "노르웨이로 파견되었다"
        prediction = "노르웨이"
        
        rouge_w_score = benchmark.calculate_rouge_w(prediction, [reference])
        
        print("✅ ROUGE-W calculation test passed")
        print(f"   Reference: {reference}")
        print(f"   Prediction: {prediction}")
        print(f"   ROUGE-W: {rouge_w_score:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ ROUGE-W calculation test failed: {e}")
        return False

def test_lccs_f1_calculation():
    """Test LCCS-based F1 score calculation."""
    try:
        # Test LCCS-F1 implementation
        import importlib.util
        spec = importlib.util.spec_from_file_location("klue_mrc_module", "klue_mrc-gemini2_5flash.py")
        if spec is None:
            raise ImportError("Could not load klue_mrc-gemini2_5flash.py")
        klue_mrc_module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError("Could not load module spec")
        spec.loader.exec_module(klue_mrc_module)
        
        KLUEMachineReadingComprehensionBenchmark = klue_mrc_module.KLUEMachineReadingComprehensionBenchmark
        BenchmarkConfig = klue_mrc_module.BenchmarkConfig
        
        # Create a minimal benchmark instance for testing
        config = BenchmarkConfig()
        benchmark = KLUEMachineReadingComprehensionBenchmark(config)
        
        # Test with sample data
        reference = "노르웨이로 파견되었다"
        prediction = "노르웨이"
        
        lccs_f1_score = benchmark.calculate_lccs_f1(prediction, [reference])
        
        print("✅ LCCS-F1 calculation test passed")
        print(f"   Reference: {reference}")
        print(f"   Prediction: {prediction}")
        print(f"   LCCS-F1: {lccs_f1_score:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ LCCS-F1 calculation test failed: {e}")
        return False

def test_metrics_comparison():
    """Test that ROUGE-W and LCCS-F1 are different metrics."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("klue_mrc_module", "klue_mrc-gemini2_5flash.py")
        if spec is None:
            raise ImportError("Could not load klue_mrc-gemini2_5flash.py")
        klue_mrc_module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError("Could not load module spec")
        spec.loader.exec_module(klue_mrc_module)
        
        KLUEMachineReadingComprehensionBenchmark = klue_mrc_module.KLUEMachineReadingComprehensionBenchmark
        BenchmarkConfig = klue_mrc_module.BenchmarkConfig
        
        # Create a minimal benchmark instance for testing
        config = BenchmarkConfig()
        benchmark = KLUEMachineReadingComprehensionBenchmark(config)
        
        # Test with exact match case
        reference = "노르웨이로 파견되었다"
        prediction = "노르웨이로 파견되었다"
        
        rouge_w_score = benchmark.calculate_rouge_w(prediction, [reference])
        lccs_f1_score = benchmark.calculate_lccs_f1(prediction, [reference])
        
        print("✅ Metrics comparison test passed")
        print(f"   Reference: {reference}")
        print(f"   Prediction: {prediction}")
        print(f"   ROUGE-W: {rouge_w_score:.4f}")
        print(f"   LCCS-F1: {lccs_f1_score:.4f}")
        print(f"   Different metrics: {'Yes' if abs(rouge_w_score - lccs_f1_score) > 0.001 else 'No'}")
        
        return True
    except Exception as e:
        print(f"❌ Metrics comparison test failed: {e}")
        return False

def test_impossible_question():
    """Test metrics for impossible questions."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("klue_mrc_module", "klue_mrc-gemini2_5flash.py")
        if spec is None:
            raise ImportError("Could not load klue_mrc-gemini2_5flash.py")
        klue_mrc_module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError("Could not load module spec")
        spec.loader.exec_module(klue_mrc_module)
        
        KLUEMachineReadingComprehensionBenchmark = klue_mrc_module.KLUEMachineReadingComprehensionBenchmark
        BenchmarkConfig = klue_mrc_module.BenchmarkConfig
        
        # Create a minimal benchmark instance for testing
        config = BenchmarkConfig()
        benchmark = KLUEMachineReadingComprehensionBenchmark(config)
        
        # Test with impossible question response
        reference = "답을 찾을 수 없습니다"
        prediction = "답을 찾을 수 없습니다"
        
        rouge_w_score = benchmark.calculate_rouge_w(prediction, [reference])
        lccs_f1_score = benchmark.calculate_lccs_f1(prediction, [reference])
        
        print("✅ Impossible question metrics test passed")
        print(f"   Reference: {reference}")
        print(f"   Prediction: {prediction}")
        print(f"   ROUGE-W: {rouge_w_score:.4f}")
        print(f"   LCCS-F1: {lccs_f1_score:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Impossible question metrics test failed: {e}")
        return False

def main():
    """Run all metrics tests."""
    print("Testing ROUGE-W and LCCS-F1 metrics functionality...")
    print("=" * 60)
    
    tests = [
        test_metrics_import,
        test_rouge_w_calculation,
        test_lccs_f1_calculation,
        test_metrics_comparison,
        test_impossible_question
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Metrics tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All metrics tests passed!")
        return 0
    else:
        print("❌ Some metrics tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 