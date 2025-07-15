#!/usr/bin/env python3
"""
Test script to verify ROUGE metrics functionality
"""

import sys
import os

def test_rouge_import():
    """Test if rouge_score can be imported."""
    try:
        from rouge_score import rouge_scorer
        print("✅ ROUGE library imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import ROUGE library: {e}")
        return False

def test_rouge_calculation():
    """Test ROUGE score calculation."""
    try:
        from rouge_score import rouge_scorer
        
        # Initialize scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Test with sample data
        reference = "노르웨이로 파견되었다"
        prediction = "노르웨이"
        
        scores = scorer.score(reference, prediction)
        
        print("✅ ROUGE calculation test passed")
        print(f"   Reference: {reference}")
        print(f"   Prediction: {prediction}")
        print(f"   ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
        print(f"   ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
        print(f"   ROUGE-L: {scores['rougeL'].fmeasure:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ ROUGE calculation test failed: {e}")
        return False

def test_impossible_question():
    """Test ROUGE calculation for impossible questions."""
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Test with impossible question response
        reference = "답을 찾을 수 없습니다"
        prediction = "답을 찾을 수 없습니다"
        
        scores = scorer.score(reference, prediction)
        
        print("✅ Impossible question ROUGE test passed")
        print(f"   Reference: {reference}")
        print(f"   Prediction: {prediction}")
        print(f"   ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
        print(f"   ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
        print(f"   ROUGE-L: {scores['rougeL'].fmeasure:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Impossible question ROUGE test failed: {e}")
        return False

def main():
    """Run all ROUGE tests."""
    print("Testing ROUGE metrics functionality...")
    print("=" * 50)
    
    tests = [
        test_rouge_import,
        test_rouge_calculation,
        test_impossible_question
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"ROUGE tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("✅ All ROUGE tests passed!")
        return 0
    else:
        print("❌ Some ROUGE tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 