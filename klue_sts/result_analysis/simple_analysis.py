#!/usr/bin/env python3
"""
Simple KLUE STS Results Analysis Script
This script analyzes benchmark results without requiring matplotlib.
"""

import json
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime

def load_results(results_file):
    """Load benchmark results from JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_metrics(results):
    """Analyze basic metrics from results."""
    total_samples = len(results)
    valid_predictions = [r for r in results if r.get('predicted_score') is not None]
    errors = [r for r in results if r.get('error')]
    
    metrics = {
        'total_samples': total_samples,
        'valid_predictions': len(valid_predictions),
        'errors': len(errors),
        'success_rate': len(valid_predictions) / total_samples * 100 if total_samples > 0 else 0,
        'error_rate': len(errors) / total_samples * 100 if total_samples > 0 else 0
    }
    
    if valid_predictions:
        true_scores = [r['true_score'] for r in valid_predictions]
        pred_scores = [r['predicted_score'] for r in valid_predictions]
        
        # Calculate correlation metrics
        try:
            from scipy.stats import pearsonr, spearmanr
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            pearson_corr, _ = pearsonr(true_scores, pred_scores)
            spearman_corr, _ = spearmanr(true_scores, pred_scores)
            mse = mean_squared_error(true_scores, pred_scores)
            mae = mean_absolute_error(true_scores, pred_scores)
            
            metrics.update({
                'pearson_correlation': pearson_corr,
                'spearman_correlation': spearman_corr,
                'mse': mse,
                'mae': mae,
                'true_score_range': (min(true_scores), max(true_scores)),
                'pred_score_range': (min(pred_scores), max(pred_scores)),
                'true_score_mean': sum(true_scores) / len(true_scores),
                'pred_score_mean': sum(pred_scores) / len(pred_scores)
            })
        except ImportError:
            print("Warning: scikit-learn not available, skipping correlation metrics")
    
    return metrics

def analyze_error_patterns(results):
    """Analyze error patterns in results."""
    errors = [r for r in results if r.get('error')]
    
    if not errors:
        return {}
    
    # Group errors by type
    error_types = {}
    for error in errors:
        error_msg = error.get('error', 'Unknown error')
        if error_msg not in error_types:
            error_types[error_msg] = []
        error_types[error_msg].append(error)
    
    # Analyze finish reasons
    finish_reasons = {}
    for result in results:
        reason = result.get('finish_reason', 'UNKNOWN')
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    
    return {
        'error_types': error_types,
        'finish_reasons': finish_reasons
    }

def generate_report(results_file, output_dir):
    """Generate comprehensive analysis report."""
    print(f"Analyzing results from: {results_file}")
    
    # Load results
    results = load_results(results_file)
    
    # Analyze metrics
    metrics = analyze_metrics(results)
    error_patterns = analyze_error_patterns(results)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"simple_analysis_report_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("KLUE STS Benchmark Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results File: {results_file}\n\n")
        
        f.write("Basic Metrics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Samples: {metrics['total_samples']}\n")
        f.write(f"Valid Predictions: {metrics['valid_predictions']}\n")
        f.write(f"Errors: {metrics['errors']}\n")
        f.write(f"Success Rate: {metrics['success_rate']:.2f}%\n")
        f.write(f"Error Rate: {metrics['error_rate']:.2f}%\n\n")
        
        if 'pearson_correlation' in metrics:
            f.write("Performance Metrics:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Pearson Correlation: {metrics['pearson_correlation']:.4f}\n")
            f.write(f"Spearman Correlation: {metrics['spearman_correlation']:.4f}\n")
            f.write(f"Mean Squared Error: {metrics['mse']:.4f}\n")
            f.write(f"Mean Absolute Error: {metrics['mae']:.4f}\n\n")
            
            f.write("Score Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"True Score Range: [{metrics['true_score_range'][0]:.2f}, {metrics['true_score_range'][1]:.2f}]\n")
            f.write(f"Predicted Score Range: [{metrics['pred_score_range'][0]:.2f}, {metrics['pred_score_range'][1]:.2f}]\n")
            f.write(f"True Score Mean: {metrics['true_score_mean']:.2f}\n")
            f.write(f"Predicted Score Mean: {metrics['pred_score_mean']:.2f}\n\n")
        
        if error_patterns.get('finish_reasons'):
            f.write("Finish Reasons:\n")
            f.write("-" * 20 + "\n")
            for reason, count in error_patterns['finish_reasons'].items():
                percentage = count / metrics['total_samples'] * 100
                f.write(f"{reason}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
        
        if error_patterns.get('error_types'):
            f.write("Error Analysis:\n")
            f.write("-" * 20 + "\n")
            for error_type, error_list in error_patterns['error_types'].items():
                f.write(f"Error Type: {error_type}\n")
                f.write(f"Count: {len(error_list)}\n")
                f.write("Sample Errors:\n")
                for i, error in enumerate(error_list[:3]):
                    f.write(f"  {i+1}. ID: {error.get('id', 'N/A')}\n")
                    f.write(f"     Sentence 1: {error.get('sentence1', 'N/A')[:50]}...\n")
                    f.write(f"     Sentence 2: {error.get('sentence2', 'N/A')[:50]}...\n")
                    f.write(f"     True Score: {error.get('true_score', 'N/A')}\n")
                    f.write(f"     Predicted Score: {error.get('predicted_score', 'N/A')}\n")
                    f.write(f"     Error: {error.get('error', 'N/A')}\n\n")
    
    print(f"Analysis report saved to: {report_file}")
    
    # Save metrics as JSON for programmatic access
    metrics_file = output_path / f"metrics_{timestamp}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"Metrics saved to: {metrics_file}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Analyze KLUE STS benchmark results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--output-dir", default="result_analysis", help="Output directory for analysis")
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        return
    
    try:
        metrics = generate_report(args.results_file, args.output_dir)
        print("\nAnalysis completed successfully!")
        if 'pearson_correlation' in metrics:
            print(f"Pearson Correlation: {metrics['pearson_correlation']:.4f}")
            print(f"Spearman Correlation: {metrics['spearman_correlation']:.4f}")
        print(f"Success Rate: {metrics['success_rate']:.2f}%")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main() 