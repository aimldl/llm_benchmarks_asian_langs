#!/usr/bin/env python3
"""
KLUE Sentence Textual Similarity (STS) Benchmark with Gemini 2.5 Flash on Vertex AI
This script benchmarks Gemini 2.5 Flash on the Korean Language Understanding Evaluation (KLUE) STS task using Google Cloud Vertex AI.
"""

import os
import json
import time
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold
)
from datasets import load_dataset   
import pandas as pd
from tqdm import tqdm
import logging
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""
    model_name: str = "gemini-2.5-flash"
    sleep_interval_between_api_calls: float = 0.04 # sec
    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 1.0
    top_k: int = 1
    batch_size: int = 1
    max_samples: Optional[int] = None
    output_dir: str = "benchmark_results"
    save_predictions: bool = True
    save_interval: int = 50  # Save intermediate results every N samples
    project_id: Optional[str] = None
    location: str = "us-central1"

class KLUESentenceTextualSimilarityBenchmark:
    """Benchmark class for KLUE STS task using Vertex AI."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = None
        self.results = []
        self.metrics = {}
        
        # Initialize Vertex AI
        self._initialize_vertex_ai()
        
        # Initialize model
        self._initialize_model()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI with project and location."""
        try:
            project_id = self.config.project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                raise ValueError("Google Cloud project ID must be provided via the --project-id flag or by setting the GOOGLE_CLOUD_PROJECT environment variable.")

            print(f"project_id: {project_id}")
            
            # Initialize genai client for Vertex AI
            self.client = genai.Client(vertexai=True, project=project_id, location=self.config.location)
            logger.info(f"Initialized Vertex AI with project: {project_id}, location: {self.config.location}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
    def _initialize_model(self):
        """Initialize the Gemini model on Vertex AI."""
        try:
            # Store model name for later use
            self.model_name = self.config.model_name
            logger.info(f"Model name set to: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load the KLUE STS dataset, convert it to a list of dictionaries,
        and efficiently limit the number of samples based on the configuration.
        """
        try:
            logger.info("Loading KLUE STS dataset...")
            
            # Load the validation split from the Hugging Face Hub.
            validation_dataset = load_dataset('klue', 'sts', split='validation')

            processed_data = []
            
            # Determine if a subset of data should be used.
            use_subset = self.config.max_samples and self.config.max_samples > 0
            if use_subset:
                 logger.info(f"Preparing to load a subset of {self.config.max_samples} samples.")

            # Efficiently iterate through the dataset.
            for item in validation_dataset:
                # If max_samples is set, break the loop once the limit is reached.
                if use_subset and self.config.max_samples and len(processed_data) >= self.config.max_samples:
                    logger.info(f"Reached sample limit of {self.config.max_samples}. Halting data loading.")
                    break
                    
                processed_data.append({
                    "id": item["guid"],
                    "sentence1": item["sentence1"],
                    "sentence2": item["sentence2"],
                    "similarity_score": item["labels"]["label"],  # The ground truth similarity score
                    "sentence1_length": len(item["sentence1"]),
                    "sentence2_length": len(item["sentence2"])
                })

            logger.info(f"✅ Successfully loaded {len(processed_data)} samples.")
            return processed_data
            
        except KeyError as e:
            logger.error(f"❌ A key was not found in the dataset item: {e}. The dataset schema may have changed.")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load or process the dataset: {e}")
            raise
    
    def create_prompt(self, sentence1: str, sentence2: str) -> str:
        """Create prompt for sentence similarity prediction."""
        prompt = f"""역할: 당신은 두 한국어 문장 간의 의미적 유사도를 정확하게 평가하는 "문장 유사도 분석 전문가"입니다.

임무: 아래에 제시된 두 문장의 의미적 유사도를 0점에서 5점까지의 척도로 평가해 주세요.

평가 기준:
- 0점: 완전히 다른 의미 (전혀 관련 없음)
- 1점: 거의 다른 의미 (매우 낮은 유사도)
- 2점: 부분적으로 다른 의미 (낮은 유사도)
- 3점: 비슷한 의미 (중간 유사도)
- 4점: 매우 유사한 의미 (높은 유사도)
- 5점: 완전히 동일한 의미 (최고 유사도)

문장 1: {sentence1}
문장 2: {sentence2}

지침:
1. 두 문장의 핵심 의미를 파악하세요.
2. 어휘, 문법, 표현 방식의 차이는 있지만 의미가 같다면 높은 점수를 주세요.
3. 반대로 어휘가 비슷해도 의미가 다르다면 낮은 점수를 주세요.
4. 정확한 숫자(0, 1, 2, 3, 4, 5 중 하나)로만 답변하세요.

유사도 점수:"""
        return prompt
    
    def configure_safety_settings(self, threshold=HarmBlockThreshold.BLOCK_NONE):
        """Configure safety settings for the model."""
        safety_settings = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=threshold
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=threshold
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=threshold
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=threshold
            )
        ]
        return safety_settings
    
    def predict_single(self, sentence1: str, sentence2: str) -> Dict[str, Any]:
        """Make a single prediction for sentence similarity."""
        try:
            # Create prompt
            prompt = self.create_prompt(sentence1, sentence2)
            
            # Configure generation parameters
            generation_config = GenerateContentConfig(
                max_output_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k
            )
            
            # Configure safety settings
            safety_settings = self.configure_safety_settings()
            
            # Generate content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generation_config,
                safety_settings=safety_settings
            )
            
            # Extract prediction
            prediction_text = response.text.strip()
            
            # Try to extract numeric score from prediction
            try:
                # Look for numbers in the response
                import re
                numbers = re.findall(r'\d+(?:\.\d+)?', prediction_text)
                if numbers:
                    predicted_score = float(numbers[0])
                    # Ensure score is within valid range [0, 5]
                    predicted_score = max(0.0, min(5.0, predicted_score))
                else:
                    predicted_score = None
            except (ValueError, IndexError):
                predicted_score = None
            
            return {
                "prediction_text": prediction_text,
                "predicted_score": predicted_score,
                "finish_reason": response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                "prediction_text": "",
                "predicted_score": None,
                "error": str(e),
                "finish_reason": "ERROR"
            }
    
    def calculate_metrics(self, y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        """Calculate correlation and error metrics for STS."""
        try:
            # Filter out None predictions
            valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
            if not valid_pairs:
                return {
                    "pearson_correlation": 0.0,
                    "spearman_correlation": 0.0,
                    "mse": 0.0,
                    "mae": 0.0,
                    "valid_predictions": 0,
                    "total_samples": len(y_true)
                }
            
            y_true_clean, y_pred_clean = zip(*valid_pairs)
            
            # Calculate metrics
            pearson_corr, _ = pearsonr(y_true_clean, y_pred_clean)
            spearman_corr, _ = spearmanr(y_true_clean, y_pred_clean)
            mse = mean_squared_error(y_true_clean, y_pred_clean)
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            
            return {
                "pearson_correlation": pearson_corr,
                "spearman_correlation": spearman_corr,
                "mse": mse,
                "mae": mae,
                "valid_predictions": len(valid_pairs),
                "total_samples": len(y_true)
            }
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            return {
                "pearson_correlation": 0.0,
                "spearman_correlation": 0.0,
                "mse": 0.0,
                "mae": 0.0,
                "valid_predictions": 0,
                "total_samples": len(y_true)
            }
    
    def run_benchmark(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the benchmark on test data."""
        logger.info("Starting benchmark...")
        
        total = len(test_data)
        start_time = time.time()
        
        # Lists to store true and predicted scores
        y_true = []
        y_pred = []
        
        for i, item in enumerate(tqdm(test_data, desc="Processing samples")):
            # Make prediction
            prediction = self.predict_single(item["sentence1"], item["sentence2"])
            
            # Store scores for metrics calculation
            y_true.append(item["similarity_score"])
            y_pred.append(prediction["predicted_score"])
            
            # Store result
            result = {
                "id": item["id"],
                "sentence1": item["sentence1"],
                "sentence2": item["sentence2"],
                "true_score": item["similarity_score"],
                "predicted_score": prediction["predicted_score"],
                "prediction_text": prediction["prediction_text"],
                "finish_reason": prediction.get("finish_reason", "UNKNOWN"),
                "error": prediction.get("error")
            }
            self.results.append(result)
            
            # Save intermediate results periodically
            if (i + 1) % self.config.save_interval == 0:
                self.save_intermediate_results(i + 1, start_time)
            
            # Add small delay to avoid rate limiting
            time.sleep(self.config.sleep_interval_between_api_calls)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        self.metrics = {
            "pearson_correlation": metrics["pearson_correlation"],
            "spearman_correlation": metrics["spearman_correlation"],
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "valid_predictions": metrics["valid_predictions"],
            "total_samples": total,
            "total_time_seconds": total_time,
            "average_time_per_sample": total_time / total if total > 0 else 0,
            "samples_per_second": total / total_time if total_time > 0 else 0
        }
        
        logger.info(f"Benchmark completed!")
        logger.info(f"Pearson Correlation: {metrics['pearson_correlation']:.4f}")
        logger.info(f"Spearman Correlation: {metrics['spearman_correlation']:.4f}")
        logger.info(f"MSE: {metrics['mse']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"Valid Predictions: {metrics['valid_predictions']}/{total}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average time per sample: {self.metrics['average_time_per_sample']:.3f} seconds")
        
        return self.metrics
    
    def save_results(self):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = os.path.join(self.config.output_dir, f"klue_sts_metrics_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Metrics saved to: {metrics_file}")
        
        # Save detailed results
        if self.config.save_predictions:
            results_file = os.path.join(self.config.output_dir, f"klue_sts_results_{timestamp}.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            logger.info(f"Detailed results saved to: {results_file}")
            
            # Save as CSV for easier analysis
            csv_file = os.path.join(self.config.output_dir, f"klue_sts_results_{timestamp}.csv")
            df = pd.DataFrame(self.results)
            
            # Reorder columns for better readability
            column_order = [
                'id', 'sentence1', 'sentence2', 'true_score', 'predicted_score', 
                'prediction_text', 'finish_reason', 'error'
            ]
            df = df[column_order]
            
            df.to_csv(csv_file, index=False, encoding='utf-8')
            logger.info(f"Results saved as CSV: {csv_file}")
            
            # Save error analysis
            self.save_error_analysis(timestamp)
    
    def save_intermediate_results(self, current_count: int, start_time: float):
        """Save intermediate results during benchmark execution."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        elapsed_time = time.time() - start_time
        
        # Calculate intermediate metrics
        y_true_intermediate = [r["true_score"] for r in self.results]
        y_pred_intermediate = [r["predicted_score"] for r in self.results]
        intermediate_metrics = self.calculate_metrics(y_true_intermediate, y_pred_intermediate)
        
        intermediate_metrics.update({
            "total_samples": current_count,
            "total_time_seconds": elapsed_time,
            "average_time_per_sample": elapsed_time / current_count if current_count > 0 else 0,
            "samples_per_second": current_count / elapsed_time if elapsed_time > 0 else 0,
            "is_intermediate": True,
            "timestamp": timestamp
        })
        
        # Save intermediate metrics
        metrics_file = os.path.join(self.config.output_dir, f"klue_sts_metrics_{current_count:06d}_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(intermediate_metrics, f, ensure_ascii=False, indent=2)
        
        # Save intermediate results
        if self.config.save_predictions:
            results_file = os.path.join(self.config.output_dir, f"klue_sts_results_{current_count:06d}_{timestamp}.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            
            # Save as CSV for easier analysis
            csv_file = os.path.join(self.config.output_dir, f"klue_sts_results_{current_count:06d}_{timestamp}.csv")
            df = pd.DataFrame(self.results)
            
            # Reorder columns for better readability
            column_order = [
                'id', 'sentence1', 'sentence2', 'true_score', 'predicted_score', 
                'prediction_text', 'finish_reason', 'error'
            ]
            df = df[column_order]
            
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            # Save intermediate error analysis
            self.save_intermediate_error_analysis(current_count, timestamp)
        
        logger.info(f"Intermediate results saved at sample {current_count} (Pearson: {intermediate_metrics['pearson_correlation']:.4f})")
    
    def save_intermediate_error_analysis(self, current_count: int, timestamp: str):
        """Save intermediate error analysis."""
        errors = [r for r in self.results if r.get("error") or r["predicted_score"] is None]
        
        if errors:
            error_file = os.path.join(self.config.output_dir, f"klue_sts_error_analysis_{current_count:06d}_{timestamp}.txt")
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write("KLUE STS Intermediate Error Analysis\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Model: {self.config.model_name}\n")
                f.write(f"Platform: Google Cloud Vertex AI\n")
                f.write(f"Project: {self.config.project_id or os.getenv('GOOGLE_CLOUD_PROJECT')}\n")
                f.write(f"Location: {self.config.location}\n")
                f.write(f"Current Sample Count: {current_count}\n")
                f.write(f"Total Errors: {len(errors)}\n")
                f.write(f"Error Rate: {len(errors)/current_count*100:.2f}%\n\n")
                
                f.write("Error Analysis:\n")
                f.write("-" * 40 + "\n")
                
                for i, error in enumerate(errors, 1):
                    f.write(f"{i}. ID: {error['id']}\n")
                    f.write(f"   Sentence 1: {error['sentence1']}\n")
                    f.write(f"   Sentence 2: {error['sentence2']}\n")
                    f.write(f"   True Score: {error['true_score']}\n")
                    f.write(f"   Predicted Score: {error['predicted_score']}\n")
                    f.write(f"   Prediction Text: {error['prediction_text']}\n")
                    if error.get('finish_reason'):
                        f.write(f"   Finish Reason: {error['finish_reason']}\n")
                    if error.get('error'):
                        f.write(f"   Error: {error['error']}\n")
                    f.write("\n")
                    
                    # Limit to first 30 errors for intermediate files
                    if i >= 30:
                        f.write(f"... and {len(errors) - 30} more errors\n")
                        break
    
    def save_error_analysis(self, timestamp: str):
        """Save error analysis to a file for later review."""
        errors = [r for r in self.results if r.get("error") or r["predicted_score"] is None]
        
        if errors:
            error_file = os.path.join(self.config.output_dir, f"klue_sts_error_analysis_{timestamp}.txt")
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write("KLUE STS Error Analysis\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Model: {self.config.model_name}\n")
                f.write(f"Platform: Google Cloud Vertex AI\n")
                f.write(f"Project: {self.config.project_id or os.getenv('GOOGLE_CLOUD_PROJECT')}\n")
                f.write(f"Location: {self.config.location}\n")
                f.write(f"Total Errors: {len(errors)}\n")
                f.write(f"Total Samples: {len(self.results)}\n")
                f.write(f"Error Rate: {len(errors)/len(self.results)*100:.2f}%\n\n")
                
                f.write("Error Analysis:\n")
                f.write("-" * 40 + "\n")
                
                for i, error in enumerate(errors, 1):
                    f.write(f"{i}. ID: {error['id']}\n")
                    f.write(f"   Sentence 1: {error['sentence1']}\n")
                    f.write(f"   Sentence 2: {error['sentence2']}\n")
                    f.write(f"   True Score: {error['true_score']}\n")
                    f.write(f"   Predicted Score: {error['predicted_score']}\n")
                    f.write(f"   Prediction Text: {error['prediction_text']}\n")
                    if error.get('finish_reason'):
                        f.write(f"   Finish Reason: {error['finish_reason']}\n")
                    if error.get('error'):
                        f.write(f"   Error: {error['error']}\n")
                    f.write("\n")
                    
                    # Limit to first 50 errors to keep file manageable
                    if i >= 50:
                        f.write(f"... and {len(errors) - 50} more errors\n")
                        break
                        
            logger.info(f"Error analysis saved to: {error_file}")
        else:
            logger.info("No errors found - no error analysis file created.")
    
    def print_detailed_metrics(self):
        """Print detailed metrics and analysis."""
        print("\n" + "="*60)
        print("KLUE Sentence Textual Similarity Benchmark Results")
        print("="*60)
        
        print(f"Model: {self.config.model_name}")
        print(f"Platform: Google Cloud Vertex AI")
        print(f"Project: {self.config.project_id or os.getenv('GOOGLE_CLOUD_PROJECT')}")
        print(f"Location: {self.config.location}")
        
        # Primary metrics
        print(f"\nPrimary Metrics:")
        print(f"  Pearson Correlation: {self.metrics['pearson_correlation']:.4f}")
        print(f"  Spearman Correlation: {self.metrics['spearman_correlation']:.4f}")
        print(f"  Mean Squared Error (MSE): {self.metrics['mse']:.4f}")
        print(f"  Mean Absolute Error (MAE): {self.metrics['mae']:.4f}")
        print(f"  Valid Predictions: {self.metrics['valid_predictions']}/{self.metrics['total_samples']}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Total Time: {self.metrics['total_time_seconds']:.2f} seconds")
        print(f"  Average Time per Sample: {self.metrics['average_time_per_sample']:.3f} seconds")
        print(f"  Samples per Second: {self.metrics['samples_per_second']:.2f}")
        
        # Score distribution analysis
        valid_results = [r for r in self.results if r["predicted_score"] is not None]
        if valid_results:
            true_scores = [r["true_score"] for r in valid_results]
            pred_scores = [r["predicted_score"] for r in valid_results]
            
            print(f"\nScore Distribution Analysis:")
            print(f"  True Score Range: [{min(true_scores):.2f}, {max(true_scores):.2f}]")
            print(f"  Predicted Score Range: [{min(pred_scores):.2f}, {max(pred_scores):.2f}]")
            print(f"  True Score Mean: {np.mean(true_scores):.2f}")
            print(f"  Predicted Score Mean: {np.mean(pred_scores):.2f}")
        
        # Error analysis
        errors = [r for r in self.results if r.get("error") or r["predicted_score"] is None]
        if errors:
            print(f"\nError Analysis (showing first 5 errors):")
            for i, error in enumerate(errors[:5]):
                print(f"  {i+1}. ID: {error['id']}")
                print(f"     Sentence 1: {error['sentence1'][:50]}...")
                print(f"     Sentence 2: {error['sentence2'][:50]}...")
                print(f"     True Score: {error['true_score']}")
                print(f"     Predicted Score: {error['predicted_score']}")
                if error.get('error'):
                    print(f"     Error: {error['error']}")
                print()

def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="KLUE STS Benchmark with Gemini 2.5 Flash on Vertex AI")
    parser.add_argument("--project-id", type=str, help="Google Cloud project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="Vertex AI location")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to test")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.1, help="Model temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum output tokens")
    parser.add_argument("--no-save-predictions", action="store_true", help="Don't save detailed predictions")
    parser.add_argument("--save-interval", type=int, default=50, help="Save intermediate results every N samples")
    
    args = parser.parse_args()
    
    # Create config
    config = BenchmarkConfig(
        project_id=args.project_id,
        location=args.location,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        save_predictions=not args.no_save_predictions,
        save_interval=args.save_interval
    )
    
    try:
        # Initialize benchmark
        benchmark = KLUESentenceTextualSimilarityBenchmark(config)
        
        # Load dataset
        test_data = benchmark.load_dataset()
        
        # Run benchmark
        metrics = benchmark.run_benchmark(test_data)
        
        # Save results
        benchmark.save_results()
        
        # Print detailed metrics
        benchmark.print_detailed_metrics()
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main() 