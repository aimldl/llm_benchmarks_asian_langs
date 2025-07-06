#!/usr/bin/env python3
"""
KLUE Natural Language Inference (NLI) Benchmark with Gemini 2.5 Flash on Vertex AI
This script benchmarks Gemini 2.5 Flash on the Korean Language Understanding Evaluation (KLUE) Natural Language Inference task using Google Cloud Vertex AI.
"""

import os
import json
import time
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, GenerationConfig
from datasets import load_dataset   
import pandas as pd
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# To verify the model name, see https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash

@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""
    model_name: str = "gemini-2.5-flash"
    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 1.0
    top_k: int = 1
    batch_size: int = 1
    max_samples: Optional[int] = None
    output_dir: str = "benchmark_results"
    save_predictions: bool = True
    project_id: Optional[str] = None
    location: str = "us-central1"

class KLUENaturalLanguageInferenceBenchmark:
    """Benchmark class for KLUE Natural Language Inference task using Vertex AI."""
    
    # KLUE NLI label mapping
    LABEL_MAP = {
        0: "entailment",      # 함의 (premise가 hypothesis를 함의함)
        1: "contradiction",   # 모순 (premise와 hypothesis가 모순됨)
        2: "neutral"          # 중립 (premise와 hypothesis가 중립적 관계)
    }
    
    REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
    
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
            # Get project ID from config or environment
            project_id = self.config.project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
            print(f"project_id: {project_id}")

            if not project_id:
                raise ValueError("Google Cloud project ID must be provided via config.project_id or GOOGLE_CLOUD_PROJECT environment variable")
            
            # Initialize Vertex AI
            aiplatform.init(
                project=project_id,
                location=self.config.location
            )
            logger.info(f"Initialized Vertex AI with project: {project_id}, location: {self.config.location}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
    def _initialize_model(self):
        """Initialize the Gemini model on Vertex AI."""
        try:
            generation_config = GenerationConfig(
                max_output_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
            
            self.model = GenerativeModel(
                model_name=self.config.model_name,
                generation_config=generation_config
            )
            logger.info(f"Initialized model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load the KLUE NLI dataset, convert it to a list of dictionaries,
        and efficiently limit the number of samples based on the configuration.
        """
        try:
            logger.info("Loading KLUE NLI dataset for natural language inference...")
            
            # Load the validation split from the Hugging Face Hub.
            validation_dataset = load_dataset('klue', 'nli', split='validation')

            processed_data = []
            
            # Determine if a subset of data should be used.
            use_subset = self.config.max_samples and self.config.max_samples > 0
            if use_subset:
                 logger.info(f"Preparing to load a subset of {self.config.max_samples} samples.")

            # Efficiently iterate through the dataset.
            for item in validation_dataset:
                # If max_samples is set, break the loop once the limit is reached.
                if use_subset and len(processed_data) >= self.config.max_samples:
                    logger.info(f"Reached sample limit of {self.config.max_samples}. Halting data loading.")
                    break
                    
                # The 'nli' dataset uses 'premise' and 'hypothesis' fields
                processed_data.append({
                    "id": item["guid"],
                    "premise": item["premise"],
                    "hypothesis": item["hypothesis"],
                    "label": item["label"],
                    "label_text": self.LABEL_MAP.get(item["label"], "Unknown Label")
                })

            logger.info(f"✅ Successfully loaded {len(processed_data)} samples.")
            return processed_data
            
        except KeyError as e:
            logger.error(f"❌ A key was not found in the dataset item: {e}. The dataset schema may have changed.")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load or process the dataset: {e}")
            raise
    
    def create_prompt(self, premise: str, hypothesis: str) -> str:
        """Create prompt for natural language inference."""
        prompt = f"""역할: 당신은 한국어 자연어 추론(Natural Language Inference)을 수행하는 "전문 언어 분석 AI"입니다.

임무: 주어진 전제(premise)와 가설(hypothesis) 사이의 논리적 관계를 분석하여 다음 세 가지 중 하나로 분류해 주세요.

분류 기준:

entailment (함의): 전제가 참이라면 가설도 반드시 참이 되는 경우
- 전제의 내용이 가설을 논리적으로 포함하거나 함의하는 관계
- 예시: 전제 "김철수는 의사다" → 가설 "김철수는 의료진이다" → entailment

contradiction (모순): 전제가 참이라면 가설이 거짓이 되는 경우  
- 전제와 가설이 논리적으로 모순되는 관계
- 예시: 전제 "김철수는 의사다" → 가설 "김철수는 의사가 아니다" → contradiction

neutral (중립): 전제가 참이어도 가설의 참거짓을 판단할 수 없는 경우
- 전제와 가설이 논리적으로 독립적이거나 관련이 없는 관계
- 예시: 전제 "김철수는 의사다" → 가설 "오늘 날씨가 좋다" → neutral

지침:
- 전제와 가설의 의미를 정확히 파악하여 논리적 관계를 분석합니다.
- 맥락과 상식을 고려하되, 전제에 명시된 정보를 우선적으로 사용합니다.
- 답변은 반드시 'entailment', 'contradiction', 'neutral' 중 하나로만 제시해 주세요.

전제: {premise}

가설: {hypothesis}

관계:"""
        return prompt
    
    def predict_single(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """Make a single prediction using Vertex AI."""
        try:
            prompt = self.create_prompt(premise, hypothesis)
            
            response = self.model.generate_content(prompt)
            prediction_text = response.text.strip().lower()
            
            # Map prediction back to label
            predicted_label = None
            predicted_label_id = None
            
            # Check for exact matches first
            for label_text, label_id in self.REVERSE_LABEL_MAP.items():
                if label_text in prediction_text:
                    predicted_label = label_text
                    predicted_label_id = label_id
                    break
            
            # If no exact match, try partial matches
            if predicted_label is None:
                if "entail" in prediction_text or "함의" in prediction_text:
                    predicted_label = "entailment"
                    predicted_label_id = 0
                elif "contradict" in prediction_text or "모순" in prediction_text:
                    predicted_label = "contradiction"
                    predicted_label_id = 1
                elif "neutral" in prediction_text or "중립" in prediction_text:
                    predicted_label = "neutral"
                    predicted_label_id = 2
            
            return {
                "prediction_text": prediction_text,
                "predicted_label": predicted_label,
                "predicted_label_id": predicted_label_id
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "prediction_text": "",
                "predicted_label": None,
                "predicted_label_id": None,
                "error": str(e)
            }
    
    def run_benchmark(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the benchmark on test data."""
        logger.info("Starting benchmark...")
        
        correct = 0
        total = len(test_data)
        start_time = time.time()
        
        for i, item in enumerate(tqdm(test_data, desc="Processing samples")):
            # Make prediction
            prediction = self.predict_single(item["premise"], item["hypothesis"])
            
            # Check if prediction is correct
            is_correct = (prediction["predicted_label_id"] == item["label"])
            if is_correct:
                correct += 1
            
            # Store result
            result = {
                "id": item["id"],
                "premise": item["premise"],
                "hypothesis": item["hypothesis"],
                "true_label": item["label"],
                "true_label_text": item["label_text"],
                "predicted_label": prediction["predicted_label"],
                "predicted_label_id": prediction["predicted_label_id"],
                "prediction_text": prediction["prediction_text"],
                "is_correct": is_correct,
                "error": prediction.get("error")
            }
            self.results.append(result)
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        
        self.metrics = {
            "accuracy": accuracy,
            "correct_predictions": correct,
            "total_samples": total,
            "total_time_seconds": total_time,
            "average_time_per_sample": total_time / total if total > 0 else 0,
            "samples_per_second": total / total_time if total_time > 0 else 0
        }
        
        logger.info(f"Benchmark completed!")
        logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average time per sample: {self.metrics['average_time_per_sample']:.3f} seconds")
        
        return self.metrics
    
    def save_results(self):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = os.path.join(self.config.output_dir, f"klue_nli_metrics_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Metrics saved to: {metrics_file}")
        
        # Save detailed results
        if self.config.save_predictions:
            results_file = os.path.join(self.config.output_dir, f"klue_nli_results_{timestamp}.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            logger.info(f"Detailed results saved to: {results_file}")
            
            # Save as CSV for easier analysis
            csv_file = os.path.join(self.config.output_dir, f"klue_nli_results_{timestamp}.csv")
            df = pd.DataFrame(self.results)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            logger.info(f"Results saved as CSV: {csv_file}")
    
    def print_detailed_metrics(self):
        """Print detailed metrics and analysis."""
        print("\n" + "="*60)
        print("KLUE Natural Language Inference Benchmark Results")
        print("="*60)
        
        print(f"Model: {self.config.model_name}")
        print(f"Platform: Google Cloud Vertex AI")
        print(f"Project: {self.config.project_id or os.getenv('GOOGLE_CLOUD_PROJECT')}")
        print(f"Location: {self.config.location}")
        print(f"Accuracy: {self.metrics['accuracy']:.4f} ({self.metrics['correct_predictions']}/{self.metrics['total_samples']})")
        print(f"Total Time: {self.metrics['total_time_seconds']:.2f} seconds")
        print(f"Average Time per Sample: {self.metrics['average_time_per_sample']:.3f} seconds")
        print(f"Samples per Second: {self.metrics['samples_per_second']:.2f}")
        
        # Per-label accuracy
        print("\nPer-label Accuracy:")
        label_correct = {label: 0 for label in self.LABEL_MAP.values()}
        label_total = {label: 0 for label in self.LABEL_MAP.values()}
        
        for result in self.results:
            true_label = result["true_label_text"]
            label_total[true_label] += 1
            if result["is_correct"]:
                label_correct[true_label] += 1
        
        for label in self.LABEL_MAP.values():
            if label_total[label] > 0:
                acc = label_correct[label] / label_total[label]
                print(f"  {label}: {acc:.4f} ({label_correct[label]}/{label_total[label]})")
        
        # Error analysis
        errors = [r for r in self.results if not r["is_correct"]]
        if errors:
            print(f"\nError Analysis (showing first 5 errors):")
            for i, error in enumerate(errors[:5]):
                print(f"  {i+1}. True: {error['true_label_text']} | Predicted: {error['predicted_label']}")
                print(f"     Premise: {error['premise'][:100]}...")
                print(f"     Hypothesis: {error['hypothesis'][:100]}...")
                print(f"     Prediction: {error['prediction_text']}")
                print()

def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="KLUE Natural Language Inference Benchmark with Gemini 2.5 Flash on Vertex AI")
    parser.add_argument("--project-id", type=str, help="Google Cloud project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="Vertex AI location")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to test")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum output tokens")
    parser.add_argument("--no-save-predictions", action="store_true", help="Don't save detailed predictions")
    
    args = parser.parse_args()
    
    # Create config
    config = BenchmarkConfig(
        project_id=args.project_id,
        location=args.location,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        save_predictions=not args.no_save_predictions
    )
    
    try:
        # Initialize benchmark
        benchmark = KLUENaturalLanguageInferenceBenchmark(config)
        
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