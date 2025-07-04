#!/usr/bin/env python3
"""
KLUE Topic Classification (TC) Benchmark with Gemini 2.5 Flash on Vertex AI
This script benchmarks Gemini 2.5 Flash on the Korean Language Understanding Evaluation (KLUE) Topic Classification task using Google Cloud Vertex AI.
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

@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""
    model_name: str = "gemini-2.0-flash-exp"
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    batch_size: int = 1
    max_samples: Optional[int] = None
    output_dir: str = "benchmark_results"
    save_predictions: bool = True
    project_id: Optional[str] = None
    location: str = "us-central1"

class KLUETopicClassificationBenchmark:
    """Benchmark class for KLUE Topic Classification task using Vertex AI."""
    
    # KLUE TC label mapping
    LABEL_MAP = {
        0: "정치",
        1: "경제", 
        2: "사회",
        3: "생활문화",
        4: "세계",
        5: "IT과학",
        6: "스포츠"
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
        """Load KLUE TC dataset."""
        try:
            logger.info("Loading KLUE TC dataset...")
            dataset = load_dataset("klue", "tc")
            
            # Convert to list of dictionaries
            test_data = []
            for item in dataset["test"]:
                test_data.append({
                    "id": item["guid"],
                    "text": item["title"] + " " + item["text"],
                    "label": item["label"],
                    "label_text": self.LABEL_MAP[item["label"]]
                })
            
            if self.config.max_samples:
                test_data = test_data[:self.config.max_samples]
            
            logger.info(f"Loaded {len(test_data)} test samples")
            return test_data
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def create_prompt(self, text: str) -> str:
        """Create prompt for topic classification."""
        prompt = f"""다음 한국어 텍스트의 주제를 분류해주세요.

주제 카테고리:
- 정치: 정치 관련 뉴스, 정책, 선거 등
- 경제: 경제, 금융, 주식, 부동산, 기업 관련 뉴스
- 사회: 사회 문제, 사건사고, 교육, 의료 등
- 생활문화: 문화, 예술, 연예, 음식, 여행, 패션 등
- 세계: 국제 뉴스, 외교, 해외 사건 등
- IT과학: 기술, 과학, IT, 인터넷, 소프트웨어 등
- 스포츠: 스포츠, 운동, 경기 결과 등

텍스트: {text}

주제:"""
        return prompt
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """Make a single prediction using Vertex AI."""
        try:
            prompt = self.create_prompt(text)
            
            response = self.model.generate_content(prompt)
            prediction_text = response.text.strip()
            
            # Map prediction back to label
            predicted_label = None
            predicted_label_id = None
            
            for label_text, label_id in self.REVERSE_LABEL_MAP.items():
                if label_text in prediction_text:
                    predicted_label = label_text
                    predicted_label_id = label_id
                    break
            
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
            prediction = self.predict_single(item["text"])
            
            # Check if prediction is correct
            is_correct = (prediction["predicted_label_id"] == item["label"])
            if is_correct:
                correct += 1
            
            # Store result
            result = {
                "id": item["id"],
                "text": item["text"],
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
        metrics_file = os.path.join(self.config.output_dir, f"klue_tc_metrics_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Metrics saved to: {metrics_file}")
        
        # Save detailed results
        if self.config.save_predictions:
            results_file = os.path.join(self.config.output_dir, f"klue_tc_results_{timestamp}.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            logger.info(f"Detailed results saved to: {results_file}")
            
            # Save as CSV for easier analysis
            csv_file = os.path.join(self.config.output_dir, f"klue_tc_results_{timestamp}.csv")
            df = pd.DataFrame(self.results)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            logger.info(f"Results saved as CSV: {csv_file}")
    
    def print_detailed_metrics(self):
        """Print detailed metrics and analysis."""
        print("\n" + "="*60)
        print("KLUE Topic Classification Benchmark Results")
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
                print(f"     Text: {error['text'][:100]}...")
                print(f"     Prediction: {error['prediction_text']}")
                print()

def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="KLUE Topic Classification Benchmark with Gemini 2.5 Flash on Vertex AI")
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
        benchmark = KLUETopicClassificationBenchmark(config)
        
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