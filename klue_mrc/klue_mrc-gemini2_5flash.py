#!/usr/bin/env python3
"""
KLUE Machine Reading Comprehension (MRC) Benchmark with Gemini 2.5 Flash on Vertex AI
This script benchmarks Gemini 2.5 Flash on the Korean Language Understanding Evaluation (KLUE) Machine Reading Comprehension task using Google Cloud Vertex AI.
"""

import os
import json
import time
import argparse
import re
from typing import List, Dict, Any, Optional, Tuple
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""
    model_name: str = "gemini-2.5-flash"
    sleep_interval_between_api_calls: float = 0.04 # sec
    max_tokens: int = 2048  # Increased for MRC task
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

class KLUEMachineReadingComprehensionBenchmark:
    """Benchmark class for KLUE Machine Reading Comprehension task using Vertex AI."""
    
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
        Load the KLUE MRC dataset, convert it to a list of dictionaries,
        and efficiently limit the number of samples based on the configuration.
        """
        try:
            logger.info("Loading KLUE MRC dataset for machine reading comprehension...")
            
            # Load the validation split from the Hugging Face Hub.
            validation_dataset = load_dataset('klue', 'mrc', split='validation')

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
                    
                # Process MRC data
                processed_data.append({
                    "id": item["guid"],
                    "title": item["title"],
                    "context": item["context"],
                    "question": item["question"],
                    "answers": item["answers"],
                    "answer_start": item.get("answer_start", []),
                    "is_impossible": item.get("is_impossible", False)
                })

            logger.info(f"✅ Successfully loaded {len(processed_data)} samples.")
            return processed_data
            
        except KeyError as e:
            logger.error(f"❌ A key was not found in the dataset item: {e}. The dataset schema may have changed.")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load or process the dataset: {e}")
            raise
    
    def create_prompt(self, title: str, context: str, question: str, is_impossible: bool = False) -> str:
        """Create detailed prompt for machine reading comprehension."""
        prompt = f"""역할: 당신은 한국어 텍스트를 읽고 질문에 답하는 "전문 독해 AI"입니다.

임무: 주어진 지문을 읽고 질문에 대한 정확한 답을 찾아주세요. 답이 지문에 없는 경우 "답을 찾을 수 없습니다"라고 답하세요.

지침:

1. **정확한 답 찾기**: 질문에 대한 답이 지문에 명확히 나와 있는지 확인하세요.
2. **문맥 이해**: 지문의 전체적인 맥락을 파악하여 정확한 답을 찾으세요.
3. **답의 형태**: 
   - 답이 지문에 있으면: 지문에서 그대로 추출하여 답하세요
   - 답이 지문에 없으면: "답을 찾을 수 없습니다"라고 답하세요
4. **한국어 특성 고려**: 한국어의 문법과 표현을 정확히 이해하여 답하세요.
5. **명확성**: 답은 간결하고 명확해야 합니다.

제목: {title}

지문:
{context}

질문: {question}

답변:"""

        if is_impossible:
            prompt += "\n\n참고: 이 질문은 지문에서 답을 찾을 수 없는 질문일 수 있습니다."
        
        return prompt
    
    def configure_safety_settings(self, threshold=HarmBlockThreshold.BLOCK_NONE):
        """Configure safety settings for the model."""
        return [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=threshold,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=threshold,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=threshold,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=threshold,
            ),
        ]
    
    def predict_single(self, title: str, context: str, question: str, is_impossible: bool = False) -> Dict[str, Any]:
        """Make a single prediction for MRC task."""
        try:
            prompt = self.create_prompt(title, context, question, is_impossible)
            
            # Configure safety settings
            safety_settings = self.configure_safety_settings()
            
            # Generate content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    safety_settings=safety_settings,
                    max_output_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                ),
            )
            
            # Extract response text
            if response and response.text:
                predicted_answer = response.text.strip()
                return {
                    "success": True,
                    "answer": predicted_answer,
                    "raw_response": response.text
                }
            else:
                logger.error("Cannot get the response text.")
                return {
                    "success": False,
                    "answer": "답을 찾을 수 없습니다",
                    "raw_response": "",
                    "error": "No response text"
                }
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "success": False,
                "answer": "답을 찾을 수 없습니다",
                "raw_response": "",
                "error": str(e)
            }
    
    def calculate_exact_match(self, predicted_answer: str, ground_truth_answers: List[str]) -> bool:
        """Calculate exact match score."""
        predicted_clean = self._normalize_answer(predicted_answer)
        
        for gt_answer in ground_truth_answers:
            gt_clean = self._normalize_answer(gt_answer)
            if predicted_clean == gt_clean:
                return True
        
        return False
    
    def calculate_f1_score(self, predicted_answer: str, ground_truth_answers: List[str]) -> float:
        """Calculate F1 score for answer prediction."""
        predicted_clean = self._normalize_answer(predicted_answer)
        
        best_f1 = 0.0
        for gt_answer in ground_truth_answers:
            gt_clean = self._normalize_answer(gt_answer)
            
            # Split into words/tokens
            pred_tokens = predicted_clean.split()
            gt_tokens = gt_clean.split()
            
            if not pred_tokens or not gt_tokens:
                continue
            
            # Calculate common tokens
            common = set(pred_tokens) & set(gt_tokens)
            
            if not common:
                continue
            
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gt_tokens)
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                best_f1 = max(best_f1, f1)
        
        return best_f1
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', answer.strip())
        # Remove punctuation for comparison (optional)
        # normalized = re.sub(r'[^\w\s]', '', normalized)
        return normalized
    
    def calculate_metrics(self, predicted_answer: str, ground_truth_answers: List[str], is_impossible: bool) -> Dict[str, Any]:
        """Calculate MRC metrics."""
        if is_impossible:
            # For impossible questions, check if model correctly says "답을 찾을 수 없습니다"
            predicted_clean = self._normalize_answer(predicted_answer)
            impossible_indicators = ["답을 찾을 수 없습니다", "답을 찾을 수 없음", "알 수 없음", "모름"]
            
            is_correct = any(indicator in predicted_clean for indicator in impossible_indicators)
            
            return {
                "exact_match": 1.0 if is_correct else 0.0,
                "f1_score": 1.0 if is_correct else 0.0,
                "is_impossible_correct": is_correct
            }
        else:
            # For answerable questions
            exact_match = self.calculate_exact_match(predicted_answer, ground_truth_answers)
            f1_score = self.calculate_f1_score(predicted_answer, ground_truth_answers)
            
            return {
                "exact_match": 1.0 if exact_match else 0.0,
                "f1_score": f1_score,
                "is_impossible_correct": True  # Not applicable for answerable questions
            }
    
    def run_benchmark(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the MRC benchmark."""
        logger.info("Starting benchmark...")
        
        start_time = time.time()
        total_samples = len(test_data)
        total_exact_match = 0
        total_f1_score = 0.0
        total_impossible_correct = 0
        impossible_count = 0
        
        # Process each sample
        for i, sample in enumerate(tqdm(test_data, desc="Processing samples")):
            try:
                # Make prediction
                prediction_result = self.predict_single(
                    sample["title"],
                    sample["context"],
                    sample["question"],
                    sample["is_impossible"]
                )
                
                # Calculate metrics
                metrics = self.calculate_metrics(
                    prediction_result.get("answer", ""),
                    sample["answers"],
                    sample["is_impossible"]
                )
                
                # Update counters
                total_exact_match += metrics["exact_match"]
                total_f1_score += metrics["f1_score"]
                
                if sample["is_impossible"]:
                    impossible_count += 1
                    if metrics["is_impossible_correct"]:
                        total_impossible_correct += 1
                
                # Store result
                result = {
                    "id": sample["id"],
                    "title": sample["title"],
                    "context": sample["context"],
                    "question": sample["question"],
                    "ground_truth_answers": sample["answers"],
                    "is_impossible": sample["is_impossible"],
                    "predicted_answer": prediction_result.get("answer", ""),
                    "metrics": metrics,
                    "success": prediction_result["success"],
                    "raw_response": prediction_result.get("raw_response", ""),
                    "error": prediction_result.get("error", "")
                }
                
                self.results.append(result)
                
                # Save intermediate results
                if (i + 1) % self.config.save_interval == 0:
                    self.save_intermediate_results(i + 1, total_exact_match, total_f1_score, start_time)
                
                # Sleep between API calls
                time.sleep(self.config.sleep_interval_between_api_calls)
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                self.results.append({
                    "id": sample["id"],
                    "title": sample["title"],
                    "context": sample["context"],
                    "question": sample["question"],
                    "ground_truth_answers": sample["answers"],
                    "is_impossible": sample["is_impossible"],
                    "predicted_answer": "답을 찾을 수 없습니다",
                    "metrics": {"exact_match": 0.0, "f1_score": 0.0, "is_impossible_correct": False},
                    "success": False,
                    "raw_response": "",
                    "error": str(e)
                })
        
        # Calculate final metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        exact_match_score = total_exact_match / total_samples if total_samples > 0 else 0.0
        f1_score = total_f1_score / total_samples if total_samples > 0 else 0.0
        impossible_accuracy = total_impossible_correct / impossible_count if impossible_count > 0 else 0.0
        
        self.metrics = {
            "total_samples": total_samples,
            "answerable_samples": total_samples - impossible_count,
            "impossible_samples": impossible_count,
            "exact_match": exact_match_score,
            "f1_score": f1_score,
            "impossible_accuracy": impossible_accuracy,
            "total_time": total_time,
            "average_time_per_sample": total_time / total_samples if total_samples > 0 else 0.0,
            "samples_per_second": total_samples / total_time if total_time > 0 else 0.0
        }
        
        logger.info("Benchmark completed!")
        logger.info(f"Exact Match: {exact_match_score:.4f}")
        logger.info(f"F1 Score: {f1_score:.4f}")
        logger.info(f"Impossible Accuracy: {impossible_accuracy:.4f}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average time per sample: {total_time / total_samples:.3f} seconds")
        
        return self.metrics
    
    def save_results(self):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = os.path.join(self.config.output_dir, f"klue_mrc_metrics_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Metrics saved to: {metrics_file}")
        
        # Save detailed results
        results_file = os.path.join(self.config.output_dir, f"klue_mrc_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Save as CSV
        csv_data = []
        for result in self.results:
            csv_data.append({
                "id": result["id"],
                "title": result["title"],
                "context": result["context"][:500] + "..." if len(result["context"]) > 500 else result["context"],
                "question": result["question"],
                "ground_truth_answers": " | ".join(result["ground_truth_answers"]),
                "is_impossible": result["is_impossible"],
                "predicted_answer": result["predicted_answer"],
                "exact_match": result["metrics"]["exact_match"],
                "f1_score": result["metrics"]["f1_score"],
                "success": result["success"],
                "error": result.get("error", "")
            })
        
        csv_file = os.path.join(self.config.output_dir, f"klue_mrc_results_{timestamp}.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"Results saved as CSV: {csv_file}")
        
        # Save error analysis
        self.save_error_analysis(timestamp)
    
    def save_intermediate_results(self, current_count: int, exact_match_count: int, f1_sum: float, start_time: float):
        """Save intermediate results."""
        if not self.config.save_predictions:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate intermediate metrics
        exact_match_score = exact_match_count / current_count if current_count > 0 else 0.0
        f1_score = f1_sum / current_count if current_count > 0 else 0.0
        
        intermediate_metrics = {
            "samples_processed": current_count,
            "exact_match": exact_match_score,
            "f1_score": f1_score,
            "timestamp": timestamp
        }
        
        # Save intermediate metrics
        metrics_file = os.path.join(self.config.output_dir, f"klue_mrc_metrics_{current_count:06d}_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(intermediate_metrics, f, ensure_ascii=False, indent=2)
        
        # Save intermediate results
        results_file = os.path.join(self.config.output_dir, f"klue_mrc_results_{current_count:06d}_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Intermediate results saved at {current_count} samples")
    
    def save_error_analysis(self, timestamp: str):
        """Save error analysis for failed predictions."""
        error_samples = [r for r in self.results if not r["success"] or r.get("error") or r["metrics"]["exact_match"] < 0.5]
        
        if not error_samples:
            logger.info("No errors to analyze")
            return
        
        error_file = os.path.join(self.config.output_dir, f"klue_mrc_error_analysis_{timestamp}.txt")
        
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write("KLUE MRC Error Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            for i, sample in enumerate(error_samples[:10], 1):  # Show first 10 errors
                f.write(f"{i}. Sample ID: {sample['id']}\n")
                f.write(f"   Title: {sample['title']}\n")
                f.write(f"   Question: {sample['question']}\n")
                f.write(f"   Context: {sample['context'][:200]}...\n")
                f.write(f"   Ground Truth: {sample['ground_truth_answers']}\n")
                f.write(f"   Predicted: {sample['predicted_answer']}\n")
                f.write(f"   Exact Match: {sample['metrics']['exact_match']:.4f}\n")
                f.write(f"   F1 Score: {sample['metrics']['f1_score']:.4f}\n")
                f.write(f"   Is Impossible: {sample['is_impossible']}\n")
                if sample.get("error"):
                    f.write(f"   Error: {sample['error']}\n")
                f.write("\n")
        
        logger.info(f"Error analysis saved to: {error_file}")
    
    def print_detailed_metrics(self):
        """Print detailed benchmark results."""
        print("=" * 60)
        print("KLUE Machine Reading Comprehension Benchmark Results")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Platform: Google Cloud Vertex AI")
        print(f"Project: {self.config.project_id or os.getenv('GOOGLE_CLOUD_PROJECT')}")
        print(f"Location: {self.config.location}")
        print(f"Exact Match: {self.metrics['exact_match']:.4f}")
        print(f"F1 Score: {self.metrics['f1_score']:.4f}")
        print(f"Impossible Accuracy: {self.metrics['impossible_accuracy']:.4f}")
        print(f"Total Samples: {self.metrics['total_samples']}")
        print(f"Answerable Samples: {self.metrics['answerable_samples']}")
        print(f"Impossible Samples: {self.metrics['impossible_samples']}")
        print(f"Total Time: {self.metrics['total_time']:.2f} seconds")
        print(f"Average Time per Sample: {self.metrics['average_time_per_sample']:.3f} seconds")
        print(f"Samples per Second: {self.metrics['samples_per_second']:.2f}")
        print()
        
        # Per-type analysis
        answerable_results = [r for r in self.results if not r["is_impossible"]]
        impossible_results = [r for r in self.results if r["is_impossible"]]
        
        if answerable_results:
            answerable_exact_match = sum(r["metrics"]["exact_match"] for r in answerable_results) / len(answerable_results)
            answerable_f1 = sum(r["metrics"]["f1_score"] for r in answerable_results) / len(answerable_results)
            print("Answerable Questions Performance:")
            print(f"  Exact Match: {answerable_exact_match:.4f}")
            print(f"  F1 Score: {answerable_f1:.4f}")
            print(f"  Sample Count: {len(answerable_results)}")
            print()
        
        if impossible_results:
            impossible_correct = sum(1 for r in impossible_results if r["metrics"]["is_impossible_correct"])
            impossible_accuracy = impossible_correct / len(impossible_results)
            print("Impossible Questions Performance:")
            print(f"  Accuracy: {impossible_accuracy:.4f}")
            print(f"  Correct: {impossible_correct}/{len(impossible_results)}")
            print()
        
        # Error analysis
        error_count = sum(1 for r in self.results if not r["success"] or r.get("error") or r["metrics"]["exact_match"] < 0.5)
        if error_count > 0:
            print(f"Error Analysis (showing first 5 errors):")
            error_samples = [r for r in self.results if not r["success"] or r.get("error") or r["metrics"]["exact_match"] < 0.5]
            for i, sample in enumerate(error_samples[:5], 1):
                print(f"  {i}. Sample ID: {sample['id']}")
                print(f"     Question: {sample['question'][:100]}...")
                print(f"     Ground Truth: {sample['ground_truth_answers']}")
                print(f"     Predicted: {sample['predicted_answer'][:100]}...")
                print(f"     Exact Match: {sample['metrics']['exact_match']:.4f} | F1: {sample['metrics']['f1_score']:.4f}")
                if sample.get("error"):
                    print(f"     Error: {sample['error']}")
                print()

def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="KLUE MRC Benchmark with Gemini 2.5 Flash")
    parser.add_argument("--project-id", type=str, help="Google Cloud project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="Vertex AI location")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to test")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.1, help="Model temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum output tokens")
    parser.add_argument("--no-save-predictions", action="store_true", help="Skip saving detailed predictions")
    parser.add_argument("--save-interval", type=int, default=50, help="Save intermediate results every N samples")
    
    args = parser.parse_args()
    
    # Create configuration
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
    
    # Create and run benchmark
    benchmark = KLUEMachineReadingComprehensionBenchmark(config)
    
    # Load dataset
    test_data = benchmark.load_dataset()
    
    # Run benchmark
    metrics = benchmark.run_benchmark(test_data)
    
    # Save results
    benchmark.save_results()
    
    # Print detailed results
    benchmark.print_detailed_metrics()

if __name__ == "__main__":
    main() 