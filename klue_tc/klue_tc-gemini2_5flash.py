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
#from google.cloud import aiplatform
#from vertexai.generative_models import GenerativeModel, GenerationConfig
from google import genai
from google.genai.types import (
    FunctionDeclaration,
    GenerateContentConfig,
    GoogleSearch,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    SafetySetting,
    ThinkingConfig,
    Tool,
    ToolCodeExecution,
)
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
    sleep_interval_between_api_calls: float = 0.04 # sec
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

class KLUETopicClassificationBenchmark:
    """Benchmark class for KLUE Topic Classification task using Vertex AI."""
    
    # KLUE TC label mapping
    # Corrected LABEL_MAP to match the order from the Hugging Face dataset features.
    # The order is: 0:IT과학, 1:경제, 2:사회, 3:생활문화, 4:세계, 5:스포츠, 6:정치
    # From
    # LABEL_MAP = {
    #     0: "정치",
    #     1: "경제", 
    #     2: "사회",
    #     3: "생활문화",
    #     4: "세계",
    #     5: "IT과학",
    #     6: "스포츠"
    # }
    # to
    LABEL_MAP = {
        0: "IT과학",
        1: "경제",
        2: "사회",
        3: "생활문화",
        4: "세계",
        5: "스포츠",
        6: "정치"
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
            # Hard-coded project ID for debugging/testing
            #   project_id = "vertex-workbench-notebook"  
            # Corrected: Use project ID from config or environment for better practice
            project_id = self.config.project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                raise ValueError("Google Cloud project ID must be provided via the --project-id flag or by setting the GOOGLE_CLOUD_PROJECT environment variable.")
    
            # Get project ID from config or environment
           #project_id = self.config.project_id or os.getenv("GOOGLE_CLOUD_PROJECT")  # Error: GOOGLE_CLOUD_PROJECT is not set
            print(f"project_id: {project_id}")

            if not project_id:
                raise ValueError("Google Cloud project ID must be provided via config.project_id or GOOGLE_CLOUD_PROJECT environment variable")
            
            # Initialize Vertex AI
            client = genai.Client(vertexai=True, project=project_id, location=self.config.location)
            # aiplatform.init(
            #     project=project_id,
            #     location=self.config.location
            # )
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
        Load the KLUE YNAT dataset, convert it to a list of dictionaries,
        and efficiently limit the number of samples based on the configuration.
        """
        try:
            logger.info("Loading KLUE YNAT dataset for topic classification...")
            
            # Load the validation split from the Hugging Face Hub.
            validation_dataset = load_dataset('klue', 'ynat', split='validation')

            # Uncomment the following line for debugging to verify the label mapping.
            # logger.info(f"Dataset label features: {validation_dataset.features['label'].names}")

            processed_data = []
            
            # Determine if a subset of data should be used.
            use_subset = self.config.max_samples and self.config.max_samples > 0
            if use_subset:
                 logger.info(f"Preparing to load a subset of {self.config.max_samples} samples.")

            # Efficiently iterate through the dataset.
            for item in validation_dataset:
                # If max_samples is set, break the loop once the limit is reached.
                # This is much more efficient than building a full list and then slicing it.
                if use_subset and len(processed_data) >= self.config.max_samples:
                    logger.info(f"Reached sample limit of {self.config.max_samples}. Halting data loading.")
                    break
                    
                # The 'ynat' dataset uses the 'title' field for the text content.
                # Using .get() for the label map is safer; it prevents a KeyError
                # if an unexpected label index appears.
                processed_data.append({
                    "id": item["guid"],
                    "text": item["title"],
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
    
    def create_prompt(self, text: str) -> str:
        """Create prompt for topic classification."""
        prompt = f"""역할: 당신은 다양한 한국어 텍스트의 핵심 주제를 정확하게 분석하고 분류하는 "전문 텍스트 분류 AI"입니다.

임무: 아래에 제시된 텍스트의 핵심 내용을 파악하여, 가장 적합한 주제 카테고리 하나를 선택해 주세요.

주제 카테고리:

정치: 국회, 정당, 대통령, 정부 부처, 정책, 선거, 입법, 행정, 지방자치 등 국내 정치 관련 소식

경제: 국내 및 국제 경제 동향, 금융, 증권, 주식, 채권, 부동산, 기업 경영, 산업, 무역, 고용, 물가 등

사회: 사회적 문제 및 현상, 사건·사고, 법원, 검찰, 교육, 노동, 환경, 의료, 복지, 인권, 시민사회 등

생활문화: 예술, 대중문화(영화, 드라마, 음악), 연예, 패션, 음식, 여행, 건강, 취미, 종교, 도서 등 일상생활과 관련된 정보

세계: 국제 정세, 외교, 분쟁, 해외 주요 사건·사고, 국가 간 관계 등 한국 외 국가에서 발생한 소식

IT과학: 정보 기술(IT), 인공지능(AI), 반도체, 인터넷, 소프트웨어, 최신 과학 연구, 우주, 생명 공학 등

스포츠: 국내 및 해외 스포츠 경기, 선수, 구단, 대회 결과, e스포츠 등

지침:

주어진 텍스트 전체의 맥락을 종합적으로 고려하여 핵심 주제를 판단합니다.

두 개 이상의 카테고리에 해당될 수 있는 내용일 경우, 텍스트에서 가장 비중 있게 다루는 주제를 우선적으로 선택합니다. 예를 들어, '정부의 IT 산업 육성 정책'에 대한 글이라면 '경제'나 'IT과학'도 관련이 있지만, 정책 발표가 핵심이므로 '정치'로 분류합니다.

답변은 지정된 카테고리명과 정확히 일치해야 합니다.

다른 설명 없이 주제 카테고리만 간결하게 제시해 주세요.

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
                "is_correct": is_correct,
                "true_label_text": item["label_text"],
                "prediction_text": prediction["prediction_text"],  
                "true_label": item["label"],
                "predicted_label": prediction["predicted_label"],
                "predicted_label_id": prediction["predicted_label_id"],    
                "error": prediction.get("error")
            }
            self.results.append(result)
            
            # Add small delay to avoid rate limiting
            time.sleep(self.config.sleep_interval_between_api_calls)
        
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