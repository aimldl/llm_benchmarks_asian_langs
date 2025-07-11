#!/usr/bin/env python3
"""
KLUE Relation Extraction (RE) Benchmark with Gemini 2.5 Flash on Vertex AI
This script benchmarks Gemini 2.5 Flash on the Korean Language Understanding Evaluation (KLUE) Relation Extraction task using Google Cloud Vertex AI.
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
    max_tokens: int = 2048  # Increased for RE task
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

class KLUERelationExtractionBenchmark:
    """Benchmark class for KLUE Relation Extraction task using Vertex AI."""
    
    # KLUE RE relation types
    RELATION_TYPES = {
        "no_relation": "관계 없음",
        "org:top_members/employees": "조직:최고경영진/직원",
        "org:members": "조직:구성원",
        "org:product": "조직:제품",
        "org:founded": "조직:설립",
        "org:alternate_names": "조직:대체명",
        "org:place_of_headquarters": "조직:본사위치",
        "org:number_of_employees/members": "조직:직원/구성원수",
        "org:website": "조직:웹사이트",
        "org:subsidiaries": "조직:자회사",
        "org:parents": "조직:상위조직",
        "org:dissolved": "조직:해산",
        "per:title": "인물:직책",
        "per:employee_of": "인물:소속",
        "per:member_of": "인물:소속조직",
        "per:schools_attended": "인물:출신학교",
        "per:works_for": "인물:근무처",
        "per:countries_of_residence": "인물:거주국",
        "per:stateorprovinces_of_residence": "인물:거주지역",
        "per:cities_of_residence": "인물:거주도시",
        "per:countries_of_birth": "인물:출생국",
        "per:stateorprovinces_of_birth": "인물:출생지역",
        "per:cities_of_birth": "인물:출생도시",
        "per:date_of_birth": "인물:출생일",
        "per:date_of_death": "인물:사망일",
        "per:place_of_birth": "인물:출생지",
        "per:place_of_death": "인물:사망지",
        "per:cause_of_death": "인물:사망원인",
        "per:origin": "인물:출신",
        "per:religion": "인물:종교",
        "per:spouse": "인물:배우자",
        "per:children": "인물:자녀",
        "per:parents": "인물:부모",
        "per:siblings": "인물:형제자매",
        "per:other_family": "인물:기타가족",
        "per:charges": "인물:혐의",
        "per:alternate_names": "인물:대체명",
        "per:age": "인물:나이",
        "per:date_of_birth": "인물:출생일",
        "per:date_of_death": "인물:사망일",
        "per:place_of_birth": "인물:출생지",
        "per:place_of_death": "인물:사망지",
        "per:cause_of_death": "인물:사망원인",
        "per:origin": "인물:출신",
        "per:religion": "인물:종교",
        "per:spouse": "인물:배우자",
        "per:children": "인물:자녀",
        "per:parents": "인물:부모",
        "per:siblings": "인물:형제자매",
        "per:other_family": "인물:기타가족",
        "per:charges": "인물:혐의",
        "per:alternate_names": "인물:대체명",
        "per:age": "인물:나이"
    }
    
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
        Load the KLUE RE dataset, convert it to a list of dictionaries,
        and efficiently limit the number of samples based on the configuration.
        """
        try:
            logger.info("Loading KLUE RE dataset for relation extraction...")
            
            # Load the validation split from the Hugging Face Hub.
            validation_dataset = load_dataset('klue', 're', split='validation')

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
                    
                # Process RE data - convert to expected format
                processed_data.append({
                    "id": item["guid"],
                    "sentence": item["sentence"],
                    "subject_entity": {
                        "text": item["subject_entity"]["word"],
                        "type": item["subject_entity"]["type"]
                    },
                    "object_entity": {
                        "text": item["object_entity"]["word"],
                        "type": item["object_entity"]["type"]
                    },
                    "label": item["label"],
                    "label_text": self.RELATION_TYPES.get(item["label"], "Unknown Relation")
                })

            logger.info(f"✅ Successfully loaded {len(processed_data)} samples.")
            return processed_data
            
        except KeyError as e:
            logger.error(f"❌ A key was not found in the dataset item: {e}. The dataset schema may have changed.")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load or process the dataset: {e}")
            raise
    
    def create_prompt(self, sentence: str, subject_entity: Dict, object_entity: Dict) -> str:
        """Create detailed prompt for relation extraction."""
        subject_text = subject_entity["text"]
        subject_type = subject_entity["type"]
        object_text = object_entity["text"]
        object_type = object_entity["type"]
        
        prompt = f"""역할: 당신은 한국어 텍스트에서 두 개체 간의 관계를 정확하게 분석하고 분류하는 "전문 관계 추출 AI"입니다.

임무: 아래에 제시된 문장에서 두 개체 간의 관계를 파악하여, 가장 적합한 관계 유형을 선택해 주세요.

문장: {sentence}

개체 1: {subject_text} (유형: {subject_type})
개체 2: {object_text} (유형: {object_type})

관계 유형 정의:

1. 조직 관련 관계 (org:):
   - org:top_members/employees: 조직의 최고경영진이나 직원 관계
   - org:members: 조직의 구성원 관계
   - org:product: 조직이 생산하는 제품 관계
   - org:founded: 조직의 설립 관계
   - org:alternate_names: 조직의 대체명이나 별칭
   - org:place_of_headquarters: 조직의 본사 위치
   - org:number_of_employees/members: 조직의 직원/구성원 수
   - org:website: 조직의 웹사이트
   - org:subsidiaries: 조직의 자회사
   - org:parents: 조직의 상위조직
   - org:dissolved: 조직의 해산

2. 인물 관련 관계 (per:):
   - per:title: 인물의 직책이나 호칭
   - per:employee_of: 인물이 소속된 조직
   - per:member_of: 인물이 속한 조직이나 단체
   - per:schools_attended: 인물이 다닌 학교
   - per:works_for: 인물이 근무하는 곳
   - per:countries_of_residence: 인물이 거주하는 국가
   - per:stateorprovinces_of_residence: 인물이 거주하는 지역
   - per:cities_of_residence: 인물이 거주하는 도시
   - per:countries_of_birth: 인물의 출생국
   - per:stateorprovinces_of_birth: 인물의 출생지역
   - per:cities_of_birth: 인물의 출생도시
   - per:date_of_birth: 인물의 출생일
   - per:date_of_death: 인물의 사망일
   - per:place_of_birth: 인물의 출생지
   - per:place_of_death: 인물의 사망지
   - per:cause_of_death: 인물의 사망원인
   - per:origin: 인물의 출신
   - per:religion: 인물의 종교
   - per:spouse: 인물의 배우자
   - per:children: 인물의 자녀
   - per:parents: 인물의 부모
   - per:siblings: 인물의 형제자매
   - per:other_family: 인물의 기타 가족
   - per:charges: 인물의 혐의나 기소
   - per:alternate_names: 인물의 대체명이나 별칭
   - per:age: 인물의 나이

3. 기타 관계:
   - no_relation: 두 개체 간에 명확한 관계가 없음

분석 지침:

1. 문장의 전체 맥락을 고려하여 두 개체 간의 관계를 분석합니다.
2. 개체의 유형(인물, 조직 등)을 고려하여 적절한 관계를 판단합니다.
3. 관계가 명확하지 않은 경우 "no_relation"을 선택합니다.
4. 가장 구체적이고 정확한 관계 유형을 선택합니다.
5. 관계의 방향성을 고려합니다 (예: A가 B의 직원인 경우 per:employee_of).

출력 형식:
관계 유형의 영어 코드만 출력하세요 (예: per:employee_of, org:product, no_relation).

관계 유형:"""
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
    
    def predict_single(self, sentence: str, subject_entity: Dict, object_entity: Dict) -> Dict[str, Any]:
        """Make a single prediction for RE task."""
        try:
            prompt = self.create_prompt(sentence, subject_entity, object_entity)
            
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
                predicted_relation = self._parse_re_response(response.text)
                return {
                    "success": True,
                    "relation": predicted_relation,
                    "raw_response": response.text
                }
            else:
                logger.error("Cannot get the response text.")
                logger.error("Cannot get the Candidate text.")
                logger.error("Response candidate content has no parts (and thus no text). The candidate is likely blocked by the safety filters.")
                if response:
                    logger.error(f"Content:\n{response.content}")
                    logger.error(f"Candidate:\n{response.candidates[0] if response.candidates else 'No candidates'}")
                    logger.error(f"Response:\n{response}")
                return {
                    "success": False,
                    "relation": "no_relation",
                    "raw_response": "",
                    "error": "No response text"
                }
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "success": False,
                "relation": "no_relation",
                "raw_response": "",
                "error": str(e)
            }
    
    def _parse_re_response(self, response_text: str) -> str:
        """Parse the RE response from the model."""
        # Clean the response text
        response_text = response_text.strip()
        
        # Look for relation type patterns
        # Common patterns: per:xxx, org:xxx, no_relation
        relation_pattern = r'\b(per:[a-z_]+|org:[a-z_]+|no_relation)\b'
        match = re.search(relation_pattern, response_text)
        
        if match:
            return match.group(1)
        
        # If no pattern found, try to extract from the text
        response_lower = response_text.lower()
        
        # Check for common relation indicators
        if "no_relation" in response_lower or "관계 없음" in response_text:
            return "no_relation"
        elif "per:" in response_lower:
            # Extract per: relation
            per_match = re.search(r'per:[a-z_]+', response_lower)
            if per_match:
                return per_match.group(0)
        elif "org:" in response_lower:
            # Extract org: relation
            org_match = re.search(r'org:[a-z_]+', response_lower)
            if org_match:
                return org_match.group(0)
        
        # Default to no_relation if unclear
        return "no_relation"
    
    def calculate_metrics(self, true_relation: str, pred_relation: str) -> Dict[str, Any]:
        """Calculate accuracy and other metrics for RE."""
        is_correct = true_relation == pred_relation
        
        return {
            "accuracy": 1.0 if is_correct else 0.0,
            "correct": 1 if is_correct else 0,
            "total": 1,
            "true_relation": true_relation,
            "predicted_relation": pred_relation
        }
    
    def run_benchmark(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the RE benchmark."""
        logger.info("Starting benchmark...")
        
        start_time = time.time()
        total_samples = len(test_data)
        correct_predictions = 0
        
        # Process each sample
        for i, sample in enumerate(tqdm(test_data, desc="Processing samples")):
            try:
                # Make prediction
                prediction_result = self.predict_single(
                    sample["sentence"], 
                    sample["subject_entity"], 
                    sample["object_entity"]
                )
                
                # Calculate metrics
                metrics = self.calculate_metrics(
                    sample["label"], 
                    prediction_result.get("relation", "no_relation")
                )
                
                # Update counters
                correct_predictions += metrics["correct"]
                
                # Store result
                result = {
                    "id": sample["id"],
                    "sentence": sample["sentence"],
                    "subject_entity": sample["subject_entity"],
                    "object_entity": sample["object_entity"],
                    "true_relation": sample["label"],
                    "predicted_relation": prediction_result.get("relation", "no_relation"),
                    "metrics": metrics,
                    "success": prediction_result["success"],
                    "raw_response": prediction_result.get("raw_response", ""),
                    "error": prediction_result.get("error", "")
                }
                
                self.results.append(result)
                
                # Save intermediate results
                if (i + 1) % self.config.save_interval == 0:
                    self.save_intermediate_results(i + 1, correct_predictions, start_time)
                
                # Sleep between API calls
                time.sleep(self.config.sleep_interval_between_api_calls)
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                self.results.append({
                    "id": sample["id"],
                    "sentence": sample["sentence"],
                    "subject_entity": sample["subject_entity"],
                    "object_entity": sample["object_entity"],
                    "true_relation": sample["label"],
                    "predicted_relation": "no_relation",
                    "metrics": {"accuracy": 0.0, "correct": 0, "total": 1, "true_relation": sample["label"], "predicted_relation": "no_relation"},
                    "success": False,
                    "raw_response": "",
                    "error": str(e)
                })
        
        # Calculate final metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        self.metrics = {
            "total_samples": total_samples,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "total_time": total_time,
            "average_time_per_sample": total_time / total_samples if total_samples > 0 else 0.0,
            "samples_per_second": total_samples / total_time if total_time > 0 else 0.0
        }
        
        logger.info("Benchmark completed!")
        logger.info(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_samples})")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average time per sample: {total_time / total_samples:.3f} seconds")
        
        return self.metrics
    
    def save_results(self):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = os.path.join(self.config.output_dir, f"klue_re_metrics_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Metrics saved to: {metrics_file}")
        
        # Save detailed results
        results_file = os.path.join(self.config.output_dir, f"klue_re_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Save as CSV
        csv_data = []
        for result in self.results:
            csv_data.append({
                "id": result["id"],
                "sentence": result["sentence"],
                "subject_entity_text": result["subject_entity"]["text"],
                "subject_entity_type": result["subject_entity"]["type"],
                "object_entity_text": result["object_entity"]["text"],
                "object_entity_type": result["object_entity"]["type"],
                "true_relation": result["true_relation"],
                "predicted_relation": result["predicted_relation"],
                "accuracy": result["metrics"]["accuracy"],
                "success": result["success"],
                "error": result.get("error", "")
            })
        
        csv_file = os.path.join(self.config.output_dir, f"klue_re_results_{timestamp}.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"Results saved as CSV: {csv_file}")
        
        # Save error analysis
        self.save_error_analysis(timestamp)
    
    def save_intermediate_results(self, current_count: int, correct_count: int, start_time: float):
        """Save intermediate results."""
        if not self.config.save_predictions:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate intermediate metrics
        intermediate_metrics = {
            "samples_processed": current_count,
            "correct_predictions": correct_count,
            "accuracy": correct_count / current_count if current_count > 0 else 0.0,
            "timestamp": timestamp
        }
        
        # Save intermediate metrics
        metrics_file = os.path.join(self.config.output_dir, f"klue_re_metrics_{current_count:06d}_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(intermediate_metrics, f, ensure_ascii=False, indent=2)
        
        # Save intermediate results
        results_file = os.path.join(self.config.output_dir, f"klue_re_results_{current_count:06d}_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Intermediate results saved at {current_count} samples")
    
    def save_error_analysis(self, timestamp: str):
        """Save error analysis for failed predictions."""
        error_samples = [r for r in self.results if not r["success"] or r.get("error") or r["true_relation"] != r["predicted_relation"]]
        
        if not error_samples:
            logger.info("No errors to analyze")
            return
        
        error_file = os.path.join(self.config.output_dir, f"klue_re_error_analysis_{timestamp}.txt")
        
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write("KLUE RE Error Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            for i, sample in enumerate(error_samples[:10], 1):  # Show first 10 errors
                f.write(f"{i}. Sample ID: {sample['id']}\n")
                f.write(f"   Sentence: {sample['sentence']}\n")
                f.write(f"   Subject: {sample['subject_entity']['text']} ({sample['subject_entity']['type']})\n")
                f.write(f"   Object: {sample['object_entity']['text']} ({sample['object_entity']['type']})\n")
                f.write(f"   True Relation: {sample['true_relation']}\n")
                f.write(f"   Predicted Relation: {sample['predicted_relation']}\n")
                if sample.get("error"):
                    f.write(f"   Error: {sample['error']}\n")
                f.write("\n")
        
        logger.info(f"Error analysis saved to: {error_file}")
    
    def print_detailed_metrics(self):
        """Print detailed benchmark results."""
        print("=" * 60)
        print("KLUE Relation Extraction Benchmark Results")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Platform: Google Cloud Vertex AI")
        print(f"Project: {self.config.project_id or os.getenv('GOOGLE_CLOUD_PROJECT')}")
        print(f"Location: {self.config.location}")
        print(f"Accuracy: {self.metrics['accuracy']:.4f} ({self.metrics['correct_predictions']}/{self.metrics['total_samples']})")
        print(f"Total Time: {self.metrics['total_time']:.2f} seconds")
        print(f"Average Time per Sample: {self.metrics['average_time_per_sample']:.3f} seconds")
        print(f"Samples per Second: {self.metrics['samples_per_second']:.2f}")
        print()
        
        # Per-relation type analysis
        relation_metrics = {}
        for result in self.results:
            relation_type = result["true_relation"]
            if relation_type not in relation_metrics:
                relation_metrics[relation_type] = {"total": 0, "correct": 0}
            relation_metrics[relation_type]["total"] += 1
            
            if result["true_relation"] == result["predicted_relation"]:
                relation_metrics[relation_type]["correct"] += 1
        
        print("Per-relation Type Performance:")
        for relation_type, metrics in relation_metrics.items():
            accuracy = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0.0
            relation_name = self.RELATION_TYPES.get(relation_type, relation_type)
            print(f"  {relation_type} ({relation_name}): {accuracy:.4f} ({metrics['correct']}/{metrics['total']})")
        
        print()
        
        # Error analysis
        error_count = sum(1 for r in self.results if not r["success"] or r.get("error") or r["true_relation"] != r["predicted_relation"])
        if error_count > 0:
            print(f"Error Analysis (showing first 5 errors):")
            error_samples = [r for r in self.results if not r["success"] or r.get("error") or r["true_relation"] != r["predicted_relation"]]
            for i, sample in enumerate(error_samples[:5], 1):
                print(f"  {i}. Sample ID: {sample['id']}")
                print(f"     Sentence: {sample['sentence'][:100]}...")
                print(f"     Subject: {sample['subject_entity']['text']} ({sample['subject_entity']['type']})")
                print(f"     Object: {sample['object_entity']['text']} ({sample['object_entity']['type']})")
                print(f"     True: {sample['true_relation']} | Predicted: {sample['predicted_relation']}")
                if sample.get("error"):
                    print(f"     Error: {sample['error']}")
                print()

def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="KLUE RE Benchmark with Gemini 2.5 Flash")
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
    benchmark = KLUERelationExtractionBenchmark(config)
    
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