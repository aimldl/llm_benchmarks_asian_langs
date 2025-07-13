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
import sys
import io
import contextlib
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
    verbose: bool = False  # Add verbose mode for detailed logging

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
        "per:age": "인물:나이"
    }
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = None
        self.results = []
        self.metrics = {}
        
        # Configure logging based on verbose mode
        if not config.verbose:
            # Suppress Google Cloud client logging in clean mode
            logging.getLogger('google.cloud').setLevel(logging.WARNING)
            logging.getLogger('google.auth').setLevel(logging.WARNING)
            logging.getLogger('google.api_core').setLevel(logging.WARNING)
            logging.getLogger('google.genai').setLevel(logging.WARNING)
        
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
        """Create a prompt for the RE task."""
        prompt = f"""Analyze the relationship between two entities in the given Korean sentence.

Sentence: {sentence}

Subject entity: {subject_entity['text']} ({subject_entity['type']})
Object entity: {object_entity['text']} ({object_entity['type']})

Choose the relationship type from the following options:

Relationship types:
- no_relation: no relationship
- per:employee_of: person is employee of organization
- per:member_of: person is member of organization
- per:works_for: person works for organization
- per:title: person's title
- per:schools_attended: schools person attended
- per:countries_of_residence: person's country of residence
- per:stateorprovinces_of_residence: person's state/province of residence
- per:cities_of_residence: person's city of residence
- per:countries_of_birth: person's country of birth
- per:stateorprovinces_of_birth: person's state/province of birth
- per:cities_of_birth: person's city of birth
- per:date_of_birth: person's date of birth
- per:date_of_death: person's date of death
- per:place_of_birth: person's place of birth
- per:place_of_death: person's place of death
- per:cause_of_death: person's cause of death
- per:origin: person's origin
- per:religion: person's religion
- per:spouse: person's spouse
- per:children: person's children
- per:parents: person's parents
- per:siblings: person's siblings
- per:other_family: person's other family
- per:charges: person's charges
- per:alternate_names: person's alternate names
- per:age: person's age
- org:top_members/employees: organization's top members/employees
- org:members: organization's members
- org:product: organization's product
- org:founded: organization's founding
- org:alternate_names: organization's alternate names
- org:place_of_headquarters: organization's headquarters location
- org:number_of_employees/members: organization's number of employees/members
- org:website: organization's website
- org:subsidiaries: organization's subsidiaries
- org:parents: organization's parent organization
- org:dissolved: organization's dissolution

Output format:
Respond with only the relationship type code (e.g., per:employee_of, org:product, no_relation).

Relationship:"""
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
            
            # Suppress Google Cloud logging during API call if not verbose
            if not self.config.verbose:
                # Redirect stdout and stderr to suppress Google Cloud logging
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
            else:
                # Generate content with full logging
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
            
            # Extract response text - improved handling
            if response and hasattr(response, 'text') and response.text:
                predicted_relation = self._parse_re_response(response.text)
                return {
                    "success": True,
                    "relation": predicted_relation,
                    "raw_response": response.text
                }
            elif response and hasattr(response, 'candidates') and response.candidates:
                # Try to get text from candidates
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        text = candidate.content.parts[0].text
                        predicted_relation = self._parse_re_response(text)
                        return {
                            "success": True,
                            "relation": predicted_relation,
                            "raw_response": text
                        }
            
            # If we get here, the response was blocked or empty
            logger.error("Cannot get the response text.")
            logger.error("Response candidate content has no parts (and thus no text). The candidate is likely blocked by the safety filters.")
            if response and hasattr(response, 'candidates') and response.candidates:
                logger.error(f"Candidate:\n{response.candidates[0] if response.candidates else 'No candidates'}")
            return {
                "success": False,
                "relation": "no_relation",
                "raw_response": "",
                "error": "No response text - likely blocked by safety filters"
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
        
        # Process each sample with reduced progress bar updates (every 4th sample)
        for i, sample in enumerate(tqdm(test_data, desc="Processing samples", mininterval=1.0, maxinterval=5.0)):
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
        """Save intermediate results during benchmark execution."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate intermediate metrics
        accuracy = correct_count / current_count if current_count > 0 else 0.0
        elapsed_time = time.time() - start_time
        
        intermediate_metrics = {
            "samples_processed": current_count,
            "correct_predictions": correct_count,
            "accuracy": accuracy,
            "elapsed_time": elapsed_time,
            "average_time_per_sample": elapsed_time / current_count if current_count > 0 else 0.0
        }
        
        # Save intermediate metrics
        metrics_file = os.path.join(self.config.output_dir, f"klue_re_metrics_{current_count:06d}_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(intermediate_metrics, f, ensure_ascii=False, indent=2)
        
        # Save intermediate results as CSV
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
        
        csv_file = os.path.join(self.config.output_dir, f"klue_re_results_{current_count:06d}_{timestamp}.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.info(f"Intermediate results saved: {metrics_file} and {csv_file}")
    
    def save_error_analysis(self, timestamp: str):
        """Save detailed error analysis."""
        error_file = os.path.join(self.config.output_dir, f"klue_re_error_analysis_{timestamp}.txt")
        
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write("KLUE RE Error Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            # Count errors by type
            error_counts = {}
            failed_predictions = 0
            
            for result in self.results:
                if not result["success"]:
                    failed_predictions += 1
                    error_type = result.get("error", "Unknown error")
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            f.write(f"Total samples: {len(self.results)}\n")
            f.write(f"Failed predictions: {failed_predictions}\n")
            f.write(f"Success rate: {(len(self.results) - failed_predictions) / len(self.results) * 100:.2f}%\n\n")
            
            f.write("Error breakdown:\n")
            for error_type, count in error_counts.items():
                f.write(f"  {error_type}: {count}\n")
            
            f.write("\nDetailed error analysis (showing first 10 errors):\n")
            f.write("-" * 50 + "\n")
            
            error_count = 0
            for result in self.results:
                if not result["success"] and error_count < 10:
                    f.write(f"\n{error_count + 1}. Sample ID: {result['id']}\n")
                    f.write(f"   Sentence: {result['sentence'][:100]}...\n")
                    f.write(f"   Subject: {result['subject_entity']['text']} ({result['subject_entity']['type']})\n")
                    f.write(f"   Object: {result['object_entity']['text']} ({result['object_entity']['type']})\n")
                    f.write(f"   True: {result['true_relation']} | Predicted: {result['predicted_relation']}\n")
                    f.write(f"   Error: {result.get('error', 'Unknown error')}\n")
                    error_count += 1
        
        logger.info(f"Error analysis saved to: {error_file}")
    
    def print_detailed_metrics(self):
        """Print detailed metrics and results."""
        print("=" * 60)
        print("KLUE Relation Extraction Benchmark Results")
        print("=" * 60)
        print(f"Model: {self.config.model_name}")
        print(f"Platform: Google Cloud Vertex AI")
        print(f"Project: {self.config.project_id or os.getenv('GOOGLE_CLOUD_PROJECT')}")
        print(f"Location: {self.config.location}")
        print(f"Accuracy: {self.metrics['accuracy']:.4f} ({self.metrics['correct_predictions']}/{self.metrics['total_samples']})")
        print(f"Total Time: {self.metrics['total_time']:.2f} seconds")
        print(f"Average Time per Sample: {self.metrics['average_time_per_sample']:.3f} seconds")
        print(f"Samples per Second: {self.metrics['samples_per_second']:.2f}")
        
        # Per-relation type performance
        print("\nPer-relation Type Performance:")
        relation_performance = {}
        for result in self.results:
            true_rel = result["true_relation"]
            if true_rel not in relation_performance:
                relation_performance[true_rel] = {"correct": 0, "total": 0}
            relation_performance[true_rel]["total"] += 1
            if result["metrics"]["correct"]:
                relation_performance[true_rel]["correct"] += 1
        
        for relation, stats in relation_performance.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            print(f"  {relation} ({relation}): {accuracy:.4f} ({stats['correct']}/{stats['total']})")
        
        # Error analysis summary
        failed_count = sum(1 for r in self.results if not r["success"])
        if failed_count > 0:
            print(f"\nFailed predictions: {failed_count}/{len(self.results)} ({failed_count/len(self.results)*100:.1f}%)")
        
        print("\nError Analysis (showing first 5 errors):")
        error_count = 0
        for result in self.results:
            if not result["success"] and error_count < 5:
                print(f"  {error_count + 1}. Sample ID: {result['id']}")
                print(f"     Sentence: {result['sentence'][:100]}...")
                print(f"     Subject: {result['subject_entity']['text']} ({result['subject_entity']['type']})")
                print(f"     Object: {result['object_entity']['text']} ({result['object_entity']['type']})")
                print(f"     True: {result['true_relation']} | Predicted: {result['predicted_relation']}")
                print(f"     Error: {result.get('error', 'Unknown error')}")
                print()
                error_count += 1

def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="KLUE RE Benchmark with Gemini 2.5 Flash")
    parser.add_argument("--project-id", type=str, help="Google Cloud project ID")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to test")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Create configuration
    config = BenchmarkConfig(
        max_samples=args.max_samples,
        project_id=args.project_id,
        verbose=args.verbose
    )
    
    try:
        # Initialize benchmark
        benchmark = KLUERelationExtractionBenchmark(config)
        
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
        sys.exit(1)

if __name__ == "__main__":
    main() 