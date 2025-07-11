#!/usr/bin/env python3
"""
KLUE Dialogue State Tracking (DST) Benchmark with Gemini 2.5 Flash on Vertex AI
This script benchmarks Gemini 2.5 Flash on the Korean Language Understanding Evaluation (KLUE) Dialogue State Tracking task using Google Cloud Vertex AI.
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
    max_tokens: int = 2048  # Increased for DST task
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

class KLUEDialogueStateTrackingBenchmark:
    """Benchmark class for KLUE Dialogue State Tracking task using Vertex AI."""
    
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
        Load the KLUE DST dataset, convert it to a list of dictionaries,
        and efficiently limit the number of samples based on the configuration.
        """
        try:
            logger.info("Loading KLUE DST dataset for dialogue state tracking...")
            
            # Load the validation split from the Hugging Face Hub.
            validation_dataset = load_dataset('klue', 'dst', split='validation')

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
                    
                # Process DST data
                processed_data.append({
                    "id": item["guid"],
                    "dialogue_id": item.get("dialogue_id", ""),
                    "turn_id": item.get("turn_id", 0),
                    "dialogue": item["dialogue"],
                    "domains": item.get("domains", []),
                    "state": item.get("state", {}),
                    "active_intent": item.get("active_intent", ""),
                    "requested_slots": item.get("requested_slots", []),
                    "slot_values": item.get("slot_values", {})
                })

            logger.info(f"✅ Successfully loaded {len(processed_data)} samples.")
            return processed_data
            
        except KeyError as e:
            logger.error(f"❌ A key was not found in the dataset item: {e}. The dataset schema may have changed.")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load or process the dataset: {e}")
            raise
    
    def create_prompt(self, dialogue: List[Dict], domains: List[str], turn_id: int) -> str:
        """Create detailed prompt for dialogue state tracking."""
        prompt = f"""역할: 당신은 한국어 대화에서 사용자의 의도와 상태를 정확하게 추적하는 "전문 대화 상태 추적 AI"입니다.

임무: 주어진 대화를 분석하여 현재 턴에서 사용자의 의도(intent), 요청된 슬롯(requested slots), 그리고 슬롯 값(slot values)을 정확히 파악해주세요.

대화 상태 추적의 핵심 요소:

1. **도메인(Domain)**: 대화의 주제 영역 (예: 레스토랑, 호텔, 영화, 음악 등)
2. **의도(Intent)**: 사용자가 원하는 행동이나 목표
3. **요청된 슬롯(Requested Slots)**: 사용자가 정보를 요청한 항목들
4. **슬롯 값(Slot Values)**: 사용자가 제공하거나 시스템이 추론한 구체적인 값들

일반적인 의도 유형:
- inform: 정보 제공
- request: 정보 요청
- confirm: 확인 요청
- deny: 부정/거부
- affirm: 긍정/동의
- book: 예약
- search: 검색
- recommend: 추천 요청

일반적인 슬롯 유형:
- location: 위치
- time: 시간
- date: 날짜
- price: 가격
- rating: 평점
- cuisine: 요리 종류
- name: 이름
- phone: 전화번호
- address: 주소
- capacity: 수용 인원
- duration: 기간
- genre: 장르
- artist: 아티스트
- title: 제목

지침:

1. **대화 맥락 이해**: 전체 대화 흐름을 파악하여 현재 상황을 이해하세요.
2. **의도 파악**: 사용자가 무엇을 원하는지 정확히 파악하세요.
3. **슬롯 추출**: 사용자가 언급한 정보나 요청한 정보를 정확히 식별하세요.
4. **값 추출**: 구체적인 값(시간, 장소, 이름 등)을 정확히 추출하세요.
5. **일관성 유지**: 이전 턴의 정보와 일관성을 유지하세요.

대화 도메인: {', '.join(domains) if domains else '일반'}
현재 턴: {turn_id}

대화 내용:
"""

        # Add dialogue turns
        for i, turn in enumerate(dialogue):
            speaker = turn.get("speaker", "unknown")
            utterance = turn.get("utterance", "")
            prompt += f"{i+1}. {speaker}: {utterance}\n"
        
        prompt += f"""

현재 턴({turn_id})에서 다음을 분석해주세요:

1. 활성 의도(Active Intent): 사용자의 현재 의도
2. 요청된 슬롯(Requested Slots): 사용자가 정보를 요청한 슬롯들
3. 슬롯 값(Slot Values): 사용자가 제공하거나 추론할 수 있는 슬롯 값들

답변 형식:
활성 의도: [의도]
요청된 슬롯: [슬롯1, 슬롯2, ...]
슬롯 값: {{"슬롯명": "값", "슬롯명2": "값2", ...}}

답변:"""

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
    
    def predict_single(self, dialogue: List[Dict], domains: List[str], turn_id: int) -> Dict[str, Any]:
        """Make a single prediction for DST task."""
        try:
            prompt = self.create_prompt(dialogue, domains, turn_id)
            
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
                predicted_response = response.text.strip()
                return {
                    "success": True,
                    "response": predicted_response,
                    "raw_response": response.text
                }
            else:
                logger.error("Cannot get the response text.")
                return {
                    "success": False,
                    "response": "",
                    "raw_response": "",
                    "error": "No response text"
                }
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "success": False,
                "response": "",
                "raw_response": "",
                "error": str(e)
            }
    
    def parse_dst_response(self, response: str) -> Dict[str, Any]:
        """Parse the DST response to extract intent, requested slots, and slot values."""
        try:
            parsed = {
                "active_intent": "",
                "requested_slots": [],
                "slot_values": {}
            }
            
            # Extract active intent
            intent_match = re.search(r'활성\s*의도\s*:\s*(.+)', response, re.IGNORECASE)
            if intent_match:
                parsed["active_intent"] = intent_match.group(1).strip()
            
            # Extract requested slots
            slots_match = re.search(r'요청된\s*슬롯\s*:\s*\[(.+)\]', response, re.IGNORECASE)
            if slots_match:
                slots_str = slots_match.group(1).strip()
                if slots_str:
                    parsed["requested_slots"] = [slot.strip() for slot in slots_str.split(',') if slot.strip()]
            
            # Extract slot values
            values_match = re.search(r'슬롯\s*값\s*:\s*\{([^}]+)\}', response, re.IGNORECASE)
            if values_match:
                values_str = values_match.group(1).strip()
                # Simple parsing for slot values
                slot_pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', values_str)
                for slot, value in slot_pairs:
                    parsed["slot_values"][slot] = value
            
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse DST response: {e}")
            return {
                "active_intent": "",
                "requested_slots": [],
                "slot_values": {}
            }
    
    def calculate_metrics(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate DST metrics."""
        metrics = {}
        
        # Intent accuracy
        predicted_intent = predicted.get("active_intent", "")
        ground_truth_intent = ground_truth.get("active_intent", "")
        metrics["intent_accuracy"] = 1.0 if predicted_intent == ground_truth_intent else 0.0
        
        # Requested slots F1
        predicted_slots = set(predicted.get("requested_slots", []))
        ground_truth_slots = set(ground_truth.get("requested_slots", []))
        
        if predicted_slots or ground_truth_slots:
            precision = len(predicted_slots & ground_truth_slots) / len(predicted_slots) if predicted_slots else 0.0
            recall = len(predicted_slots & ground_truth_slots) / len(ground_truth_slots) if ground_truth_slots else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics["requested_slots_f1"] = f1
            metrics["requested_slots_precision"] = precision
            metrics["requested_slots_recall"] = recall
        else:
            metrics["requested_slots_f1"] = 1.0
            metrics["requested_slots_precision"] = 1.0
            metrics["requested_slots_recall"] = 1.0
        
        # Slot values F1
        predicted_values = predicted.get("slot_values", {})
        ground_truth_values = ground_truth.get("slot_values", {})
        
        all_slots = set(predicted_values.keys()) | set(ground_truth_values.keys())
        correct_slots = 0
        predicted_slot_count = len(predicted_values)
        ground_truth_slot_count = len(ground_truth_values)
        
        for slot in all_slots:
            if predicted_values.get(slot) == ground_truth_values.get(slot):
                correct_slots += 1
        
        if predicted_slot_count or ground_truth_slot_count:
            precision = correct_slots / predicted_slot_count if predicted_slot_count else 0.0
            recall = correct_slots / ground_truth_slot_count if ground_truth_slot_count else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics["slot_values_f1"] = f1
            metrics["slot_values_precision"] = precision
            metrics["slot_values_recall"] = recall
        else:
            metrics["slot_values_f1"] = 1.0
            metrics["slot_values_precision"] = 1.0
            metrics["slot_values_recall"] = 1.0
        
        # Overall F1 (average of all F1 scores)
        f1_scores = [metrics["requested_slots_f1"], metrics["slot_values_f1"]]
        metrics["overall_f1"] = sum(f1_scores) / len(f1_scores)
        
        return metrics
    
    def run_benchmark(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the DST benchmark."""
        logger.info("Starting benchmark...")
        
        start_time = time.time()
        total_samples = len(test_data)
        total_intent_accuracy = 0
        total_requested_slots_f1 = 0
        total_slot_values_f1 = 0
        total_overall_f1 = 0
        
        # Process each sample
        for i, sample in enumerate(tqdm(test_data, desc="Processing samples")):
            try:
                # Make prediction
                prediction_result = self.predict_single(
                    sample["dialogue"],
                    sample["domains"],
                    sample["turn_id"]
                )
                
                # Parse the response
                parsed_prediction = self.parse_dst_response(prediction_result.get("response", ""))
                
                # Calculate metrics
                metrics = self.calculate_metrics(parsed_prediction, {
                    "active_intent": sample["active_intent"],
                    "requested_slots": sample["requested_slots"],
                    "slot_values": sample["slot_values"]
                })
                
                # Update counters
                total_intent_accuracy += metrics["intent_accuracy"]
                total_requested_slots_f1 += metrics["requested_slots_f1"]
                total_slot_values_f1 += metrics["slot_values_f1"]
                total_overall_f1 += metrics["overall_f1"]
                
                # Store result
                result = {
                    "id": sample["id"],
                    "dialogue_id": sample["dialogue_id"],
                    "turn_id": sample["turn_id"],
                    "domains": sample["domains"],
                    "dialogue": sample["dialogue"],
                    "ground_truth": {
                        "active_intent": sample["active_intent"],
                        "requested_slots": sample["requested_slots"],
                        "slot_values": sample["slot_values"]
                    },
                    "predicted": parsed_prediction,
                    "metrics": metrics,
                    "success": prediction_result["success"],
                    "raw_response": prediction_result.get("raw_response", ""),
                    "error": prediction_result.get("error", "")
                }
                
                self.results.append(result)
                
                # Save intermediate results
                if (i + 1) % self.config.save_interval == 0:
                    self.save_intermediate_results(i + 1, total_intent_accuracy, total_requested_slots_f1, total_slot_values_f1, total_overall_f1, start_time)
                
                # Sleep between API calls
                time.sleep(self.config.sleep_interval_between_api_calls)
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                self.results.append({
                    "id": sample["id"],
                    "dialogue_id": sample["dialogue_id"],
                    "turn_id": sample["turn_id"],
                    "domains": sample["domains"],
                    "dialogue": sample["dialogue"],
                    "ground_truth": {
                        "active_intent": sample["active_intent"],
                        "requested_slots": sample["requested_slots"],
                        "slot_values": sample["slot_values"]
                    },
                    "predicted": {"active_intent": "", "requested_slots": [], "slot_values": {}},
                    "metrics": {"intent_accuracy": 0.0, "requested_slots_f1": 0.0, "slot_values_f1": 0.0, "overall_f1": 0.0},
                    "success": False,
                    "raw_response": "",
                    "error": str(e)
                })
        
        # Calculate final metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        intent_accuracy = total_intent_accuracy / total_samples if total_samples > 0 else 0.0
        requested_slots_f1 = total_requested_slots_f1 / total_samples if total_samples > 0 else 0.0
        slot_values_f1 = total_slot_values_f1 / total_samples if total_samples > 0 else 0.0
        overall_f1 = total_overall_f1 / total_samples if total_samples > 0 else 0.0
        
        self.metrics = {
            "total_samples": total_samples,
            "intent_accuracy": intent_accuracy,
            "requested_slots_f1": requested_slots_f1,
            "slot_values_f1": slot_values_f1,
            "overall_f1": overall_f1,
            "total_time": total_time,
            "average_time_per_sample": total_time / total_samples if total_samples > 0 else 0.0,
            "samples_per_second": total_samples / total_time if total_time > 0 else 0.0
        }
        
        logger.info("Benchmark completed!")
        logger.info(f"Intent Accuracy: {intent_accuracy:.4f}")
        logger.info(f"Requested Slots F1: {requested_slots_f1:.4f}")
        logger.info(f"Slot Values F1: {slot_values_f1:.4f}")
        logger.info(f"Overall F1: {overall_f1:.4f}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average time per sample: {total_time / total_samples:.3f} seconds")
        
        return self.metrics
    
    def save_results(self):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = os.path.join(self.config.output_dir, f"klue_dst_metrics_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Metrics saved to: {metrics_file}")
        
        # Save detailed results
        results_file = os.path.join(self.config.output_dir, f"klue_dst_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Save as CSV
        csv_data = []
        for result in self.results:
            csv_data.append({
                "id": result["id"],
                "dialogue_id": result["dialogue_id"],
                "turn_id": result["turn_id"],
                "domains": ", ".join(result["domains"]),
                "ground_truth_intent": result["ground_truth"]["active_intent"],
                "predicted_intent": result["predicted"]["active_intent"],
                "ground_truth_requested_slots": ", ".join(result["ground_truth"]["requested_slots"]),
                "predicted_requested_slots": ", ".join(result["predicted"]["requested_slots"]),
                "ground_truth_slot_values": str(result["ground_truth"]["slot_values"]),
                "predicted_slot_values": str(result["predicted"]["slot_values"]),
                "intent_accuracy": result["metrics"]["intent_accuracy"],
                "requested_slots_f1": result["metrics"]["requested_slots_f1"],
                "slot_values_f1": result["metrics"]["slot_values_f1"],
                "overall_f1": result["metrics"]["overall_f1"],
                "success": result["success"],
                "error": result.get("error", "")
            })
        
        csv_file = os.path.join(self.config.output_dir, f"klue_dst_results_{timestamp}.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"Results saved as CSV: {csv_file}")
        
        # Save error analysis
        self.save_error_analysis(timestamp)
    
    def save_intermediate_results(self, current_count: int, intent_accuracy_sum: float, requested_slots_f1_sum: float, slot_values_f1_sum: float, overall_f1_sum: float, start_time: float):
        """Save intermediate results."""
        if not self.config.save_predictions:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate intermediate metrics
        intent_accuracy = intent_accuracy_sum / current_count if current_count > 0 else 0.0
        requested_slots_f1 = requested_slots_f1_sum / current_count if current_count > 0 else 0.0
        slot_values_f1 = slot_values_f1_sum / current_count if current_count > 0 else 0.0
        overall_f1 = overall_f1_sum / current_count if current_count > 0 else 0.0
        
        intermediate_metrics = {
            "samples_processed": current_count,
            "intent_accuracy": intent_accuracy,
            "requested_slots_f1": requested_slots_f1,
            "slot_values_f1": slot_values_f1,
            "overall_f1": overall_f1,
            "timestamp": timestamp
        }
        
        # Save intermediate metrics
        metrics_file = os.path.join(self.config.output_dir, f"klue_dst_metrics_{current_count:06d}_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(intermediate_metrics, f, ensure_ascii=False, indent=2)
        
        # Save intermediate results
        results_file = os.path.join(self.config.output_dir, f"klue_dst_results_{current_count:06d}_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Intermediate results saved at {current_count} samples")
    
    def save_error_analysis(self, timestamp: str):
        """Save error analysis for failed predictions."""
        error_samples = [r for r in self.results if not r["success"] or r.get("error") or r["metrics"]["overall_f1"] < 0.5]
        
        if not error_samples:
            logger.info("No errors to analyze")
            return
        
        error_file = os.path.join(self.config.output_dir, f"klue_dst_error_analysis_{timestamp}.txt")
        
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write("KLUE DST Error Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            for i, sample in enumerate(error_samples[:10], 1):  # Show first 10 errors
                f.write(f"{i}. Sample ID: {sample['id']}\n")
                f.write(f"   Dialogue ID: {sample['dialogue_id']}\n")
                f.write(f"   Turn ID: {sample['turn_id']}\n")
                f.write(f"   Domains: {sample['domains']}\n")
                f.write(f"   Ground Truth Intent: {sample['ground_truth']['active_intent']}\n")
                f.write(f"   Predicted Intent: {sample['predicted']['active_intent']}\n")
                f.write(f"   Ground Truth Requested Slots: {sample['ground_truth']['requested_slots']}\n")
                f.write(f"   Predicted Requested Slots: {sample['predicted']['requested_slots']}\n")
                f.write(f"   Ground Truth Slot Values: {sample['ground_truth']['slot_values']}\n")
                f.write(f"   Predicted Slot Values: {sample['predicted']['slot_values']}\n")
                f.write(f"   Intent Accuracy: {sample['metrics']['intent_accuracy']:.4f}\n")
                f.write(f"   Requested Slots F1: {sample['metrics']['requested_slots_f1']:.4f}\n")
                f.write(f"   Slot Values F1: {sample['metrics']['slot_values_f1']:.4f}\n")
                f.write(f"   Overall F1: {sample['metrics']['overall_f1']:.4f}\n")
                if sample.get("error"):
                    f.write(f"   Error: {sample['error']}\n")
                f.write("\n")
        
        logger.info(f"Error analysis saved to: {error_file}")
    
    def print_detailed_metrics(self):
        """Print detailed benchmark results."""
        print("=" * 60)
        print("KLUE Dialogue State Tracking Benchmark Results")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Platform: Google Cloud Vertex AI")
        print(f"Project: {self.config.project_id or os.getenv('GOOGLE_CLOUD_PROJECT')}")
        print(f"Location: {self.config.location}")
        print(f"Intent Accuracy: {self.metrics['intent_accuracy']:.4f}")
        print(f"Requested Slots F1: {self.metrics['requested_slots_f1']:.4f}")
        print(f"Slot Values F1: {self.metrics['slot_values_f1']:.4f}")
        print(f"Overall F1: {self.metrics['overall_f1']:.4f}")
        print(f"Total Samples: {self.metrics['total_samples']}")
        print(f"Total Time: {self.metrics['total_time']:.2f} seconds")
        print(f"Average Time per Sample: {self.metrics['average_time_per_sample']:.3f} seconds")
        print(f"Samples per Second: {self.metrics['samples_per_second']:.2f}")
        print()
        
        # Per-domain analysis
        domain_results = {}
        for result in self.results:
            for domain in result["domains"]:
                if domain not in domain_results:
                    domain_results[domain] = {"count": 0, "f1_sum": 0}
                domain_results[domain]["count"] += 1
                domain_results[domain]["f1_sum"] += result["metrics"]["overall_f1"]
        
        if domain_results:
            print("Per-Domain Performance:")
            for domain, stats in domain_results.items():
                avg_f1 = stats["f1_sum"] / stats["count"]
                print(f"  {domain}: F1 = {avg_f1:.4f} (n={stats['count']})")
            print()
        
        # Error analysis
        error_count = sum(1 for r in self.results if not r["success"] or r.get("error") or r["metrics"]["overall_f1"] < 0.5)
        if error_count > 0:
            print(f"Error Analysis (showing first 5 errors):")
            error_samples = [r for r in self.results if not r["success"] or r.get("error") or r["metrics"]["overall_f1"] < 0.5]
            for i, sample in enumerate(error_samples[:5], 1):
                print(f"  {i}. Sample ID: {sample['id']}")
                print(f"     Turn ID: {sample['turn_id']}")
                print(f"     Ground Truth Intent: {sample['ground_truth']['active_intent']}")
                print(f"     Predicted Intent: {sample['predicted']['active_intent']}")
                print(f"     Overall F1: {sample['metrics']['overall_f1']:.4f}")
                if sample.get("error"):
                    print(f"     Error: {sample['error']}")
                print()

def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="KLUE DST Benchmark with Gemini 2.5 Flash")
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
    benchmark = KLUEDialogueStateTrackingBenchmark(config)
    
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