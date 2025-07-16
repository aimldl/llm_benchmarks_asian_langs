#!/usr/bin/env python3
"""
KLUE Named Entity Recognition (NER) Benchmark with Gemini 2.5 Flash on Vertex AI
This script benchmarks Gemini 2.5 Flash on the Korean Language Understanding Evaluation (KLUE) Named Entity Recognition task using Google Cloud Vertex AI.
"""

import os
import json
import time
import argparse
import re
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
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
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduce verbosity of Google Cloud client logging
logging.getLogger('google.cloud.aiplatform').setLevel(logging.ERROR)
logging.getLogger('google.auth').setLevel(logging.ERROR)
logging.getLogger('google.api_core').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('google.genai').setLevel(logging.ERROR)
logging.getLogger('google.cloud').setLevel(logging.ERROR)

@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""
    model_name: str = "gemini-2.5-flash"
    sleep_interval_between_api_calls: float = 0.04 # sec
    max_tokens: int = 2048  # Increased for NER task
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
    verbose: bool = False  # Control logging verbosity

class KLUENamedEntityRecognitionBenchmark:
    """Benchmark class for KLUE Named Entity Recognition task using Vertex AI."""
    
    # KLUE NER entity types
    ENTITY_TYPES = {
        "PS": "인물(Person)",
        "LC": "지명(Location)",
        "OG": "기관(Organization)",
        "DT": "날짜(Date)",
        "TI": "시간(Time)",
        "QT": "수량(Quantity)"
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
        Load the KLUE NER dataset, convert it to a list of dictionaries,
        and efficiently limit the number of samples based on the configuration.
        """
        try:
            logger.info("Loading KLUE NER dataset for named entity recognition...")
            
            # Load the validation split from the Hugging Face Hub.
            validation_dataset = load_dataset('klue', 'ner', split='validation')

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
                    
                # Process NER data
                processed_data.append({
                    "id": f"ner-val_{len(processed_data):06d}",  # Generate unique ID since guid is not available
                    "tokens": item["tokens"],
                    "ner_tags": item["ner_tags"],
                    "text": " ".join(item["tokens"]),
                    "entities": self._extract_entities_from_tags(item["tokens"], item["ner_tags"])
                })

            logger.info(f"✅ Successfully loaded {len(processed_data)} samples.")
            return processed_data
            
        except KeyError as e:
            logger.error(f"❌ A key was not found in the dataset item: {e}. The dataset schema may have changed.")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load or process the dataset: {e}")
            raise
    
    def _extract_entities_from_tags(self, tokens: List[str], ner_tags: List[int]) -> List[Dict[str, Any]]:
        """Extract entities from BIO tags (integer format)."""
        # Define the label mapping based on the dataset features
        label_names = ['B-DT', 'I-DT', 'B-LC', 'I-LC', 'B-OG', 'I-OG', 'B-PS', 'I-PS', 'B-QT', 'I-QT', 'B-TI', 'I-TI', 'O']
        
        entities = []
        current_entity = None
        
        for i, (token, tag_idx) in enumerate(zip(tokens, ner_tags)):
            # Convert integer tag to string label
            if tag_idx < len(label_names):
                tag = label_names[tag_idx]
            else:
                tag = 'O'  # Default to 'O' if index is out of range
            
            if tag.startswith('B-'):  # Beginning of entity
                if current_entity:
                    entities.append(current_entity)
                entity_type = tag[2:]  # Remove 'B-' prefix
                current_entity = {
                    'type': entity_type,
                    'text': token,
                    'start': i,
                    'end': i
                }
            elif tag.startswith('I-') and current_entity:  # Inside entity
                entity_type = tag[2:]  # Remove 'I-' prefix
                if entity_type == current_entity['type']:
                    current_entity['text'] += ' ' + token
                    current_entity['end'] = i
            else:  # 'O' tag or mismatched I- tag
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Don't forget the last entity
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def create_prompt(self, text: str) -> str:
        """Create detailed prompt for named entity recognition."""
        prompt = f"""역할: 당신은 한국어 텍스트에서 명명된 개체(Named Entity)를 정확하게 식별하고 분류하는 "전문 개체명 인식 AI"입니다.

임무: 아래에 제시된 한국어 텍스트에서 명명된 개체들을 찾아서 지정된 형식으로 출력해 주세요.

개체 유형 정의:

1. 인물(PS: Person): 사람의 이름, 별명, 호칭 등
   - 예시: "김철수", "박영희", "이사장", "대통령", "교수님"

2. 지명(LC: Location): 장소, 지역, 국가, 도시, 건물명 등
   - 예시: "서울", "부산", "한국", "미국", "강남구", "롯데월드타워"

3. 기관(OG: Organization): 회사, 학교, 정부기관, 단체, 협회 등
   - 예시: "삼성전자", "서울대학교", "국회", "정부", "한국은행"

4. 날짜(DT: Date): 년, 월, 일, 요일, 기념일 등
   - 예시: "2024년", "3월 15일", "월요일", "크리스마스", "설날"

5. 시간(TI: Time): 시, 분, 초, 오전/오후, 시간대 등
   - 예시: "오후 3시", "30분", "새벽", "저녁", "한 시간"

6. 수량(QT: Quantity): 숫자, 단위, 금액, 비율, 개수 등
   - 예시: "100개", "5천만원", "50%", "3배", "1킬로그램"

출력 형식:
각 개체를 다음 형식으로 출력하세요:
[개체텍스트:개체유형]

주의사항:
1. 개체의 경계를 정확히 파악하여 전체 개체명을 포함하세요.
2. 동일한 개체가 여러 번 나타나면 모두 찾아주세요.
3. 개체 유형은 정확히 일치해야 합니다 (PS, LC, OG, DT, TI, QT).
4. 개체가 없는 경우 "개체 없음"이라고 출력하세요.
5. 개체를 찾을 때는 문맥을 고려하여 정확한 유형을 판단하세요.

텍스트: {text}

개체명 인식 결과:"""
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
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """Make a single prediction for NER task."""
        try:
            prompt = self.create_prompt(text)
            
            # Configure safety settings
            safety_settings = self.configure_safety_settings()
            
            # Generate content with optional logging suppression
            if not self.config.verbose:
                # In default mode, suppress all output from the API call
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
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
                # In verbose mode, allow all output
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
                predicted_entities = self._parse_ner_response(response.text)
                return {
                    "success": True,
                    "entities": predicted_entities,
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
                    "entities": [],
                    "raw_response": "",
                    "error": "No response text"
                }
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "success": False,
                "entities": [],
                "raw_response": "",
                "error": str(e)
            }
    
    def _parse_ner_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse the NER response from the model."""
        entities = []
        
        # Look for patterns like [텍스트:유형]
        pattern = r'\[([^:]+):([^\]]+)\]'
        matches = re.findall(pattern, response_text)
        
        for text, entity_type in matches:
            text = text.strip()
            entity_type = entity_type.strip()
            
            # Map Korean entity types to English codes
            type_mapping = {
                "인물": "PS", "PS": "PS",
                "지명": "LC", "LC": "LC", 
                "기관": "OG", "OG": "OG",
                "날짜": "DT", "DT": "DT",
                "시간": "TI", "TI": "TI",
                "수량": "QT", "QT": "QT"
            }
            
            mapped_type = type_mapping.get(entity_type, entity_type)
            
            entities.append({
                "text": text,
                "type": mapped_type,
                "start": -1,  # Will be calculated later if needed
                "end": -1
            })
        
        return entities
    
    def calculate_entity_level_metrics(self, true_entities: List[Dict], pred_entities: List[Dict]) -> Dict[str, Any]:
        """Calculate Entity-level Macro F1 with exact boundary and type match."""
        if not pred_entities:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "correct": 0,
                "predicted": 0,
                "total": len(true_entities)
            }
        
        if not true_entities:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "correct": 0,
                "predicted": len(pred_entities),
                "total": 0
            }
        
        # Entity-level evaluation: exact boundary match + correct type
        correct = 0
        for true_entity in true_entities:
            for pred_entity in pred_entities:
                # Check exact boundary match and correct type
                if (true_entity["start"] == pred_entity["start"] and 
                    true_entity["end"] == pred_entity["end"] and 
                    true_entity["type"] == pred_entity["type"]):
                    correct += 1
                    break
        
        precision = correct / len(pred_entities) if pred_entities else 0.0
        recall = correct / len(true_entities) if true_entities else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "correct": correct,
            "predicted": len(pred_entities),
            "total": len(true_entities)
        }
    
    def calculate_character_level_metrics(self, text: str, true_entities: List[Dict], pred_entities: List[Dict]) -> Dict[str, Any]:
        """Calculate Character-level Macro F1 for Korean agglutinative language."""
        # Create character-level labels for the entire text
        char_labels_true = ['O'] * len(text)
        char_labels_pred = ['O'] * len(text)
        
        # Fill in true entity labels at character level
        for entity in true_entities:
            start_char = sum(len(text.split()[i]) + 1 for i in range(entity["start"])) if entity["start"] > 0 else 0
            end_char = sum(len(text.split()[i]) + 1 for i in range(entity["end"] + 1)) - 1
            
            # Mark characters as entity type
            for i in range(start_char, min(end_char + 1, len(text))):
                if i < len(char_labels_true):
                    char_labels_true[i] = entity["type"]
        
        # Fill in predicted entity labels at character level
        for entity in pred_entities:
            start_char = sum(len(text.split()[i]) + 1 for i in range(entity["start"])) if entity["start"] > 0 else 0
            end_char = sum(len(text.split()[i]) + 1 for i in range(entity["end"] + 1)) - 1
            
            # Mark characters as entity type
            for i in range(start_char, min(end_char + 1, len(text))):
                if i < len(char_labels_pred):
                    char_labels_pred[i] = entity["type"]
        
        # Calculate character-level metrics using seqeval
        try:
            # Convert to format expected by seqeval
            y_true = [char_labels_true]
            y_pred = [char_labels_pred]
            
            # Get unique labels (excluding 'O')
            labels = list(set([label for label in char_labels_true + char_labels_pred if label != 'O']))
            
            if not labels:
                return {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "correct_chars": 0,
                    "total_chars": len(char_labels_true),
                    "predicted_chars": len([l for l in char_labels_pred if l != 'O'])
                }
            
            # Calculate metrics
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            
            # Count correct character predictions
            correct_chars = sum(1 for i in range(len(char_labels_true)) 
                              if char_labels_true[i] != 'O' and char_labels_true[i] == char_labels_pred[i])
            
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "correct_chars": correct_chars,
                "total_chars": len([l for l in char_labels_true if l != 'O']),
                "predicted_chars": len([l for l in char_labels_pred if l != 'O'])
            }
            
        except Exception as e:
            logger.warning(f"Error calculating character-level metrics: {e}")
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "correct_chars": 0,
                "total_chars": len([l for l in char_labels_true if l != 'O']),
                "predicted_chars": len([l for l in char_labels_pred if l != 'O'])
            }
    
    def calculate_metrics(self, text: str, true_entities: List[Dict], pred_entities: List[Dict]) -> Dict[str, Any]:
        """Calculate both Entity-level and Character-level metrics."""
        # Calculate entity-level metrics
        entity_metrics = self.calculate_entity_level_metrics(true_entities, pred_entities)
        
        # Calculate character-level metrics
        char_metrics = self.calculate_character_level_metrics(text, true_entities, pred_entities)
        
        return {
            "entity_level": entity_metrics,
            "character_level": char_metrics
        }
    
    def run_benchmark(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the NER benchmark."""
        logger.info("Starting benchmark...")
        
        # Configure logging based on verbose mode
        if not self.config.verbose:
            # In default mode, completely suppress Google Cloud logging during benchmark
            original_levels = {}
            for logger_name in ['google.cloud.aiplatform', 'google.auth', 'google.api_core', 'urllib3', 'google.genai', 'google.cloud', 'google']:
                logger_obj = logging.getLogger(logger_name)
                original_levels[logger_name] = logger_obj.level
                logger_obj.setLevel(logging.CRITICAL)  # Only show critical errors
                logger_obj.disabled = True  # Completely disable
        else:
            # In verbose mode, allow all logging
            original_levels = {}
            for logger_name in ['google.cloud.aiplatform', 'google.auth', 'google.api_core', 'urllib3', 'google.genai', 'google.cloud', 'google']:
                logger_obj = logging.getLogger(logger_name)
                original_levels[logger_name] = logger_obj.level
                logger_obj.setLevel(logging.INFO)
                logger_obj.disabled = False
        
        start_time = time.time()
        total_samples = len(test_data)
        correct_predictions = 0
        total_entities = 0
        predicted_entities = 0
        
        # Process each sample with very infrequent progress bar updates (about 25% of current frequency)
        for i, sample in enumerate(tqdm(test_data, desc="Processing samples", mininterval=10.0, maxinterval=30.0, leave=False, unit="samples", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')):
            try:
                # Make prediction
                prediction_result = self.predict_single(sample["text"])
                
                # Calculate metrics
                metrics = self.calculate_metrics(sample["text"], sample["entities"], prediction_result.get("entities", []))
                
                # Update counters (using entity-level metrics for overall stats)
                correct_predictions += metrics["entity_level"]["correct"]
                total_entities += metrics["entity_level"]["total"]
                predicted_entities += metrics["entity_level"]["predicted"]
                
                # Store result
                result = {
                    "id": sample["id"],
                    "success": prediction_result["success"],
                    "raw_response": prediction_result.get("raw_response", ""),
                    "true_entities": sample["entities"],
                    "predicted_entities": prediction_result.get("entities", []),
                    "metrics": metrics,
                    "text": sample["text"],
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
                    "success": False,
                    "raw_response": "",
                    "true_entities": sample["entities"],
                    "predicted_entities": [],
                    "metrics": {
                        "entity_level": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "correct": 0, "predicted": 0, "total": len(sample["entities"])},
                        "character_level": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "correct_chars": 0, "total_chars": 0, "predicted_chars": 0}
                    },
                    "text": sample["text"],
                    "error": str(e)
                })
        
        # Calculate final metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate overall entity-level metrics
        overall_precision = correct_predictions / predicted_entities if predicted_entities > 0 else 0.0
        overall_recall = correct_predictions / total_entities if total_entities > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        # Calculate overall character-level metrics
        total_char_correct = sum(r["metrics"]["character_level"]["correct_chars"] for r in self.results)
        total_char_true = sum(r["metrics"]["character_level"]["total_chars"] for r in self.results)
        total_char_pred = sum(r["metrics"]["character_level"]["predicted_chars"] for r in self.results)
        
        char_precision = total_char_correct / total_char_pred if total_char_pred > 0 else 0.0
        char_recall = total_char_correct / total_char_true if total_char_true > 0 else 0.0
        char_f1 = 2 * (char_precision * char_recall) / (char_precision + char_recall) if (char_precision + char_recall) > 0 else 0.0
        
        self.metrics = {
            "total_samples": total_samples,
            "total_time": total_time,
            "average_time_per_sample": total_time / total_samples if total_samples > 0 else 0.0,
            "samples_per_second": total_samples / total_time if total_time > 0 else 0.0,
            "entity_level": {
                "total_entities": total_entities,
                "predicted_entities": predicted_entities,
                "correct_entities": correct_predictions,
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1
            },
            "character_level": {
                "total_chars": total_char_true,
                "predicted_chars": total_char_pred,
                "correct_chars": total_char_correct,
                "precision": char_precision,
                "recall": char_recall,
                "f1": char_f1
            }
        }
        
        # Restore original logging levels
        for logger_name, original_level in original_levels.items():
            logger_obj = logging.getLogger(logger_name)
            logger_obj.setLevel(original_level)
            logger_obj.disabled = False  # Re-enable all loggers
        
        logger.info("Benchmark completed!")
        logger.info(f"Entity-level F1: {overall_f1:.4f} ({correct_predictions}/{total_entities})")
        logger.info(f"Entity-level Precision: {overall_precision:.4f}")
        logger.info(f"Entity-level Recall: {overall_recall:.4f}")
        logger.info(f"Character-level F1: {char_f1:.4f} ({total_char_correct}/{total_char_true})")
        logger.info(f"Character-level Precision: {char_precision:.4f}")
        logger.info(f"Character-level Recall: {char_recall:.4f}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average time per sample: {total_time / total_samples:.3f} seconds")
        
        return self.metrics
    
    def save_results(self):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = os.path.join(self.config.output_dir, f"klue_ner_metrics_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Metrics saved to: {metrics_file}")
        
        # Save detailed results
        results_file = os.path.join(self.config.output_dir, f"klue_ner_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Save as CSV
        csv_data = []
        for result in self.results:
            csv_data.append({
                "id": result["id"],
                "success": result["success"],
                "entity_correct": result["metrics"]["entity_level"]["correct"],
                "entity_true_count": result["metrics"]["entity_level"]["total"],
                "entity_pred_count": result["metrics"]["entity_level"]["predicted"],
                "entity_precision": result["metrics"]["entity_level"]["precision"],
                "entity_recall": result["metrics"]["entity_level"]["recall"],
                "entity_f1": result["metrics"]["entity_level"]["f1"],
                "char_correct": result["metrics"]["character_level"]["correct_chars"],
                "char_true_count": result["metrics"]["character_level"]["total_chars"],
                "char_pred_count": result["metrics"]["character_level"]["predicted_chars"],
                "char_precision": result["metrics"]["character_level"]["precision"],
                "char_recall": result["metrics"]["character_level"]["recall"],
                "char_f1": result["metrics"]["character_level"]["f1"],
                "text": result["text"],
                "error": result.get("error", "")
            })
        
        csv_file = os.path.join(self.config.output_dir, f"klue_ner_results_{timestamp}.csv")
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
            "correct_entities": correct_count,
            "timestamp": timestamp
        }
        
        # Save intermediate metrics
        metrics_file = os.path.join(self.config.output_dir, f"klue_ner_metrics_{current_count:06d}_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(intermediate_metrics, f, ensure_ascii=False, indent=2)
        
        # Save intermediate results as JSON
        results_file = os.path.join(self.config.output_dir, f"klue_ner_results_{current_count:06d}_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # Save intermediate results as CSV
        csv_data = []
        for result in self.results:
            csv_data.append({
                "id": result["id"],
                "success": result["success"],
                "entity_correct": result["metrics"]["entity_level"]["correct"],
                "entity_true_count": result["metrics"]["entity_level"]["total"],
                "entity_pred_count": result["metrics"]["entity_level"]["predicted"],
                "entity_precision": result["metrics"]["entity_level"]["precision"],
                "entity_recall": result["metrics"]["entity_level"]["recall"],
                "entity_f1": result["metrics"]["entity_level"]["f1"],
                "char_correct": result["metrics"]["character_level"]["correct_chars"],
                "char_true_count": result["metrics"]["character_level"]["total_chars"],
                "char_pred_count": result["metrics"]["character_level"]["predicted_chars"],
                "char_precision": result["metrics"]["character_level"]["precision"],
                "char_recall": result["metrics"]["character_level"]["recall"],
                "char_f1": result["metrics"]["character_level"]["f1"],
                "text": result["text"],
                "error": result.get("error", "")
            })
        
        csv_file = os.path.join(self.config.output_dir, f"klue_ner_results_{current_count:06d}_{timestamp}.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.info(f"Intermediate results saved at {current_count} samples (JSON + CSV)")
    
    def save_error_analysis(self, timestamp: str):
        """Save error analysis for failed predictions."""
        error_samples = [r for r in self.results if not r["success"] or r.get("error")]
        
        if not error_samples:
            logger.info("No errors to analyze")
            return
        
        error_file = os.path.join(self.config.output_dir, f"klue_ner_error_analysis_{timestamp}.txt")
        
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write("KLUE NER Error Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            for i, sample in enumerate(error_samples[:10], 1):  # Show first 10 errors
                f.write(f"{i}. Sample ID: {sample['id']}\n")
                f.write(f"   Text: {sample['text']}\n")
                f.write(f"   True Entities: {sample['true_entities']}\n")
                f.write(f"   Predicted Entities: {sample['predicted_entities']}\n")
                if sample.get("error"):
                    f.write(f"   Error: {sample['error']}\n")
                f.write("\n")
        
        logger.info(f"Error analysis saved to: {error_file}")
    
    def print_detailed_metrics(self):
        """Print detailed benchmark results."""
        print("=" * 60)
        print("KLUE Named Entity Recognition Benchmark Results")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Platform: Google Cloud Vertex AI")
        print(f"Project: {self.config.project_id or os.getenv('GOOGLE_CLOUD_PROJECT')}")
        print(f"Location: {self.config.location}")
        print(f"F1 Score: {self.metrics['f1']:.4f} ({self.metrics['correct_entities']}/{self.metrics['total_entities']})")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall: {self.metrics['recall']:.4f}")
        print(f"Total Time: {self.metrics['total_time']:.2f} seconds")
        print(f"Average Time per Sample: {self.metrics['average_time_per_sample']:.3f} seconds")
        print(f"Samples per Second: {self.metrics['samples_per_second']:.2f}")
        print()
        
        # Per-entity type analysis
        entity_metrics = {}
        for result in self.results:
            for entity in result["true_entities"]:
                entity_type = entity["type"]
                if entity_type not in entity_metrics:
                    entity_metrics[entity_type] = {"total": 0, "correct": 0}
                entity_metrics[entity_type]["total"] += 1
                
                # Check if this entity was correctly predicted
                for pred_entity in result["predicted_entities"]:
                    if (entity["text"] == pred_entity["text"] and 
                        entity["type"] == pred_entity["type"]):
                        entity_metrics[entity_type]["correct"] += 1
                        break
        
        print("Per-entity Type Performance:")
        for entity_type, metrics in entity_metrics.items():
            accuracy = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0.0
            print(f"  {entity_type} ({self.ENTITY_TYPES.get(entity_type, entity_type)}): {accuracy:.4f} ({metrics['correct']}/{metrics['total']})")
        
        print()
        
        # Error analysis
        error_count = sum(1 for r in self.results if not r["success"] or r.get("error"))
        if error_count > 0:
            print(f"Error Analysis (showing first 5 errors):")
            error_samples = [r for r in self.results if not r["success"] or r.get("error")]
            for i, sample in enumerate(error_samples[:5], 1):
                print(f"  {i}. Sample ID: {sample['id']}")
                print(f"     Text: {sample['text'][:100]}...")
                print(f"     True Entities: {len(sample['true_entities'])} entities")
                print(f"     Predicted Entities: {len(sample['predicted_entities'])} entities")
                if sample.get("error"):
                    print(f"     Error: {sample['error']}")
                print()

def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="KLUE NER Benchmark with Gemini 2.5 Flash")
    parser.add_argument("--project-id", type=str, help="Google Cloud project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="Vertex AI location")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to test")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.1, help="Model temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum output tokens")
    parser.add_argument("--no-save-predictions", action="store_true", help="Skip saving detailed predictions")
    parser.add_argument("--save-interval", type=int, default=50, help="Save intermediate results every N samples")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (shows Google Cloud API details)")
    
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
        save_interval=args.save_interval,
        verbose=args.verbose
    )
    
    # Create and run benchmark
    benchmark = KLUENamedEntityRecognitionBenchmark(config)
    
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