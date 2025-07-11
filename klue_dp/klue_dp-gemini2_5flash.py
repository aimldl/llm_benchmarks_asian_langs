#!/usr/bin/env python3
"""
KLUE Dependency Parsing (DP) Benchmark with Gemini 2.5 Flash on Vertex AI
This script benchmarks Gemini 2.5 Flash on the Korean Language Understanding Evaluation (KLUE) Dependency Parsing task using Google Cloud Vertex AI.
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
    max_tokens: int = 4096  # Increased for DP task
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

class KLUEDependencyParsingBenchmark:
    """Benchmark class for KLUE Dependency Parsing task using Vertex AI."""
    
    # Korean POS tags
    POS_TAGS = {
        "NNG": "일반명사",
        "NNP": "고유명사", 
        "NNB": "의존명사",
        "NNBC": "단위를 나타내는 명사",
        "NR": "수사",
        "NP": "대명사",
        "VV": "동사",
        "VA": "형용사",
        "VX": "보조용언",
        "VCP": "긍정지정사",
        "VCN": "부정지정사",
        "MM": "관형사",
        "MAG": "일반부사",
        "MAJ": "접속부사",
        "IC": "감탄사",
        "JKS": "주격조사",
        "JKC": "보격조사",
        "JKG": "관형격조사",
        "JKO": "목적격조사",
        "JKB": "부사격조사",
        "JKV": "호격조사",
        "JKQ": "인용격조사",
        "JX": "보조사",
        "JC": "접속조사",
        "EP": "선어말어미",
        "EF": "종결어미",
        "EC": "연결어미",
        "ETN": "명사형전성어미",
        "ETM": "관형형전성어미",
        "XPN": "체언접두사",
        "XSN": "명사파생접미사",
        "XSV": "동사파생접미사",
        "XSA": "형용사파생접미사",
        "XR": "어근",
        "SF": "마침표,물음표,느낌표",
        "SP": "쉼표,가운뎃점,콜론,빗금",
        "SS": "따옴표,괄호표,줄표",
        "SE": "줄임표",
        "SO": "붙임표(물결,숨김,빠짐)",
        "SW": "기타기호(논리수학기호,화폐기호)",
        "SL": "외국어",
        "SH": "한자",
        "SN": "숫자"
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
        Load the KLUE DP dataset, convert it to a list of dictionaries,
        and efficiently limit the number of samples based on the configuration.
        """
        try:
            logger.info("Loading KLUE DP dataset for dependency parsing...")
            
            # Load the validation split from the Hugging Face Hub.
            validation_dataset = load_dataset('klue', 'dp', split='validation')

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
                    
                # Process DP data
                processed_data.append({
                    "id": item["guid"],
                    "sentence": item["sentence"],
                    "words": item["words"],
                    "pos_tags": item["pos_tags"],
                    "heads": item["heads"],
                    "deprels": item["deprels"]
                })

            logger.info(f"✅ Successfully loaded {len(processed_data)} samples.")
            return processed_data
            
        except KeyError as e:
            logger.error(f"❌ A key was not found in the dataset item: {e}. The dataset schema may have changed.")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load or process the dataset: {e}")
            raise
    
    def create_prompt(self, sentence: str, words: List[str], pos_tags: List[str]) -> str:
        """Create detailed prompt for dependency parsing."""
        # Create word-pos pairs
        word_pos_pairs = []
        for i, (word, pos) in enumerate(zip(words, pos_tags)):
            pos_name = self.POS_TAGS.get(pos, pos)
            word_pos_pairs.append(f"{i+1}. {word} ({pos}: {pos_name})")
        
        word_pos_text = "\n".join(word_pos_pairs)
        
        prompt = f"""역할: 당신은 한국어 문장의 의존 구문 분석을 수행하는 "전문 구문 분석 AI"입니다.

임무: 주어진 한국어 문장의 각 단어에 대해 품사 태그와 의존 관계를 분석하여, 각 단어의 의존소(부모 단어)와 의존 관계를 파악해 주세요.

문장: {sentence}

단어와 품사:
{word_pos_text}

의존 관계 유형:

1. 주어-서술어 관계:
   - nsubj: 주어 (명사가 동사/형용사를 수식)
   - nsubj:pass: 피동 주어 (피동문의 주어)

2. 목적어 관계:
   - obj: 직접목적어 (동사의 직접적인 대상)
   - iobj: 간접목적어 (동사의 간접적인 대상)

3. 보어 관계:
   - ccomp: 보문 (동사나 형용사의 보어)
   - xcomp: 개방형 보문 (주어가 없는 보문)

4. 수식어 관계:
   - amod: 형용사 수식어 (명사를 수식하는 형용사)
   - nummod: 수사 수식어 (명사를 수식하는 수사)
   - det: 한정사 (명사를 한정하는 요소)

5. 부사어 관계:
   - advmod: 부사 수식어 (동사나 형용사를 수식하는 부사)
   - advcl: 부사절 (부사적 기능을 하는 절)

6. 조사 관계:
   - case: 격조사 (명사에 붙는 조사)
   - mark: 접속조사 (절을 연결하는 조사)

7. 기타 관계:
   - root: 루트 (문장의 중심 동사)
   - punct: 구두점
   - compound: 복합어
   - flat: 평면 구조 (이름, 날짜 등)
   - list: 나열
   - parataxis: 병렬 구조
   - discourse: 담화 표지
   - vocative: 호격
   - expl: 가주어/가목적어
   - aux: 보조동사
   - cop: 연결동사
   - acl: 관계절
   - acl:relcl: 관계절
   - appos: 동격어
   - dislocated: 도치된 요소
   - orphan: 고아 요소
   - goeswith: 연결된 요소
   - reparandum: 수정된 요소
   - dep: 기타 의존 관계

분석 지침:

1. 각 단어는 정확히 하나의 의존소를 가져야 합니다 (루트 제외).
2. 루트는 문장의 중심 동사나 형용사입니다.
3. 의존 관계는 의미적이고 문법적 관계를 모두 고려합니다.
4. 한국어의 특성상 조사가 많이 사용되므로 주의 깊게 분석합니다.
5. 복합어나 관용구는 하나의 단위로 처리할 수 있습니다.

출력 형식:
각 단어에 대해 "단어번호: 의존소번호(의존관계)" 형태로 출력하세요.
예시:
1: 0(root)
2: 1(nsubj)
3: 1(obj)

의존 구문 분석 결과:"""
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
    
    def predict_single(self, sentence: str, words: List[str], pos_tags: List[str]) -> Dict[str, Any]:
        """Make a single prediction for DP task."""
        try:
            prompt = self.create_prompt(sentence, words, pos_tags)
            
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
                predicted_deps = self._parse_dp_response(response.text, len(words))
                return {
                    "success": True,
                    "heads": predicted_deps["heads"],
                    "deprels": predicted_deps["deprels"],
                    "raw_response": response.text
                }
            else:
                logger.error("Cannot get the response text.")
                return {
                    "success": False,
                    "heads": [0] * len(words),
                    "deprels": ["dep"] * len(words),
                    "raw_response": "",
                    "error": "No response text"
                }
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "success": False,
                "heads": [0] * len(words),
                "deprels": ["dep"] * len(words),
                "raw_response": "",
                "error": str(e)
            }
    
    def _parse_dp_response(self, response_text: str, num_words: int) -> Dict[str, List]:
        """Parse the DP response from the model."""
        # Clean the response text
        response_text = response_text.strip()
        
        # Initialize default values
        heads = [0] * num_words
        deprels = ["dep"] * num_words
        
        # Look for dependency patterns
        # Pattern: "word_index: head_index(deprel)"
        dep_pattern = r'(\d+):\s*(\d+)\s*\(([^)]+)\)'
        matches = re.findall(dep_pattern, response_text)
        
        for match in matches:
            try:
                word_idx = int(match[0]) - 1  # Convert to 0-based index
                head_idx = int(match[1])
                deprel = match[2].strip()
                
                # Validate indices
                if 0 <= word_idx < num_words:
                    heads[word_idx] = head_idx
                    deprels[word_idx] = deprel
            except (ValueError, IndexError):
                continue
        
        return {
            "heads": heads,
            "deprels": deprels
        }
    
    def calculate_metrics(self, true_heads: List[int], true_deprels: List[str], 
                         pred_heads: List[int], pred_deprels: List[str]) -> Dict[str, Any]:
        """Calculate UAS and LAS metrics for dependency parsing."""
        if len(true_heads) != len(pred_heads) or len(true_deprels) != len(pred_deprels):
            return {
                "uas": 0.0,
                "las": 0.0,
                "correct_heads": 0,
                "correct_labels": 0,
                "total": len(true_heads)
            }
        
        correct_heads = sum(1 for t, p in zip(true_heads, pred_heads) if t == p)
        correct_labels = sum(1 for th, ph, td, pd in zip(true_heads, pred_heads, true_deprels, pred_deprels) 
                           if th == ph and td == pd)
        
        total = len(true_heads)
        uas = correct_heads / total if total > 0 else 0.0
        las = correct_labels / total if total > 0 else 0.0
        
        return {
            "uas": uas,
            "las": las,
            "correct_heads": correct_heads,
            "correct_labels": correct_labels,
            "total": total
        }
    
    def run_benchmark(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the DP benchmark."""
        logger.info("Starting benchmark...")
        
        start_time = time.time()
        total_samples = len(test_data)
        total_words = 0
        total_correct_heads = 0
        total_correct_labels = 0
        
        # Process each sample
        for i, sample in enumerate(tqdm(test_data, desc="Processing samples")):
            try:
                # Make prediction
                prediction_result = self.predict_single(
                    sample["sentence"], 
                    sample["words"], 
                    sample["pos_tags"]
                )
                
                # Calculate metrics
                metrics = self.calculate_metrics(
                    sample["heads"], 
                    sample["deprels"],
                    prediction_result.get("heads", [0] * len(sample["words"])),
                    prediction_result.get("deprels", ["dep"] * len(sample["words"]))
                )
                
                # Update counters
                total_words += metrics["total"]
                total_correct_heads += metrics["correct_heads"]
                total_correct_labels += metrics["correct_labels"]
                
                # Store result
                result = {
                    "id": sample["id"],
                    "sentence": sample["sentence"],
                    "words": sample["words"],
                    "pos_tags": sample["pos_tags"],
                    "true_heads": sample["heads"],
                    "true_deprels": sample["deprels"],
                    "predicted_heads": prediction_result.get("heads", [0] * len(sample["words"])),
                    "predicted_deprels": prediction_result.get("deprels", ["dep"] * len(sample["words"])),
                    "metrics": metrics,
                    "success": prediction_result["success"],
                    "raw_response": prediction_result.get("raw_response", ""),
                    "error": prediction_result.get("error", "")
                }
                
                self.results.append(result)
                
                # Save intermediate results
                if (i + 1) % self.config.save_interval == 0:
                    self.save_intermediate_results(i + 1, total_correct_heads, total_correct_labels, start_time)
                
                # Sleep between API calls
                time.sleep(self.config.sleep_interval_between_api_calls)
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                self.results.append({
                    "id": sample["id"],
                    "sentence": sample["sentence"],
                    "words": sample["words"],
                    "pos_tags": sample["pos_tags"],
                    "true_heads": sample["heads"],
                    "true_deprels": sample["deprels"],
                    "predicted_heads": [0] * len(sample["words"]),
                    "predicted_deprels": ["dep"] * len(sample["words"]),
                    "metrics": {"uas": 0.0, "las": 0.0, "correct_heads": 0, "correct_labels": 0, "total": len(sample["words"])},
                    "success": False,
                    "raw_response": "",
                    "error": str(e)
                })
        
        # Calculate final metrics
        end_time = time.time()
        total_time = end_time - start_time
        
        uas = total_correct_heads / total_words if total_words > 0 else 0.0
        las = total_correct_labels / total_words if total_words > 0 else 0.0
        
        self.metrics = {
            "total_samples": total_samples,
            "total_words": total_words,
            "correct_heads": total_correct_heads,
            "correct_labels": total_correct_labels,
            "uas": uas,
            "las": las,
            "total_time": total_time,
            "average_time_per_sample": total_time / total_samples if total_samples > 0 else 0.0,
            "samples_per_second": total_samples / total_time if total_time > 0 else 0.0
        }
        
        logger.info("Benchmark completed!")
        logger.info(f"UAS: {uas:.4f} ({total_correct_heads}/{total_words})")
        logger.info(f"LAS: {las:.4f} ({total_correct_labels}/{total_words})")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average time per sample: {total_time / total_samples:.3f} seconds")
        
        return self.metrics
    
    def save_results(self):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = os.path.join(self.config.output_dir, f"klue_dp_metrics_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Metrics saved to: {metrics_file}")
        
        # Save detailed results
        results_file = os.path.join(self.config.output_dir, f"klue_dp_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Save as CSV
        csv_data = []
        for result in self.results:
            csv_data.append({
                "id": result["id"],
                "sentence": result["sentence"],
                "words": " ".join(result["words"]),
                "pos_tags": " ".join(result["pos_tags"]),
                "true_heads": " ".join(map(str, result["true_heads"])),
                "true_deprels": " ".join(result["true_deprels"]),
                "predicted_heads": " ".join(map(str, result["predicted_heads"])),
                "predicted_deprels": " ".join(result["predicted_deprels"]),
                "uas": result["metrics"]["uas"],
                "las": result["metrics"]["las"],
                "success": result["success"],
                "error": result.get("error", "")
            })
        
        csv_file = os.path.join(self.config.output_dir, f"klue_dp_results_{timestamp}.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"Results saved as CSV: {csv_file}")
        
        # Save error analysis
        self.save_error_analysis(timestamp)
    
    def save_intermediate_results(self, current_count: int, correct_heads: int, correct_labels: int, start_time: float):
        """Save intermediate results."""
        if not self.config.save_predictions:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate intermediate metrics
        total_words = sum(r["metrics"]["total"] for r in self.results)
        uas = correct_heads / total_words if total_words > 0 else 0.0
        las = correct_labels / total_words if total_words > 0 else 0.0
        
        intermediate_metrics = {
            "samples_processed": current_count,
            "total_words": total_words,
            "correct_heads": correct_heads,
            "correct_labels": correct_labels,
            "uas": uas,
            "las": las,
            "timestamp": timestamp
        }
        
        # Save intermediate metrics
        metrics_file = os.path.join(self.config.output_dir, f"klue_dp_metrics_{current_count:06d}_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(intermediate_metrics, f, ensure_ascii=False, indent=2)
        
        # Save intermediate results
        results_file = os.path.join(self.config.output_dir, f"klue_dp_results_{current_count:06d}_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Intermediate results saved at {current_count} samples")
    
    def save_error_analysis(self, timestamp: str):
        """Save error analysis for failed predictions."""
        error_samples = [r for r in self.results if not r["success"] or r.get("error") or r["metrics"]["uas"] < 0.5]
        
        if not error_samples:
            logger.info("No errors to analyze")
            return
        
        error_file = os.path.join(self.config.output_dir, f"klue_dp_error_analysis_{timestamp}.txt")
        
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write("KLUE DP Error Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            for i, sample in enumerate(error_samples[:10], 1):  # Show first 10 errors
                f.write(f"{i}. Sample ID: {sample['id']}\n")
                f.write(f"   Sentence: {sample['sentence']}\n")
                f.write(f"   Words: {' '.join(sample['words'])}\n")
                f.write(f"   POS Tags: {' '.join(sample['pos_tags'])}\n")
                f.write(f"   True Heads: {sample['true_heads']}\n")
                f.write(f"   Predicted Heads: {sample['predicted_heads']}\n")
                f.write(f"   UAS: {sample['metrics']['uas']:.4f}\n")
                f.write(f"   LAS: {sample['metrics']['las']:.4f}\n")
                if sample.get("error"):
                    f.write(f"   Error: {sample['error']}\n")
                f.write("\n")
        
        logger.info(f"Error analysis saved to: {error_file}")
    
    def print_detailed_metrics(self):
        """Print detailed benchmark results."""
        print("=" * 60)
        print("KLUE Dependency Parsing Benchmark Results")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Platform: Google Cloud Vertex AI")
        print(f"Project: {self.config.project_id or os.getenv('GOOGLE_CLOUD_PROJECT')}")
        print(f"Location: {self.config.location}")
        print(f"UAS: {self.metrics['uas']:.4f} ({self.metrics['correct_heads']}/{self.metrics['total_words']})")
        print(f"LAS: {self.metrics['las']:.4f} ({self.metrics['correct_labels']}/{self.metrics['total_words']})")
        print(f"Total Time: {self.metrics['total_time']:.2f} seconds")
        print(f"Average Time per Sample: {self.metrics['average_time_per_sample']:.3f} seconds")
        print(f"Samples per Second: {self.metrics['samples_per_second']:.2f}")
        print()
        
        # Per-POS analysis
        pos_metrics = {}
        for result in self.results:
            for word, pos, true_head, pred_head, true_deprel, pred_deprel in zip(
                result["words"], result["pos_tags"], result["true_heads"], 
                result["predicted_heads"], result["true_deprels"], result["predicted_deprels"]
            ):
                if pos not in pos_metrics:
                    pos_metrics[pos] = {"total": 0, "correct_heads": 0, "correct_labels": 0}
                pos_metrics[pos]["total"] += 1
                
                if true_head == pred_head:
                    pos_metrics[pos]["correct_heads"] += 1
                    if true_deprel == pred_deprel:
                        pos_metrics[pos]["correct_labels"] += 1
        
        print("Per-POS Performance:")
        for pos, metrics in pos_metrics.items():
            uas = metrics["correct_heads"] / metrics["total"] if metrics["total"] > 0 else 0.0
            las = metrics["correct_labels"] / metrics["total"] if metrics["total"] > 0 else 0.0
            pos_name = self.POS_TAGS.get(pos, pos)
            print(f"  {pos} ({pos_name}): UAS={uas:.4f}, LAS={las:.4f} ({metrics['total']} words)")
        
        print()
        
        # Error analysis
        error_count = sum(1 for r in self.results if not r["success"] or r.get("error") or r["metrics"]["uas"] < 0.5)
        if error_count > 0:
            print(f"Error Analysis (showing first 5 errors):")
            error_samples = [r for r in self.results if not r["success"] or r.get("error") or r["metrics"]["uas"] < 0.5]
            for i, sample in enumerate(error_samples[:5], 1):
                print(f"  {i}. Sample ID: {sample['id']}")
                print(f"     Sentence: {sample['sentence'][:100]}...")
                print(f"     Words: {' '.join(sample['words'][:10])}...")
                print(f"     UAS: {sample['metrics']['uas']:.4f} | LAS: {sample['metrics']['las']:.4f}")
                if sample.get("error"):
                    print(f"     Error: {sample['error']}")
                print()

def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="KLUE DP Benchmark with Gemini 2.5 Flash")
    parser.add_argument("--project-id", type=str, help="Google Cloud project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="Vertex AI location")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to test")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.1, help="Model temperature")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum output tokens")
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
    benchmark = KLUEDependencyParsingBenchmark(config)
    
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