#!/usr/bin/env python3
"""
SEA-HELM Benchmark with Gemini 2.5 Flash on Vertex AI
This script benchmarks Gemini 2.5 Flash on the SEA-HELM (SouthEast Asian Holistic Evaluation of Language Models) 
using Google Cloud Vertex AI.
"""

import os
import json
import time
import argparse
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import sys

# Add the current directory to the path so we can import sea_helm modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from seahelm_evaluation import SeaHelmEvaluation
from serving.gemini_serving import GeminiServing
from base_logger import setup_root_logger, get_logger

# Configure logging
logger = get_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for the SEA-HELM benchmark with Gemini 2.5 Flash."""
    model_name: str = "gemini-2.5-flash"
    project_id: Optional[str] = None
    location: str = "us-central1"
    output_dir: str = "benchmark_results"
    tasks_configuration: str = "seahelm"
    max_samples: Optional[int] = None
    sleep_interval: float = 0.04
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 1.0
    top_k: int = 1
    is_base_model: bool = False
    is_vision_model: bool = False
    is_reasoning_model: bool = False
    num_in_context_examples: Optional[int] = None
    fewshot_as_multiturn: bool = False
    inference_file_type: str = "csv"
    tokenize_prompts: bool = True
    skip_task: Optional[List[str]] = None
    limit: Optional[int] = None
    no_batching: bool = True
    use_cached_results: bool = True
    rerun_cached_results: bool = False


class SEAHELMGemini25FlashBenchmark:
    """Benchmark class for SEA-HELM using Gemini 2.5 Flash on Vertex AI."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.llm = None
        self.evaluator = None
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize the benchmark
        self._initialize_benchmark()
        
    def _initialize_benchmark(self):
        """Initialize the benchmark components."""
        try:
            # Initialize Gemini serving
            self.llm = GeminiServing(
                model=self.config.model_name,
                project_id=self.config.project_id,
                location=self.config.location,
                is_base_model=self.config.is_base_model,
                sleep_interval=self.config.sleep_interval,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
            
            # Initialize SEA-HELM evaluator
            self.evaluator = SeaHelmEvaluation(
                llm=self.llm,
                tasks_configuration=self.config.tasks_configuration,
                output_dir=self.config.output_dir,
                model_name=self.config.model_name,
                is_base_model=self.config.is_base_model,
                is_vision_model=self.config.is_vision_model,
                is_reasoning_model=self.config.is_reasoning_model,
                num_in_context_examples=self.config.num_in_context_examples,
                fewshot_as_multiturn=self.config.fewshot_as_multiturn,
                inference_file_type=self.config.inference_file_type,
                tokenize_prompts=self.config.tokenize_prompts,
                skip_task=self.config.skip_task,
                limit=self.config.limit,
                no_batching=self.config.no_batching,
            )
            
            logger.info(f"Initialized SEA-HELM benchmark with Gemini 2.5 Flash")
            logger.info(f"Model: {self.config.model_name}")
            logger.info(f"Output directory: {self.config.output_dir}")
            logger.info(f"Tasks configuration: {self.config.tasks_configuration}")
            
        except Exception as e:
            logger.error(f"Failed to initialize benchmark: {e}")
            raise
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete SEA-HELM benchmark.
        
        Returns:
            Dictionary containing benchmark results
        """
        try:
            logger.info("Starting SEA-HELM benchmark with Gemini 2.5 Flash...")
            start_time = time.time()
            
            # Run the evaluation
            results = self.evaluator.run_evaluation(
                llm=self.llm,
                use_cached_results=self.config.use_cached_results
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"Benchmark completed in {total_time:.2f} seconds")
            
            # Save final results
            self._save_final_results(results, total_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            raise
    
    def _save_final_results(self, results: Dict[str, Any], total_time: float):
        """Save final benchmark results."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                self.config.output_dir, 
                f"seahelm_gemini25flash_results_{timestamp}.json"
            )
            
            # Add metadata to results
            results_with_metadata = {
                "model": self.config.model_name,
                "timestamp": timestamp,
                "total_time_seconds": total_time,
                "config": {
                    "project_id": self.config.project_id,
                    "location": self.config.location,
                    "tasks_configuration": self.config.tasks_configuration,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "is_base_model": self.config.is_base_model,
                    "is_vision_model": self.config.is_vision_model,
                    "is_reasoning_model": self.config.is_reasoning_model,
                },
                "results": results
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Final results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving final results: {e}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of the benchmark results."""
        try:
            print("\n" + "="*80)
            print("SEA-HELM GEMINI 2.5 FLASH BENCHMARK SUMMARY")
            print("="*80)
            print(f"Model: {self.config.model_name}")
            print(f"Tasks Configuration: {self.config.tasks_configuration}")
            print(f"Output Directory: {self.config.output_dir}")
            print(f"Base Model: {self.config.is_base_model}")
            print(f"Vision Model: {self.config.is_vision_model}")
            print(f"Reasoning Model: {self.config.is_reasoning_model}")
            print("-"*80)
            
            # Print results summary if available
            if results and "results" in results:
                print("Results Summary:")
                for key, value in results["results"].items():
                    if isinstance(value, dict) and "metrics" in value:
                        print(f"  {key}: {value['metrics']}")
                    else:
                        print(f"  {key}: {value}")
            
            print("="*80)
            
        except Exception as e:
            logger.error(f"Error printing summary: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SEA-HELM Benchmark with Gemini 2.5 Flash on Vertex AI"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash)"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=None,
        help="Google Cloud project ID (default: from GOOGLE_CLOUD_PROJECT env var)"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="us-central1",
        help="Google Cloud location (default: us-central1)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)"
    )
    parser.add_argument(
        "--tasks-configuration",
        type=str,
        default="seahelm",
        help="Tasks configuration to run (default: seahelm)"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter (default: 1.0)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Top-k sampling parameter (default: 1)"
    )
    parser.add_argument(
        "--sleep-interval",
        type=float,
        default=0.04,
        help="Sleep interval between API calls in seconds (default: 0.04)"
    )
    
    # Model type flags
    parser.add_argument(
        "--base-model",
        action="store_true",
        help="Treat as base model"
    )
    parser.add_argument(
        "--vision-model",
        action="store_true",
        help="Treat as vision model"
    )
    parser.add_argument(
        "--reasoning-model",
        action="store_true",
        help="Treat as reasoning model"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task"
    )
    parser.add_argument(
        "--skip-task",
        type=str,
        nargs="+",
        default=None,
        help="Tasks to skip"
    )
    parser.add_argument(
        "--no-batching",
        action="store_true",
        default=True,
        help="Disable batching (default: True)"
    )
    parser.add_argument(
        "--no-cached-results",
        action="store_true",
        help="Don't use cached results"
    )
    parser.add_argument(
        "--rerun-cached-results",
        action="store_true",
        help="Rerun cached results"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the SEA-HELM benchmark with Gemini 2.5 Flash."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Create configuration
        config = BenchmarkConfig(
            model_name=args.model_name,
            project_id=args.project_id,
            location=args.location,
            output_dir=args.output_dir,
            tasks_configuration=args.tasks_configuration,
            max_samples=args.max_samples,
            sleep_interval=args.sleep_interval,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            is_base_model=args.base_model,
            is_vision_model=args.vision_model,
            is_reasoning_model=args.reasoning_model,
            limit=args.limit,
            skip_task=args.skip_task,
            no_batching=args.no_batching,
            use_cached_results=not args.no_cached_results,
            rerun_cached_results=args.rerun_cached_results,
        )
        
        # Initialize and run benchmark
        benchmark = SEAHELMGemini25FlashBenchmark(config)
        results = benchmark.run_benchmark()
        
        # Print summary
        benchmark.print_summary(results)
        
        logger.info("SEA-HELM benchmark with Gemini 2.5 Flash completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 