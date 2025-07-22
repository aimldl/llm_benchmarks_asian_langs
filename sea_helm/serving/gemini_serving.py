import os
import time
import asyncio
import importlib_metadata
from typing import List, Dict, Any, Optional
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold
)

from base_logger import get_logger
from serving.base_serving import BaseServing

logger = get_logger(__name__)


class GeminiServing(BaseServing):
    """Serving class for Gemini models on Vertex AI."""
    
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        project_id: Optional[str] = None,
        location: str = "us-central1",
        api_key: Optional[str] = None,
        is_base_model: bool = False,
        sleep_interval: float = 0.04,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 1.0,
        top_k: int = 1,
    ):
        """
        Initialize Gemini serving.
        
        Args:
            model: Gemini model name (e.g., "gemini-2.5-flash", "gemini-2.5-pro")
            project_id: Google Cloud project ID
            location: Google Cloud location
            api_key: API key (not used for Vertex AI)
            is_base_model: Whether this is a base model
            sleep_interval: Sleep interval between API calls
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
        """
        self.model = model
        self.is_base_model = is_base_model
        self.sleep_interval = sleep_interval
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        # Initialize Vertex AI
        self._initialize_vertex_ai(project_id, location)
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"Initialized Gemini serving with model: {model}")
    
    def _initialize_vertex_ai(self, project_id: Optional[str], location: str):
        """Initialize Vertex AI client."""
        try:
            project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                raise ValueError(
                    "Google Cloud project ID must be provided via project_id parameter "
                    "or by setting the GOOGLE_CLOUD_PROJECT environment variable."
                )
            
            self.project_id = project_id
            self.location = location
            
            # Initialize genai client for Vertex AI
            self.client = genai.Client(
                vertexai=True, 
                project=project_id, 
                location=location
            )
            
            logger.info(f"Initialized Vertex AI with project: {project_id}, location: {location}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
    def _initialize_model(self):
        """Initialize the Gemini model."""
        try:
            # Store model name for later use
            self.model_name = self.model
            logger.info(f"Model name set to: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _configure_generation_config(self, **kwargs) -> GenerateContentConfig:
        """Configure generation parameters."""
        # Use instance defaults, but allow override via kwargs
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        temperature = kwargs.get('temperature', self.temperature)
        top_p = kwargs.get('top_p', self.top_p)
        top_k = kwargs.get('top_k', self.top_k)
        
        return GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    
    def _configure_safety_settings(self, threshold=HarmBlockThreshold.BLOCK_NONE):
        """Configure safety settings."""
        return [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=threshold
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=threshold
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=threshold
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=threshold
            ),
        ]
    
    def _messages_to_content(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert messages format to Gemini content format."""
        content = []
        
        for message in messages:
            role = message.get("role", "user")
            content_text = message.get("content", "")
            
            # Map roles to Gemini format
            if role == "user":
                content.append({"role": "user", "parts": [{"text": content_text}]})
            elif role == "assistant":
                content.append({"role": "model", "parts": [{"text": content_text}]})
            elif role == "system":
                # For system messages, prepend to user message or handle specially
                if content and content[-1]["role"] == "user":
                    content[-1]["parts"][0]["text"] = f"{content_text}\n\n{content[-1]['parts'][0]['text']}"
                else:
                    content.append({"role": "user", "parts": [{"text": content_text}]})
        
        return content
    
    def generate(self, messages: List[Dict[str, str]], logprobs: bool = False, **generation_kwargs) -> Dict[str, Any]:
        """
        Generate a single response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            logprobs: Whether to return log probabilities (not supported by Gemini)
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing the generated response
        """
        try:
            # Convert messages to Gemini format
            content = self._messages_to_content(messages)
            
            # Configure generation parameters
            generation_config = self._configure_generation_config(**generation_kwargs)
            safety_settings = self._configure_safety_settings()
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=content,
                config=generation_config,
            )
            
            # Sleep to respect rate limits
            time.sleep(self.sleep_interval)
            
            # Extract response text
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                generated_text = response.candidates[0].content.parts[0].text
            else:
                generated_text = ""
            
            return {
                "text": generated_text,
                "finish_reason": getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None,
                "usage": {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0),
                } if hasattr(response, 'usage_metadata') and response.usage_metadata else None,
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "text": "",
                "finish_reason": "error",
                "error": str(e),
            }
    
    def batch_generate(
        self, 
        batch_messages: List[List[Dict[str, str]]], 
        logprobs: bool = False, 
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of message lists.
        
        Args:
            batch_messages: List of message lists
            logprobs: Whether to return log probabilities (not supported by Gemini)
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of response dictionaries
        """
        results = []
        
        for messages in batch_messages:
            result = self.generate(messages, logprobs, **generation_kwargs)
            results.append(result)
        
        return results
    
    def get_run_env(self) -> Dict[str, Any]:
        """
        Get runtime environment information for logging and debugging.
        
        Returns:
            Dictionary containing environment information
        """
        try:
            return {
                "google_genai_version": importlib_metadata.version("google-genai"),
                "google_cloud_aiplatform_version": importlib_metadata.version("google-cloud-aiplatform"),
                "model_name": self.model_name,
                "project_id": self.project_id,
                "location": self.location,
            }
        except Exception as e:
            return {
                "model_name": self.model_name,
                "project_id": self.project_id,
                "location": self.location,
                "error": f"Failed to get versions: {e}",
            }
    
    def parse_outputs(
        self, generated_outputs: List[Dict[str, Any]], conversations=None, tokenize_prompts=False
    ):
        """
        Parse generated outputs to extract text responses, errors, and tokenized prompts.
        
        Args:
            generated_outputs: List of generated output dictionaries
            conversations: List of conversation messages (not used for Gemini)
            tokenize_prompts: Whether to tokenize prompts (not supported for Gemini)
            
        Returns:
            Tuple of (responses, errors, tokenized_prompts)
        """
        responses = []
        errors = []
        tokenized_prompts = []

        for output in generated_outputs:
            # Check for errors
            if "error" in output:
                responses.append(None)
                errors.append(output["error"])
            else:
                text = output.get("text", "")
                responses.append(text)
                errors.append(None)

        # Tokenized prompts not supported for Gemini API
        if tokenize_prompts:
            logger.warning("Tokenized prompts not supported for Gemini API")

        return responses, errors, tokenized_prompts 