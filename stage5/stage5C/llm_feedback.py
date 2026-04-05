"""
LLM Feedback Generator for ClassifLLM v3
========================================

Uses real LLM API calls to generate human-like feedback for code quality assessment.
Supports multiple providers: OpenAI, Anthropic, local models via transformers.
"""

import json
import time
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM feedback generation."""
    provider: str = "openai"  # openai, anthropic, local, huggingface
    model_name: str = "gpt-4"  # or "claude-3-sonnet-20240229", "microsoft/codebert-base", etc.
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 5  # How many samples to process in parallel

@dataclass
class CodeQualityFeedback:
    """Structured feedback from LLM."""
    consistent: float  # -2 to +2 scale
    correct: float     # -2 to +2 scale
    useful: float      # -2 to +2 scale
    explanation: str   # Text explanation
    confidence: float  # 0-1 confidence score

class BaseLLMProvider:
    """Base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config

    async def generate_feedback(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None
    ) -> CodeQualityFeedback:
        """Generate feedback for a question-answer pair."""
        raise NotImplementedError

    def _parse_feedback_response(self, response: str) -> CodeQualityFeedback:
        """Parse LLM response into structured feedback."""
        try:
            # Try to parse JSON first
            data = json.loads(response.strip())

            # Normalize scores to -2 to +2 range
            def normalize_score(score: Union[int, float], min_val: int = -2, max_val: int = 2) -> float:
                if isinstance(score, (int, float)):
                    return max(min_val, min(max_val, float(score)))
                return 0.0

            return CodeQualityFeedback(
                consistent=normalize_score(data.get('consistent', 0)),
                correct=normalize_score(data.get('correct', 0)),
                useful=normalize_score(data.get('useful', 0)),
                explanation=data.get('explanation', 'No explanation provided'),
                confidence=min(1.0, max(0.0, float(data.get('confidence', 0.8))))
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"Failed to parse LLM response: {e}")
            # Return neutral feedback with low confidence - will be blended with human labels
            return CodeQualityFeedback(
                consistent=0.5,
                correct=0.5,
                useful=0.5,
                explanation="Parse failed - using neutral defaults",
                confidence=0.2  # Very low confidence so human labels are preferred
            )

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.api_base,
                timeout=config.timeout
            )
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")

    async def generate_feedback(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None
    ) -> CodeQualityFeedback:

        prompt = self._build_prompt(question, answer, context)

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout
                )

                content = response.choices[0].message.content
                return self._parse_feedback_response(content)

            except Exception as e:
                logger.warning(f"OpenAI API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise

    def _build_prompt(self, question: str, answer: str, context: Optional[str] = None) -> str:
        """Build the evaluation prompt for OpenAI models."""

        base_prompt = f"""You are an expert code reviewer evaluating the quality of programming answers.

Please analyze this question and answer pair:

QUESTION: {question}

ANSWER: {answer}

{context if context else ''}

Evaluate the answer on three criteria using a scale from -2 to +2:
- consistent: How well does the answer match what was asked? (-2 = completely irrelevant, +2 = perfectly on-topic)
- correct: Is the code technically correct and functional? (-2 = contains errors, +2 = perfect implementation)
- useful: How practical and helpful is this solution? (-2 = harmful/useless, +2 = highly valuable)

Also provide:
- explanation: A brief explanation of your evaluation (2-3 sentences)
- confidence: Your confidence in this assessment (0.0 to 1.0)

Respond with valid JSON in this exact format:
{{
    "consistent": <number>,
    "correct": <number>,
    "useful": <number>,
    "explanation": "<text>",
    "confidence": <number>
}}

Important: Use only the JSON format above. No additional text or markdown."""

        return base_prompt

class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Inference API provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from huggingface_hub import InferenceClient

            # Check if model is supported for inference
            supported_models = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                              'microsoft/DialoGPT-small', 'microsoft/DialoGPT-medium',
                              'microsoft/DialoGPT-large']

            if config.model_name not in supported_models:
                logger.warning(f"Model {config.model_name} may not be supported by free Inference API. "
                             f"Supported models include: {supported_models[:3]}...")

            self.client = InferenceClient(
                model=config.model_name,
                token=config.api_key
            )

            logger.info(f"Initialized HuggingFace Inference API for model: {config.model_name}")

        except ImportError:
            raise ImportError("huggingface_hub package not installed. Install with: pip install huggingface_hub")

    async def generate_feedback(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None
    ) -> CodeQualityFeedback:

        prompt = self._build_prompt(question, answer, context)

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"HuggingFace API call attempt {attempt + 1} for question: {question[:50]}...")
                # Use async call to HuggingFace Inference API with timeout
                response = await asyncio.wait_for(
                    self._async_inference_call(prompt),
                    timeout=self.config.timeout
                )
                logger.debug(f"HuggingFace API call successful on attempt {attempt + 1}")
                return self._parse_feedback_response(response)

            except asyncio.TimeoutError:
                logger.warning(f"HuggingFace API timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"HuggingFace API call failed (attempt {attempt + 1}): {type(e).__name__}: {e}")

            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                # Fallback to heuristic evaluation
                logger.warning("HuggingFace API failed after all retries, using heuristic fallback")
                return self._heuristic_feedback(question, answer)

    async def _async_inference_call(self, prompt: str) -> str:
        """Make async call to HuggingFace Inference API."""
        try:
            # Use asyncio.to_thread for Python 3.9+ or fallback to ThreadPoolExecutor
            if hasattr(asyncio, 'to_thread'):
                response = await asyncio.to_thread(self._sync_inference_call, prompt)
            else:
                import concurrent.futures
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    response = await loop.run_in_executor(executor, self._sync_inference_call, prompt)
            return response
        except StopIteration:
            # StopIteration cannot be raised into asyncio Futures (Python 3.7+)
            # Convert to RuntimeError to propagate properly
            logger.warning("StopIteration caught in async inference - converting to RuntimeError")
            raise RuntimeError("HuggingFace API returned empty iterator")
        except Exception as e:
            logger.error(f"Async inference call failed: {e}")
            raise

    def _sync_inference_call(self, prompt: str) -> str:
        """Synchronous call to HuggingFace Inference API."""
        try:
            logger.debug(f"Making HF API call for model {self.config.model_name}")

            response = self.client.text_generation(
                prompt,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=0.95,
                do_sample=True,
                return_full_text=False
            )

            logger.debug(f"HF API response type: {type(response)}")

            # Handle different response formats
            if isinstance(response, str):
                result = response
            elif isinstance(response, list) and len(response) > 0:
                result = str(response[0])
            elif hasattr(response, 'generated_text'):
                result = response.generated_text
            elif hasattr(response, '__iter__') and not isinstance(response, (str, list)):
                # Handle generator/iterator safely - convert to list first to avoid
                # StopIteration leaking through asyncio boundaries (Python 3.7+ issue)
                try:
                    items = list(response)  # Materialize the generator/iterator
                    result = str(items[0]) if items else ""
                except (TypeError, IndexError):
                    # Not iterable or empty
                    result = ""
            else:
                result = str(response)

            logger.debug(f"Processed response: {result[:100]}...")

            # Ensure we have some response
            if not result or not result.strip():
                logger.warning("Empty response from HuggingFace API")
                result = "No response generated"

            return result

        except StopIteration:
            # StopIteration cannot be raised into asyncio Futures (Python 3.7+)
            # Return empty string to trigger fallback handling
            logger.warning("HuggingFace API returned empty iterator (StopIteration)")
            return ""
        except Exception as e:
            logger.error(f"HuggingFace sync call failed: {type(e).__name__}: {e}")
            raise

    def _build_prompt(self, question: str, answer: str, context: Optional[str] = None) -> str:
        """Build prompt for HuggingFace models."""

        base_prompt = f"""You are an expert code reviewer evaluating the quality of programming answers.

Please analyze this question and answer pair:

QUESTION: {question}

ANSWER: {answer}

{context if context else ''}

Evaluate the answer on three criteria using a scale from -2 to +2:
- consistent: How well does the answer match what was asked? (-2 = completely irrelevant, +2 = perfectly on-topic)
- correct: Is the code technically correct and functional? (-2 = contains errors, +2 = perfect implementation)
- useful: How practical and helpful is this solution? (-2 = harmful/useless, +2 = highly valuable)

Also provide:
- explanation: A brief explanation of your evaluation (2-3 sentences)
- confidence: Your confidence in this assessment (0.0 to 1.0)

Respond with valid JSON in this exact format:
{{
    "consistent": <number>,
    "correct": <number>,
    "useful": <number>,
    "explanation": "<text>",
    "confidence": <number>
}}

Important: Use only the JSON format above. No additional text or markdown."""

        return base_prompt

    def _heuristic_feedback(self, question: str, answer: str) -> CodeQualityFeedback:
        """Generate feedback using simple heuristics when HuggingFace API fails."""

        # Simple heuristics for code quality assessment
        score = 0.0
        explanation_parts = []

        # Check answer length (reasonable code should be 10-200 chars)
        answer_len = len(answer.strip())
        if 10 <= answer_len <= 200:
            score += 0.5
            explanation_parts.append("appropriate length")
        elif answer_len < 5:
            score -= 0.5
            explanation_parts.append("too short")

        # Check for code-like elements
        code_indicators = ['def ', 'class ', 'import ', 'for ', 'if ', 'return ', '=', '(', ')']
        code_matches = sum(1 for indicator in code_indicators if indicator in answer)
        if code_matches >= 2:
            score += 0.5
            explanation_parts.append("contains code elements")

        # Check question-answer relevance (simple keyword matching)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        common_words = question_words.intersection(answer_words)
        if len(common_words) > 2:
            score += 0.3
            explanation_parts.append("relevant keywords")

        # Normalize to -2/+2 range
        final_score = max(-2.0, min(2.0, score))

        explanation = f"HuggingFace API failed, using heuristic evaluation: {'; '.join(explanation_parts) if explanation_parts else 'minimal code structure'}"

        return CodeQualityFeedback(
            consistent=final_score,
            correct=final_score,
            useful=final_score,
            explanation=explanation,
            confidence=0.4  # Lower confidence for API failure fallback
        )


class LocalTransformersProvider(BaseLLMProvider):
    """Local transformers model provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._torch = None  # Store torch module reference
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            self._torch = torch  # Save for later use

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading local model: {config.model_name} on {self.device}")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=True)
                # Don't use device_map="auto" - causes DLL issues on Windows
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    local_files_only=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            except Exception as _e:
                logger.warning(f"Cache load failed: {_e}, trying network")
                self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
                # Don't use device_map="auto" - causes DLL issues on Windows
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            self.model.to(self.device)

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except ImportError as e:
            logger.warning(f"Transformers not available: {e}. Using fallback.")
            self.model = None
            self.tokenizer = None
            self.device = "cpu"
        except Exception as e:
            logger.warning(f"Failed to load local model {config.model_name}: {e}. Using fallback.")
            self.model = None
            self.tokenizer = None
            self.device = "cpu"

    async def generate_feedback(
        self,
        question: str,
        answer: str,
        context: Optional[str] = None
    ) -> CodeQualityFeedback:

        # Fallback if model not loaded
        if self.model is None or self.tokenizer is None or self._torch is None:
            logger.warning("Local model not available, using heuristic fallback")
            return self._heuristic_feedback(question, answer)

        prompt = self._build_prompt(question, answer, context)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with self._torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Try to parse, but small models like DialoGPT usually can't output valid JSON
            if response.strip():
                try:
                    import json
                    # Try to find JSON in response
                    json_match = None
                    if '{' in response and '}' in response:
                        start = response.find('{')
                        end = response.rfind('}') + 1
                        json_match = response[start:end]
                    
                    if json_match:
                        data = json.loads(json_match)
                        if all(k in data for k in ['consistent', 'correct', 'useful']):
                            return CodeQualityFeedback(
                                consistent=float(data['consistent']),
                                correct=float(data['correct']),
                                useful=float(data['useful']),
                                explanation=data.get('explanation', 'Local model evaluation'),
                                confidence=min(1.0, max(0.0, float(data.get('confidence', 0.6))))
                            )
                except:
                    pass  # Fall through to heuristic
            
            # Local models usually can't generate structured feedback - use heuristic
            return self._heuristic_feedback(question, answer)

        except Exception as e:
            logger.debug(f"Local model inference failed: {e}")
            # Fallback to heuristic
            return self._heuristic_feedback(question, answer)

    def _heuristic_feedback(self, question: str, answer: str) -> CodeQualityFeedback:
        """Generate feedback using improved heuristics when LLM is unavailable."""
        import ast
        import re
        
        consistent_score = 0.0
        correct_score = 0.0
        useful_score = 0.0
        explanation_parts = []
        
        answer_stripped = answer.strip()
        answer_len = len(answer_stripped)
        
        # === SYNTAX CHECK (for correct) ===
        try:
            ast.parse(answer_stripped)
            correct_score += 1.0
            explanation_parts.append("valid syntax")
        except SyntaxError:
            correct_score -= 1.0
            explanation_parts.append("syntax error")
        except Exception:
            # Not valid Python at all
            correct_score -= 0.5
        
        # === LENGTH CHECK ===
        if answer_len < 5:
            useful_score -= 1.5
            correct_score -= 0.5
            explanation_parts.append("too short")
        elif answer_len < 20:
            useful_score -= 0.5
            explanation_parts.append("very brief")
        elif answer_len > 500:
            # Very long might be repetitive/broken
            useful_score -= 0.3
            explanation_parts.append("very long")
        
        # === REPETITION DETECTION (common LLM failure) ===
        lines = answer_stripped.split('\n')
        if len(lines) > 3:
            unique_lines = set(lines)
            repetition_ratio = len(unique_lines) / len(lines)
            if repetition_ratio < 0.3:
                correct_score -= 1.5
                useful_score -= 1.5
                explanation_parts.append("highly repetitive")
            elif repetition_ratio < 0.6:
                correct_score -= 0.5
                useful_score -= 0.5
                explanation_parts.append("some repetition")
        
        # === CODE STRUCTURE CHECK ===
        has_function = 'def ' in answer
        has_class = 'class ' in answer
        has_import = 'import ' in answer
        has_return = 'return ' in answer
        has_assignment = '=' in answer and '==' not in answer
        
        structure_score = sum([has_function, has_class, has_return, has_assignment])
        if structure_score >= 2:
            correct_score += 0.5
            useful_score += 0.3
            explanation_parts.append("good code structure")
        elif structure_score == 0 and answer_len > 20:
            correct_score -= 0.3
            explanation_parts.append("minimal structure")
        
        # === QUESTION-ANSWER RELEVANCE (for consistent) ===
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Extract meaningful words (length > 3, no common words)
        common_stopwords = {'the', 'and', 'for', 'that', 'this', 'with', 'from', 'are', 'was', 'were', 'been', 'have', 'has', 'how', 'what', 'when', 'where', 'which', 'who', 'will', 'would', 'could', 'should'}
        question_words = set(w for w in re.findall(r'\b\w+\b', question_lower) if len(w) > 3 and w not in common_stopwords)
        answer_words = set(w for w in re.findall(r'\b\w+\b', answer_lower) if len(w) > 3 and w not in common_stopwords)
        
        if question_words:
            overlap = len(question_words.intersection(answer_words))
            overlap_ratio = overlap / len(question_words)
            
            if overlap_ratio > 0.3:
                consistent_score += 1.0
                explanation_parts.append("highly relevant")
            elif overlap_ratio > 0.1:
                consistent_score += 0.3
                explanation_parts.append("somewhat relevant")
            else:
                consistent_score -= 0.5
                explanation_parts.append("low relevance")
        
        # === GARBAGE DETECTION ===
        # Check for nonsense patterns
        if re.search(r'(.)\1{10,}', answer):  # Repeated characters
            correct_score -= 2.0
            useful_score -= 2.0
            explanation_parts.append("garbage output")
        
        if 'error' in answer_lower and 'traceback' in answer_lower:
            correct_score -= 1.0
            explanation_parts.append("contains error")
        
        # === NORMALIZE SCORES to [-2, 2] ===
        consistent_score = max(-2.0, min(2.0, consistent_score))
        correct_score = max(-2.0, min(2.0, correct_score))
        useful_score = max(-2.0, min(2.0, useful_score))
        
        # Convert to 0-1 probability range for training
        consistent_prob = (consistent_score + 2) / 4.0
        correct_prob = (correct_score + 2) / 4.0
        useful_prob = (useful_score + 2) / 4.0
        
        explanation = f"Heuristic: {'; '.join(explanation_parts) if explanation_parts else 'evaluated'}"
        
        return CodeQualityFeedback(
            consistent=consistent_prob,
            correct=correct_prob,
            useful=useful_prob,
            explanation=explanation,
            confidence=0.4  # Lower confidence for heuristics
        )

    def _build_prompt(self, question: str, answer: str, context: Optional[str] = None) -> str:
        """Build prompt for local transformers models."""

        # Use a simpler prompt for smaller local models
        prompt = f"""Evaluate this code answer:

Question: {question}
Answer: {answer}

Rate from -2 to +2:
- consistent: How relevant is the answer? (-2=irrelevant, +2=perfect match)
- correct: Is the code correct? (-2=wrong, +2=perfect)
- useful: How useful is this? (-2=useless, +2=very helpful)

Format: {{"consistent": number, "correct": number, "useful": number, "explanation": "text", "confidence": number}}

Response:"""

        return prompt


class LLMFeedbackGenerator:
    """Main class for generating LLM feedback."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = self._create_provider(config)
        self.cache = {}  # Simple in-memory cache
        _default_cache = Path(__file__).resolve().parent / "llm_feedback_cache.json"
        if _default_cache.is_file():
            self.load_cache(str(_default_cache))

    def _create_provider(self, config: LLMConfig) -> BaseLLMProvider:
        """Create the appropriate provider based on config."""

        if config.provider == "openai":
            return OpenAIProvider(config)
        elif config.provider == "local":
            return LocalTransformersProvider(config)
        elif config.provider == "huggingface":
            return HuggingFaceProvider(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    async def generate_feedback_batch(
        self,
        samples: List[Dict[str, Any]],
        use_cache: bool = True,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate feedback for a batch of samples.

        Args:
            samples: List of dicts with 'question', 'answer', 'metadata' keys
            use_cache: Whether to use cached results
            show_progress: Whether to show progress

        Returns:
            Samples with added feedback data
        """

        import asyncio
        if show_progress:
            from tqdm.asyncio import tqdm
        else:
            def tqdm(x, **kwargs):
                return x

        async def process_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
            question = sample.get('question', '')
            answer = sample.get('answer', '')

            cache_key = hashlib.sha256(
                f"{question}\n{answer}".encode("utf-8", errors="replace")
            ).hexdigest()

            # Check cache
            if use_cache and cache_key in self.cache:
                feedback = self.cache[cache_key]
            else:
                try:
                    # Generate new feedback
                    feedback = await self.provider.generate_feedback(
                        question=question,
                        answer=answer,
                        context=sample.get('metadata', {}).get('context')
                    )

                    # Cache result
                    if use_cache:
                        self.cache[cache_key] = feedback

                except Exception as e:
                    # Handle any exception including StopIteration, timeouts, etc.
                    logger.warning(f"Feedback generation failed for sample: {type(e).__name__}: {e}")
                    # Use fallback heuristic feedback
                    from stage5.stage5C.llm_feedback import CodeQualityFeedback
                    feedback = CodeQualityFeedback(
                        consistent=0.0, correct=0.0, useful=0.0,
                        explanation=f"Feedback generation failed: {type(e).__name__}",
                        confidence=0.1  # Very low confidence
                    )

                    # Still cache failed results to avoid retries
                    if use_cache:
                        self.cache[cache_key] = feedback

            # Convert to the format expected by the training system
            sample_copy = sample.copy()
            
            # Check if feedback is already in 0-1 range (from improved heuristic)
            # or in -2/+2 range (from real LLM)
            def normalize_score(score):
                if -2 <= score <= 2 and (score < 0 or score > 1):
                    # Score is in -2/+2 range, convert to 0-1
                    return (score + 2) / 4
                else:
                    # Score is already in 0-1 range
                    return max(0.0, min(1.0, score))
            
            llm_labels = {
                'consistent': normalize_score(feedback.consistent),
                'correct': normalize_score(feedback.correct),
                'useful': normalize_score(feedback.useful)
            }
            
            # PRESERVE original human labels if they exist, blend with LLM feedback
            original_labels = sample.get('labels', {})
            if original_labels and feedback.confidence < 0.8:
                # If we have human labels and LLM confidence is low, prefer human labels
                confidence = feedback.confidence
                sample_copy['labels'] = {
                    'consistent': original_labels.get('consistent', llm_labels['consistent']) * (1 - confidence) + llm_labels['consistent'] * confidence,
                    'correct': original_labels.get('correct', llm_labels['correct']) * (1 - confidence) + llm_labels['correct'] * confidence,
                    'useful': original_labels.get('useful', llm_labels['useful']) * (1 - confidence) + llm_labels['useful'] * confidence
                }
            else:
                # High confidence LLM feedback or no original labels
                sample_copy['labels'] = llm_labels
            
            sample_copy['feedback'] = {
                'consistent_raw': feedback.consistent,
                'correct_raw': feedback.correct,
                'useful_raw': feedback.useful,
                'explanation': feedback.explanation,
                'confidence': feedback.confidence,
                'source': 'llm_' + self.config.provider
            }

            return sample_copy

        async def process_sample_fallback(sample: Dict[str, Any]) -> Dict[str, Any]:
            """Process sample with fallback heuristic feedback."""
            question = sample.get('question', '')
            answer = sample.get('answer', '')

            # Use heuristic feedback
            from stage5.stage5C.llm_feedback import CodeQualityFeedback
            feedback = CodeQualityFeedback(
                consistent=0.0, correct=0.0, useful=0.0,
                explanation="Batch processing failed, using heuristic fallback",
                confidence=0.1
            )

            # Convert to the format expected by the training system
            sample_copy = sample.copy()
            sample_copy['labels'] = {
                'consistent': (feedback.consistent + 2) / 4,
                'correct': (feedback.correct + 2) / 4,
                'useful': (feedback.useful + 2) / 4
            }
            sample_copy['feedback'] = {
                'consistent_raw': feedback.consistent,
                'correct_raw': feedback.correct,
                'useful_raw': feedback.useful,
                'explanation': feedback.explanation,
                'confidence': feedback.confidence,
                'source': 'fallback_heuristic'
            }

            return sample_copy

        # Process in batches to avoid overwhelming the API
        results = []
        batch_size = self.config.batch_size

        for i in tqdm(range(0, len(samples), batch_size), desc="Generating LLM feedback"):
            batch = samples[i:i + batch_size]

            # Process batch concurrently with error handling
            batch_tasks = [process_sample(sample) for sample in batch]
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                # Handle any exceptions that occurred
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Task {j} in batch failed: {type(result).__name__}: {result}")
                        # Create fallback result
                        sample = batch[j]
                        fallback_result = await process_sample_fallback(sample)
                        batch_results[j] = fallback_result

            except Exception as e:
                logger.error(f"Batch processing failed: {type(e).__name__}: {e}")
                # Fallback: process sequentially
                batch_results = []
                for sample in batch:
                    try:
                        result = await process_sample(sample)
                        batch_results.append(result)
                    except Exception as sample_e:
                        logger.error(f"Sample processing failed: {type(sample_e).__name__}: {sample_e}")
                        fallback_result = await process_sample_fallback(sample)
                        batch_results.append(fallback_result)

            results.extend(batch_results)

            # Small delay between batches to be respectful to API
            if i + batch_size < len(samples):
                await asyncio.sleep(0.5)

        if use_cache and results:
            _p = Path(__file__).resolve().parent / "llm_feedback_cache.json"
            try:
                self.save_cache(str(_p))
            except OSError as e:
                logger.warning("Could not persist LLM feedback cache: %s", e)

        return results

    def save_cache(self, filepath: str):
        """Save feedback cache to file."""
        cache_data = {}
        for key, feedback in self.cache.items():
            cache_data[key] = {
                'consistent': feedback.consistent,
                'correct': feedback.correct,
                'useful': feedback.useful,
                'explanation': feedback.explanation,
                'confidence': feedback.confidence
            }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved feedback cache to {filepath} ({len(cache_data)} entries)")

    def load_cache(self, filepath: str):
        """Load feedback cache from file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            self.cache = {}
            for key, data in cache_data.items():
                self.cache[key] = CodeQualityFeedback(
                    consistent=data['consistent'],
                    correct=data['correct'],
                    useful=data['useful'],
                    explanation=data['explanation'],
                    confidence=data.get('confidence', 0.8)
                )

            logger.info(f"Loaded feedback cache from {filepath} ({len(self.cache)} entries)")

        except FileNotFoundError:
            logger.warning(f"Cache file not found: {filepath}")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")

def create_default_config(provider: str = "openai") -> LLMConfig:
    """Create a default configuration for the specified provider."""

    if provider == "openai":
        return LLMConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            batch_size=3,  # Be respectful to OpenAI API limits
            max_retries=3,
            retry_delay=2.0
        )

    elif provider == "local":
        return LLMConfig(
            provider="local",
            model_name="microsoft/DialoGPT-medium",  # Smaller model for testing
            temperature=0.8,
            max_tokens=500,
            batch_size=2,
            max_retries=1
        )

    else:
        raise ValueError(f"No default config for provider: {provider}")

# Example usage
async def example_usage():
    """Example of how to use the LLM feedback generator."""

    # Create config (replace with your API key)
    config = LLMConfig(
        provider="openai",
        model_name="gpt-4",
        api_key="YOUR_API_KEY_HERE",  # Replace with actual key
        temperature=0.7,
        batch_size=2
    )

    # Create generator
    generator = LLMFeedbackGenerator(config)

    # Example samples
    samples = [
        {
            'question': 'How to sort a list in Python?',
            'answer': 'sorted_list = sorted(my_list)',
            'metadata': {'source': 'test'}
        },
        {
            'question': 'How to read a file?',
            'answer': 'with open("file.txt", "r") as f: content = f.read()',
            'metadata': {'source': 'test'}
        }
    ]

    # Generate feedback
    results = await generator.generate_feedback_batch(samples, show_progress=True)

    # Print results
    for i, result in enumerate(results):
        print(f"\nSample {i+1}:")
        print(f"Question: {result['question'][:50]}...")
        print(f"Labels: {result['labels']}")
        print(f"Feedback: {result['feedback']['explanation'][:100]}...")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
