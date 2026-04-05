"""
Stage 2B - GPT-2 Feedback Generator
===================================

Uses GPT-2 to generate synthetic feedback scores (Consistency, Agreement, Usefulness)
for code question-answer pairs.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import re


class GPT2FeedbackGenerator(nn.Module):
    """
    GPT-2 based feedback generator for code quality assessment.
    
    Generates scores for:
    - Consistency: Does the answer logically follow from the question?
    - Agreement: Does the answer align with expected coding practices?
    - Usefulness: Is the answer practically useful for the given task?
    """
    
    PROMPT_TEMPLATE = """Evaluate the following code answer for a programming question.

Question: {question}

Code Answer: {answer}

Rate the answer on three criteria (1-5 scale):
1. Consistency - Does the code logically address the question?
2. Agreement - Does the code follow good programming practices?
3. Usefulness - Is the code practically useful?

Scores:
Consistency:"""

    SCORE_PATTERNS = {
        'consistency': r'[Cc]onsistency[:\s]*(\d)',
        'agreement': r'[Aa]greement[:\s]*(\d)',
        'usefulness': r'[Uu]sefulness[:\s]*(\d)',
    }
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cuda",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        
        print(f"[GPT2FeedbackGenerator] Loading model: {model_name}")
        
        self._model_available = False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
            self._model_available = True
        except Exception as e:
            print(f"[GPT2FeedbackGenerator] Error loading {model_name}: {e}")
            # Fallback to gpt2 local cache
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
                self.model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True)
                self._model_available = True
            except Exception as e2:
                print(f"[GPT2FeedbackGenerator] GPT-2 not in cache: {e2}")
                print("[GPT2FeedbackGenerator] Using text-feature fallback scorer (no GPT-2)")
                self.tokenizer = None
                self.model = None
        
        if self._model_available:
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id

            self.model.to(device)
            self.model.eval()

            # Initialize score predictor head for faster inference
            self.score_predictor = ScorePredictor(
                hidden_size=self.model.config.n_embd,
                num_criteria=3
            ).to(device)

            print(f"[GPT2FeedbackGenerator] Model loaded successfully")
            print(f"[GPT2FeedbackGenerator] Model size: {self._count_parameters()} parameters")
        else:
            self.score_predictor = None
            print("[GPT2FeedbackGenerator] Running in text-feature fallback mode")
    
    def _count_parameters(self) -> str:
        """Count model parameters."""
        params = sum(p.numel() for p in self.model.parameters())
        return f"{params / 1e6:.1f}M"
    
    def generate_feedback_text(
        self,
        question: str,
        answer: str,
        max_new_tokens: int = 100
    ) -> str:
        """Generate feedback text using GPT-2."""
        prompt = self.PROMPT_TEMPLATE.format(question=question, answer=answer)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - max_new_tokens
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=1
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        feedback = generated[len(prompt):]
        
        return feedback
    
    def parse_scores_from_text(self, text: str) -> Dict[str, float]:
        """Parse scores from generated text."""
        scores = {}
        
        for criterion, pattern in self.SCORE_PATTERNS.items():
            match = re.search(pattern, text)
            if match:
                score = int(match.group(1))
                scores[criterion] = min(max(score, 1), 5) / 5.0  # Normalize to 0-1
            else:
                scores[criterion] = 0.5  # Default
        
        return scores
    
    def _text_feature_scores(
        self,
        questions: List[str],
        answers: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Fallback scorer using simple text features when GPT-2 is unavailable."""
        batch_size = len(questions)
        consistency = []
        agreement = []
        usefulness = []
        for q, a in zip(questions, answers):
            q_tokens = set(q.lower().split())
            a_tokens = set(a.lower().split())
            overlap = len(q_tokens & a_tokens) / max(len(q_tokens), 1)
            code_tokens = sum(1 for t in a.split() if any(c in t for c in ['(', ')', '=', '.', ':', '[', ']']))
            code_ratio = min(code_tokens / max(len(a.split()), 1), 1.0)
            len_ratio = min(len(a.split()) / max(len(q.split()), 1), 3.0) / 3.0
            consistency.append(float(np.clip(overlap + 0.3, 0.0, 1.0)))
            agreement.append(float(np.clip(code_ratio + 0.2, 0.0, 1.0)))
            usefulness.append(float(np.clip(len_ratio * 0.5 + 0.25, 0.0, 1.0)))
        return {
            'consistency': torch.tensor(consistency, device=self.device),
            'agreement': torch.tensor(agreement, device=self.device),
            'usefulness': torch.tensor(usefulness, device=self.device),
        }

    def generate_scores_fast(
        self,
        questions: List[str],
        answers: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Fast score generation using the score predictor head.
        More efficient for batch processing during training.
        Falls back to text-feature scoring if GPT-2 is unavailable.
        """
        if not self._model_available:
            return self._text_feature_scores(questions, answers)

        batch_size = len(questions)

        # Combine question and answer
        combined_texts = [
            f"Question: {q}\nAnswer: {a}"
            for q, a in zip(questions, answers)
        ]

        # Tokenize
        inputs = self.tokenizer(
            combined_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.no_grad():
            # Get GPT-2 hidden states
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )

            # Use last hidden state
            hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden]

            # Pool: use last token (like GPT-2 classification)
            pooled = hidden_states[:, -1, :]  # [batch, hidden]

            # Predict scores
            scores = self.score_predictor(pooled)  # [batch, 3]

        return {
            'consistency': scores[:, 0],
            'agreement': scores[:, 1],
            'usefulness': scores[:, 2],
        }
    
    def generate_feedback_batch(
        self,
        questions: List[str],
        answers: List[str],
        use_fast: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate feedback scores for a batch of Q-A pairs.
        
        Args:
            questions: List of questions
            answers: List of answers
            use_fast: Use fast score predictor (default) or text generation
        
        Returns:
            Dictionary with tensors for consistency, agreement, usefulness
        """
        if use_fast:
            return self.generate_scores_fast(questions, answers)
        
        # Slower text-based generation
        all_scores = {
            'consistency': [],
            'agreement': [],
            'usefulness': []
        }
        
        for q, a in zip(questions, answers):
            text = self.generate_feedback_text(q, a)
            scores = self.parse_scores_from_text(text)
            
            for key in all_scores:
                all_scores[key].append(scores.get(key, 0.5))
        
        return {
            k: torch.tensor(v, device=self.device)
            for k, v in all_scores.items()
        }
    
    def estimate_code_quality(
        self,
        question: str,
        answer: str
    ) -> Dict[str, float]:
        """
        Estimate code quality using heuristics + model.
        Provides additional quality signals beyond GPT-2 feedback.
        """
        quality = {
            'has_code': False,
            'syntax_valid': False,
            'length_appropriate': False,
            'has_keywords': False,
        }
        
        # Check for code markers
        code_markers = ['def ', 'class ', 'import ', '=', '(', ')', '[', ']', '{', '}']
        quality['has_code'] = any(m in answer for m in code_markers)
        
        # Basic syntax check
        try:
            compile(answer, '<string>', 'exec')
            quality['syntax_valid'] = True
        except:
            # Try as expression
            try:
                compile(answer, '<string>', 'eval')
                quality['syntax_valid'] = True
            except:
                quality['syntax_valid'] = False
        
        # Length check
        quality['length_appropriate'] = 10 <= len(answer) <= 500
        
        # Python keywords
        python_keywords = ['def', 'return', 'if', 'for', 'while', 'import', 'from', 'class', 'try', 'except']
        quality['has_keywords'] = any(kw in answer.lower() for kw in python_keywords)
        
        # Calculate overall quality score
        quality_score = sum(quality.values()) / len(quality)
        
        return {
            **quality,
            'quality_score': quality_score
        }
    
    def forward(
        self,
        questions: List[str],
        answers: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        return self.generate_feedback_batch(questions, answers, use_fast=True)


class ScorePredictor(nn.Module):
    """
    Fast score predictor head for GPT-2.
    Predicts Consistency, Agreement, Usefulness scores from hidden states.
    """
    
    def __init__(self, hidden_size: int = 768, num_criteria: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_criteria),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict scores from hidden states.
        
        Args:
            hidden_states: [batch, hidden_size]
        
        Returns:
            scores: [batch, num_criteria] in range [0, 1]
        """
        return self.predictor(hidden_states)


class FeedbackLoss(nn.Module):
    """
    Loss function for training with GPT-2 feedback.
    """
    
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        predicted_scores: Dict[str, torch.Tensor],
        feedback_scores: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute loss between predicted and feedback scores.
        
        Args:
            predicted_scores: Model predictions
            feedback_scores: GPT-2 generated feedback
            weights: Optional weights for each criterion
        
        Returns:
            Total loss
        """
        if weights is None:
            weights = {'consistency': 1.0, 'agreement': 1.0, 'usefulness': 1.0}
        
        total_loss = 0.0
        
        for criterion in ['consistency', 'agreement', 'usefulness']:
            if criterion in predicted_scores and criterion in feedback_scores:
                pred = predicted_scores[criterion]
                target = feedback_scores[criterion]
                
                # Apply label smoothing
                if self.label_smoothing > 0:
                    target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
                
                # MSE loss
                loss = F.mse_loss(pred, target)
                total_loss += weights.get(criterion, 1.0) * loss
        
        return total_loss
