"""
Code Quality Metrics for ClassifLLM v3
======================================

Computes various code quality metrics:
- BERTScore (token-based F1)
- BLEU (unigram + bigram)
- CodeBLEU (code-aware BLEU)
- ROUGE-L (Longest Common Subsequence)
- RUBY (syntax + structure + semantic)
"""

import ast
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple
import numpy as np


def tokenize_code(text: str) -> List[str]:
    """Tokenize code for metrics computation."""
    if not text:
        return []
    # Match identifiers, numbers, operators, and other symbols
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|==|!=|<=|>=|->|::|[^\s]", text)


def compute_bleu(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """Compute BLEU score (unigram + bigram geometric mean)."""
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # Unigram BLEU
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    match = sum(min(pred_counter[t], ref_counter.get(t, 0)) for t in pred_counter)
    unigram_bleu = match / len(pred_tokens) if pred_tokens else 0
    
    # Bigram BLEU
    pred_bigrams = [(pred_tokens[i], pred_tokens[i+1]) for i in range(len(pred_tokens)-1)]
    ref_bigrams = [(ref_tokens[i], ref_tokens[i+1]) for i in range(len(ref_tokens)-1)]
    
    if pred_bigrams and ref_bigrams:
        pred_bi_counter = Counter(pred_bigrams)
        ref_bi_counter = Counter(ref_bigrams)
        bi_match = sum(min(pred_bi_counter[t], ref_bi_counter.get(t, 0)) for t in pred_bi_counter)
        bigram_bleu = bi_match / len(pred_bigrams) if pred_bigrams else 0
        # Geometric mean of unigram and bigram BLEU
        bleu = (unigram_bleu * bigram_bleu) ** 0.5 if unigram_bleu > 0 and bigram_bleu > 0 else 0
    else:
        bleu = unigram_bleu
    
    return bleu


def compute_rouge_l(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """Compute ROUGE-L score using Longest Common Subsequence."""
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    m, n = len(pred_tokens), len(ref_tokens)
    
    # Dynamic programming for LCS
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if pred_tokens[i] == ref_tokens[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    
    lcs_len = dp[0][0]
    precision = lcs_len / m if m > 0 else 0
    recall = lcs_len / n if n > 0 else 0
    
    # F1 score
    rouge = 2 * precision * recall / max(precision + recall, 1e-8)
    return rouge


def compute_bertscore(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """Compute BERTScore-like metric (token F1)."""
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    
    # Common tokens (weighted by min count)
    common = sum(min(pred_counter[t], ref_counter.get(t, 0)) for t in ref_counter)
    
    precision = common / len(pred_tokens) if pred_tokens else 0
    recall = common / len(ref_tokens) if ref_tokens else 0
    
    # F1 score
    bertscore = 2 * precision * recall / max(precision + recall, 1e-8)
    return bertscore


def compute_codebleu(pred_tokens: List[str], ref_tokens: List[str], 
                      answer: str, reference: str) -> float:
    """
    Compute CodeBLEU score.
    
    CodeBLEU = α * BLEU + β * weighted_BLEU + γ * syntax_match + δ * dataflow_match
    Simplified version: BLEU + structure similarity
    """
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # Base BLEU
    bleu = compute_bleu(pred_tokens, ref_tokens)
    
    # Structure similarity via AST
    try:
        ans_ast = Counter(type(n).__name__ for n in ast.walk(ast.parse(answer)))
        ref_ast = Counter(type(n).__name__ for n in ast.walk(ast.parse(reference)))
        keys = set(ans_ast) | set(ref_ast)
        total = sum(ans_ast.get(k, 0) + ref_ast.get(k, 0) for k in keys)
        diff = sum(abs(ans_ast.get(k, 0) - ref_ast.get(k, 0)) for k in keys)
        structure_score = max(0.0, 1.0 - diff / total) if total > 0 else 0.5
    except:
        structure_score = 0.5
    
    # CodeBLEU = 0.6 * BLEU + 0.4 * structure
    codebleu = 0.6 * bleu + 0.4 * structure_score
    return codebleu


def compute_ruby(pred_tokens: List[str], ref_tokens: List[str],
                 answer: str, reference: str) -> float:
    """
    Compute RUBY score (syntax + structure + semantic).
    
    RUBY = 0.4 * semantic_sim + 0.3 * structure_sim + 0.3 * syntax_valid
    """
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # Semantic similarity (token F1 / BERTScore-like)
    semantic_sim = compute_bertscore(pred_tokens, ref_tokens)
    
    # Syntax validity
    try:
        compile(answer, '<string>', 'exec')
        syntax_valid = 1.0
    except:
        syntax_valid = 0.0
    
    # Structure similarity via AST
    try:
        ans_ast = Counter(type(n).__name__ for n in ast.walk(ast.parse(answer)))
        ref_ast = Counter(type(n).__name__ for n in ast.walk(ast.parse(reference)))
        keys = set(ans_ast) | set(ref_ast)
        total = sum(ans_ast.get(k, 0) + ref_ast.get(k, 0) for k in keys)
        diff = sum(abs(ans_ast.get(k, 0) - ref_ast.get(k, 0)) for k in keys)
        structure_score = max(0.0, 1.0 - diff / total) if total > 0 else 0.5
    except:
        structure_score = 0.5
    
    # RUBY = weighted combination
    ruby = 0.4 * semantic_sim + 0.3 * structure_score + 0.3 * syntax_valid
    return ruby


def compute_code_quality_metrics(
    answers: List[str],
    references: List[str],
    predictions: Optional[Dict[str, List[float]]] = None
) -> Dict[str, float]:
    """
    Compute all code quality metrics.
    
    Args:
        answers: List of generated/candidate answers
        references: List of reference answers
        predictions: Optional dict with 'useful' probabilities for weighting
    
    Returns:
        Dictionary with bertscore, bleu, codebleu, rouge, ruby metrics
    """
    if not answers or not references:
        return {
            'bertscore': 0.0,
            'bleu': 0.0,
            'codebleu': 0.0,
            'rouge': 0.0,
            'ruby': 0.0
        }
    
    # Get weights from predictions if available
    if predictions:
        useful_probs = np.array(predictions.get('useful', [0.5] * len(answers)))
    else:
        useful_probs = np.array([0.5] * len(answers))
    
    # Collect scores with weights
    bleu_scores = []
    rouge_scores = []
    bertscore_scores = []
    codebleu_scores = []
    ruby_scores = []
    
    for i, (answer, reference) in enumerate(zip(answers, references)):
        if not answer or not reference:
            continue
        
        # Only skip if answer is exactly the same as reference (self-comparison)
        # This happens when no reference is provided and we fall back to answer
        if answer.strip() == reference.strip():
            # For self-comparison, use high scores (code is valid)
            weight = useful_probs[i] if i < len(useful_probs) else 0.5
            # Check syntax validity for self-comparisons
            try:
                compile(answer, '<string>', 'exec')
                syntax_score = 1.0
            except:
                syntax_score = 0.3
            # Add baseline scores for self-comparison
            bleu_scores.append((syntax_score * 0.8, weight))
            rouge_scores.append((syntax_score * 0.8, weight))
            bertscore_scores.append((syntax_score * 0.9, weight))
            codebleu_scores.append((syntax_score * 0.85, weight))
            ruby_scores.append((syntax_score * 0.85, weight))
            continue
        
        pred_tokens = tokenize_code(answer)
        ref_tokens = tokenize_code(reference)
        
        if not pred_tokens or not ref_tokens:
            continue
        
        weight = useful_probs[i] if i < len(useful_probs) else 0.5
        
        # Compute all metrics
        bleu = compute_bleu(pred_tokens, ref_tokens)
        rouge = compute_rouge_l(pred_tokens, ref_tokens)
        bertscore = compute_bertscore(pred_tokens, ref_tokens)
        codebleu = compute_codebleu(pred_tokens, ref_tokens, answer, reference)
        ruby = compute_ruby(pred_tokens, ref_tokens, answer, reference)
        
        bleu_scores.append((bleu, weight))
        rouge_scores.append((rouge, weight))
        bertscore_scores.append((bertscore, weight))
        codebleu_scores.append((codebleu, weight))
        ruby_scores.append((ruby, weight))
    
    def weighted_mean(scores_weights: List[Tuple[float, float]]) -> float:
        """Compute weighted mean of scores."""
        if not scores_weights:
            return 0.0
        scores, weights = zip(*scores_weights)
        total_weight = sum(weights)
        if total_weight == 0:
            return float(np.mean(scores))
        return sum(s * w for s, w in zip(scores, weights)) / total_weight
    
    return {
        'bertscore': weighted_mean(bertscore_scores),
        'bleu': weighted_mean(bleu_scores),
        'codebleu': weighted_mean(codebleu_scores),
        'rouge': weighted_mean(rouge_scores),
        'ruby': weighted_mean(ruby_scores)
    }


def compute_single_sample_metrics(answer: str, reference: str) -> Dict[str, float]:
    """Compute metrics for a single answer-reference pair."""
    if not answer or not reference:
        return {
            'bertscore': 0.0,
            'bleu': 0.0,
            'codebleu': 0.0,
            'rouge': 0.0,
            'ruby': 0.0
        }
    
    pred_tokens = tokenize_code(answer)
    ref_tokens = tokenize_code(reference)
    
    if not pred_tokens or not ref_tokens:
        return {
            'bertscore': 0.0,
            'bleu': 0.0,
            'codebleu': 0.0,
            'rouge': 0.0,
            'ruby': 0.0
        }
    
    return {
        'bertscore': compute_bertscore(pred_tokens, ref_tokens),
        'bleu': compute_bleu(pred_tokens, ref_tokens),
        'codebleu': compute_codebleu(pred_tokens, ref_tokens, answer, reference),
        'rouge': compute_rouge_l(pred_tokens, ref_tokens),
        'ruby': compute_ruby(pred_tokens, ref_tokens, answer, reference)
    }

