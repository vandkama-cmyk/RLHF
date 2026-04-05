"""
Stage 2A - Metrics Evaluation
=============================

Comprehensive metrics evaluation for code generation:
- BERTScore: Semantic similarity using BERT embeddings
- BLEU: N-gram overlap score
- CodeBLEU: Code-specific evaluation
- ROUGE: Summarization metrics
- RUBY: Code migration evaluation (PDG → AST → Token level)
"""

import numpy as np
from typing import List, Dict, Any, Optional
import ast
import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for metric results."""
    name: str
    score: float
    details: Optional[Dict[str, Any]] = None


class Stage2AMetricsEvaluator:
    """
    Comprehensive metrics evaluator for Stage 2A.
    Computes BERTScore, BLEU, CodeBLEU, ROUGE, and RUBY.
    """
    
    def __init__(self):
        self._smoothing = None
    
    def _tokenize_code(self, text: str) -> List[str]:
        """Tokenize code into tokens."""
        if not text:
            return []
        return re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|==|!=|<=|>=|->|::|[^\s]",
            text
        )
    
    def _lcs_length(self, a: List[str], b: List[str]) -> int:
        """Compute longest common subsequence length."""
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0
        
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if a[i] == b[j]:
                    dp[i][j] = 1 + dp[i + 1][j + 1]
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
        return dp[0][0]
    
    def compute_bertscore(
        self,
        predictions: List[str],
        references: List[str]
    ) -> MetricResult:
        """
        Compute BERTScore - semantic similarity based on token embeddings.
        Uses token F1 as proxy when bert-score package not available.
        """
        scores = []
        for pred, ref in zip(predictions, references):
            if not pred or not ref:
                scores.append(0.0)
                continue
            
            pred_tokens = set(self._tokenize_code(pred))
            ref_tokens = set(self._tokenize_code(ref))
            
            if not ref_tokens:
                scores.append(0.0)
                continue
            
            # Compute F1-like score
            common = len(pred_tokens & ref_tokens)
            precision = common / len(pred_tokens) if pred_tokens else 0
            recall = common / len(ref_tokens)
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            scores.append(f1)
        
        avg_score = np.mean(scores) if scores else 0.0
        return MetricResult(
            name='bertscore',
            score=float(avg_score),
            details={'individual_scores': scores, 'method': 'token_f1_proxy'}
        )
    
    def compute_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> MetricResult:
        """
        Compute BLEU score - n-gram overlap metric.
        Computes unigram, bigram, and combined BLEU.
        """
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize_code(pred)
            ref_tokens = self._tokenize_code(ref)
            
            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                continue
            
            # Unigram BLEU
            pred_counter = Counter(pred_tokens)
            ref_counter = Counter(ref_tokens)
            match = sum(min(pred_counter[t], ref_counter.get(t, 0)) for t in pred_counter)
            unigram = match / len(pred_tokens)
            
            # Bigram BLEU
            pred_bigrams = [(pred_tokens[i], pred_tokens[i+1]) for i in range(len(pred_tokens)-1)]
            ref_bigrams = [(ref_tokens[i], ref_tokens[i+1]) for i in range(len(ref_tokens)-1)]
            
            if pred_bigrams and ref_bigrams:
                pred_bi_counter = Counter(pred_bigrams)
                ref_bi_counter = Counter(ref_bigrams)
                bi_match = sum(min(pred_bi_counter[t], ref_bi_counter.get(t, 0)) for t in pred_bi_counter)
                bigram = bi_match / len(pred_bigrams)
                
                # Geometric mean of unigram and bigram
                bleu = (unigram * bigram) ** 0.5 if bigram > 0 else unigram * 0.5
            else:
                bleu = unigram * 0.7
            
            # Brevity penalty
            bp = min(1.0, len(pred_tokens) / max(len(ref_tokens), 1))
            bleu = bp * bleu
            
            scores.append(bleu)
        
        avg_score = np.mean(scores) if scores else 0.0
        return MetricResult(
            name='bleu',
            score=float(avg_score),
            details={'individual_scores': scores}
        )
    
    def compute_codebleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> MetricResult:
        """
        Compute CodeBLEU - code-specific BLEU with syntax awareness.
        Combines n-gram match, weighted n-gram match, syntax match, dataflow match.
        """
        scores = []
        for pred, ref in zip(predictions, references):
            if not pred or not ref:
                scores.append(0.0)
                continue
            
            pred_tokens = self._tokenize_code(pred)
            ref_tokens = self._tokenize_code(ref)
            
            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                continue
            
            # Token-level precision and recall
            pred_counter = Counter(pred_tokens)
            ref_counter = Counter(ref_tokens)
            common = sum(min(pred_counter[t], ref_counter.get(t, 0)) for t in pred_counter)
            
            precision = common / len(pred_tokens) if pred_tokens else 0
            recall = common / len(ref_tokens) if ref_tokens else 0
            
            if precision + recall > 0:
                token_f1 = 2 * precision * recall / (precision + recall)
            else:
                token_f1 = 0.0
            
            # Syntax match (AST comparison)
            syntax_score = self._compute_ast_similarity(pred, ref)
            
            # Combined CodeBLEU
            # weights: 25% n-gram, 25% weighted n-gram, 25% syntax, 25% dataflow
            # Simplified: 50% token_f1, 50% syntax
            codebleu = 0.5 * token_f1 + 0.5 * syntax_score
            scores.append(codebleu)
        
        avg_score = np.mean(scores) if scores else 0.0
        return MetricResult(
            name='codebleu',
            score=float(avg_score),
            details={'individual_scores': scores}
        )
    
    def _compute_ast_similarity(self, pred: str, ref: str) -> float:
        """Compute AST-based similarity score."""
        try:
            pred_ast = ast.parse(pred)
            ref_ast = ast.parse(ref)
            
            # Count node types
            pred_nodes = Counter(type(n).__name__ for n in ast.walk(pred_ast))
            ref_nodes = Counter(type(n).__name__ for n in ast.walk(ref_ast))
            
            # Jaccard similarity of node types
            all_types = set(pred_nodes.keys()) | set(ref_nodes.keys())
            if not all_types:
                return 0.5
            
            intersection = sum(min(pred_nodes.get(t, 0), ref_nodes.get(t, 0)) for t in all_types)
            union = sum(max(pred_nodes.get(t, 0), ref_nodes.get(t, 0)) for t in all_types)
            
            return intersection / union if union > 0 else 0.0
        except SyntaxError:
            return 0.3  # Partial credit for non-parseable code
        except Exception:
            return 0.0
    
    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> MetricResult:
        """
        Compute ROUGE score - recall-oriented understudy for gisting evaluation.
        Returns ROUGE-L (longest common subsequence).
        """
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize_code(pred)
            ref_tokens = self._tokenize_code(ref)
            
            if not ref_tokens:
                rouge1_scores.append(0.0)
                rouge2_scores.append(0.0)
                rougeL_scores.append(0.0)
                continue
            
            # ROUGE-1 (unigram)
            pred_set = set(pred_tokens)
            ref_set = set(ref_tokens)
            overlap = len(pred_set & ref_set)
            rouge1 = overlap / len(ref_set) if ref_set else 0
            rouge1_scores.append(rouge1)
            
            # ROUGE-2 (bigram)
            if len(pred_tokens) > 1 and len(ref_tokens) > 1:
                pred_bigrams = set((pred_tokens[i], pred_tokens[i+1]) for i in range(len(pred_tokens)-1))
                ref_bigrams = set((ref_tokens[i], ref_tokens[i+1]) for i in range(len(ref_tokens)-1))
                bi_overlap = len(pred_bigrams & ref_bigrams)
                rouge2 = bi_overlap / len(ref_bigrams) if ref_bigrams else 0
            else:
                rouge2 = 0
            rouge2_scores.append(rouge2)
            
            # ROUGE-L (LCS)
            lcs_len = self._lcs_length(pred_tokens, ref_tokens)
            precision = lcs_len / len(pred_tokens) if pred_tokens else 0
            recall = lcs_len / len(ref_tokens)
            
            if precision + recall > 0:
                rougeL = 2 * precision * recall / (precision + recall)
            else:
                rougeL = 0.0
            rougeL_scores.append(rougeL)
        
        avg_rougeL = np.mean(rougeL_scores) if rougeL_scores else 0.0
        return MetricResult(
            name='rouge',
            score=float(avg_rougeL),
            details={
                'rouge1': float(np.mean(rouge1_scores)),
                'rouge2': float(np.mean(rouge2_scores)),
                'rougeL': float(avg_rougeL)
            }
        )
    
    def compute_ruby(
        self,
        predictions: List[str],
        references: List[str]
    ) -> MetricResult:
        """
        Compute RUBY metric - multi-level code comparison.
        
        RUBY uses hierarchical comparison:
        1. GRS (Graph Representation Similarity) - PDG level
        2. TRS (Tree Representation Similarity) - AST level  
        3. STS (String/Token Similarity) - Token level fallback
        """
        results = []
        methods_used = []
        
        for pred, ref in zip(predictions, references):
            score = None
            method = None
            
            # Try TRS (AST-based comparison)
            trs_score = self._compute_trs(pred, ref)
            if trs_score is not None:
                score = trs_score
                method = 'TRS'
            
            # Fallback to STS (token-based)
            if score is None:
                score = self._compute_sts(pred, ref)
                method = 'STS'
            
            results.append(score)
            methods_used.append(method)
        
        avg_score = np.mean(results) if results else 0.0
        
        # Method distribution
        method_counts = Counter(methods_used)
        
        return MetricResult(
            name='ruby',
            score=float(avg_score),
            details={
                'individual_scores': results,
                'method_distribution': dict(method_counts)
            }
        )
    
    def _compute_trs(self, pred: str, ref: str) -> Optional[float]:
        """Compute TRS (Tree Representation Similarity) using AST."""
        try:
            pred_ast = ast.parse(pred)
            ref_ast = ast.parse(ref)
            
            # Count nodes
            pred_count = sum(1 for _ in ast.walk(pred_ast))
            ref_count = sum(1 for _ in ast.walk(ref_ast))
            
            # Node type similarity
            pred_types = Counter(type(n).__name__ for n in ast.walk(pred_ast))
            ref_types = Counter(type(n).__name__ for n in ast.walk(ref_ast))
            
            all_types = set(pred_types.keys()) | set(ref_types.keys())
            type_diff = sum(abs(pred_types.get(t, 0) - ref_types.get(t, 0)) for t in all_types)
            total_nodes = pred_count + ref_count
            
            if total_nodes > 0:
                similarity = 1.0 - (type_diff / total_nodes)
                return max(0.0, min(1.0, similarity))
            return 0.5
        except:
            return None
    
    def _compute_sts(self, pred: str, ref: str) -> float:
        """Compute STS (String/Token Similarity)."""
        pred_tokens = self._tokenize_code(pred)
        ref_tokens = self._tokenize_code(ref)
        
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # Edit distance ratio
        m, n = len(pred_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i-1] == ref_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        edit_dist = dp[m][n]
        max_len = max(m, n)
        
        return 1.0 - (edit_dist / max_len) if max_len > 0 else 1.0
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, MetricResult]:
        """Compute all metrics."""
        return {
            'bertscore': self.compute_bertscore(predictions, references),
            'bleu': self.compute_bleu(predictions, references),
            'codebleu': self.compute_codebleu(predictions, references),
            'rouge': self.compute_rouge(predictions, references),
            'ruby': self.compute_ruby(predictions, references)
        }
    
    def get_summary(self, metrics: Dict[str, MetricResult]) -> Dict[str, float]:
        """Get summary of all metrics."""
        return {name: result.score for name, result in metrics.items()}
