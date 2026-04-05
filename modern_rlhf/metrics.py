"""
Modern Evaluation Metrics for Code Generation
============================================

Comprehensive evaluation metrics for code generation tasks including:
- BERTScore for semantic similarity
- CodeBLEU for code-specific evaluation
- BLEU for n-gram overlap
- ROUGE for summarization metrics
- RUBY metric for code migration evaluation (multi-level comparison: PDG → AST → Token)
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass
import re
import ast
import subprocess
import tempfile
import os
from pathlib import Path

# Project root (parent of ``modern_rlhf/``) for resolving bundled data paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Import evaluation libraries
try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logging.warning("BERTScore not available. Install with: pip install bert-score")

try:
    from codebleu import calc_codebleu
    CODEBLEU_AVAILABLE = True
except ImportError:
    CODEBLEU_AVAILABLE = False
    logging.warning("CodeBLEU not available. Install with: pip install codebleu")

try:
    from codegen_metrics.metrics import ruby as codegen_ruby_pair
    CODEGEN_RUBY_AVAILABLE = True
except ImportError:
    CODEGEN_RUBY_AVAILABLE = False
    logging.warning(
        "codegen-metrics RUBY not available. Install with: pip install codegen-metrics "
        "(requires Python <3.13 and dependencies; see package README on Windows)."
    )

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("ROUGE not available. Install with: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    logging.warning("BLEU not available. Install with: pip install nltk")

logger = logging.getLogger(__name__)


def get_metric_float(
    metrics: Dict[str, "MetricResult"],
    *key_options: str,
    default: float = float("nan"),
) -> float:
    """Return ``.score`` for the first present key in ``key_options`` (uniform CodeBLEU / RUBY aliases)."""
    for k in key_options:
        if k in metrics and metrics[k] is not None:
            s = metrics[k].score
            if s is None:
                continue
            return float(s)
    return default


@dataclass
class MetricResult:
    """Container for metric evaluation results."""
    metric_name: str
    score: float
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class CodeQualityAnalyzer:
    """Analyzer for code quality metrics."""
    
    def __init__(self):
        self.smoothing = SmoothingFunction().method1 if BLEU_AVAILABLE else None
    
    def analyze_syntax(self, code: str) -> Dict[str, Any]:
        """Analyze syntax correctness of code."""
        try:
            # Try to parse the code
            ast.parse(code)
            return {
                "syntax_correct": True,
                "syntax_error": None,
                "syntax_score": 1.0
            }
        except SyntaxError as e:
            return {
                "syntax_correct": False,
                "syntax_error": str(e),
                "syntax_score": 0.0
            }
        except Exception as e:
            return {
                "syntax_correct": False,
                "syntax_error": f"Parse error: {str(e)}",
                "syntax_score": 0.0
            }
    
    def analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity metrics."""
        try:
            tree = ast.parse(code)
            
            # Count different constructs
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            loops = len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))])
            conditionals = len([node for node in ast.walk(tree) if isinstance(node, ast.If)])
            
            # Calculate complexity score (simplified)
            complexity_score = min(1.0, max(0.0, 1.0 - (functions + classes + loops + conditionals) / 20.0))
            
            return {
                "functions": functions,
                "classes": classes,
                "loops": loops,
                "conditionals": conditionals,
                "complexity_score": complexity_score
            }
        except Exception as e:
            return {
                "functions": 0,
                "classes": 0,
                "loops": 0,
                "conditionals": 0,
                "complexity_score": 0.0,
                "error": str(e)
            }
    
    def analyze_style(self, code: str) -> Dict[str, Any]:
        """Analyze code style metrics."""
        lines = code.split('\n')
        
        # Basic style metrics
        avg_line_length = np.mean([len(line) for line in lines if line.strip()])
        long_lines = sum(1 for line in lines if len(line) > 80)
        empty_lines = sum(1 for line in lines if not line.strip())
        
        # Style score (simplified)
        style_score = 1.0
        if avg_line_length > 100:
            style_score -= 0.2
        if long_lines / len(lines) > 0.1:
            style_score -= 0.2
        if empty_lines / len(lines) > 0.3:
            style_score -= 0.1
        
        return {
            "avg_line_length": avg_line_length,
            "long_lines": long_lines,
            "empty_lines": empty_lines,
            "style_score": max(0.0, style_score)
        }


class ModernMetricsEvaluator:
    """Modern metrics evaluator for code generation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.code_analyzer = CodeQualityAnalyzer()
        self.rouge_scorer = None
        
        # Initialize ROUGE scorer if available
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def compute_bertscore(self, predictions: List[str], references: List[str]) -> MetricResult:
        """Compute BERTScore for semantic similarity."""
        # Fallback to a simple token-overlap based proxy if bert-score package is unavailable
        if not BERTSCORE_AVAILABLE:
            try:
                scores = []
                for pred, ref in zip(predictions, references):
                    if not pred or not ref:
                        scores.append(0.0)
                        continue
                    pred_tokens = set(pred.split())
                    ref_tokens = set(ref.split())
                    if not ref_tokens:
                        scores.append(0.0)
                        continue
                    overlap = len(pred_tokens & ref_tokens) / max(1, len(ref_tokens))
                    scores.append(overlap)
                score = float(np.mean(scores)) if scores else 0.0
                return MetricResult(metric_name="bertscore", score=score, details={"method": "token_overlap_proxy"})
            except Exception as e:
                return MetricResult(metric_name="bertscore", score=0.0, error=str(e))
        
        try:
            P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
            
            # Return F1 score (harmonic mean of precision and recall)
            score = float(F1.mean())
            
            return MetricResult(
                metric_name="bertscore",
                score=score,
                details={
                    "precision": float(P.mean()),
                    "recall": float(R.mean()),
                    "f1": score
                }
            )
        except Exception as e:
            return MetricResult(
                metric_name="bertscore",
                score=0.0,
                error=str(e)
            )
    
    def compute_codebleu(self, predictions: List[str], references: List[str]) -> MetricResult:
        """Compute CodeBLEU using the official ``codebleu.calc_codebleu`` implementation when available."""
        if CODEBLEU_AVAILABLE:
            try:
                refs_wrapped: List[List[str]] = [[r] for r in references]
                out = calc_codebleu(refs_wrapped, predictions, lang="python")
                score = float(out.get("codebleu", 0.0))
                return MetricResult(
                    metric_name="codebleu",
                    score=score,
                    details={"method": "codebleu.calc_codebleu", **out},
                )
            except Exception as e:
                logger.warning("calc_codebleu failed (%s); falling back to token F1 proxy", e)

        try:
            scores = []
            for pred, ref in zip(predictions, references):
                if not pred or not ref:
                    scores.append(0.0)
                    continue
                p_tokens = pred.split()
                r_tokens = ref.split()
                if not p_tokens or not r_tokens:
                    scores.append(0.0)
                    continue
                common = len([t for t in p_tokens if t in r_tokens])
                prec = common / len(p_tokens) if len(p_tokens) > 0 else 0.0
                rec = common / len(r_tokens) if len(r_tokens) > 0 else 0.0
                if prec + rec > 0:
                    f1 = 2 * prec * rec / (prec + rec)
                else:
                    f1 = 0.0
                scores.append(f1)

            score = float(np.mean(scores)) if scores else 0.0
            return MetricResult(
                metric_name="codebleu_proxy",
                score=score,
                details={"method": "token_f1_proxy", "individual_scores": scores},
            )
        except Exception as e:
            return MetricResult(metric_name="codebleu_proxy", score=0.0, error=str(e))
    
    def compute_bleu(self, predictions: List[str], references: List[str]) -> MetricResult:
        """Compute BLEU score for n-gram overlap."""
        # Fallback simple unigram-precision BLEU if nltk not available
        if not BLEU_AVAILABLE:
            try:
                results = []
                for pred, ref in zip(predictions, references):
                    pred_tokens = pred.split()
                    ref_tokens = ref.split()
                    if len(pred_tokens) == 0:
                        results.append(0.0)
                        continue
                    match = sum(1 for t in pred_tokens if t in ref_tokens)
                    score = match / max(1, len(pred_tokens))
                    results.append(score)
                score = float(np.mean(results)) if results else 0.0
                return MetricResult(metric_name="bleu", score=score, details={"method": "unigram_precision"})
            except Exception as e:
                return MetricResult(metric_name="bleu", score=0.0, error=str(e))
        
        try:
            results = []
            for pred, ref in zip(predictions, references):
                # Tokenize (simple whitespace tokenization)
                pred_tokens = pred.split()
                ref_tokens = ref.split()
                
                if len(pred_tokens) == 0:
                    results.append(0.0)
                    continue
                
                # Compute BLEU score
                score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.code_analyzer.smoothing)
                results.append(score)
            
            score = np.mean(results)
            
            return MetricResult(
                metric_name="bleu",
                score=score,
                details={"individual_scores": results}
            )
        except Exception as e:
            return MetricResult(
                metric_name="bleu",
                score=0.0,
                error=str(e)
            )
    
    def compute_rouge(self, predictions: List[str], references: List[str]) -> MetricResult:
        """Compute ROUGE scores for summarization metrics."""
        # Fallback: approximate ROUGE-L by longest-common-subsequence ratio if rouge-score not installed
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            try:
                def lcs_len(a: List[str], b: List[str]) -> int:
                    # simple DP LCS
                    la, lb = len(a), len(b)
                    dp = [[0] * (lb + 1) for _ in range(la + 1)]
                    for i in range(la - 1, -1, -1):
                        for j in range(lb - 1, -1, -1):
                            if a[i] == b[j]:
                                dp[i][j] = 1 + dp[i + 1][j + 1]
                            else:
                                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
                    return dp[0][0]

                rougeL_scores = []
                for pred, ref in zip(predictions, references):
                    p_tokens = pred.split()
                    r_tokens = ref.split()
                    if not r_tokens:
                        rougeL_scores.append(0.0)
                        continue
                    lcs = lcs_len(p_tokens, r_tokens)
                    rougeL_scores.append(lcs / max(1, len(r_tokens)))
                score = float(np.mean(rougeL_scores)) if rougeL_scores else 0.0
                return MetricResult(metric_name="rouge", score=score, details={"method": "lcs_proxy"})
            except Exception as e:
                return MetricResult(metric_name="rouge", score=0.0, error=str(e))
        
        try:
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for pred, ref in zip(predictions, references):
                scores = self.rouge_scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            # Return average ROUGE-L score
            score = np.mean(rougeL_scores)
            
            return MetricResult(
                metric_name="rouge",
                score=score,
                details={
                    "rouge1": np.mean(rouge1_scores),
                    "rouge2": np.mean(rouge2_scores),
                    "rougeL": score
                }
            )
        except Exception as e:
            return MetricResult(
                metric_name="rouge",
                score=0.0,
                error=str(e)
            )
    
    def compute_ruby(self, predictions: List[str], references: List[str]) -> MetricResult:
        """Compute RUBY — prefer JetBrains ``codegen_metrics.metrics.ruby`` when importable."""
        if CODEGEN_RUBY_AVAILABLE:
            try:
                scores = []
                for pred, ref in zip(predictions, references):
                    if not pred.strip() and not ref.strip():
                        scores.append(1.0)
                        continue
                    if not pred or not ref:
                        scores.append(0.0)
                        continue
                    # API: ruby(reference_code, generated_code)
                    scores.append(float(codegen_ruby_pair(ref, pred)))
                score = float(np.mean(scores)) if scores else 0.0
                return MetricResult(
                    metric_name="ruby",
                    score=score,
                    details={"implementation": "codegen_metrics.metrics.ruby", "individual_scores": scores},
                )
            except Exception as e:
                logger.warning("codegen_metrics RUBY failed (%s); using built-in fallback", e)

        # Built-in PDG/AST/token fallback (when codegen_metrics is unavailable).
        try:
            results = []
            method_used = []
            
            for pred, ref in zip(predictions, references):
                ruby_score = None
                method = None
                
                # 1. Try GRS (Graph Representation Similarity) - PDG level
                grs_score = self._compute_grs(ref, pred)
                if grs_score is not None:
                    ruby_score = grs_score
                    method = "GRS"
                
                # 2. Try TRS (Tree Representation Similarity) - AST level
                if ruby_score is None:
                    trs_score = self._compute_trs(ref, pred)
                    if trs_score is not None:
                        ruby_score = trs_score
                        method = "TRS"
                
                # 3. Fallback to STS (String/Token Similarity) - token level
                if ruby_score is None:
                    ruby_score = self._compute_sts(ref, pred)
                    method = "STS"
                
                results.append(ruby_score)
                method_used.append(method)
            
            score = np.mean(results) if results else 0.0
            
            return MetricResult(
                metric_name="ruby_like_heuristic",
                score=score,
                details={
                    "method_distribution": {
                        method: method_used.count(method) for method in set(method_used)
                    },
                    "individual_scores": results,
                    "method_used": method_used
                }
            )
        except Exception as e:
            return MetricResult(
                metric_name="ruby_like_heuristic",
                score=0.0,
                error=str(e)
            )
    
    def _compute_grs(self, reference_code: str, translated_code: str) -> Optional[float]:
        """Compute GRS (Graph Representation Similarity) using PDG.
        
        Сравнивает Program Dependence Graphs используя graph edit distance.
        Это самый высокий уровень представления согласно статье.
        """
        try:
            import networkx as nx
            
            pdg_ref = self._build_pdg(reference_code)
            pdg_trans = self._build_pdg(translated_code)
            
            if pdg_ref is None or pdg_trans is None:
                return None
            
            # Graph Edit Distance (GED)
            ged = self._graph_edit_distance(pdg_ref, pdg_trans)
            
            # Normalize by graph size
            pdg_size = len(pdg_ref.nodes) + len(pdg_ref.edges) + len(pdg_trans.nodes) + len(pdg_trans.edges)
            
            if pdg_size > 0:
                similarity = 1.0 - (ged / pdg_size)
                return max(0.0, min(1.0, similarity))
            else:
                return 0.0
        except Exception:
            return None
    
    def _compute_trs(self, reference_code: str, translated_code: str) -> Optional[float]:
        """Compute TRS (Tree Representation Similarity) using AST.
        
        Сравнивает Abstract Syntax Trees используя tree edit distance.
        Это средний уровень представления согласно статье.
        """
        try:
            ast_ref = self._parse_ast(reference_code)
            ast_trans = self._parse_ast(translated_code)
            
            if ast_ref is None or ast_trans is None:
                return None
            
            # Tree Edit Distance (TED)
            ted = self._tree_edit_distance(ast_ref, ast_trans)
            
            # Normalize by tree size
            tree_size = self._count_ast_nodes(ast_ref) + self._count_ast_nodes(ast_trans)
            
            if tree_size > 0:
                similarity = 1.0 - (ted / tree_size)
                return max(0.0, min(1.0, similarity))
            else:
                return 0.0
        except Exception:
            return None
    
    def _compute_sts(self, reference_code: str, translated_code: str) -> float:
        """Compute STS (String/Token Similarity) using token-level comparison.
        
        Fallback метод на токен-уровне когда PDG и AST не могут быть построены.
        """
        try:
            tokens_ref = self._tokenize_code(reference_code)
            tokens_trans = self._tokenize_code(translated_code)
            
            # String Edit Distance (Levenshtein distance)
            sed = self._string_edit_distance(tokens_ref, tokens_trans)
            
            max_length = max(len(tokens_ref), len(tokens_trans))
            
            if max_length > 0:
                similarity = 1.0 - (sed / max_length)
                return max(0.0, min(1.0, similarity))
            else:
                return 1.0  # Both empty - perfect match
        except Exception:
            return 0.0
    
    def _build_pdg(self, code: str):
        """Build Program Dependence Graph from code.
        
        Упрощенная версия: создает граф из AST узлов.
        Полная реализация должна включать data и control dependencies.
        """
        try:
            import networkx as nx
            
            tree = self._parse_ast(code)
            if tree is None:
                return None
            
            pdg = nx.DiGraph()
            
            # Simplified PDG: add nodes for each AST node
            # Full implementation would include:
            # - Data dependencies (variable definitions and uses)
            # - Control dependencies (control flow edges)
            nodes = list(ast.walk(tree))
            for i, node in enumerate(nodes):
                pdg.add_node(i, type=type(node).__name__)
            
            # Add basic control flow edges based on AST structure
            # Traverse AST and connect parent-child relationships
            def add_edges(node, parent_idx=None):
                current_idx = nodes.index(node) if node in nodes else None
                if current_idx is not None and parent_idx is not None:
                    pdg.add_edge(parent_idx, current_idx)
                for child in ast.iter_child_nodes(node):
                    if child in nodes:
                        add_edges(child, current_idx)
            
            add_edges(tree)
            
            return pdg
        except Exception:
            return None
    
    def _parse_ast(self, code: str):
        """Parse code into AST."""
        try:
            return ast.parse(code)
        except SyntaxError:
            return None
        except Exception:
            return None
    
    def _tokenize_code(self, code: str) -> List[str]:
        """Tokenize code into tokens."""
        tokens = []
        current_token = ""
        
        for char in code:
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            elif char in '(){}[];.,=+-*/<>!&|':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            else:
                current_token += char
        
        if current_token:
            tokens.append(current_token)
        
        return tokens
    
    def _string_edit_distance(self, tokens1: List[str], tokens2: List[str]) -> int:
        """Compute Levenshtein distance between token sequences."""
        m, n = len(tokens1), len(tokens2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif tokens1[i-1] == tokens2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    def _tree_edit_distance(self, tree1, tree2) -> int:
        """Compute simplified tree edit distance between ASTs.
        
        Упрощенная версия: использует разницу в количестве узлов.
        Полная реализация должна использовать алгоритм Zhang-Shasha или аналогичный.
        """
        count1 = self._count_ast_nodes(tree1)
        count2 = self._count_ast_nodes(tree2)
        return abs(count1 - count2)
    
    def _count_ast_nodes(self, node) -> int:
        """Count nodes in AST."""
        count = 1
        for child in ast.iter_child_nodes(node):
            count += self._count_ast_nodes(child)
        return count
    
    def _graph_edit_distance(self, graph1, graph2) -> int:
        """Compute simplified graph edit distance between PDGs.
        
        Упрощенная версия: использует разницу в количестве узлов и рёбер.
        Полная реализация должна использовать точный алгоритм GED.
        """
        node_diff = abs(len(graph1.nodes) - len(graph2.nodes))
        edge_diff = abs(len(graph1.edges) - len(graph2.edges))
        return node_diff + edge_diff
    
    def _test_execution(self, code: str) -> float:
        """Test if code can be executed (simplified version)."""
        try:
            # Create a safe execution environment
            safe_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round,
                    'int': int,
                    'float': float,
                    'str': str,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'bool': bool,
                    'type': type,
                    'isinstance': isinstance,
                    'hasattr': hasattr,
                    'getattr': getattr,
                    'setattr': setattr,
                }
            }
            
            # Try to compile and execute
            compiled = compile(code, '<string>', 'exec')
            exec(compiled, safe_globals)
            return 1.0
            
        except Exception:
            return 0.0
    
    def compute_all_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, MetricResult]:
        """Compute all available metrics.

        Keys are **uniform**: ``codebleu`` (official) *or* ``codebleu_proxy``; ``ruby`` (codegen_metrics)
        *or* ``ruby_like_heuristic``. Consumers should use :func:`get_metric_float` for robust access.
        """
        metrics: Dict[str, MetricResult] = {}
        metrics["bertscore"] = self.compute_bertscore(predictions, references)
        cb = self.compute_codebleu(predictions, references)
        metrics[cb.metric_name] = cb
        metrics["bleu"] = self.compute_bleu(predictions, references)
        metrics["rouge"] = self.compute_rouge(predictions, references)
        rb = self.compute_ruby(predictions, references)
        metrics[rb.metric_name] = rb
        return metrics
    
    def evaluate_against_targets(self, metrics: Dict[str, MetricResult], targets: Dict[str, float]) -> Dict[str, bool]:
        """Evaluate if metrics meet target thresholds."""
        results = {}
        
        for metric_name, target in targets.items():
            if metric_name in metrics:
                results[metric_name] = metrics[metric_name].score >= target
            else:
                results[metric_name] = False
        
        return results
    
    def get_summary(self, metrics: Dict[str, MetricResult]) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        summary = {
            "scores": {},
            "errors": {},
            "overall_success": True
        }
        
        for metric_name, result in metrics.items():
            summary["scores"][metric_name] = result.score
            if result.error:
                summary["errors"][metric_name] = result.error
                summary["overall_success"] = False
        
        return summary


# Utility functions for batch evaluation
def evaluate_batch(
    predictions: List[str],
    references: List[str],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Evaluate a batch of predictions against references."""
    evaluator = ModernMetricsEvaluator(config)
    metrics = evaluator.compute_all_metrics(predictions, references)
    summary = evaluator.get_summary(metrics)
    
    return {
        "metrics": metrics,
        "summary": summary
    }


def evaluate_single(
    prediction: str,
    reference: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Evaluate a single prediction against a reference."""
    return evaluate_batch([prediction], [reference], config)
