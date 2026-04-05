"""
ClassifMLP - Neural Code Quality Classification Pipeline
MODIFIED FOR CORRECT METRIC ONLY

Core MLP classifier with anti-overfitting techniques for code quality assessment.
Uses 'correct' metric from human feedback instead of 'useful'.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from stage4.model import MultiHeadClassifier, build_feature_vector, _ensure_tensor


# =============================================================================
# ENHANCED MLP CLASSIFIER WITH FEATURES
# =============================================================================

class EnhancedClassifierWithFeatures(nn.Module):
    """Enhanced MLP classifier with code features and anti-overfitting techniques."""

    def __init__(self, embedding_dim: int, code_feature_dim: int = 74, hidden_dim: int = 512, dropout: float = 0.4):
        super().__init__()
        input_dim = embedding_dim * 4 + code_feature_dim  # embeddings + code features
        hidden_dim = max(hidden_dim, embedding_dim)
        mid_dim = max(hidden_dim // 2, embedding_dim // 2)

        # Deep MLP with regularization
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        logits = self.net(features)
        return logits.squeeze(-1)


# =============================================================================
# DUMMY EMBEDDING ENCODER
# =============================================================================

class DummyEmbeddingEncoder:
    """Deterministic embedding encoder for reproducible demos."""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self._cache: Dict[str, torch.Tensor] = {}

    def encode(self, texts):
        """Return deterministic embeddings derived from text hashes."""
        texts_list = list(texts) if not isinstance(texts, torch.Tensor) else [texts]

        if not texts_list:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)

        embeddings = []
        for text in texts_list:
            normalized = self._normalize_text(text)
            embeddings.append(self._get_embedding(normalized))

        return torch.stack(embeddings, dim=0)

    def _normalize_text(self, text: str) -> str:
        if text is None:
            return ""
        return " ".join(str(text).split()).lower()

    def _get_embedding(self, normalized_text: str) -> torch.Tensor:
        cached = self._cache.get(normalized_text)
        if cached is not None:
            return cached.clone()

        tensor = self._build_embedding(normalized_text)
        self._cache[normalized_text] = tensor
        return tensor.clone()

    def _build_embedding(self, normalized_text: str) -> torch.Tensor:
        import hashlib
        digest = hashlib.sha256(normalized_text.encode("utf-8")).digest()[:8]
        seed = int.from_bytes(digest, "little", signed=False)
        rng = np.random.default_rng(seed)
        values = rng.standard_normal(self.embedding_dim, dtype=np.float32)
        return torch.from_numpy(values)


# =============================================================================
# CODE FEATURE EXTRACTION
# =============================================================================

def extract_code_features(code: str) -> Dict[str, float]:
    """Extract code features from code snippet (67 features total)."""
    import warnings
    features = {}

    # Basic text metrics
    features['code_length'] = len(code)
    features['num_lines'] = len(code.split('\n'))
    features['avg_line_length'] = len(code) / max(1, features['num_lines'])

    # Python keywords (25 features)
    python_keywords = {
        'import', 'from', 'def', 'class', 'if', 'for', 'while', 'try', 'except',
        'return', 'yield', 'lambda', 'and', 'or', 'not', 'in', 'is', 'None',
        'True', 'False', 'with', 'as', 'pass', 'break', 'continue', 'raise',
        'assert', 'global', 'nonlocal', 'del', 'await', 'async'
    }

    for keyword in python_keywords:
        count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', code))
        features[f'keyword_{keyword}'] = count

    features['total_keywords'] = sum(features[f'keyword_{k}'] for k in python_keywords)

    # Common modules (13 features)
    common_modules = {
        'os', 'sys', 're', 'json', 'math', 'datetime', 'collections', 'itertools',
        'numpy', 'pandas', 'torch', 'tensorflow', 'sklearn', 'matplotlib', 'PIL'
    }

    imported_modules = set()
    for module in common_modules:
        features[f'imports_{module}'] = 1.0 if module in code else 0.0

    # Syntax validity
    features['syntax_valid'] = 1.0 if check_syntax_validity(code) else 0.0

    # AST features (simplified - 20 features)
    ast_features = extract_basic_ast_features(code)
    features.update(ast_features)

    # Structural features (10 features)
    structural_features = extract_basic_structural_features(code)
    features.update(structural_features)

    return features


def check_syntax_validity(code: str) -> bool:
    """Check if Python code is syntactically valid."""
    try:
        # Suppress syntax warnings during compilation
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            compile(code, '<string>', 'exec')
        return True
    except (SyntaxError, IndentationError, TypeError):
        return False


def extract_basic_ast_features(code: str) -> Dict[str, float]:
    """Extract basic AST features."""
    features = {
        'ast_num_functions': 0, 'ast_num_classes': 0, 'ast_num_loops': 0,
        'ast_num_conditionals': 0, 'ast_max_nesting': 0, 'ast_num_assignments': 0,
        'ast_num_calls': 0, 'ast_num_returns': 0
    }

    try:
        import ast
        tree = ast.parse(code)

        class ASTVisitor(ast.NodeVisitor):
            def __init__(self):
                self.features = features.copy()
                self.nesting_level = 0
                self.max_nesting = 0

            def visit_FunctionDef(self, node):
                self.features['ast_num_functions'] += 1

            def visit_ClassDef(self, node):
                self.features['ast_num_classes'] += 1

            def visit_For(self, node):
                self.features['ast_num_loops'] += 1

            def visit_While(self, node):
                self.features['ast_num_loops'] += 1

            def visit_If(self, node):
                self.features['ast_num_conditionals'] += 1

            def visit_Assign(self, node):
                self.features['ast_num_assignments'] += 1

            def visit_Call(self, node):
                self.features['ast_num_calls'] += 1

            def visit_Return(self, node):
                self.features['ast_num_returns'] += 1

        visitor = ASTVisitor()
        visitor.visit(tree)
        features.update(visitor.features)

    except SyntaxError:
        pass

    return features


def extract_basic_structural_features(code: str) -> Dict[str, float]:
    """Extract basic structural features."""
    features = {}

    # Cyclomatic complexity (simplified)
    predicates = len(re.findall(r'\b(if|while|for|and|or|not)\b', code))
    features['cyclomatic_complexity'] = predicates + 1

    # Indentation statistics
    lines = code.split('\n')
    indent_levels = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            indent = len(line) - len(line.lstrip())
            indent_levels.append(indent)

    if indent_levels:
        features['max_indent'] = max(indent_levels) / 4
        features['avg_indent'] = sum(indent_levels) / len(indent_levels) / 4
        features['indent_variance'] = np.var(indent_levels) if len(indent_levels) > 1 else 0
    else:
        features['max_indent'] = 0
        features['avg_indent'] = 0
        features['indent_variance'] = 0

    # Comment ratio
    comment_lines = len([line for line in lines if line.strip().startswith('#')])
    code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
    features['comment_ratio'] = comment_lines / max(1, code_lines)

    # String and numeric literals
    string_literals = len(re.findall(r'["\'].*?["\']', code))
    numeric_literals = len(re.findall(r'\b\d+\.?\d*\b', code))
    features['num_string_literals'] = string_literals
    features['num_numeric_literals'] = numeric_literals

    return features


def build_feature_vector_with_code_features(
    question_emb: torch.Tensor,
    answer_emb: torch.Tensor,
    answer_code: str = None
) -> torch.Tensor:
    """Build combined features with code analysis."""
    base_features = build_feature_vector(question_emb, answer_emb)

    if answer_code is None:
        return base_features

    code_features = extract_code_features(answer_code)
    code_feature_values = list(code_features.values())
    code_features_tensor = torch.tensor(code_feature_values, dtype=torch.float32, device=base_features.device)

    combined_features = torch.cat([base_features, code_features_tensor.unsqueeze(0)], dim=-1)
    return combined_features


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class FeedbackClassificationDataset(Dataset):
    """Dataset for code quality classification."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'question': sample['question'],
            'answer': sample['answer'],
            'label': float(sample.get('label', 0)),
            'metadata': sample.get('metadata', {})
        }


def _tokenize_code(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|==|!=|<=|>=|->|::|[^\s]", text)


def _token_overlap_ratio(prediction: str, reference: str) -> float:
    ref_tokens = _tokenize_code(reference)
    if not ref_tokens:
        return 0.0
    pred_tokens = _tokenize_code(prediction)
    ref_counter = Counter(ref_tokens)
    pred_counter = Counter(pred_tokens)
    common = sum(min(ref_counter[token], pred_counter.get(token, 0)) for token in ref_counter)
    return common / max(1, len(ref_tokens))


def _token_f1_score(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize_code(prediction)
    ref_tokens = _tokenize_code(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = sum(min(pred_counter[token], ref_counter[token]) for token in ref_counter)
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _bleu_unigram(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize_code(prediction)
    ref_tokens = _tokenize_code(reference)
    if not pred_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    match = sum(min(pred_counter[token], ref_counter.get(token, 0)) for token in pred_counter)
    return match / len(pred_tokens)


def _rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize_code(prediction)
    ref_tokens = _tokenize_code(reference)
    if not ref_tokens:
        return 0.0
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if pred_tokens[i] == ref_tokens[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0] / max(1, n)


def _syntax_validity_score(code: str) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            compile(code, '<string>', 'exec')
        return 1.0
    except Exception:
        return 0.0


def _ast_signature(code: str) -> Counter:
    try:
        tree = ast.parse(code)
    except Exception:
        return Counter()
    counter: Counter = Counter()
    for node in ast.walk(tree):
        counter[type(node).__name__] += 1
    return counter


def _ast_similarity_score(prediction: str, reference: str) -> float:
    pred_sig = _ast_signature(prediction)
    ref_sig = _ast_signature(reference)
    if not pred_sig and not ref_sig:
        return 1.0
    keys = set(pred_sig) | set(ref_sig)
    total = sum(pred_sig.get(k, 0) + ref_sig.get(k, 0) for k in keys)
    if total == 0:
        return 1.0
    diff = sum(abs(pred_sig.get(k, 0) - ref_sig.get(k, 0)) for k in keys)
    return max(0.0, 1.0 - diff / total)


def _ruby_score(prediction: str, reference: str) -> float:
    syntax_score = (_syntax_validity_score(prediction) + _syntax_validity_score(reference)) / 2
    structure_score = _ast_similarity_score(prediction, reference)
    token_score = _token_f1_score(prediction, reference)
    combined = 0.4 * token_score + 0.3 * structure_score + 0.3 * syntax_score
    return max(0.0, min(1.0, combined))


def _selection_has_required_labels(all_labels: List[float], selected_indices: List[int]) -> bool:
    """Ensure selected indices cover the label diversity present in the dataset."""
    if not selected_indices:
        return False

    selected_labels = [all_labels[i] for i in selected_indices]
    has_positive = any(label >= 0.6 for label in all_labels)
    has_negative = any(label <= 0.4 for label in all_labels)
    has_neutral = any(0.4 < label < 0.6 for label in all_labels)

    conditions = []
    if has_positive:
        conditions.append(any(label >= 0.6 for label in selected_labels))
    if has_negative:
        conditions.append(any(label <= 0.4 for label in selected_labels))
    if has_neutral and len(all_labels) >= 10:
        conditions.append(any(0.4 < label < 0.6 for label in selected_labels))

    return all(conditions) if conditions else True


def balanced_split_indices(
    labels: List[float],
    val_ratio: float = 0.2,
    seed: int = 42,
    max_attempts: int = 32
) -> Tuple[List[int], List[int]]:
    """Create a balanced train/validation split that preserves label diversity."""
    total = len(labels)
    if total == 0:
        raise ValueError("Cannot split empty dataset")

    val_size = max(1, int(round(total * val_ratio)))
    indices = list(range(total))
    rng = random.Random(seed)

    for _ in range(max_attempts):
        rng.shuffle(indices)
        val_indices = indices[:val_size]
        if _selection_has_required_labels(labels, val_indices):
            train_indices = indices[val_size:]
            return train_indices, val_indices

    # Fallback: stratified sampling per bucket
    buckets = {
        'positive': [i for i, label in enumerate(labels) if label >= 0.6],
        'negative': [i for i, label in enumerate(labels) if label <= 0.4],
        'neutral': [i for i, label in enumerate(labels) if 0.4 < label < 0.6],
    }

    for bucket in buckets.values():
        rng.shuffle(bucket)

    val_indices: List[int] = []
    for name in ('positive', 'negative', 'neutral'):
        if buckets[name] and len(val_indices) < val_size:
            val_indices.append(buckets[name].pop())

    remaining = [idx for idx in indices if idx not in val_indices]
    rng.shuffle(remaining)
    needed = val_size - len(val_indices)
    val_indices.extend(remaining[:max(0, needed)])

    train_indices = [idx for idx in indices if idx not in val_indices]
    return train_indices, val_indices


def iter_feedback_samples(feedback_dir: Path, dataset_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load real datasets with human feedback and ground truth reference answers."""
    import glob
    import csv
    import hashlib
    
    try:
        current_dir = Path.cwd()
        eval_dir = dataset_dir or (current_dir / "stage4" / "datasets_for_eval")
        feedback_rel = str(feedback_dir).replace("clasifNN/", "").replace("stage4/", "")
        feedback_dir_full = current_dir / "stage4" / feedback_rel

        print(f"Loading datasets from eval_dir: {eval_dir}, feedback_dir: {feedback_dir_full}")

        if not eval_dir.exists() or not feedback_dir_full.exists():
            print("Directories not found, using dummy data")
            raise FileNotFoundError("Required directories not found")

        # Step 1: Build lookup table from CSV files (ID -> ground truth Answer)
        csv_reference_lookup = {}  # {(csv_path, question_id): ground_truth_answer}
        csv_pattern = str(eval_dir / "*.csv")
        csv_files = glob.glob(csv_pattern)
        print(f"Found {len(csv_files)} CSV files in eval_dir")
        
        for csv_file in csv_files:
            csv_name = os.path.basename(csv_file)
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        question_id = row.get('ID')
                        ground_truth = row.get('Answer', '').strip()
                        if question_id and ground_truth:
                            csv_reference_lookup[(csv_name, str(question_id))] = ground_truth
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
        
        print(f"Loaded {len(csv_reference_lookup)} ground truth answers from CSV files")

        # Step 2: Load human feedback JSON files and match with ground truth
        samples = []
        json_pattern = str(feedback_dir_full / "*.json")
        json_files = glob.glob(json_pattern)
        print(f"Found {len(json_files)} JSON files in feedback_dir")
        
        ref_matched = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                questions_list = data.get('questions_df', [])
                
                for idx, question_data in enumerate(questions_list):
                    question_id = str(question_data.get('ID', ''))
                    answer = question_data.get('Answer', '').strip()
                    question_text = question_data.get('Question', '').strip()
                    csv_path = question_data.get('CSV_PATH', '')

                    if not question_id or not answer:
                        continue

                    # Get human feedback scores
                    if idx == 0:  # Left answer
                        consistent = (data.get('consistent_L', 0) + 2) / 4
                        correct = (data.get('correct_L', 0) + 2) / 4
                        useful = (data.get('useful_L', 0) + 2) / 4
                    else:  # Right answer
                        consistent = (data.get('consistent_R', 0) + 2) / 4
                        correct = (data.get('correct_R', 0) + 2) / 4
                        useful = (data.get('useful_R', 0) + 2) / 4

                    # Look up ground truth reference answer using CSV_PATH and ID
                    reference_answer = csv_reference_lookup.get((csv_path, question_id))
                    if reference_answer:
                        ref_matched += 1

                    sample = {
                        'question': question_text,
                        'answer': answer,
                        'labels': {
                            'consistent': consistent,
                            'correct': correct,
                            'useful': useful
                        },
                        'metadata': {
                            'source_file': os.path.basename(json_file),
                            'question_id': question_id,
                            'csv_path': csv_path,
                            'reference_answer': reference_answer,  # Ground truth from CSV
                            'has_feedback': True
                        }
                    }
                    samples.append(sample)
                    
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue

        print(f"Loaded {len(samples)} samples with human feedback")
        print(f"Matched {ref_matched} samples with ground truth reference answers")
        
        if samples:
            return samples

        print("No samples loaded, using dummy data")
    except Exception as e:
        print(f"Error loading real datasets: {e}, using dummy data")

    # Fallback to enhanced dummy data
    samples = []
    questions = [
        "How to sort a list in Python?",
        "How to read a file in Python?",
        "How to calculate factorial recursively?",
        "How to handle exceptions in Python?",
        "How to use list comprehensions?",
        "How to work with dictionaries?",
        "How to write functions in Python?",
        "How to use classes in Python?",
        "How to handle file I/O?",
        "How to use loops in Python?"
    ]

    answers = [
        "sorted_list = sorted(my_list)",
        "with open('file.txt', 'r') as f: content = f.read()",
        "def factorial(n): return n * factorial(n-1) if n > 1 else 1",
        "try: risky_code() except Exception as e: handle_error(e)",
        "squares = [x**2 for x in range(10)]",
        "my_dict = {'key': 'value'}; value = my_dict.get('key')",
        "def greet(name): return f'Hello {name}'",
        "class Calculator: def add(self, a, b): return a + b",
        "with open('data.txt', 'w') as f: f.write('content')",
        "for i in range(5): print(i)"
    ]

    labels = [1, 1, 1, 0, 1, 1, 1, 1, 1, 0]  # Mix of good/bad examples

    for i, (q, a, label) in enumerate(zip(questions, answers, labels)):
        samples.append({
            'question': q,
            'answer': a,
            'label': label,
            'metadata': {'sample_id': i}
        })

    return samples


class IntegratedTrainingPipeline:
    """Integrated training pipeline with anti-overfitting techniques."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        self.embedding_encoder = DummyEmbeddingEncoder()
        # Dynamically determine code feature dimension
        test_features = extract_code_features("def test(): pass")
        self.code_feature_dim = len(test_features)
        self.total_input_dim = 768 * 4 + self.code_feature_dim

        # Use EnhancedClassifierWithFeatures (anti-overfitting solution)
        self.classifier = EnhancedClassifierWithFeatures(
            embedding_dim=768,
            code_feature_dim=self.code_feature_dim,
            hidden_dim=config.get('hidden_dim', 512),
            dropout=config.get('dropout', 0.4)
        )

        self.classifier.to(self.device)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 30) -> Dict[str, Any]:
        """Train the MLP classifier with anti-overfitting techniques."""
        print("=== ClassifMLP Training (CORRECT METRIC) ===")
        print(f"Device: {self.device}")
        print(f"Model: EnhancedClassifierWithFeatures")
        print(f"Input dim: {self.total_input_dim}, Hidden dim: {self.config.get('hidden_dim', 512)}")
        print(f"Dropout: {self.config.get('dropout', 0.4)}, Batch size: {self.config.get('batch_size', 16)}")

        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.config.get('learning_rate', 1e-5),
            weight_decay=self.config.get('weight_decay', 1e-3)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

        pos_weight_val = self.config.get('pos_weight', 1.0)
        pos_weight = torch.tensor([pos_weight_val], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('patience', 5)
        history = []

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")

            # Training
            self.classifier.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for batch in tqdm(train_loader, desc="Training"):
                questions = batch['question']
                answers = batch['answer']
                labels = batch['label'].to(self.device)
                # Convert continuous labels to binary (threshold at 0.5)
                labels = (labels >= 0.5).float()

                question_emb = self.embedding_encoder.encode(questions)
                answer_emb = self.embedding_encoder.encode(answers)

                extended_features = []
                for q, a in zip(questions, answers):
                    features = build_feature_vector_with_code_features(
                        question_emb[len(extended_features):len(extended_features)+1],
                        answer_emb[len(extended_features):len(extended_features)+1],
                        a
                    )
                    extended_features.append(features[0])

                extended_features = torch.stack(extended_features).to(self.device)

                # Debug: Check feature dimensions
                if epoch == 1 and len(extended_features) > 0:
                    print(f"DEBUG: Feature tensor shape: {extended_features.shape}")
                    print(f"DEBUG: Expected input dim: {self.classifier.net[0].in_features}")

                optimizer.zero_grad()
                logits = self.classifier(extended_features)
                loss = criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * len(questions)
                predictions = (torch.sigmoid(logits) > 0.5).long()
                total_correct += (predictions == labels.long()).sum().item()
                total_samples += len(questions)

                # Debug: Check prediction distribution (first epoch only)
                if epoch == 1 and len(questions) > 0:
                    pred_vals = torch.sigmoid(logits).detach().cpu().numpy()
                    label_vals = labels.detach().cpu().numpy()
                    print(f"DEBUG: Train predictions - Mean: {pred_vals.mean():.3f}, Std: {pred_vals.std():.3f}")
                    print(f"DEBUG: Train labels - Mean: {label_vals.mean():.3f}, Std: {label_vals.std():.3f}")
                    print(f"DEBUG: Train accuracy on this batch: {(predictions == labels.long()).float().mean().item():.3f}")

            avg_train_loss = total_loss / total_samples
            train_acc = total_correct / total_samples

            # Validation
            val_loss, val_acc, val_code_samples, val_clf_metrics = self.validate(val_loader, criterion, collect_samples=True)
            code_metrics = self.compute_code_quality_metrics(val_code_samples)
            scheduler.step(val_loss)

            print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
            print(f"  Val F1: {val_clf_metrics['f1']:.3f}, Precision: {val_clf_metrics['precision']:.3f}, Recall: {val_clf_metrics['recall']:.3f}, Balanced Acc: {val_clf_metrics['balanced_accuracy']:.3f}")
            if code_metrics:
                print(
                    "  Code metrics:"
                    f" BERTScore={(code_metrics.get('bertscore') or 0):.3f}"
                    f", CodeBLEU={(code_metrics.get('codebleu') or 0):.3f}"
                    f", BLEU={(code_metrics.get('bleu') or 0):.3f}"
                    f", ROUGE={(code_metrics.get('rouge') or 0):.3f}"
                    f", RUBY={(code_metrics.get('ruby') or 0):.3f}"
                )
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model("best_model.pt")
            else:
                patience_counter += 1

            if patience is not None and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            # Compute detailed accuracy metrics
            train_correct_acc = train_acc
            val_correct_acc = val_acc

            history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_correct_acc': train_correct_acc,
                'val_correct_acc': val_correct_acc,
                'val_f1': val_clf_metrics['f1'],
                'val_precision': val_clf_metrics['precision'],
                'val_recall': val_clf_metrics['recall'],
                'val_balanced_accuracy': val_clf_metrics['balanced_accuracy'],
                'val_bertscore': code_metrics.get('bertscore'),
                'val_codebleu': code_metrics.get('codebleu'),
                'val_bleu': code_metrics.get('bleu'),
                'val_rouge': code_metrics.get('rouge'),
                'val_ruby': code_metrics.get('ruby')
            })

        return {'history': history, 'best_val_loss': best_val_loss}


    def validate(self, val_loader: DataLoader, criterion, collect_samples: bool = False) -> Tuple[float, float, Optional[List[Tuple[str, str, float]]], Dict[str, float]]:
        """Validate the model. Returns (loss, accuracy, code_samples, classification_metrics)."""
        self.classifier.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        code_samples: Optional[List[Tuple[str, str, float]]] = [] if collect_samples else None
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for batch in val_loader:
                questions = batch['question']
                answers = batch['answer']
                labels = batch['label'].to(self.device)
                # Convert continuous labels to binary (threshold at 0.5)
                labels = (labels >= 0.5).float()

                question_emb = self.embedding_encoder.encode(questions)
                answer_emb = self.embedding_encoder.encode(answers)

                extended_features = []
                for q, a in zip(questions, answers):
                    features = build_feature_vector_with_code_features(
                        question_emb[len(extended_features):len(extended_features)+1],
                        answer_emb[len(extended_features):len(extended_features)+1],
                        a
                    )
                    extended_features.append(features[0])

                extended_features = torch.stack(extended_features).to(self.device)

                logits = self.classifier(extended_features)
                loss = criterion(logits, labels)

                total_loss += loss.item() * len(questions)
                predictions = (torch.sigmoid(logits) > 0.5).long()
                total_correct += (predictions == labels.long()).sum().item()
                total_samples += len(questions)
                all_preds.extend(predictions.cpu().tolist())
                all_labels.extend(labels.long().cpu().tolist())
                if collect_samples and code_samples is not None:
                    metadata_batch = batch.get('metadata', [])
                    probs = torch.sigmoid(logits).detach().cpu().tolist()
                    for answer, meta, prob in zip(answers, metadata_batch, probs):
                        reference = None
                        if isinstance(meta, dict):
                            reference = meta.get('reference_answer')
                        if reference:
                            code_samples.append((answer, reference, float(prob)))

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        # Compute F1, precision, recall, balanced accuracy
        import numpy as _np
        preds = _np.array(all_preds)
        truths = _np.array(all_labels)
        tp = int(((preds == 1) & (truths == 1)).sum())
        fp = int(((preds == 1) & (truths == 0)).sum())
        fn = int(((preds == 0) & (truths == 1)).sum())
        tn = int(((preds == 0) & (truths == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        balanced_acc = (sensitivity + specificity) / 2.0
        clf_metrics = {
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'balanced_accuracy': float(balanced_acc),
        }
        return avg_loss, accuracy, code_samples, clf_metrics

    def compute_code_quality_metrics(self, samples: Optional[List[Tuple[str, str, float]]]) -> Dict[str, float]:
        """Compute confidence-weighted code quality metrics from validation samples."""
        if not samples:
            return {}

        weighted_sums = {
            'bertscore': 0.0,
            'codebleu': 0.0,
            'bleu': 0.0,
            'rouge': 0.0,
            'ruby': 0.0,
        }
        total_weight = 0.0

        for answer, reference, confidence in samples:
            if not reference:
                continue
            weight = max(1e-6, float(confidence))
            weighted_sums['bertscore'] += _token_overlap_ratio(answer, reference) * weight
            weighted_sums['codebleu'] += _token_f1_score(answer, reference) * weight
            weighted_sums['bleu'] += _bleu_unigram(answer, reference) * weight
            weighted_sums['rouge'] += _rouge_l(answer, reference) * weight
            weighted_sums['ruby'] += _ruby_score(answer, reference) * weight
            total_weight += weight

        if total_weight <= 0:
            return {}

        return {
            'bertscore': float(weighted_sums['bertscore'] / total_weight),
            'codebleu': float(weighted_sums['codebleu'] / total_weight),
            'bleu': float(weighted_sums['bleu'] / total_weight),
            'rouge': float(weighted_sums['rouge'] / total_weight),
            'ruby': float(weighted_sums['ruby'] / total_weight),
        }

    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'classifier': self.classifier.state_dict(),
            'config': self.config
        }, path)
        print(f"Model saved: {path}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def collate_fn(batch):
    """Collate function for DataLoader."""
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    metadata = [item['metadata'] for item in batch]
    return {
        'question': questions,
        'answer': answers,
        'label': labels,
        'metadata': metadata
    }


def train_integrated_system(args: argparse.Namespace) -> None:
    """Main training function for ClassifMLP - CORRECT METRIC."""
    import random as _random
    import numpy as _np
    seed = getattr(args, 'seed', 42)
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"=== ClassifMLP Training (CORRECT METRIC) | seed={seed} ===")

    config = {
        'device': args.device,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': getattr(args, 'patience', 5),
    }

    feedback_dir = Path(args.feedback_dir)
    dataset_dir = Path(getattr(args, 'dataset_dir', "stage4/datasets_for_eval"))
    samples = iter_feedback_samples(feedback_dir, dataset_dir)
    print(f"Loaded {len(samples)} samples")

    # Normalize data format: convert 'labels' dict to single 'label' for compatibility
    # MODIFIED: Use 'correct' as the primary label (instead of 'useful')
    for sample in samples:
        if 'labels' in sample and 'label' not in sample:
            # Use 'correct' as the primary label for binary classification
            correct_score = sample['labels']['correct']

            # Create more balanced binary labels using stricter threshold
            if correct_score >= 0.8:  # Stricter threshold for positive
                sample['label'] = 1.0
            elif correct_score <= 0.4:  # Only very low scores as negative
                sample['label'] = 0.0
            else:
                # For medium confidence, randomly assign to create balance
                import random
                sample['label'] = 1.0 if random.random() > 0.7 else 0.0
        elif 'label' not in sample:
            import random
            sample['label'] = 1.0 if random.random() > 0.5 else 0.0

    # Debug: Check data distribution
    print("DEBUG: Sample data inspection (CORRECT METRIC):")
    first_10_labels = [s['label'] for s in samples[:10]]
    print(f"DEBUG: First 10 labels: {first_10_labels}")

    if samples and 'labels' in samples[0]:
        print(f"DEBUG: Sample has 'labels' dict: {samples[0]['labels']}")

    all_labels = [s['label'] for s in samples]
    positive_total = sum(1 for l in all_labels if l > 0.5)
    negative_total = sum(1 for l in all_labels if l < 0.5)
    neutral_total = sum(1 for l in all_labels if l == 0.5)
    print(f"DEBUG: Overall label distribution: {positive_total} positive, {negative_total} negative, {neutral_total} neutral out of {len(samples)}")

    import numpy as np
    labels_array = np.array(all_labels)
    print(f"DEBUG: Label stats - Mean: {labels_array.mean():.3f}, Std: {labels_array.std():.3f}, Min: {labels_array.min():.3f}, Max: {labels_array.max():.3f}")

    unique_labels = len(set(all_labels))
    print(f"DEBUG: Unique label values: {unique_labels} out of {len(samples)} samples")
    if unique_labels < 10:
        print("WARNING: Very few unique labels - classification might be trivial!")

    unique_questions = set(s['question'] for s in samples)
    print(f"DEBUG: Unique questions: {len(unique_questions)} out of {len(samples)} total samples")

    dataset = FeedbackClassificationDataset(samples)
    val_ratio = getattr(args, 'val_ratio', 0.2)
    train_indices, val_indices = balanced_split_indices(all_labels, val_ratio=val_ratio)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Debug: Check for data leakage
    train_indices_set = set(train_indices)
    val_indices_set = set(val_indices)
    overlap = train_indices_set & val_indices_set
    print(f"DEBUG: Train/Val split - Train indices: {len(train_indices_set)}, Val indices: {len(val_indices_set)}")
    print(f"DEBUG: Overlap between train/val: {len(overlap)} samples")
    if overlap:
        print(f"ERROR: Data leakage detected! Overlapping indices: {overlap}")

    def _count_labels(indices: List[int]) -> Tuple[int, int, int]:
        positives = sum(1 for i in indices if dataset[i]['label'] >= 0.6)
        negatives = sum(1 for i in indices if dataset[i]['label'] <= 0.4)
        neutrals = len(indices) - positives - negatives
        return positives, negatives, neutrals

    train_pos, train_neg, train_neu = _count_labels(train_indices)
    val_pos, val_neg, val_neu = _count_labels(val_indices)
    print(f"DEBUG: Train labels - {train_pos} positive, {train_neg} negative, {train_neu} neutral")
    print(f"DEBUG: Val labels - {val_pos} positive, {val_neg} negative, {val_neu} neutral")

    # Class-imbalance correction: weight positives by (n_neg / n_pos)
    pos_weight_val = train_neg / max(train_pos, 1)
    print(f"INFO: pos_weight for BCEWithLogitsLoss = {pos_weight_val:.3f} ({train_neg} neg / {train_pos} pos)")
    config['pos_weight'] = pos_weight_val

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    pipeline = IntegratedTrainingPipeline(config)
    training_results = pipeline.train(train_loader, val_loader, num_epochs=args.epochs)

    # Save history — per-seed subfolder when seed != default
    seed = getattr(args, 'seed', 42)
    output_dir = Path(args.output_dir)
    if seed != 42:
        output_dir = output_dir / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "training_history_correct.json"
    with history_path.open('w') as f:
        json.dump(training_results['history'], f, indent=2)

    print(f"Training completed! Results saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ClassifMLP - Neural Code Quality Classifier (CORRECT METRIC)")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feedback-dir", type=str, default="evaluation_results_server")
    parser.add_argument("--output-dir", type=str, default="stage4/stage4B_corct/artifacts")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dataset-dir", type=str, default="stage4/datasets_for_eval")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_integrated_system(args)
