"""
Modern Data Loader for RLHF
===========================

A comprehensive data loader that handles:
- Training data preparation
- Evaluation data loading
- Human feedback integration
- Data preprocessing and augmentation
"""

import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import random
from pathlib import Path
from datasets import load_dataset  # Added import for Hugging Face datasets
import sys
from contextlib import contextmanager
import tempfile
from typing import cast
import time
import re
try:
    import torch
except Exception:
    torch = None
try:
    from huggingface_hub import hf_hub_download, list_repo_files
    _HF_HUB_AVAILABLE = True
except Exception:
    _HF_HUB_AVAILABLE = False

from .config import ModernRLHFConfig, DataConfig

logger = logging.getLogger(__name__)

@dataclass
class DataSample:
    """Container for a single data sample."""
    prompt: str
    response: str
    reference: Optional[str] = None
    rating: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class ModernDataLoader:
    """Modern data loader for RLHF training."""
    
    def __init__(self, config: ModernRLHFConfig):
        self.config = config
        self.data_config = config.data
        
        # Set random seed for reproducibility
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        logger.info(f"Initialized ModernDataLoader with config: {self.data_config}")

    def _sample_to_dict(self, item: Any) -> Dict[str, Any]:
        """Normalize a sample to a dict (accepts dict or DataSample objects)."""
        if isinstance(item, dict):
            return item
        try:
            return {
                'prompt': getattr(item, 'prompt', '') or '',
                'response': getattr(item, 'response', '') or '',
                'reference': getattr(item, 'reference', None),
                'rating': getattr(item, 'rating', None),
                'metadata': getattr(item, 'metadata', None)
            }
        except Exception:
            return {'prompt': '', 'response': '', 'reference': None, 'rating': None, 'metadata': None}

    @contextmanager
    def _no_local_dataset_scripts(self):
        """Temporarily remove project paths from sys.path to avoid picking local conala.py."""
        project_root = Path(__file__).resolve().parents[2]  # repo root
        removed = []
        original_sys_path = list(sys.path)
        for p in list(sys.path):
            try:
                pr = Path(p).resolve()
                if project_root in pr.parents or pr == project_root or p in ("", "."):
                    sys.path.remove(p)
                    removed.append(p)
            except Exception:
                # Non-pathy entries, ignore
                if p in ("", "."):
                    try:
                        sys.path.remove(p)
                        removed.append(p)
                    except Exception:
                        pass
        try:
            yield
        finally:
            # Restore original sys.path order
            sys.path[:] = original_sys_path

    @contextmanager
    def _temp_cwd(self):
        """Temporarily change working directory to a safe temp folder."""
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                yield
            finally:
                os.chdir(old_cwd)

    def _load_conala_split(self, split: str):
        """Load CoNaLa curated split directly from Hugging Face Hub without dataset scripts.

        Strategy:
        1) Try discovering curated parquet files via huggingface_hub and load with pandas.
        2) If that fails, try datasets parquet builder with explicit URLs.
        3) Finally, try repo id APIs.
        """
        # 0) Prefer local corpus if provided
        local = self._load_conala_local(split)
        if local is not None:
            return local
        # 1) Discover and load curated parquet(s) via Hub API
        if _HF_HUB_AVAILABLE:
            try:
                files = list_repo_files(repo_id="neulab/conala", repo_type="dataset")
                # Prefer curated parquet paths containing the split name
                candidate_paths = [
                    f for f in files
                    if f.lower().endswith('.parquet') and (
                        ('/curated/' in f.replace('\\', '/')) or ('curated' in f)
                    ) and (f"/{split}" in f.replace('\\', '/') or f"{split}-" in f or f"{split}.parquet" in f)
                ]
                # If nothing found under curated, fall back to any parquet with split in name
                if not candidate_paths:
                    candidate_paths = [
                        f for f in files
                        if f.lower().endswith('.parquet') and (f"/{split}" in f.replace('\\', '/') or f"{split}-" in f)
                    ]
                if candidate_paths:
                    dfs = []
                    for rel_path in candidate_paths:
                        try:
                            local_path = hf_hub_download(repo_id="neulab/conala", filename=rel_path, repo_type="dataset")
                            dfs.append(pd.read_parquet(local_path))
                        except Exception as e_dl:
                            logger.warning(f"Failed to download/read parquet {rel_path}: {e_dl}")
                    if dfs:
                        df = pd.concat(dfs, ignore_index=True)
                        return df.to_dict(orient='records')
            except Exception as e:
                logger.warning(f"hf_hub listing/parquet load failed: {e}")

            # Additional direct download attempts for common curated parquet names
            try:
                possible_paths = [
                    f"curated/{split}.parquet",
                    f"curated/{split}-00000-of-00001.parquet",
                    f"{split}.parquet",
                    f"{split}-00000-of-00001.parquet",
                ]
                dfs = []
                for rel_path in possible_paths:
                    try:
                        local_path = hf_hub_download(repo_id="neulab/conala", filename=rel_path, repo_type="dataset")
                        dfs.append(pd.read_parquet(local_path))
                    except Exception:
                        continue
                if dfs:
                    df = pd.concat(dfs, ignore_index=True)
                    return df.to_dict(orient='records')
            except Exception as e_extra:
                logger.warning(f"hf_hub direct curated parquet attempts failed: {e_extra}")

        # 2) Datasets parquet builder with explicit URL(s)
        with self._no_local_dataset_scripts(), self._temp_cwd():
            try:
                url = f"https://huggingface.co/datasets/neulab/conala/resolve/main/curated/{split}-00000-of-00001.parquet"
                ds = load_dataset("parquet", data_files={split: [url]}, split=split)
                return ds
            except Exception as e1:
                logger.warning(f"datasets parquet builder failed: {e1}")
                try:
                    # Прямой доступ к подконфигурации на HF
                    return load_dataset("hf://datasets/neulab/conala/curated", split=split)
                except Exception as e2:
                    logger.warning(f"Direct curated load failed: {e2}")
                    try:
                        # Резерв: стандартный ID
                        return load_dataset("neulab/conala", "curated", split=split)
                    except Exception as e3:
                        logger.error(f"All loading methods failed for split '{split}': {e3}")
                        raise

    def _load_conala_local(self, split: str) -> Optional[List[Dict[str, Any]]]:
        """Load CoNaLa curated split from a local corpus directory if available.

        Supports common file names and formats in the official corpus:
        - conala-train.json / conala-test.json (JSON array or JSONL)
        - curated_train.json / curated_test.json
        - train.json / test.json
        """
        root = getattr(self.data_config, 'conala_local_path', None)
        if not root:
            return None
        corpus_dir = Path(root)
        if not corpus_dir.exists():
            logger.warning(f"Conala local path not found: {corpus_dir}")
            return None

        candidates = [
            f"conala-{split}.json",
            f"conala_{split}.json",
            f"curated_{split}.json",
            f"{split}.json",
            f"conala-{split}.jsonl",
            f"conala_{split}.jsonl",
            f"curated_{split}.jsonl",
            f"{split}.jsonl",
            # nested under 'conala-corpus' subdir if user points to parent
            f"conala-corpus/conala-{split}.json",
            f"conala-corpus/conala_{split}.json",
        ]

        file_path = None
        for name in candidates:
            p = corpus_dir / name
            if p.exists():
                file_path = p
                break

        if file_path is None:
            # Try to locate any json/jsonl mentioning split in name
            for p in corpus_dir.rglob("*.json*"):
                if split in p.name.lower() and ("train" in p.name.lower() or "test" in p.name.lower()):
                    file_path = p
                    break

        if file_path is None:
            logger.warning(f"No local CoNaLa file found for split '{split}' in {corpus_dir}")
            return None

        logger.info(f"Loading local CoNaLa {split} from: {file_path}")

        records: List[Dict[str, Any]] = []
        try:
            if file_path.suffix == '.jsonl' or file_path.suffixes[-1] == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        records.append(json.loads(line))
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                    if isinstance(obj, dict) and split in obj:
                        records = obj[split]
                    elif isinstance(obj, list):
                        records = obj
                    else:
                        # Some dumps store under keys like 'data'
                        for key in ['data', 'examples', 'items']:
                            if isinstance(obj, dict) and key in obj and isinstance(obj[key], list):
                                records = obj[key]
                                break
            # Normalize to expected fields
            normalized = []
            for item in records:
                prompt = item.get('rewritten_intent') or item.get('intent') or item.get('question') or ""
                response = item.get('snippet') or item.get('code') or item.get('answer') or ""
                qid = item.get('question_id') or item.get('id')
                normalized.append({
                    'prompt': prompt,
                    'response': response,
                    'reference': response,
                    'rating': None,
                    'metadata': {'source': f'conala_{split}_local', 'question_id': qid}
                })
            return normalized
        except Exception as e:
            logger.error(f"Failed to load local CoNaLa {split} from {file_path}: {e}")
            return None

    def _load_conala_train_samples(self) -> List[DataSample]:
        """Load CoNaLa curated train split and return as DataSample entries."""
        logger.info("Loading CoNaLa train split (local or Hugging Face)...")

        dataset = self._load_conala_split('train')

        samples: List[DataSample] = []
        for item in dataset:
            if isinstance(item, dict):
                prompt = (
                    item.get('prompt')
                    or item.get('rewritten_intent')
                    or item.get('intent')
                    or item.get('question')
                    or ""
                )
                response = (
                    item.get('response')
                    or item.get('snippet')
                    or item.get('code')
                    or item.get('answer')
                    or ""
                )
                qid = item.get('question_id') or item.get('id')
            else:
                try:
                    prompt = item['prompt'] if item['prompt'] else item['rewritten_intent']
                except Exception:
                    try:
                        prompt = item['rewritten_intent'] if item['rewritten_intent'] else item['intent']
                    except Exception:
                        prompt = item.get('intent', "")  # type: ignore[attr-defined]
                try:
                    response = item['response'] if item['response'] else item['snippet']
                except Exception:
                    response = item.get('snippet', "")  # type: ignore[attr-defined]
                qid = item.get('question_id', None)  # type: ignore[attr-defined]

            samples.append(DataSample(
                prompt=str(prompt),
                response=str(response),
                reference=str(response) if response else None,
                rating=None,
                metadata={'source': 'conala_train', 'question_id': qid}
            ))

        logger.info(f"Loaded {len(samples)} raw CoNaLa training samples")
        return samples

    def load_training_data(self) -> List[DataSample]:
        """Load training data from various sources."""
        logger.info("Loading training data...")

        all_samples: List[DataSample] = []

        # Prefer CoNaLa curated dataset
        try:
            conala_samples = self._load_conala_train_samples()
            all_samples.extend(conala_samples)
            logger.info(f"Added {len(conala_samples)} CoNaLa samples")
        except Exception as e:
            logger.warning(f"Failed to load CoNaLa training data: {e}")

        # Load from different sources
        sources = [
            self._load_sft_data,
            self._load_preference_data,
            self._load_synthetic_data
        ]
        
        for source_func in sources:
            try:
                samples = source_func()
                all_samples.extend(samples)
                logger.info(f"Loaded {len(samples)} samples from {source_func.__name__}")
            except Exception as e:
                logger.warning(f"Failed to load from {source_func.__name__}: {e}")
        
        # Filter and clean data
        filtered_samples = self._filter_samples(all_samples)

        if self.data_config.use_data_augmentation and filtered_samples:
            filtered_samples = self.augment_data(filtered_samples)

        # Limit samples if specified
        if self.data_config.max_train_samples > 0:
            filtered_samples = filtered_samples[:self.data_config.max_train_samples]
        
        logger.info(f"Total training samples available: {len(filtered_samples)}")
        
        return filtered_samples
    
    def load_evaluation_data(self) -> List[DataSample]:
        """Load evaluation data.

        Strategy:
        1) If a local CoNaLa corpus path is provided or HF hub is available, prefer loading the
           curated CoNaLa 'test' split (to evaluate on the canonical dataset).
        2) Otherwise, attempt to load CSV evaluation datasets from `eval_data_path` listed in
           `self.config.evaluation.eval_datasets`.
        """
        logger.info("Loading evaluation data...")

        all_samples: List[DataSample] = []

        # First preference: local CoNaLa corpus (if provided) -> try any matching JSON/JSONL files
        loaded = False
        try:
            local_root = getattr(self.data_config, 'conala_local_path', None)
            if local_root:
                corpus_dir = Path(local_root)
                if corpus_dir.exists():
                    # Prefer an explicit conala-test.json if present
                    explicit_test = corpus_dir / 'conala-test.json'
                    explicit_testl = corpus_dir / 'conala-test.jsonl'
                    if explicit_test.exists():
                        matched_files = [explicit_test]
                    elif explicit_testl.exists():
                        matched_files = [explicit_testl]
                    else:
                        # find files likely matching the test split
                        json_files = sorted(list(corpus_dir.glob('*.json*')))
                        matched_files = [p for p in json_files if 'test' in p.name.lower()]
                        # fallback: if no explicit test files, accept any json/jsonl
                        if not matched_files:
                            matched_files = json_files

                    for file_path in matched_files:
                        try:
                            logger.info(f"Loading local CoNaLa evaluation from: {file_path}")
                            with open(file_path, 'r', encoding='utf-8') as f:
                                if file_path.suffix == '.jsonl' or file_path.name.lower().endswith('.jsonl'):
                                    records = [json.loads(line) for line in f if line.strip()]
                                else:
                                    obj = json.load(f)
                                    if isinstance(obj, list):
                                        records = obj
                                    elif isinstance(obj, dict):
                                        # try common keys
                                        for key in ['test', 'data', 'examples', 'items']:
                                            if key in obj and isinstance(obj[key], list):
                                                records = obj[key]
                                                break
                                        else:
                                            # flatten values if dict of dicts
                                            records = [v for v in obj.values() if isinstance(v, dict)]

                            # normalize records to DataSample and append
                            for item in records:
                                if isinstance(item, dict):
                                    prompt = item.get('rewritten_intent') or item.get('intent') or item.get('question') or item.get('text') or ""
                                    snippet = item.get('snippet') or item.get('code') or item.get('answer') or item.get('response') or ""
                                    qid = item.get('question_id') or item.get('id')
                                else:
                                    # fallback generic
                                    prompt = str(item)
                                    snippet = ""
                                    qid = None

                                all_samples.append(DataSample(
                                    prompt=str(prompt),
                                    response="",
                                    reference=str(snippet) if snippet is not None else None,
                                    rating=None,
                                    metadata={'source': 'conala_test_local', 'question_id': qid, 'file': str(file_path)}
                                ))
                            loaded = True
                        except Exception as e_file:
                            logger.warning(f"Failed to read local CoNaLa file {file_path}: {e_file}")

        except Exception as e:
            logger.warning(f"Local CoNaLa load check failed: {e}")

        # If local corpus not loaded, try HF curated split (robust) as before
        if not loaded:
            try:
                conala = self._load_conala_split('test')
                if conala is not None:
                    ds = conala
                else:
                    # Try direct datasets load as a fallback (force HF)
                    try:
                        from datasets import load_dataset
                        with self._no_local_dataset_scripts(), self._temp_cwd():
                            ds = load_dataset('neulab/conala', 'curated', split='test')
                    except Exception:
                        ds = None

                if ds is not None:
                    logger.info("Loading evaluation data from CoNaLa curated split (HF)")
                    # datasets.Dataset supports iteration and dict-like access
                    for item in ds:
                        if isinstance(item, dict):
                            prompt = item.get('rewritten_intent') or item.get('intent') or item.get('question') or ""
                            snippet = item.get('snippet') or item.get('code') or item.get('answer') or ""
                            qid = item.get('question_id') or item.get('id')
                        else:
                            # fallback
                            prompt = getattr(item, 'intent', "")
                            snippet = getattr(item, 'snippet', "")
                            qid = getattr(item, 'question_id', None)

                        all_samples.append(DataSample(
                            prompt=str(prompt),
                            response="",
                            reference=str(snippet) if snippet is not None else None,
                            rating=None,
                            metadata={'source': 'conala_test', 'question_id': qid}
                        ))
                    loaded = True
            except Exception as e:
                logger.warning(f"CoNaLa curated load failed or not available: {e}")

        # Fallback: load evaluation CSVs from eval_data_path
        if not all_samples:
            eval_path = Path(self.data_config.eval_data_path)
            if eval_path.exists():
                for dataset_file in self.config.evaluation.eval_datasets:
                    try:
                        samples = self._load_evaluation_dataset(eval_path / dataset_file)
                        all_samples.extend(samples)
                        logger.info(f"Loaded {len(samples)} samples from {dataset_file}")
                    except Exception as e:
                        logger.warning(f"Failed to load {dataset_file}: {e}")

        # Final cleaning and filtering
        filtered_samples = self._filter_samples(all_samples, allow_empty_response=True)
        if self.data_config.max_eval_samples > 0:
            filtered_samples = filtered_samples[:self.data_config.max_eval_samples]

        # If still empty, create synthetic evaluation samples so metrics can be computed
        if not filtered_samples:
            logger.warning("No evaluation samples found; generating synthetic evaluation set")
            n = getattr(self.data_config, 'max_eval_samples', None) or 100
            n = min(n, getattr(self.config.evaluation, 'eval_samples', 100))
            synth = self._generate_synthetic_data()
            # Use generated synthetic responses as references, but keep response empty for generation
            synth_eval: List[DataSample] = []
            for i, s in enumerate(synth[:n]):
                synth_eval.append(DataSample(
                    prompt=s.prompt,
                    response="",
                    reference=s.response,
                    rating=None,
                    metadata={'source': 'synthetic_eval', 'index': i}
                ))
            filtered_samples = synth_eval

        logger.info(f"Total evaluation samples loaded: {len(filtered_samples)}")
        return filtered_samples
    
    def load_human_feedback(self) -> Optional[str]:
        """Load human feedback data."""
        logger.info("Loading human feedback data...")

        feedback_dir = Path(self.data_config.human_feedback_path)

        if feedback_dir.exists():
            json_files = sorted([p for p in feedback_dir.glob("*.json")], key=os.path.getmtime)
            if json_files:
                latest_file = json_files[-1]
                logger.info(f"Found human feedback file: {latest_file}")
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read human feedback file {latest_file}: {e}")
                    return None

                # Normalize to list of feedback dicts
                items: List[Dict[str, Any]] = []
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    for key in ['data', 'items', 'examples', 'feedback']:
                        v = data.get(key)
                        if isinstance(v, list):
                            items = v
                            break
                # If still empty but dict of pairs, convert to list
                if not items and isinstance(data, dict):
                    # Try to interpret dict values as list entries
                    for v in data.values():
                        if isinstance(v, dict):
                            items.append(v)

                logger.info(f"Loaded {len(items)} human feedback entries from {latest_file}")
                return items

        # If no feedback files exist, create a small synthetic feedback file and return it
        logger.warning(f"Human feedback directory not found or empty: {feedback_dir} — generating synthetic feedback")
        items = self.generate_synthetic_human_feedback()
        return items

    def integrate_human_feedback(self, samples: List[DataSample], feedback_items: List[Dict[str, Any]]) -> None:
        """Attach human feedback to matching samples by question_id or prompt/response match.

        This mutates `samples` in place: it adds metadata['human_rating'] and
        prepends a short human-feedback token to the prompt so models receive it as context.
        """
        if not feedback_items:
            return

        # Index samples by question_id (if present) and by prompt text
        by_qid = {}
        by_prompt = {}
        for s in samples:
            qid = getattr(s.metadata or {}, 'get', lambda k, d=None: None)('question_id', None)
            if qid:
                by_qid[str(qid)] = s
            by_prompt[str(s.prompt).strip()] = s

        matched = 0
        for fb in feedback_items:
            rating = fb.get('rating')
            prompt = fb.get('prompt') or fb.get('rewritten_intent') or fb.get('intent') or ""
            response = fb.get('response') or fb.get('snippet') or fb.get('code') or ""
            qid = fb.get('question_id') or fb.get('id')

            target = None
            if qid and str(qid) in by_qid:
                target = by_qid[str(qid)]
            elif prompt and prompt.strip() in by_prompt:
                target = by_prompt[prompt.strip()]
            else:
                # Try fuzzy match by substring on prompt
                for p_text, s in by_prompt.items():
                    if prompt.strip() and prompt.strip() in p_text:
                        target = s
                        break

            if target is not None:
                if target.metadata is None:
                    target.metadata = {}
                # store rating and raw feedback
                if rating is not None:
                    target.metadata['human_rating'] = float(rating)
                    # also set the primary rating field so existing code paths can use it
                    try:
                        target.rating = float(rating)
                    except Exception:
                        target.rating = None
                if fb.get('comment'):
                    target.metadata['human_comment'] = fb.get('comment')

                # Prepend short rating context to prompt for model context
                try:
                    rating_str = f"[HumanRating:{int(float(rating))}] " if rating is not None else ""
                except Exception:
                    rating_str = ""
                target.prompt = f"{rating_str}{target.prompt}"
                matched += 1

        logger.info(f"Integrated human feedback: matched {matched} entries into {len(samples)} samples")

    def generate_synthetic_human_feedback(self, n: int = 200, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate synthetic human-feedback dataset based on CoNaLa data with realistic ratings.

        This creates feedback entries using actual CoNaLa prompts and snippets, with ratings
        based on code quality heuristics (syntax, length, complexity).
        """
        import random
        output_dir = output_dir or self.data_config.human_feedback_path
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        items: List[Dict[str, Any]] = []

        # Load CoNaLa train data to get real prompts and code
        try:
            logger.info("Loading CoNaLa data for synthetic feedback generation...")
            conala_samples = self._load_conala_train_samples()
            
            if not conala_samples:
                logger.warning("No CoNaLa samples available, falling back to generic feedback")
                return self._generate_generic_feedback(n, out_path)
            
            # Sample n entries from CoNaLa
            sample_size = min(n, len(conala_samples))
            selected_samples = random.sample(conala_samples, sample_size)
            
            logger.info(f"Generating {sample_size} synthetic feedback entries from CoNaLa data...")
            
            for i, sample in enumerate(selected_samples):
                prompt = sample.prompt
                response = sample.response
                qid = sample.metadata.get('question_id') if sample.metadata else f'synth_{i}'
                
                # Generate realistic rating based on code quality heuristics
                rating = self._evaluate_code_quality(response)
                
                # Add some randomness to make it more realistic
                rating = max(1, min(5, rating + random.randint(-1, 1)))
                
                items.append({
                    'id': qid,
                    'question_id': qid,
                    'prompt': prompt,
                    'response': response,
                    'rating': rating,
                    'comment': f"Synthetic rating based on code quality: {rating}/5",
                    'source': 'conala_synthetic'
                })
            
            logger.info(f"[OK] Generated {len(items)} CoNaLa-based feedback entries")
            
        except Exception as e:
            logger.warning(f"Failed to generate CoNaLa-based feedback: {e}")
            logger.info("Falling back to generic feedback generation")
            return self._generate_generic_feedback(n, out_path)

        # Save to a timestamped file so load_human_feedback can find it
        fname = out_path / f"synthetic_human_feedback_{int(time.time())}.json"
        try:
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(items, f, indent=2)
            logger.info(f"[OK] Wrote synthetic human feedback to {fname}")
        except Exception as e:
            logger.warning(f"Failed to write synthetic human feedback: {e}")

        return items
    
    def _evaluate_code_quality(self, code: str) -> int:
        """Evaluate code quality and return a rating from 1-5.
        
        Heuristics:
        - Syntax correctness (can it be parsed?)
        - Length (too short or too long is bad)
        - Complexity indicators (functions, classes, etc.)
        - Style (proper indentation, naming)
        """
        if not code or not code.strip():
            return 1
        
        score = 3  # Start with neutral
        
        # Check syntax
        try:
            import ast
            ast.parse(code)
            score += 1  # Valid syntax
        except:
            score -= 1  # Invalid syntax
        
        # Check length (reasonable code should be 20-500 chars)
        code_len = len(code.strip())
        if 20 <= code_len <= 500:
            score += 1
        elif code_len < 10:
            score -= 2
        elif code_len > 1000:
            score -= 1
        
        # Check for good practices
        code_lower = code.lower()
        if 'def ' in code or 'class ' in code:
            score += 1  # Has functions/classes
        if any(kw in code_lower for kw in ['import ', 'from ']):
            score += 0.5  # Uses imports
        if code.count('\n') >= 2:
            score += 0.5  # Multi-line code
        
        # Clamp to 1-5 range
        return max(1, min(5, int(round(score))))
    
    def _generate_generic_feedback(self, n: int, out_path: Path) -> List[Dict[str, Any]]:
        """Fallback: generate generic synthetic feedback."""
        import random
        
        items: List[Dict[str, Any]] = []
        
        for i in range(n):
            prompt = f"Example prompt asking for solution {i}"
            response = f"Example response content {i}"
            rating = random.randint(1, 5)
            
            items.append({
                'id': f'synth_{i}',
                'prompt': prompt,
                'response': response,
                'rating': rating,
                'comment': f"Generic synthetic rating {rating}"
            })
        
        # Save
        fname = out_path / f"synthetic_human_feedback_{int(time.time())}.json"
        try:
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(items, f, indent=2)
            logger.info(f"Wrote generic synthetic feedback to {fname}")
        except Exception as e:
            logger.warning(f"Failed to write generic feedback: {e}")
        
        return items
    
    def _load_sft_data(self) -> List[DataSample]:
        """Load supervised fine-tuning data."""
        samples = []
        
        sft_path = Path(self.data_config.train_data_path) / "sft_dataset.csv"
        
        if sft_path.exists():
            df = pd.read_csv(sft_path)
            
            # Find appropriate columns
            prompt_col = self._find_column(df, ['prompt', 'instruction', 'question', 'input'])
            response_col = self._find_column(df, ['response', 'answer', 'output', 'completion'])
            
            if prompt_col and response_col:
                for _, row in df.iterrows():
                    sample = DataSample(
                        prompt=str(row[prompt_col]),
                        response=str(row[response_col]),
                        metadata={'source': 'sft', 'row_id': row.name}
                    )
                    samples.append(sample)
        
        return samples
    
    def _load_preference_data(self) -> List[DataSample]:
        """Load preference data."""
        samples = []
        
        pref_path = Path(self.data_config.train_data_path) / "pairwise_prefs.csv"
        
        if pref_path.exists():
            df = pd.read_csv(pref_path)
            
            # Find appropriate columns
            prompt_col = self._find_column(df, ['prompt', 'instruction', 'question', 'input'])
            chosen_col = self._find_column(df, ['chosen', 'preferred', 'better'])
            rejected_col = self._find_column(df, ['rejected', 'not_preferred', 'worse'])
            rating_col = self._find_column(df, ['rating', 'score', 'preference'])
            
            if prompt_col and chosen_col and rejected_col:
                for _, row in df.iterrows():
                    # Create sample for chosen response
                    chosen_sample = DataSample(
                        prompt=str(row[prompt_col]),
                        response=str(row[chosen_col]),
                        rating=float(row[rating_col]) if rating_col else 1.0,
                        metadata={'source': 'preference', 'type': 'chosen', 'row_id': row.name}
                    )
                    samples.append(chosen_sample)
                    
                    # Create sample for rejected response
                    rejected_sample = DataSample(
                        prompt=str(row[prompt_col]),
                        response=str(row[rejected_col]),
                        rating=0.0,
                        metadata={'source': 'preference', 'type': 'rejected', 'row_id': row.name}
                    )
                    samples.append(rejected_sample)
        
        return samples
    
    def _load_synthetic_data(self) -> List[DataSample]:
        """Load synthetic data or generate if needed."""
        samples = []
        
        # Check for existing synthetic data
        synthetic_path = Path(self.data_config.train_data_path) / "synthetic_data.csv"
        
        if synthetic_path.exists():
            df = pd.read_csv(synthetic_path)
            
            prompt_col = self._find_column(df, ['prompt', 'instruction', 'question', 'input'])
            response_col = self._find_column(df, ['response', 'answer', 'output', 'completion'])
            
            if prompt_col and response_col:
                for _, row in df.iterrows():
                    sample = DataSample(
                        prompt=str(row[prompt_col]),
                        response=str(row[response_col]),
                        metadata={'source': 'synthetic', 'row_id': row.name}
                    )
                    samples.append(sample)
        else:
            # Generate some basic synthetic data if none exists
            samples = self._generate_synthetic_data()
        
        return samples
    
    def _load_evaluation_dataset(self, dataset_path: Path) -> List[DataSample]:
        """Load a specific evaluation dataset."""
        samples = []
        
        if dataset_path.suffix == '.csv':
            df = pd.read_csv(dataset_path)
            
            # Find appropriate columns
            prompt_col = self._find_column(df, ['prompt', 'instruction', 'question', 'input', 'text'])
            response_col = self._find_column(df, ['response', 'answer', 'output', 'completion', 'code'])
            reference_col = self._find_column(df, ['reference', 'ground_truth', 'expected'])
            
            if prompt_col:
                for _, row in df.iterrows():
                    sample = DataSample(
                        prompt=str(row[prompt_col]),
                        response=str(row[response_col]) if response_col else "",
                        reference=str(row[reference_col]) if reference_col else None,
                        metadata={'source': 'evaluation', 'dataset': dataset_path.name, 'row_id': row.name}
                    )
                    samples.append(sample)
        
        return samples
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find a column with one of the possible names."""
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    def _filter_samples(self, samples: List[Dict[str, Any]], allow_empty_response: bool = False) -> List[Dict[str, Any]]:
        """Filter and clean samples based on criteria.
        If allow_empty_response is True, do not enforce min response length and allow empty responses (for eval).
        """

    def _filter_samples(self, samples: List[DataSample]) -> List[DataSample]:
        """Filter and clean samples based on criteria."""

    def _filter_samples(self, samples: List[DataSample]) -> List[DataSample]:
        """Filter and clean samples based on criteria."""


    def _filter_samples(self, samples: List[DataSample], allow_empty_response: bool = False) -> List[DataSample]:
        """Filter and clean samples based on criteria.

        Returns a list of `DataSample` instances that pass the configured filters.
        Accepts input samples as either dicts or `DataSample` objects.
        """
        filtered_samples: List[DataSample] = []

        for sample in samples:
            s = self._sample_to_dict(sample)
            prompt = str(s.get('prompt', '') or '')
            response = str(s.get('response', '') or '')

            if len(prompt) < self.data_config.min_prompt_length:
                continue
            if len(prompt) > self.data_config.max_prompt_length:
                continue
            if not allow_empty_response:
                if len(response) < self.data_config.min_response_length:
                    continue
            if len(response) > self.data_config.max_response_length:
                continue

            # Check for empty or invalid content
            if not prompt.strip():
                continue
            if not allow_empty_response and not response.strip():
                continue

            # Determine acceptance
            require_code_like = getattr(self.data_config, 'require_code_like', True)

            if allow_empty_response:
                accept = True
            elif not require_code_like:
                accept = True
            else:
                accept = self._is_code_like(prompt) or self._is_code_like(response)

            if not accept:
                continue

            filtered_samples.append(DataSample(
                prompt=prompt,
                response=response,
                reference=s.get('reference'),
                rating=s.get('rating'),
                metadata=s.get('metadata')
            ))

        logger.info(f"Filtered {len(samples)} samples to {len(filtered_samples)} valid samples")

        return filtered_samples
    
    def _is_code_like(self, text: str) -> bool:
        """Check if text looks like code."""
        # Simple heuristics for code detection
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ',
            'return ', 'print(', 'function', 'var ', 'let ', 'const ',
            '{', '}', '(', ')', ';', '=', '==', '!='
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in code_indicators)
    
    def _generate_synthetic_data(self) -> List[DataSample]:
        """Generate basic synthetic data for training."""
        samples = []
        
        # Basic code generation prompts
        basic_prompts = [
            "Write a function to calculate the factorial of a number",
            "Create a function that reverses a string",
            "Write a function to check if a number is prime",
            "Create a function that finds the maximum element in a list",
            "Write a function to sort a list of numbers",
            "Create a function that counts the frequency of each character in a string",
            "Write a function to find the greatest common divisor of two numbers",
            "Create a function that checks if a string is a palindrome",
            "Write a function to generate the Fibonacci sequence",
            "Create a function that removes duplicates from a list"
        ]
        
        # Basic responses (these would be improved with actual code generation)
        basic_responses = [
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
            "def reverse_string(s):\n    return s[::-1]",
            "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
            "def find_max(lst):\n    return max(lst)",
            "def sort_list(lst):\n    return sorted(lst)",
            "def count_chars(s):\n    return {char: s.count(char) for char in set(s)}",
            "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
            "def is_palindrome(s):\n    return s == s[::-1]",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "def remove_duplicates(lst):\n    return list(set(lst))"
        ]
        
        for prompt, response in zip(basic_prompts, basic_responses):
            sample = DataSample(
                prompt=prompt,
                response=response,
                metadata={'source': 'synthetic', 'generated': True}
            )
            samples.append(sample)
        
        logger.info(f"Generated {len(samples)} synthetic samples")
        
        return samples
    
    def augment_data(self, samples: List[DataSample]) -> List[DataSample]:
        """Augment training data if enabled."""
        if not self.data_config.use_data_augmentation:
            return samples
        
        logger.info("Augmenting training data...")
        
        augmented_samples = samples.copy()
        augmentation_count = int(len(samples) * self.data_config.augmentation_ratio)
        
        # Select random samples for augmentation
        indices_to_augment = random.sample(range(len(samples)), augmentation_count)
        
        for idx in indices_to_augment:
            original_sample = samples[idx]
            
            # Simple augmentation: add variations to prompts
            augmented_prompt = self._augment_prompt(original_sample.prompt)
            
            augmented_sample = DataSample(
                prompt=augmented_prompt,
                response=original_sample.response,
                reference=original_sample.reference,
                rating=original_sample.rating,
                metadata={**(original_sample.metadata or {}), 'augmented': True}
            )
            
            augmented_samples.append(augmented_sample)
        
        logger.info(f"Augmented {augmentation_count} samples, total: {len(augmented_samples)}")
        
        return augmented_samples
    
    def _augment_prompt(self, prompt: str) -> str:
        """Augment a single prompt."""
        # Simple augmentation strategies
        augmentations = [
            lambda p: f"Please {p.lower()}",
            lambda p: f"Can you {p.lower()}?",
            lambda p: f"I need help with: {p}",
            lambda p: f"Write code to {p.lower()}",
            lambda p: f"Create a solution for: {p}"
        ]
        
        # Randomly select an augmentation
        augmentation = random.choice(augmentations)
        return augmentation(prompt)
    
    def create_train_test_split(self, samples: List[DataSample]) -> Tuple[List[DataSample], List[DataSample]]:
        """Create train-test split."""
        random.shuffle(samples)
        
        split_idx = int(len(samples) * self.data_config.train_test_split)
        
        train_samples = samples[:split_idx]
        test_samples = samples[split_idx:]
        
        logger.info(f"Created train-test split: {len(train_samples)} train, {len(test_samples)} test")
        
        return train_samples, test_samples
    
    def save_samples(self, samples: List[DataSample], filepath: str):
        """Save samples to a file."""
        data = []
        
        for sample in samples:
            data.append({
                'prompt': sample.prompt,
                'response': sample.response,
                'reference': sample.reference,
                'rating': sample.rating,
                'metadata': sample.metadata
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Saved {len(samples)} samples to {filepath}")
    
    def load_samples(self, filepath: str) -> List[DataSample]:
        """Load samples from a file."""
        df = pd.read_csv(filepath)
        
        samples = []
        for _, row in df.iterrows():
            sample = DataSample(
                prompt=str(row['prompt']),
                response=str(row['response']),
                reference=str(row['reference']) if 'reference' in row and pd.notna(row['reference']) else None,
                rating=float(row['rating']) if 'rating' in row and pd.notna(row['rating']) else None,
                metadata=json.loads(row['metadata']) if 'metadata' in row and pd.notna(row['metadata']) else None
            )
            samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples from {filepath}")
        
        return samples

    def generate_human_feedback_dataset(self, size: int):
        import datetime
        import random
        import json
        import os
        from pathlib import Path

        output_dir = Path(self.data_config.human_feedback_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load conala train for questions
        dataset = self._load_conala_split('train')
        if len(dataset) < size:
            size = len(dataset)
        selected = random.sample(range(len(dataset)), size)

        for idx in selected:
            item = dataset[idx]
            question = item.get('rewritten_intent') or item.get('intent') or ""
            id_val = item.get('question_id') or random.randint(10000000, 99999999)
            good_answer = item.get('snippet') or "good code"
            other_idx = random.choice([i for i in range(len(dataset)) if i != idx])
            bad_answer = dataset[other_idx].get('snippet') or "pass"

            # Randomize L and R
            if random.random() > 0.5:
                answer_l = good_answer
                answer_r = bad_answer
                consistent_l = random.randint(1,2)
                correct_l = random.randint(1,2)
                useful_l = random.randint(1,2)
                consistent_r = random.randint(-2,-1)
                correct_r = random.randint(-2,-1)
                useful_r = random.randint(-2,-1)
            else:
                answer_l = bad_answer
                answer_r = good_answer
                consistent_l = random.randint(-2,-1)
                correct_l = random.randint(-2,-1)
                useful_l = random.randint(-2,-1)
                consistent_r = random.randint(1,2)
                correct_r = random.randint(1,2)
                useful_r = random.randint(1,2)

            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
            filename = f"{timestamp}-Synthetic.json"
            filepath = output_dir / filename

            data = {
                "name_input": "Synthetic",
                "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "address": "127.0.0.1",
                "comparison_slider": random.randint(-100, 100),
                "consistent_L": consistent_l,
                "correct_L": correct_l,
                "useful_L": useful_l,
                "consistent_R": consistent_r,
                "correct_R": correct_r,
                "useful_R": useful_r,
                "questions_df": [
                    {
                        "level_0": random.randint(0, 5000),
                        "index": random.randint(0, 500),
                        "ID": id_val,
                        "Question": question,
                        "Answer": answer_l,
                        "CSV_PATH": "synthetic_l.csv",
                        "CODE_FORMATTING": False,
                        "MODEL_TAG": "Model L"
                    },
                    {
                        "level_0": random.randint(0, 5000),
                        "index": random.randint(0, 500),
                        "ID": id_val,
                        "Question": question,
                        "Answer": answer_r,
                        "CSV_PATH": "synthetic_r.csv",
                        "CODE_FORMATTING": False,
                        "MODEL_TAG": "Model R"
                    }
                ]
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

        logger.info(f"Generated {size} synthetic human feedback files in {output_dir}")