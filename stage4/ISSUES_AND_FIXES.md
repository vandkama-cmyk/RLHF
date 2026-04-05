# Stage 4 — Known Issues and Fixes

Use **`train_fixed.py`** and **`integrated_system_fixed.py`** for new experiments. Legacy scripts are suffixed with **`_legacy.py`** (older behaviour, not recommended).

## Issues addressed in `_fixed` pipelines

| Issue | Description | Fix |
|--------|-------------|-----|
| **Static / proxy metrics** | Earlier runs sometimes logged placeholder or token-overlap proxies as if they were full BERTScore / CodeBLEU. | `_fixed` paths use `modern_rlhf.metrics` with explicit **uniform keys** (`codebleu` vs `codebleu_proxy`, `ruby` vs `ruby_like_heuristic`) and documented fallbacks. |
| **Dummy embeddings** | `DummyEmbeddingEncoder` produced deterministic vectors from hashes for demos; not representative of semantic quality. | `_fixed` integration prefers real encoder features where configured; legacy demos remain in `*_legacy.py` only. |
| **Self-reference in eval** | Comparing a generated answer to itself inflated BERTScore / ROUGE. | Validation uses true **reference** strings only; empty references are excluded from misleading self-comparisons. |
| **Hash collisions** | Using Python `hash()` for cache keys caused collisions across runs. | Stable keys (e.g. SHA-256 over question+answer) in downstream consumers (see Stage 5C LLM cache). |
| **Random / unstable labels** | Non-deterministic label noise broke reproducibility. | Fixed splits and seeds in `_fixed` training scripts; legacy paths may still use older sampling. |

## File naming

| Legacy (old) | Use instead |
|----------------|-------------|
| `integrated_system_legacy.py` | `integrated_system_fixed.py` |
| `train_improved_legacy.py` | `train_fixed.py` |
| Per-head scripts without `_fixed` anti-overfitting | `stage4A_consist` / `stage4B_corct` / `stage4C_useful` + `train_*.py` pointing at integrated systems |

## Stage 4C usefulness head

The usefulness MLP (`stage4C_useful/`) supports **focal loss** and class rebalancing via `pos_weight` to reduce collapse toward a single class. Enable debug logging with `debug_logging: true` in the config dict passed to `IntegratedTrainingPipeline`.
