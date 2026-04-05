"""Quick metrics computation on a small sample."""
import pandas as pd
import sys
sys.path.insert(0, '.')
from modern_rlhf.metrics import ModernMetricsEvaluator, get_metric_float

# Load a small sample
df = pd.read_csv('clasifNN/datasets_for_eval/T2C-CONALA-CODEGEN-FINETUNED-SO.csv')
predictions = df['ModelAnswer'].fillna('').astype(str).tolist()[:100]
references = df['Answer'].fillna('').astype(str).tolist()[:100]

print(f'Computing metrics on {len(predictions)} samples...')

evaluator = ModernMetricsEvaluator()
results = evaluator.compute_all_metrics(predictions, references)

print(f'Ruby: {get_metric_float(results, "ruby", "ruby_like_heuristic"):.4f}')
print(f'CodeBLEU: {get_metric_float(results, "codebleu", "codebleu_proxy"):.4f}')
print(f'BERTScore: {results["bertscore"].score:.4f}')
print(f'ROUGE: {results["rouge"].score:.4f}')
print(f'BLEU: {results["bleu"].score:.4f}')
