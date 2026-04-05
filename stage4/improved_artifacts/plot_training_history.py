import json
import matplotlib.pyplot as plt

with open('training_history_fixed.json') as f:
    history = json.load(f)

epochs = [entry['epoch'] for entry in history]

# Losses
plt.figure(figsize=(10, 5))
plt.plot(epochs, [entry['train_loss'] for entry in history], label='Train Loss')
plt.plot(epochs, [entry['val_loss'] for entry in history], label='Val Loss')
plt.title('Losses over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('losses.png')

# Accuracies
plt.figure(figsize=(10, 5))
for metric in ['consistent_acc', 'correct_acc', 'useful_acc']:
    plt.plot(epochs, [entry[f'train_{metric}'] for entry in history], label=f'Train {metric}')
    plt.plot(epochs, [entry[f'val_{metric}'] for entry in history], label=f'Val {metric}')
plt.title('Accuracies over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracies.png')

# F1 Scores
plt.figure(figsize=(10, 5))
for metric in ['consistent_f1', 'correct_f1', 'useful_f1']:
    plt.plot(epochs, [entry[f'val_{metric}'] for entry in history], label=f'Val {metric}')
plt.title('F1 Scores over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.savefig('f1_scores.png')

# Other metrics
other_metrics = ['bertscore', 'codebleu', 'bleu', 'rouge', 'ruby']
plt.figure(figsize=(10, 5))
for metric in other_metrics:
    plt.plot(epochs, [entry[f'val_{metric}'] for entry in history], label=metric.capitalize())
plt.title('Other Metrics over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.savefig('other_metrics.png')
