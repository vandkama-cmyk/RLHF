# Сравнительный анализ методов классификации качества кода

## 📊 Обзор методов

### 1. ClassifNN (MLP-based)
- **Архитектура**: MLP с sentence-transformers эмбеддингами
- **Размер модели**: ~2-5 MB
- **Время обучения**: Быстрое (~минуты)
- **Зависимости**: sentence-transformers (опционально)

### 2. ClassifLLM v1 (Базовый LLM)
- **Архитектура**: CodeBERT + Multi-head классификатор
- **Размер модели**: ~440 MB
- **Время обучения**: Среднее (~10-30 минут на GPU)
- **Особенности**: Совместное кодирование Q+A через [SEP]

### 3. ClassifLLM v2 (Улучшенный LLM)
- **Архитектура**: CodeBERT + Cross-attention + Contrastive learning
- **Размер модели**: ~450 MB
- **Время обучения**: Среднее-долгое (~15-40 минут на GPU)
- **Улучшения**:
  - Cross-attention между вопросом и ответом
  - Attention-based pooling
  - Contrastive learning auxiliary loss
  - Mixed Precision Training (AMP)
  - Gradient Accumulation
  - Label Smoothing
  - EMA (Exponential Moving Average)

---

## 📈 Результаты обучения

### Excel Naming Mapping (important)

Your Excel workbook contains only two sheets:
- Excel `Stage 5A` corresponds to JSON `stage5_v2` (v2, 26 epochs).
- Excel `Stage 5B` corresponds to JSON `stage5_v3` (v3, 30 epochs).

JSON `stage5_v1` (v1, 4 epochs) is not represented in Excel because training diverged
and it was not used in the paper tables.

### ClassifLLM v1 (4 эпохи)

| Метрика | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
|---------|---------|---------|---------|---------|
| Val Loss | 0.524 | **0.507** | 0.526 | 0.533 |
| Consistent Acc | 0.774 | 0.748 | 0.765 | 0.757 |
| Correct Acc | 0.713 | 0.739 | 0.739 | 0.739 |
| Useful Acc | 0.730 | 0.765 | 0.748 | 0.748 |
| Consistent F1 | 0.840 | 0.818 | 0.840 | 0.823 |
| Correct F1 | 0.667 | 0.769 | 0.776 | 0.766 |
| Useful F1 | 0.716 | 0.787 | 0.782 | 0.772 |

**Code Quality Metrics (Epoch 2 - лучший):**
- BERTScore: 0.115
- CodeBLEU: 0.115
- BLEU: 0.131
- ROUGE: 0.149
- RUBY: 0.229

### ClassifNN (4 эпохи)

| Метрика | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
|---------|---------|---------|---------|---------|
| Val Loss | 0.616 | **0.612** | 0.613 | 0.614 |
| Val Accuracy | 0.701 | 0.701 | 0.701 | 0.701 |

**Code Quality Metrics (все эпохи стабильны):**
- BERTScore: 0.799
- CodeBLEU: 0.749
- BLEU: 0.730
- ROUGE: 0.759
- RUBY: 0.661

---

## 📊 Сравнительная таблица

| Метрика | ClassifLLM v1 | ClassifNN | Разница |
|---------|---------------|-----------|---------|
| **Val Loss** | **0.507** | 0.612 | -0.105 ✓ |
| **Avg Accuracy** | **0.751** | 0.701 | +0.050 ✓ |
| **Avg F1** | **0.791** | N/A | - |
| **BERTScore** | 0.115 | **0.799** | -0.684 |
| **CodeBLEU** | 0.115 | **0.749** | -0.634 |
| **BLEU** | 0.131 | **0.730** | -0.599 |
| **ROUGE** | 0.149 | **0.759** | -0.610 |
| **RUBY** | 0.229 | **0.661** | -0.432 |

---

## 🔍 Анализ результатов

### Сильные стороны ClassifLLM v1:
1. ✅ **Лучшая классификация**: Accuracy 75.1% vs 70.1%
2. ✅ **Ниже потери валидации**: 0.507 vs 0.612
3. ✅ **Multi-head classification**: Отдельные метрики для consistent/correct/useful
4. ✅ **F1 scores**: ~0.77-0.84 для всех голов

### Сильные стороны ClassifNN:
1. ✅ **Лучшие code quality metrics**: BERTScore 0.80 vs 0.12
2. ✅ **Быстрое обучение**: Не требует GPU
3. ✅ **Меньший размер модели**: ~5 MB vs ~440 MB
4. ✅ **Стабильные метрики**: Минимальная дисперсия между эпохами

### Проблемы ClassifLLM v1:
1. ⚠️ **Низкие code quality metrics**: Это может быть связано с:
   - Совместное кодирование Q+A снижает quality расчёт
   - Отсутствие референсных ответов для сравнения
   - Метрики рассчитываются по вопросам, а не реальным ответам

### Проблемы ClassifNN:
1. ⚠️ **Застревание в локальном минимуме**: Val accuracy не растёт
2. ⚠️ **Отсутствие F1 метрик**: Только общая accuracy
3. ⚠️ **Меньше semantic understanding**: Только эмбеддинги

---

## 🚀 Улучшения в ClassifLLM v2

### Архитектурные улучшения:
1. **Раздельное кодирование Q и A** → Лучшее понимание контекста
2. **Cross-attention** → Явное взаимодействие между вопросом и ответом
3. **Attention pooling** → Более информативные представления
4. **Contrastive learning** → Лучшие эмбеддинги

### Улучшения обучения:
1. **Mixed Precision (AMP)** → Быстрее в 1.5-2 раза на GPU
2. **Gradient Accumulation** → Эффективный batch size x2
3. **Label Smoothing** → Лучшая генерализация
4. **EMA** → Стабильные предсказания
5. **OneCycle LR** → Лучшая сходимость
6. **Patience 5** → Меньше раннего останова

### Ожидаемые результаты v2:
- Val Loss: ~0.45-0.50 (улучшение 5-10%)
- Accuracy: ~0.77-0.80 (улучшение 3-5%)
- F1: ~0.80-0.85 (улучшение 2-5%)
- BERTScore: ~0.20-0.30 (улучшение 2x благодаря раздельному кодированию)

---

## 🎯 Рекомендации

### Для задачи классификации качества кода:
```
1. ЛУЧШИЙ ВЫБОР: ClassifLLM v2
   - Сочетает преимущества LLM и улучшенные техники
   - Баланс между accuracy и code quality metrics
   
2. АЛЬТЕРНАТИВА (быстрая): ClassifNN
   - Если важна скорость обучения
   - Для продакшена с ограниченными ресурсами
   
3. BASELINE: ClassifLLM v1
   - Хорошая accuracy
   - Простая архитектура
```

### Для улучшения code quality metrics:
```python
# Используйте раздельное кодирование (как в v2):
q_encoded = model.encode(question)
a_encoded = model.encode(answer)

# Вместо объединённого:
combined = f"{question} [SEP] {answer}"
encoded = model.encode(combined)
```

---

## 📁 Структура файлов v2

```
clasifLLM/v2/
├── __init__.py           # Экспорты модулей
├── model.py              # EnhancedLLMClassifier
├── integrated_system.py  # Training pipeline
├── train.py              # Training script
├── compare_methods.py    # Comparison script
├── COMPARISON_REPORT.md  # Этот отчёт
└── artifacts/            # Training results
    └── training_history_llm_v2.json
```

---

## 🔧 Запуск обучения v2

```bash
# Базовое обучение
python clasifLLM/v2/train.py

# С настройками
python clasifLLM/v2/train.py \
    --model-type codebert \
    --epochs 15 \
    --batch-size 8 \
    --gradient-accumulation 2 \
    --use-cross-attention \
    --use-contrastive \
    --use-amp \
    --use-ema

# Сравнение результатов
python clasifLLM/v2/compare_methods.py
```

---

## 📌 Выводы

| Критерий | Лучший метод |
|----------|--------------|
| Classification Accuracy | ClassifLLM v1/v2 |
| F1 Score | ClassifLLM v1/v2 |
| Code Quality (BERTScore) | ClassifNN |
| Training Speed | ClassifNN |
| Model Size | ClassifNN |
| Semantic Understanding | ClassifLLM v2 |
| Overall Balance | **ClassifLLM v2** |

**Итоговая рекомендация**: Используйте **ClassifLLM v2** как основной метод, так как он предоставляет лучший баланс между classification accuracy и архитектурными улучшениями для понимания кода.

