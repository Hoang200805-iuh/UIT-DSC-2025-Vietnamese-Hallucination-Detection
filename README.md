# UIT-DSC 2025 Challenge B: Vietnamese Hallucination Detection

A modular, production-ready solution for detecting and classifying hallucinations in Vietnamese language model responses.

## 🎯 Challenge Overview

**Task**: Classify whether an LLM response contains hallucinations
- **Classes**: 3-way classification
  - `no` - No hallucination
  - `intrinsic` - Contradicts context
  - `extrinsic` - Contains new facts not in context
- **Dataset**: ViHallu (Vietnamese Hallucination Detection)
- **Base Model**: [PhoBERT-base](https://huggingface.co/vinai/phobert-base)
- **Metric**: Macro F1-score

## 📦 Features

✅ **Modular Architecture** - 10 independent, reusable modules  
✅ **Advanced Techniques** - Retriever 2.0, TTA, R-Drop, FGM, EMA, Temperature Scaling  
✅ **Production Ready** - Logging, error handling, checkpoint management  
✅ **Well Documented** - Docstrings, type hints, usage examples  
✅ **Easy Configuration** - Centralized hyperparameter management  

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Hoang200805-iuh/UIT-DSC-2025-Vietnamese-Hallucination-Detection
cd uit-dsc-2025
pip install -r requirements.txt
```

### 2. Prepare Data
Place your CSV files in `data/` folder:
```
data/
├── vihallu-train.csv      # Training set
└── vihallu-public-test.csv # Test set
```

**CSV Format**:
```
id,context,prompt,response,label
0,"Context text...", "Question?", "Response...", extrinsic
```

### 3. Setup & Train
```python
from src import config, utils, model, training
import pandas as pd

# Setup
utils.set_all_seeds(config.SEED)
train_df = pd.read_csv(config.TRAIN_PATH)

# Create & train
my_model = model.create_model()
trainer = training.WeightedTrainer(model=my_model)
trainer.train()
```

### 4. Generate Predictions
```python
from src import inference
import pandas as pd

# Inference with TTA
test_logits = inference.inference_with_tta(model, test_loader, num_passes=9)

# Save submission
processor = inference.InferencePostProcessor(config.ID_TO_LABEL)
results = processor.process(test_logits)

submit_df = pd.DataFrame({
    'id': test_ids,
    'predict_label': results['predictions']
})
submit_df.to_csv('outputs/submit.csv', index=False)
```

## 📁 Project Structure

```
.
├── src/                          # 10 Python modules
│   ├── config.py                 # Centralized configuration (50+ settings)
│   ├── utils.py                  # Utilities (logging, seeds, I/O)
│   ├── tokenizer.py              # Vietnamese NLP & IDF builder
│   ├── retriever.py              # Retriever 2.0 (context selection)
│   ├── model.py                  # Model architecture (dual-head)
│   ├── augmentation.py           # Training techniques (R-Drop, FGM, EMA)
│   ├── training.py               # Training pipeline & callbacks
│   ├── inference.py              # Inference & TTA
│   ├── metrics.py                # Evaluation & calibration
│   └── __init__.py
│
├── configs/                      # Configuration files
│   └── config_base.yaml
│
├── data/                         # Data directory
│   ├── vihallu-train.csv
│   └── vihallu-public-test.csv
│
├── models/                       # Model checkpoints
├── outputs/                      # Results & predictions
├── notebooks/                    # Analysis notebooks
├── logs/                         # Training logs
├── requirements.txt              # Python dependencies
├── COMPLETE_DOCUMENTATION.md     # Detailed documentation
└── README.md                     # This file
```

## 🔧 Module Overview

| Module | Purpose |
|--------|---------|
| **config.py** | 50+ centralized hyperparameters |
| **utils.py** | Logging, seeds, device, file I/O |
| **tokenizer.py** | Vietnamese text processing & IDF |
| **retriever.py** | Retriever 2.0 (IDF + neighbor + dynamic budget) |
| **model.py** | PhoBERT + custom dual-head architecture |
| **augmentation.py** | R-Drop, FGM, EMA, Focal Loss, etc. |
| **training.py** | Training pipeline with callbacks |
| **inference.py** | TTA inference & post-processing |
| **metrics.py** | Evaluation & temperature scaling |

## 📊 Model Architecture

```
PhoBERT-base (256 tokens, 768 hidden)
    ↓
[Global Correlation Features: 18 dims]
    ↓
[Dual-Head Architecture]
├─ Main Classifier (3-way)
├─ Hall Classifier (binary)
└─ IE Classifier (binary)
    ↓
[Inference Fusion: Main 0.8 + Aux 0.2]
[TTA: 9 retriever views]
[Temperature Scaling: Per-class calibration]
```

## 🎯 Training Strategy

- **K-Fold**: 5-fold cross-validation
- **Optimizer**: AdamW (lr=2e-5)
- **Scheduler**: Linear warmup + cosine decay
- **Epochs**: 5 per fold
- **Batch Size**: 6 (with gradient accumulation ×2)
- **Loss**: CE + Hall + IE + R-Drop + FGM
- **Augmentation**: R-Drop, FGM, EMA
- **Expected F1**: OOF ~0.82-0.85, Test ~0.80-0.83

## 💡 Usage Examples

### Change Configuration
```python
from src import config
config.SEED = 123
config.EPOCHS = 10
config.BATCH_SIZE = 8
```

### Set Reproducible Seeds
```python
from src import utils
utils.set_all_seeds(42)
```

### Get Device Information
```python
device = utils.get_device()  # "cuda" or "cpu"
```

### Build IDF Index
```python
from src import tokenizer
idf_builder = tokenizer.IDFBuilder()
idf_builder.build(texts)
```

### Create Model
```python
from src import model
my_model = model.create_model()  # PhoBERT + custom head
```

### Inference with TTA
```python
from src import inference
tta_logits = inference.inference_with_tta(model, test_loader, num_passes=9)
```

### Temperature Scaling
```python
from src import metrics
scaler = metrics.TemperatureScaling()
temps = scaler.fit(oof_logits, oof_labels)
calibrated = scaler.apply(test_logits)
```

## 🎓 Key Concepts

### Hallucination Detection
- **Intrinsic**: Response contradicts given context
- **Extrinsic**: Response introduces facts not in context
- **No Hallucination**: Response is consistent with context

### Retriever 2.0
- IDF-based context sentence ranking
- Multi-view retrieval (IDF, numeric boost, capitalization boost)
- Dynamic budget allocation based on token counts

### Test Time Augmentation (TTA)
- 9 forward passes with different retriever configurations
- Predictions averaged for robust inference
- Significantly improves performance

### Temperature Scaling
- Per-class temperature learned on validation set
- Improves confidence calibration
- Reduces overconfidence on common classes

## 📝 Best Practices

✅ **DO**
- Use centralized `config.py` for all hyperparameters
- Set seeds before training with `utils.set_all_seeds()`
- Use logger instead of print statements
- Save and track metrics after training
- Use temperature scaling for calibration
- Use TTA for final predictions

❌ **DON'T**
- Hardcode hyperparameters in notebooks
- Use wildcard imports
- Skip reproducibility setup
- Train without checkpoints
- Forget to validate on proper splits

## ⚙️ Configuration

Key hyperparameters in `src/config.py`:
```python
MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 256
SEED = 42
FOLDS = 5
EPOCHS = 5
BATCH_SIZE = 6
LEARNING_RATE = 2e-5
TTA_PASSES = 9
FUSION_MAIN_WEIGHT = 0.8
```

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Ensure `src/` folder exists in project root |
| Out of memory | Reduce `BATCH_SIZE` in config |
| Wrong predictions | Verify temperature scaling is applied |
| Slow training | Use GPU with `CUDA_VISIBLE_DEVICES=0` |
| Module not found | Check that all files in `src/` exist |

## 📚 Documentation

- **[COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md)** - Detailed reference guide with all modules and usage patterns

## 🤝 Contributing

Contributions are welcome! Please:
1. Create a new branch
2. Make your changes
3. Test your code
4. Submit a pull request

## 📄 License

This project is part of the UIT-DSC 2025 Challenge.

## 🏆 Challenge Info

- **Event**: UIT-DSC 2025 (University of Information Technology - Data Science Challenge)
- **Challenge**: Challenge B - Vietnamese Hallucination Detection
- **Focus**: NLP, Text Classification, Model Calibration

---

**Status**: ✅ Production Ready  
**Last Updated**: January 2025  
**Python**: 3.8+  
**CUDA**: 11.7+ (optional, for GPU training)
