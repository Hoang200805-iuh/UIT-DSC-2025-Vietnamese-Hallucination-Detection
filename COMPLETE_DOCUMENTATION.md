# 📚 UIT-DSC 2025 Challenge B - Documentation

**Challenge**: Vietnamese Hallucination Detection  
**Base Model**: PhoBERT-base (256 tokens)  
**Approach**: Advanced techniques (Retriever 2.0, TTA, R-Drop, FGM, EMA, Temperature Scaling)

## 📋 TABLE OF CONTENTS
1. [Quick Start](#quick-start)
2. [Project Setup](#project-setup)
3. [Module Reference](#modules)
4. [Usage Guide](#usage)

---

## 🚀 QUICK START {#quick-start}

### Project Structure
```
UIT_DSC/
├── src/                    # 10 Python modules
├── data/                   # Place CSV files here
├── models/                 # Model checkpoints
├── outputs/                # Results & submit.csv
├── configs/                # Configuration
├── notebooks/              # Your analysis
├── logs/                   # Training logs
├── requirements.txt        # Dependencies
└── uit-dsc.ipynb          # Your original model
```

### 10 Modules in `src/`
| Module | Purpose |
|--------|---------|
| `config.py` | 50+ hyperparameters (centralized) |
| `utils.py` | Logging, seeds, file I/O, device |
| `tokenizer.py` | Vietnamese NLP & IDF builder |
| `retriever.py` | Retriever 2.0 (context selection) |
| `model.py` | Model architecture (dual-head) |
| `augmentation.py` | R-Drop, FGM, EMA, focal loss |
| `training.py` | Training pipeline & callbacks |
| `inference.py` | Inference & TTA (9 passes) |
| `metrics.py` | Evaluation & temperature scaling |
| `__init__.py` | Package initialization |

### Installation & Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup in Python notebook or script
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src import config, utils, model
utils.set_all_seeds(config.SEED)
```

---

## 📊 PROJECT SETUP {#project-setup}

### Challenge Overview
- **Dataset**: ViHallu (Vietnamese Hallucination Detection)
- **Classes**: 3-way (no, intrinsic, extrinsic)
- **Base Model**: PhoBERT-base (256 tokens, 768 hidden)
- **Goal**: Maximize Macro F1-score

### Input/Output Format
**Input CSV**:
```
id,context,prompt,response,label
0,"Toyota, Honda...", "Xe nào?", "Xe Nhật...", extrinsic
```

**Output CSV** (submit.csv):
```
id,predict_label
0,extrinsic
1,no
2,intrinsic
```

### Model Features
- ✅ Retriever 2.0 (IDF + neighbor + dynamic budget)
- ✅ Global correlation features (18 dims)
- ✅ Dual-head training (main + hall + IE)
- ✅ TTA with 9 retriever views
- ✅ R-Drop, FGM, EMA augmentation
- ✅ 5-Fold cross-validation
- ✅ Temperature scaling calibration

**Expected**: OOF F1 ~0.82-0.85, Test F1 ~0.80-0.83

---

## 🔧 MODULE REFERENCE {#modules}

### 1. Configuration (`config.py`)
```python
from src import config
config.MODEL_NAME = "vinai/phobert-base"
config.MAX_LEN = 256
config.SEED = 42
config.EPOCHS = 5
config.BATCH_SIZE = 6
config.LEARNING_RATE = 2e-5
config.TTA_PASSES = 9
```

### 2. Utilities (`utils.py`)
```python
from src import utils
utils.set_all_seeds(42)              # Reproducibility
utils.setup_logging("train.log")     # Logging
device = utils.get_device()          # "cuda" or "cpu"
utils.save_json(data, "file.json")   # File I/O
```

### 3. Tokenizer & IDF (`tokenizer.py`)
```python
from src import tokenizer
idf_builder = tokenizer.IDFBuilder()
idf_builder.build(texts)
tokens = tokenizer.simple_tokenize("text")
```

### 4. Retriever (`retriever.py`)
```python
from src import retriever
ret = retriever.Retriever(idf_dict)
selected = ret.retrieve(context, prompt, response, max_tokens=200)
```

### 5. Model (`model.py`)
```python
from src import model
my_model = model.create_model()  # PhoBERT + custom head
```

### 6. Augmentation (`augmentation.py`)
```python
from src import augmentation
rdrop = augmentation.RDropLoss(alpha=0.5)
fgm = augmentation.FGM(model, eps=1.0)
ema = augmentation.EMA(model, decay=0.9999)
```

### 7. Training (`training.py`)
```python
from src import training
optimizer = training.get_optimizer(model, lr=2e-5)
scheduler = training.get_scheduler(optimizer, num_steps)
trainer = training.WeightedTrainer(model=model, use_ema=True)
```

### 8. Inference (`inference.py`)
```python
from src import inference
tta_logits = inference.inference_with_tta(model, test_loader, num_passes=9)
processor = inference.InferencePostProcessor(config.ID_TO_LABEL)
results = processor.process(tta_logits)
```

### 9. Metrics (`metrics.py`)
```python
from src import metrics
mc = metrics.MetricsComputer()
all_metrics = mc.compute(y_true, y_pred, logits)
temp_scaler = metrics.TemperatureScaling()
temps = temp_scaler.fit(oof_logits, oof_labels)
```

---

## 💡 USAGE GUIDE {#usage}

### Workflow 1: Full Training
```python
from src import config, utils, model, training
import pandas as pd

utils.set_all_seeds(config.SEED)
train_df = pd.read_csv(config.TRAIN_PATH)
my_model = model.create_model()
trainer = training.WeightedTrainer(model=my_model)
trainer.train()
```

### Workflow 2: Inference with TTA
```python
from src import config, inference
import pandas as pd

test_logits = inference.inference_with_tta(model, test_loader, num_passes=9)
processor = inference.InferencePostProcessor(config.ID_TO_LABEL)
results = processor.process(test_logits)

submit_df = pd.DataFrame({
    'id': test_ids,
    'predict_label': results['predictions']
})
submit_df.to_csv('outputs/submit.csv', index=False)
```

### Workflow 3: Calibration
```python
from src import metrics
temp_scaler = metrics.TemperatureScaling()
temps = temp_scaler.fit(oof_logits, oof_labels)
test_logits_calibrated = temp_scaler.apply(test_logits)
```

### Quick Tips
```python
# Change config
config.SEED = 123
config.EPOCHS = 10

# Set reproducible seeds
utils.set_all_seeds(config.SEED)

# Get device
device = utils.get_device()

# Count parameters
num_params = utils.count_parameters(model)

# Save/load metrics
utils.save_json(metrics, "metrics.json")
data = utils.load_json("metrics.json")
```

---

## 🎯 QUICK REFERENCE

### Key Hyperparameters
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
CLASS_BIAS = [-0.05, 0.02, 0.02]
```

### Class Labels
```python
LABEL_TO_ID = {"no": 0, "intrinsic": 1, "extrinsic": 2}
ID_TO_LABEL = {0: "no", 1: "intrinsic", 2: "extrinsic"}
```

### Data Format
- **Train**: `data/vihallu-train.csv` (~1000 samples)
- **Test**: `data/vihallu-public-test.csv` (~200-400 samples)
- K-Fold (5 folds)
- Optimizer: AdamW (lr=2e-5)
- Scheduler: Linear warmup + cosine decay
- Loss: CE + Hall + IE + FGM + R-Drop
- Augmentation: R-Drop, FGM, EMA
- Batch size: 6 (+ grad accumulation ×2)

---

## ⚠️ TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| Module not found | Ensure `src/` folder is in project root |
| Out of memory | Reduce `BATCH_SIZE` in config.py |
| Wrong predictions | Check temperature scaling & class bias |
| Import error | Verify all files in `src/` exist |
| Slow training | Use GPU (`CUDA_VISIBLE_DEVICES=0`) or reduce batch size |

---

## 📝 BEST PRACTICES

✅ **DO**
- Use centralized config.py
- Set seeds for reproducibility
- Use logger instead of print()
- Save metrics after training
- Use temperature scaling

❌ **DON'T**
- Hardcode hyperparameters
- Use wildcard imports
- Skip seed setting
- Train without checkpoints

---

## 🏁 SUMMARY

**Repository Structure**: 28 files (10 modules, 6 docs, configs, directories)  
**Code**: ~2000 lines | **Documentation**: ~3000 lines  
**Status**: ✅ Production Ready

### Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place data in `data/` folder
4. Follow usage examples in this documentation
5. Start training!

### Project Links
- **Challenge**: UIT-DSC 2025 Challenge B (Hallucination Detection)
- **Dataset**: ViHallu (Vietnamese Hallucination Detection)
- **Base Model**: [PhoBERT](https://huggingface.co/vinai/phobert-base)

---

**Made for UIT-DSC 2025 Challenge** | Last Updated: January 2025
