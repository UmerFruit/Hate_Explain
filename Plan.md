# HateXplain Python 3.7 → 3.13 Migration Plan

## Executive Summary
This document outlines the comprehensive migration plan to update HateXplain from Python 3.7 to Python 3.13. The project is a hate speech detection and explanation system using BERT and RNN models with attention mechanisms.

**STATUS: MIGRATION COMPLETED** ✅

The following changes have been applied to the codebase:
- Updated `requirements.txt` with Python 3.13 compatible packages
- Fixed all deprecated imports (transformers, sklearn, keras, tqdm)
- Removed Neptune logging dependencies (now uses local logging)
- Fixed torch.uint8 deprecation to torch.bool
- Implemented custom pad_sequences to replace keras dependency

---

## Table of Contents
1. [Files to Delete](#1-files-to-delete)
2. [Dependency Updates](#2-dependency-updates)
3. [Code Changes Required](#3-code-changes-required)
4. [File-by-File Analysis](#4-file-by-file-analysis)
5. [Testing Strategy](#5-testing-strategy)

---

## 1. Files to Delete

### Unnecessary Files (Can be safely removed)
Based on the `.gitignore` and project structure, these files/folders can be deleted if present:
- `*.pkl`, `*.pickle` files (cached data - will be regenerated)
- `*.pyc`, `__pycache__/` directories
- `.ipynb_checkpoints/`
- `*.model` files (will be regenerated during training)
- `explanations_dicts/` (generated output)
- `Dataset_Eraser_Format/` (generated)
- `Saved/` directory (model checkpoints)
- `Data/Evaluation/` (generated)
- `glove.*.txt` files (large embedding files downloaded separately)

### Files to Keep
- All `.py` source files
- All `.ipynb` notebooks
- `Data/dataset.json` (main dataset)
- `Data/classes.npy`, `Data/classes_two.npy` (label files)
- `Data/post_id_divisions.json` (train/val/test splits)
- `best_model_json/` directory (model configurations)
- `requirements.txt` (will be updated)
- All documentation files (`README.md`, `LICENSE`, etc.)

---

## 2. Dependency Updates

### Current requirements.txt (Python 3.7)
```
scipy==1.4.1
spacy==2.3.2
tqdm==4.43.0
Keras==2.3.1
waiting==1.4.1
ekphrasis==0.5.1
pandas==1.0.3
transformers==2.5.1
lime==0.2.0.1
numpy==1.16.3
matplotlib==3.2.1
gensim==3.8.1
neptune_client==0.4.107
knockknock==0.1.7
torch==1.1.0
apex==0.9.10dev
dataclasses==0.8
GPUtil==1.4.0
scikit_learn==0.23.2
```

### Updated requirements.txt (Python 3.13)
```
# Core ML/DL Libraries
torch>=2.1.0
transformers>=4.36.0
scipy>=1.11.0
numpy>=1.26.0
pandas>=2.1.0
scikit-learn>=1.3.0

# NLP Libraries
spacy>=3.7.0
gensim>=4.3.0
ekphrasis>=0.5.4

# Visualization
matplotlib>=3.8.0

# Utilities
tqdm>=4.66.0
lime>=0.2.0.1
GPUtil>=1.4.0

# Optional (Neptune for experiment tracking - may need update)
# neptune>=1.8.0

# Additional required packages
more-itertools>=10.0.0
```

### Dependency Changes Explained

| Old Package | New Package | Reason |
|------------|-------------|--------|
| `torch==1.1.0` | `torch>=2.1.0` | Python 3.13 support, new features |
| `transformers==2.5.1` | `transformers>=4.36.0` | Major API changes, Python 3.13 support |
| `numpy==1.16.3` | `numpy>=1.26.0` | Python 3.13 support |
| `pandas==1.0.3` | `pandas>=2.1.0` | Python 3.13 support |
| `scipy==1.4.1` | `scipy>=1.11.0` | Python 3.13 support |
| `spacy==2.3.2` | `spacy>=3.7.0` | Major API changes |
| `gensim==3.8.1` | `gensim>=4.3.0` | API changes for KeyedVectors |
| `Keras==2.3.1` | *(removed)* | Using keras from tensorflow or standalone |
| `scikit_learn==0.23.2` | `scikit-learn>=1.3.0` | Package name fix, Python 3.13 |
| `dataclasses==0.8` | *(removed)* | Built into Python 3.7+ |
| `apex==0.9.10dev` | *(removed)* | Not needed, PyTorch has native amp |
| `neptune_client==0.4.107` | `neptune>=1.8.0` | Package renamed |
| `knockknock==0.1.7` | *(optional)* | May be outdated |

---

## 3. Code Changes Required

### 3.1 Critical Breaking Changes

#### A. Transformers Library (2.5.1 → 4.x)

**File: `Models/bertModels.py`**
```python
# OLD (line 1):
from transformers.modeling_bert import *

# NEW:
from transformers import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss
import torch.nn as nn
```

**File: `manual_training_inference.py`**
```python
# OLD imports at top:
from transformers import *
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig

# NEW:
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    BertConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW  # AdamW moved to torch.optim
```

#### B. Keras pad_sequences Removal

**File: `TensorDataset/dataLoader.py`**
```python
# OLD (line 3):
from keras.preprocessing.sequence import pad_sequences

# NEW (use torch.nn.utils.rnn or custom implementation):
from torch.nn.utils.rnn import pad_sequence
# OR implement custom padding:
def pad_sequences(sequences, maxlen, dtype="long", value=0, truncating="post", padding="post"):
    import numpy as np
    padded = np.full((len(sequences), maxlen), value, dtype=dtype)
    for i, seq in enumerate(sequences):
        if truncating == "post":
            trunc = seq[:maxlen]
        else:
            trunc = seq[-maxlen:]
        if padding == "post":
            padded[i, :len(trunc)] = trunc
        else:
            padded[i, -len(trunc):] = trunc
    return padded
```

#### C. Gensim API Changes (3.x → 4.x)

**File: `TensorDataset/datsetSplitter.py`**
```python
# OLD:
word2vecmodel1 = KeyedVectors.load("Data/word2vec.model")

# This should still work, but loading method may differ
# The save/load API is similar in gensim 4.x
```

**File: `Example_HateExplain.ipynb` (cell for glove conversion)**
```python
# OLD:
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec('Data/glove.42B.300d.txt', 'Data/glove.42B.300d_w2v.txt')

# NEW (gensim 4.x):
# glove2word2vec still exists but import may need adjustment
from gensim.scripts.glove2word2vec import glove2word2vec
# OR use KeyedVectors directly:
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('Data/glove.42B.300d.txt', no_header=True)
```

#### D. Spacy Changes (2.x → 3.x)

**File: `Preprocess/preProcess.py`**
```python
# OLD (line 8):
nlp2 = spacy.load('en_core_web_sm')

# NEW: Same, but need to download the model differently
# python -m spacy download en_core_web_sm
```

The spacy imports for tokenization rules have changed paths:
```python
# These imports may need updating based on spacy 3.x structure
from spacy.lang.char_classes import LIST_PUNCT, LIST_ELLIPSES, LIST_QUOTES, LIST_CURRENCY
from spacy.lang.char_classes import LIST_ICONS, HYPHENS, CURRENCY, UNITS
from spacy.lang.char_classes import CONCAT_QUOTES, ALPHA_LOWER, ALPHA_UPPER, ALPHA, PUNCT
```

#### E. scikit-learn API Changes

**File: `TensorDataset/datsetSplitter.py` and others**
```python
# OLD:
from sklearn.utils import class_weight
params['weights'] = class_weight.compute_class_weight('balanced', np.unique(y_test), y_test)

# NEW (sklearn 1.x):
from sklearn.utils.class_weight import compute_class_weight
params['weights'] = compute_class_weight('balanced', classes=np.unique(y_test), y=y_test)
```

#### F. Neptune Client Changes

**File: `manual_training_inference.py`**
```python
# OLD:
import neptune
neptune.init(project_name, api_token=api_token)
neptune.create_experiment(...)
neptune.log_metric(...)

# NEW (neptune 1.x):
import neptune
run = neptune.init_run(project=project_name, api_token=api_token)
run["metric_name"].append(value)
```
*Note: Neptune logging is optional and can be disabled by setting `params['logging']='local'`*

#### G. PyTorch Tensor Type Changes

**File: `TensorDataset/dataLoader.py`**
```python
# OLD (line 47):
masks = torch.tensor(np.array(att_masks), dtype=torch.uint8)

# NEW (torch.uint8 is deprecated for boolean masks):
masks = torch.tensor(np.array(att_masks), dtype=torch.bool)
```

### 3.2 Python 3.10+ Syntax Opportunities (Optional)

These are optional improvements using newer Python syntax:

```python
# Match statements (Python 3.10+)
# Dictionary union operator |= (Python 3.9+)
# Type hints with | instead of Union (Python 3.10+)
```

---

## 4. File-by-File Analysis

### Core Files Requiring Changes

| File | Priority | Changes Required |
|------|----------|------------------|
| `Models/bertModels.py` | HIGH | Update transformers imports |
| `TensorDataset/dataLoader.py` | HIGH | Remove keras dependency, fix torch.uint8 |
| `manual_training_inference.py` | HIGH | Update imports, AdamW location |
| `Preprocess/preProcess.py` | MEDIUM | Verify spacy 3.x compatibility |
| `TensorDataset/datsetSplitter.py` | MEDIUM | Update sklearn class_weight call |
| `testing_with_rational.py` | MEDIUM | Same as manual_training_inference.py |
| `testing_for_bias.py` | MEDIUM | Similar updates |
| `testing_with_lime.py` | MEDIUM | Similar updates |
| `requirements.txt` | HIGH | Complete rewrite |

### Files Likely Compatible (Minor or No Changes)

| File | Notes |
|------|-------|
| `Models/attentionLayer.py` | Pure PyTorch, should work |
| `Models/otherModels.py` | Pure PyTorch, should work |
| `Models/utils.py` | Minor: tqdm_notebook → tqdm.notebook |
| `Preprocess/attentionCal.py` | NumPy operations, should work |
| `Preprocess/spanMatcher.py` | Minor import adjustments |
| `Preprocess/utils.py` | Pure Python, compatible |
| `Preprocess/dataCollect.py` | Depends on other modules |
| `eraserbenchmark/` | Dataclasses built-in, should work |

### Notebook Files

| Notebook | Changes Required |
|----------|------------------|
| `Example_HateExplain.ipynb` | Update imports in cells, gensim usage |
| `Bias_Calculation_NB.ipynb` | Minor updates |
| `Explainability_Calculation_NB.ipynb` | Minor updates |

---

## 5. Testing Strategy

### Phase 1: Environment Setup
1. Create fresh Python 3.13 virtual environment
2. Install updated requirements
3. Download spacy model: `python -m spacy download en_core_web_sm`
4. Download GloVe embeddings (if needed)

### Phase 2: Import Testing
```bash
# Test each module can be imported
python -c "from Models.bertModels import *"
python -c "from Models.otherModels import *"
python -c "from Preprocess.dataCollect import *"
python -c "from TensorDataset.dataLoader import *"
```

### Phase 3: Data Loading Test
```python
from Preprocess.dataCollect import get_annotated_data
params = {
    'num_classes': 3,
    'data_file': 'Data/dataset.json',
    'class_names': 'Data/classes.npy'
}
data = get_annotated_data(params)
print(f"Loaded {len(data)} samples")
```

### Phase 4: Full Notebook Execution
Run `Example_HateExplain.ipynb` cells sequentially, fixing issues as they arise.

---

## 6. Implementation Order

### Step 1: Update requirements.txt
Create new requirements file with updated versions.

### Step 2: Fix Models/bertModels.py
Update transformers imports - this is the most critical change.

### Step 3: Fix TensorDataset/dataLoader.py
Remove keras dependency, implement custom pad_sequences.

### Step 4: Fix manual_training_inference.py
Update all imports for new library versions.

### Step 5: Fix sklearn usage
Update class_weight API calls across all files.

### Step 6: Fix remaining files
Update testing scripts and other modules.

### Step 7: Test Example_HateExplain.ipynb
Run through notebook and fix any remaining issues.

---

## 7. Known Issues and Workarounds

### Issue 1: CUDA Compatibility
- PyTorch 2.x may require different CUDA versions
- Solution: Install CPU-only version for testing, or match CUDA version

### Issue 2: Transformers Model Loading
- Old saved BERT models may not load with new transformers
- Solution: May need to retrain models or use conversion utilities

### Issue 3: Neptune Logging
- Neptune API completely changed
- Solution: Set `params['logging']='local'` to disable, or update neptune code

### Issue 4: apex Library
- NVIDIA apex is deprecated for mixed precision
- Solution: Use PyTorch native `torch.cuda.amp` instead

---

## 8. Quick Reference: Import Changes

```python
# OLD → NEW Import Mappings

# transformers
"from transformers.modeling_bert import *"
→ "from transformers import BertModel, BertPreTrainedModel"

"from transformers import AdamW"  
→ "from torch.optim import AdamW"

# keras
"from keras.preprocessing.sequence import pad_sequences"
→ Custom implementation (see Section 3.1.B)

# sklearn
"class_weight.compute_class_weight('balanced', np.unique(y), y)"
→ "compute_class_weight('balanced', classes=np.unique(y), y=y)"

# tqdm
"from tqdm import tqdm_notebook"
→ "from tqdm.notebook import tqdm"

# dataclasses (Python 3.7+ built-in)
"from dataclasses import dataclass"  # No change needed, remove from requirements
```

---

## 9. Estimated Effort

| Task | Time Estimate |
|------|---------------|
| Requirements update | 30 minutes |
| bertModels.py fixes | 1 hour |
| dataLoader.py fixes | 1 hour |
| manual_training_inference.py | 2 hours |
| Other Python files | 2 hours |
| Notebook updates | 1 hour |
| Testing & debugging | 2-4 hours |
| **Total** | **9-11 hours** |

---

## 10. Rollback Plan

If migration fails:
1. Keep original files backed up
2. Use Python 3.7 virtual environment as fallback
3. Consider Docker container with Python 3.7 for legacy support

---

*Document created: Migration plan for HateXplain Python 3.7 → 3.13*
*Target: Run Example_HateExplain.ipynb successfully on Python 3.13*
