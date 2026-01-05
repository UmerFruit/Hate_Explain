# HateXplain: Comprehensive Project Documentation

**Version:** 2.0  
**Last Updated:** January 5, 2026  
**Python Version:** 3.13 (Migrated from 3.7)  
**Project Status:** Production-Ready  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Design](#2-architecture--design)
3. [Data Pipeline & Preprocessing](#3-data-pipeline--preprocessing)
4. [Model Architectures](#4-model-architectures)
5. [Training & Inference](#5-training--inference)
6. [Explainability & Interpretability](#6-explainability--interpretability)
7. [API Reference](#7-api-reference)
8. [Configuration & Parameters](#8-configuration--parameters)
9. [File Structure Reference](#9-file-structure-reference)
10. [Usage Examples](#10-usage-examples)

---

## 1. Project Overview

### 1.1 Introduction

**HateXplain** is a benchmark dataset and deep learning framework for explainable hate speech detection, published at AAAI 2021. The project addresses three critical aspects of hate speech detection:

1. **Classification**: 3-class classification (hate speech, offensive, normal) or 2-class (toxic, non-toxic)
2. **Target Community Detection**: Identifying which community is targeted by hate/offensive speech
3. **Explainability**: Providing human-understandable rationales for classification decisions

### 1.2 Key Features

- **Multi-annotator Dataset**: Each post annotated by 3 annotators with:
  - Label (hatespeech/offensive/normal)
  - Target community identification
  - Token-level rationales (which words justify the label)

- **Multiple Model Architectures**:
  - BERT-based models (with attention supervision)
  - BiRNN (Bidirectional RNN with LSTM/GRU)
  - BiRNN with Attention
  - CNN-GRU hybrid models

- **Explainability Methods**:
  - Attention-based explanations
  - LIME (Local Interpretable Model-agnostic Explanations)
  - Supervised attention training with human rationales

- **Bias Evaluation**: Framework for measuring unintended bias towards target communities

### 1.3 Research Contributions

The project demonstrates that:
- Models with high classification accuracy don't necessarily provide good explanations
- Training with human rationales improves explainability metrics (plausibility, faithfulness)
- Supervised attention helps reduce unintended bias towards target communities

### 1.4 Published Work

**Paper**: "HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection"  
**Authors**: Binny Mathew, Punyajoy Saha, Seid Muhie Yimam, Chris Biemann, Pawan Goyal, Animesh Mukherjee  
**Conference**: AAAI 2021  
**ArXiv**: https://arxiv.org/abs/2012.10289  
**Dataset**: Available on HuggingFace  
**Pre-trained Models**: Available on HuggingFace Model Hub  

### 1.5 Dataset Statistics

- **Total Posts**: ~20,000 social media posts (Twitter + Gab)
- **Annotators**: 3 per post
- **Classes**: 
  - 3-class: hate speech, offensive, normal
  - 2-class: toxic (hate+offensive), non-toxic
- **Target Communities**: African, Arab, Asian, Caucasian, Christian, Hispanic, Hindu, Jewish, Muslim, Women, Homosexual, Other
- **Data Split**: Train/Val/Test = 8:1:1 (fixed splits provided)

---

## 2. Architecture & Design

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HateXplain System Architecture                │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐
│   Raw Data   │ (dataset.json)
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                        │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐     │
│  │ Ekphrasis  │→ │ Tokenization │→ │ Attention       │     │
│  │ Processor  │  │ (BERT/GloVe) │  │ Aggregation     │     │
│  └────────────┘  └──────────────┘  └─────────────────┘     │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Train Split  │  │  Val Split   │  │  Test Split  │      │
│  │   (80%)      │  │    (10%)     │  │    (10%)     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │               │
│         └──────────────────┴──────────────────┘               │
│                            ▼                                   │
│                   ┌──────────────────┐                        │
│                   │  DataLoader      │                        │
│                   │  (batched)       │                        │
│                   └──────────────────┘                        │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    MODEL LAYER                                │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  BERT-based Models                                    │   │
│  │  ┌──────────────┐     ┌──────────────────────┐      │   │
│  │  │ BERT Encoder │ ──→ │ Classification Head  │      │   │
│  │  │ (12 layers)  │     │ + Attention Loss     │      │   │
│  │  └──────────────┘     └──────────────────────┘      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Non-BERT Models (BiRNN, CNN-GRU)                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────┐    │   │
│  │  │Embedding │→ │ RNN/CNN  │→ │ Classification │    │   │
│  │  │ (GloVe)  │  │ Encoder  │  │ Head + Attn    │    │   │
│  │  └──────────┘  └──────────┘  └────────────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                  OUTPUT & EVALUATION                          │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐     │
│  │Class Label │  │  Attention   │  │  Explainability │     │
│  │Predictions │  │   Weights    │  │    Metrics      │     │
│  └────────────┘  └──────────────┘  └─────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Module Organization

The codebase is organized into distinct modules with clear responsibilities:

#### **Preprocess/** - Data Preprocessing Module
- **dataCollect.py**: Main data collection and dataset creation
- **preProcess.py**: Text preprocessing using Ekphrasis and spaCy
- **attentionCal.py**: Attention aggregation from multiple annotators
- **spanMatcher.py**: Token-level rationale matching
- **utils.py**: Helper functions

#### **Models/** - Neural Network Architectures
- **bertModels.py**: BERT-based models with supervised attention
- **otherModels.py**: BiRNN, CNN-GRU implementations
- **attentionLayer.py**: Custom attention mechanisms
- **utils.py**: Model saving/loading, loss functions

#### **TensorDataset/** - Data Loading & Encoding
- **dataLoader.py**: PyTorch DataLoader creation
- **datsetSplitter.py**: Train/val/test splitting and encoding

#### **eraserbenchmark/** - Evaluation Framework
- External benchmark for explainability metrics
- Metrics: Plausibility, Faithfulness

### 2.3 Design Principles

1. **Modularity**: Each component is self-contained and reusable
2. **Configurability**: All hyperparameters in JSON configs
3. **Reproducibility**: Fixed random seeds and deterministic operations
4. **Scalability**: GPU support with automatic device selection
5. **Extensibility**: Easy to add new models and preprocessing methods

---

## 3. Data Pipeline & Preprocessing

### 3.1 Dataset Format

The main dataset is stored in `Data/dataset.json` with the following structure:

```json
{
  "24198545_gab": {
    "post_id": "24198545_gab",
    "annotators": [
      {
        "label": "hatespeech",
        "annotator_id": 4,
        "target": ["African"]
      },
      {
        "label": "hatespeech",
        "annotator_id": 3,
        "target": ["African"]
      },
      {
        "label": "offensive",
        "annotator_id": 5,
        "target": ["African"]
      }
    ],
    "rationales": [
      [0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
      [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ],
    "post_tokens": ["and","this","is","why","i","end","up","with","nigger","trainee","doctors","who","can","not","speak","properly","lack","basic","knowledge","of","biology","it","truly","scary","if","the","public","only","knew"]
  }
}
```

**Key Fields**:
- `post_id`: Unique identifier (platform_postid)
- `annotators`: List of 3 annotator judgments
  - `label`: hatespeech/offensive/normal
  - `annotator_id`: Annotator identifier
  - `target`: List of target communities
- `rationales`: 3 binary vectors (1=token is part of rationale, 0=not)
- `post_tokens`: List of tokens from the post

### 3.2 Label Encoding

**3-Class Classification** (`Data/classes.npy`):
- Class 0: hatespeech
- Class 1: normal
- Class 2: offensive

**2-Class Classification** (`Data/classes_two.npy`):
- Class 0: toxic (hatespeech + offensive)
- Class 1: non-toxic (normal)

**Majority Voting**:
- Final label determined by majority (≥2 annotators agree)
- Posts with no majority (all 3 annotators disagree) labeled as "undecided" and excluded

### 3.3 Data Split

Fixed splits provided in `Data/post_id_divisions.json`:
```json
{
  "train": ["post_id1", "post_id2", ...],
  "val": ["post_id3", "post_id4", ...],
  "test": ["post_id5", "post_id6", ...]
}
```

- **Train**: 80% of data
- **Validation**: 10% of data
- **Test**: 10% of data

### 3.4 Text Preprocessing Pipeline

#### 3.4.1 Ekphrasis Processing

The Ekphrasis library handles social media text normalization:

**Normalizations**:
- URLs → `<url>`
- Emails → `<email>`
- Percentages → `<percent>`
- Money amounts → `<money>`
- Phone numbers → `<phone>`
- User mentions → `<user>`
- Time expressions → `<time>`
- Dates → `<date>`
- Numbers → `<number>`

**Annotations**:
- Hashtags → `<hashtag>` ... `</hashtag>`
- All caps → `<allcaps>` ... `</allcaps>`
- Elongated words → `<elongated>`
- Repeated characters → `<repeated>`
- Emphasis → `<emphasis>`
- Censored words → `<censored>`

**Additional Processing**:
- Hashtag segmentation (e.g., #happyholidays → happy holidays)
- Contraction expansion (e.g., can't → can not)
- Emoticon dictionary lookup

**Configuration** (`params['include_special']`):
- `True`: Keep special tags (e.g., `<hashtag>`)
- `False`: Remove special tags, keep only content

#### 3.4.2 Tokenization

**For BERT Models** (`params['bert_tokens'] = True`):
- Uses `BertTokenizer` from HuggingFace
- Subword tokenization (WordPiece)
- Special tokens: `[CLS]` at start, `[SEP]` at end
- Max length: 128 tokens (configurable)
- Truncation: Post-truncation (cut from end)

**For Non-BERT Models** (`params['bert_tokens'] = False`):
- Word-level tokenization
- Removes punctuation
- Maps to GloVe embeddings (300-dim)
- Unknown words mapped to 'unk' token

#### 3.4.3 Attention Aggregation

Multiple annotators provide rationales (which tokens justify their label). These need to be aggregated into a single attention vector.

**Process** (`Preprocess/attentionCal.py`):

1. **For "normal"/"non-toxic" posts**:
   - Uniform attention: `1/sequence_length` for all tokens
   - Rationale: No specific tokens are "hateful" or "offensive"

2. **For "hatespeech"/"offensive"/"toxic" posts**:
   
   a. **Variance Scaling** (`params['variance']`):
      ```python
      attention_vector = variance * attention_vector
      # Default variance = 5 or 10
      ```
   
   b. **Mean Across Annotators**:
      ```python
      attention_vector = mean(attention_vectors_from_3_annotators)
      ```
   
   c. **Normalization** (`params['type_attention']`):
      - `'softmax'`: Standard softmax (default)
        ```python
        exp(x) / sum(exp(x))
        ```
      - `'neg_softmax'`: Inverted softmax
        ```python
        exp(-x) / sum(exp(-x))
        ```
      - `'sigmoid'`: Element-wise sigmoid
        ```python
        1 / (1 + exp(-x))
        ```

3. **Optional: Attention Decay** (`params['decay'] = True`):
   - Spreads attention to neighboring tokens
   - Controlled by `window`, `alpha`, `p_value`, `method`
   - Methods: 'additive' or 'geometric'
   - Purpose: Decentralize attention from single tokens

**Example**:
```python
# 3 annotators mark tokens [0,0,1,1,0] [0,1,1,0,0] [0,0,1,1,1]
raw_attention = [[0,0,1,1,0], [0,1,1,0,0], [0,0,1,1,1]]

# Step 1: Variance scaling (variance=5)
scaled = [[0,0,5,5,0], [0,5,5,0,0], [0,0,5,5,5]]

# Step 2: Mean
mean_attention = [0, 1.67, 5, 3.33, 1.67]

# Step 3: Softmax normalization
final_attention = [0.006, 0.035, 0.866, 0.086, 0.007]
```

### 3.5 Data Loading

#### 3.5.1 Encoding (`TensorDataset/datsetSplitter.py`)

**For BERT Models**:
- Input: Token IDs from BertTokenizer
- Attention: Normalized attention vector
- Mask: Binary mask (1=real token, 0=padding)
- Labels: Integer class labels

**For Non-BERT Models**:
- Creates vocabulary from training data
- Maps words to embeddings using GloVe
- Unknown words → 'unk' embedding
- Padding token → zero vector

**Vocabulary Object** (`Vocab_own`):
```python
class Vocab_own:
    itos: dict  # index to string
    stoi: dict  # string to index
    vocab: dict  # word frequencies
    embeddings: np.array  # (vocab_size, 300)
```

#### 3.5.2 DataLoader Creation (`TensorDataset/dataLoader.py`)

**Sequence Padding**:
- All sequences padded to `max_length` (default 128)
- Padding value: 0 for token IDs, 0.0 for attention
- Truncation: From end ('post')

**Batch Creation**:
```python
# Each batch contains:
batch = (
    input_ids,      # (batch_size, max_length) - token IDs
    attention_vals, # (batch_size, max_length) - attention values
    attention_mask, # (batch_size, max_length) - padding mask
    labels          # (batch_size,) - class labels
)
```

**Samplers**:
- Training: `RandomSampler` (shuffled)
- Validation/Test: `SequentialSampler` (ordered)

---

## 4. Model Architectures

### 4.1 Overview

The project supports two main categories of models:

1. **BERT-based Models**: Transformer-based with supervised attention
2. **Non-BERT Models**: RNN and CNN-based architectures

All models output:
- Class predictions (logits)
- Attention weights (for explainability)

### 4.2 BERT-based Models

#### 4.2.1 SC_weighted_BERT (Supervised Classification BERT)

**Location**: `Models/bertModels.py`

**Architecture**:
```
Input Tokens
    ↓
BERT Encoder (12 layers)
    ├→ Hidden States (768-dim per token)
    ├→ Pooled Output ([CLS] token representation)
    └→ Attention Weights (12 layers × 12 heads)
    ↓
Dropout (dropout_bert)
    ↓
Linear Classifier (768 → num_classes)
    ↓
Logits + Loss
```

**Key Features**:

1. **Classification Loss**:
   ```python
   CrossEntropyLoss(logits, labels, weight=class_weights)
   ```

2. **Supervised Attention Loss** (when `train_att=True`):
   ```python
   # Extract attention from specific layer and heads
   attention_weights = bert_outputs[layer][batch, head, 0, :]
   
   # Compare with human rationales
   loss_att = masked_cross_entropy(attention_weights, 
                                    human_rationales, 
                                    attention_mask)
   
   # Combined loss
   total_loss = classification_loss + lambda * attention_loss
   ```

3. **Configurable Supervision**:
   - `supervised_layer_pos`: Which BERT layer to supervise (0-11, default 11)
   - `num_supervised_heads`: How many attention heads to supervise (default 6)
   - `att_lambda`: Weight for attention loss (default 0.001)

**Hyperparameters**:
```python
{
    "path_files": "bert-base-uncased",
    "num_labels": 3,
    "dropout_bert": 0.1,
    "train_att": True/False,
    "supervised_layer_pos": 11,
    "num_supervised_heads": 6,
    "att_lambda": 0.001
}
```

**Forward Pass**:
```python
outputs = model(
    input_ids=tokens,           # (batch, seq_len)
    attention_mask=pad_mask,    # (batch, seq_len)
    attention_vals=rationales,  # (batch, seq_len)
    labels=labels,              # (batch,)
    device=device
)

# Returns:
# outputs[0]: loss (if labels provided)
# outputs[1]: logits (batch, num_classes)
# outputs[2:]: hidden_states, attentions
```

### 4.3 Non-BERT Models

All non-BERT models use GloVe embeddings (300-dim) as input.

#### 4.3.1 BiRNN (Bidirectional RNN)

**Location**: `Models/otherModels.py`

**Architecture**:
```
Input Token IDs
    ↓
Embedding Layer (vocab_size → 300)
    ↓
Dropout2d (drop_embed)
    ↓
BiLSTM/BiGRU (300 → hidden_size × 2)
    ↓
Hidden State Concatenation
    ↓
Dropout (drop_fc)
    ↓
Linear1 (hidden_size × 2 → hidden_size)
    ↓
ReLU
    ↓
Dropout (drop_fc)
    ↓
Linear2 (hidden_size → num_classes)
    ↓
Logits
```

**Key Features**:
- Bidirectional LSTM or GRU
- Embedding layer can be frozen or trainable (`train_embed`)
- Uses final hidden states (not sequence output)

**Hyperparameters**:
```python
{
    "hidden_size": 256,
    "embed_size": 300,
    "seq_model": "lstm" or "gru",
    "drop_embed": 0.3,
    "drop_fc": 0.2,
    "drop_hidden": 0.3,
    "train_embed": False
}
```

#### 4.3.2 BiAtt_RNN (BiRNN with Attention)

**Location**: `Models/otherModels.py`

**Architecture**:
```
Input Token IDs
    ↓
Embedding Layer (vocab_size → 300)
    ↓
Dropout2d (drop_embed)
    ↓
BiLSTM/BiGRU (300 → hidden_size × 2)
    ↓
Sequence of Hidden States
    ↓
Attention Mechanism (LBSA)
    ├→ Attention Weights (for explainability)
    └→ Weighted Sum of Hidden States
    ↓
Dropout (drop_fc)
    ↓
Linear1 (hidden_size × 2 → batch_size)
    ↓
ReLU
    ↓
Dropout (drop_fc)
    ↓
Linear2 (batch_size → num_classes)
    ↓
Logits
```

**Attention Mechanism** (LBSA - Location-Based Self Attention):

```python
# Attention calculation
scores = tanh(W × hidden_states + b)
scores = context_vector × scores
attention_weights = softmax(scores)
attended_output = sum(attention_weights × hidden_states)
```

**Supervised Attention Training** (when `train_att=True`):
```python
# Attention loss
loss_att = masked_cross_entropy(predicted_attention, 
                                 human_rationales,
                                 attention_mask)

# Total loss
total_loss = classification_loss + lambda * loss_att
```

**Two Variants**:
1. **birnnatt**: Uses softmax for attention normalization
2. **birnnscrat**: Uses sigmoid for attention (allows multiple focus points)

#### 4.3.3 CNN_GRU (CNN-GRU Hybrid)

**Location**: `Models/otherModels.py`

**Architecture**:
```
Input Token IDs
    ↓
Embedding Layer (vocab_size → 300)
    ↓
Dropout (drop_embed)
    ↓
┌────────────────────────────────────┐
│  Parallel CNN Layers               │
│  ├─ Conv1D (kernel_size=2) → 100   │
│  ├─ Conv1D (kernel_size=3) → 100   │
│  └─ Conv1D (kernel_size=4) → 100   │
│       ↓                             │
│  MaxPool1D (kernel=4, stride=4)    │
└────────────────────────────────────┘
    ↓
Concatenate (→ 300 features)
    ↓
MaxPool1D (kernel=4, stride=4)
    ↓
GRU (100 → 100, single direction)
    ↓
Global Max Pooling
    ↓
Dropout (drop_fc)
    ↓
Linear (100 → num_classes)
    ↓
Logits
```

**Key Features**:
- Multi-scale CNN feature extraction (2, 3, 4-gram patterns)
- Temporal modeling with GRU
- Global max pooling for fixed-size representation

### 4.4 Attention Layers

#### 4.4.1 Attention_LBSA (Location-Based Self Attention)

**Location**: `Models/attentionLayer.py`

**Mechanism**:
```python
# Learnable parameters
W: (feature_dim, feature_dim)         # Weight matrix
b: (feature_dim,)                      # Bias
context_vector: (feature_dim, 1)       # Context vector

# Forward pass
temp = hidden_states.reshape(-1, feature_dim)
scores = tanh(matmul(temp, W) + b)
scores = matmul(scores, context_vector)
scores = scores.reshape(batch, seq_len)

# Masking and normalization
scores[~mask] = -inf
attention_weights = softmax(scores, dim=1)

# Weighted sum
attended = sum(attention_weights * hidden_states)
```

**Output**:
- `attended`: Weighted sum of hidden states
- `attention_weights`: Attention distribution (for explainability)

#### 4.4.2 Attention_LBSA_sigmoid

**Variant**: Uses sigmoid instead of softmax
- Allows multiple tokens to have high attention
- Doesn't force normalization to sum to 1
- Better for tasks where multiple tokens are independently important

### 4.5 Model Selection

**Function**: `select_model(params, embeddings)`

```python
if params['bert_tokens']:
    # BERT-based model
    model = SC_weighted_BERT.from_pretrained(
        params['path_files'],
        num_labels=params['num_classes'],
        output_attentions=True,
        hidden_dropout_prob=params['dropout_bert'],
        params=params
    )
else:
    # Non-BERT models
    if params['model_name'] == "birnn":
        model = BiRNN(params, embeddings)
    elif params['model_name'] == "birnnatt":
        model = BiAtt_RNN(params, embeddings, return_att=False)
    elif params['model_name'] == "birnnscrat":
        model = BiAtt_RNN(params, embeddings, return_att=True)
    elif params['model_name'] == "cnn_gru":
        model = CNN_GRU(params, embeddings)
```

### 4.6 Loss Functions

#### 4.6.1 Classification Loss

```python
CrossEntropyLoss(
    weight=class_weights,  # Handle class imbalance
    reduction='mean'
)
```

**Class Weight Calculation** (when `auto_weights=True`):
```python
from sklearn.utils.class_weight import compute_class_weight

weights = compute_class_weight(
    'balanced',
    classes=unique_classes,
    y=training_labels
)
```

#### 4.6.2 Masked Cross Entropy (for Attention)

**Location**: `Models/utils.py`

```python
def masked_cross_entropy(predicted_attention, target_attention, mask):
    """
    Computes cross-entropy only on non-padded tokens
    
    Args:
        predicted_attention: (batch, seq_len) - Model's attention
        target_attention: (batch, seq_len) - Human rationales
        mask: (batch, seq_len) - 1=real token, 0=padding
    
    Returns:
        loss: Scalar loss value
    """
    loss = 0
    for i in range(mask.shape[0]):
        # Only compute loss on real tokens
        pred = predicted_attention[i][mask[i]]
        target = target_attention[i][mask[i]]
        loss += cross_entropy(pred, target)
    
    return loss / mask.shape[0]
```

### 4.7 Pre-trained Models

Available on HuggingFace and in `best_model_json/`:

1. **BERT (No Attention Supervision)**:
   - Config: `bestModel_bert_base_uncased_Attn_train_FALSE.json`
   - Best for: Pure classification
   - Test F1: ~0.69

2. **BERT (With Attention Supervision)**:
   - Config: `bestModel_bert_base_uncased_Attn_train_TRUE.json`
   - Best for: Classification + Explainability
   - Test F1: ~0.68
   - Better explainability metrics

3. **BiRNN**:
   - Config: `bestModel_birnn.json`
   - Lightweight, fast inference

4. **BiRNN with Attention**:
   - Config: `bestModel_birnnatt.json`
   - Good balance of performance and interpretability

5. **BiRNN-SCRAT**:
   - Config: `bestModel_birnnscrat.json`
   - Sigmoid attention for multi-focus

6. **CNN-GRU**:
   - Config: `bestModel_cnn_gru.json`
   - Fast, good for pattern detection

---

## 5. Training & Inference

### 5.1 Training Pipeline

**Main Script**: `manual_training_inference.py`

#### 5.1.1 Training Flow

```
1. Load Configuration
   └→ From JSON file or inline params

2. Prepare Data
   ├→ Load dataset (dataset.json)
   ├→ Apply preprocessing
   ├→ Create train/val/test splits
   └→ Create DataLoaders

3. Initialize Model
   ├→ Select architecture (BERT/BiRNN/etc.)
   ├→ Load embeddings (for non-BERT)
   └→ Move to GPU/CPU

4. Setup Training
   ├→ Initialize optimizer (AdamW)
   ├→ Setup learning rate scheduler
   ├→ Calculate class weights (if auto_weights)
   └→ Set random seeds

5. Training Loop (for each epoch)
   ├→ Train Phase
   │  ├─ Forward pass
   │  ├─ Calculate losses
   │  ├─ Backward pass
   │  ├─ Gradient clipping
   │  └─ Update parameters
   │
   └→ Evaluation Phase
      ├─ Evaluate on train set
      ├─ Evaluate on validation set
      ├─ Evaluate on test set
      └─ Save best model (based on val F1)

6. Save Final Model
   └→ Best checkpoint based on validation F1
```

#### 5.1.2 Training Configuration

**Command Line Usage**:
```bash
python manual_training_inference.py \
    --path_to_json best_model_json/bestModel_bert_base_uncased_Attn_train_TRUE.json \
    --use_from_file True \
    --attention_lambda 0.001
```

**Key Parameters**:

```python
# Training hyperparameters
{
    "epochs": 20,
    "batch_size": 16,
    "learning_rate": 2e-5,      # BERT: ~2e-5, RNN: 0.001-0.1
    "epsilon": 1e-8,             # Adam epsilon
    "device": "cuda",            # "cuda" or "cpu"
    "random_seed": 42,           # For reproducibility
    
    # Model-specific
    "train_att": True/False,     # Train with attention supervision
    "att_lambda": 0.001,         # Attention loss weight
    
    # Class balancing
    "auto_weights": True,        # Auto-calculate class weights
    "weights": [1.08, 0.82, 1.17] # Or manual weights
}
```

#### 5.1.3 Optimizer & Scheduler

**For BERT Models**:
```python
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    eps=1e-8
)

# Learning rate warmup
total_steps = len(train_dataloader) * epochs
warmup_steps = total_steps // 10

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

**For Non-BERT Models**:
```python
optimizer = AdamW(
    model.parameters(),
    lr=0.001,  # Higher learning rate
    eps=1e-8
)
# No scheduler typically used
```

#### 5.1.4 Training Loop Details

```python
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in train_dataloader:
        # Unpack batch
        input_ids = batch[0].to(device)
        attention_vals = batch[1].to(device)
        attention_mask = batch[2].to(device)
        labels = batch[3].to(device)
        
        # Forward pass
        model.zero_grad()
        outputs = model(
            input_ids,
            attention_vals=attention_vals,
            attention_mask=attention_mask,
            labels=labels,
            device=device
        )
        
        loss = outputs[0]
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters
        optimizer.step()
        
        # Update learning rate (BERT only)
        if bert_tokens:
            scheduler.step()
    
    # Epoch metrics
    avg_train_loss = total_loss / len(train_dataloader)
    
    # Evaluate
    train_f1, train_acc, ... = eval_phase('train')
    val_f1, val_acc, ... = eval_phase('val')
    test_f1, test_acc, ... = eval_phase('test')
    
    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        save_model(model, params)
```

### 5.2 Evaluation

#### 5.2.1 Evaluation Metrics

```python
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

# Classification metrics
accuracy = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels, average='macro')
precision = precision_score(true_labels, pred_labels, average='macro')
recall = recall_score(true_labels, pred_labels, average='macro')

# For 3-class problems
roc_auc = roc_auc_score(
    true_labels,
    probabilities,
    multi_class='ovo',  # One-vs-One
    average='macro'
)
```

#### 5.2.2 Evaluation Phase Function

```python
def Eval_phase(params, which_files, model, dataloader, device):
    """
    Evaluate model on train/val/test set
    
    Returns:
        f1_score, accuracy, precision, recall, roc_auc, logits
    """
    model.eval()
    
    true_labels = []
    pred_labels = []
    logits_all = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_vals = batch[1].to(device)
            attention_mask = batch[2].to(device)
            labels = batch[3].to(device)
            
            outputs = model(
                input_ids,
                attention_vals=attention_vals,
                attention_mask=attention_mask,
                labels=None,
                device=device
            )
            
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = labels.detach().cpu().numpy()
            
            pred_labels += list(np.argmax(logits, axis=1))
            true_labels += list(label_ids)
            logits_all += list(logits)
    
    # Calculate metrics
    f1 = f1_score(true_labels, pred_labels, average='macro')
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    
    # Convert logits to probabilities
    probabilities = [softmax(logit) for logit in logits_all]
    
    if num_classes == 3:
        roc_auc = roc_auc_score(true_labels, probabilities, 
                                multi_class='ovo', average='macro')
    else:
        roc_auc = 0
    
    return f1, accuracy, precision, recall, roc_auc, probabilities
```

### 5.3 Inference

#### 5.3.1 Loading Saved Models

**For BERT Models**:
```python
from transformers import BertTokenizer
from Models.bertModels import SC_weighted_BERT

# Load model
model = SC_weighted_BERT.from_pretrained(
    saved_model_path,
    num_labels=3,
    output_attentions=True,
    params=params
)
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```

**For Non-BERT Models**:
```python
from Models.utils import load_model

# Initialize model
model = select_model(params, embeddings)

# Load weights
model = load_model(model, params)
model.eval()
```

#### 5.3.2 Inference on New Text

**Example** (`testing_with_lime.py` demonstrates this):

```python
class modelPred:
    def __init__(self, model_to_use, params):
        # Initialize model, tokenizer, vocab
        self.model = select_model(params, embeddings)
        self.model.eval()
        
    def return_probab(self, sentences_list):
        """
        Args:
            sentences_list: List of strings
        
        Returns:
            probabilities: List of probability distributions
        """
        # Transform to internal format
        temp_data = transform_dummy_data(sentences_list)
        
        # Preprocess
        test_data = get_test_data(temp_data, params)
        
        # Encode
        test_encoded = encodeData(test_data, vocab, params)
        
        # Create dataloader
        test_dataloader = combine_features(test_encoded, params)
        
        # Inference
        logits_all = []
        for batch in test_dataloader:
            outputs = self.model(batch[0], batch[1], batch[2])
            logits = outputs[0].detach().cpu().numpy()
            logits_all += list(logits)
        
        # Convert to probabilities
        probabilities = [softmax(logit) for logit in logits_all]
        
        return probabilities

# Usage
predictor = modelPred('bert', params)
sentences = ["I hate you", "You are great"]
probs = predictor.return_probab(sentences)
# probs[0] = [P(hate), P(normal), P(offensive)]
```

### 5.4 Model Saving & Loading

#### 5.4.1 Save BERT Model

```python
def save_bert_model(model, tokenizer, params):
    """
    Saves BERT model and tokenizer
    
    Directory structure:
    Saved/bert-base-uncased_11_6_3_0.001/
        ├── config.json
        ├── pytorch_model.bin
        └── tokenizer files
    """
    output_dir = f"Saved/{params['path_files']}_"
    
    if params['train_att']:
        output_dir += f"{params['supervised_layer_pos']}_"
        output_dir += f"{params['num_supervised_heads']}_"
        output_dir += f"{params['num_classes']}_"
        output_dir += f"{params['att_lambda']}"
    else:
        output_dir += f"{params['num_classes']}"
    
    # Create directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model
    model.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")
```

#### 5.4.2 Save Non-BERT Model

```python
def save_normal_model(model, params):
    """
    Saves PyTorch state dict
    
    Filename format:
    Saved/birnnatt_lstm_256_3_0.001.pth
    """
    if params['train_att']:
        filename = f"Saved/{params['model_name']}_"
        filename += f"{params['seq_model']}_"
        filename += f"{params['hidden_size']}_"
        filename += f"{params['num_classes']}_"
        filename += f"{params['att_lambda']}.pth"
    else:
        filename = f"Saved/{params['model_name']}_"
        filename += f"{params['seq_model']}_"
        filename += f"{params['hidden_size']}_"
        filename += f"{params['num_classes']}.pth"
    
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")
```

### 5.5 GPU Management

```python
def get_gpu(params):
    """
    Automatically selects available GPU with low load
    
    Uses GPUtil to find GPU with:
    - Low memory usage
    - Low computational load
    
    Waits until a GPU becomes available
    """
    import GPUtil
    
    # BERT models need more memory
    if params['bert_tokens']:
        max_load = 0.07
        max_memory = 0.07
    else:
        max_load = 0.5
        max_memory = 0.5
    
    while True:
        available_gpus = GPUtil.getAvailable(
            order='memory',
            limit=1,
            maxLoad=max_load,
            maxMemory=max_memory
        )
        
        if available_gpus:
            gpu_id = available_gpus[0]
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            return [gpu_id]
        else:
            print("No GPU available, waiting...")
            time.sleep(5)

# Usage
if params['device'] == 'cuda' and torch.cuda.is_available():
    device_id = get_gpu(params)
    torch.cuda.set_device(device_id[0])
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

---

## 6. Explainability & Interpretability

### 6.1 Overview

The project provides three main explainability approaches:

1. **Attention-based Explanations**: From model's attention weights
2. **LIME Explanations**: Model-agnostic local explanations
3. **Rationale Evaluation**: Using ERASER benchmark metrics

### 6.2 Attention-based Explanations

#### 6.2.1 Extracting Attention Weights

**For BERT Models**:
```python
outputs = model(
    input_ids,
    attention_mask=attention_mask,
    output_attentions=True
)

# outputs[1] contains attention from all layers
# Shape: (num_layers, batch, num_heads, seq_len, seq_len)

# Extract attention from layer 11 (last layer)
layer_11_attention = outputs[1][11]  # (batch, 12, seq_len, seq_len)

# Average across attention heads
avg_attention = layer_11_attention.mean(dim=1)  # (batch, seq_len, seq_len)

# Extract attention to [CLS] token (position 0)
cls_attention = avg_attention[:, 0, :]  # (batch, seq_len)

# This represents how much each token attends to [CLS]
```

**For Non-BERT Models with Attention**:
```python
outputs = model(input_ids, attention_mask=attention_mask)

# outputs[1] contains attention weights
attention_weights = outputs[1]  # (batch, seq_len)

# Already normalized (softmax applied in model)
```

#### 6.2.2 Visualizing Attention

```python
def visualize_attention(tokens, attention_weights, true_rationale=None):
    """
    Args:
        tokens: List of string tokens
        attention_weights: numpy array of attention values
        true_rationale: Optional human rationale for comparison
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 2))
    
    # Normalize attention for visualization
    attn = attention_weights / attention_weights.max()
    
    # Color map
    colors = plt.cm.Reds(attn)
    
    # Plot tokens with attention-based coloring
    for i, (token, color) in enumerate(zip(tokens, colors)):
        ax.text(i, 0, token, 
                bbox=dict(facecolor=color, alpha=0.8),
                fontsize=12)
    
    # Overlay true rationale if provided
    if true_rationale is not None:
        for i, is_rationale in enumerate(true_rationale):
            if is_rationale:
                ax.plot([i, i], [-0.5, 0.5], 'g-', linewidth=3)
    
    ax.set_xlim(-0.5, len(tokens)-0.5)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
```

### 6.3 LIME Explanations

**Script**: `testing_with_lime.py`

LIME (Local Interpretable Model-agnostic Explanations) provides explanations by:
1. Creating perturbed versions of input
2. Getting model predictions on perturbed inputs
3. Training a simple linear model to approximate the complex model locally
4. Using linear model weights as feature importance

#### 6.3.1 LIME Setup

```python
from lime.lime_text import LimeTextExplainer

# Initialize explainer
explainer = LimeTextExplainer(
    class_names=['hatespeech', 'normal', 'offensive']
)

# Prediction function for LIME
def predict_proba(texts):
    """
    Args:
        texts: List of strings
    
    Returns:
        probabilities: numpy array (len(texts), num_classes)
    """
    # Preprocess texts
    processed_data = preprocess_texts(texts, params)
    
    # Get model predictions
    dataloader = create_dataloader(processed_data, params)
    
    probabilities = []
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(batch[0], batch[1], batch[2])
            logits = outputs[0].cpu().numpy()
            probs = softmax(logits, axis=1)
            probabilities.extend(probs)
    
    return np.array(probabilities)

# Generate explanation for a single instance
explanation = explainer.explain_instance(
    text_instance,
    predict_proba,
    num_features=10,  # Top 10 important words
    num_samples=5000  # Number of perturbed samples
)

# Get feature importance
feature_importance = explanation.as_list()
# [('word1', 0.45), ('word2', -0.23), ...]
```

#### 6.3.2 LIME Interpretation

```python
# Show in notebook
explanation.show_in_notebook(text=True)

# Get top positive and negative features
positive_features = [
    (word, weight) for word, weight in feature_importance 
    if weight > 0
][:5]

negative_features = [
    (word, weight) for word, weight in feature_importance 
    if weight < 0
][:5]

print("Words supporting current prediction:")
for word, weight in positive_features:
    print(f"  {word}: {weight:.3f}")

print("\nWords against current prediction:")
for word, weight in negative_features:
    print(f"  {word}: {weight:.3f}")
```

### 6.4 Rationale Evaluation with Human Annotations

**Script**: `testing_with_rational.py`

#### 6.4.1 Extracting Model Rationales

```python
def extract_rationales(model, test_dataloader, params, device):
    """
    Extract attention-based rationales from model
    
    Returns:
        List of dictionaries with:
        - annotation_id: post_id
        - classification: predicted label
        - classification_scores: probability distribution
        - rationales: list of rationales (one per annotator)
    """
    results = []
    
    model.eval()
    with torch.no_grad():
        for batch, post_ids in zip(test_dataloader, post_id_list):
            input_ids = batch[0].to(device)
            attention_mask = batch[2].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            
            # Get attention weights
            if params['bert_tokens']:
                # BERT: average attention from layer 11
                attention_weights = outputs[1][11][:, :, 0, :]
                attention_weights = attention_weights.mean(dim=1)
            else:
                # Non-BERT: direct attention output
                attention_weights = outputs[1]
            
            # Convert to probabilities
            probs = softmax(logits.cpu().numpy())
            predictions = np.argmax(probs, axis=1)
            
            # Format results
            for i in range(len(post_ids)):
                result = {
                    'annotation_id': post_ids[i],
                    'classification': class_names[predictions[i]],
                    'classification_scores': {
                        class_names[j]: float(probs[i][j])
                        for j in range(len(class_names))
                    },
                    'rationales': [{
                        'docid': post_ids[i],
                        'hard_rationale_predictions': [
                            {'end_token': k+1, 'start_token': k}
                            for k in range(len(attention_weights[i]))
                            if attention_weights[i][k] > threshold
                        ],
                        'soft_rationale_predictions': 
                            attention_weights[i].cpu().numpy().tolist()
                    }]
                }
                results.append(result)
    
    return results
```

#### 6.4.2 ERASER Benchmark Metrics

The project uses the ERASER benchmark for evaluating rationale quality.

**Metrics**:

1. **Token-level F1** (Plausibility):
   - Measures agreement between model rationales and human rationales
   - Higher is better (model identifies same tokens as humans)

2. **AUPRC** (Area Under Precision-Recall Curve):
   - Evaluates quality of soft attention scores
   - Higher is better

3. **Comprehensiveness**:
   - How much does removing rationales hurt performance?
   - Measures if model truly relies on identified tokens
   - Higher is better (performance drops more when rationales removed)

4. **Sufficiency**:
   - Can the model maintain performance using only rationales?
   - Measures if rationales alone are enough for prediction
   - Lower is better (rationales should be sufficient)

**Running ERASER Evaluation**:
```bash
# From eraserbenchmark directory
python rationale_benchmark/metrics.py \
    --data_dir ../Data \
    --results ../model_rationales.json \
    --score_file ../scores.json
```

### 6.5 Bias Evaluation

**Script**: `testing_for_bias.py`

#### 6.5.1 Unintended Bias Measurement

Measures if model is biased against specific target communities:

```python
def evaluate_bias(model, test_data, target_community):
    """
    Evaluate model bias towards a specific community
    
    Approach:
    1. Filter posts mentioning the target community
    2. Check if model is more likely to predict "toxic" 
       when community is mentioned
    3. Compare to baseline (all posts)
    """
    # Filter by target community
    community_posts = test_data[
        test_data['targets'].apply(lambda x: target_community in x)
    ]
    
    # Get predictions
    community_preds = predict(model, community_posts)
    all_preds = predict(model, test_data)
    
    # Calculate bias metrics
    community_toxic_rate = (community_preds == 'toxic').mean()
    overall_toxic_rate = (all_preds == 'toxic').mean()
    
    bias_score = community_toxic_rate - overall_toxic_rate
    
    return {
        'community': target_community,
        'community_toxic_rate': community_toxic_rate,
        'overall_toxic_rate': overall_toxic_rate,
        'bias_score': bias_score
    }
```

#### 6.5.2 Bias Mitigation

The paper shows that supervised attention training helps reduce bias:

```python
# Regular BERT (no attention supervision)
bias_scores_regular = evaluate_all_communities(model_regular)

# BERT with supervised attention
bias_scores_supervised = evaluate_all_communities(model_supervised)

# Supervised attention typically shows lower bias scores
# because it's trained to focus on actual hateful tokens
# rather than just community mentions
```

### 6.6 Evaluation Notebooks

#### 6.6.1 Example_HateExplain.ipynb

Demonstrates end-to-end usage:
- Loading data
- Preprocessing
- Training a model
- Getting predictions
- Visualizing attention

#### 6.6.2 Explainability_Calculation_NB.ipynb

Computes explainability metrics:
- Extracts model rationales
- Compares with human rationales
- Calculates ERASER metrics

#### 6.6.3 Bias_Calculation_NB.ipynb

Evaluates model bias:
- Tests each target community
- Measures unintended bias
- Compares different models

---

## 7. API Reference

### 7.1 Core Functions

#### 7.1.1 Data Collection

**Function**: `collect_data(params)`  
**Location**: `Preprocess/dataCollect.py`

```python
def collect_data(params):
    """
    Main data collection and preprocessing pipeline
    
    Args:
        params (dict): Configuration dictionary with:
            - data_file: Path to dataset.json
            - bert_tokens: Whether to use BERT tokenization
            - class_names: Path to class encoder
            - max_length: Maximum sequence length
            - type_attention: Attention aggregation method
            - variance: Attention variance scaling
            ... (see Parameters section)
    
    Returns:
        pd.DataFrame: Processed data with columns:
            - Post_id: Unique identifier
            - Text: List of token IDs
            - Attention: Attention vector
            - Label: Class label (string)
    
    Process:
        1. Load dataset.json
        2. Determine final labels via majority voting
        3. Preprocess text (Ekphrasis + tokenization)
        4. Aggregate rationales into attention vectors
        5. Create dataframe
        6. Save as pickle file
    
    Notes:
        - Caches processed data as pickle
        - Posts without majority label excluded
        - Attention aggregated from 3 annotators
    """
```

**Usage**:
```python
params = {
    'data_file': 'Data/dataset.json',
    'bert_tokens': True,
    'class_names': 'Data/classes.npy',
    'max_length': 128,
    'type_attention': 'softmax',
    'variance': 5,
    # ... other params
}

processed_data = collect_data(params)
```

#### 7.1.2 Dataset Splitting

**Function**: `createDatasetSplit(params)`  
**Location**: `TensorDataset/datsetSplitter.py`

```python
def createDatasetSplit(params):
    """
    Create train/val/test splits
    
    Args:
        params (dict): Configuration dictionary
    
    Returns:
        For BERT models:
            train, val, test (list of tuples)
            Each tuple: (token_ids, attention, label)
        
        For non-BERT models:
            train, val, test, vocab_own
            vocab_own: Vocab_own object with embeddings
    
    Process:
        1. Load or create processed data
        2. Split by post_id_divisions.json (80/10/10)
        3. Create vocabulary (non-BERT only)
        4. Encode tokens to IDs
        5. Cache results
    
    Notes:
        - Uses fixed splits for reproducibility
        - Non-BERT: creates GloVe-based vocab
        - Caches splits as pickle files
    """
```

**Usage**:
```python
# For BERT
train, val, test = createDatasetSplit(params)

# For non-BERT
train, val, test, vocab = createDatasetSplit(params)
```

#### 7.1.3 DataLoader Creation

**Function**: `combine_features(tuple_data, params, is_train)`  
**Location**: `TensorDataset/dataLoader.py`

```python
def combine_features(tuple_data, params, is_train=False):
    """
    Create PyTorch DataLoader from encoded data
    
    Args:
        tuple_data: List of (token_ids, attention, label) tuples
        params (dict): Configuration dictionary
        is_train (bool): Whether this is training data
    
    Returns:
        torch.utils.data.DataLoader:
            Batch structure:
            - batch[0]: input_ids (batch, max_length)
            - batch[1]: attention_vals (batch, max_length)
            - batch[2]: attention_mask (batch, max_length)
            - batch[3]: labels (batch,)
    
    Process:
        1. Extract components from tuples
        2. Encode labels with LabelEncoder
        3. Pad sequences to max_length
        4. Create attention masks
        5. Convert to PyTorch tensors
        6. Create DataLoader with appropriate sampler
    
    Notes:
        - Training: RandomSampler (shuffled)
        - Val/Test: SequentialSampler (ordered)
        - Padding value: 0 for tokens, 0.0 for attention
    """
```

**Usage**:
```python
train_loader = combine_features(train, params, is_train=True)
val_loader = combine_features(val, params, is_train=False)
test_loader = combine_features(test, params, is_train=False)
```

#### 7.1.4 Model Selection

**Function**: `select_model(params, embeddings)`  
**Location**: `manual_training_inference.py`, `testing_with_lime.py`

```python
def select_model(params, embeddings=None):
    """
    Initialize model based on configuration
    
    Args:
        params (dict): Configuration dictionary
        embeddings (np.ndarray): GloVe embeddings (non-BERT only)
    
    Returns:
        torch.nn.Module: Initialized model
    
    Supported Models:
        BERT-based (bert_tokens=True):
            - 'weighted': SC_weighted_BERT
        
        Non-BERT (bert_tokens=False):
            - 'birnn': BiRNN
            - 'birnnatt': BiAtt_RNN (softmax attention)
            - 'birnnscrat': BiAtt_RNN (sigmoid attention)
            - 'cnn_gru': CNN_GRU
    
    Notes:
        - BERT models loaded from HuggingFace
        - Non-BERT models require embeddings parameter
        - Models initialized with random weights
        - Use load_model() to load trained weights
    """
```

**Usage**:
```python
# BERT model
params['bert_tokens'] = True
params['what_bert'] = 'weighted'
params['path_files'] = 'bert-base-uncased'
model = select_model(params)

# Non-BERT model
params['bert_tokens'] = False
params['model_name'] = 'birnnatt'
model = select_model(params, embeddings)
```

#### 7.1.5 Training

**Function**: `train_model(params, device)`  
**Location**: `manual_training_inference.py`

```python
def train_model(params, device):
    """
    Complete training pipeline
    
    Args:
        params (dict): Full configuration dictionary
        device (torch.device): Device to train on
    
    Returns:
        int: 1 (success indicator)
    
    Process:
        1. Create data splits
        2. Calculate class weights (if auto_weights)
        3. Initialize model
        4. Setup optimizer and scheduler
        5. Training loop:
            - Forward/backward pass
            - Gradient clipping
            - Parameter updates
            - Learning rate scheduling
        6. Evaluation after each epoch
        7. Save best model (by val F1)
    
    Notes:
        - Saves best model to Saved/ directory
        - Logs metrics (local or Neptune)
        - Uses early stopping implicitly (saves best)
        - Clears CUDA cache after training
    """
```

**Usage**:
```python
params = load_params('best_model_json/bestModel_bert.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_model(params, device)
```

#### 7.1.6 Evaluation

**Function**: `Eval_phase(params, which_files, model, dataloader, device)`  
**Location**: `manual_training_inference.py`

```python
def Eval_phase(params, which_files, model, dataloader, device):
    """
    Evaluate model on a dataset
    
    Args:
        params (dict): Configuration dictionary
        which_files (str): 'train', 'val', or 'test'
        model (torch.nn.Module): Model to evaluate
        dataloader (DataLoader): Data to evaluate on
        device (torch.device): Device to run on
    
    Returns:
        tuple: (f1, accuracy, precision, recall, roc_auc, logits)
            - f1 (float): Macro F1 score
            - accuracy (float): Accuracy
            - precision (float): Macro precision
            - recall (float): Macro recall
            - roc_auc (float): ROC AUC (0 for 2-class)
            - logits (list): Probability distributions
    
    Process:
        1. Set model to eval mode
        2. Iterate through dataloader
        3. Get predictions (no gradient)
        4. Calculate metrics
        5. Log results
    
    Notes:
        - Uses macro averaging for multi-class
        - Converts logits to probabilities
        - Prints metrics if logging='local'
    """
```

**Usage**:
```python
f1, acc, prec, rec, auc, probs = Eval_phase(
    params, 
    'test', 
    model, 
    test_dataloader, 
    device
)
```

### 7.2 Utility Functions

#### 7.2.1 Attention Aggregation

**Function**: `aggregate_attention(at_mask, row, params)`  
**Location**: `Preprocess/attentionCal.py`

```python
def aggregate_attention(at_mask, row, params):
    """
    Aggregate attention vectors from multiple annotators
    
    Args:
        at_mask: List of binary attention vectors from annotators
        row: DataFrame row with post information
        params (dict): Configuration with:
            - type_attention: 'softmax', 'neg_softmax', 'sigmoid'
            - variance: Scaling factor
            - decay: Whether to apply attention decay
    
    Returns:
        np.ndarray: Aggregated attention vector
    
    Process:
        1. Check if post is normal/non-toxic
            - If yes: uniform attention (1/length)
        2. Otherwise:
            - Scale by variance
            - Average across annotators
            - Normalize (softmax/sigmoid)
            - Optionally apply decay
    
    Notes:
        - Normal posts get uniform attention
        - Hate/offensive posts get focused attention
        - Decay spreads attention to neighbors
    """
```

**Usage**:
```python
# 3 annotators' rationales
rationales = [[0,0,1,1,0], [0,1,1,0,0], [0,0,1,1,1]]

# Aggregate
params = {'type_attention': 'softmax', 'variance': 5}
attention_vector = aggregate_attention(rationales, row, params)
# [0.006, 0.035, 0.866, 0.086, 0.007]
```

#### 7.2.2 Text Preprocessing

**Function**: `ek_extra_preprocess(text, params, tokenizer)`  
**Location**: `Preprocess/preProcess.py`

```python
def ek_extra_preprocess(text, params, tokenizer=None):
    """
    Preprocess text using Ekphrasis + tokenization
    
    Args:
        text (str): Raw input text
        params (dict): Configuration with:
            - include_special: Keep special tags
            - bert_tokens: Use BERT tokenization
        tokenizer: BertTokenizer (if bert_tokens=True)
    
    Returns:
        list: Token IDs (BERT) or token strings (non-BERT)
    
    Process:
        1. Apply Ekphrasis preprocessing
        2. Optionally remove special tags
        3. For BERT: subword tokenization
        4. For non-BERT: word tokenization, remove punctuation
    
    Notes:
        - Ekphrasis handles social media normalization
        - BERT: WordPiece tokenization
        - Non-BERT: word-level tokens
    """
```

**Usage**:
```python
# BERT
params = {'bert_tokens': True, 'include_special': False}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = ek_extra_preprocess("#HateIsNotCool u suck!!!", params, tokenizer)
# [7123, 2003, 2025, 4658, 1057, 11891]

# Non-BERT
params = {'bert_tokens': False, 'include_special': False}
tokens = ek_extra_preprocess("#HateIsNotCool u suck!!!", params, None)
# ['hate', 'is', 'not', 'cool', 'u', 'suck']
```

#### 7.2.3 Model Saving/Loading

**Function**: `save_bert_model(model, tokenizer, params)`  
**Location**: `Models/utils.py`

```python
def save_bert_model(model, tokenizer, params):
    """
    Save BERT model and tokenizer
    
    Args:
        model: SC_weighted_BERT model
        tokenizer: BertTokenizer
        params (dict): Configuration for naming
    
    Saves:
        Saved/{model_name}_{config}/
            ├── config.json
            ├── pytorch_model.bin
            ├── vocab.txt
            └── tokenizer files
    
    Notes:
        - Uses HuggingFace save_pretrained
        - Directory name encodes configuration
        - Can be loaded with from_pretrained
    """
```

**Function**: `load_model(model, params)`  
**Location**: `Models/utils.py`

```python
def load_model(model, params, use_cuda=False):
    """
    Load trained weights for non-BERT models
    
    Args:
        model: Initialized model (empty weights)
        params (dict): Configuration for path
        use_cuda (bool): Deprecated (use map_location)
    
    Returns:
        torch.nn.Module: Model with loaded weights
    
    Notes:
        - Loads from Saved/ directory
        - Path constructed from params
        - Uses torch.load with CPU map_location
    """
```

**Usage**:
```python
# Save BERT
save_bert_model(model, tokenizer, params)

# Load BERT
model = SC_weighted_BERT.from_pretrained('Saved/bert-base-uncased_11_6_3_0.001')

# Load non-BERT
model = BiRNN(params, embeddings)
model = load_model(model, params)
```

#### 7.2.4 Parameter Loading

**Function**: `return_params(path_name, att_lambda, num_classes)`  
**Location**: `Models/utils.py`

```python
def return_params(path_name, att_lambda, num_classes=3):
    """
    Load parameters from JSON file
    
    Args:
        path_name (str): Path to JSON config file
        att_lambda (float): Attention loss weight
        num_classes (int): Number of classes (2 or 3)
    
    Returns:
        dict: Full configuration dictionary
    
    Process:
        1. Load JSON file
        2. Convert string booleans to bool
        3. Convert numeric strings to int
        4. Parse weight arrays
        5. Add att_lambda and num_classes
        6. Construct save paths
        7. Add data_file and class_names paths
    
    Notes:
        - Handles JSON type conversions
        - Constructs full save paths
        - Adds default values
    """
```

**Usage**:
```python
params = return_params(
    'best_model_json/bestModel_bert.json',
    att_lambda=0.001,
    num_classes=3
)
```

### 7.3 Loss Functions

#### 7.3.1 Masked Cross Entropy

**Function**: `masked_cross_entropy(input1, target, mask)`  
**Location**: `Models/utils.py`

```python
def masked_cross_entropy(input1, target, mask):
    """
    Cross-entropy loss ignoring padded positions
    
    Args:
        input1 (torch.Tensor): Predicted attention (batch, seq_len)
        target (torch.Tensor): Target attention (batch, seq_len)
        mask (torch.Tensor): Binary mask (batch, seq_len)
            1 = real token, 0 = padding
    
    Returns:
        torch.Tensor: Scalar loss value
    
    Process:
        For each sample in batch:
            1. Extract non-padded positions using mask
            2. Compute cross-entropy on those positions
            3. Average across batch
    
    Notes:
        - Essential for attention supervision
        - Prevents loss from padding tokens
        - Uses torch.nn.LogSoftmax internally
    """
```

**Usage**:
```python
# During training
attention_loss = masked_cross_entropy(
    predicted_attention,  # Model's attention
    human_rationales,     # Ground truth
    attention_mask        # Padding mask
)

total_loss = classification_loss + lambda * attention_loss
```

### 7.4 Embeddings

#### 7.4.1 GloVe to Word2Vec Conversion

**Script**: `convert_to_word2vec.py`

```python
from gensim.scripts.glove2word2vec import glove2word2vec

# Convert GloVe format to Word2Vec format
glove_input_file = 'Data/glove.840B.300d.txt'
word2vec_output_file = 'Data/word2vec.txt'

glove2word2vec(glove_input_file, word2vec_output_file)

# Load as KeyedVectors
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format(word2vec_output_file)

# Save in Gensim format
model.save('Data/word2vec.model')
```

#### 7.4.2 Vocabulary Creation

**Class**: `Vocab_own`  
**Location**: `TensorDataset/datsetSplitter.py`

```python
class Vocab_own:
    """
    Vocabulary with GloVe embeddings
    
    Attributes:
        itos (dict): Index to string mapping
        stoi (dict): String to index mapping
        vocab (dict): Word frequencies
        embeddings (np.ndarray): Embedding matrix (vocab_size, 300)
    
    Methods:
        load_embeddings(word): Get embedding for word
        create_vocab(): Build vocab from training data
    """
    
    def create_vocab(self):
        """
        Create vocabulary from training dataframe
        
        Process:
            1. Iterate through all tokens in training data
            2. Look up embedding in GloVe
            3. Add to vocabulary if new
            4. Build index mappings
            5. Create embedding matrix
        
        Notes:
            - Index 0 reserved for <pad> (zero vector)
            - Unknown words mapped to 'unk' embedding
            - Builds embeddings matrix for nn.Embedding
        """
```

**Usage**:
```python
# Load Word2Vec model
word2vec_model = KeyedVectors.load("Data/word2vec.model")

# Create vocabulary
vocab = Vocab_own(train_dataframe, word2vec_model)
vocab.create_vocab()

# Access
token_id = vocab.stoi['hate']
token = vocab.itos[token_id]
embedding = vocab.embeddings[token_id]  # (300,)
```

---

## 8. Configuration & Parameters

### 8.1 Complete Parameter Reference

All parameters are documented in `Parameters_description.md`. Here's a comprehensive reference:

#### 8.1.1 Data Parameters

| Parameter | Type | Description | Default | Options |
|-----------|------|-------------|---------|---------|
| `data_file` | str | Path to dataset.json | `'Data/dataset.json'` | Any valid path |
| `class_names` | str | Path to class encoder .npy | `'Data/classes.npy'` | `classes.npy` (3-class) or `classes_two.npy` (2-class) |
| `num_classes` | int | Number of output classes | `3` | `2` or `3` |
| `max_length` | int | Maximum sequence length | `128` | Any positive int |
| `include_special` | bool | Keep Ekphrasis special tags | `False` | `True`/`False` |
| `majority` | int | Minimum annotators for majority | `2` | `2` or `3` |

#### 8.1.2 Preprocessing Parameters

| Parameter | Type | Description | Default | Options |
|-----------|------|-------------|---------|---------|
| `bert_tokens` | bool | Use BERT tokenization | `True/False` | `True`: BERT, `False`: Word-level |
| `type_attention` | str | Attention normalization | `'softmax'` | `'softmax'`, `'neg_softmax'`, `'sigmoid'` |
| `variance` | int | Attention scaling factor | `5` or `10` | Any positive int |
| `decay` | bool | Apply attention decay | `False` | `True`/`False` |
| `window` | int | Decay window size | `4` | Any positive int |
| `alpha` | float | Decay strength | `0.5` | 0.0 to 1.0 |
| `p_value` | float | Geometric decay parameter | `0.8` | 0.0 to 1.0 |
| `method` | str | Decay method | `'additive'` | `'additive'`, `'geometric'` |
| `normalized` | bool | Normalize after decay | `False` | `True`/`False` |

#### 8.1.3 Model Parameters - BERT

| Parameter | Type | Description | Default | Options |
|-----------|------|-------------|---------|---------|
| `path_files` | str | HuggingFace model name | `'bert-base-uncased'` | Any BERT-compatible model |
| `what_bert` | str | BERT variant | `'weighted'` | `'weighted'` (only option) |
| `dropout_bert` | float | Dropout after BERT | `0.1` | 0.0 to 1.0 |
| `train_att` | bool | Train with attention supervision | `True/False` | `True`/`False` |
| `supervised_layer_pos` | int | Which BERT layer to supervise | `11` | 0 to 11 |
| `num_supervised_heads` | int | Number of heads to supervise | `6` | 1 to 12 |
| `save_only_bert` | bool | Save only BERT (not classifier) | `False` | `True`/`False` |

#### 8.1.4 Model Parameters - Non-BERT

| Parameter | Type | Description | Default | Options |
|-----------|------|-------------|---------|---------|
| `model_name` | str | Model architecture | `'birnn'` | `'birnn'`, `'birnnatt'`, `'birnnscrat'`, `'cnn_gru'` |
| `seq_model` | str | RNN type | `'lstm'` | `'lstm'`, `'gru'` |
| `hidden_size` | int | RNN hidden dimension | `256` | Any positive int |
| `embed_size` | int | Embedding dimension | `300` | 300 (GloVe) |
| `drop_embed` | float | Dropout after embedding | `0.3` | 0.0 to 1.0 |
| `drop_fc` | float | Dropout after FC layers | `0.2` | 0.0 to 1.0 |
| `drop_hidden` | float | RNN dropout | `0.3` | 0.0 to 1.0 |
| `train_embed` | bool | Train embedding layer | `False` | `True`/`False` |
| `attention` | str | Attention type (BiAtt models) | `'softmax'` | `'softmax'`, `'sigmoid'` |

#### 8.1.5 Training Parameters

| Parameter | Type | Description | Default | Options |
|-----------|------|-------------|---------|---------|
| `epochs` | int | Number of training epochs | `20` | Any positive int |
| `batch_size` | int | Training batch size | `16` (BERT), `32` (others) | Any positive int |
| `learning_rate` | float | Optimizer learning rate | `2e-5` (BERT), `0.001` (others) | Any positive float |
| `epsilon` | float | Adam epsilon | `1e-8` | Small positive float |
| `att_lambda` | float | Attention loss weight | `0.001` | 0.0 to 1.0 |
| `auto_weights` | bool | Auto-calculate class weights | `True` | `True`/`False` |
| `weights` | list | Manual class weights | `[1.0, 1.0, 1.0]` | List of floats |
| `random_seed` | int | Random seed | `42` | Any int |

#### 8.1.6 System Parameters

| Parameter | Type | Description | Default | Options |
|-----------|------|-------------|---------|---------|
| `device` | str | Compute device | `'cuda'` | `'cuda'`, `'cpu'` |
| `logging` | str | Logging backend | `'local'` | `'local'`, `'neptune'` |
| `to_save` | bool | Save trained model | `True` | `True`/`False` |
| `is_model` | bool | Model already loaded | `False` | `True`/`False` |

### 8.2 Configuration Files

#### 8.2.1 BERT with Attention Supervision

**File**: `best_model_json/bestModel_bert_base_uncased_Attn_train_TRUE.json`

```json
{
    "bert_tokens": "True",
    "what_bert": "weighted",
    "path_files": "bert-base-uncased",
    "num_classes": 3.0,
    "batch_size": 16.0,
    "epochs": 20.0,
    "learning_rate": 2e-05,
    "dropout_bert": 0.1,
    "train_att": "True",
    "supervised_layer_pos": 11.0,
    "num_supervised_heads": 6.0,
    "att_lambda": 0.001,
    "max_length": 128.0,
    "type_attention": "softmax",
    "variance": 5.0,
    "auto_weights": "True",
    "device": "cuda",
    "random_seed": 42.0
}
```

**Use Case**: Best explainability, good classification performance

#### 8.2.2 BERT without Attention Supervision

**File**: `best_model_json/bestModel_bert_base_uncased_Attn_train_FALSE.json`

```json
{
    "bert_tokens": "True",
    "what_bert": "weighted",
    "path_files": "bert-base-uncased",
    "num_classes": 3.0,
    "batch_size": 16.0,
    "epochs": 20.0,
    "learning_rate": 2e-05,
    "dropout_bert": 0.1,
    "train_att": "False",
    "att_lambda": 0.0,
    "max_length": 128.0,
    "type_attention": "softmax",
    "variance": 5.0,
    "auto_weights": "True",
    "device": "cuda",
    "random_seed": 42.0
}
```

**Use Case**: Best classification performance, baseline explainability

#### 8.2.3 BiRNN with Attention

**File**: `best_model_json/bestModel_birnnatt.json`

```json
{
    "bert_tokens": "False",
    "model_name": "birnnatt",
    "seq_model": "lstm",
    "hidden_size": 256.0,
    "embed_size": 300.0,
    "num_classes": 3.0,
    "batch_size": 32.0,
    "epochs": 20.0,
    "learning_rate": 0.001,
    "drop_embed": 0.3,
    "drop_fc": 0.2,
    "drop_hidden": 0.3,
    "train_embed": "False",
    "train_att": "True",
    "att_lambda": 0.01,
    "attention": "softmax",
    "max_length": 128.0,
    "type_attention": "softmax",
    "variance": 10.0,
    "auto_weights": "True",
    "device": "cuda",
    "random_seed": 42.0
}
```

**Use Case**: Lighter model, faster inference, good explainability

### 8.3 Creating Custom Configurations

#### 8.3.1 Template

```python
custom_params = {
    # Data
    "data_file": "Data/dataset.json",
    "class_names": "Data/classes.npy",
    "num_classes": 3,
    
    # Model Selection
    "bert_tokens": True,  # or False
    
    # If BERT
    "what_bert": "weighted",
    "path_files": "bert-base-uncased",
    "dropout_bert": 0.1,
    "train_att": True,
    "supervised_layer_pos": 11,
    "num_supervised_heads": 6,
    
    # If Non-BERT
    "model_name": "birnnatt",
    "seq_model": "lstm",
    "hidden_size": 256,
    "embed_size": 300,
    "drop_embed": 0.3,
    "drop_fc": 0.2,
    "drop_hidden": 0.3,
    "train_embed": False,
    "attention": "softmax",
    
    # Training
    "epochs": 20,
    "batch_size": 16,  # 16 for BERT, 32 for others
    "learning_rate": 2e-5,  # 2e-5 for BERT, 0.001 for others
    "epsilon": 1e-8,
    "att_lambda": 0.001,
    "auto_weights": True,
    
    # Preprocessing
    "max_length": 128,
    "type_attention": "softmax",
    "variance": 5,
    "include_special": False,
    "decay": False,
    
    # System
    "device": "cuda",
    "random_seed": 42,
    "logging": "local",
    "to_save": True,
    "is_model": True
}
```

#### 8.3.2 Saving Custom Config

```python
import json

with open('my_config.json', 'w') as f:
    json.dump(custom_params, f, indent=4)
```

#### 8.3.3 Loading Custom Config

```python
from Models.utils import return_params

params = return_params(
    'my_config.json',
    att_lambda=0.001,
    num_classes=3
)
```

### 8.4 Parameter Tuning Guidelines

#### 8.4.1 For Better Classification

- **Increase epochs**: 20-30 for complex datasets
- **Tune learning rate**: 
  - BERT: 1e-5 to 5e-5
  - RNN: 0.0001 to 0.01
- **Batch size**: 16-32 (limited by GPU memory)
- **Dropout**: 0.1-0.3 (higher if overfitting)
- **Class weights**: Use `auto_weights=True` for imbalanced data

#### 8.4.2 For Better Explainability

- **Enable attention training**: `train_att=True`
- **Tune att_lambda**: 0.0001 to 0.01
  - Higher = more focus on explanations
  - Lower = more focus on classification
- **Supervise more heads**: `num_supervised_heads=6-12`
- **Use last layer**: `supervised_layer_pos=11`
- **Higher variance**: `variance=10` (sharper attention)

#### 8.4.3 For Faster Training

- **Smaller model**: Use BiRNN instead of BERT
- **Smaller batch**: 8-16 (more updates)
- **Fewer epochs**: 10-15 (with early stopping)
- **Freeze embeddings**: `train_embed=False`
- **Single GPU**: Avoid distributed training overhead

#### 8.4.4 For Better Generalization

- **Higher dropout**: 0.3-0.5
- **Data augmentation**: (not implemented, but possible)
- **Regularization**: L2 weight decay in optimizer
- **Class balancing**: `auto_weights=True`
- **Cross-validation**: Train on different splits

---

## 9. File Structure Reference

### 9.1 Project Directory Tree

```
HateXplain/
│
├── Data/                          # Dataset files
│   ├── dataset.json              # Main dataset (20k posts)
│   ├── post_id_divisions.json    # Train/val/test splits
│   ├── classes.npy               # 3-class encoder
│   ├── classes_two.npy           # 2-class encoder
│   ├── glove.840B.300d.txt       # GloVe embeddings (download separately)
│   ├── word2vec.model            # Converted embeddings
│   └── README.md                 # Data documentation
│
├── Preprocess/                    # Preprocessing pipeline
│   ├── __init__.py
│   ├── dataCollect.py            # Main data collection
│   ├── preProcess.py             # Ekphrasis + tokenization
│   ├── attentionCal.py           # Attention aggregation
│   ├── spanMatcher.py            # Rationale matching
│   └── utils.py                  # Helper functions
│
├── Models/                        # Neural network models
│   ├── __init__.py
│   ├── bertModels.py             # BERT-based models
│   ├── otherModels.py            # BiRNN, CNN-GRU
│   ├── attentionLayer.py         # Attention mechanisms
│   └── utils.py                  # Model utilities
│
├── TensorDataset/                 # Data loading
│   ├── __init__.py
│   ├── dataLoader.py             # PyTorch DataLoaders
│   └── datsetSplitter.py         # Train/val/test splitting
│
├── best_model_json/               # Pre-trained model configs
│   ├── bestModel_bert_base_uncased_Attn_train_TRUE.json
│   ├── bestModel_bert_base_uncased_Attn_train_FALSE.json
│   ├── bestModel_birnn.json
│   ├── bestModel_birnnatt.json
│   ├── bestModel_birnnscrat.json
│   └── bestModel_cnn_gru.json
│
├── Saved/                         # Saved model checkpoints
│   └── (generated during training)
│
├── eraserbenchmark/               # Explainability evaluation
│   ├── rationale_benchmark/
│   │   ├── metrics.py           # ERASER metrics
│   │   └── utils.py
│   └── params/                   # ERASER configs
│
├── Figures/                       # Visualizations
│   └── (generated figures)
│
├── manual_training_inference.py   # Main training script
├── testing_with_lime.py          # LIME explanations
├── testing_with_rational.py      # Rationale evaluation
├── testing_for_bias.py           # Bias evaluation
├── parameters_selection.py       # Hyperparameter search
├── convert_to_word2vec.py        # GloVe conversion
│
├── Example_HateExplain.ipynb     # Usage tutorial
├── Explainability_Calculation_NB.ipynb  # Explainability eval
├── Bias_Calculation_NB.ipynb     # Bias eval
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
├── Parameters_description.md      # Parameter documentation
├── Plan.md                        # Migration plan (3.7→3.13)
├── LICENSE                        # License
└── COMPREHENSIVE_DOCUMENTATION.md # This file
```

### 9.2 Generated Files & Directories

#### 9.2.1 Preprocessed Data

**Location**: `Data/`

```
Data/Total_data_bert_softmax_5_128_3.pickle
Data/Total_data_bert_softmax_5_128_3/
    ├── train_data.pickle
    ├── val_data.pickle
    ├── test_data.pickle
    └── vocab_own.pickle (non-BERT only)
```

**Naming Convention**:
```
Total_data_{bert/normal}_{attention_type}_{variance}_{max_length}_{num_classes}
```

#### 9.2.2 Saved Models

**For BERT**:
```
Saved/bert-base-uncased_11_6_3_0.001/
    ├── config.json
    ├── pytorch_model.bin
    ├── vocab.txt
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

**For Non-BERT**:
```
Saved/birnnatt_lstm_256_3_0.01.pth
```

**Naming Convention**:
- BERT: `{model}_{layer}_{heads}_{classes}_{lambda}/`
- Non-BERT: `{model}_{rnn}_{hidden}_{classes}_{lambda}.pth`

#### 9.2.3 Evaluation Results

```
explanations_dicts/
    ├── model_rationales.json     # Model explanations
    └── scores.json               # ERASER metrics

Dataset_Eraser_Format/
    └── (ERASER-formatted data)
```

### 9.3 Important Files Description

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `manual_training_inference.py` | Main training script | `train_model()`, `Eval_phase()`, `select_model()` |
| `testing_with_lime.py` | LIME explanations | `modelPred`, LIME integration |
| `testing_with_rational.py` | Rationale evaluation | `standaloneEval_with_rational()` |
| `testing_for_bias.py` | Bias measurement | Bias evaluation functions |
| `Preprocess/dataCollect.py` | Data preprocessing | `collect_data()`, `get_training_data()` |
| `Preprocess/attentionCal.py` | Attention aggregation | `aggregate_attention()`, `softmax()` |
| `Models/bertModels.py` | BERT architecture | `SC_weighted_BERT` class |
| `Models/otherModels.py` | Non-BERT architectures | `BiRNN`, `BiAtt_RNN`, `CNN_GRU` |
| `Models/attentionLayer.py` | Attention mechanisms | `Attention_LBSA`, `Attention_LBSA_sigmoid` |
| `TensorDataset/dataLoader.py` | Data loading | `combine_features()`, `pad_sequences()` |
| `TensorDataset/datsetSplitter.py` | Data splitting | `createDatasetSplit()`, `Vocab_own` |

### 9.4 Dependencies & Requirements

**File**: `requirements.txt`

```
# Core ML/DL Libraries - Python 3.13 compatible
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
more-itertools>=10.0.0
```

**Installation**:
```bash
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

---

## 10. Usage Examples

### 10.1 Quick Start

#### 10.1.1 Training a BERT Model

```python
# Load configuration
from Models.utils import return_params
import torch

params = return_params(
    'best_model_json/bestModel_bert_base_uncased_Attn_train_TRUE.json',
    att_lambda=0.001,
    num_classes=3
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Train
from manual_training_inference import train_model
train_model(params, device)
```

**Expected Output**:
```
Loading data...
total_data 19826
Creating train/val/test splits...
train: 15861, val: 1983, test: 1982

Training...
Epoch 1/20
  avg_train_loss: 0.876
  Train - fscore: 0.45, accuracy: 0.52
  Val   - fscore: 0.51, accuracy: 0.58
  Test  - fscore: 0.50, accuracy: 0.57

Epoch 2/20
  ...

Best validation F1: 0.68
Best test F1: 0.67
Model saved to Saved/bert-base-uncased_11_6_3_0.001/
```

#### 10.1.2 Training a BiRNN Model

```python
params = return_params(
    'best_model_json/bestModel_birnnatt.json',
    att_lambda=0.01,
    num_classes=3
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ensure GloVe embeddings are downloaded and converted
# See section 10.2.1 for GloVe setup

train_model(params, device)
```

### 10.2 Setup & Preparation

#### 10.2.1 Download & Convert GloVe Embeddings

```bash
# Download GloVe (2GB)
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip -d Data/

# Convert to Word2Vec format
python convert_to_word2vec.py
```

**convert_to_word2vec.py** (if needed):
```python
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Convert
glove_input = 'Data/glove.840B.300d.txt'
word2vec_output = 'Data/word2vec.txt'

print("Converting GloVe to Word2Vec format...")
glove2word2vec(glove_input, word2vec_output)

# Load and save as Gensim model
print("Loading and saving as Gensim model...")
model = KeyedVectors.load_word2vec_format(word2vec_output, binary=False)
model.save('Data/word2vec.model')

print("Done! Saved to Data/word2vec.model")
```

#### 10.2.2 Verify Dataset

```python
import json
import numpy as np

# Load dataset
with open('Data/dataset.json', 'r') as f:
    dataset = json.load(f)

print(f"Total posts: {len(dataset)}")

# Load splits
with open('Data/post_id_divisions.json', 'r') as f:
    splits = json.load(f)

print(f"Train: {len(splits['train'])}")
print(f"Val: {len(splits['val'])}")
print(f"Test: {len(splits['test'])}")

# Load class encoders
classes_3 = np.load('Data/classes.npy', allow_pickle=True)
classes_2 = np.load('Data/classes_two.npy', allow_pickle=True)

print(f"3-class: {classes_3}")
print(f"2-class: {classes_2}")
```

### 10.3 Inference on New Text

#### 10.3.1 Using Pre-trained BERT Model

```python
import torch
from transformers import BertTokenizer
from Models.bertModels import SC_weighted_BERT
import numpy as np

# Load model
model_path = 'Saved/bert-base-uncased_11_6_3_0.001/'
model = SC_weighted_BERT.from_pretrained(
    model_path,
    num_labels=3,
    output_attentions=True
)
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Class names
classes = ['hatespeech', 'normal', 'offensive']

def predict_text(text):
    # Tokenize
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(
            encoded['input_ids'],
            attention_mask=encoded['attention_mask']
        )
    
    # Get prediction
    logits = outputs[0][0]
    probs = torch.softmax(logits, dim=0).numpy()
    pred_class = classes[np.argmax(probs)]
    
    # Get attention
    attention = outputs[1][11][0].mean(dim=0)[0].numpy()
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    
    return {
        'text': text,
        'prediction': pred_class,
        'probabilities': {
            classes[i]: float(probs[i]) for i in range(len(classes))
        },
        'tokens': tokens,
        'attention': attention.tolist()
    }

# Example
result = predict_text("I hate you so much!")
print(f"Prediction: {result['prediction']}")
print(f"Probabilities: {result['probabilities']}")
```

#### 10.3.2 Using Pre-trained BiRNN Model

```python
import torch
from gensim.models import KeyedVectors
from Models.otherModels import BiAtt_RNN
from Models.utils import load_model
import numpy as np

# Load configuration
from Models.utils import return_params
params = return_params(
    'best_model_json/bestModel_birnnatt.json',
    att_lambda=0.01,
    num_classes=3
)

# Load embeddings
word2vec = KeyedVectors.load('Data/word2vec.model')

# Load vocabulary (from training)
import pickle
with open('Data/Total_data_normal_softmax_10_128_3/vocab_own.pickle', 'rb') as f:
    vocab = pickle.load(f)

# Initialize model
params['embed_size'] = vocab.embeddings.shape[1]
params['vocab_size'] = vocab.embeddings.shape[0]

model = BiAtt_RNN(params, vocab.embeddings, return_att=True)
model = load_model(model, params)
model.eval()

# Class names
classes = ['hatespeech', 'normal', 'offensive']

def predict_text_rnn(text):
    from Preprocess.preProcess import ek_extra_preprocess
    
    # Preprocess
    tokens = ek_extra_preprocess(text, params, None)
    
    # Convert to IDs
    token_ids = []
    for token in tokens:
        try:
            idx = vocab.stoi[token]
        except KeyError:
            idx = vocab.stoi['unk']
        token_ids.append(idx)
    
    # Pad
    from TensorDataset.dataLoader import pad_sequences
    input_ids = pad_sequences([token_ids], maxlen=128, dtype="long")
    
    # Create attention mask
    attention_mask = [[int(tid > 0) for tid in input_ids[0]]]
    
    # Convert to tensors
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
    
    # Predict
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            attention_vals=None,
            labels=None
        )
    
    # Get results
    logits = outputs[0][0]
    attention = outputs[1][0]
    probs = torch.softmax(logits, dim=0).numpy()
    pred_class = classes[np.argmax(probs)]
    
    return {
        'text': text,
        'prediction': pred_class,
        'probabilities': {
            classes[i]: float(probs[i]) for i in range(len(classes))
        },
        'tokens': tokens,
        'attention': attention.numpy().tolist()
    }

# Example
result = predict_text_rnn("I hate you so much!")
print(f"Prediction: {result['prediction']}")
```

### 10.4 Generating Explanations

#### 10.4.1 LIME Explanations

```python
from lime.lime_text import LimeTextExplainer
import numpy as np

# Initialize explainer
explainer = LimeTextExplainer(
    class_names=['hatespeech', 'normal', 'offensive']
)

# Create prediction function
def predict_proba_for_lime(texts):
    """Wrapper for LIME that takes list of texts"""
    results = []
    for text in texts:
        result = predict_text(text)  # Using function from 10.3.1
        probs = [result['probabilities'][c] for c in classes]
        results.append(probs)
    return np.array(results)

# Generate explanation
text = "You stupid Muslim terrorist go back to your country"
explanation = explainer.explain_instance(
    text,
    predict_proba_for_lime,
    num_features=10,
    num_samples=5000
)

# Show results
print(f"Prediction: {classes[explanation.top_labels[0]]}")
print("\nTop features:")
for word, weight in explanation.as_list():
    print(f"  {word:20s}: {weight:+.3f}")

# Visualize
explanation.show_in_notebook(text=True)
```

#### 10.4.2 Attention Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(tokens, attention, title="Attention Visualization"):
    """
    Visualize attention weights over tokens
    
    Args:
        tokens: List of token strings
        attention: List of attention weights (same length as tokens)
        title: Plot title
    """
    # Normalize attention
    attention = np.array(attention)
    attention = attention / attention.max()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 3))
    
    # Plot each token with attention-based coloring
    for i, (token, attn) in enumerate(zip(tokens, attention)):
        color = plt.cm.Reds(attn)
        ax.text(i, 0, token,
                bbox=dict(facecolor=color, alpha=0.8, boxstyle='round,pad=0.5'),
                fontsize=12, ha='center', va='center')
    
    ax.set_xlim(-0.5, len(tokens) - 0.5)
    ax.set_ylim(-1, 1)
    ax.axis('off')
    ax.set_title(title, fontsize=14, pad=20)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, 
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', 
                        pad=0.1, aspect=30)
    cbar.set_label('Attention Weight', fontsize=12)
    
    plt.tight_layout()
    return fig

# Example
result = predict_text("You stupid Muslim terrorist go back")
fig = visualize_attention(
    result['tokens'],
    result['attention'],
    title=f"Prediction: {result['prediction']}"
)
plt.show()
```

### 10.5 Batch Processing

#### 10.5.1 Process Multiple Texts

```python
import pandas as pd
from tqdm import tqdm

def batch_predict(texts, batch_size=32):
    """
    Predict on multiple texts efficiently
    
    Args:
        texts: List of strings
        batch_size: Batch size for processing
    
    Returns:
        List of prediction dictionaries
    """
    results = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        
        # Tokenize batch
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(
                encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
        
        # Process results
        logits = outputs[0]
        probs = torch.softmax(logits, dim=1).numpy()
        predictions = classes[np.argmax(probs, axis=1)]
        
        for j, text in enumerate(batch):
            results.append({
                'text': text,
                'prediction': predictions[j],
                'probabilities': {
                    classes[k]: float(probs[j][k]) 
                    for k in range(len(classes))
                }
            })
    
    return results

# Example
test_texts = [
    "I love everyone!",
    "You are terrible",
    "Great work!",
    "Kill all [slur]",
    # ... more texts
]

predictions = batch_predict(test_texts)
df = pd.DataFrame(predictions)
print(df)
```

### 10.6 Evaluation

#### 10.6.1 Evaluate on Test Set

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load test data
from TensorDataset.datsetSplitter import createDatasetSplit
from TensorDataset.dataLoader import combine_features

params = return_params(
    'best_model_json/bestModel_bert_base_uncased_Attn_train_TRUE.json',
    att_lambda=0.001,
    num_classes=3
)

train, val, test = createDatasetSplit(params)
test_loader = combine_features(test, params, is_train=False)

# Evaluate
from manual_training_inference import Eval_phase

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
f1, acc, prec, rec, auc, probs = Eval_phase(
    params, 'test', model, test_loader, device
)

print(f"Test F1: {f1:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall: {rec:.4f}")

# Detailed classification report
true_labels = [test[i][2] for i in range(len(test))]
pred_labels = [classes[np.argmax(p)] for p in probs]

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels))

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

---

## 11. Troubleshooting & FAQs

### 11.1 Common Issues

#### 11.1.1 Out of Memory (OOM) Error

**Problem**: CUDA out of memory during training

**Solutions**:
1. Reduce batch size:
   ```python
   params['batch_size'] = 8  # Instead of 16 or 32
   ```

2. Reduce sequence length:
   ```python
   params['max_length'] = 64  # Instead of 128
   ```

3. Use gradient accumulation:
   ```python
   # Accumulate gradients over multiple batches
   accumulation_steps = 4
   for i, batch in enumerate(train_loader):
       loss = model(...)
       loss = loss / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

4. Clear cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

#### 11.1.2 GloVe Embeddings Not Found

**Problem**: `FileNotFoundError: Data/word2vec.model`

**Solution**:
```bash
# Download GloVe
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip -d Data/

# Convert
python convert_to_word2vec.py
```

#### 11.1.3 Slow Training

**Problem**: Training is very slow

**Solutions**:
1. Check GPU usage:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device: {torch.cuda.get_device_name(0)}")
   ```

2. Increase batch size (if memory allows):
   ```python
   params['batch_size'] = 32  # or 64
   ```

3. Use DataLoader num_workers:
   ```python
   dataloader = DataLoader(
       dataset,
       batch_size=32,
       num_workers=4,  # Parallel data loading
       pin_memory=True  # Faster GPU transfer
   )
   ```

4. Use mixed precision training (PyTorch 1.6+):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   for batch in train_loader:
       with autocast():
           outputs = model(batch)
           loss = outputs[0]
       
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
       optimizer.zero_grad()
   ```

#### 11.1.4 Low F1 Score

**Problem**: Model achieves low F1 score (<0.5)

**Solutions**:
1. Check class imbalance:
   ```python
   params['auto_weights'] = True
   ```

2. Tune learning rate:
   ```python
   # Try different values
   params['learning_rate'] = 1e-5  # Lower
   # or
   params['learning_rate'] = 5e-5  # Higher
   ```

3. Train longer:
   ```python
   params['epochs'] = 30  # Instead of 20
   ```

4. Check data quality:
   ```python
   # Verify preprocessing
   from Preprocess.dataCollect import collect_data
   data = collect_data(params)
   print(data.head())
   print(data['Label'].value_counts())
   ```

### 11.2 Frequently Asked Questions

**Q1: Can I use other BERT models (e.g., RoBERTa, DistilBERT)?**

A: Yes, but you'll need to modify the code:
```python
# In Models/bertModels.py
from transformers import RobertaModel, RobertaConfig

class SC_weighted_RoBERTa(RobertaPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        # ... rest similar to SC_weighted_BERT
```

**Q2: How do I train on 2-class instead of 3-class?**

A: Change the class encoder:
```python
params['num_classes'] = 2
params['class_names'] = 'Data/classes_two.npy'
```

**Q3: Can I use the models for other hate speech datasets?**

A: Yes, but you'll need to:
1. Format your data like `dataset.json`
2. Create your own `post_id_divisions.json`
3. Optionally create new rationales (or use dummy rationales)

**Q4: How do I interpret attention weights?**

A: Higher attention = model focused more on that token for its decision. Use visualization (section 10.4.2) to see which words influenced the prediction.

**Q5: What's the difference between `train_att=True` and `False`?**

A:
- `True`: Model is trained to match human rationales (better explainability)
- `False`: Model learns attention naturally (possibly better classification)

**Q6: How long does training take?**

A: Approximate times (on single GPU):
- BERT: 2-4 hours per epoch
- BiRNN: 15-30 minutes per epoch
- CNN-GRU: 10-20 minutes per epoch

**Q7: Can I use multiple GPUs?**

A: Yes, modify the training script to use `DataParallel` or `DistributedDataParallel`:
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

**Q8: How do I save predictions to a file?**

A:
```python
import json

predictions = batch_predict(texts)

with open('predictions.json', 'w') as f:
    json.dump(predictions, f, indent=2)
```

---

## 12. Citation & License

### 12.1 Citation

If you use this code or dataset, please cite:

```bibtex
@inproceedings{mathew2021hatexplain,
  title={HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection},
  author={Mathew, Binny and Saha, Punyajoy and Yimam, Seid Muhie and Biemann, Chris and Goyal, Pawan and Mukherjee, Animesh},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={17},
  pages={14867--14875},
  year={2021}
}
```

### 12.2 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### 12.3 Acknowledgments

- **AAAI 2021**: For accepting the paper
- **HuggingFace**: For transformer models and hosting
- **ERASER Benchmark**: For explainability evaluation framework
- **Contributors**: All contributors to the repository

---

## 13. Additional Resources

### 13.1 Related Papers

1. **Original Paper**: [HateXplain on ArXiv](https://arxiv.org/abs/2012.10289)
2. **ERASER Benchmark**: DeYoung et al., "ERASER: A Benchmark to Evaluate Rationalized NLP Models"
3. **LIME**: Ribeiro et al., ""Why Should I Trust You?": Explaining the Predictions of Any Classifier"

### 13.2 External Links

- **Dataset on HuggingFace**: https://huggingface.co/datasets/hatexplain
- **Pre-trained Models**: https://huggingface.co/models?search=hatexplain
- **GitHub Repository**: https://github.com/punyajoy/HateXplain
- **Project Website**: (if available)

### 13.3 Contact & Support

For questions, issues, or contributions:
- **GitHub Issues**: https://github.com/punyajoy/HateXplain/issues
- **Email**: Contact authors (see paper)

---

**END OF DOCUMENTATION**

*This comprehensive documentation was generated on January 5, 2026 for the HateXplain project (Python 3.13 version). For the latest updates, check the GitHub repository.*
