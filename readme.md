
### Layer Details

| Layer | Type | Parameters |
|-------|------|------------|
| 1 | Input | shape=(None, 120) |
| 2 | Embedding (GloVe) | vocabulary=20,000, dim=300, trainable=True |
| 3 | Bidirectional LSTM | 128 units, return_sequences=True, dropout=0.2, L2=1e-4 |
| 4 | Batch Normalization | - |
| 5 | Bidirectional LSTM | 64 units, return_sequences=True, dropout=0.2, L2=1e-4 |
| 6 | Self-Attention | Tanh → Softmax → Weighted Sum |
| 7 | Dense | 128 units, ReLU, L2=1e-4 |
| 8 | Dropout | 0.5 |
| 9 | Dense | 64 units, ReLU, L2=1e-4 |
| 10 | Dropout | 0.3 |
| 11 | Output (Softmax) | 6 units |

**Total Parameters:** 6,630,471

---

## Model Performance

### Accuracy Comparison

| Model | Accuracy |
|-------|----------|
| **Improved LSTM (Ours)** | **92.81%** |
| SVM (linear) | 88.81% |
| Basic LSTM | 88.75% |
| Logistic Regression | 88.00% |
| Random Forest | 87.97% |
| Multinomial NB | 78.69% |

### Per-Class Performance (F1-Score)

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Anger | 0.94 | 0.94 | 0.94 |
| Fear | 0.88 | 0.93 | 0.90 |
| Joy | 0.97 | 0.91 | 0.94 |
| Love | 0.77 | 0.89 | 0.82 |
| Sadness | 0.97 | 0.97 | 0.97 |
| Surprise | 0.81 | 0.83 | 0.82 |

### Confusion Matrix

| True/Predicted | Anger | Fear | Joy | Love | Sadness | Surprise |
|----------------|-------|------|-----|------|---------|----------|
| Anger | 406 | 5 | 2 | 9 | 5 | 5 |
| Fear | 3 | 361 | 4 | 6 | 10 | 3 |
| Joy | 12 | 15 | 972 | 27 | 41 | 5 |
| Love | 2 | 4 | 14 | 233 | 7 | 1 |
| Sadness | 6 | 8 | 6 | 8 | 901 | 4 |
| Surprise | 1 | 4 | 6 | 2 | 6 | 96 |

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Dataset Size | 16,000 samples |
| Training Split | 80% (12,800) |
| Testing Split | 20% (3,200) |
| Validation Split | 10% of training |
| Batch Size | 64 |
| Epochs | 20 (early stopping at 9) |
| Optimizer | Adam |
| Learning Rate | 0.001 (adaptive) |
| Loss Function | Categorical Cross-Entropy |

### Training Progress

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
|-------|-----------|---------|------------|----------|
| 1 | 46.27% | 73.59% | 1.3681 | 0.7569 |
| 2 | 82.14% | 88.44% | 0.5325 | 0.3460 |
| 3 | 93.08% | 88.83% | 0.2346 | 0.2992 |
| 4 | 95.81% | 90.08% | 0.1374 | 0.3116 |
| 5 | 96.82% | 90.08% | 0.1085 | 0.3699 |
| 6 | 97.63% | 88.98% | 0.0810 | 0.3909 |
| 7 | 95.75% | 93.20% | 0.2322 | 0.2900 |
| 8 | 96.76% | 93.91% | 0.2104 | 0.2938 |

---

## Key Techniques Used

### 1. GloVe Embeddings (300-dim)
- Pre-trained on 6 billion tokens
- Captures semantic relationships between words
- Improves accuracy by ~4% over random embeddings

### 2. Bidirectional LSTM
- Processes text in both forward and backward directions
- Captures context from preceding and following words

### 3. Self-Attention Mechanism
- Identifies emotionally salient words in the input
- Assigns higher weights to emotionally important tokens

### 4. Dropout Regularization
- Dropout rates: 0.2 (LSTM), 0.5 (Dense), 0.3 (Dense)
- Prevents overfitting

### 5. L2 Regularization
- Factor: 1×10⁻⁴
- Keeps weights small and balanced

---

## Files in `saved_models/`

| File | Description | Size |
|------|-------------|------|
| `emotion_lstm_final.keras` | Trained LSTM model | ~25 MB |
| `tokenizer_glove.pkl` | GloVe tokenizer | ~2 MB |
| `label_encoder.pkl` | Label encoder | 1 KB |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer (for baseline models) | ~10 MB |
| `best_ml_model.pkl` | Best traditional ML model (SVM) | ~15 MB |
| `training_history.pkl` | Training history logs | 5 KB |

---

## How to Load and Use the Model

### Prerequisites

```bash
pip install tensorflow numpy nltk scikit-learn