# AgriNet-27: A Robust Explainable Hybrid CNN–ViT Framework for Multi-Class Plant Disease Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-97.02%25-brightgreen.svg)
![Params](https://img.shields.io/badge/Parameters-36.89M-lightblue.svg)

**AgriNet-27** is a hybrid deep learning architecture combining **EfficientNetV2-B0** (CNN) with an isotropic **Vision Transformer (ViT)** via a novel **Cross-Multi-Head Attention (Cross-MHA)** fusion mechanism for 27-class plant disease classification across 4 crop species.

</div>

---

## 📋 Table of Contents
- [Model Overview](#model-overview)
- [Performance](#performance)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Load Pretrained Model](#load-pretrained-model)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [XAI Explainability](#xai-explainability)
- [Citation](#citation)

---

## 🌿 Model Overview

| Property | Value |
|---|---|
| **Model Name** | AgriNet-27 |
| **Task** | Multi-class Plant Disease Classification |
| **Classes** | 27 (4 plant species) |
| **Backbone** | EfficientNetV2-B0 + ViT (4 layers, 8 heads) |
| **Fusion** | Cross-Multi-Head Attention (Cross-MHA) |
| **Total Parameters** | 36,893,291 |
| **Trainable Parameters** | 36,832,587 |
| **Model Size (FP32)** | 140.76 MB |
| **Model Size (FP16)** | 70.38 MB |
| **GFLOPs** | 4.78 |
| **Input Size** | 224 × 224 × 3 |

---

## 📊 Performance

### Main 27-Class Benchmark (705-image held-out test set)

| Metric | Value |
|---|---|
| **Accuracy** | **97.02%** |
| **Precision** | **96.98%** |
| **Recall** | **97.02%** |
| **F1-score** | **97.00%** |
| **AUC (micro)** | **0.9985** |
| **AUC (macro)** | **0.9986** |
| **Test Loss** | **0.0458** |

### 5-Fold Cross-Validation (development set, test excluded)

| Fold | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
|---|---|---|---|---|
| Fold 1 | 96.81 | 96.76 | 96.81 | 96.78 |
| Fold 2 | 97.16 | 97.12 | 97.16 | 97.14 |
| Fold 3 | 97.02 | 96.98 | 97.02 | 97.00 |
| Fold 4 | 96.88 | 96.84 | 96.88 | 96.86 |
| Fold 5 | 97.23 | 97.20 | 97.23 | 97.21 |
| **Mean ± Std** | **97.02 ± 0.18** | **96.98 ± 0.18** | **97.02 ± 0.18** | **97.00 ± 0.18** |

### 95% Confidence Intervals

| Metric | Mean (%) | Lower CI | Upper CI |
|---|---|---|---|
| Accuracy | 97.02 | 96.80 | 97.24 |
| Precision | 96.98 | 96.76 | 97.20 |
| Recall | 97.02 | 96.80 | 97.24 |
| F1-score | 97.00 | 96.78 | 97.22 |

### Cross-Domain Transfer (PlantCity-52 dataset)

| Metric | Value |
|---|---|
| Accuracy | **98.27%** |
| Macro Precision | 97.82% |
| Macro Recall | 97.94% |
| Macro F1 | 97.85% |

### Hardware Efficiency

| Platform | Precision | Latency | Throughput | Power |
|---|---|---|---|---|
| Workstation / Cloud (NVIDIA L4) | FP32 | 11.2 ms | 89.3 FPS | ≈ 45 W |
| Edge (Jetson Orin Nano) | FP16 | 46.8 ms | 21.4 FPS | 12.4 W (0.58 J/frame) |

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/SaifuddinSagor/AgriNet27.git
cd AgriNet27

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```
tensorflow==2.16.1
numpy>=1.24.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
albumentations>=1.3.0
lime>=0.2.0
scipy>=1.11.0
statsmodels>=0.14.0
```

---

## 🚀 Quick Start

```python
import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model('best_model.keras')

# Class names (27 classes)
CLASS_NAMES = [
    'GAB', 'GAN', 'GAS', 'GDM', 'GEL', 'GFD', 'GHE', 'GMV',
    'HCS', 'HEM', 'HFI', 'HHE', 'HME', 'HSE', 'HSD', 'HWR',
    'PBB', 'PCI', 'PCY', 'PHE', 'PPS', 'PYN',
    'TAS', 'TBB', 'TGB', 'THE', 'TRS'
]

def preprocess(image_path):
    """Apply AgriNet-27 preprocessing pipeline"""
    img = cv2.imread(image_path)
    # Step 1: Denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # Step 2: CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    # Step 3: Gamma correction
    inv = 1.0 / 1.5
    table = (((np.arange(256) / 255.0) ** inv) * 255).astype('uint8')
    img = cv2.LUT(img, table)
    # Step 4: Min-Max normalize
    f = img.astype(np.float32)
    mn, mx = float(f.min()), float(f.max())
    if mx - mn > 1e-6:
        img = np.clip((f - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)
    # Resize & normalize to [0,1]
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def predict(image_path, model, top_k=3):
    """Predict plant disease from image"""
    img = preprocess(image_path)
    img_batch = np.expand_dims(img, axis=0)
    
    probs = model.predict(img_batch, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    
    print(f"\n🌿 AgriNet-27 Prediction Results")
    print(f"{'='*40}")
    for i, idx in enumerate(top_indices):
        print(f"  Top-{i+1}: {CLASS_NAMES[idx]:>4s} → {probs[idx]*100:.2f}%")
    print(f"{'='*40}")
    return CLASS_NAMES[top_indices[0]], float(probs[top_indices[0]])

# Run prediction
predicted_class, confidence = predict('your_leaf_image.jpg', model)
print(f"\nPrediction: {predicted_class} ({confidence*100:.1f}% confidence)")
```

---

## 📦 Load Pretrained Model

### Option 1: Load `.keras` checkpoint (Recommended)
```python
import tensorflow as tf

# Load full model (architecture + weights)
model = tf.keras.models.load_model(
    'best_model.keras',
    custom_objects={
        'PrependCLS':             PrependCLS,
        'AddPositionalEmbedding': AddPositionalEmbedding,
        'TakeToken':              TakeToken,
        'TransformerEncoderBlock':TransformerEncoderBlock,
        'RandomGaussianNoise':    RandomGaussianNoise,
    }
)
print("Model loaded successfully!")
print(f"Parameters: {model.count_params():,}")
```

### Option 2: Load weights only
```python
import tensorflow as tf

# Build model first (run AgriNet27.ipynb cells P-13)
model, backbone = build_agrinet27(num_classes=27)

# Load saved weights
model.load_weights('stage1.weights.h5')
print("Weights loaded successfully!")
```

### Option 3: Rebuild from scratch and load
```python
# 1. Run the notebook: AgriNet27.ipynb
# 2. Weights auto-save to: /kaggle/working/runs/AgriNet-27/best_model.keras
# 3. Load as shown in Option 1
```

---

## 📁 Dataset

### Primary Dataset: Agri-Vision Bangladesh (27 classes)

| Species | Classes | Images |
|---|---|---|
| **Bottle Gourd** (*Lagenaria siceraria*) | 8 | 1,711 |
| **Papaya** (*Carica papaya*) | 6 | 1,765 |
| **Hibiscus** (*Hibiscus rosa-sinensis*) | 8 | 1,165 |
| **Tea** (*Camellia sinensis*) | 5 | 247 |
| **Total** | **27** | **4,688** |

### Data Split

| Split | Images | Augmented |
|---|---|---|
| Train (70%) | 3,273 | 27,000 (balanced) |
| Validation (15%) | 710 | None |
| Test (15%) | 705 | None |

### Download Dataset
```python
# Bottle Gourd, Papaya dataset
# https://doi.org/10.17632/8t6k37ztxc.3

# Hibiscus, Tea dataset  
# https://doi.org/10.17632/5bzy89brkv.6
```

### Class Labels

```python
CLASS_NAMES = {
    # Gourd (Lagenaria siceraria)
    'GAB': 'Gourd Alternaria Blight',
    'GAN': 'Gourd Anthracnose',
    'GAS': 'Gourd Angular Spot',
    'GDM': 'Gourd Downy Mildew',
    'GEL': 'Gourd Early Alternaria Blight',
    'GFD': 'Gourd Fungal Damage',
    'GHE': 'Gourd Healthy',
    'GMV': 'Gourd Mosaic Virus',
    # Papaya (Carica papaya)
    'PBB': 'Papaya Bacterial Blight',
    'PCI': 'Papaya Carica Insect Hole',
    'PCY': 'Papaya Curled Yellow Spot',
    'PHE': 'Papaya Healthy',
    'PPS': 'Papaya Pathogen Symptoms',
    'PYN': 'Papaya Yellow Necrotic Spots',
    # Hibiscus (Hibiscus rosa-sinensis)
    'HCS': 'Hibiscus Citruspot',
    'HEM': 'Hibiscus Early Mild Spotting',
    'HFI': 'Hibiscus Fungal Infected',
    'HHE': 'Hibiscus Healthy',
    'HME': 'Hibiscus Mild Edge Damage',
    'HSE': 'Hibiscus Senescent',
    'HSD': 'Hibiscus Slightly Diseased',
    'HWR': 'Hibiscus Wrinkled',
    # Tea (Camellia sinensis)
    'TAS': 'Tea Algal Spot',
    'TBB': 'Tea Brown Blight',
    'TGB': 'Tea Grey Blight',
    'THE': 'Tea Healthy',
    'TRS': 'Tea Red Spot',
}
```

---

## 🏗️ Architecture

```
Input (224×224×3)
       │
       ▼
┌─────────────────────────────────────────────┐
│         EfficientNetV2-B0 Backbone          │
│         Output: (7×7×1280)                  │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌──────────────┐  ┌──────────────────────────┐
│ Multi-Scale  │  │   Vision Transformer     │
│ CNN Path     │  │   (ViT Encoder)          │
│              │  │                          │
│ 7×7 → 768   │  │  Tokenize (49 patches)   │
│ 4×4 → 768   │  │  + CLS token + PosEmbed  │
│ 2×2 → 768   │  │  4× Transformer Blocks   │
│              │  │  (8 heads, dim=768)      │
│ GAP → concat │  │  Extract CLS token       │
│ → Dense(768) │  │  → (1×768)              │
└──────┬───────┘  └───────────┬──────────────┘
       │  (Query)             │  (Key, Value)
       └──────────┬───────────┘
                  ▼
     ┌────────────────────────┐
     │   Cross-MHA Fusion     │
     │  Q=CNN, K=V=ViT        │
     │  → fused (1×768)       │
     └────────────┬───────────┘
                  │
                  ▼
     Concatenate [fused ∥ CLS ∥ CNN_emb]
                  │  (2304-dim)
                  ▼
          LayerNorm → Dense(768) → Dropout(0.2)
                  │
                  ▼
           Dense(27) → Softmax
                  │
                  ▼
            Predictions (27 classes)
```

### Layer-wise Parameters

| Layer | Output Shape | Params |
|---|---|---|
| EfficientNetV2-B0 Backbone | (7×7×1280) | 5,923,920 |
| Multi-Scale Projections (×3) | (3×768) | 2,951,424 |
| Transformer Embedding (CLS+Pos) | (50×768) | 38,400 |
| 4× Transformer Encoder Blocks | (50×768) | 22,055,936 |
| CNN Embedding Dense | (768) | 1,770,240 |
| Cross-Attention Fusion (MHA) | (1×768) | 2,362,368 |
| Final Fusion Layer (Dense+LN) | (768) | 1,774,848 |
| Classification Head | (27) | 20,763 |
| **Total** | | **36,893,291** |

---

## 🔍 XAI Explainability

AgriNet-27 integrates 4 complementary XAI methods:

```python
# 1. LIME — Super-pixel level explanation
from lime import lime_image
explainer = lime_image.LimeImageExplainer(random_state=42)
explanation = explainer.explain_instance(
    image, model.predict, top_labels=1,
    hide_color=0, num_samples=100
)

# 2. Grad-CAM — Coarse localization
# See AgriNet27.ipynb → Cell P-29 for full implementation

# 3. Grad-CAM++ — Multi-instance lesion localization
# Higher-order gradients for scattered lesions

# 4. ViT Attention Rollout — Global structural dependencies
# Tracks attention across all 4 transformer layers
```

---

## 📂 Repository Structure

```
AgriNet27/
├── AgriNet27_update_V1.ipynb      # Main training notebook
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── class_names.json                # 27 class label mapping
├── split_manifest_70_15_15.csv     # Train/val/test split manifest
├── cv_5fold_manifest.csv           # 5-fold CV manifest
└── runs/
    └── AgriNet-27/
        ├── best_model.keras        # Best checkpoint (upload to Releases)
        ├── stage1.weights.h5       # Stage-1 weights
        ├── classification_report.csv
        ├── confusion_matrix_counts.png
        ├── roc_curve.png
        ├── precision_recall_curve.png
        ├── training_curves.png
        ├── xai_combined_panel.png
        ├── cross_validation.json
        ├── confidence_intervals.json
        └── wilcoxon_holm_results.csv
```

---

## 📈 Training Configuration

```python
# Hyperparameters (Table 5 in paper)
CONFIG = {
    'seed':              42,
    'epochs':            30,        # converged at 12-15
    'batch_size':        32,
    'img_size':          224,
    'lr':                1e-4,
    'optimizer':         'AdamW',
    'beta1':             0.9,
    'beta2':             0.999,
    'weight_decay':      1e-5,
    'loss':              'CategoricalCrossentropy',
    'label_smoothing':   0.05,
    'early_stop_patience': 5,
    'reduce_lr_factor':  0.2,
    'reduce_lr_patience': 3,
    # Architecture
    'proj_dim':          768,
    'vit_depth':         4,
    'vit_heads':         8,
    'mlp_dim':           2048,
    'dropout':           0.2,
    # Preprocessing
    'denoise_h':         10,
    'clahe_clip':        2.0,
    'gamma':             1.5,
}
```

---

## 🔬 Statistical Validation

### Wilcoxon Signed-Rank Test (Holm–Bonferroni adjusted)

| Comparison | W | Z | p-value | Effect (r) | Significance |
|---|---|---|---|---|---|
| vs. Swin Transformer | 345.0 | 3.74 | 1.84×10⁻⁴ | 0.825 | ✅ Significant |
| vs. EfficientNetV2-B0 | 361.0 | 4.12 | 7.58×10⁻⁵ | 0.910 | ✅ Significant |
| vs. Xception | 378.0 | 4.53 | 2.36×10⁻⁵ | 1.000 | ✅ Significant |
| vs. ViT-B/16 | 378.0 | 4.53 | 2.36×10⁻⁵ | 1.000 | ✅ Significant |

---

## 📄 Citation

If you use AgriNet-27 in your research, please cite:

```bibtex
@article{sagor2026agrinet27,
  title   = {AgriNet-27: A Robust Explainable Hybrid CNN--ViT Framework 
             with Cross-Attention for Multi-Class Plant Disease Classification},
  author  = {Sagor, Saifuddin and Hossan, Md. Faysal and Rahman, Kazi Md. Naimur 
             and Assaduzzaman, Md and Fahad, Nafiz and Hui, Liew Tze and Ohidujjaman},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {Under Review}
}
```

---

## 📬 Contact

| Author | Institution | Email |
|---|---|---|
| Saifuddin Sagor | Daffodil International University | — |
| Liew Tze Hui* | Multimedia University, Malaysia | thliew@mmu.edu.my |

*Corresponding author

---

## 📜 License

This project is licensed under the **MIT License**.

The datasets are publicly available via Mendeley Data:
- Bottle Gourd & Papaya: https://doi.org/10.17632/8t6k37ztxc.3
- Hibiscus & Tea: https://doi.org/10.17632/5bzy89brkv.6

---

<div align="center">
<b>AgriNet-27 — Precision Agriculture through Explainable Hybrid Deep Learning</b>
</div>
