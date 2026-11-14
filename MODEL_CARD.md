# 🌿 AgriNet-27 — Model Card

---

## 📌 Model Overview

| Property | Detail |
|---|---|
| **Model Name** | AgriNet-27 |
| **Developed by** | Saifuddin Sagor, Md. Faysal Hossan, Kazi Md. Naimur Rahman, Md. Assaduzzaman, Nafiz Fahad, Liew Tze Hui, Ohidujjaman |
| **Affiliations** | Daffodil International University, Bangladesh & Multimedia University, Malaysia |
| **Model Type** | Hybrid CNN + Vision Transformer (Cross-MHA Fusion) |
| **Framework** | TensorFlow 2.16.1 / Keras |
| **Task** | Multi-class Plant Disease Classification |
| **Classes** | 27 (4 plant species) |
| **License** | MIT |

---

## 🎯 Intended Use

### ✅ Primary Use Cases
- Automated detection and diagnosis of plant diseases from leaf images
- Precision agriculture surveillance and early warning systems
- Agricultural decision support tools for field practitioners
- Research baseline for plant pathology deep learning

### ❌ Out-of-Scope Use
- Plant species not in training set — requires fine-tuning for other crops
- Images from significantly different imaging conditions without domain adaptation
- Medical, clinical, or non-agricultural applications

---

## 📊 Evaluation Results

### Test Set Performance — 705 held-out images

| Metric | Score |
|---|---|
| **Accuracy** | **97.02%** |
| **Macro Precision** | 96.98% |
| **Macro Recall** | 97.02% |
| **Macro F1-score** | 97.00% |
| **Micro-AUC** | 0.9985 |
| **Macro-AUC** | 0.9986 |
| **Cohen's Kappa** | 0.9689 |
| **MCC** | 0.9693 |
| **Test Loss** | 0.0458 |

### 5-Fold Cross-Validation — development set (3,983 images, test excluded)

| Fold | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
|---|---|---|---|---|
| Fold 1 | 96.81 | 96.76 | 96.81 | 96.78 |
| Fold 2 | 97.16 | 97.12 | 97.16 | 97.14 |
| Fold 3 | 97.02 | 96.98 | 97.02 | 97.00 |
| Fold 4 | 96.88 | 96.84 | 96.88 | 96.86 |
| Fold 5 | 97.23 | 97.20 | 97.23 | 97.21 |
| **Mean ± Std** | **97.02 ± 0.18** | **96.98 ± 0.18** | **97.02 ± 0.18** | **97.00 ± 0.18** |

### 95% Confidence Intervals (t-distribution, df=4)

| Metric | Mean (%) | Lower CI | Upper CI |
|---|---|---|---|
| Accuracy | 97.02 | 96.80 | 97.24 |
| Precision | 96.98 | 96.76 | 97.20 |
| Recall | 97.02 | 96.80 | 97.24 |
| F1-score | 97.00 | 96.78 | 97.22 |

### Cross-Domain Transfer — PlantCity-52 (supervised fine-tuning, 25% held-out)

| Metric | Score |
|---|---|
| **Accuracy** | **98.27%** |
| Macro Precision | 97.82% |
| Macro Recall | 97.94% |
| Macro F1-score | 97.85% |

### Statistical Significance — Wilcoxon Signed-Rank Test (Holm–Bonferroni adjusted)

| Comparison | W | Z | Holm p-adj | Effect (r) | Result |
|---|---|---|---|---|---|
| vs. Swin Transformer | 345.0 | 3.74 | 1.84×10⁻⁴ | 0.825 | ✅ Significant |
| vs. EfficientNetV2-B0 | 361.0 | 4.12 | 7.58×10⁻⁵ | 0.910 | ✅ Significant |
| vs. Xception | 378.0 | 4.53 | 2.36×10⁻⁵ | 1.000 | ✅ Significant |
| vs. ViT-B/16 | 378.0 | 4.53 | 2.36×10⁻⁵ | 1.000 | ✅ Significant |

---

## 🏗️ Architecture

| Component | Detail |
|---|---|
| **Backbone** | EfficientNetV2-B0 (trainable, ImageNet-1k init) |
| **Multi-Scale CNN Path** | 7×7, 4×4, 2×2 → GAP → concat → Dense(768) |
| **ViT Encoder** | 4 layers, 8 heads, dim=768, MLP=2048 |
| **Fusion Mechanism** | Cross-Multi-Head Attention (Cross-MHA) |
| **Fusion Query** | CNN multi-scale embedding |
| **Fusion Key/Value** | Full ViT transformer sequence |
| **Classification Head** | LayerNorm → Dense(768) → Dropout(0.2) → Dense(27) |
| **Total Parameters** | 36,893,291 |
| **Trainable Parameters** | 36,832,587 |
| **Non-trainable Parameters** | 60,704 |
| **Model Size (FP32)** | 140.76 MB |
| **Model Size (FP16)** | 70.38 MB |
| **GFLOPs** | 4.78 |

---

## 🏋️ Training Configuration

| Setting | Value |
|---|---|
| **Platform** | Lightning AI — NVIDIA L4 GPU (20 GB VRAM) |
| **Input Size** | 224 × 224 × 3 |
| **Epochs** | 30 (converged at 12–15) |
| **Batch Size** | 32 |
| **Optimizer** | AdamW (β₁=0.9, β₂=0.999) |
| **Learning Rate** | 1×10⁻⁴ |
| **Weight Decay** | 1×10⁻⁵ |
| **Gradient ClipNorm** | 1.0 |
| **Loss Function** | Categorical Cross-Entropy |
| **Label Smoothing** | 0.05 |
| **Early Stopping** | patience=12, monitor=val_auc |
| **LR Scheduler** | ReduceLROnPlateau (factor=0.5, patience=12, monitor=val_loss) |
| **Random Seed** | 42 |

---

## 📁 Training Data

### Dataset: Agri-Vision Bangladesh — 27 Classes, 4 Species

| Species | Classes | Total Images | Collection Site |
|---|---|---|---|
| Bottle Gourd (*Lagenaria siceraria*) | 8 | 1,711 | Daffodil Smart City, Bangladesh |
| Papaya (*Carica papaya*) | 6 | 1,765 | Daffodil Smart City, Bangladesh |
| Hibiscus (*Hibiscus rosa-sinensis*) | 8 | 1,165 | Daffodil International University & Sylhet |
| Tea (*Camellia sinensis*) | 5 | 247 | Sylhet, Bangladesh |
| **Total** | **27** | **4,688** | — |

### Class Labels

| Code | Full Name | Species |
|---|---|---|
| GAB | Gourd Alternaria Blight | Bottle Gourd |
| GAN | Gourd Anthracnose | Bottle Gourd |
| GAS | Gourd Angular Spot | Bottle Gourd |
| GDM | Gourd Downy Mildew | Bottle Gourd |
| GEL | Gourd Early Alternaria Blight | Bottle Gourd |
| GFD | Gourd Fungal Damage | Bottle Gourd |
| GHE | Gourd Healthy | Bottle Gourd |
| GMV | Gourd Mosaic Virus | Bottle Gourd |
| PBB | Papaya Bacterial Blight | Papaya |
| PCI | Papaya Carica Insect Hole | Papaya |
| PCY | Papaya Curled Yellow Spot | Papaya |
| PHE | Papaya Healthy | Papaya |
| PPS | Papaya Pathogen Symptoms | Papaya |
| PYN | Papaya Yellow Necrotic Spots | Papaya |
| HCS | Hibiscus Citruspot | Hibiscus |
| HEM | Hibiscus Early Mild Spotting | Hibiscus |
| HFI | Hibiscus Fungal Infected | Hibiscus |
| HHE | Hibiscus Healthy | Hibiscus |
| HME | Hibiscus Mild Edge Damage | Hibiscus |
| HSE | Hibiscus Senescent | Hibiscus |
| HSD | Hibiscus Slightly Diseased | Hibiscus |
| HWR | Hibiscus Wrinkled | Hibiscus |
| TAS | Tea Algal Spot | Tea |
| TBB | Tea Brown Blight | Tea |
| TGB | Tea Grey Blight | Tea |
| THE | Tea Healthy | Tea |
| TRS | Tea Red Spot | Tea |

### Data Split

| Split | Raw Images | Augmented | Note |
|---|---|---|---|
| Train (70%) | 3,273 | 27,000 | Balanced to 1,000/class |
| Validation (15%) | 710 | None | Unaugmented, real field photos |
| Test (15%) | 705 | None | Unaugmented, strictly quarantined |

### Preprocessing Pipeline (applied to all splits)
1. **FastNlMeans Denoising** — h=10, hColor=10, templateWindowSize=7
2. **CLAHE** — clipLimit=2.0, tileGridSize=8×8
3. **Gamma Correction** — γ=1.5
4. **Min-Max Normalization** — [0, 255]

### Augmentation (training split only)
Horizontal flip · Random zoom (0.6–0.8×) · Rotation (±30°) · Affine shift (10–20%) · Gaussian noise · Brightness & contrast (±30%)

---

## ⚡ Hardware Efficiency

| Platform | Precision | Parameters | GFLOPs | Latency | Throughput | Power |
|---|---|---|---|---|---|---|
| Workstation / Cloud (NVIDIA L4) | FP32 | 36.89 M | 4.78 | 11.2 ms | 89.3 FPS | ≈ 45 W |
| Edge (Jetson Orin Nano) | FP16 | 36.89 M | 4.78 | 46.8 ms | 21.4 FPS | 12.4 W (0.58 J/frame) |

---

## 🔍 Explainability (XAI)

| Method | Type | Purpose |
|---|---|---|
| **LIME** | Perturbation-based | Super-pixel pathological boundary identification |
| **Grad-CAM** | Gradient-based | Coarse class-discriminative region localization |
| **Grad-CAM++** | Gradient-based | Multi-instance fine-grained lesion localization |
| **ViT Attention Rollout** | Attention-based | Global structural leaf deformation analysis |

---

## ⚠️ Limitations & Biases

- **Geographic bias** — Dataset collected in Bangladesh; may not generalize to all regions without fine-tuning
- **Species scope** — 4 plant species only; other crops require transfer learning
- **Image-level split** — Partitioning is image-level stratified (not plant/session-disjoint) due to absence of specimen-level metadata in source repositories
- **Edge constraints** — 21.4 FPS on Jetson Orin Nano; ultra-low-power MCUs require INT8 quantization and structured pruning
- **Co-infection** — Complex polymicrobial co-infections (e.g., simultaneous fungal + viral) not explicitly modeled

---

## 📦 Artifacts & Reproducibility

| Artifact | Description |
|---|---|
| `AgriNet27.ipynb` | Full training, evaluation & XAI notebook |
| `best_model.keras` | Best checkpoint (upload to GitHub Releases) |
| `stage1.weights.h5` | Stage-1 trained weights |
| `split_manifest_70_15_15.csv` | Exact train/val/test image-level split |
| `cv_5fold_manifest.csv` | 5-fold CV fold assignments |
| `class_names.json` | Class index → label mapping |
| `requirements.txt` | Pinned dependency versions |

---

## 🗄️ Dataset Access

| Dataset | DOI |
|---|---|
| Bottle Gourd & Papaya (Agri-Vision Bangladesh) | https://doi.org/10.17632/8t6k37ztxc.3 |
| Hibiscus & Tea | https://doi.org/10.17632/5bzy89brkv.6 |
| PlantCity-52 (transfer learning) | https://doi.org/10.17632/w8kh2xkspx.4 |
