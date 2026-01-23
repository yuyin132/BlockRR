# BlockRR
A Unified Framework of RR-type Algorithms for Label Differential Privacy
# LabelDP Classification with Randomized Response Variants

This repository implements **label-randomized (Label Differential Privacy, LabelDP) image classification** under class imbalance, combined with **ResNet-based models, mixup training, and symmetric cross-entropy loss**.

The core idea is:
> **Randomize training labels under a LabelDP mechanism *before* model training**, then train a standard classifier on the randomized labels.

The implementation supports multiple randomized response variants that leverage **class priors and weighted label transitions**.

---

## ✨ Features

- **Label Differential Privacy (LabelDP)**
  - Classic Randomized Response (`rr`)
  - Prior-aware Randomized Response (`rrwithprior`)
  - Weighted Randomized Response (`rrwithweight`)
- **Private estimation of class prior**
  - Laplace mechanism with privacy budget `ε_p`
- **Robust training**
  - Mixup
  - Symmetric Cross Entropy (SCE)
- **Imbalanced CIFAR-10 benchmarks**
- **Pre-activation ResNet18 (ResNet v2)**
- **Multiple-run experiments with logging & optional wandb**

---

## 📁 Project Structure

```text
.
├── train_classification.py     # Main training & experiment entry
├── randomized.py               # Label randomization mechanisms (core)
├── datasets.py                 # Imbalanced CIFAR-10 construction
├── models.py                   # ResNet / PreActResNet definitions
├── utils/
│   ├── augmentation.py         # Cutout, Mixup, data augmentation
│   ├── report_log.py           # Logger utilities
│   └── privacy_randomized.py   # Generic DP label noise engine (optional)
└── README.md
