# BlockRR
This repository implements BlockRR code for LabelDP training under class imbalance for the paper **A Unified Framework of RR-type Algorithms for Label Differential Privacy**.

# Abstack
In this paper, we introduce \textit{BlockRR}, a novel and unified randomized-response mechanism for label differential privacy. This framework generalizes classical RR and RRWithPrior as special cases under specific parameter settings, which eliminates the need for separate, case-by-case analysis. Theoretically, we prove that BlockRR satisfies $\epsilon$-label DP. We also design a partition method for BlockRR based on a weight matrix derived from label prior information; the parallel composition principle ensures that the composition of two such mechanisms remains $\epsilon$-label DP. Empirically, we evaluate BlockRR on two variants of CIFAR-10 with varying degrees of class imbalance. Results show that in the high-privacy and moderate-privacy regimes ($\epsilon \leq 3.0$), our propsed method gets a better balance between test accuaracy and the average of per-class accuracy. In the low-privacy regime ($\epsilon \geq 4.0$), all methods reduce BlockRR to standard RR without additional performance loss.

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
