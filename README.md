## BlockRR
This repository implements BlockRR code for LabelDP training under class imbalance for the paper **A Unified Framework of RR-type Algorithms for Label Differential Privacy**.

## Abstack
In this paper, we introduce \textit{BlockRR}, a novel and unified randomized-response mechanism for label differential privacy. This framework generalizes classical RR and RRWithPrior as special cases under specific parameter settings, which eliminates the need for separate, case-by-case analysis. Theoretically, we prove that BlockRR satisfies $\epsilon$-label DP. We also design a partition method for BlockRR based on a weight matrix derived from label prior information; the parallel composition principle ensures that the composition of two such mechanisms remains $\epsilon$-label DP. Empirically, we evaluate BlockRR on two variants of CIFAR-10 with varying degrees of class imbalance. Results show that in the high-privacy and moderate-privacy regimes ($\epsilon \leq 3.0$), our propsed method gets a better balance between test accuaracy and the average of per-class accuracy. In the low-privacy regime ($\epsilon \geq 4.0$), all methods reduce BlockRR to standard RR without additional performance loss.

### Quick Start
Run a single trial.This runs one experiment `--num_exp=1`on an imbalanced CIFAR-10_1 split.
```
python train_classification.py --num_exp=1
```
Logs are written to `--log_dir` as timestamped `.txt` files and also printed to console.

### Detailed
Choose a dataset split, implemented imbalanced CIFAR-10 splits:`CIFAR-10_1`and`CIFAR-10_2`. Choose a LabelDP mechanism, supported mechanisms:`rr`, `rrwithprior` and `rrwithweight` for our BlockRR. Note that the values of the hyperparameters `--sigma` and `--delta` only affect the training of rrwithweight, while the other two mechanisms are not affected.The functions `compute_randomized_labels_classification()` and `compute_randomized_labels_classification_infty()` respectively represent the perturbation with privacy budget epsilon $\leq 8.0$ and the perturbation without privacy protection. Therefore, when calling `compute_randomized_labels_classification_infty()`, you only need to set epsilon to a value greater than 8.0.

An example like
```
python train_classification.py \
  --dataset=cifar10_1 \
  --mechanism=rrwithweight \
  --epsilon=1.0 \
  --epsilon_p=1.0 \
  --sigma=1.0 \
  --delta=1 \
  --num_exp=10
```

### Logging & optional Weights & Biases (wandb)
A timestamped log file is created under `--log_dir`, and logs are also printed to stdout.

Use the following command to run if you want to use wandb:
```
python train_classification.py --use_wandb --wandb_project dp
```
The value of `--wandb_project` depends on the account and project that you have created youerself. And there is also an option for offline mode:
```
python train_classification.py --use_wandb --wandb_offline
```






