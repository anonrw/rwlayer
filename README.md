# RW-Layer: Read-Write Layer for Neural Networks

**A learnable data correction layer that can be integrated into any neural network architecture.**

> *Paper under review at KDD 2026*

## Key Idea

Traditional ML operates under a **read-only paradigm**: models learn from data that is never modified. The RW-Layer introduces a **read-write paradigm** where the model learns to correct input features during training.

The RW-Layer computes:

```
z = W^k x
```

where `W` is initialized as the **identity matrix** and has **no activation function**. This means:
- At initialization: `z = x` (data is preserved)
- During training: `W` deviates from identity only when corrections reduce the loss
- The transformation remains **linear**, preserving the input feature space

## Results

RW-MLP achieves **Rank 1** among 15 methods on 39 regression datasets:

| Method | Mean R² | Rank |
|--------|---------|------|
| **RW-MLP** | **0.5912** | **1** |
| MLP | 0.5862 | 2 |
| RW-FT-Transformer | 0.5790 | 3 |
| FT-Transformer | 0.5774 | 4 |
| RW-ResNet | 0.5772 | 5 |
| CatBoost | 0.5762 | 6 |
| RandomForest | 0.5749 | 7 |
| RTDL-ResNet | 0.5730 | 8 |
| LightGBM | 0.5613 | 9 |
| XGBoost | 0.5409 | 11 |

**RW-Layer helps MLP-based architectures (54-62% win rate) but hurts attention-based models like NODE and TabNet (38-41% win rate).**

## Quick Start

### Try it yourself (< 5 minutes)

```bash
pip install torch scikit-learn numpy matplotlib
python try_it.py
```

This runs the RW-Layer on 5 regression datasets, comparing MLP vs RW-MLP with multiple k values, and visualizes what the correction matrix learned.

### Full installation

```bash
pip install -r requirements.txt
```

## RW-Layer Implementation

The full RW-Layer is only ~15 lines of PyTorch:

```python
class RWLayer(nn.Module):
    def __init__(self, n_features, k=1):
        super().__init__()
        self.k = k
        self.W = nn.Parameter(torch.eye(n_features))   # Identity init
        self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        W_k = self.W
        for _ in range(self.k - 1):
            W_k = W_k @ self.W                         # Matrix power W^k
        return x @ W_k.T + self.bias                    # No activation
```

To add the RW-Layer to **any** model, simply prepend it:

```python
class RWMLP(nn.Module):
    def __init__(self, input_dim, k=1):
        super().__init__()
        self.rw = RWLayer(input_dim, k=k)
        self.mlp = MLP(input_dim)

    def forward(self, x):
        x = self.rw(x)       # Correct the data
        return self.mlp(x)   # Then predict
```

## Why Does It Work?

The RW-Layer succeeds because it implements **data correction** rather than data transformation:

1. **Identity initialization** (`W = I`): The layer starts by preserving the input exactly
2. **No activation function**: The output stays in the same space as the input
3. **Matrix power W^k**: Amplifies small learned corrections by factor k

Together, the layer learns only the adjustments necessary to improve predictions while preserving the original feature structure. Neither component alone works -- the synergy is essential.

## Ablation Study

| Variant | Init | Activation | W^k | Win Rate vs Linear-MLP |
|---------|------|------------|-----|----------------------|
| MLP | - | - | - | Baseline |
| Linear-MLP | Random | ReLU | No | Baseline |
| Identity-MLP | Identity | ReLU | No | 62.1% |
| NoAct-MLP | Random | None | No | 69.0% |
| IdentityNoAct-MLP | Identity | None | k=1 | 82.8% |
| **RW-MLP** | **Identity** | **None** | **Yes** | **86.2%** |

## Repository Structure

```
rw-layer/
├── README.md                              # This file
├── try_it.py                              # Quick demo (< 5 min)
├── requirements.txt                       # Dependencies
├── rw_layer.py                            # Core RW-Layer module
│
├── experiments/
│   ├── rw_layer_benchmark_full.py         # Main benchmark (Table 1, Fig 2a-c)
│   ├── established_methods_experiment.py  # ResNet, NODE, TabNet (Table 2, Fig 2d)
│   └── ablation_study.py                 # Ablation study (Table 3, Fig 2e-f)
│
└── results/
    ├── combined_method_rankings.csv       # Final rankings (15 methods)
    └── analysis/
        └── rw_layer_analysis.py           # Publication figures & tables
```

## Reproducing Paper Results

### Experiment 1: Main benchmark (Table 1, Figure 2a-c)

MLP, RW-MLP, FT-Transformer, RW-FT-Transformer + 4 tree-based methods on 39 datasets, 10 seeds:

```bash
# Quick test (5 datasets, ~30 min)
python experiments/rw_layer_benchmark_full.py

# Full benchmark (39+ datasets, ~24h)
python experiments/rw_layer_benchmark_full.py --full --benchmark combined
```

### Experiment 2: Established architectures (Table 2, Figure 2d)

RW-ResNet, RW-NODE, RW-TabNet on 39 datasets:

```bash
python experiments/established_methods_experiment.py
```

### Experiment 3: Ablation study (Table 3, Figure 2e-f)

6 variants isolating identity init, no activation, and W^k:

```bash
python experiments/ablation_study.py
```

## Evaluation Protocol

Following the [TabPFN](https://www.nature.com/articles/s41586-024-08328-6) evaluation protocol:

- **Data**: 39 regression datasets from OpenML-CTR23 + AMLB Suite 269
- **Split**: 90% train / 10% test; training further split 89/11 for validation
- **Repetitions**: 10 random seeds per dataset
- **Metrics**: R², RMSE, MAE, Spearman correlation
- **k selection**: Tuned from {1, 2, 3, 5, 10} via validation loss
- **Training**: Adam optimizer, weight decay 1e-4, early stopping (patience 20)
- **Subsampling**: Datasets > 10,000 samples subsampled with fixed seed

## Citation

```bibtex
@article{rwlayer2025,
  title={Read \& Write Layer},
  author={Anonymous},
  journal={Under review at KDD 2026},
  year={2025}
}
```

## License

This project is released for research purposes.
