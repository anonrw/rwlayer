"""
Try It: RW-Layer Quick Demo
============================

Demonstrates the RW-Layer on 5 regression datasets from scikit-learn/OpenML.
Compares MLP vs RW-MLP and visualizes what the correction matrix W learned.

Usage:
    python try_it.py

Requirements:
    pip install torch scikit-learn numpy matplotlib

Runtime: ~3-5 minutes on CPU
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.datasets import (
    fetch_california_housing,
    load_diabetes,
    load_wine,
)
import warnings
import os
import random

warnings.filterwarnings('ignore')

# ==============================================================================
# RW-Layer (self-contained for easy copy-paste)
# ==============================================================================
class RWLayer(nn.Module):
    """Read-Write Layer: z = x @ W^k + bias, identity init, no activation."""
    def __init__(self, n_features, k=1):
        super().__init__()
        self.k = k
        self.W = nn.Parameter(torch.eye(n_features))
        self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        W_k = self.W
        for _ in range(self.k - 1):
            W_k = W_k @ self.W
        return x @ W_k.T + self.bias


class MLP(nn.Module):
    """Standard 2-layer MLP."""
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RWMLP(nn.Module):
    """MLP with RW-Layer prepended."""
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1, k=1):
        super().__init__()
        self.rw = RWLayer(input_dim, k=k)
        self.mlp = MLP(input_dim, hidden_dim, dropout)

    def forward(self, x):
        return self.mlp(self.rw(x))


# ==============================================================================
# Training
# ==============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test,
                       scaler_y, epochs=100, lr=1e-3, patience=15, batch_size=64, seed=42):
    """Train model with early stopping, return R² on test set."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_va = torch.FloatTensor(X_val).to(device)
    y_va = torch.FloatTensor(y_val).to(device)

    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

    best_val_loss = float('inf')
    best_state = None
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(loss.item())

        train_losses.append(np.mean(epoch_loss))

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va), y_va).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)

    # Evaluate on test
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()

    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    y_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    return r2_score(y_orig, pred), train_losses, val_losses


# ==============================================================================
# Datasets
# ==============================================================================
def load_datasets():
    """Load 5 diverse regression datasets."""
    datasets = {}

    # 1. California Housing (subsample to 2000 for speed)
    data = fetch_california_housing()
    np.random.seed(42)
    idx = np.random.choice(len(data.data), 2000, replace=False)
    datasets['California Housing'] = (data.data[idx], data.target[idx])

    # 2. Diabetes
    data = load_diabetes()
    datasets['Diabetes'] = (data.data, data.target)

    # 3. Wine Quality (use wine dataset features for regression)
    data = load_wine()
    datasets['Wine'] = (data.data, data.target.astype(float))

    # 4-5. Synthetic datasets with known noise patterns
    np.random.seed(42)
    n = 1000

    # Friedman #1 with noise
    X = np.random.uniform(0, 1, (n, 10))
    y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5)**2 + 10 * X[:, 3] + 5 * X[:, 4]
    # Add noise to some features (RW-Layer should correct this)
    X_noisy = X.copy()
    X_noisy[:, 0] += np.random.normal(0, 0.3, n)
    X_noisy[:, 1] += np.random.normal(0, 0.3, n)
    datasets['Friedman #1 (noisy features)'] = (X_noisy, y)

    # Correlated features with redundancy
    X = np.random.randn(n, 8)
    X[:, 4] = X[:, 0] * 0.8 + np.random.randn(n) * 0.2  # correlated with feature 0
    X[:, 5] = X[:, 1] * 0.7 + np.random.randn(n) * 0.3  # correlated with feature 1
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] + np.random.randn(n) * 0.5
    datasets['Correlated Features'] = (X, y)

    return datasets


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 70)
    print("  RW-Layer: Quick Demo")
    print("  Comparing MLP vs RW-MLP on 5 regression datasets")
    print("=" * 70)
    print(f"\nDevice: {device}")

    datasets = load_datasets()
    k_values = [1, 3, 5]
    seeds = [42, 142, 242]

    all_results = {}
    rw_models = {}  # Store for visualization

    for ds_name, (X, y) in datasets.items():
        print(f"\n{'-' * 60}")
        print(f"Dataset: {ds_name} ({X.shape[0]} samples, {X.shape[1]} features)")
        print(f"{'-' * 60}")

        mlp_r2s = []
        rw_r2s = {k: [] for k in k_values}

        for seed in seeds:
            set_seed(seed)

            # Split: 80% train, 10% val, 10% test
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=seed)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.11, random_state=seed)

            # Scale
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_train_s = scaler_X.fit_transform(X_train)
            X_val_s = scaler_X.transform(X_val)
            X_test_s = scaler_X.transform(X_test)
            y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
            y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

            n_features = X_train_s.shape[1]

            # MLP baseline
            set_seed(seed)
            model = MLP(n_features)
            r2, _, _ = train_and_evaluate(model, X_train_s, y_train_s, X_val_s, y_val_s,
                                          X_test_s, y_test_s, scaler_y, seed=seed)
            mlp_r2s.append(r2)

            # RW-MLP with different k values
            for k in k_values:
                set_seed(seed)
                model = RWMLP(n_features, k=k)
                r2, _, _ = train_and_evaluate(model, X_train_s, y_train_s, X_val_s, y_val_s,
                                              X_test_s, y_test_s, scaler_y, seed=seed)
                rw_r2s[k].append(r2)

                # Save last model for visualization
                if seed == seeds[-1]:
                    rw_models[(ds_name, k)] = model

        # Print results
        mlp_mean = np.mean(mlp_r2s)
        print(f"\n  {'Method':<20} {'R² (mean ± std)':>20}")
        print(f"  {'-' * 42}")
        print(f"  {'MLP':<20} {mlp_mean:>8.4f} +/- {np.std(mlp_r2s):.4f}")

        best_k, best_r2 = 1, -999
        for k in k_values:
            mean_r2 = np.mean(rw_r2s[k])
            marker = " <-- best" if mean_r2 == max(np.mean(rw_r2s[kk]) for kk in k_values) else ""
            print(f"  {'RW-MLP (k=' + str(k) + ')':<20} {mean_r2:>8.4f} +/- {np.std(rw_r2s[k]):.4f}{marker}")
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_k = k

        improvement = best_r2 - mlp_mean
        if improvement > 0:
            print(f"\n  RW-MLP improves by +{improvement:.4f} R² (k={best_k})")
        else:
            print(f"\n  MLP wins by {-improvement:.4f} R²")

        all_results[ds_name] = {
            'MLP': mlp_mean,
            'best_RW-MLP': best_r2,
            'best_k': best_k,
            'improved': improvement > 0
        }

    # ===========================================================================
    # Summary
    # ===========================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    wins = sum(1 for r in all_results.values() if r['improved'])
    total = len(all_results)

    print(f"\n  {'Dataset':<30} {'MLP':>8} {'RW-MLP':>8} {'k':>4} {'Winner':>8}")
    print(f"  {'-' * 62}")
    for ds_name, r in all_results.items():
        winner = "RW-MLP" if r['improved'] else "MLP"
        print(f"  {ds_name:<30} {r['MLP']:>8.4f} {r['best_RW-MLP']:>8.4f} {r['best_k']:>4} {winner:>8}")

    print(f"\n  RW-MLP wins on {wins}/{total} datasets ({wins/total*100:.0f}%)")

    # ===========================================================================
    # Visualization: W matrix deviation from identity
    # ===========================================================================
    print("\nGenerating visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Plot 1: R² comparison bar chart
    ax = axes[0, 0]
    ds_names = list(all_results.keys())
    ds_short = [n[:15] + '...' if len(n) > 15 else n for n in ds_names]
    mlp_vals = [all_results[n]['MLP'] for n in ds_names]
    rw_vals = [all_results[n]['best_RW-MLP'] for n in ds_names]

    x = np.arange(len(ds_names))
    width = 0.35
    ax.bar(x - width/2, mlp_vals, width, label='MLP', color='#ff7f0e', alpha=0.8)
    ax.bar(x + width/2, rw_vals, width, label='RW-MLP', color='#1f77b4', alpha=0.8)
    ax.set_ylabel('R²')
    ax.set_title('MLP vs RW-MLP')
    ax.set_xticks(x)
    ax.set_xticklabels(ds_short, rotation=45, ha='right', fontsize=7)
    ax.legend()

    # Plots 2-6: W matrix heatmaps (deviation from identity)
    plot_idx = 1
    for ds_name in ds_names:
        if plot_idx >= 6:
            break
        ax = axes[plot_idx // 3, plot_idx % 3]

        # Find the best k model for this dataset
        best_k = all_results[ds_name]['best_k']
        key = (ds_name, best_k)

        if key in rw_models:
            model = rw_models[key]
            W = model.rw.W.detach().cpu().numpy()
            deviation = W - np.eye(W.shape[0])

            # For large matrices, show top-left corner
            show_dim = min(W.shape[0], 15)
            im = ax.imshow(deviation[:show_dim, :show_dim], cmap='RdBu_r',
                          vmin=-np.abs(deviation).max(), vmax=np.abs(deviation).max(),
                          aspect='auto')
            plt.colorbar(im, ax=ax, shrink=0.8)
            title = ds_name[:20]
            ax.set_title(f'W-I ({title}, k={best_k})', fontsize=9)
            ax.set_xlabel('Feature')
            ax.set_ylabel('Feature')
        plot_idx += 1

    plt.suptitle('RW-Layer Demo: Learned Corrections (W - I)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = 'rw_layer_demo_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to: {output_path}")

    print("\n" + "=" * 70)
    print("  The heatmaps show W - I (deviation from identity).")
    print("  Blue = feature value decreased, Red = feature value increased.")
    print("  Off-diagonal elements show learned feature interactions.")
    print("=" * 70)


if __name__ == '__main__':
    main()
