# -*- coding: utf-8 -*-
"""
RW-Layer Ablation Study
========================

This script tests what makes RW-Layer work by isolating:
1. Identity initialization
2. No activation function
3. Matrix power W^k

Ablation Models:
----------------
| Model              | Init     | Activation | Formula      | What it tests           |
|--------------------+----------+------------+--------------+-------------------------|
| MLP                | -        | -          | x → MLP      | Baseline (no extra layer)|
| Linear-MLP         | Random   | ReLU       | ReLU(xW+b)   | Standard approach       |
| Identity-MLP       | Identity | ReLU       | ReLU(xI+b)   | Identity init effect    |
| NoAct-MLP          | Random   | None       | xW+b         | No activation effect    |
| IdentityNoAct-MLP  | Identity | None       | xI+b (k=1)   | Both, but no W^k        |
| RW-MLP             | Identity | None       | xW^k+b       | Full RW-Layer           |

Usage:
    python ablation_study.py
    python ablation_study.py --resume
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from sklearn.datasets import fetch_openml
import time
import warnings
import os
import random
from typing import Optional
import argparse

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - SAME AS ORIGINAL EXPERIMENTS
# =============================================================================
N_REPETITIONS = 10
TEST_SIZE = 0.10
VAL_SIZE = 0.11
EPOCHS = 150
PATIENCE = 20
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
BASE_SEED = 42
RW_K = 3  # Same k as in original RW-MLP experiments

SEEDS = [BASE_SEED + i * 100 for i in range(N_REPETITIONS)]

OUTPUT_FILE = 'ablation_results.csv'

# =============================================================================
# DATASETS - Same 30 that worked (excluding mismatched)
# =============================================================================
DATASETS = [
    (505, 'tecator', None),
    (507, 'space_ga', None),
    (531, 'boston', None),
    (541, 'socmob', None),
    (546, 'sensory', None),
    (547, 'no2', None),
    (550, 'quake', None),
    (574, 'house_16H', 10000),
    (1028, 'SWD', None),
    (1030, 'ERA', None),
    (1096, 'FacultySalaries', None),
    (4545, 'wine_quality', 10000),
    (41021, 'Moneyball', None),
    (41540, 'black_friday', 10000),
    (41702, 'wine-quality-white', None),
    (42225, 'diamonds', 10000),
    (42563, 'Mercedes_Benz_Greener_Manufacturing', None),
    (42570, 'Brazilian_houses', 10000),
    (42571, 'Bike_Sharing_Demand', 10000),
    (42688, 'abalone', 10000),
    (42726, 'us_crime', None),
    (42728, 'nyc-taxi-green-dec-2016', 10000),
    (42730, 'yprop_4_1', None),
    (42731, 'analcatdata_supreme', 10000),
    (216, 'elevators', 10000),
    (189, 'kin8nm', None),
    (537, 'houses', None),
    (42165, 'california_housing', 10000),
    (42705, 'Yolanda', 10000),
    (44061, 'jannis', 10000),
]

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# DATASET LOADING
# =============================================================================
def load_dataset(dataset_id: int, name: str, max_samples: Optional[int] = None):
    try:
        data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
        X = data.data
        y = data.target
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                X[col] = pd.Categorical(X[col]).codes
        
        X = X.values.astype(np.float32)
        y = y.values.astype(np.float32) if hasattr(y, 'values') else np.array(y, dtype=np.float32)
        
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]
        
        if max_samples and len(X) > max_samples:
            np.random.seed(42)
            idx = np.random.choice(len(X), max_samples, replace=False)
            X, y = X[idx], y[idx]
        
        return X, y
    except Exception as e:
        print(f"    Error loading: {e}")
        return None, None

# =============================================================================
# BASE MLP
# =============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# =============================================================================
# ABLATION LAYERS
# =============================================================================

class LinearLayer(nn.Module):
    """Standard Linear: Random init + ReLU"""
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, n_features)  # Random init (default)
    
    def forward(self, x):
        return F.relu(self.linear(x))  # With ReLU


class IdentityInitLayer(nn.Module):
    """Identity init + ReLU (no W^k)"""
    def __init__(self, n_features):
        super().__init__()
        self.W = nn.Parameter(torch.eye(n_features))  # Identity init
        self.bias = nn.Parameter(torch.zeros(n_features))
    
    def forward(self, x):
        return F.relu(x @ self.W.T + self.bias)  # With ReLU


class NoActivationLayer(nn.Module):
    """Random init + No activation"""
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, n_features)  # Random init
    
    def forward(self, x):
        return self.linear(x)  # No activation


class IdentityNoActLayer(nn.Module):
    """Identity init + No activation (k=1 only, no matrix power)"""
    def __init__(self, n_features):
        super().__init__()
        self.W = nn.Parameter(torch.eye(n_features))  # Identity init
        self.bias = nn.Parameter(torch.zeros(n_features))
    
    def forward(self, x):
        return x @ self.W.T + self.bias  # No activation, no W^k


class RWLayer(nn.Module):
    """Full RW-Layer: Identity init + No activation + W^k"""
    def __init__(self, n_features, k=3):
        super().__init__()
        self.k = k
        self.W = nn.Parameter(torch.eye(n_features))  # Identity init
        self.bias = nn.Parameter(torch.zeros(n_features))
    
    def forward(self, x):
        W_k = self.W
        for _ in range(self.k - 1):
            W_k = W_k @ self.W  # Matrix power
        return x @ W_k.T + self.bias  # No activation

# =============================================================================
# ABLATION MODELS
# =============================================================================

class LinearMLP(nn.Module):
    """Ablation 1: Random init + ReLU"""
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.layer = LinearLayer(input_dim)
        self.mlp = MLP(input_dim, hidden_dim, n_layers, dropout)
    
    def forward(self, x):
        return self.mlp(self.layer(x))


class IdentityMLP(nn.Module):
    """Ablation 2: Identity init + ReLU"""
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.layer = IdentityInitLayer(input_dim)
        self.mlp = MLP(input_dim, hidden_dim, n_layers, dropout)
    
    def forward(self, x):
        return self.mlp(self.layer(x))


class NoActMLP(nn.Module):
    """Ablation 3: Random init + No activation"""
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.layer = NoActivationLayer(input_dim)
        self.mlp = MLP(input_dim, hidden_dim, n_layers, dropout)
    
    def forward(self, x):
        return self.mlp(self.layer(x))


class IdentityNoActMLP(nn.Module):
    """Ablation 4: Identity init + No activation (k=1)"""
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.layer = IdentityNoActLayer(input_dim)
        self.mlp = MLP(input_dim, hidden_dim, n_layers, dropout)
    
    def forward(self, x):
        return self.mlp(self.layer(x))


class RWMLP(nn.Module):
    """Full RW-MLP: Identity init + No activation + W^k"""
    def __init__(self, input_dim, hidden_dim=128, n_layers=2, dropout=0.1, k=3):
        super().__init__()
        self.layer = RWLayer(input_dim, k=k)
        self.mlp = MLP(input_dim, hidden_dim, n_layers, dropout)
    
    def forward(self, x):
        return self.mlp(self.layer(x))

# =============================================================================
# TRAINING & EVALUATION
# =============================================================================
def train_model(model, X_train, y_train, X_val, y_val, seed):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)
    
    return model


def evaluate_model(model, X_test, y_test, scaler_y):
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        pred_tensor = model(X_test_t).cpu()
    
    pred_scaled = np.array(pred_tensor.tolist())
    
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    y_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    
    return {
        'r2': r2_score(y_original, pred),
        'rmse': np.sqrt(mean_squared_error(y_original, pred)),
        'mae': mean_absolute_error(y_original, pred),
        'spearman': spearmanr(y_original, pred)[0]
    }


def get_completed():
    """Get already completed (dataset, method) pairs"""
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        return set(zip(df['Dataset'], df['Method']))
    return set()


def save_result(result):
    """Append a single result to CSV"""
    df = pd.DataFrame([result])
    if os.path.exists(OUTPUT_FILE):
        df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(OUTPUT_FILE, index=False)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================
def run_ablation(resume=False):
    print("=" * 70)
    print("RW-LAYER ABLATION STUDY")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 70)
    
    # Ablation methods
    ablation_methods = [
        ('MLP', lambda d: MLP(d)),
        ('Linear-MLP', lambda d: LinearMLP(d)),
        ('Identity-MLP', lambda d: IdentityMLP(d)),
        ('NoAct-MLP', lambda d: NoActMLP(d)),
        ('IdentityNoAct-MLP', lambda d: IdentityNoActMLP(d)),
        ('RW-MLP', lambda d: RWMLP(d, k=RW_K)),
    ]
    
    print("\nAblation Methods:")
    print("-" * 70)
    print(f"{'Method':<20} {'Init':<12} {'Activation':<12} {'W^k':<8}")
    print("-" * 70)
    print(f"{'MLP':<20} {'-':<12} {'-':<12} {'-':<8}")
    print(f"{'Linear-MLP':<20} {'Random':<12} {'ReLU':<12} {'No':<8}")
    print(f"{'Identity-MLP':<20} {'Identity':<12} {'ReLU':<12} {'No':<8}")
    print(f"{'NoAct-MLP':<20} {'Random':<12} {'None':<12} {'No':<8}")
    print(f"{'IdentityNoAct-MLP':<20} {'Identity':<12} {'None':<12} {'No (k=1)':<8}")
    print(f"{'RW-MLP':<20} {'Identity':<12} {'None':<12} {'Yes (k={RW_K})':<8}")
    print("=" * 70)
    
    # Check completed
    completed = set()
    if resume:
        completed = get_completed()
        print(f"\nResuming: {len(completed)} (dataset, method) pairs already done")
    else:
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
    
    start_time = time.time()
    total_tasks = len(DATASETS) * len(ablation_methods)
    completed_tasks = len(completed)
    
    for ds_idx, (dataset_id, name, max_samples) in enumerate(DATASETS):
        print(f"\n{'='*70}")
        print(f"[{ds_idx+1}/{len(DATASETS)}] Dataset: {name}")
        print("=" * 70)
        
        # Load dataset once
        X, y = load_dataset(dataset_id, name, max_samples)
        if X is None or len(X) == 0:
            print(f"  FAILED to load - skipping all methods")
            continue
        
        n_features = X.shape[1]
        print(f"  Shape: {X.shape}, Features: {n_features}")
        
        for method_name, model_fn in ablation_methods:
            # Check if already done
            if (name, method_name) in completed:
                print(f"  {method_name}: already done - skipping")
                continue
            
            print(f"\n  {method_name}:")
            results = []
            
            for seed_idx, seed in enumerate(SEEDS):
                set_seed(seed)
                
                # Split
                X_train_full, X_test, y_train_full, y_test = train_test_split(
                    X, y, test_size=TEST_SIZE, random_state=seed
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_full, y_train_full, test_size=VAL_SIZE, random_state=seed
                )
                
                # Scale
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                
                X_train_s = scaler_X.fit_transform(X_train)
                X_val_s = scaler_X.transform(X_val)
                X_test_s = scaler_X.transform(X_test)
                
                y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
                y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
                y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
                
                # Train
                set_seed(seed)
                model = model_fn(n_features)
                model = train_model(model, X_train_s, y_train_s, X_val_s, y_val_s, seed)
                
                # Evaluate
                metrics = evaluate_model(model, X_test_s, y_test_s, scaler_y)
                results.append(metrics)
                
                print(f"    Seed {seed_idx+1}/10: R²={metrics['r2']:.4f}")
            
            # Aggregate
            r2s = [r['r2'] for r in results]
            rmses = [r['rmse'] for r in results]
            maes = [r['mae'] for r in results]
            spearmans = [r['spearman'] for r in results]
            
            result = {
                'Dataset': name,
                'Method': method_name,
                'R²': np.mean(r2s),
                'R² std': np.std(r2s),
                'RMSE': np.mean(rmses),
                'RMSE std': np.std(rmses),
                'MAE': np.mean(maes),
                'MAE std': np.std(maes),
                'Spearman': np.mean(spearmans),
                'Spearman std': np.std(spearmans),
            }
            
            # Save immediately
            save_result(result)
            completed_tasks += 1
            
            print(f"  ✓ SAVED: R²={result['R²']:.4f}±{result['R² std']:.4f}")
            
            # Progress
            elapsed = time.time() - start_time
            remaining = (elapsed / completed_tasks) * (total_tasks - completed_tasks) if completed_tasks > 0 else 0
            print(f"  Progress: {completed_tasks}/{total_tasks} ({completed_tasks/total_tasks*100:.1f}%), "
                  f"ETA: {remaining/60:.1f} min")
    
    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        print(f"\nResults saved to: {OUTPUT_FILE}")
        print(f"Total rows: {len(df)}")
        
        # Summary per method
        print("\n" + "-" * 70)
        print("SUMMARY BY METHOD")
        print("-" * 70)
        
        summary = df.groupby('Method')['R²'].agg(['mean', 'std', 'count'])
        summary = summary.sort_values('mean', ascending=False)
        
        print(f"\n{'Method':<20} {'Mean R²':>10} {'Std':>10} {'Count':>8}")
        print("-" * 55)
        for method, row in summary.iterrows():
            print(f"{method:<20} {row['mean']:>10.4f} {row['std']:>10.4f} {row['count']:>8.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from existing CSV')
    args = parser.parse_args()
    
    run_ablation(resume=args.resume)
