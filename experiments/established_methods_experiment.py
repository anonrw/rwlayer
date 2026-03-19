"""
RW-Layer: Universal Neural Network Improvement for Tabular Data
================================================================
Full experiment on 39 datasets with 10 seeds.

Methods (MLP already in Experiment 1 - NOT included):
1. RTDL-ResNet   - Official Yandex ResNet (NeurIPS 2021)
2. RW-ResNet     - RW + RTDL-ResNet (end-to-end gradients)
3. NODE          - Neural Oblivious Decision Ensembles (Yandex, ICLR 2020)
4. RW-NODE       - RW + NODE (end-to-end gradients)
5. TabNet        - Google's TabNet (AAAI 2021)
6. RW-TabNet     - RW + TabNet

Install: pip install pytorch-tabnet rtdl
"""

import subprocess
import sys
import io

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet", "rtdl", "-q"])

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_openml
import warnings
import random
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

import rtdl
from pytorch_tabnet.tab_model import TabNetRegressor

print("✓ All libraries loaded")

# =============================================================================
# CONFIG - SAME AS EXPERIMENT 1
# =============================================================================
SEEDS = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]  # 10 seeds - SAME AS EXPERIMENT 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Seeds: {SEEDS}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# =============================================================================
# ALL 39 DATASETS - SAME AS EXPERIMENT 1
# =============================================================================
ALL_DATASETS = [
    # OpenML-CTR23 (26 datasets)
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
    (42727, 'MIP-2016-regression', None),
    (42728, 'nyc-taxi-green-dec-2016', 10000),
    (42729, 'delays_zurich_transport', 10000),
    (42730, 'yprop_4_1', None),
    (42731, 'analcatdata_supreme', 10000),
    # Additional AMLB datasets (13 datasets)
    (216, 'elevators', 10000),
    (189, 'kin8nm', None),
    (537, 'houses', None),
    (42165, 'california_housing', 10000),
    (42464, 'pol', None),
    (42724, 'OnlineNewsPopularity', 10000),
    (42705, 'Yolanda', 10000),
    (44135, 'bank8FM', None),
    (44136, 'pumadyn32nm', None),
    (44137, 'cpu_act', 10000),
    (44129, 'topo_2_1', 10000),
    (44056, 'superconduct', 10000),
    (44061, 'jannis', 10000),
]

# =============================================================================
# ADAPTIVE MODEL SIZING (only for TabNet to prevent overfitting)
# =============================================================================
def get_tabnet_config(n_samples, n_features):
    """
    Dynamically adjust TabNet size based on dataset characteristics.
    Small datasets → smaller TabNet to prevent overfitting.
    ResNet and NODE stay at fixed (full) size.
    """
    data_size = n_samples * n_features
    
    if n_samples < 500 or data_size < 5000:
        # Small dataset - minimal TabNet
        config = {
            'tabnet_n_d': 4,
            'tabnet_n_a': 4,
            'tabnet_n_steps': 2,
            'batch_size': min(64, max(16, n_samples // 4)),
        }
    elif n_samples < 2000 or data_size < 20000:
        # Medium dataset
        config = {
            'tabnet_n_d': 8,
            'tabnet_n_a': 8,
            'tabnet_n_steps': 3,
            'batch_size': 128,
        }
    else:
        # Large dataset - full TabNet
        config = {
            'tabnet_n_d': 16,
            'tabnet_n_a': 16,
            'tabnet_n_steps': 4,
            'batch_size': 256,
        }
    
    return config


# =============================================================================
# RW-LAYER
# =============================================================================
class RWLayer(nn.Module):
    """Random Walk Layer with identity initialization"""
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


class RWModel(nn.Module):
    """Wraps any PyTorch model with RW-Layer. Gradients flow through both!"""
    def __init__(self, rw_layer, model):
        super().__init__()
        self.rw = rw_layer
        self.model = model
    
    def forward(self, x):
        return self.model(self.rw(x))


# =============================================================================
# NODE (Neural Oblivious Decision Ensembles) - Yandex ICLR 2020
# =============================================================================
class ObliviousDecisionTree(nn.Module):
    """Single differentiable oblivious decision tree"""
    def __init__(self, input_dim, depth=6, output_dim=1, temperature=1.0):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.temperature = temperature
        
        self.feature_selection = nn.Linear(input_dim, depth, bias=False)
        self.thresholds = nn.Parameter(torch.zeros(depth))
        self.response = nn.Parameter(torch.randn(self.n_leaves, output_dim) * 0.01)
    
    def forward(self, x):
        features = self.feature_selection(x)
        decisions = torch.sigmoid((features - self.thresholds) / self.temperature)
        
        batch_size = x.size(0)
        leaf_probs = torch.ones(batch_size, self.n_leaves, device=x.device)
        
        for d in range(self.depth):
            leaf_indices = torch.arange(self.n_leaves, device=x.device)
            goes_right = ((leaf_indices >> (self.depth - 1 - d)) & 1).float()
            decision_d = decisions[:, d:d+1]
            goes_right = goes_right.unsqueeze(0)
            leaf_probs = leaf_probs * (goes_right * decision_d + (1 - goes_right) * (1 - decision_d))
        
        return torch.matmul(leaf_probs, self.response)


class NODE(nn.Module):
    """
    Neural Oblivious Decision Ensembles
    Reference: https://arxiv.org/abs/1909.06312 (Yandex, ICLR 2020)
    """
    def __init__(self, input_dim, n_trees=20, depth=5, tree_output_dim=1):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.trees = nn.ModuleList([
            ObliviousDecisionTree(input_dim, depth, tree_output_dim)
            for _ in range(n_trees)
        ])
        self.output = nn.Linear(n_trees * tree_output_dim, 1)
    
    def forward(self, x):
        x = self.input_bn(x)
        tree_outputs = [tree(x) for tree in self.trees]
        combined = torch.cat(tree_outputs, dim=-1)
        return self.output(combined).squeeze(-1)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================
def train_pytorch_model(model, X_train, y_train, X_val, y_val, 
                        epochs=100, lr=1e-3, patience=15, batch_size=256):
    """Train PyTorch model with early stopping"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            if pred.dim() > 1:
                pred = pred.squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            if val_pred.dim() > 1:
                val_pred = val_pred.squeeze(-1)
            val_loss = criterion(val_pred, y_val_t).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_pytorch(model, X_test, y_test, scaler_y):
    """Evaluate PyTorch model"""
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        pred_scaled = model(X_test_t)
        if pred_scaled.dim() > 1:
            pred_scaled = pred_scaled.squeeze(-1)
        pred_scaled = pred_scaled.cpu().numpy()
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    y_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    return r2_score(y_orig, pred), np.sqrt(mean_squared_error(y_orig, pred))


# =============================================================================
# TABNET (with RW preprocessing via black-box optimization)
# =============================================================================
class RWPreprocessor:
    """Learns RW transformation via validation loss (for sklearn-style models)"""
    def __init__(self, n_features, k=3, n_iter=15, lr=0.02):
        self.n_features = n_features
        self.k = k
        self.n_iter = n_iter
        self.lr = lr
        self.W = np.eye(n_features)
        self.bias = np.zeros(n_features)
    
    def transform(self, X):
        W_k = self.W.copy()
        for _ in range(self.k - 1):
            W_k = W_k @ self.W
        return X @ W_k.T + self.bias
    
    def fit(self, X_train, y_train, X_val, y_val, model_fn, seed=42):
        """Learn RW params by minimizing validation MSE"""
        best_loss = float('inf')
        best_W, best_bias = self.W.copy(), self.bias.copy()
        
        for i in range(self.n_iter):
            X_tr = self.transform(X_train)
            X_va = self.transform(X_val)
            
            model = model_fn(seed + i)
            # Suppress output
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                model.fit(X_tr, y_train.reshape(-1, 1),
                         eval_set=[(X_va, y_val.reshape(-1, 1))],
                         max_epochs=50, patience=10, batch_size=256,
                         virtual_batch_size=128)
            finally:
                sys.stdout = old_stdout
            
            pred = model.predict(X_va).ravel()
            loss = mean_squared_error(y_val, pred)
            
            if loss < best_loss:
                best_loss = loss
                best_W, best_bias = self.W.copy(), self.bias.copy()
            
            if i < self.n_iter - 1:
                self.W = best_W + np.random.randn(*self.W.shape) * self.lr * (1 - i/self.n_iter)
                self.bias = best_bias + np.random.randn(*self.bias.shape) * self.lr * (1 - i/self.n_iter)
        
        self.W, self.bias = best_W, best_bias
        return self


def run_tabnet(X_train, y_train, X_val, y_val, X_test, y_test, scaler_y, seed, use_rw=False, k=3, cfg=None):
    """Run TabNet with optional RW preprocessing and adaptive config"""
    # Default config if not provided
    if cfg is None:
        cfg = {
            'tabnet_n_d': 8,
            'tabnet_n_a': 8,
            'tabnet_n_steps': 3,
            'batch_size': 256,
        }
    
    def make_tabnet(s):
        return TabNetRegressor(
            n_d=cfg['tabnet_n_d'], 
            n_a=cfg['tabnet_n_a'], 
            n_steps=cfg['tabnet_n_steps'], 
            gamma=1.3,
            seed=s, verbose=0, device_name=str(device)
        )
    
    if use_rw:
        rw = RWPreprocessor(X_train.shape[1], k=k, n_iter=15)
        rw.fit(X_train, y_train, X_val, y_val, make_tabnet, seed)
        X_train = rw.transform(X_train)
        X_val = rw.transform(X_val)
        X_test = rw.transform(X_test)
    
    model = make_tabnet(seed)
    # Suppress output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        model.fit(X_train, y_train.reshape(-1, 1),
                 eval_set=[(X_val, y_val.reshape(-1, 1))],
                 max_epochs=100, patience=15, 
                 batch_size=cfg['batch_size'],
                 virtual_batch_size=min(128, cfg['batch_size']))
    finally:
        sys.stdout = old_stdout
    
    pred_scaled = model.predict(X_test).ravel()
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    y_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    return r2_score(y_orig, pred), np.sqrt(mean_squared_error(y_orig, pred))


# =============================================================================
# DATASET LOADING
# =============================================================================
def load_dataset(dataset_id, name, max_samples=None):
    """Load dataset with optional max_samples"""
    if max_samples is None:
        max_samples = 10000  # Default cap
    try:
        data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
        X = data.data.values if hasattr(data.data, 'values') else np.array(data.data)
        y = data.target.values if hasattr(data.target, 'values') else np.array(data.target)
        
        X_clean = np.zeros((X.shape[0], X.shape[1]))
        for i in range(X.shape[1]):
            try:
                X_clean[:, i] = X[:, i].astype(float)
            except:
                X_clean[:, i] = LabelEncoder().fit_transform(X[:, i].astype(str))
        
        X = np.nan_to_num(X_clean)
        y = y.astype(float)
        y = np.nan_to_num(y, nan=np.nanmean(y))
        
        if max_samples and len(X) > max_samples:
            np.random.seed(SEEDS[0])  # Use first seed for consistent subsampling
            idx = np.random.choice(len(X), max_samples, replace=False)
            X, y = X[idx], y[idx]
        
        return X, y
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None, None


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================
def run_experiment():
    print("=" * 70)
    print("RW-LAYER: NEURAL NETWORK IMPROVEMENT EXPERIMENT")
    print("=" * 70)
    print("\nMethods (MLP already in Experiment 1 - NOT included):")
    print("  1. RTDL-ResNet   - Yandex NeurIPS 2021 (Residual Network)")
    print("  2. RW-ResNet     - RW + RTDL-ResNet (end-to-end gradients)")
    print("  3. NODE          - Yandex ICLR 2020 (Differentiable Trees)")
    print("  4. RW-NODE       - RW + NODE (end-to-end gradients)")
    print("  5. TabNet        - Google AAAI 2021 (Sparse Attention)")
    print("  6. RW-TabNet     - RW + TabNet (black-box optimization)")
    print(f"\nDatasets: {len(ALL_DATASETS)} (Same as Experiment 1)")
    print(f"Seeds: {SEEDS} (Same as Experiment 1)")
    print(f"Train/Test Split: 90%/10% (Same as Experiment 1)")
    print("=" * 70)
    
    all_results = []
    start_time = time.time()
    
    for ds_idx, (dataset_id, name, max_samples) in enumerate(ALL_DATASETS):
        print(f"\n{'='*60}")
        print(f"[{ds_idx+1}/{len(ALL_DATASETS)}] Dataset: {name}")
        print(f"{'='*60}")
        
        X, y = load_dataset(dataset_id, name, max_samples)
        if X is None:
            continue
        
        n_features = X.shape[1]
        n_samples = X.shape[0]
        print(f"Shape: {X.shape}")
        
        # Get adaptive TabNet config (ResNet/NODE stay fixed)
        tabnet_cfg = get_tabnet_config(n_samples, n_features)
        print(f"TabNet config: n_d={tabnet_cfg['tabnet_n_d']}, batch={tabnet_cfg['batch_size']}")
        
        results = {m: {'r2': [], 'rmse': []} for m in 
                   ['RTDL-ResNet', 'RW-ResNet', 'NODE', 'RW-NODE', 'TabNet', 'RW-TabNet']}
        
        for seed_idx, seed in enumerate(SEEDS):
            print(f"  Seed {seed_idx+1}/{len(SEEDS)} ({seed})...", end=" ", flush=True)
            set_seed(seed)
            
            # Split - SAME AS EXPERIMENT 1: 90% train, 10% test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.10, random_state=seed
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.11, random_state=seed
            )
            
            # Scale
            scaler_X, scaler_y = StandardScaler(), StandardScaler()
            X_train_s = scaler_X.fit_transform(X_train)
            X_val_s = scaler_X.transform(X_val)
            X_test_s = scaler_X.transform(X_test)
            y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
            y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
            
            # 1. RTDL-ResNet (FIXED SIZE)
            set_seed(seed)
            model = rtdl.ResNet.make_baseline(
                d_in=n_features, 
                n_blocks=2, 
                d_main=256, 
                d_hidden=512,
                dropout_first=0.1, 
                dropout_second=0.0, 
                d_out=1
            )
            model = train_pytorch_model(model, X_train_s, y_train_s, X_val_s, y_val_s, 
                                        batch_size=256)
            r2, rmse = evaluate_pytorch(model, X_test_s, y_test_s, scaler_y)
            results['RTDL-ResNet']['r2'].append(r2)
            results['RTDL-ResNet']['rmse'].append(rmse)
            
            # 2. RW + RTDL-ResNet (FIXED SIZE)
            set_seed(seed)
            rw = RWLayer(n_features, k=3)
            base = rtdl.ResNet.make_baseline(
                d_in=n_features, 
                n_blocks=2, 
                d_main=256, 
                d_hidden=512,
                dropout_first=0.1, 
                dropout_second=0.0, 
                d_out=1
            )
            model = RWModel(rw, base)
            model = train_pytorch_model(model, X_train_s, y_train_s, X_val_s, y_val_s,
                                        batch_size=256)
            r2, rmse = evaluate_pytorch(model, X_test_s, y_test_s, scaler_y)
            results['RW-ResNet']['r2'].append(r2)
            results['RW-ResNet']['rmse'].append(rmse)
            
            # 3. NODE (FIXED SIZE)
            set_seed(seed)
            model = NODE(n_features, n_trees=20, depth=5)
            model = train_pytorch_model(model, X_train_s, y_train_s, X_val_s, y_val_s,
                                        batch_size=256)
            r2, rmse = evaluate_pytorch(model, X_test_s, y_test_s, scaler_y)
            results['NODE']['r2'].append(r2)
            results['NODE']['rmse'].append(rmse)
            
            # 4. RW + NODE (FIXED SIZE)
            set_seed(seed)
            rw = RWLayer(n_features, k=3)
            base = NODE(n_features, n_trees=20, depth=5)
            model = RWModel(rw, base)
            model = train_pytorch_model(model, X_train_s, y_train_s, X_val_s, y_val_s,
                                        batch_size=256)
            r2, rmse = evaluate_pytorch(model, X_test_s, y_test_s, scaler_y)
            results['RW-NODE']['r2'].append(r2)
            results['RW-NODE']['rmse'].append(rmse)
            
            # 5. TabNet (ADAPTIVE SIZE)
            set_seed(seed)
            r2, rmse = run_tabnet(X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s, 
                                   scaler_y, seed, use_rw=False, cfg=tabnet_cfg)
            results['TabNet']['r2'].append(r2)
            results['TabNet']['rmse'].append(rmse)
            
            # 6. RW + TabNet (ADAPTIVE SIZE)
            set_seed(seed)
            r2, rmse = run_tabnet(X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s,
                                   scaler_y, seed, use_rw=True, k=3, cfg=tabnet_cfg)
            results['RW-TabNet']['r2'].append(r2)
            results['RW-TabNet']['rmse'].append(rmse)
            
            print("Done")
        
        # Print results for this dataset
        print(f"\nResults for {name} (mean ± std over {len(SEEDS)} seeds):")
        print(f"{'Method':<12} {'R²':>14} {'RMSE':>14}")
        print("-" * 42)
        
        for method, data in results.items():
            r2_m, r2_s = np.mean(data['r2']), np.std(data['r2'])
            rmse_m, rmse_s = np.mean(data['rmse']), np.std(data['rmse'])
            print(f"{method:<12} {r2_m:>6.4f}±{r2_s:.4f} {rmse_m:>6.2f}±{rmse_s:.2f}")
            
            all_results.append({
                'dataset': name,
                'method': method,
                'r2_mean': r2_m,
                'r2_std': r2_s,
                'rmse_mean': rmse_m,
                'rmse_std': rmse_s,
            })
        
        # RW improvements
        print(f"\n  RW Improvements (R²):")
        for base, rw in [('RTDL-ResNet', 'RW-ResNet'), ('NODE', 'RW-NODE'), ('TabNet', 'RW-TabNet')]:
            b_r2 = np.mean(results[base]['r2'])
            r_r2 = np.mean(results[rw]['r2'])
            imp = (r_r2 - b_r2) / abs(b_r2) * 100 if b_r2 != 0 else 0
            marker = "✓" if r_r2 > b_r2 else "✗"
            print(f"    {rw} vs {base}: {imp:+.1f}% {marker}")
        
        # Save intermediate results
        df = pd.DataFrame(all_results)
        df.to_csv('neural_rw_results.csv', index=False)
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total time: {elapsed/3600:.1f} hours")
    
    df = pd.DataFrame(all_results)
    
    for base, rw in [('RTDL-ResNet', 'RW-ResNet'), ('NODE', 'RW-NODE'), ('TabNet', 'RW-TabNet')]:
        wins = 0
        total = 0
        improvements = []
        
        for dataset_id, ds_name, _ in ALL_DATASETS:
            base_rows = df[(df['dataset'] == ds_name) & (df['method'] == base)]
            rw_rows = df[(df['dataset'] == ds_name) & (df['method'] == rw)]
            
            if len(base_rows) > 0 and len(rw_rows) > 0:
                base_r2 = base_rows['r2_mean'].values[0]
                rw_r2 = rw_rows['r2_mean'].values[0]
                
                if not np.isnan(base_r2) and not np.isnan(rw_r2):
                    if rw_r2 > base_r2:
                        wins += 1
                    total += 1
                    imp = (rw_r2 - base_r2) / abs(base_r2) * 100 if base_r2 != 0 else 0
                    improvements.append(imp)
        
        if total > 0:
            print(f"\n  {rw} vs {base}:")
            print(f"    Win rate: {wins}/{total} ({wins/total*100:.0f}%)")
            print(f"    Mean improvement: {np.mean(improvements):+.1f}%")
            print(f"    Median improvement: {np.median(improvements):+.1f}%")
    
    # Overall
    print("\n" + "-" * 40)
    total_wins = 0
    total_comparisons = 0
    for base, rw in [('RTDL-ResNet', 'RW-ResNet'), ('NODE', 'RW-NODE'), ('TabNet', 'RW-TabNet')]:
        for dataset_id, ds_name, _ in ALL_DATASETS:
            base_rows = df[(df['dataset'] == ds_name) & (df['method'] == base)]
            rw_rows = df[(df['dataset'] == ds_name) & (df['method'] == rw)]
            if len(base_rows) > 0 and len(rw_rows) > 0:
                if rw_rows['r2_mean'].values[0] > base_rows['r2_mean'].values[0]:
                    total_wins += 1
                total_comparisons += 1
    
    print(f"\n  OVERALL: RW improves {total_wins}/{total_comparisons} ({total_wins/total_comparisons*100:.0f}%) comparisons")
    
    print(f"\nResults saved to: neural_rw_results.csv")
    
    return df


if __name__ == "__main__":
    df = run_experiment()