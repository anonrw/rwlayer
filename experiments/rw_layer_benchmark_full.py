"""
RW-Layer Benchmark: Full Replication of TabPFN + AMLB Methodology
==================================================================

Replicating the evaluation protocols from:
1. "Accurate predictions on small data with a tabular foundation model" 
   (TabPFN, Nature 2025) - OpenML-CTR23 benchmark
2. "AMLB: an AutoML Benchmark" (JMLR 2024) - OpenML Suite 269

Combined Benchmark Features:
- OpenML-CTR23: 26 regression datasets (TabPFN paper)
- AMLB Suite 269: 33 regression datasets (AutoML Benchmark)
- Combined: ~47 unique datasets covering both benchmarks

Protocol:
- 90% train / 10% test split (paper's exact protocol)
- 10 repetitions with different random seeds
- Multiple metrics: R², RMSE, MAE, Spearman correlation
- Normalized scores (0 = worst, 1 = best per dataset)
- Store ALL data: train/val losses, predictions, metrics per repetition

Methods Compared:
- Tree-based: XGBoost, CatBoost, LightGBM, RandomForest
- Neural: MLP, RW-MLP, FT-Transformer, RW-FT-Transformer

Usage:
  python rw_layer_benchmark_full.py --full --benchmark combined  # Most comprehensive
  python rw_layer_benchmark_full.py --full --benchmark ctr23     # TabPFN only
  python rw_layer_benchmark_full.py --full --benchmark amlb      # AMLB only
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_openml
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import time
import warnings
import os
import random
import json
import pickle
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class BenchmarkConfig:
    """Configuration matching TabPFN paper protocol"""
    # Evaluation protocol (from paper)
    n_repetitions: int = 10          # Paper: 10 repetitions
    test_size: float = 0.10          # Paper: 90% train, 10% test
    val_size: float = 0.11           # ~10% of remaining 90% for validation
    
    # Hyperparameter search
    k_values: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    
    # Neural network training
    epochs: int = 150
    patience: int = 20
    batch_size: int = 64
    lr_mlp: float = 1e-3
    lr_transformer: float = 1e-4
    
    # Seeds for 10 repetitions
    base_seed: int = 42
    
    # GPU acceleration
    use_gpu: bool = True             # Enable GPU for tree methods (XGBoost, CatBoost, LightGBM)
    
    # Output
    output_dir: str = "benchmark_results"
    save_predictions: bool = True
    save_losses: bool = True
    
    def get_seeds(self) -> List[int]:
        """Generate seeds for repetitions"""
        return [self.base_seed + i * 100 for i in range(self.n_repetitions)]


# =============================================================================
# DATA STRUCTURES FOR STORING RESULTS
# =============================================================================
@dataclass
class TrainingHistory:
    """Store training history for a single run"""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    total_epochs: int = 0
    
@dataclass
class RepetitionResult:
    """Results for a single repetition"""
    seed: int
    rep_idx: int
    
    # Metrics (all on original scale)
    r2: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    spearman: float = 0.0
    
    # Timing
    train_time: float = 0.0
    
    # Optional: best hyperparameters
    best_k: Optional[int] = None
    
    # Training history (for neural methods)
    history: Optional[TrainingHistory] = None
    
    # Predictions (optional, can be large)
    predictions: Optional[np.ndarray] = None
    ground_truth: Optional[np.ndarray] = None

@dataclass  
class MethodResults:
    """Results for a method across all repetitions"""
    method_name: str
    repetitions: List[RepetitionResult] = field(default_factory=list)
    
    # Aggregated metrics (computed after all reps)
    mean_r2: float = 0.0
    std_r2: float = 0.0
    mean_rmse: float = 0.0
    std_rmse: float = 0.0
    mean_mae: float = 0.0
    std_mae: float = 0.0
    mean_spearman: float = 0.0
    std_spearman: float = 0.0
    mean_time: float = 0.0
    
    def compute_aggregates(self):
        """Compute mean and std across repetitions"""
        if not self.repetitions:
            return
        r2s = [r.r2 for r in self.repetitions]
        rmses = [r.rmse for r in self.repetitions]
        maes = [r.mae for r in self.repetitions]
        spearmans = [r.spearman for r in self.repetitions]
        times = [r.train_time for r in self.repetitions]
        
        self.mean_r2, self.std_r2 = np.mean(r2s), np.std(r2s)
        self.mean_rmse, self.std_rmse = np.mean(rmses), np.std(rmses)
        self.mean_mae, self.std_mae = np.mean(maes), np.std(maes)
        self.mean_spearman, self.std_spearman = np.mean(spearmans), np.std(spearmans)
        self.mean_time = np.mean(times)

@dataclass
class DatasetResults:
    """Results for a single dataset"""
    dataset_name: str
    dataset_id: int
    n_samples: int
    n_features: int
    
    # Results per method
    methods: Dict[str, MethodResults] = field(default_factory=dict)
    
    # Normalized scores (computed after all methods run)
    normalized_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)

@dataclass
class BenchmarkResults:
    """Complete benchmark results"""
    config: BenchmarkConfig
    datasets: Dict[str, DatasetResults] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def save(self, filepath: str):
        """Save results to file"""
        # Convert to serializable format
        data = {
            'timestamp': self.timestamp,
            'config': asdict(self.config),
            'datasets': {}
        }
        for ds_name, ds_result in self.datasets.items():
            ds_data = {
                'dataset_name': ds_result.dataset_name,
                'dataset_id': ds_result.dataset_id,
                'n_samples': ds_result.n_samples,
                'n_features': ds_result.n_features,
                'methods': {},
                'normalized_scores': ds_result.normalized_scores
            }
            for method_name, method_result in ds_result.methods.items():
                method_data = {
                    'method_name': method_result.method_name,
                    'mean_r2': method_result.mean_r2,
                    'std_r2': method_result.std_r2,
                    'mean_rmse': method_result.mean_rmse,
                    'std_rmse': method_result.std_rmse,
                    'mean_mae': method_result.mean_mae,
                    'std_mae': method_result.std_mae,
                    'mean_spearman': method_result.mean_spearman,
                    'std_spearman': method_result.std_spearman,
                    'mean_time': method_result.mean_time,
                    'repetitions': []
                }
                for rep in method_result.repetitions:
                    rep_data = {
                        'seed': rep.seed,
                        'rep_idx': rep.rep_idx,
                        'r2': rep.r2,
                        'rmse': rep.rmse,
                        'mae': rep.mae,
                        'spearman': rep.spearman,
                        'train_time': rep.train_time,
                        'best_k': rep.best_k
                    }
                    if rep.history:
                        rep_data['history'] = {
                            'train_losses': rep.history.train_losses,
                            'val_losses': rep.history.val_losses,
                            'best_epoch': rep.history.best_epoch,
                            'best_val_loss': rep.history.best_val_loss,
                            'total_epochs': rep.history.total_epochs
                        }
                    method_data['repetitions'].append(rep_data)
                ds_data['methods'][method_name] = method_data
            data['datasets'][ds_name] = ds_data
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed: int):
    """Set all random seeds for full reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For full determinism (may slow down slightly)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass  # Older PyTorch versions


def get_generator(seed: int) -> torch.Generator:
    """Get a seeded generator for DataLoader"""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int):
    """Initialize worker with deterministic seed"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# METRICS COMPUTATION
# =============================================================================
def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all metrics on original scale"""
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'spearman': spearmanr(y_true, y_pred)[0]
    }


# =============================================================================
# NEURAL NETWORK MODULES
# =============================================================================
class RWLayer(nn.Module):
    """Random Walk Layer with identity initialization"""
    def __init__(self, n_features: int, k: int = 1):
        super().__init__()
        self.k = k
        self.W = nn.Parameter(torch.eye(n_features))
        self.bias = nn.Parameter(torch.zeros(n_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_k = self.W
        for _ in range(self.k - 1):
            W_k = W_k @ self.W
        return x @ W_k.T + self.bias


class MLP(nn.Module):
    """Standard MLP for regression"""
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RWMLP(nn.Module):
    """MLP with RW-Layer preprocessing"""
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2, 
                 dropout: float = 0.1, k: int = 1):
        super().__init__()
        self.rw = RWLayer(input_dim, k=k)
        self.mlp = MLP(input_dim, hidden_dim, n_layers, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rw(x)
        return self.mlp(x)


class FTTransformer(nn.Module):
    """FT-Transformer for tabular data"""
    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4, 
                 n_layers: int = 2, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        embeddings = [self.feature_embeddings[i](x[:, i:i+1]) for i in range(self.n_features)]
        x = torch.stack(embeddings, dim=1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.head(x[:, 0]).squeeze(-1)


class RWFTTransformer(nn.Module):
    """FT-Transformer with RW-Layer preprocessing"""
    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 128, dropout: float = 0.1, k: int = 1):
        super().__init__()
        self.rw = RWLayer(n_features, k=k)
        self.ft = FTTransformer(n_features, d_model, n_heads, n_layers, d_ff, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rw(x)
        return self.ft(x)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================
def train_neural_model(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    config: BenchmarkConfig,
    lr: float,
    seed: int
) -> Tuple[nn.Module, TrainingHistory]:
    """Train neural model and return model + full history"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    
    history = TrainingHistory()
    best_state = None
    patience_counter = 0
    
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    # Use seeded generator for reproducible shuffling
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        generator=get_generator(seed),
        worker_init_fn=worker_init_fn
    )
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        epoch_train_loss = np.mean(train_losses)
        history.train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
        history.val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break
    
    history.total_epochs = epoch + 1
    
    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)
    
    return model, history


def evaluate_neural_model(
    model: nn.Module,
    X_test: np.ndarray, y_test: np.ndarray,
    scaler_y: StandardScaler
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate and return predictions on original scale"""
    model.eval()
    X_test_t = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        pred_scaled = model(X_test_t).cpu().numpy()
    
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    y_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    
    return pred, y_original


# =============================================================================
# DATASET LOADING
# =============================================================================
# Combined benchmark from:
# 1. OpenML-CTR23 Regression datasets (TabPFN paper - 26 datasets)
# 2. AMLB AutoML Benchmark Regression (OpenML Suite 269 - 33 datasets)
#
# Together provides ~45 unique regression datasets covering both benchmarks
# =============================================================================

# OpenML-CTR23 Regression datasets (TabPFN paper basis)
OPENML_CTR23_REGRESSION = [
    # OpenML-CTR23 datasets
    (505, 'tecator', None),
    (507, 'space_ga', None),
    (531, 'boston', None),
    (541, 'socmob', None),
    (546, 'sensory', None),
    (547, 'no2', None),
    (550, 'quake', None),
    (574, 'house_16H', 10000),         # 22784 samples - subsample
    (1028, 'SWD', None),
    (1030, 'ERA', None),
    (1096, 'FacultySalaries', None),
    (4545, 'wine_quality', 10000),     # 39644 samples - subsample
    (41021, 'Moneyball', None),
    (41540, 'black_friday', 10000),
    (41702, 'wine-quality-white', None),
    (42225, 'diamonds', 10000),
    (42563, 'Mercedes_Benz_Greener_Manufacturing', None),
    (42570, 'Brazilian_houses', 10000),
    (42571, 'Bike_Sharing_Demand', 10000),  # 188318 samples - subsample
    (42688, 'abalone', 10000),              # 10692 samples - subsample  
    (42726, 'us_crime', None),
    (42727, 'MIP-2016-regression', None),
    (42728, 'nyc-taxi-green-dec-2016', 10000),
    (42729, 'delays_zurich_transport', 10000),
    (42730, 'yprop_4_1', None),
    (42731, 'analcatdata_supreme', 10000),  # 21613 samples - subsample
]

# AMLB AutoML Benchmark Regression Suite 269 (33 datasets)
# These are additional datasets used by AMLB paper (Gijsbers et al., JMLR 2024)
# Source: https://www.openml.org/s/269
AMLB_REGRESSION = [
    # Medium-sized datasets
    (287, 'wine_quality', 10000),           # Also in CTR23 but different variant
    (216, 'elevators', 10000),              # 16599 samples
    (218, 'house_8L', None),                # 22784 samples (same as house_16H)
    (4550, 'MiceProtein', None),            # 1080 samples
    (42464, 'pol', None),                   # 15000 samples
    (42724, 'OnlineNewsPopularity', 10000), # 39644 samples - subsample
    (42705, 'Yolanda', 10000),              # 400000 samples - subsample
    
    # Classic regression datasets
    (189, 'kin8nm', None),                  # 8192 samples - robot arm kinematics
    (1191, 'BNG_wine_quality', 10000),      # wine quality variant
    (1193, 'BNG_cpu_small', 10000),         # CPU performance
    (1196, 'BNG_pbc', None),                # Liver disease survival
    (1199, 'BNG_echoMonths', None),         # Echocardiogram
    (1201, 'BNG_lowbwt', None),             # Birth weight
    
    # Housing/Real Estate datasets  
    (537, 'houses', None),                  # California housing variant
    (42092, 'ames_housing', None),          # Ames Iowa housing (Kaggle-style)
    (42165, 'california_housing', 10000),   # 20640 samples
    (42492, 'house_sales', 10000),          # King County house sales
    
    # Scientific/Technical datasets
    (44090, 'year', 10000),                 # Million Song subset - year prediction
    (44091, 'sarcos', 10000),               # Robot inverse dynamics
    (44128, 'sulfur', None),                # Chemistry - sulfur recovery
    (44135, 'bank8FM', None),               # Finance - bank rejection
    (44136, 'pumadyn32nm', None),           # Robot arm dynamics
    (44137, 'cpu_act', None),               # CPU activity
    
    # Challenging real-world datasets
    (44125, 'colleges', None),              # US college statistics
    (44126, 'house_prices_nominal', None),  # Nominal house prices
    (44129, 'topo_2_1', None),              # Topological data
    (44132, 'visualizing_soil', None),      # Soil visualization
    
    # Large-scale datasets (subsampled)
    (41979, 'airlines', 10000),             # Flight delays (millions of rows)
    (44056, 'superconduct', 10000),         # Superconductor critical temp
    (44057, 'MiniBooNE', 10000),            # Particle physics
    (44061, 'jannis', 10000),               # Large classification as regression
    (44063, 'covertype', 10000),            # Forest cover type
]

# Combined full benchmark: CTR23 + AMLB unique datasets
# This covers both benchmarks comprehensively
FULL_BENCHMARK_DATASETS = OPENML_CTR23_REGRESSION + [
    # Add AMLB datasets that are NOT already in CTR23 (avoiding duplicates by name)
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
    (44129, 'topo_2_1', 10000),       # Very large, subsample
    (44056, 'superconduct', 10000),
    (44061, 'jannis', 10000),
]

# Note: The following datasets were removed due to issues:
# - house_sales (42492): Target column issue
# - MiceProtein (4550): Object dtype casting issue  
# - ames_housing (42092): Duplicate of analcatdata_supreme (same data)
# - year (44090): Object dtype issue
# - sarcos (44091): Object dtype issue
# - sulfur (44128): Object dtype issue
# - colleges (44125): Object dtype issue
# - airlines (41979): Dataset not found on OpenML

# Quick test subset
QUICK_TEST_DATASETS = [
    (531, 'boston', None),
    (546, 'sensory', None),
    (547, 'no2', None),
    (550, 'quake', None),
    (4545, 'wine_quality', None),
]


def load_openml_dataset(dataset_id: int, name: str, max_samples: Optional[int] = None, 
                        seed: int = 42) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Load dataset from OpenML"""
    try:
        data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
        X = data.data
        y = data.target
        
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Handle categorical features
        X_processed = np.zeros((X.shape[0], X.shape[1]))
        for i in range(X.shape[1]):
            col = X[:, i]
            try:
                X_processed[:, i] = col.astype(float)
            except (ValueError, TypeError):
                le = LabelEncoder()
                col_str = np.array([str(x) if pd.notna(x) else 'missing' for x in col])
                X_processed[:, i] = le.fit_transform(col_str)
        
        X = X_processed
        y = y.astype(float)
        
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=np.nanmean(y))
        
        # Subsample if needed
        if max_samples and len(X) > max_samples:
            np.random.seed(seed)
            idx = np.random.choice(len(X), max_samples, replace=False)
            X, y = X[idx], y[idx]
        
        return X, y, None
    except Exception as e:
        return None, None, str(e)


# =============================================================================
# METHOD RUNNERS
# =============================================================================
def run_tree_method(
    method_name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    scaler_y: StandardScaler,
    seed: int, rep_idx: int,
    use_gpu: bool = True
) -> RepetitionResult:
    """Run a tree-based method with optional GPU acceleration"""
    t0 = time.time()
    
    if method_name == 'XGBoost':
        if use_gpu:
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, random_state=seed, verbosity=0,
                tree_method='hist', device='cuda'
            )
        else:
            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=seed, verbosity=0)
    elif method_name == 'CatBoost':
        if use_gpu:
            model = CatBoostRegressor(
                iterations=100, depth=6, random_state=seed, verbose=False,
                task_type='GPU', devices='0'
            )
        else:
            model = CatBoostRegressor(iterations=100, depth=6, random_state=seed, verbose=False)
    elif method_name == 'LightGBM':
        # LightGBM GPU requires special build, fallback to CPU if fails
        if use_gpu:
            try:
                model = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=6, random_state=seed, verbosity=-1,
                    device='gpu', gpu_platform_id=0, gpu_device_id=0
                )
            except Exception:
                model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, random_state=seed, verbosity=-1)
        else:
            model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, random_state=seed, verbosity=-1)
    elif method_name == 'RandomForest':
        # RandomForest has no GPU support in sklearn, use all CPU cores
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    model.fit(X_train, y_train)
    pred_scaled = model.predict(X_test)
    
    # Convert to original scale
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    y_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    
    train_time = time.time() - t0
    metrics = compute_all_metrics(y_original, pred)
    
    return RepetitionResult(
        seed=seed,
        rep_idx=rep_idx,
        r2=metrics['r2'],
        rmse=metrics['rmse'],
        mae=metrics['mae'],
        spearman=metrics['spearman'],
        train_time=train_time,
        predictions=pred,
        ground_truth=y_original
    )


def run_neural_method(
    method_name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    scaler_y: StandardScaler,
    n_features: int,
    config: BenchmarkConfig,
    seed: int, rep_idx: int
) -> RepetitionResult:
    """Run a neural method (MLP, RW-MLP, FT-Transformer, RW-FT)"""
    set_seed(seed)
    t0 = time.time()
    
    is_transformer = 'FT' in method_name or 'Transformer' in method_name
    lr = config.lr_transformer if is_transformer else config.lr_mlp
    
    if method_name == 'MLP':
        model = MLP(n_features)
        model, history = train_neural_model(model, X_train, y_train, X_val, y_val, config, lr, seed)
        pred, y_original = evaluate_neural_model(model, X_test, y_test, scaler_y)
        best_k = None
        
    elif method_name == 'RW-MLP':
        # Try different k values, pick best on validation
        best_val_loss = float('inf')
        best_model = None
        best_history = None
        best_k = 1
        
        for k in config.k_values:
            set_seed(seed)
            model = RWMLP(n_features, k=k)
            model, history = train_neural_model(model, X_train, y_train, X_val, y_val, config, lr, seed)
            if history.best_val_loss < best_val_loss:
                best_val_loss = history.best_val_loss
                best_model = model
                best_history = history
                best_k = k
        
        pred, y_original = evaluate_neural_model(best_model, X_test, y_test, scaler_y)
        history = best_history
        
    elif method_name == 'FT-Transformer':
        model = FTTransformer(n_features)
        model, history = train_neural_model(model, X_train, y_train, X_val, y_val, config, lr, seed)
        pred, y_original = evaluate_neural_model(model, X_test, y_test, scaler_y)
        best_k = None
        
    elif method_name == 'RW-FT-Transformer':
        best_val_loss = float('inf')
        best_model = None
        best_history = None
        best_k = 1
        
        for k in config.k_values:
            set_seed(seed)
            model = RWFTTransformer(n_features, k=k)
            model, history = train_neural_model(model, X_train, y_train, X_val, y_val, config, lr, seed)
            if history.best_val_loss < best_val_loss:
                best_val_loss = history.best_val_loss
                best_model = model
                best_history = history
                best_k = k
        
        pred, y_original = evaluate_neural_model(best_model, X_test, y_test, scaler_y)
        history = best_history
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    train_time = time.time() - t0
    metrics = compute_all_metrics(y_original, pred)
    
    return RepetitionResult(
        seed=seed,
        rep_idx=rep_idx,
        r2=metrics['r2'],
        rmse=metrics['rmse'],
        mae=metrics['mae'],
        spearman=metrics['spearman'],
        train_time=train_time,
        best_k=best_k,
        history=history,
        predictions=pred,
        ground_truth=y_original
    )


# =============================================================================
# NORMALIZATION (Paper's method)
# =============================================================================
def normalize_scores(dataset_results: DatasetResults, metric: str = 'rmse'):
    """
    Normalize scores per dataset (paper's method):
    - 0.0 = worst performance among all methods
    - 1.0 = best performance among all methods
    
    For RMSE/MAE: lower is better, so we invert
    For R2/Spearman: higher is better
    """
    lower_is_better = metric in ['rmse', 'mae']
    
    # Get all method scores for this metric
    scores = {}
    for method_name, method_result in dataset_results.methods.items():
        if metric == 'rmse':
            scores[method_name] = method_result.mean_rmse
        elif metric == 'mae':
            scores[method_name] = method_result.mean_mae
        elif metric == 'r2':
            scores[method_name] = method_result.mean_r2
        elif metric == 'spearman':
            scores[method_name] = method_result.mean_spearman
    
    if not scores:
        return
    
    min_score = min(scores.values())
    max_score = max(scores.values())
    
    if max_score == min_score:
        # All methods have same score
        normalized = {m: 1.0 for m in scores}
    elif lower_is_better:
        # For RMSE/MAE: invert so that lower raw = higher normalized
        normalized = {m: (max_score - s) / (max_score - min_score) for m, s in scores.items()}
    else:
        # For R2/Spearman: higher raw = higher normalized
        normalized = {m: (s - min_score) / (max_score - min_score) for m, s in scores.items()}
    
    if metric not in dataset_results.normalized_scores:
        dataset_results.normalized_scores[metric] = {}
    dataset_results.normalized_scores[metric] = normalized


# =============================================================================
# VISUALIZATION (Replicating paper figures)
# =============================================================================
def create_paper_figures(results: BenchmarkResults, output_dir: str):
    """Create figures matching TabPFN paper style"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect data for plotting
    methods = ['XGBoost', 'CatBoost', 'LightGBM', 'RandomForest', 'MLP', 'RW-MLP', 'FT-Transformer', 'RW-FT-Transformer']
    metrics = ['rmse', 'r2', 'mae', 'spearman']
    
    # Aggregate normalized scores across datasets
    norm_scores = {m: {metric: [] for metric in metrics} for m in methods}
    raw_scores = {m: {metric: [] for metric in metrics} for m in methods}
    
    for ds_name, ds_result in results.datasets.items():
        for method in methods:
            if method in ds_result.methods:
                method_result = ds_result.methods[method]
                raw_scores[method]['rmse'].append(method_result.mean_rmse)
                raw_scores[method]['r2'].append(method_result.mean_r2)
                raw_scores[method]['mae'].append(method_result.mean_mae)
                raw_scores[method]['spearman'].append(method_result.mean_spearman)
                
                for metric in metrics:
                    if metric in ds_result.normalized_scores and method in ds_result.normalized_scores[metric]:
                        norm_scores[method][metric].append(ds_result.normalized_scores[metric][method])
    
    # ==========================================================================
    # Figure 1: Average Normalized Scores (like Fig 4a)
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#2ecc71', '#27ae60', '#3498db', '#9b59b6', '#e74c3c', '#c0392b', '#f39c12', '#d35400']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        means = [np.mean(norm_scores[m][metric]) if norm_scores[m][metric] else 0 for m in methods]
        stds = [np.std(norm_scores[m][metric]) / np.sqrt(len(norm_scores[m][metric])) 
                if len(norm_scores[m][metric]) > 1 else 0 for m in methods]
        
        bars = ax.bar(range(len(methods)), means, yerr=stds, color=colors, capsize=3)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([m.replace('-', '\n') for m in methods], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(f'Normalized {metric.upper()}')
        ax.set_title(f'Average Normalized {metric.upper()} (1=best)')
        ax.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{mean:.3f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_normalized_scores.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==========================================================================
    # Figure 2: Per-Dataset Scatter Plots (like Fig 4b)
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    comparisons = [
        ('MLP', 'RW-MLP', 'RW-MLP vs MLP'),
        ('FT-Transformer', 'RW-FT-Transformer', 'RW-FT vs FT-Transformer'),
        ('XGBoost', 'RW-MLP', 'RW-MLP vs XGBoost'),
        ('CatBoost', 'RW-FT-Transformer', 'RW-FT vs CatBoost'),
    ]
    
    for idx, (m1, m2, title) in enumerate(comparisons):
        ax = axes[idx // 2, idx % 2]
        
        # Collect per-dataset normalized RMSE
        x_vals, y_vals = [], []
        for ds_name, ds_result in results.datasets.items():
            if 'rmse' in ds_result.normalized_scores:
                if m1 in ds_result.normalized_scores['rmse'] and m2 in ds_result.normalized_scores['rmse']:
                    x_vals.append(ds_result.normalized_scores['rmse'][m1])
                    y_vals.append(ds_result.normalized_scores['rmse'][m2])
        
        ax.scatter(x_vals, y_vals, c='#3498db', s=80, alpha=0.7, edgecolors='white', linewidth=0.5)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Equal')
        ax.set_xlabel(f'{m1} (Normalized RMSE)')
        ax.set_ylabel(f'{m2} (Normalized RMSE)')
        ax.set_title(title)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Count wins
        wins_m2 = sum(1 for x, y in zip(x_vals, y_vals) if y > x)
        wins_m1 = sum(1 for x, y in zip(x_vals, y_vals) if x > y)
        ax.text(0.05, 0.95, f'{m2} wins: {wins_m2}', transform=ax.transAxes, fontsize=9)
        ax.text(0.95, 0.05, f'{m1} wins: {wins_m1}', transform=ax.transAxes, ha='right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_per_dataset_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==========================================================================
    # Figure 3: Win Counts (like part of Fig 4)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Count wins per method (using normalized RMSE)
    win_counts = {m: 0 for m in methods}
    n_datasets = len(results.datasets)
    
    for ds_name, ds_result in results.datasets.items():
        if 'rmse' in ds_result.normalized_scores:
            best_method = max(ds_result.normalized_scores['rmse'].items(), key=lambda x: x[1])[0]
            if best_method in win_counts:
                win_counts[best_method] += 1
    
    wins = [win_counts[m] for m in methods]
    bars = ax.bar(range(len(methods)), wins, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('-', '\n') for m in methods], rotation=45, ha='right')
    ax.set_ylabel('Number of Wins')
    ax.set_title(f'Win Counts (n={n_datasets} datasets)')
    
    for bar, win in zip(bars, wins):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
               f'{win}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_win_counts.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==========================================================================
    # Figure 4: Improvement Distribution (RW methods vs base)
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # RW-MLP vs MLP improvement
    improvements_mlp = []
    for ds_name, ds_result in results.datasets.items():
        if 'MLP' in ds_result.methods and 'RW-MLP' in ds_result.methods:
            mlp_rmse = ds_result.methods['MLP'].mean_rmse
            rwmlp_rmse = ds_result.methods['RW-MLP'].mean_rmse
            if mlp_rmse > 0:
                improvement = (mlp_rmse - rwmlp_rmse) / mlp_rmse * 100
                improvements_mlp.append(improvement)
    
    ax = axes[0]
    colors_bar = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements_mlp]
    ax.bar(range(len(improvements_mlp)), improvements_mlp, color=colors_bar)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('RMSE Improvement (%)')
    ax.set_title(f'RW-MLP vs MLP: {sum(1 for x in improvements_mlp if x > 0)}/{len(improvements_mlp)} improvements')
    
    # RW-FT vs FT improvement
    improvements_ft = []
    for ds_name, ds_result in results.datasets.items():
        if 'FT-Transformer' in ds_result.methods and 'RW-FT-Transformer' in ds_result.methods:
            ft_rmse = ds_result.methods['FT-Transformer'].mean_rmse
            rwft_rmse = ds_result.methods['RW-FT-Transformer'].mean_rmse
            if ft_rmse > 0:
                improvement = (ft_rmse - rwft_rmse) / ft_rmse * 100
                improvements_ft.append(improvement)
    
    ax = axes[1]
    colors_bar = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements_ft]
    ax.bar(range(len(improvements_ft)), improvements_ft, color=colors_bar)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('RMSE Improvement (%)')
    ax.set_title(f'RW-FT vs FT-Trans: {sum(1 for x in improvements_ft if x > 0)}/{len(improvements_ft)} improvements')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_improvements.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==========================================================================
    # Figure 5: Average Rank (like paper's ranking analysis)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute ranks per dataset
    ranks = {m: [] for m in methods}
    for ds_name, ds_result in results.datasets.items():
        if 'rmse' in ds_result.normalized_scores:
            # Sort methods by normalized score (higher is better)
            sorted_methods = sorted(ds_result.normalized_scores['rmse'].items(), 
                                   key=lambda x: x[1], reverse=True)
            for rank, (method, _) in enumerate(sorted_methods, 1):
                if method in ranks:
                    ranks[method].append(rank)
    
    avg_ranks = [np.mean(ranks[m]) if ranks[m] else len(methods) for m in methods]
    bars = ax.bar(range(len(methods)), avg_ranks, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('-', '\n') for m in methods], rotation=45, ha='right')
    ax.set_ylabel('Average Rank')
    ax.set_title('Average Rank (lower = better)')
    ax.set_ylim(0, len(methods) + 1)
    
    for bar, rank in zip(bars, avg_ranks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{rank:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_average_rank.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==========================================================================
    # Figure 6: Training Curves (sample from one dataset)
    # ==========================================================================
    # Find a dataset with training history
    sample_ds = None
    for ds_name, ds_result in results.datasets.items():
        if 'RW-MLP' in ds_result.methods:
            reps = ds_result.methods['RW-MLP'].repetitions
            if reps and reps[0].history:
                sample_ds = (ds_name, ds_result)
                break
    
    if sample_ds:
        ds_name, ds_result = sample_ds
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        neural_methods = ['MLP', 'RW-MLP', 'FT-Transformer', 'RW-FT-Transformer']
        
        for idx, method in enumerate(neural_methods):
            ax = axes[idx // 2, idx % 2]
            if method in ds_result.methods:
                rep = ds_result.methods[method].repetitions[0]
                if rep.history:
                    epochs = range(1, len(rep.history.train_losses) + 1)
                    ax.plot(epochs, rep.history.train_losses, label='Train Loss', color='#3498db')
                    ax.plot(epochs, rep.history.val_losses, label='Val Loss', color='#e74c3c')
                    ax.axvline(x=rep.history.best_epoch + 1, color='green', linestyle='--', 
                              label=f'Best Epoch ({rep.history.best_epoch + 1})')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('MSE Loss')
                    ax.set_title(f'{method} Training Curve ({ds_name})')
                    ax.legend()
                    ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fig6_training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Figures saved to {output_dir}/")


def create_summary_table(results: BenchmarkResults, output_dir: str):
    """Create summary tables (like Extended Data Tables)"""
    methods = ['XGBoost', 'CatBoost', 'LightGBM', 'RandomForest', 'MLP', 'RW-MLP', 'FT-Transformer', 'RW-FT-Transformer']
    
    # Collect aggregated metrics
    data = []
    for method in methods:
        row = {'Method': method}
        
        # Aggregate across datasets
        all_r2, all_rmse, all_mae, all_spearman, all_time = [], [], [], [], []
        all_norm_rmse = []
        
        for ds_name, ds_result in results.datasets.items():
            if method in ds_result.methods:
                mr = ds_result.methods[method]
                all_r2.append(mr.mean_r2)
                all_rmse.append(mr.mean_rmse)
                all_mae.append(mr.mean_mae)
                all_spearman.append(mr.mean_spearman)
                all_time.append(mr.mean_time)
                
                if 'rmse' in ds_result.normalized_scores and method in ds_result.normalized_scores['rmse']:
                    all_norm_rmse.append(ds_result.normalized_scores['rmse'][method])
        
        row['Mean R²'] = f"{np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}" if all_r2 else "N/A"
        row['Mean RMSE'] = f"{np.mean(all_rmse):.4f}" if all_rmse else "N/A"
        row['Mean MAE'] = f"{np.mean(all_mae):.4f}" if all_mae else "N/A"
        row['Mean Spearman'] = f"{np.mean(all_spearman):.4f}" if all_spearman else "N/A"
        row['Norm RMSE'] = f"{np.mean(all_norm_rmse):.4f} ± {np.std(all_norm_rmse):.4f}" if all_norm_rmse else "N/A"
        row['Mean Time (s)'] = f"{np.mean(all_time):.2f}" if all_time else "N/A"
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'summary_table.csv'), index=False)
    
    # Also save detailed per-dataset results
    detailed_data = []
    for ds_name, ds_result in results.datasets.items():
        for method in methods:
            if method in ds_result.methods:
                mr = ds_result.methods[method]
                detailed_data.append({
                    'Dataset': ds_name,
                    'Method': method,
                    'R²': mr.mean_r2,
                    'R² std': mr.std_r2,
                    'RMSE': mr.mean_rmse,
                    'RMSE std': mr.std_rmse,
                    'MAE': mr.mean_mae,
                    'MAE std': mr.std_mae,
                    'Spearman': mr.mean_spearman,
                    'Spearman std': mr.std_spearman,
                    'Time (s)': mr.mean_time,
                    'Best k': mr.repetitions[0].best_k if mr.repetitions else None
                })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(os.path.join(output_dir, 'detailed_results.csv'), index=False)
    
    print(f"\nSummary Table:")
    print(df.to_string(index=False))
    
    return df


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================
def run_benchmark(
    use_quick_test: bool = True,
    n_repetitions: int = 10,
    output_dir: str = "benchmark_results",
    use_gpu: bool = True,
    benchmark_type: str = "combined"  # "ctr23", "amlb", or "combined"
) -> BenchmarkResults:
    """Run the full benchmark
    
    Args:
        use_quick_test: If True, use quick test datasets (5 small datasets)
        n_repetitions: Number of repetitions per dataset (default 10)
        output_dir: Directory for output files
        use_gpu: Enable GPU acceleration for tree methods
        benchmark_type: Which benchmark suite to use:
            - "ctr23": OpenML-CTR23 only (26 datasets, TabPFN paper)
            - "amlb": AMLB only (33 datasets, AMLB paper)
            - "combined": Both CTR23 + AMLB unique datasets (~47 datasets)
    """
    
    # Configuration
    config = BenchmarkConfig(
        n_repetitions=n_repetitions,
        output_dir=output_dir,
        use_gpu=use_gpu
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("RW-LAYER BENCHMARK: TabPFN + AMLB Protocol Replication")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"GPU for tree methods: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"Repetitions per dataset: {config.n_repetitions}")
    print(f"Train/Test split: {100*(1-config.test_size):.0f}% / {100*config.test_size:.0f}%")
    print(f"Seeds: {config.get_seeds()}")
    print(f"Benchmark type: {benchmark_type}")
    
    # Select datasets based on benchmark type
    if use_quick_test:
        datasets_list = QUICK_TEST_DATASETS
        benchmark_desc = "quick test"
    elif benchmark_type == "ctr23":
        datasets_list = OPENML_CTR23_REGRESSION
        benchmark_desc = "OpenML-CTR23 (TabPFN paper)"
    elif benchmark_type == "amlb":
        datasets_list = AMLB_REGRESSION
        benchmark_desc = "AMLB (AutoML Benchmark)"
    else:  # combined
        datasets_list = FULL_BENCHMARK_DATASETS
        benchmark_desc = "Combined (CTR23 + AMLB)"
    
    print(f"\nDatasets: {len(datasets_list)} ({benchmark_desc})")
    
    # Initialize results
    results = BenchmarkResults(config=config)
    
    # Methods to evaluate
    tree_methods = ['XGBoost', 'CatBoost', 'LightGBM', 'RandomForest']
    neural_methods = ['MLP', 'RW-MLP', 'FT-Transformer', 'RW-FT-Transformer']
    all_methods = tree_methods + neural_methods
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = {}
    for dataset_id, name, max_samples in datasets_list:
        X, y, error = load_openml_dataset(dataset_id, name, max_samples, seed=config.base_seed)
        if X is not None and len(X) >= 50:
            datasets[name] = {'X': X, 'y': y, 'id': dataset_id}
            print(f"  ✓ {name}: {X.shape}")
        else:
            print(f"  ✗ {name}: {'SKIPPED' if X is not None else f'FAILED - {error}'}")
    
    print(f"\nTotal: {len(datasets)} datasets loaded")
    
    # Run benchmark
    for ds_name, ds_info in tqdm(datasets.items(), desc="Datasets"):
        X, y = ds_info['X'], ds_info['y']
        n_samples, n_features = X.shape
        
        print(f"\n{'='*60}")
        print(f"{ds_name}: {X.shape}, {config.n_repetitions} repetitions")
        print(f"{'='*60}")
        
        # Initialize dataset results
        ds_result = DatasetResults(
            dataset_name=ds_name,
            dataset_id=ds_info['id'],
            n_samples=n_samples,
            n_features=n_features
        )
        
        # Initialize method results
        for method in all_methods:
            ds_result.methods[method] = MethodResults(method_name=method)
        
        # Run all repetitions
        for rep_idx, seed in enumerate(config.get_seeds()):
            set_seed(seed)
            
            # Split data (paper's 90/10 split)
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X, y, test_size=config.test_size, random_state=seed
            )
            
            # Further split for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=config.val_size, random_state=seed
            )
            
            # Scale data
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_s = scaler_X.fit_transform(X_train)
            X_val_s = scaler_X.transform(X_val)
            X_test_s = scaler_X.transform(X_test)
            
            y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_val_s = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
            y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
            
            # For tree methods, use train + val combined
            X_train_tree = np.vstack([X_train_s, X_val_s])
            y_train_tree = np.concatenate([y_train_s, y_val_s])
            
            # Run tree methods
            for method in tree_methods:
                rep_result = run_tree_method(
                    method, X_train_tree, y_train_tree, X_test_s, y_test_s,
                    scaler_y, seed, rep_idx, use_gpu=config.use_gpu
                )
                ds_result.methods[method].repetitions.append(rep_result)
            
            # Run neural methods
            for method in neural_methods:
                rep_result = run_neural_method(
                    method, X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s,
                    scaler_y, n_features, config, seed, rep_idx
                )
                ds_result.methods[method].repetitions.append(rep_result)
            
            # Progress
            if (rep_idx + 1) % 5 == 0:
                print(f"  Completed {rep_idx + 1}/{config.n_repetitions} repetitions")
        
        # Compute aggregates
        for method in all_methods:
            ds_result.methods[method].compute_aggregates()
        
        # Compute normalized scores
        for metric in ['rmse', 'r2', 'mae', 'spearman']:
            normalize_scores(ds_result, metric)
        
        # Print summary for this dataset
        print(f"\n  Results for {ds_name}:")
        print(f"  {'Method':<20} {'R²':>10} {'RMSE':>10} {'Norm RMSE':>10}")
        print(f"  {'-'*50}")
        for method in all_methods:
            mr = ds_result.methods[method]
            norm = ds_result.normalized_scores.get('rmse', {}).get(method, 0)
            print(f"  {method:<20} {mr.mean_r2:>10.4f} {mr.mean_rmse:>10.4f} {norm:>10.4f}")
        
        results.datasets[ds_name] = ds_result
    
    # Save results
    results_file = os.path.join(output_dir, 'benchmark_results.json')
    results.save(results_file)
    print(f"\nResults saved to: {results_file}")
    
    # Create figures
    create_paper_figures(results, output_dir)
    
    # Create summary table
    create_summary_table(results, output_dir)
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='RW-Layer Benchmark: Replicating TabPFN + AMLB Evaluation Protocol',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Benchmark Types:
  ctr23     OpenML-CTR23 only (26 datasets, TabPFN Nature paper)
  amlb      AMLB only (33 datasets, AutoML Benchmark JMLR 2024)
  combined  Both CTR23 + unique AMLB datasets (~47 datasets)

Examples:
  # Quick test (5 datasets, 10 reps) - ~30 min
  python rw_layer_benchmark_full.py
  
  # Full TabPFN benchmark (26 datasets)
  python rw_layer_benchmark_full.py --full --benchmark ctr23
  
  # Full AMLB benchmark (33 datasets)
  python rw_layer_benchmark_full.py --full --benchmark amlb
  
  # Combined benchmark (47 datasets) - most comprehensive
  python rw_layer_benchmark_full.py --full --benchmark combined
  
  # Disable GPU for tree methods
  python rw_layer_benchmark_full.py --full --no-gpu
"""
    )
    parser.add_argument('--full', action='store_true', help='Run full benchmark (all datasets)')
    parser.add_argument('--benchmark', type=str, default='combined', 
                       choices=['ctr23', 'amlb', 'combined'],
                       help='Benchmark type: ctr23 (TabPFN), amlb (AutoML), or combined (default)')
    parser.add_argument('--reps', type=int, default=10, help='Number of repetitions (default: 10)')
    parser.add_argument('--output', type=str, default='benchmark_results', help='Output directory')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU for tree methods')
    
    args = parser.parse_args()
    
    results = run_benchmark(
        use_quick_test=not args.full,
        n_repetitions=args.reps,
        output_dir=args.output,
        use_gpu=not args.no_gpu,
        benchmark_type=args.benchmark
    )
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)