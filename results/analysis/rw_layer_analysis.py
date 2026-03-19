#!/usr/bin/env python3
"""
RW-Layer Benchmark Analysis - Nature-style Tables and Figures
Inspired by TabPFN (Hollmann et al., Nature 2025)

Usage:
    python rw_layer_analysis.py

Data files should be in ./data/ folder:
    - detailed_results.csv
    - neural_rw_results.csv (or 1_neural_rw_results.csv)
    - linear_mlp_results.csv
    - ablation_results.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = './data'
OUTPUT_DIR = './output'

# Nature-style color palette (similar to TabPFN paper)
COLORS = {
    'RW-MLP': '#1f77b4',        # Blue (main method)
    'MLP': '#ff7f0e',           # Orange
    'CatBoost': '#2ca02c',      # Green
    'XGBoost': '#d62728',       # Red
    'LightGBM': '#9467bd',      # Purple
    'RandomForest': '#8c564b',  # Brown
    'FT-Transformer': '#e377c2', # Pink
    'RW-FT-Transformer': '#7f7f7f', # Gray
    'Linear-MLP': '#bcbd22',    # Yellow-green
    'RTDL-ResNet': '#17becf',   # Cyan
    'RW-ResNet': '#aec7e8',     # Light blue
    'NODE': '#ffbb78',          # Light orange
    'RW-NODE': '#98df8a',       # Light green
    'TabNet': '#ff9896',        # Light red
    'RW-TabNet': '#c5b0d5',     # Light purple
    'Identity-MLP': '#c49c94',
    'NoAct-MLP': '#f7b6d2',
    'IdentityNoAct-MLP': '#c7c7c7',
}

# Nature figure style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load all result files from DATA_DIR"""
    print(f"Loading data from {DATA_DIR}...")
    
    # Load detailed results
    detailed_path = os.path.join(DATA_DIR, 'detailed_results.csv')
    detailed = pd.read_csv(detailed_path)
    print(f"  ✓ detailed_results.csv: {len(detailed)} rows")
    
    # Load neural RW results (try both naming conventions)
    neural_path = os.path.join(DATA_DIR, 'neural_rw_results.csv')
    if not os.path.exists(neural_path):
        neural_path = os.path.join(DATA_DIR, '1_neural_rw_results.csv')
    neural_rw = pd.read_csv(neural_path)
    print(f"  ✓ neural_rw_results.csv: {len(neural_rw)} rows")
    
    # Load linear MLP results
    linear_path = os.path.join(DATA_DIR, 'linear_mlp_results.csv')
    linear_mlp = pd.read_csv(linear_path)
    print(f"  ✓ linear_mlp_results.csv: {len(linear_mlp)} rows")
    
    # Load ablation results
    ablation_path = os.path.join(DATA_DIR, 'ablation_results.csv')
    ablation = pd.read_csv(ablation_path)
    print(f"  ✓ ablation_results.csv: {len(ablation)} rows")
    
    return detailed, neural_rw, linear_mlp, ablation


def combine_all_results(detailed, neural_rw, linear_mlp):
    """Combine all results into a unified dataframe"""
    # Standardize detailed results
    df1 = detailed[['Dataset', 'Method', 'R²', 'R² std']].copy()
    df1.columns = ['dataset', 'method', 'r2_mean', 'r2_std']
    
    # Neural RW results already standardized
    df2 = neural_rw[['dataset', 'method', 'r2_mean', 'r2_std']].copy()
    
    # Linear-MLP results
    df3 = linear_mlp[['Dataset', 'Method', 'R²', 'R² std']].copy()
    df3.columns = ['dataset', 'method', 'r2_mean', 'r2_std']
    
    # Combine
    combined = pd.concat([df1, df2, df3], ignore_index=True)
    return combined


# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def normalize_scores(pivot_df):
    """Normalize scores per dataset (0=worst, 1=best)"""
    normalized = pivot_df.copy()
    for idx in normalized.index:
        row_min = normalized.loc[idx].min()
        row_max = normalized.loc[idx].max()
        if row_max > row_min:
            normalized.loc[idx] = (normalized.loc[idx] - row_min) / (row_max - row_min)
        else:
            normalized.loc[idx] = 1.0
    return normalized


def compute_ranks(pivot_df):
    """Compute ranks per dataset (1=best)"""
    return pivot_df.rank(axis=1, ascending=False)


def wilcoxon_test(scores1, scores2):
    """Compute Wilcoxon signed-rank test"""
    common_idx = scores1.dropna().index.intersection(scores2.dropna().index)
    if len(common_idx) < 5:
        return np.nan
    s1 = scores1.loc[common_idx]
    s2 = scores2.loc[common_idx]
    try:
        stat, pval = stats.wilcoxon(s1, s2)
        return pval
    except:
        return np.nan


def win_rate(scores1, scores2):
    """Compute win rate of method 1 over method 2"""
    common_idx = scores1.dropna().index.intersection(scores2.dropna().index)
    if len(common_idx) == 0:
        return np.nan, 0
    wins = (scores1.loc[common_idx] > scores2.loc[common_idx]).sum()
    return wins / len(common_idx), len(common_idx)


def create_pairwise_comparison(pivot, method1, method2):
    """Create pairwise comparison statistics"""
    common = pivot[[method1, method2]].dropna()
    
    wins = (common[method1] > common[method2]).sum()
    total = len(common)
    win_pct = wins / total * 100
    
    mean_diff = (common[method1] - common[method2]).mean()
    
    try:
        stat, pval = stats.wilcoxon(common[method1], common[method2])
    except:
        pval = np.nan
    
    return {
        'comparison': f'{method1} vs {method2}',
        'wins': wins,
        'total': total,
        'win_rate': win_pct,
        'mean_diff': mean_diff,
        'p_value': pval
    }


# ============================================================================
# TABLE GENERATION
# ============================================================================

def create_main_results_table(combined_df):
    """Create main results table (Extended Data Table style)"""
    pivot = combined_df.pivot_table(
        index='dataset', columns='method', values='r2_mean', aggfunc='mean'
    )
    
    normalized = normalize_scores(pivot)
    ranks = compute_ranks(pivot)
    
    results = []
    for method in pivot.columns:
        scores = pivot[method].dropna()
        norm_scores = normalized[method].dropna()
        rank_scores = ranks[method].dropna()
        
        results.append({
            'Method': method,
            'Mean R²': scores.mean(),
            'Std R²': scores.std(),
            'Norm R² (↑)': norm_scores.mean(),
            'Norm Std': norm_scores.std(),
            'Avg Rank (↓)': rank_scores.mean(),
            'N': len(scores)
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Avg Rank (↓)')
    
    return results_df, pivot


def create_rw_effect_table(pivot):
    """Create table showing RW-Layer effect on each architecture"""
    comparisons = [
        ('RW-MLP', 'MLP'),
        ('RW-MLP', 'Linear-MLP'),
        ('RW-FT-Transformer', 'FT-Transformer'),
        ('RW-ResNet', 'RTDL-ResNet'),
        ('RW-NODE', 'NODE'),
        ('RW-TabNet', 'TabNet'),
    ]
    
    results = []
    for rw_method, base_method in comparisons:
        if rw_method in pivot.columns and base_method in pivot.columns:
            result = create_pairwise_comparison(pivot, rw_method, base_method)
            result['RW Method'] = rw_method
            result['Base Method'] = base_method
            results.append(result)
    
    return pd.DataFrame(results)


def create_ablation_table(ablation_df):
    """Create ablation study table"""
    pivot = ablation_df.pivot_table(
        index='Dataset', columns='Method', values='R²', aggfunc='mean'
    )
    
    methods = ['MLP', 'Linear-MLP', 'Identity-MLP', 'NoAct-MLP', 'IdentityNoAct-MLP', 'RW-MLP']
    methods = [m for m in methods if m in pivot.columns]
    
    normalized = normalize_scores(pivot[methods])
    ranks = compute_ranks(pivot[methods])
    
    results = []
    for method in methods:
        scores = pivot[method].dropna()
        
        has_identity = 'Identity' in method or method == 'RW-MLP'
        has_noact = 'NoAct' in method or method == 'RW-MLP'
        has_wk = method == 'RW-MLP'
        
        results.append({
            'Method': method,
            'Identity Init': '✓' if has_identity else '',
            'No Activation': '✓' if has_noact else '',
            'W^k': '✓' if has_wk else '',
            'Mean R²': scores.mean(),
            'Std': scores.std(),
            'Norm R²': normalized[method].mean(),
            'Avg Rank': ranks[method].mean(),
        })
    
    return pd.DataFrame(results)


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def plot_main_figure(combined, ablation, output_dir):
    """Create Figure 4 style combined figure (6 panels)"""
    
    pivot = combined.pivot_table(index='dataset', columns='method', values='r2_mean', aggfunc='mean')
    normalized = normalize_scores(pivot)
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], wspace=0.3, hspace=0.35)
    
    # ========== Panel A: Normalized R² Bar Chart ==========
    ax_a = fig.add_subplot(gs[0, 0])
    
    mean_norm = normalized.mean().sort_values(ascending=False)
    std_norm = normalized.std()
    n = len(pivot)
    
    methods = mean_norm.index.tolist()[:10]
    values = mean_norm[methods].values
    errors = std_norm[methods].values / np.sqrt(n) * 1.96
    
    colors = [COLORS.get(m, '#666666') for m in methods]
    
    bars = ax_a.bar(range(len(methods)), values, yerr=errors, color=colors, 
                    alpha=0.8, capsize=3, error_kw={'linewidth': 0.8})
    
    ax_a.set_xticks(range(len(methods)))
    ax_a.set_xticklabels(methods, rotation=45, ha='right', fontsize=7)
    ax_a.set_ylabel('Normalized R² (↑)')
    ax_a.set_title('a  Performance Comparison (Top 10)', fontweight='bold', loc='left')
    ax_a.set_ylim(0, 1.05)
    
    for i, m in enumerate(methods):
        if 'RW-' in m:
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(1.5)
    
    # ========== Panel B: RW-MLP vs MLP Scatter ==========
    ax_b = fig.add_subplot(gs[0, 1])
    
    if 'RW-MLP' in pivot.columns and 'MLP' in pivot.columns:
        common = pivot[['MLP', 'RW-MLP']].dropna()
        x, y = common['MLP'].values, common['RW-MLP'].values
        
        colors_scatter = ['#1f77b4' if yi > xi else '#d62728' for xi, yi in zip(x, y)]
        ax_b.scatter(x, y, c=colors_scatter, alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
        
        lims = [min(min(x), min(y)) - 0.05, max(max(x), max(y)) + 0.05]
        ax_b.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
        
        ax_b.set_xlabel('MLP R²')
        ax_b.set_ylabel('RW-MLP R²')
        
        wins = (y > x).sum()
        total = len(x)
        try:
            stat, pval = stats.wilcoxon(y, x)
            pval_str = f'p={pval:.3f}' if pval > 0.001 else 'p<0.001'
        except:
            pval_str = ''
        
        ax_b.set_title(f'b  RW-MLP vs MLP ({wins}/{total} wins, {pval_str})', 
                      fontweight='bold', loc='left')
        
        ax_b.text(0.95, 0.05, 'MLP\nstronger', transform=ax_b.transAxes,
                 ha='right', va='bottom', fontsize=7, color='#d62728')
        ax_b.text(0.05, 0.95, 'RW-MLP\nstronger', transform=ax_b.transAxes,
                 ha='left', va='top', fontsize=7, color='#1f77b4')
    
    # ========== Panel C: RW-MLP vs CatBoost Scatter ==========
    ax_c = fig.add_subplot(gs[0, 2])
    
    if 'RW-MLP' in pivot.columns and 'CatBoost' in pivot.columns:
        common = pivot[['CatBoost', 'RW-MLP']].dropna()
        x, y = common['CatBoost'].values, common['RW-MLP'].values
        
        colors_scatter = ['#1f77b4' if yi > xi else '#2ca02c' for xi, yi in zip(x, y)]
        ax_c.scatter(x, y, c=colors_scatter, alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
        
        lims = [min(min(x), min(y)) - 0.05, max(max(x), max(y)) + 0.05]
        ax_c.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
        
        ax_c.set_xlabel('CatBoost R²')
        ax_c.set_ylabel('RW-MLP R²')
        
        wins = (y > x).sum()
        total = len(x)
        try:
            stat, pval = stats.wilcoxon(y, x)
            pval_str = f'p={pval:.3f}' if pval > 0.001 else 'p<0.001'
        except:
            pval_str = ''
        
        ax_c.set_title(f'c  RW-MLP vs CatBoost ({wins}/{total} wins, {pval_str})', 
                      fontweight='bold', loc='left')
        
        ax_c.text(0.95, 0.05, 'CatBoost\nstronger', transform=ax_c.transAxes,
                 ha='right', va='bottom', fontsize=7, color='#2ca02c')
        ax_c.text(0.05, 0.95, 'RW-MLP\nstronger', transform=ax_c.transAxes,
                 ha='left', va='top', fontsize=7, color='#1f77b4')
    
    # ========== Panel D: RW-Layer Effect by Architecture ==========
    ax_d = fig.add_subplot(gs[1, 0])
    
    comparisons = [
        ('RW-MLP', 'MLP'),
        ('RW-FT-Transformer', 'FT-Transformer'),
        ('RW-ResNet', 'RTDL-ResNet'),
        ('RW-NODE', 'NODE'),
        ('RW-TabNet', 'TabNet'),
    ]
    
    arch_names = []
    win_rates = []
    p_values = []
    
    for rw, base in comparisons:
        if rw in pivot.columns and base in pivot.columns:
            common = pivot[[rw, base]].dropna()
            wins = (common[rw] > common[base]).sum()
            total = len(common)
            
            try:
                stat, pval = stats.wilcoxon(common[rw], common[base])
            except:
                pval = 1.0
            
            arch_names.append(base.replace('RTDL-', '').replace('FT-', 'FT-'))
            win_rates.append(wins / total * 100)
            p_values.append(pval)
    
    colors_bar = ['#1f77b4' if wr > 50 else '#d62728' for wr in win_rates]
    
    bars = ax_d.barh(range(len(arch_names)), win_rates, color=colors_bar, alpha=0.8)
    ax_d.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
    
    ax_d.set_yticks(range(len(arch_names)))
    ax_d.set_yticklabels(arch_names)
    ax_d.set_xlabel('Win Rate of RW-variant (%)')
    ax_d.set_title('d  RW-Layer Effect by Architecture', fontweight='bold', loc='left')
    ax_d.set_xlim(0, 100)
    
    for i, (wr, pv) in enumerate(zip(win_rates, p_values)):
        sig = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else ''))
        ax_d.text(wr + 2, i, f'{wr:.1f}%{sig}', va='center', fontsize=7)
    
    # ========== Panel E: Ablation Study ==========
    ax_e = fig.add_subplot(gs[1, 1])
    
    ablation_pivot = ablation.pivot_table(index='Dataset', columns='Method', values='R²', aggfunc='mean')
    methods_order = ['Linear-MLP', 'Identity-MLP', 'NoAct-MLP', 'MLP', 'IdentityNoAct-MLP', 'RW-MLP']
    methods_order = [m for m in methods_order if m in ablation_pivot.columns]
    
    ablation_norm = normalize_scores(ablation_pivot[methods_order])
    means = ablation_norm.mean()
    stds = ablation_norm.std()
    
    colors_abl = [COLORS.get(m, '#666666') for m in methods_order]
    
    bars = ax_e.bar(range(len(methods_order)), means.values, 
                    yerr=stds.values / np.sqrt(len(ablation_pivot)) * 1.96,
                    color=colors_abl, alpha=0.8, capsize=3)
    
    ax_e.set_xticks(range(len(methods_order)))
    ax_e.set_xticklabels(methods_order, rotation=45, ha='right', fontsize=7)
    ax_e.set_ylabel('Normalized R²')
    ax_e.set_title('e  Ablation Study', fontweight='bold', loc='left')
    ax_e.set_ylim(0, 1.1)
    
    # ========== Panel F: Win Rate vs Linear-MLP ==========
    ax_f = fig.add_subplot(gs[1, 2])
    
    win_rates_abl = []
    methods_for_plot = []
    
    for method in methods_order:
        if method != 'Linear-MLP':
            common = ablation_pivot[['Linear-MLP', method]].dropna()
            wins = (common[method] > common['Linear-MLP']).sum()
            total = len(common)
            win_rates_abl.append(wins / total * 100)
            methods_for_plot.append(method)
    
    colors_abl_f = [COLORS.get(m, '#666666') for m in methods_for_plot]
    
    bars = ax_f.bar(range(len(methods_for_plot)), win_rates_abl, color=colors_abl_f, alpha=0.8)
    ax_f.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    
    ax_f.set_xticks(range(len(methods_for_plot)))
    ax_f.set_xticklabels(methods_for_plot, rotation=45, ha='right', fontsize=7)
    ax_f.set_ylabel('Win Rate vs Linear-MLP (%)')
    ax_f.set_title('f  Ablation Win Rates', fontweight='bold', loc='left')
    ax_f.set_ylim(0, 100)
    
    for i, v in enumerate(win_rates_abl):
        ax_f.text(i, v + 2, f'{v:.0f}%', ha='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_main_combined.png'), dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(output_dir, 'figure_main_combined.pdf'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✓ figure_main_combined.png/pdf")


def plot_individual_figures(pivot, ablation, output_dir):
    """Create individual figures"""
    normalized = normalize_scores(pivot)
    
    # Figure 1: Normalized performance bars
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean_norm = normalized.mean().sort_values(ascending=True)
    std_norm = normalized.std()
    n = len(pivot)
    
    methods = mean_norm.index.tolist()
    values = mean_norm.values
    errors = std_norm[methods].values / np.sqrt(n) * 1.96
    colors = [COLORS.get(m, '#666666') for m in methods]
    
    bars = ax.barh(range(len(methods)), values, xerr=errors, color=colors, 
                   alpha=0.8, capsize=3)
    
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel('Normalized R² (↑)')
    ax.set_title('Regression Performance Comparison', fontweight='bold')
    ax.set_xlim(0, 1.05)
    
    for i, m in enumerate(methods):
        if 'RW-' in m:
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_normalized_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig1_normalized_performance.png")
    
    # Figure 2: Rank comparison
    ranks = compute_ranks(pivot)
    mean_rank = ranks.mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = mean_rank.index.tolist()
    rank_values = mean_rank.values
    colors = [COLORS.get(m, '#666666') for m in methods]
    
    bars = ax.barh(range(len(methods)), rank_values, color=colors, alpha=0.8)
    
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel('Average Rank (↓ better)')
    ax.set_title('Method Rankings Across Datasets', fontweight='bold')
    
    for i, v in enumerate(rank_values):
        ax.text(v + 0.1, i, f'{v:.2f}', va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_rank_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig2_rank_comparison.png")
    
    # Figure 3-5: Scatter plots
    scatter_pairs = [
        ('RW-MLP', 'MLP', 'fig3_rwmlp_vs_mlp.png'),
        ('RW-MLP', 'Linear-MLP', 'fig4_rwmlp_vs_linear.png'),
        ('RW-MLP', 'CatBoost', 'fig5_rwmlp_vs_catboost.png'),
    ]
    
    for method1, method2, filename in scatter_pairs:
        if method1 in pivot.columns and method2 in pivot.columns:
            fig, ax = plt.subplots(figsize=(5, 5))
            
            common = pivot[[method1, method2]].dropna()
            x = common[method2].values
            y = common[method1].values
            
            colors_scatter = ['#1f77b4' if yi > xi else '#d62728' for xi, yi in zip(x, y)]
            ax.scatter(x, y, c=colors_scatter, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
            
            lims = [min(min(x), min(y)) - 0.05, max(max(x), max(y)) + 0.05]
            ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
            
            ax.set_xlabel(f'{method2} R²')
            ax.set_ylabel(f'{method1} R²')
            
            wins = (y > x).sum()
            total = len(x)
            try:
                stat, pval = stats.wilcoxon(y, x)
                pval_str = f'p={pval:.4f}' if pval > 0.0001 else 'p<0.0001'
            except:
                pval_str = ''
            
            ax.set_title(f'{method1} vs {method2}\nWin rate: {wins}/{total} ({wins/total*100:.1f}%), {pval_str}',
                        fontsize=9)
            
            ax.text(0.95, 0.05, f'{method2}\nstronger', transform=ax.transAxes,
                   ha='right', va='bottom', fontsize=8, color='#d62728')
            ax.text(0.05, 0.95, f'{method1}\nstronger', transform=ax.transAxes,
                   ha='left', va='top', fontsize=8, color='#1f77b4')
            
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ {filename}")
    
    # Figure 6: RW effect summary
    fig, ax = plt.subplots(figsize=(8, 5))
    
    comparisons = [
        ('RW-MLP', 'MLP'),
        ('RW-FT-Transformer', 'FT-Transformer'),
        ('RW-ResNet', 'RTDL-ResNet'),
        ('RW-NODE', 'NODE'),
        ('RW-TabNet', 'TabNet'),
    ]
    
    arch_names = []
    win_rates = []
    p_values = []
    
    for rw, base in comparisons:
        if rw in pivot.columns and base in pivot.columns:
            common = pivot[[rw, base]].dropna()
            wins = (common[rw] > common[base]).sum()
            total = len(common)
            
            try:
                stat, pval = stats.wilcoxon(common[rw], common[base])
            except:
                pval = 1.0
            
            arch_names.append(base.replace('RTDL-', '').replace('FT-', 'FT-'))
            win_rates.append(wins / total * 100)
            p_values.append(pval)
    
    colors_bar = ['#1f77b4' if wr > 50 else '#d62728' for wr in win_rates]
    
    bars = ax.barh(range(len(arch_names)), win_rates, color=colors_bar, alpha=0.8)
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
    
    ax.set_yticks(range(len(arch_names)))
    ax.set_yticklabels(arch_names)
    ax.set_xlabel('Win Rate of RW-variant (%)')
    ax.set_title('RW-Layer Effect by Architecture', fontweight='bold')
    ax.set_xlim(0, 100)
    
    for i, (wr, pv) in enumerate(zip(win_rates, p_values)):
        sig = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else ''))
        ax.text(wr + 2, i, f'{wr:.1f}%{sig}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_rw_effect_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig6_rw_effect_summary.png")
    
    # Figure 7: Ablation study
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ablation_pivot = ablation.pivot_table(index='Dataset', columns='Method', values='R²', aggfunc='mean')
    methods_order = ['Linear-MLP', 'Identity-MLP', 'NoAct-MLP', 'MLP', 'IdentityNoAct-MLP', 'RW-MLP']
    methods_order = [m for m in methods_order if m in ablation_pivot.columns]
    
    # Panel a
    ax = axes[0]
    ablation_norm = normalize_scores(ablation_pivot[methods_order])
    means = ablation_norm.mean()
    stds = ablation_norm.std()
    
    colors_abl = [COLORS.get(m, '#666666') for m in methods_order]
    
    bars = ax.bar(range(len(methods_order)), means.values, 
                  yerr=stds.values / np.sqrt(len(ablation_pivot)) * 1.96,
                  color=colors_abl, alpha=0.8, capsize=4)
    
    ax.set_xticks(range(len(methods_order)))
    ax.set_xticklabels(methods_order, rotation=45, ha='right')
    ax.set_ylabel('Normalized R²')
    ax.set_title('a) Ablation Study: Component Effects', fontweight='bold')
    ax.set_ylim(0, 1.1)
    
    # Panel b
    ax = axes[1]
    
    win_rates_abl = []
    methods_for_plot = []
    for method in methods_order:
        if method != 'Linear-MLP':
            common = ablation_pivot[['Linear-MLP', method]].dropna()
            wins = (common[method] > common['Linear-MLP']).sum()
            total = len(common)
            win_rates_abl.append(wins / total * 100)
            methods_for_plot.append(method)
    
    colors_plot = [COLORS.get(m, '#666666') for m in methods_for_plot]
    
    bars = ax.bar(range(len(methods_for_plot)), win_rates_abl, color=colors_plot, alpha=0.8)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    
    ax.set_xticks(range(len(methods_for_plot)))
    ax.set_xticklabels(methods_for_plot, rotation=45, ha='right')
    ax.set_ylabel('Win Rate vs Linear-MLP (%)')
    ax.set_title('b) Win Rate Comparison', fontweight='bold')
    ax.set_ylim(0, 100)
    
    for i, v in enumerate(win_rates_abl):
        ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig7_ablation_study.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ fig7_ablation_study.png")


def create_extended_table_image(results_df, output_dir):
    """Create Extended Data Table visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    cell_text = []
    for _, row in results_df.iterrows():
        cell_text.append([
            row['Method'],
            f"{row['Mean R²']:.4f} ± {row['Std R²']:.4f}",
            f"{row['Norm R² (↑)']:.3f} ± {row['Norm Std']:.3f}",
            f"{row['Avg Rank (↓)']:.2f}",
            str(int(row['N']))
        ])
    
    columns = ['Method', 'Mean R² (↑)', 'Norm R² (↑)', 'Avg Rank (↓)', 'N']
    
    table = ax.table(cellText=cell_text, colLabels=columns, loc='center',
                     cellLoc='center', colColours=['#f0f0f0']*5)
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    for i in range(len(columns)):
        table[(0, i)].set_text_props(fontweight='bold')
    
    for i in range(1, 4):
        for j in range(len(columns)):
            table[(i, j)].set_facecolor('#e6f3ff')
    
    plt.title('Extended Data Table: Aggregated Regression Results',
              fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'extended_data_table.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ extended_data_table.png")


def generate_latex_table(results_df, output_dir):
    """Generate LaTeX table"""
    latex = r"""\begin{table}[htbp]
\centering
\caption{Aggregated results on regression benchmark datasets. Scores are normalized per dataset (0=worst, 1=best).}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Mean R²} & \textbf{Norm R² ($\uparrow$)} & \textbf{Avg Rank ($\downarrow$)} & \textbf{N} \\
\midrule
"""
    
    for _, row in results_df.iterrows():
        method = row['Method'].replace('_', r'\_')
        latex += f"{method} & {row['Mean R²']:.4f} & {row['Norm R² (↑)']:.3f} & {row['Avg Rank (↓)']:.2f} & {int(row['N'])} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(os.path.join(output_dir, 'table_latex.tex'), 'w') as f:
        f.write(latex)
    
    print(f"  ✓ table_latex.tex")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("RW-LAYER BENCHMARK ANALYSIS")
    print("Nature-style Tables and Figures")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    detailed, neural_rw, linear_mlp, ablation = load_data()
    
    # Combine results
    print("\n2. Combining results...")
    combined = combine_all_results(detailed, neural_rw, linear_mlp)
    print(f"   Combined: {len(combined)} rows, {combined['method'].nunique()} methods, {combined['dataset'].nunique()} datasets")
    
    # Create main results table
    print("\n3. Creating tables...")
    results_df, pivot = create_main_results_table(combined)
    
    print("\n" + "=" * 70)
    print("MAIN RESULTS TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # Save tables
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'main_results_summary.csv'), index=False)
    print(f"\n  ✓ main_results_summary.csv")
    
    # RW effect table
    rw_effect = create_rw_effect_table(pivot)
    print("\n" + "=" * 70)
    print("RW-LAYER EFFECT TABLE")
    print("=" * 70)
    print(rw_effect.to_string(index=False))
    rw_effect.to_csv(os.path.join(OUTPUT_DIR, 'rw_effect_table.csv'), index=False)
    print(f"\n  ✓ rw_effect_table.csv")
    
    # Ablation table
    ablation_table = create_ablation_table(ablation)
    print("\n" + "=" * 70)
    print("ABLATION STUDY TABLE")
    print("=" * 70)
    print(ablation_table.to_string(index=False))
    ablation_table.to_csv(os.path.join(OUTPUT_DIR, 'ablation_table.csv'), index=False)
    print(f"\n  ✓ ablation_table.csv")
    
    # Key comparisons
    print("\n" + "=" * 70)
    print("KEY PAIRWISE COMPARISONS")
    print("=" * 70)
    
    key_comparisons = [
        ('RW-MLP', 'MLP'),
        ('RW-MLP', 'Linear-MLP'),
        ('RW-MLP', 'CatBoost'),
        ('RW-FT-Transformer', 'FT-Transformer'),
    ]
    
    for m1, m2 in key_comparisons:
        if m1 in pivot.columns and m2 in pivot.columns:
            result = create_pairwise_comparison(pivot, m1, m2)
            sig = '***' if result['p_value'] < 0.001 else ('**' if result['p_value'] < 0.01 else ('*' if result['p_value'] < 0.05 else ''))
            print(f"{m1} vs {m2}: {result['wins']}/{result['total']} wins ({result['win_rate']:.1f}%), "
                  f"Δ={result['mean_diff']:+.4f}, p={result['p_value']:.4f}{sig}")
    
    # Create figures
    print("\n4. Creating figures...")
    plot_main_figure(combined, ablation, OUTPUT_DIR)
    plot_individual_figures(pivot, ablation, OUTPUT_DIR)
    create_extended_table_image(results_df, OUTPUT_DIR)
    generate_latex_table(results_df, OUTPUT_DIR)
    
    # Save full pivot table
    pivot.to_csv(os.path.join(OUTPUT_DIR, 'full_pivot_table.csv'))
    print(f"  ✓ full_pivot_table.csv")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n🏆 Top 5 Methods by Average Rank:")
    for i, (_, row) in enumerate(results_df.head(5).iterrows()):
        print(f"   {i+1}. {row['Method']}: Rank {row['Avg Rank (↓)']:.2f}, Norm R² {row['Norm R² (↑)']:.3f}")
    
    print(f"\n📊 RW-Layer Win Rates:")
    for _, row in rw_effect.iterrows():
        status = "✅ Helps" if row['win_rate'] > 50 else "❌ Hurts"
        print(f"   {row['RW Method']} vs {row['Base Method']}: {row['win_rate']:.1f}% {status}")
    
    print("\n" + "=" * 70)
    print(f"Analysis complete! Files saved to {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
