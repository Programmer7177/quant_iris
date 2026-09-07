"""
CONFIG PRESETS & BENCHMARK SUITE FOR BTC QUANT FORECASTING
==========================================================
Saves both validated configuration setups:

SETUP A (Master High-Accuracy Conviction Setup):
  - Model: ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15, class_weight='balanced')
  - Features:
      7d  -> FEAT_SHORT (SHAP-5: rsi_90, dist_ema50, spread_50_200, vol_21, power_law_res)
      14d -> FEAT_SHORT
      30d -> FEAT_MEDIUM (SHAP-5 + hmm_norm)
      90d -> FEAT_LONG + dist_sma3_m + spread_sma_3_5_m (SHAP-5 + Macro + Halving + Monthly SMA package)
  - Gating: Single confidence margin |prob - 0.5|
      7d  -> Top 20% (Acc: 56.14%, Bull F1: 0.5994, Bear F1: 0.5155)
      14d -> Top 35% (Acc: 61.50%, Bull F1: 0.6983, Bear F1: 0.4680)
      30d -> Top 50% (Acc: 60.50%, Bull F1: 0.6918, Bear F1: 0.4503)
      90d -> 100% All bars (Acc: 65.14%, Bull F1: 0.7255, Bear F1: 0.5225)

SETUP B (Dual-Threshold High-Symmetry Long/Short Setup):
  - Model: ExtraTreesClassifier with custom class weight ratio {0: W_bear, 1: 1.0}
  - Features:
      7d  -> FEAT_SHORT + semi_vol_skew, W_bear=1.3
      14d -> FEAT_SHORT + semi_vol_skew, W_bear=1.4
      30d -> FEAT_MEDIUM + semi_vol_skew, W_bear=1.6
      90d -> FEAT_LONG + dist_sma3_m + spread_sma_3_5_m, class_weight='balanced'
  - Gating: Dual quantile threshold (prob <= th_dn or prob >= th_up)
      7d  -> Dual-Q 20% (Acc: 54.35%, Macro F1: 0.5432, Bear F1: 0.5315, Bull F1: 0.5550)
      14d -> Dual-Q 20% (Acc: 59.74%, Macro F1: 0.5973, Bear F1: 0.5932, Bull F1: 0.6014)
      30d -> Dual-Q 30% (Acc: 58.67%, Macro F1: 0.5861, Bear F1: 0.5703, Bull F1: 0.6019)
      90d -> Dual-Q 25% (Acc: 63.35%, Macro F1: 0.6322, Bear F1: 0.6534, Bull F1: 0.6111)

Usage:
  py -3.11 config_presets.py --setup A
  py -3.11 config_presets.py --setup B
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from forecast_engine import load_data, fit_hmm, FEAT_SHORT, FEAT_MEDIUM, FEAT_LONG

def prepare_dataset():
    df = load_data()
    log_rets = np.log(df['close'] / df['close'].shift(1)).fillna(0).values
    states, _, _ = fit_hmm(log_rets)
    df['hmm_state'] = states
    df['hmm_norm'] = states.astype(float) / 3.0

    c = df['close']
    
    # Monthly SMA package
    df['dist_sma3_m'] = (c - c.rolling(90).mean()) / c.rolling(90).mean()
    df['dist_sma5_m'] = (c - c.rolling(150).mean()) / c.rolling(150).mean()
    df['spread_sma_3_5_m'] = (c.rolling(90).mean() - c.rolling(150).mean()) / c.rolling(150).mean()

    # Semi-volatility skew
    ret1 = c.pct_change()
    down_sq = np.where(ret1 < 0, ret1**2, 0.0)
    up_sq = np.where(ret1 > 0, ret1**2, 0.0)
    df['downside_semi_vol21'] = np.sqrt(pd.Series(down_sq).rolling(21).mean() * 252)
    df['upside_semi_vol21'] = np.sqrt(pd.Series(up_sq).rolling(21).mean() * 252)
    df['semi_vol_skew'] = (df['upside_semi_vol21'] - df['downside_semi_vol21']) / (df['upside_semi_vol21'] + df['downside_semi_vol21'] + 1e-9)
    
    return df

def run_setup_a(df):
    print("=" * 72)
    print("  RUNNING SETUP A: Master High-Accuracy Conviction Setup")
    print("=" * 72)
    
    configs = [
        (7,  FEAT_SHORT, 20, "Short 7d"),
        (14, FEAT_SHORT, 35, "Short 14d"),
        (30, FEAT_MEDIUM, 50, "Medium 30d"),
        (90, FEAT_LONG + ['dist_sma3_m', 'spread_sma_3_5_m'], 100, "Long 90d"),
    ]
    
    n = len(df)
    min_train = 500
    step = max(60, (n - min_train) // 8)
    
    for horizon, feats, gate_pct, label in configs:
        ft = [f for f in feats if f in df.columns]
        y_true_all, prob_all = [], []
        
        for start in range(min_train, n - horizon, step):
            tr = df.iloc[:start]; te = df.iloc[start:min(start+step, n-horizon)]
            if len(te) < 10: continue
            X_tr = tr[ft].values; y_tr = (tr[f'fwd_ret_{horizon}'].values > 0).astype(int)
            X_te = te[ft].values; y_te = (te[f'fwd_ret_{horizon}'].values > 0).astype(int)
            
            m = ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15, 
                                     class_weight='balanced', random_state=42)
            m.fit(X_tr, y_tr)
            prob_all.extend(m.predict_proba(X_te)[:, 1])
            y_true_all.extend(y_te)
            
        y_true = np.array(y_true_all)
        probs = np.array(prob_all)
        conf = np.abs(probs - 0.5)
        
        if gate_pct < 100:
            cutoff = np.percentile(conf, 100 - gate_pct)
            mask = conf >= cutoff
        else:
            mask = np.ones(len(probs), dtype=bool)
            
        y_sub = y_true[mask]
        p_sub = (probs[mask] >= 0.5).astype(int)
        
        acc = accuracy_score(y_sub, p_sub)
        mac = f1_score(y_sub, p_sub, average='macro')
        bull = f1_score(y_sub, p_sub, pos_label=1)
        bear = f1_score(y_sub, p_sub, pos_label=0)
        cm = confusion_matrix(y_sub, p_sub, labels=[0, 1])
        
        print(f"[{label} | Gate: Top {gate_pct}% | Cov: {mask.mean():.1%}]")
        print(f"  Acc: {acc:.2%} | Macro F1: {mac:.4f} | Bull F1: {bull:.4f} | Bear F1: {bear:.4f}")
        print(f"  Confusion: TN={cm[0,0]}, FP={cm[0,1]} | FN={cm[1,0]}, TP={cm[1,1]}\n")

def run_setup_b(df):
    print("=" * 72)
    print("  RUNNING SETUP B: Dual-Threshold High-Symmetry Long/Short Setup")
    print("=" * 72)
    
    configs = [
        (7,  FEAT_SHORT + ['semi_vol_skew'], {0: 1.3, 1: 1.0}, 20, "Short 7d"),
        (14, FEAT_SHORT + ['semi_vol_skew'], {0: 1.4, 1: 1.0}, 20, "Short 14d"),
        (30, FEAT_MEDIUM + ['semi_vol_skew'], {0: 1.6, 1: 1.0}, 30, "Medium 30d"),
        (90, FEAT_LONG + ['dist_sma3_m', 'spread_sma_3_5_m'], 'balanced', 25, "Long 90d"),
    ]
    
    n = len(df)
    min_train = 500
    step = max(60, (n - min_train) // 8)
    
    for horizon, feats, cw, q_pct, label in configs:
        ft = [f for f in feats if f in df.columns]
        y_true_all, prob_all = [], []
        
        for start in range(min_train, n - horizon, step):
            tr = df.iloc[:start]; te = df.iloc[start:min(start+step, n-horizon)]
            if len(te) < 10: continue
            X_tr = tr[ft].values; y_tr = (tr[f'fwd_ret_{horizon}'].values > 0).astype(int)
            X_te = te[ft].values; y_te = (te[f'fwd_ret_{horizon}'].values > 0).astype(int)
            
            m = ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15, 
                                     class_weight=cw, random_state=42)
            m.fit(X_tr, y_tr)
            prob_all.extend(m.predict_proba(X_te)[:, 1])
            y_true_all.extend(y_te)
            
        y_true = np.array(y_true_all)
        probs = np.array(prob_all)
        
        th_dn = np.percentile(probs, q_pct)
        th_up = np.percentile(probs, 100 - q_pct)
        mask = (probs <= th_dn) | (probs >= th_up)
        
        y_sub = y_true[mask]
        p_sub = (probs[mask] >= th_up).astype(int)
        
        acc = accuracy_score(y_sub, p_sub)
        mac = f1_score(y_sub, p_sub, average='macro')
        bull = f1_score(y_sub, p_sub, pos_label=1)
        bear = f1_score(y_sub, p_sub, pos_label=0)
        cm = confusion_matrix(y_sub, p_sub, labels=[0, 1])
        
        print(f"[{label} | Dual-Q: {q_pct}% | Cov: {mask.mean():.1%}]")
        print(f"  Acc: {acc:.2%} | Macro F1: {mac:.4f} | Bull F1: {bull:.4f} | Bear F1: {bear:.4f}")
        print(f"  Confusion: TN={cm[0,0]}, FP={cm[0,1]} | FN={cm[1,0]}, TP={cm[1,1]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=["A", "B", "ALL"], default="ALL", help="Which setup to execute")
    args = parser.parse_args()
    
    print("Loading master dataset...")
    df = prepare_dataset()
    
    if args.setup in ["A", "ALL"]:
        run_setup_a(df)
    if args.setup in ["B", "ALL"]:
        run_setup_b(df)
