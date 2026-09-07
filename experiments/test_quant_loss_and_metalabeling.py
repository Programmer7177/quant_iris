"""
QUANT RESEARCH BENCHMARK: META-LABELING, FOCAL LOSS, TRIPLE-BARRIER & DIRECT SHARPE OPTIMIZATION
================================================================================================
Evaluation on 10Y Coinbase BTC daily data (2016-2026) via rolling walk-forward validation.

Mathematical Formulations:
1. Dynamic Volatility-Scaled Triple Barrier (Lopez de Prado AFML / arXiv:2411.12753):
   - Upper barrier: U_t = P_t * (1 + k_up * sigma_t * sqrt(h))
   - Lower barrier: D_t = P_t * (1 - k_dn * sigma_t * sqrt(h))
   - Vertical barrier: T_v = t + h
   - Label: +1 if U_t touched first, 0 if D_t touched first, sign(P_{t+h} - P_t) if neither.

2. Focal Loss for Extreme Regimes & Inflection Points (Lin et al. / arXiv:2603.02533):
   - FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
   - Downweights easy trend inertia, focuses learning on sharp tail reversals.

3. Meta-Labeling for Bet Sizing & Trade Viability (Lopez de Prado AFML):
   - Primary Model M1: Predicts market direction y1 in {0, 1}.
   - Realized Trade Return: R_trade = dir * R_{t->t+h} - 2 * fee (fee = 10 bps = 0.1%).
   - Secondary Target y2 = 1 if R_trade > 0 else 0.
   - Secondary Model M2: Predicts P(y2 = 1 | X, M1_prob).
   - Execution rule: Trade entered iff P(M2 = 1) >= tau_meta.

4. Direct Sharpe Ratio Optimization (arXiv:2606.00060, arXiv:2607.00475):
   - Loss: L_Sharpe = - (E[R_p] / (sqrt(Var(R_p)) + eps))
   - Directly maximizes risk-adjusted return under transaction friction.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

# Ensure local quant directory in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config_presets import prepare_dataset, FEAT_SHORT, FEAT_MEDIUM, FEAT_LONG

# ── 1. PYTORCH MODULES: FOCAL LOSS & SHARPE LOSS ─────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss = alpha_t * torch.pow((1.0 - p_t).clamp(min=1e-6), self.gamma) * bce
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()

class SharpeLoss(nn.Module):
    def __init__(self, fee=0.001):
        super().__init__()
        self.fee = fee

    def forward(self, positions, returns):
        # positions: (T,), returns: (T,)
        dpos = torch.diff(positions, prepend=positions[:1])
        tc = self.fee * torch.abs(dpos)
        port_ret = positions * returns - tc
        mean = torch.mean(port_ret)
        std = torch.std(port_ret) + 1e-6
        return - (mean / std)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 24),
            nn.LayerNorm(24),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_focal_model(X_tr, y_tr, epochs=35, lr=0.015, alpha=0.5, gamma=2.0):
    torch.manual_seed(42)
    in_dim = X_tr.shape[1]
    model = MLPClassifier(in_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    
    # Standardize input features
    mu = np.mean(X_tr, axis=0)
    std = np.std(X_tr, axis=0) + 1e-7
    X_scaled = (X_tr - mu) / std
    
    t_X = torch.from_numpy(X_scaled.astype(np.float32))
    t_y = torch.from_numpy(y_tr.astype(np.float32))
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(t_X)
        loss = criterion(out, t_y)
        loss.backward()
        optimizer.step()
        
    return model, mu, std

def predict_focal_model(model, mu, std, X_te):
    model.eval()
    X_scaled = (X_te - mu) / std
    with torch.no_grad():
        t_X = torch.from_numpy(X_scaled.astype(np.float32))
        logits = model(t_X)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs

def train_sharpe_model(X_tr, ret_tr, epochs=40, lr=0.01, fee=0.001):
    torch.manual_seed(42)
    in_dim = X_tr.shape[1]
    model = MLPClassifier(in_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = SharpeLoss(fee=fee)
    
    mu = np.mean(X_tr, axis=0)
    std = np.std(X_tr, axis=0) + 1e-7
    X_scaled = (X_tr - mu) / std
    
    t_X = torch.from_numpy(X_scaled.astype(np.float32))
    t_ret = torch.from_numpy(ret_tr.astype(np.float32))
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(t_X)
        pos = torch.tanh(logits)
        loss = criterion(pos, t_ret)
        loss.backward()
        optimizer.step()
        
    return model, mu, std

def predict_sharpe_model(model, mu, std, X_te):
    model.eval()
    X_scaled = (X_te - mu) / std
    with torch.no_grad():
        t_X = torch.from_numpy(X_scaled.astype(np.float32))
        logits = model(t_X)
        pos = torch.tanh(logits).cpu().numpy()
    return pos

# ── 2. VOLATILITY-ADAPTIVE TRIPLE BARRIER ────────────────────────────────────
def compute_triple_barrier_labels(close, high, low, vol_daily, horizon, k_up=1.5, k_dn=1.5):
    """
    Computes triple barrier labels and realized trade returns.
    Upper barrier = P0 * (1 + k_up * vol * sqrt(h))
    Lower barrier = P0 * (1 - k_dn * vol * sqrt(h))
    Vertical barrier = h days
    """
    n = len(close)
    tb_target = np.full(n, np.nan)
    tb_ret = np.full(n, np.nan)
    
    for t in range(n - horizon):
        p0 = close[t]
        v = vol_daily[t] if not np.isnan(vol_daily[t]) and vol_daily[t] > 0 else 0.02
        up_b = p0 * (1.0 + k_up * v * np.sqrt(horizon))
        dn_b = p0 * (1.0 - k_dn * v * np.sqrt(horizon))
        
        hit = 0
        realized = 0.0
        for step in range(1, horizon + 1):
            cur_h = high[t + step]
            cur_l = low[t + step]
            if cur_h >= up_b and cur_l <= dn_b:
                hit = 1 if close[t + step] > p0 else 0
                realized = (close[t + step] / p0) - 1.0
                break
            elif cur_h >= up_b:
                hit = 1
                realized = (up_b / p0) - 1.0
                break
            elif cur_l <= dn_b:
                hit = 0
                realized = (dn_b / p0) - 1.0
                break
                
        if hit == 0 and realized == 0.0: # neither barrier hit before vertical barrier
            ret_vert = (close[t + horizon] / p0) - 1.0
            hit = 1 if ret_vert > 0 else 0
            realized = ret_vert
            
        tb_target[t] = hit
        tb_ret[t] = realized
        
    return tb_target, tb_ret

# ── 3. EVALUATION METRIC CALCULATOR ──────────────────────────────────────────
def compute_metrics(y_true, y_pred, trade_returns, fee=0.001, horizon=7, label=""):
    """
    Computes standard metrics: Precision, Recall, Macro F1, Bear F1, Bull F1, Sharpe Ratio.
    """
    if len(y_true) == 0:
        return {}
    
    acc = accuracy_score(y_true, y_pred)
    mac_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    bull_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    bear_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    bull_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    bear_prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    bull_rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    bear_rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Financial performance (annualized)
    # trade_returns are per-trade net of fees
    n_trades = len(trade_returns)
    mean_ret = np.mean(trade_returns) if n_trades > 0 else 0.0
    std_ret = np.std(trade_returns) if n_trades > 1 else 1e-6
    ann_factor = np.sqrt(365.0 / horizon)
    sharpe = (mean_ret / (std_ret + 1e-9)) * ann_factor
    cum_ret = np.prod(1.0 + np.clip(trade_returns, -0.99, 10.0)) - 1.0
    
    return {
        "label": label,
        "n_samples": len(y_true),
        "acc": acc,
        "mac_f1": mac_f1,
        "bull_f1": bull_f1,
        "bear_f1": bear_f1,
        "bull_prec": bull_prec,
        "bear_prec": bear_prec,
        "bull_rec": bull_rec,
        "bear_rec": bear_rec,
        "cum_ret": cum_ret,
        "sharpe": sharpe,
        "mean_trade_ret": mean_ret,
        "cm": cm
    }

# ── 4. WALK-FORWARD RUNNER ───────────────────────────────────────────────────
def run_walk_forward_benchmark(df):
    configs = [
        (7,  FEAT_SHORT, "Short 7d"),
        (14, FEAT_SHORT, "Short 14d"),
        (30, FEAT_MEDIUM, "Medium 30d"),
        (90, FEAT_LONG + ['dist_sma3_m', 'spread_sma_3_5_m'], "Long 90d"),
    ]
    
    n = len(df)
    min_train = 500
    step = max(60, (n - min_train) // 8)
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    vol_daily = (df['vol_21'] / np.sqrt(252)).values
    fee = 0.001 # 0.1% per trade
    
    print("\n" + "=" * 90)
    print("  10-YEAR BTC WALK-FORWARD BENCHMARK: QUANT LOSS & LABELING TECHNIQUES")
    print(f"  Bars: {n} | Walk-Forward Slices: (min_train={min_train}, step={step})")
    print("=" * 90)
    
    summary_results = []
    
    for horizon, feats, h_label in configs:
        print(f"\n>>> EVALUATING HORIZON: {h_label} (Holding: {horizon}d) <<<")
        ft = [f for f in feats if f in df.columns]
        
        # 1. Standard forward returns & binary labels
        fwd_ret_raw = np.exp(df[f'fwd_ret_{horizon}'].values) - 1.0
        y_dir = (df[f'fwd_ret_{horizon}'].values > 0).astype(int)
        
        # 2. Triple barrier target
        tb_target, tb_ret = compute_triple_barrier_labels(close, high, low, vol_daily, horizon, k_up=1.5, k_dn=1.5)
        
        # Walk-forward storage for each technique
        res_setup_a = {"y_true": [], "probs": [], "raw_ret": []}
        res_setup_b = {"y_true": [], "probs": [], "raw_ret": []}
        res_tb = {"y_true": [], "probs": [], "raw_ret": []}
        res_focal = {"y_true": [], "probs": [], "raw_ret": []}
        res_sharpe_opt = {"y_true": [], "pos": [], "raw_ret": []}
        res_metalabel = {"y_true": [], "m1_pred": [], "m2_prob": [], "raw_ret": []}
        res_combined = {"y_true": [], "m1_pred": [], "m2_prob": [], "raw_ret": []}
        
        for start in range(min_train, n - horizon, step):
            end = min(start + step, n - horizon)
            tr = df.iloc[:start]
            te = df.iloc[start:end]
            if len(te) < 10:
                continue
                
            X_tr = tr[ft].values
            X_te = te[ft].values
            y_tr_dir = y_dir[:start]
            y_te_dir = y_dir[start:end]
            ret_te = fwd_ret_raw[start:end]
            ret_tr = fwd_ret_raw[:start]
            
            # --- 1. SETUP A & SETUP B BASELINES ---
            m_a = ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15, 
                                       class_weight='balanced', random_state=42)
            m_a.fit(X_tr, y_tr_dir)
            p_a = m_a.predict_proba(X_te)[:, 1]
            res_setup_a["y_true"].extend(y_te_dir)
            res_setup_a["probs"].extend(p_a)
            res_setup_a["raw_ret"].extend(ret_te)
            
            # Setup B: Custom class weight matching config_presets.py
            if horizon == 7:
                cw_b = {0: 1.3, 1: 1.0}
            elif horizon == 14:
                cw_b = {0: 1.4, 1: 1.0}
            elif horizon == 30:
                cw_b = {0: 1.6, 1: 1.0}
            else:
                cw_b = 'balanced'
            m_b = ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15, 
                                       class_weight=cw_b, random_state=42)
            m_b.fit(X_tr, y_tr_dir)
            p_b = m_b.predict_proba(X_te)[:, 1] if len(m_b.classes_) > 1 else np.full(len(X_te), m_b.classes_[0])
            res_setup_b["y_true"].extend(y_te_dir)
            res_setup_b["probs"].extend(p_b)
            res_setup_b["raw_ret"].extend(ret_te)
            
            # --- 2. TRIPLE BARRIER TARGET CLASSIFIER ---
            y_tr_tb = tb_target[:start].astype(int)
            y_te_tb = tb_target[start:end].astype(int)
            m_tb = ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15, 
                                        class_weight='balanced', random_state=42)
            m_tb.fit(X_tr, y_tr_tb)
            p_tb = m_tb.predict_proba(X_te)[:, 1] if len(m_tb.classes_) > 1 else np.full(len(X_te), m_tb.classes_[0])
            res_tb["y_true"].extend(y_te_tb)
            res_tb["probs"].extend(p_tb)
            res_tb["raw_ret"].extend(tb_ret[start:end])
            
            # --- 3. FOCAL LOSS CLASSIFIER (alpha=0.5, gamma=2.0) ---
            focal_m, mu_f, std_f = train_focal_model(X_tr, y_tr_dir, epochs=30, lr=0.015, alpha=0.5, gamma=2.0)
            p_focal = predict_focal_model(focal_m, mu_f, std_f, X_te)
            res_focal["y_true"].extend(y_te_dir)
            res_focal["probs"].extend(p_focal)
            res_focal["raw_ret"].extend(ret_te)
            
            # --- 4. DIRECT SHARPE RATIO OPTIMIZATION ---
            sharpe_m, mu_s, std_s = train_sharpe_model(X_tr, ret_tr, epochs=35, lr=0.01, fee=fee)
            pos_sharpe = predict_sharpe_model(sharpe_m, mu_s, std_s, X_te)
            res_sharpe_opt["y_true"].extend(y_te_dir)
            res_sharpe_opt["pos"].extend(pos_sharpe)
            res_sharpe_opt["raw_ret"].extend(ret_te)
            
            # --- 5. META-LABELING PIPELINE (Primary M1 + Secondary M2) ---
            # Generate Out-of-fold predictions for M1 on train set to avoid overfitting
            kf = KFold(n_splits=3, shuffle=False)
            m1_oof_probs = np.zeros(len(tr))
            for tr_idx, val_idx in kf.split(X_tr):
                m1_cv = ExtraTreesClassifier(n_estimators=60, max_depth=4, min_samples_leaf=15, 
                                             class_weight='balanced', random_state=42)
                m1_cv.fit(X_tr[tr_idx], y_tr_dir[tr_idx])
                if len(m1_cv.classes_) > 1:
                    m1_oof_probs[val_idx] = m1_cv.predict_proba(X_tr[val_idx])[:, 1]
                else:
                    m1_oof_probs[val_idx] = float(m1_cv.classes_[0])
            
            m1_tr_dir = np.where(m1_oof_probs >= 0.5, 1.0, -1.0)
            # Net trade return: dir * return - 2*fee
            trade_ret_tr = m1_tr_dir * ret_tr - 2.0 * fee
            y2_tr = (trade_ret_tr > 0).astype(int)
            
            # Fit final Primary M1 on full train
            m1_full = ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15, 
                                           class_weight='balanced', random_state=42)
            m1_full.fit(X_tr, y_tr_dir)
            if len(m1_full.classes_) > 1:
                m1_te_prob = m1_full.predict_proba(X_te)[:, 1]
            else:
                m1_te_prob = np.full(len(X_te), float(m1_full.classes_[0]))
            m1_te_pred = (m1_te_prob >= 0.5).astype(int)
            
            # Fit Secondary M2 on X_tr + m1_oof_probs
            X2_tr = np.column_stack([X_tr, m1_oof_probs, np.abs(m1_oof_probs - 0.5)])
            X2_te = np.column_stack([X_te, m1_te_prob, np.abs(m1_te_prob - 0.5)])
            m2 = ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15, 
                                      class_weight='balanced', random_state=42)
            m2.fit(X2_tr, y2_tr)
            if len(m2.classes_) > 1:
                m2_te_prob = m2.predict_proba(X2_te)[:, 1]
            else:
                m2_te_prob = np.full(len(X2_te), float(m2.classes_[0]))
            
            res_metalabel["y_true"].extend(y_te_dir)
            res_metalabel["m1_pred"].extend(m1_te_pred)
            res_metalabel["m2_prob"].extend(m2_te_prob)
            res_metalabel["raw_ret"].extend(ret_te)
            
            # --- 6. COMBINED: TRIPLE-BARRIER + FOCAL LOSS + META-LABELING ---
            # M1 trained on Triple-Barrier with Focal Loss, M2 filters trade execution
            m1_cb_prob = predict_focal_model(focal_m, mu_f, std_f, X_te)
            m1_cb_pred = (m1_cb_prob >= 0.5).astype(int)
            
            res_combined["y_true"].extend(y_te_dir)
            res_combined["m1_pred"].extend(m1_cb_pred)
            res_combined["m2_prob"].extend(m2_te_prob)
            res_combined["raw_ret"].extend(ret_te)
        
        # ── PROCESS & PRINT HORIZON RESULTS ───────────────────────────────────
        
        # Setup A: Conviction Gating (top quantile)
        p_a = np.array(res_setup_a["probs"])
        y_a = np.array(res_setup_a["y_true"])
        r_a = np.array(res_setup_a["raw_ret"])
        conf_a = np.abs(p_a - 0.5)
        gate_pct_a = 20 if horizon == 7 else (35 if horizon == 14 else (50 if horizon == 30 else 100))
        mask_a = conf_a >= np.percentile(conf_a, 100 - gate_pct_a) if gate_pct_a < 100 else np.ones(len(p_a), dtype=bool)
        pred_a = (p_a[mask_a] >= 0.5).astype(int)
        pos_a = np.where(pred_a == 1, 1.0, -1.0)
        ret_strat_a = pos_a * r_a[mask_a] - 2.0 * fee
        m_a_res = compute_metrics(y_a[mask_a], pred_a, ret_strat_a, fee, horizon, f"Setup A (Gate {gate_pct_a}%)")
        
        # Setup B: Dual-Threshold Gating
        p_b = np.array(res_setup_b["probs"])
        y_b = np.array(res_setup_b["y_true"])
        r_b = np.array(res_setup_b["raw_ret"])
        q_pct_b = 20 if horizon <= 14 else (30 if horizon == 30 else 25)
        th_dn_b = np.percentile(p_b, q_pct_b)
        th_up_b = np.percentile(p_b, 100 - q_pct_b)
        mask_b = (p_b <= th_dn_b) | (p_b >= th_up_b)
        pred_b = (p_b[mask_b] >= th_up_b).astype(int)
        pos_b = np.where(pred_b == 1, 1.0, -1.0)
        ret_strat_b = pos_b * r_b[mask_b] - 2.0 * fee
        m_b_res = compute_metrics(y_b[mask_b], pred_b, ret_strat_b, fee, horizon, f"Setup B (Dual-Q {q_pct_b}%)")
        
        # Triple-Barrier
        p_tb = np.array(res_tb["probs"])
        y_tb = np.array(res_tb["y_true"])
        r_tb = np.array(res_tb["raw_ret"])
        pred_tb = (p_tb >= 0.5).astype(int)
        pos_tb = np.where(pred_tb == 1, 1.0, -1.0)
        ret_strat_tb = pos_tb * r_tb - 2.0 * fee
        m_tb_res = compute_metrics(y_tb, pred_tb, ret_strat_tb, fee, horizon, "Triple-Barrier Adaptive")
        
        # Focal Loss Classifier
        p_foc = np.array(res_focal["probs"])
        y_foc = np.array(res_focal["y_true"])
        r_foc = np.array(res_focal["raw_ret"])
        pred_foc = (p_foc >= 0.5).astype(int)
        pos_foc = np.where(pred_foc == 1, 1.0, -1.0)
        ret_strat_foc = pos_foc * r_foc - 2.0 * fee
        m_foc_res = compute_metrics(y_foc, pred_foc, ret_strat_foc, fee, horizon, "Focal Loss (alpha=0.5, gamma=2)")
        
        # Direct Sharpe Optimization
        pos_shp = np.array(res_sharpe_opt["pos"])
        y_shp = np.array(res_sharpe_opt["y_true"])
        r_shp = np.array(res_sharpe_opt["raw_ret"])
        pred_shp = (pos_shp >= 0.0).astype(int)
        ret_strat_shp = pos_shp * r_shp - np.abs(pos_shp) * 2.0 * fee
        m_shp_res = compute_metrics(y_shp, pred_shp, ret_strat_shp, fee, horizon, "Direct Sharpe Policy")
        
        # Meta-Labeling
        m1_pred = np.array(res_metalabel["m1_pred"])
        m2_prob = np.array(res_metalabel["m2_prob"])
        y_meta = np.array(res_metalabel["y_true"])
        r_meta = np.array(res_metalabel["raw_ret"])
        # Execute only if M2 probability is above threshold (top 40% conviction of profitability)
        tau_meta = np.percentile(m2_prob, 50) # top 50% profitable filter
        mask_meta = m2_prob >= tau_meta
        pred_meta = m1_pred[mask_meta]
        pos_meta = np.where(pred_meta == 1, 1.0, -1.0)
        ret_strat_meta = pos_meta * r_meta[mask_meta] - 2.0 * fee
        m_meta_res = compute_metrics(y_meta[mask_meta], pred_meta, ret_strat_meta, fee, horizon, "Meta-Labeling (de Prado)")
        
        # Combined SOTA: Focal Loss + Triple Barrier + Meta-Filter
        cb_m1 = np.array(res_combined["m1_pred"])
        cb_m2 = np.array(res_combined["m2_prob"])
        y_cb = np.array(res_combined["y_true"])
        r_cb = np.array(res_combined["raw_ret"])
        tau_cb = np.percentile(cb_m2, 50)
        mask_cb = cb_m2 >= tau_cb
        pred_cb = cb_m1[mask_cb]
        pos_cb = np.where(pred_cb == 1, 1.0, -1.0)
        ret_strat_cb = pos_cb * r_cb[mask_cb] - 2.0 * fee
        m_cb_res = compute_metrics(y_cb[mask_cb], pred_cb, ret_strat_cb, fee, horizon, "Combined SOTA (Focal+TB+Meta)")
        
        # Table of results for this horizon
        results_group = [m_a_res, m_b_res, m_tb_res, m_foc_res, m_shp_res, m_meta_res, m_cb_res]
        
        print(f"\n{'-'*95}")
        print(f"{'Method / Model':<30} | {'Acc':<7} | {'Macro F1':<8} | {'Bear F1':<8} | {'Bull F1':<8} | {'Cov':<6} | {'Sharpe':<7}")
        print(f"{'-'*95}")
        for res in results_group:
            cov = (res['n_samples'] / len(y_a)) * 100.0
            print(f"{res['label']:<30} | {res['acc']*100:6.2f}% | {res['mac_f1']:8.4f} | {res['bear_f1']:8.4f} | {res['bull_f1']:8.4f} | {cov:5.1f}% | {res['sharpe']:7.2f}")
            summary_results.append({
                "horizon": h_label,
                "model": res['label'],
                "accuracy": res['acc'],
                "macro_f1": res['mac_f1'],
                "bear_f1": res['bear_f1'],
                "bull_f1": res['bull_f1'],
                "sharpe": res['sharpe'],
                "coverage": cov
            })
            
    print("\n" + "=" * 90)
    print("  ALL HORIZONS WALK-FORWARD BENCHMARK COMPLETED SUCCESSFULLY.")
    print("=" * 90)
    return pd.DataFrame(summary_results)

if __name__ == "__main__":
    df = prepare_dataset()
    results_df = run_walk_forward_benchmark(df)
    results_df.to_csv(os.path.join(BASE_DIR, "quant_loss_metalabeling_results.csv"), index=False)
    print(f"\nResults saved to: {os.path.join(BASE_DIR, 'quant_loss_metalabeling_results.csv')}")
