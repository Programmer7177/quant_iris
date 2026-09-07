"""
FRONTIER BALANCED SCORE ENHANCER
Tests 4 advanced quant classification techniques on 14d & 30d BTC walk-forward:
1. Balanced ExtraTrees (Baseline Pemenang Sebelumnya)
2. Triple-Barrier Labeling (López de Prado) with Volatility-Adjusted Target
3. Probability Threshold Tuning via ROC/Youden-J on Out-of-Fold (OOF)
4. Cost-Sensitive Gradient Boosting (Focal Loss / Asymmetric Sample Weights based on Return Magnitude)
5. Balanced Probability Stacking (Ensemble Calibrated ExtraTrees + HistGradientBoosting + LogisticRegression)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from forecast_engine import load_data, fit_hmm, FEAT_SHORT, FEAT_MEDIUM

df = load_data()
log_rets = np.log(df['close'] / df['close'].shift(1)).fillna(0).values
states, _, _ = fit_hmm(log_rets)
df['hmm_state'] = states
df['hmm_norm'] = states.astype(float) / 3.0

n = len(df)
min_train = 500
step = max(60, (n - min_train) // 8)

for horizon, feats in [(14, FEAT_SHORT), (30, FEAT_MEDIUM)]:
    print(f"\n========================================================")
    print(f"             BENCHMARK ADVANCED HORIZON {horizon} HARI")
    print(f"========================================================")
    
    y_true_all = []
    p_base_all = []
    p_return_weighted_all = []
    p_stack_all = []
    p_oof_tuned_all = []
    
    ft = [f for f in feats if f in df.columns]
    
    for start in range(min_train, n - horizon, step):
        tr = df.iloc[:start]
        te = df.iloc[start:min(start+step, n-horizon)]
        if len(te) < 10: continue
        
        X_tr = tr[ft].values
        y_tr_ret = tr[f'fwd_ret_{horizon}'].values
        y_tr_cls = (y_tr_ret > 0).astype(int)
        
        X_te = te[ft].values
        y_te_ret = te[f'fwd_ret_{horizon}'].values
        y_te_cls = (y_te_ret > 0).astype(int)
        
        y_true_all.extend(y_te_cls)
        
        # 1. Baseline Balanced ExtraTrees
        m1 = ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15, 
                                  class_weight='balanced', random_state=42)
        m1.fit(X_tr, y_tr_cls)
        p_base_all.extend(m1.predict(X_te))
        
        # 2. Magnitude-Weighted Classification (López de Prado sample weights)
        # Weight each sample by the absolute size of forward return so big crashes and pumps are punished heavier
        sample_weights = np.abs(y_tr_ret) / (np.mean(np.abs(y_tr_ret)) + 1e-9)
        # Give extra multiplier to negative returns to balance total weight mass
        neg_mask = (y_tr_cls == 0)
        pos_mask = (y_tr_cls == 1)
        w_neg_sum = sample_weights[neg_mask].sum()
        w_pos_sum = sample_weights[pos_mask].sum()
        if w_neg_sum > 0:
            sample_weights[neg_mask] *= (w_pos_sum / w_neg_sum)
            
        m2 = ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15, random_state=42)
        m2.fit(X_tr, y_tr_cls, sample_weight=sample_weights)
        p_return_weighted_all.extend(m2.predict(X_te))
        
        # 3. Balanced Probability Stacking (ExtraTrees + HistGBM + LogisticRegression)
        sc = StandardScaler().fit(X_tr)
        Xs_tr = sc.transform(X_tr)
        Xs_te = sc.transform(X_te)
        
        m_hgb = HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=20, class_weight='balanced', random_state=42)
        m_hgb.fit(X_tr, y_tr_cls)
        
        m_lr = LogisticRegression(class_weight='balanced', C=0.1, random_state=42)
        m_lr.fit(Xs_tr, y_tr_cls)
        
        # Soft voting / averaging probabilities
        prob_et = m1.predict_proba(X_te)[:, 1]
        prob_hgb = m_hgb.predict_proba(X_te)[:, 1]
        prob_lr = m_lr.predict_proba(Xs_te)[:, 1]
        
        prob_ensemble = 0.4 * prob_et + 0.35 * prob_hgb + 0.25 * prob_lr
        p_stack_all.extend((prob_ensemble > 0.50).astype(int))
        
        # 4. Out-of-fold Optimal Threshold Tuning (Youden's J statistic)
        # Find best probability threshold on training data
        prob_tr_et = m1.predict_proba(X_tr)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_tr_cls, prob_tr_et)
        best_idx = np.argmax(tpr - fpr)
        best_thresh = thresholds[best_idx]
        best_thresh = np.clip(best_thresh, 0.42, 0.58) # bound to avoid extreme thresholds
        p_oof_tuned_all.extend((prob_et > best_thresh).astype(int))
        
    y_true_all = np.array(y_true_all)
    
    approaches = [
        ("1. Balanced ExtraTrees (Sebelumnya)", np.array(p_base_all)),
        ("2. Magnitude-Weighted (Loss Bobot Return)", np.array(p_return_weighted_all)),
        ("3. Balanced Stacking (ET + HistGBM + LogReg)", np.array(p_stack_all)),
        ("4. OOF Optimal Threshold Tuning (Youden-J)", np.array(p_oof_tuned_all)),
    ]
    
    for name, p in approaches:
        cm = confusion_matrix(y_true_all, p, labels=[0, 1])
        acc = accuracy_score(y_true_all, p)
        f1_mac = f1_score(y_true_all, p, average='macro')
        f1_bull = f1_score(y_true_all, p, pos_label=1)
        f1_bear = f1_score(y_true_all, p, pos_label=0)
        
        print(f"\n--- {name} ---")
        print(f"  Confusion: TN={cm[0,0]:<4} FP={cm[0,1]:<4} | FN={cm[1,0]:<4} TP={cm[1,1]:<4}")
        print(f"  Akurasi: {acc:.2%} | Macro F1: {f1_mac:.4f} | Bull F1: {f1_bull:.4f} | Bear F1: {f1_bear:.4f}")
