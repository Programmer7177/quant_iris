"""
TEST BALANCED DIRECTIONAL FORECASTING
Compares 3 approaches on 14d and 30d:
1. Baseline Regression (Zero threshold)
2. Median/Quantile Threshold Calibration
3. Balanced Classifier (ExtraTreesClassifier with class_weight='balanced')
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
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
    print(f"\n==================================================")
    print(f"            BENCHMARK HORIZON {horizon} HARI")
    print(f"==================================================")
    
    y_true_all = []
    y_reg_zero = []
    y_reg_median = []
    y_clf_bal = []
    
    for start in range(min_train, n - horizon, step):
        tr = df.iloc[:start]
        te = df.iloc[start:min(start+step, n-horizon)]
        if len(te) < 10: continue
        
        ft = [f for f in feats if f in df.columns]
        X_tr = tr[ft].values; y_tr_ret = tr[f'fwd_ret_{horizon}'].values
        X_te = te[ft].values; y_te_ret = te[f'fwd_ret_{horizon}'].values
        
        y_te_true = (y_te_ret > 0).astype(int)
        y_true_all.extend(y_te_true)
        
        # 1. Regressor Baseline (Zero Threshold)
        m_reg = ExtraTreesRegressor(n_estimators=100, max_depth=4, min_samples_leaf=15, random_state=42)
        m_reg.fit(X_tr, y_tr_ret)
        preds_reg = m_reg.predict(X_te)
        y_reg_zero.extend((preds_reg > 0).astype(int))
        
        # 2. Regressor with Median Threshold Calibration
        # Threshold is the median of training set predictions or median of training returns
        median_thresh = np.median(m_reg.predict(X_tr))
        y_reg_median.extend((preds_reg > median_thresh).astype(int))
        
        # 3. Balanced Classifier
        y_tr_cls = (y_tr_ret > 0).astype(int)
        m_cls = ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15, 
                                     class_weight='balanced', random_state=42)
        m_cls.fit(X_tr, y_tr_cls)
        y_clf_bal.extend(m_cls.predict(X_te))
        
    y_true_all = np.array(y_true_all)
    
    models = [
        ("1. Regresi Baseline (Threshold 0)", np.array(y_reg_zero)),
        ("2. Regresi + Median Calibrated", np.array(y_reg_median)),
        ("3. Balanced Classifier (Tree)", np.array(y_clf_bal)),
    ]
    
    for name, p in models:
        cm = confusion_matrix(y_true_all, p, labels=[0, 1])
        acc = accuracy_score(y_true_all, p)
        f1_mac = f1_score(y_true_all, p, average='macro')
        f1_bull = f1_score(y_true_all, p, pos_label=1)
        f1_bear = f1_score(y_true_all, p, pos_label=0)
        
        print(f"\n--- {name} ---")
        print(f"Confusion Matrix [TN={cm[0,0]}, FP={cm[0,1]} / FN={cm[1,0]}, TP={cm[1,1]}]:")
        print(f"  Tebak Turun | Tebak Naik")
        print(f"Actual Turun:  {cm[0,0]:>5}     | {cm[0,1]:>5}")
        print(f"Actual Naik :  {cm[1,0]:>5}     | {cm[1,1]:>5}")
        print(f"Akurasi: {acc:.2%} | Macro F1: {f1_mac:.4f} | Bull F1: {f1_bull:.4f} | Bear F1: {f1_bear:.4f}")
