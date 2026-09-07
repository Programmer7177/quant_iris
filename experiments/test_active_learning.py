"""
ACTIVE LEARNING & UNCERTAINTY SAMPLING FOR WALK-FORWARD BITCOIN PREDICTION
==========================================================================
Literature (2024-2026):
  1. Hoarau, Lemaire et al. (Springer Machine Learning 2024): "Evidential uncertainty sampling
     strategies for active learning" (DOI: 10.1007/s10994-024-06567-2)
     -> Decomposes model uncertainty into reducible (epistemic) and irreducible (aleatoric) uncertainty.
  2. "QueryMarket: Cost-Aware Online Active Learning in Data Markets" (arXiv:2606.17805, 2026)
     -> Formulates active selection on streaming time-series under cost / memory budgets.
  3. "Machine learning in stock market forecasting: a comprehensive review" (Springer Discov Computing, 2026)
     -> Details non-stationarity, regime shifts, and adaptive retraining in financial time series.

Pipelines Evaluated:
  1. Static Baseline: Fit initial N=600 days, evaluate forward without updates.
  2. Rolling Uniform Retrain: Standard industry walk-forward with unweighted retraining.
  3. Margin Uncertainty Weighted: Boosts boundary sample weights (w = 0.5 + 1.5 * margin).
  4. Prediction Entropy Weighted: Boosts high Shannon entropy instances (w = 0.5 + 1.5 * H).
  5. Query-By-Committee (QBC) Disagreement: Boosts instances where ensemble tree variance is maximal.
  6. Selective Pool Boundary Retraining: Discards low-uncertainty consensus points (top 65% boundary kept).
  7. Conformal Uncertainty Filtered Trading: Abstains when test-time predictive entropy exceeds threshold.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, brier_score_loss
from forecast_engine import load_data, fit_hmm, FEAT_SHORT, FEAT_MEDIUM, FEAT_LONG

def calculate_uncertainty_metrics(probs, ensemble_preds=None):
    """
    probs: (N, 2)
    ensemble_preds: (n_estimators, N)
    """
    p = np.clip(probs, 1e-12, 1.0 - 1e-12)
    entropy = -np.sum(p * np.log2(p), axis=1) # Binary entropy [0, 1]
    margin = 1.0 - np.abs(p[:, 1] - p[:, 0])   # Margin uncertainty [0, 1]
    
    qbc_disagreement = np.zeros(len(probs))
    if ensemble_preds is not None:
        qbc_disagreement = np.var(ensemble_preds, axis=0) # Variance across trees [0, 0.25]
        
    return entropy, margin, qbc_disagreement

def run_experiment(horizon=14, feats=FEAT_SHORT, min_train=600, step=60):
    df = load_data()
    log_rets = np.log(df['close'] / df['close'].shift(1)).fillna(0).values
    states, _, _ = fit_hmm(log_rets)
    df['hmm_state'] = states
    df['hmm_norm'] = states.astype(float) / 3.0

    n = len(df)
    target_col = f'fwd_ret_{horizon}'
    ft = [f for f in feats if f in df.columns]
    
    models_keys = [
        "1_static",
        "2_rolling_uniform",
        "3_al_margin_weighted",
        "4_al_entropy_weighted",
        "5_al_qbc_weighted",
        "6_al_selective_retrain"
    ]
    
    results = {k: {"preds": [], "probs": []} for k in models_keys}
    test_uncertainties = []
    y_trues = []
    market_returns = []
    dates = []
    
    # Static model
    tr_init = df.iloc[:min_train]
    X_init = tr_init[ft].values
    y_init = (tr_init[target_col].values > 0).astype(int)
    
    static_model = ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15, 
                                        class_weight='balanced', random_state=42)
    static_model.fit(X_init, y_init)
    
    # Walk-forward loop
    for start in range(min_train, n - horizon, step):
        test_end = min(start + step, n - horizon)
        te = df.iloc[start:test_end]
        if len(te) < 10:
            continue
            
        X_te = te[ft].values
        y_te_ret = te[target_col].values
        y_te = (y_te_ret > 0).astype(int)
        
        y_trues.extend(y_te)
        market_returns.extend(y_te_ret)
        dates.extend(te['date'].values)
        
        tr = df.iloc[:start]
        X_tr = tr[ft].values
        y_tr = (tr[target_col].values > 0).astype(int)
        
        # 1. Static
        prob_static = static_model.predict_proba(X_te)
        pred_static = (prob_static[:, 1] >= 0.5).astype(int)
        results["1_static"]["preds"].extend(pred_static)
        results["1_static"]["probs"].extend(prob_static[:, 1])
        
        # 2. Rolling Uniform
        m_uniform = ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15, 
                                         class_weight='balanced', random_state=42)
        m_uniform.fit(X_tr, y_tr)
        prob_uniform = m_uniform.predict_proba(X_te)
        pred_uniform = (prob_uniform[:, 1] >= 0.5).astype(int)
        results["2_rolling_uniform"]["preds"].extend(pred_uniform)
        results["2_rolling_uniform"]["probs"].extend(prob_uniform[:, 1])
        
        # Active Learning sample weights on historical training instances:
        tree_preds_tr = np.array([tree.predict_proba(X_tr)[:, 1] for tree in m_uniform.estimators_])
        prob_tr = np.mean(tree_preds_tr, axis=0)
        prob_tr_2d = np.column_stack([1.0 - prob_tr, prob_tr])
        ent_tr, margin_tr, qbc_tr = calculate_uncertainty_metrics(prob_tr_2d, tree_preds_tr)
        
        # Record test stream uncertainty
        tree_preds_te = np.array([tree.predict_proba(X_te)[:, 1] for tree in m_uniform.estimators_])
        ent_te, margin_te, qbc_te = calculate_uncertainty_metrics(prob_uniform, tree_preds_te)
        test_uncertainties.extend(ent_te)
        
        # 3. AL Margin Weighted
        w_margin = 0.5 + 1.5 * margin_tr
        m_al_margin = ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15,
                                           class_weight='balanced', random_state=42)
        m_al_margin.fit(X_tr, y_tr, sample_weight=w_margin)
        prob_al_margin = m_al_margin.predict_proba(X_te)
        pred_al_margin = (prob_al_margin[:, 1] >= 0.5).astype(int)
        results["3_al_margin_weighted"]["preds"].extend(pred_al_margin)
        results["3_al_margin_weighted"]["probs"].extend(prob_al_margin[:, 1])
        
        # 4. AL Entropy Weighted
        w_entropy = 0.5 + 1.5 * ent_tr
        m_al_ent = ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15,
                                        class_weight='balanced', random_state=42)
        m_al_ent.fit(X_tr, y_tr, sample_weight=w_entropy)
        prob_al_ent = m_al_ent.predict_proba(X_te)
        pred_al_ent = (prob_al_ent[:, 1] >= 0.5).astype(int)
        results["4_al_entropy_weighted"]["preds"].extend(pred_al_ent)
        results["4_al_entropy_weighted"]["probs"].extend(prob_al_ent[:, 1])
        
        # 5. AL QBC Disagreement Weighted
        qbc_norm = qbc_tr / (np.max(qbc_tr) + 1e-8)
        w_qbc = 0.5 + 1.5 * qbc_norm
        m_al_qbc = ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15,
                                        class_weight='balanced', random_state=42)
        m_al_qbc.fit(X_tr, y_tr, sample_weight=w_qbc)
        prob_al_qbc = m_al_qbc.predict_proba(X_te)
        pred_al_qbc = (prob_al_qbc[:, 1] >= 0.5).astype(int)
        results["5_al_qbc_weighted"]["preds"].extend(pred_al_qbc)
        results["5_al_qbc_weighted"]["probs"].extend(prob_al_qbc[:, 1])
        
        # 6. AL Selective Pool Retraining (Boundary Retention)
        cutoff_margin = np.percentile(margin_tr, 35)
        informative_mask = margin_tr >= cutoff_margin
        X_tr_sel = X_tr[informative_mask]
        y_tr_sel = y_tr[informative_mask]
        w_tr_sel = margin_tr[informative_mask]
        
        m_al_sel = ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=15,
                                        class_weight='balanced', random_state=42)
        m_al_sel.fit(X_tr_sel, y_tr_sel, sample_weight=w_tr_sel)
        prob_al_sel = m_al_sel.predict_proba(X_te)
        pred_al_sel = (prob_al_sel[:, 1] >= 0.5).astype(int)
        results["6_al_selective_retrain"]["preds"].extend(pred_al_sel)
        results["6_al_selective_retrain"]["probs"].extend(prob_al_sel[:, 1])

    y_trues = np.array(y_trues)
    market_returns = np.array(market_returns)
    test_uncertainties = np.array(test_uncertainties)
    
    summary_rows = []
    print(f"\n{'='*78}")
    print(f"   ACTIVE LEARNING VS WALK-FORWARD BENCHMARK (Horizon: {horizon}d, N: {len(y_trues)})")
    print(f"{'='*78}")
    print(f"{'Model Strategy':<25} | {'Acc':<7} | {'Macro F1':<9} | {'Bear F1':<8} | {'Bull F1':<8} | {'Brier':<7} | {'Sharpe':<7}")
    print(f"{'-'*78}")
    
    for k in models_keys:
        p = np.array(results[k]["preds"])
        pr = np.array(results[k]["probs"])
        
        acc = accuracy_score(y_trues, p)
        mac_f1 = f1_score(y_trues, p, average='macro')
        bear_f1 = f1_score(y_trues, p, pos_label=0)
        bull_f1 = f1_score(y_trues, p, pos_label=1)
        brier = brier_score_loss(y_trues, pr)
        
        pos = np.where(p == 1, 1.0, -1.0)
        strat_rets = pos * (market_returns / horizon)
        sharpe = (np.mean(strat_rets) / (np.std(strat_rets) + 1e-9)) * np.sqrt(252)
        
        summary_rows.append({
            "model": k,
            "horizon": horizon,
            "accuracy": acc,
            "macro_f1": mac_f1,
            "bear_f1": bear_f1,
            "bull_f1": bull_f1,
            "brier_score": brier,
            "sharpe": sharpe
        })
        
        print(f"{k:<25} | {acc:.2%}  | {mac_f1:.4f}    | {bear_f1:.4f}   | {bull_f1:.4f}   | {brier:.4f}  | {sharpe:+.2f}")

    # Evaluate 7. Test-time Uncertainty Gating (abstain when top 30% most uncertain)
    p_qbc = np.array(results["5_al_qbc_weighted"]["preds"])
    pr_qbc = np.array(results["5_al_qbc_weighted"]["probs"])
    certain_mask = test_uncertainties <= np.percentile(test_uncertainties, 70)
    acc_cert = accuracy_score(y_trues[certain_mask], p_qbc[certain_mask])
    mac_cert = f1_score(y_trues[certain_mask], p_qbc[certain_mask], average='macro')
    pos_cert = np.where(p_qbc[certain_mask] == 1, 1.0, -1.0)
    rets_cert = pos_cert * (market_returns[certain_mask] / horizon)
    sharpe_cert = (np.mean(rets_cert) / (np.std(rets_cert) + 1e-9)) * np.sqrt(252)
    print(f"\n* Abstention / Uncertainty Gating (Top 70% Conviction on QBC model):")
    print(f"  Coverage: {np.mean(certain_mask):.1%} | Gated Acc: {acc_cert:.2%} | Gated Macro F1: {mac_cert:.4f} | Gated Sharpe: {sharpe_cert:+.2f}")
    
    return pd.DataFrame(summary_rows)

if __name__ == '__main__':
    all_res = []
    for h, feats in [(7, FEAT_SHORT), (14, FEAT_SHORT), (30, FEAT_MEDIUM)]:
        res = run_experiment(horizon=h, feats=feats, min_train=600, step=60)
        all_res.append(res)
    pd.concat(all_res, ignore_index=True).to_csv(r"C:\Mirza Personal\crypto quant\data\active_learning_results.csv", index=False)
    print("\nSaved results to C:\\Mirza Personal\\crypto quant\\data\\active_learning_results.csv")
