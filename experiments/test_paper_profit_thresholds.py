import os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

DATA_PATH = r"C:\Mirza Personal\crypto quant\data\btc_coinbase_10y.csv"

def compute_features(df):
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    vol = df['volume']
    
    # Technical & volatility features
    # Parkinson Volatility (21d)
    hl_ratio = np.log(high / low) ** 2
    df['park_vol'] = np.sqrt(hl_ratio.rolling(21).sum() / (4 * 21 * np.log(2)))
    
    # Realized Volatility (21d)
    log_ret = np.log(close / close.shift(1))
    df['vol_21'] = log_ret.rolling(21).std() * np.sqrt(365)
    
    # ATR (14d)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_pct_14'] = df['atr_14'] / close
    
    # RSI 14 and 90
    for rsi_period in [14, 90]:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        roll_gain = gain.rolling(rsi_period).mean()
        roll_loss = loss.rolling(rsi_period).mean()
        rs = roll_gain / (roll_loss + 1e-9)
        df[f'rsi_{rsi_period}'] = 100.0 - (100.0 / (1.0 + rs))
        
    # Moving average distances
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    df['dist_ema50'] = (close - ema50) / ema50
    df['dist_ema200'] = (close - ema200) / ema200
    df['spread_50_200'] = (ema50 / ema200) - 1.0
    
    # Halving harmonics & power law
    dates = pd.to_datetime(df['open_time'])
    days_since_genesis = (dates - pd.Timestamp('2009-01-03')).dt.days
    df['halving_cos'] = np.cos(2 * np.pi * days_since_genesis / 1460.0)
    df['halving_sin'] = np.sin(2 * np.pi * days_since_genesis / 1460.0)
    df['power_law_res'] = np.log(close) - (-17.0 + 5.8 * np.log(days_since_genesis.clip(lower=1)))
    
    # Momentum
    df['ret_7d'] = close.pct_change(7)
    df['ret_14d'] = close.pct_change(14)
    df['ret_30d'] = close.pct_change(30)
    
    # Amihud illiquidity
    dollar_vol = close * vol
    df['amihud'] = (np.abs(log_ret) / (dollar_vol + 1e-5)).rolling(21).mean()
    
    return df

def create_labels(df, horizon=7, hurdle=0.05, dynamic_vol_mult=1.5, cost_barrier=0.002):
    close = df['close']
    high = df['high']
    low = df['low']
    n = len(df)
    
    # 1. Naive Binary Label: Close[t+7] > Close[t]
    fwd_ret_7d = (close.shift(-horizon) - close) / close
    naive_binary = (fwd_ret_7d > 0).astype(int)
    
    # 2. arXiv:2608.26174 Hurdle Label: Close[t+7] / Close[t] - 1 > 5% (covering round-trip fees)
    hurdle_5pct = (fwd_ret_7d > hurdle).astype(int)
    
    # 3. Dynamic Volatility Hurdle: Return > max(5%, dynamic_vol_mult * 7d_vol)
    vol_7d_scaled = df['atr_pct_14'] * np.sqrt(horizon)
    dynamic_hurdle = np.maximum(hurdle, dynamic_vol_mult * vol_7d_scaled)
    hurdle_dynamic_vol = (fwd_ret_7d > dynamic_hurdle).astype(int)
    
    # 4. Marcos Lopez de Prado Triple Barrier Labeling
    # Upper barrier = hurdle (or dynamic ATR), Lower barrier = hurdle, Horizontal = 7 days
    # Return 1 if hit upper before lower, 0 otherwise
    tb_labels = np.zeros(n, dtype=int)
    tb_realized_returns = np.zeros(n, dtype=float)
    
    for i in range(n - horizon):
        entry_p = close.iloc[i]
        barr_pct = max(hurdle, dynamic_vol_mult * vol_7d_scaled.iloc[i]) if not np.isnan(vol_7d_scaled.iloc[i]) else hurdle
        up_barrier = entry_p * (1.0 + barr_pct)
        down_barrier = entry_p * (1.0 - barr_pct)
        
        hit = 0
        realized_ret = (close.iloc[i + horizon] - entry_p) / entry_p
        for j in range(1, horizon + 1):
            cur_high = high.iloc[i + j]
            cur_low = low.iloc[i + j]
            
            # check barrier hits
            if cur_high >= up_barrier and cur_low <= down_barrier:
                # Both touched intraday: conservative penalty (treat as stop out or no win)
                hit = 0
                realized_ret = -barr_pct
                break
            elif cur_high >= up_barrier:
                hit = 1
                realized_ret = barr_pct
                break
            elif cur_low <= down_barrier:
                hit = 0
                realized_ret = -barr_pct
                break
                
        tb_labels[i] = hit
        tb_realized_returns[i] = realized_ret
        
    df['fwd_ret_7d'] = fwd_ret_7d
    df['label_naive'] = naive_binary
    df['label_hurdle_5pct'] = hurdle_5pct
    df['label_dynamic_vol'] = hurdle_dynamic_vol
    df['label_triple_barrier'] = tb_labels
    df['tb_realized_return'] = tb_realized_returns
    
    return df

def optimize_threshold_paper(probs_val, y_val, g=0.05, c=0.001):
    # From arXiv:2608.26174 Eq (3) & (6):
    # tau* = argmax_tau [ TP(tau) * g - (TP(tau) + FP(tau)) * c ]
    best_tau = 0.5
    best_profit = -1e9
    taus = np.linspace(0.01, 0.99, 99)
    for tau in taus:
        preds = (probs_val >= tau).astype(int)
        tp = np.sum((preds == 1) & (y_val == 1))
        fp = np.sum((preds == 1) & (y_val == 0))
        profit = tp * g - (tp + fp) * c
        if profit > best_profit:
            best_profit = profit
            best_tau = tau
    return best_tau, best_profit

def run_walk_forward_evaluation(df, features, target_col, n_splits=6, min_train=1000, horizon=7, cost_per_trade=0.001):
    valid_df = df.dropna(subset=features + [target_col, 'fwd_ret_7d']).copy()
    n = len(valid_df)
    step = (n - min_train - horizon) // n_splits
    
    records = []
    
    for split_idx in range(n_splits):
        train_end = min_train + split_idx * step
        test_end = min(train_end + step, n - horizon)
        
        train_data = valid_df.iloc[:train_end]
        test_data = valid_df.iloc[train_end:test_end]
        
        if len(test_data) == 0:
            break
            
        X_train = train_data[features].values
        y_train = train_data[target_col].values
        
        X_test = test_data[features].values
        y_test = test_data[target_col].values
        actual_ret = test_data['fwd_ret_7d'].values
        
        # Train ExtraTreesClassifier with class_weight='balanced'
        model = ExtraTreesClassifier(n_estimators=100, max_depth=5, min_samples_leaf=15, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        # Validation fold inside training window (last 20% of train) to tune threshold
        val_size = max(50, int(len(train_data) * 0.2))
        X_val_tune = X_train[-val_size:]
        y_val_tune = y_train[-val_size:]
        val_probs = model.predict_proba(X_val_tune)[:, 1] if len(model.classes_) == 2 else np.zeros(len(y_val_tune))
        
        # Optimize threshold via paper profit equation
        opt_tau, _ = optimize_threshold_paper(val_probs, y_val_tune, g=0.05, c=cost_per_trade)
        
        test_probs = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else np.zeros(len(y_test))
        
        # Predictions using standard 0.5 threshold vs paper profit-optimized threshold
        preds_fixed = (test_probs >= 0.5).astype(int)
        preds_opt = (test_probs >= opt_tau).astype(int)
        
        for p_type, preds, tau_used in [('fixed_0.5', preds_fixed, 0.5), ('profit_opt', preds_opt, opt_tau)]:
            acc = np.mean(preds == y_test)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
            
            try:
                auc = roc_auc_score(y_test, test_probs)
            except:
                auc = 0.5
                
            # Economic & Trading metrics:
            # When preds == 1: take long position holding 7 days
            # Simulated trading returns (unlevered): actual 7d return minus 2 * fee
            n_trades = np.sum(preds == 1)
            trade_returns = actual_ret[preds == 1] - (2 * cost_per_trade)
            
            hit_rate = np.mean(actual_ret[preds == 1] > 0) if n_trades > 0 else 0.0
            econ_hit_rate = np.mean(actual_ret[preds == 1] > 0.05) if n_trades > 0 else 0.0
            
            # Paper Eq (3) simulated paper profit
            tp = np.sum((preds == 1) & (y_test == 1))
            fp = np.sum((preds == 1) & (y_test == 0))
            paper_profit = tp * 0.05 - (tp + fp) * cost_per_trade
            
            # Realized PnL and Sharpe (annualized for 7-day holding steps)
            cum_ret = np.sum(trade_returns) if n_trades > 0 else 0.0
            ret_mean = np.mean(trade_returns) if n_trades > 1 else 0.0
            ret_std = np.std(trade_returns) if n_trades > 1 else 1e-6
            sharpe = (ret_mean / (ret_std + 1e-9)) * np.sqrt(365.0 / 7.0) if n_trades > 1 else 0.0
            
            records.append({
                'split': split_idx + 1,
                'target': target_col,
                'mode': p_type,
                'tau': tau_used,
                'n_test': len(y_test),
                'n_trades': n_trades,
                'pos_rate': np.mean(y_test),
                'auc': auc,
                'acc': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'macro_f1': macro_f1,
                'hit_rate': hit_rate,
                'econ_hit_rate': econ_hit_rate,
                'paper_profit': paper_profit,
                'cum_ret': cum_ret,
                'sharpe': sharpe
            })
            
    return pd.DataFrame(records)

def main():
    print("="*80)
    print("ARXIV:2608.26174 BTC PROFIT-OPTIMIZED THRESHOLDS WALK-FORWARD BENCHMARK")
    print("="*80)
    
    df = pd.read_csv(DATA_PATH)
    df = compute_features(df)
    df = create_labels(df, horizon=7, hurdle=0.05, dynamic_vol_mult=1.5, cost_barrier=0.001)
    
    features = [
        'park_vol', 'vol_21', 'atr_pct_14', 'rsi_14', 'rsi_90', 
        'dist_ema50', 'dist_ema200', 'spread_50_200', 'halving_cos', 
        'halving_sin', 'power_law_res', 'ret_7d', 'ret_14d', 'ret_30d', 'amihud'
    ]
    
    targets = [
        ('label_naive', 'Naive Binary Directional (>0%)'),
        ('label_hurdle_5pct', 'arXiv:2608.26174 Hurdle Move (>5%)'),
        ('label_dynamic_vol', 'Dynamic Volatility Hurdle (max(5%, 1.5x Vol))'),
        ('label_triple_barrier', 'Lopez de Prado Triple Barrier')
    ]
    
    all_results = []
    
    for col, desc in targets:
        print(f"\nRunning walk-forward for: {desc} [{col}]...")
        res = run_walk_forward_evaluation(df, features, target_col=col, n_splits=6, min_train=1200, horizon=7)
        all_results.append(res)
        
    res_df = pd.concat(all_results, ignore_index=True)
    
    # Aggregated Summary
    summary = res_df.groupby(['target', 'mode']).agg({
        'auc': 'mean',
        'acc': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'macro_f1': 'mean',
        'n_trades': 'sum',
        'hit_rate': 'mean',
        'econ_hit_rate': 'mean',
        'paper_profit': 'sum',
        'cum_ret': 'sum',
        'sharpe': 'mean'
    }).reset_index()
    
    print("\n" + "="*80)
    print("AGGREGATED WALK-FORWARD RESULTS ACROSS ALL FOLDS (2016 - 2026)")
    print("="*80)
    print(summary.to_string(index=False))
    
    # Detailed breakdown per fold for the Hurdle 5% model
    print("\n" + "="*80)
    print("DETAILED FOLD-BY-FOLD COMPARISON: arXiv:2608.26174 Hurdle Move (>5%)")
    print("="*80)
    hurdle_sub = res_df[res_df['target'] == 'label_hurdle_5pct'][['split', 'mode', 'tau', 'n_trades', 'auc', 'precision', 'recall', 'macro_f1', 'econ_hit_rate', 'paper_profit', 'cum_ret', 'sharpe']]
    print(hurdle_sub.to_string(index=False))
    
    print("\n" + "="*80)
    print("DETAILED FOLD-BY-FOLD COMPARISON: Naive Binary Classification (>0%)")
    print("="*80)
    naive_sub = res_df[res_df['target'] == 'label_naive'][['split', 'mode', 'tau', 'n_trades', 'auc', 'precision', 'recall', 'macro_f1', 'hit_rate', 'paper_profit', 'cum_ret', 'sharpe']]
    print(naive_sub.to_string(index=False))

if __name__ == '__main__':
    main()
