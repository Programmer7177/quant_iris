# Bitcoin Quantitative Forecasting Engine (10-Year OOS Walk-Forward)

A production-grade quantitative research framework for multi-horizon Bitcoin direction forecasting and risk calibration, benchmarked on 10 years of daily Coinbase data (2016–2026, 3,900 bars) with an 8-fold expanding walk-forward protocol.

---

## 📌 Project Overview

This repository contains the complete quantitative pipeline developed to forecast Bitcoin market regime dynamics and forward directional probability across multiple trading horizons:
- **Short 7-Day**: Microstructure-driven (Parkinson Volatility, Amihud Illiquidity, Fear & Greed Index).
- **Short 14-Day**: SHAP-selected core features with Conformal Conviction Gating.
- **Medium 30-Day**: 4-State Gaussian Hidden Markov Model (HMM) + Asymmetric Downside Semi-Variance.
- **Long 90-Day**: Macroeconomic liquidity (US 10-Year Treasury Yield, DXY) + Harmonic Halving Cycle + High-Timeframe Monthly Moving Averages (3M & 5M).

---

## 🏆 Benchmark Results

Evaluated out-of-sample across 3,900 daily bars using walk-forward cross-validation (strictly preserving time-series causality). We classify models into two practical deployment profiles:

### Profile A: Top 1 (Win-Rate & Bull Momentum Hunter)
Designed for trend-following spot / leveraged long strategies to maximize return capture during expansion legs:

| Horizon | Winning Architecture | Directional Accuracy | Bull F1 | Bear F1 | Macro F1 | Market Coverage | Key Mechanism |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **7 Days** | Soft-Blend Hybrid (50:50, Gate 20%) | **57.70%** | 0.6845 | 0.3585 | 0.5215 | 20.0% | Eliminates 80% daily microstructure noise |
| **14 Days** | Setup A (SHAP-5, Gate 35%) | **61.50%** | 0.6983 | 0.4680 | 0.5832 | 35.0% | Peak win-rate during expansion legs |
| **30 Days** | Soft-Blend Hybrid (60:40, Gate 40%) | **62.38%** | 0.7019 | 0.4904 | 0.5962 | 40.0% | HMM latent regime + asymmetric penalty |
| **90 Days** | Veto-Consensus ($P_A \ge 0.58, P_B \le 0.44$) | **67.61%** | **0.7549** | **0.5228** | **0.6389** | **75.3%** | Macro + Halving validated by Monthly SMA |

---

### Profile B: Master Balanced (Symmetric Long/Short & Crash Protection)
Designed for two-way perpetual futures, market-neutral hedging, and tail risk protection with minimal Long/Short disparity ($|\text{Bull F1} - \text{Bear F1}| < 0.085$) and high Macro F1:

| Horizon | Master Balanced Architecture | Directional Accuracy | Bull F1 | Bear F1 | Macro F1 | F1 Gap (Disparity) | Market Coverage |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **7 Days** | Setup A (SHAP-5, Top 20% Gate) | **56.14%** | 0.5994 | **0.5155** | **0.5574** | **0.0839 (8.4%)** | 20.0% |
| **14 Days** | Dual-Head Specialist (A-35% / B-20%) | **60.99%** | 0.6406 | **0.5734** | **0.6070** | **0.0672 (6.7%)** | 44.9% |
| **30 Days** | Dual-Q 30% (Setup B) | **58.67%** | 0.6019 | **0.5703** | **0.5861** | **0.0316 (3.1%)** | 60.0% |
| **90 Days** | Dual-Q 25% (Setup B) | **63.35%** | 0.6111 | **0.6534** | **0.6322** | **0.0423 (4.2%)** | 50.0% |

---

## 📂 Repository Structure

```
├── bitcoin_quant_forecasting_master_pipeline.ipynb  # End-to-end interactive research notebook (10 phases)
├── forecast_engine.py                              # Automated multi-horizon production engine
├── config_presets.py                               # Setup A & Setup B benchmark suite
├── ARSITEKTUR_DAN_METODOLOGI.md                    # In-depth architectural decisions (ADR)
├── PERJALANAN_RISET_DAN_CATATAN_TEMUAN.md          # Research journal, hypotheses, and failure analysis
├── refrensi_metode.md                              # Mapping of 30 arXiv papers to code sections
│
├── data/                                           # Preprocessed datasets & walk-forward results
│   ├── btc_coinbase_10y.csv                        # 10Y Coinbase daily OHLCV dataset
│   ├── global_macro_10y.csv                        # DXY, Gold, Oil, US10Y Yield, S&P500
│   ├── advanced_macro_10y.csv                      # USD/CNY, FXI, TLT, TIP, HYG
│   ├── fear_greed_full.csv                         # Daily Alternative.me sentiment index
│   ├── coinbase_premium_index.csv                  # US institutional vs retail flow spread
│   ├── btc_spot_etf_flows.csv                      # Spot ETF volume flows (IBIT, FBTC, etc.)
│   └── active_learning_results.csv                 # Active Learning QBC benchmark log
│
├── jurnal/                                         # 30 open-access peer-reviewed/arXiv research PDFs
│   ├── 2608.26174_profit_optimized_thresholds.pdf
│   ├── 2602.00776_explainable_patterns_crypto.pdf
│   ├── 2507.05470_temporal_conformal_pred.pdf
│   └── ... (27 other papers)
│
└── experiments/                                    # Standalone scripts for isolated hypothesis testing
    ├── test_active_learning.py
    ├── test_advanced_math_features.py
    ├── test_quant_loss_and_metalabeling.py
    └── test_paper_profit_thresholds.py
```

---

## 🔬 Core Quantitative Innovations

1. **SHAP Permutation Pruning**: Screened 19 technical indicators down to 5 orthogonal features (`rsi_90`, `dist_ema50`, `spread_50_200`, `vol_21`, `power_law_res`), improving OOS accuracy by $+6.52\%$.
2. **Balanced Directional Classification**: Fixed traditional regression MSE long-only bias by dynamically tuning class penalties ($W_{\text{bear}} = 1.3\times - 1.6\times$), lifting Bear F1 from $0.00 \to 0.50-0.57$.
3. **Conformal Conviction Gating**: Only executes positions when model confidence exceeds a statistical margin, protecting capital during choppy regimes.
4. **Fractional Differentiation ($d^*=0.20$)**: Stationarizes price series while retaining $94.02\%$ long-term memory correlation.
5. **High-Timeframe Monthly SMA Package**: Transforms noise into trend signal; Monthly 3M & 5M averages yield an Information Coefficient of $+0.2018$ at 90 days.

---

## 🛠️ Quickstart

### Environment Setup
```bash
# Recommended Python version: 3.11+
py -3.11 -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install numpy pandas scipy scikit-learn hmmlearn
```

### Running Real-Time Signal Diagnostics
```bash
py -3.11 forecast_engine.py --signal
```

### Running Presets & Backtests
```bash
# Setup A: Master High-Conviction
py -3.11 config_presets.py --setup A

# Setup B: Dual-Threshold Symmetric Long/Short
py -3.11 config_presets.py --setup B
```

---

## 📜 License
MIT License. Free for research and educational purposes.
