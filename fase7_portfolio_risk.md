# FASE 7: PORTFOLIO CONSTRUCTION & TAIL RISK MANAGEMENT

> Universe: BTC, ETH, SOL, BNB (1,000 Daily Observations, 2023–2026)
> Method: Classical Markowitz vs Rockafellar-Uryasev Mean-CVaR LP, Multi-Asset Fractional Kelly, Crash Correlation Analysis

---

## 1. HISTORICAL ASSET DYNAMICS

| Aset | Annualized Return | Annualized Volatility | Sharpe Ratio | Catatan Kuantitatif |
|---|---|---|---|---|
| **BTC** | **$+23.97\%$** | **$47.08\%$** | **$0.51$** | Jangkar volatilitas terendah di ekosistem crypto |
| **ETH** | $+4.65\%$ | $67.66\%$ | $0.07$ | Kinerja Sharpe rendah akibat kompresi margin L2 |
| **SOL** | $+16.09\%$ | $80.32\%$ | $0.20$ | Volatilitas tertinggi, beta spekulatif tinggi |
| **BNB** | **$+39.81\%$** | **$53.02\%$** | **$0.75$** | Sharpe rasio historis tertinggi dalam universe |

---

## 2. PORTFOLIO OPTIMIZATION: MIN-VARIANCE VS MIN-CVAR (95%)

Formulasi program linier Rockafellar-Uryasev (2000) meminimalkan kerugian bersyarat pada kuantil terburuk 5% (CVaR), bukan sekadar varians:
$$\min_{w, \xi, z} \xi + \frac{1}{(1-\alpha)T}\sum_{t=1}^T z_t \quad \text{s.t.} \quad z_t \ge -r_t^T w - \xi, \quad z_t \ge 0, \quad \sum w_i = 1$$

### Alokasi Optimal:
- **Minimasi Varians (Markowitz)**: $70.7\%\text{ BTC} + 29.3\%\text{ BNB}$ ($0\%\text{ ETH, } 0\%\text{ SOL}$).
- **Minimasi CVaR 95% (Tail Risk)**: **$52.5\%\text{ BTC} + 47.5\%\text{ BNB}$** ($0\%\text{ ETH, } 0\%\text{ SOL}$).
- **Hasil Risiko Portofolio**: CVaR(95%) harian terkompresi menjadi **$0.20\%/\text{hari}$** ($3.84\%$ *annualized tail risk*).

---

## 3. MULTI-ASSET FRACTIONAL KELLY CRITERION

Formula unconstrained growth-optimal:
$$f^* = \Sigma^{-1} \mu$$
- **Full Kelly**: Leverage total **$2.12\times$** (Long $+2.44\times\text{ BTC}$, Long $+2.18\times\text{ BNB}$, Short $-2.07\times\text{ ETH}$).
- **Quarter Kelly ($f^*/4$) — Rekomendasi Quant**:
  - Leverage total: **$0.53\times$** (Cadangan kas / stablecoin $47\%$).
  - Alokasi: $+0.61\times\text{ BTC}$ dan $+0.55\times\text{ BNB}$.
  - Mengeliminasi risiko likuidasi paksa (*margin call ruin*) saat terjadi *volatility shock*.

---

## 4. TAIL DEPENDENCE & "CRASHING TOGETHER"

- Korelasi dasar antar aset crypto berkisar antara **$+0.68$ hingga $+0.82$**.
- Saat terjadi koreksi tajam (kuantil terendah 5% BTC), korelasi tetap solid di atas $+0.55$ hingga $+0.75$.
- **Implikasi**: Portofolio murni multi-crypto tidak menawarkan diversifikasi sejati saat krisis sistemik; perlindungan portofolio yang efektif memerlukan instrumen pelindung nilai (Short Perp, Cash & Carry, atau Long Put Options).
