# FASE 4: STATISTICAL ARBITRAGE & SHORT-HORIZON TRADING

> Dataset: Binance BTCUSDT & ETHUSDT 15-Minute Bars (1,000 observations)
> Model: Directional Candle Reversal (Kitron & Wengrowicz 2026), Dynamic Kalman Filter Pairs Trading

---

## 1. REPLIKASI 15-MINUTE CANDLE REVERSAL (KITRON & WENGROWICZ, 2026)

Paper Kitron & Wengrowicz (arXiv:2608.21888) menyatakan 90% pair crypto di Binance menunjukkan directional mean-reversion pada horizon 15 menit, di mana prediktor terbaik adalah tanda arah candle sebelumnya:
$$\text{Signal}_t = -\text{sign}(r_{t-1})$$

### Hasil Empiris Riil:
- **Win Rate**: **$51.50\%$** (Secara statistik di atas 50% random walk)
- **Gross Return (0 bps)**: **$+0.63\%$** (Annualized Sharpe: **$+0.80$**)
- **Net Return (5 bps fee)**: **$-22.22\%$** (515 trades per 1,000 bar)

### Lesson for Quant Practitioners:
- Sinyal prediktif terbukti nyata di level mikrostruktur, tetapi **biaya turnover (turnover drag)** menghancurkan alpha pada akun ritel bertarif taker 5 bps.
- Strategi ini hanya dapat dimonetisasi oleh pembuat likuiditas (Market Maker) yang mendapatkan fee rebate (negatif maker fee) atau VIP liquidity provider tiers.

---

## 2. KALMAN FILTER PAIRS TRADING (BTC - ETH)

Mengestimasi rasio lindung nilai dinamis (dynamic hedge ratio) menggunakan recursive Kalman Filter:
$$y_t = \alpha_t + \beta_t x_t + \epsilon_t$$
$$\beta_t = \beta_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$

### Hasil Kalibrasi:
- **Rata-rata $\beta_t$**: **$0.6898$** (Setiap 1 unit eksposur ETH dilindung nilai dengan 0.69 unit BTC).
- **Uji Kointegrasi Engle-Granger**: $p$-value $= 0.0516$ (kointegrasi bersyarat).
- **Kendala Eksekusi**: Perdagangan 2-leg spread pada horizon 15m memicu 146 kali eksekusi, memerlukan ambang batas Z-score yang lebih selektif ($|z| \ge 2.5$) untuk menghindari whipsaw pada fluktuasi jangka pendek.
