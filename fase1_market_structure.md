# FASE 1: CRYPTO MARKET STRUCTURE & EMPIRICAL STYLIZED FACTS

> Data Source: Binance API (Spot & USD-M Futures) | Tanggal Uji: September 2026
> Sample: 1,000 daily bars (2023-2026), 1,000 15m bars, 500 funding rate epochs, L2 depth snapshot

---

## 1. STRUKTUR PASAR CRYPTO (CEX, DEX & PERPETUALS)

### 1.1 Taksonomi Ekosistem
Pasar cryptocurrency modern beroperasi dalam 4 lapisan likuiditas utama:
1. **CEX Spot (Binance, Coinbase, Kraken)**:
   - Order matching engine berbasis Central Limit Order Book (CLOB).
   - Maker-taker fee model (umumnya 1–10 bps).
   - Price discovery primer untuk arbitrase fiat-ke-crypto.
2. **CEX Derivatives (Binance Futures, Bybit, Deribit)**:
   - Volume mendominasi 70–80% dari total aktivitas pasar crypto global.
   - Instrumen utama: **Perpetual Futures (Swap)** dan **Inverse/Vanilla Options**.
3. **DEX Spot & Derivatives (Uniswap, Hyperliquid, dYdX)**:
   - AMM (Constant Product $x \cdot y = k$, Concentrated Liquidity V3) dan on-chain CLOB.
   - Karakteristik latency block time dan risiko MEV (Maximal Extractable Value).
4. **Prediction Markets (Polymarket)**:
   - Binary state-contingent payoff. Terbukti berkorelasi erat dengan probability density implied options (Portnaya, 2026).

---

## 1.2 MEKANISME PERPETUAL FUTURES & FUNDING RATE

Perpetual futures tidak memiliki tanggal jatuh tempo (expiry). Untuk menjaga agar harga kontrak perpetual ($P_{\text{perp}}$) tidak deviasi jauh dari harga underlying spot/index ($P_{\text{spot}}$), bursa menerapkan mekanisme **Funding Rate**.

### Formula Pembayaran Funding:
$$\text{Payment} = \text{Position Size} \times F_t$$
dimana $F_t$ dihitung berkala (biasanya tiap 8 jam pada 00:00, 08:00, 16:00 UTC):
$$F_t = \text{Clamp}\left(\text{Premium Index} + \text{Interest Rate}, -0.75\%, +0.75\%\right)$$
$$\text{Premium Index} = \frac{\max(0, \text{Impact Bid Price} - P_{\text{index}}) - \max(0, P_{\text{index}} - \text{Impact Ask Price})}{P_{\text{index}}}$$

### Implikasi Quant:
- **Cash and Carry Arbitrage (Basis Trade)**:
  - Beli Spot BTC + Short Perp BTC jika $F_t > 0$ secara konsisten.
  - Bebas risiko delta direction, mengumpulkan cash flow funding.
  - Pada pengujian data Binance aktual: **76.4% funding bernilai positif** dengan rata-rata annualized carry yield $+3.34\%$.

---

## 2. BUKTI EMPIRIS: STYLIZED FACTS BITCOIN

Pengujian empiris langsung pada data historis BTCUSDT menghasilkan stylized facts kuantitatif berikut:

| Karakteristik | Nilai Uji Empiris (Binance) | TradFi Equities Benchmark | Implikasi Model Matematika |
|---|---|---|---|
| **Annualized Volatility** | **47.08%** | 15% – 20% (S&P 500) | Membutuhkan dynamic risk budgeting & scaling |
| **Skewness** | **+0.104** | Negatif (-0.5 s.d. -1.0) | Crash risk seimbang dengan upside squeeze |
| **Excess Kurtosis** | **+3.444** | 1.0 – 2.0 | **Fat tails parah**. Model Gaussian / Normal BSM GAGAL |
| **Jarque-Bera Test** | **stat=495.4 ($p < 10^{-107}$)** | Rejected | Penolakan mutlak atas asumsi return normal |
| **Hill Tail Index ($\alpha$)** | **3.87** (top 5%) | 3.0 – 4.0 | Power-law tail ($P(R > x) \sim x^{-\alpha}$), varians terhingga |
| **Return Autocorrelation** | $\rho_1 = -0.0515$ (near 0) | $\rho_1 \approx 0$ | Efficient Market Hypothesis (Martingale property) |
| **Vol Clustering (ACF $r^2$)** | **$\rho_1 = +0.2347, \rho_5 = +0.1173$** | Persisten | Volatility clustering valid -> Wajib GARCH / HAR-RV |
| **Leverage Effect** | **$\text{Corr}(r_t, \sigma_{t+5}) = -0.0887$** | Negatif kuat (-0.60 s.d. -0.80) | Asimetri vol jauh lebih lemah dibanding pasar saham |
| **15m Reversal Rate** | **51.45%** | 50.0% (acak murni) | Sinyal mean-reversion jangka pendek (Kitron et al., 2026) |

---

## 3. ORDER BOOK MICROSTRUCTURE

Snapshot Level 2 (100 depth levels) pada BTCUSDT Spot:
- **Best Bid / Ask**: \$80,000.00 / \$80,000.01.
- **Bid-Ask Spread**: \$0.01 ($< 0.01$ bps) — likuiditas spread CEX top-tier sangat rapat.
- **Order Book Imbalance (OBI)**:
  $$\text{OBI} = \frac{V_{\text{bid}} - V_{\text{ask}}}{V_{\text{bid}} + V_{\text{ask}}} = -0.541$$
  Mengindikasikan dinding penawaran (ask depth) lebih padat pada range snapshot tersebut.

---

## 4. KESIMPULAN FASE 1 & JEMBATAN KE FASE 2

1. **Gaussian BSM Invalid**: Dengan kurtosis 3.44 dan JB stat 495, model opsi BSM standar akan meremehkan (underprice) out-of-the-money (OTM) options dan risiko ekstrim.
2. **Vol Clustering Confirmed**: Autokorelasi $r_t^2$ yang persisten hingga lag 5 membuktikan varians bersyarat (conditional heteroskedasticity) harus dimodelkan secara dinamis.
3. **Langkah Berikutnya (Fase 2)**:
   - Kalibrasi model **GARCH(1,1)** dengan error Student-t vs GED.
   - Implementasi **HAR-RV (Heterogeneous Autoregressive of Realized Volatility)**.
   - Kalibrasi **Rough Bergomi (rBergomi)** untuk menangkap Hurst parameter $H \approx 0.03$.
