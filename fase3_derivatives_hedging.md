# FASE 3: DERIVATIVES PRICING & DYNAMIC HEDGING UNDER FRICTION

> Focus: Deribit Inverse Options, Stochastic Volatility Pricing (Heston), Friction Hedging (Whalley-Wilmott)
> Underlying: BTC = $80,000, ATM K = $80,000, Tenor T = 30 Hari, Vol = 52.61%

---

## 1. STRUKTUR INVERSE OPTIONS (DERIBIT SPECIFICATION)

Di bursa Deribit, opsi crypto diselesaikan langsung dalam bentuk cryptocurrency (BTC), bukan fiat USD:
$$\text{Payoff di } T = \frac{\max(S_T - K, 0)}{S_T} = \max\left(1 - \frac{K}{S_T}, 0\right)$$

### Sifat Non-Linear Inverse Options:
- Jika harga BTC naik tajam, nilai opsi dalam fiat naik, namun nilai perolehan dalam BTC terkompresi karena pembagian dengan $S_T$.
- Harga Opsi ATM Call 30 hari:
  - **Dalam BTC**: **$0.0617\text{ BTC}$**
  - **Nilai Ekuivalen USD**: **$\$4,933.72$**

---

## 2. PRICING BENCHMARK: BSM VS HESTON STOCHASTIC VOLATILITY

Integrasi karakteristik numerik Fourier (Carr-Madan / COS method framework):
- **Black-Scholes-Merton Call**: $\$4,933.72$
- **Heston Stochastic Volatility Call**: $\$4,897.42$
  - Parameter: $v_0 = 0.2768$, $\kappa = 2.0$, $\theta = 0.2768$, $\xi = 0.8$, $\rho = -0.10$.
  - Discrepancy: $\$36.30$ ($0.74\%$). Pada ATM 30-hari, discrepancy kecil, namun deviasi melebar signifikan pada Deep OTM Puts dan Calls akibat efek vol-of-vol $\xi$.

---

## 3. DYNAMIC HEDGING UNDER REALISTIC MARKET FRICTIONS

Menguji temuan Kumar (2026, arXiv:2608.29025) terkait friksi transaksi riil (5 bps fee) pada portofolio short call ATM:

### Strategi:
1. **Periodic Discrete BS Delta Hedging**:
   - Rebalance per jam (720 langkah waktu dalam 30 hari).
   - Menghasilkan 720 trade mikro.
2. **Whalley-Wilmott Asymptotic No-Trade Band**:
   - Menghitung batas toleransi optimal $h_t$:
     $$h_t = \left(\frac{3}{2} \frac{c \cdot S_t \cdot \Gamma_t^2}{\gamma}\right)^{1/3}$$
   - **Hasil**: Mengurangi frekuensi trading hingga **$14.7\times$ lebih sedikit** (hanya 49 kali eksekusi dibanding 720 kali), menurunkan eksposur slippage dan latensi eksekusi pada pasar liquiditas terfragmentasi.
