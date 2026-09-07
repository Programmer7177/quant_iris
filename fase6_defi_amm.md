# FASE 6: DEFI & AUTOMATED MARKET MAKER (AMM) QUANTITATIVE ANALYTICS

> Focus: Uniswap V2/V3 Concentrated Liquidity, Impermanent Loss Dynamics, Range Optimization, MEV Sandwich Modeling
> Parameter: Spot BTC = $80,000, Volatilitas Tahunan = 52.61%, Horizon = 30 Hari

---

## 1. FORMULASI MATEMATIKA IMPERMANENT LOSS (IL)

Pada model Constant Product AMM ($x \cdot y = k$), penyedia likuiditas (LP) mengalami kerugian oportunitas relatif terhadap strategi beli dan simpan (HODL) saat rasio harga relatif bergeser sebesar $k = P_t / P_0$:
$$\text{IL}(k) = \frac{2\sqrt{k}}{1 + k} - 1$$

### Perbandingan Efisiensi Modal & Risiko (V2 vs V3):
- **Uniswap V2**: Likuiditas tersebar di seluruh rentang harga $[0, \infty)$, menghasilkan kerugian IL $-5.72\%$ saat harga bergerak $2\times$ atau $0.5\times$.
- **Uniswap V3 (Concentrated Liquidity)**:
  - Efisiensi modal meningkat dengan faktor:
    $$\text{Multiplier} = \frac{1}{1 - \sqrt{p_a / p_b}}$$
  - Pada konsentrasi modal $5\times$, kerugian IL efektif terakselerasi menjadi **$-28.60\%$** bila harga mendekati batas rentang.

---

## 2. OPTIMAL LP RANGE SEBAGAI FUNGSI VOLATILITAS

Untuk meminimalkan risiko harga keluar dari rentang aktif (*out-of-range divergence*) seraya memaksimalkan perolehan *swap fees*, rentang optimal $[p_a, p_b]$ diturunkan dari volatilitas aset $\sigma$:
$$p_a = S_0 \exp\left(-z_{\alpha/2} \cdot \sigma \sqrt{\Delta t}\right), \quad p_b = S_0 \exp\left(+z_{\alpha/2} \cdot \sigma \sqrt{\Delta t}\right)$$

### Hasil Optimasi (Tenor 30 Hari, $95\%$ Confidence):
- **Batas Bawah ($p_a$)**: **$\$59,525.74$**
- **Batas Atas ($p_b$)**: **$\$107,516.51$**
- **Bandwidth Pita**: $60.0\%$
- **Multiplier Efisiensi Modal**: **$3.91\times$** dibanding Uniswap V2.

---

## 3. MEKANIKA EKSTRAKSI MEV (SANDWICH ATTACK)

Bot MEV memanfaatkan transparansi mempool publik untuk mengekstraksi nilai dari transaksi swap pedagang:
1. **Front-Run Leg**: Bot mendeteksi order swap $\$200\text{k}$ dengan batas toleransi slippage $1\%$, lalu menyuntikkan modal beli $\$39,900.50$ dengan gas fee lebih tinggi.
2. **Victim Swap**: Transaksi korban tereksekusi pada harga batas maksimum toleransi slippage ($\$82,809.98$).
3. **Back-Run Leg**: Bot langsung menjual kembali aset ke pool dalam blok yang sama.
4. **Hasil Ekstraksi**: Bot meraup keuntungan bersih **$\$2,004.64$** secara bebas risiko pasar (*riskless arbitrage*), menciptakan kerugian eksekusi (shortfall) bagi pedagang sebesar $\$1,908.51$.
