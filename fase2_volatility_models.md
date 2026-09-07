# FASE 2: VOLATILITY MODELING & REGIME DYNAMICS

> Underlying Data: Binance BTCUSDT Spot (1,000 Daily Bars, 2023–2026)
> Model Kalibrasi: GARCH(1,1)-t, HAR-RV, rBergomi (Rough Vol), 2-State Markov Switching

---

## 1. KALIBRASI GARCH(1,1) STUDENT-T (MLE)

Untuk mengakomodasi kelebihan kurtosis (excess kurtosis = 3.44) yang terbukti di Fase 1, kita mengestimasi GARCH(1,1) dengan distribusi inovasi **Student-t**:
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
$$\epsilon_t = \sigma_t z_t, \quad z_t \sim t(\nu)$$

### Parameter Hasil Kalibrasi:
- **$\omega$**: $3.03 \times 10^{-5}$
- **$\alpha$ (ARCH - reaksi shock)**: $0.0800$
- **$\beta$ (GARCH - persistensi)**: $0.8800$
- **Persistensi ($\alpha + \beta$)**: **$0.9600$** (mendekati Integrated GARCH)
- **$\nu$ (Degrees of Freedom)**: **$5.00$**
  - $\nu \approx 5$ memvalidasi bahwa ekor distribusi Bitcoin sangat tebal (jauh dari Gaussian $\nu > 30$).
- **Unconditional Volatility (Annualized)**: **$52.61\%$**

---

## 2. HAR-RV (HETEROGENEOUS AUTOREGRESSIVE REALIZED VOLATILITY)

Model Corsi (2009) menangkap perilaku pelaku pasar dengan cakupan waktu berbeda (intraday traders, swing traders, long-term investors):
$$\text{RV}_{t+1} = c + \beta_d \text{RV}_t + \beta_w \overline{\text{RV}}_t^{(5)} + \beta_m \overline{\text{RV}}_t^{(22)} + \epsilon_{t+1}$$

### Koefisien Terestimasi:
- **Konstanta $c$**: $0.000359$
- **$\beta_{\text{daily}}$**: $0.1952$
- **$\beta_{\text{weekly}}$**: $0.1201$
- **$\beta_{\text{monthly}}$**: $0.0940$
- **Decay Profile**: $\beta_d > \beta_w > \beta_m$ mengonfirmasi bahwa varians jangka pendek memiliki dampak prediktif langsung tertinggi terhadap volatilitas esok hari.

---

## 3. ROUGH VOLATILITY: ROUGH BERGOMI (rBergomi)

Berdasarkan temuan mutakhir Caruso (2026, arXiv:2608.27575) pada surface opsi Deribit BTC:
$$v_t = \xi_0 \exp\left(\eta \widetilde{W}_t^H - \frac{1}{2}\eta^2 t^{2H}\right)$$
- **Hurst Exponent $H = 0.03$** (kondisi *genuinely rough*, jauh di bawah Brownian motion standar $H = 0.5$ maupun ekuitas $H \approx 0.10$).
- **Vol-of-Vol $\eta = 1.90$**: Menghasilkan fluktuasi tajam pada kurva varians lokal, menjelaskan fenomena *steep skew* pada opsi OTM jangka pendek (short-dated options).

---

## 4. DUA REZIM VOLATILITAS (MARKOV SWITCHING)

- **Rezim Volatilitas Rendah (State 0)**:
  - Volatilitas Tahunan: **$12.81\%$**
  - Probabilitas Bertahan: $70.49\%$ (Rata-rata durasi: 3.4 hari)
- **Rezim Volatilitas Tinggi (State 1)**:
  - Volatilitas Tahunan: **$65.34\%$** (**$5.1\times$ lipat lebih tinggi**)
  - Probabilitas Bertahan: $31.00\%$ (Rata-rata durasi: 1.4 hari)
- **Insight Trading**: Lonjakan volatilitas di Bitcoin cenderung meledak tajam (bursty) namun terkompresi dalam durasi singkat sebelum kembali ke rezim konsolidasi.
