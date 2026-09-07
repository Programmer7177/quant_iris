# FASE 0: MATHEMATICAL FOUNDATIONS — Quant Crypto
## Stochastic Calculus, Volatility Models, Risk Measures, Portfolio Optimization

> Status: IN PROGRESS | Referensi: 30+ papers dari arXiv q-fin
> File implementasi: fase0_implementations.py

---

## 0A. STOCHASTIC CALCULUS — Fondasi Semua Pricing Model

### 0A.1 Brownian Motion (Wiener Process)
Definisi: proses $W_t$ dengan properties:
- $W_0 = 0$
- Increments independent: $W_t - W_s \perp W_s - W_u$ untuk $u < s < t$
- $W_t - W_s \sim \mathcal{N}(0, t-s)$
- Path continuous (a.s.)

**Quadratic Variation**: $[W,W]_t = t$ (deterministic!)
→ Ini yang membedakan Brownian motion dari path biasa

### 0A.2 Itô's Lemma (Fundamental Theorem of Stochastic Calculus)

Untuk $f(t, X_t)$ dimana $dX_t = \mu_t dt + \sigma_t dW_t$:

$$df(t, X_t) = \left(\frac{\partial f}{\partial t} + \mu_t \frac{\partial f}{\partial x} + \frac{1}{2}\sigma_t^2 \frac{\partial^2 f}{\partial x^2}\right)dt + \sigma_t \frac{\partial f}{\partial x} dW_t$$

**Key**: extra term $\frac{1}{2}\sigma^2 f_{xx}$ berasal dari quadratic variation. Tidak ada di calculus biasa.

**Aplikasi Bitcoin**: jika $S_t$ = BTC price, $f = \ln S_t$:
$$d(\ln S_t) = \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma dW_t$$
→ Log-return = $\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} Z$ dimana $Z \sim \mathcal{N}(0,1)$

### 0A.3 Black-Scholes-Merton Framework

SDE geometric Brownian motion:
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Solution: $S_t = S_0 \exp\left[\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right]$

BSM option pricing (European call):
$$C = S_0 N(d_1) - Ke^{-rT} N(d_2)$$
$$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

**Asumsi BSM yang GAGAL di Crypto**:
1. $\sigma$ constant → SALAH, crypto vol sangat time-varying
2. Continuous price → SALAH, BTC punya jumps besar
3. No transaction costs → SALAH, spread + gas fees signifikan
4. No market impact → SALAH di thin markets

### 0A.4 Lévy Processes dan Jump-Diffusion

Lévy process $X_t$ = Brownian motion + Poisson jumps:
$$X_t = \mu t + \sigma W_t + \sum_{i=1}^{N_t} J_i$$

dimana $N_t$ = Poisson process (intensitas $\lambda$), $J_i$ = jump sizes

**Merton (1976) Jump-Diffusion**:
$$dS_t = \mu S_t dt + \sigma S_t dW_t + S_{t^-}(e^{J} - 1) dN_t$$

BSM analog dengan jump correction:
$$C^{Merton} = \sum_{n=0}^{\infty} \frac{e^{-\lambda T}(\lambda T)^n}{n!} C_{BSM}\left(S_0, K, T, r_n, \sigma_n\right)$$

dimana $r_n = r - \lambda(e^{\mu_J + \sigma_J^2/2} - 1) + \frac{n\mu_J}{T}$, $\sigma_n^2 = \sigma^2 + \frac{n\sigma_J^2}{T}$

**Referensi**:
- [2305.10678] "Option pricing under jump diffusion model" — Lévy double jump, series solution
- [2604.06068] "Beyond Black-Scholes: Heston, GARCH, Jump Diffusion" — computational comparison
- [1811.11379] "Option Pricing in Regime Switching Jump Diffusion"

### 0A.5 Hawkes Process (Self-Exciting Point Process)

Intensitas $\lambda_t$ yang self-exciting:
$$\lambda_t = \mu + \sum_{t_i < t} \alpha e^{-\beta(t - t_i)}$$

dimana tiap event meningkatkan intensitas event berikutnya → clustering

**Aplikasi BTC**: price jumps, mining blocks, liquidation cascades semua exhibit Hawkes dynamics
**Referensi**: [2203.16666] "Hawkes Process Modeling of Block Arrivals in Bitcoin Blockchain"

---

## 0B. VOLATILITY MODELING

### 0B.1 ARCH/GARCH Family

**ARCH(p)** (Engle, 1982):
$$\sigma_t^2 = \omega + \sum_{i=1}^{p} \alpha_i \epsilon_{t-i}^2$$

**GARCH(1,1)** (Bollerslev, 1986) — paling umum dipakai:
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
- $\alpha$: ARCH coefficient (short-memory, reaction ke shocks)
- $\beta$: GARCH coefficient (persistence, long-memory component)
- $\alpha + \beta < 1$: stationarity condition

**Persistence**: unconditional variance $= \frac{\omega}{1 - \alpha - \beta}$

**Untuk BTC**: $\alpha + \beta$ sering mendekati 1 (near-integrated GARCH) → volatility sangat persistent

**GJR-GARCH (TGARCH)** — asymmetric leverage effect:
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \gamma \epsilon_{t-1}^2 \mathbf{1}[\epsilon_{t-1} < 0] + \beta \sigma_{t-1}^2$$
- $\gamma > 0$: negative returns → higher volatility (leverage effect)
- Crypto: leverage effect ada tapi LEMAH vs equity (pasar dua arah)

**EGARCH** (exponential):
$$\ln \sigma_t^2 = \omega + \alpha \left(\frac{|\epsilon_{t-1}|}{\sigma_{t-1}} - \sqrt{2/\pi}\right) + \gamma \frac{\epsilon_{t-1}}{\sigma_{t-1}} + \beta \ln \sigma_{t-1}^2$$
- Model log-variance → tidak perlu constraint positivity
- Lebih flexible untuk extreme events

**FIGARCH** (fractional integrated):
$$\sigma_t^2 = \omega + [1 - (1-L)^d \Phi(L)] \epsilon_t^2 + \beta \sigma_{t-1}^2$$
- $d \in (0,1)$: long memory parameter → shocks decay hyperbolic bukan exponential
- BTC: d ≈ 0.4-0.6 menurut beberapa studi

**Error distributions for BTC**:
- Normal: too thin tails → UNDERESTIMATES tail risk
- Student-t (ν df): lebih baik, ν ≈ 3-6 untuk BTC
- GED (Generalized Error Distribution): flexible
- NIG (Normal Inverse Gaussian): best fit untuk high-frequency BTC

**Referensi**:
- [1812.09452] "GARCH Evidence from Bitcoin High Frequency Data"
- [1909.04903] "Estimating Bitcoin Volatility using GARCH models" (sGARCH, iGARCH, tGARCH)
- [1906.03828] "Efficient Bayesian estimation for GARCH-type models via SMC"
- [2104.09879] "GARCH-UGH: Bias-reduced Extreme VaR estimation"

### 0B.2 Stochastic Volatility — Heston Model

$$dS_t = rS_t dt + \sqrt{v_t} S_t dW_t^S$$
$$dv_t = \kappa(\theta - v_t)dt + \xi \sqrt{v_t} dW_t^v$$
$$d\langle W^S, W^v \rangle_t = \rho \, dt$$

Parameters:
- $\kappa$: mean reversion speed of variance
- $\theta$: long-run variance
- $\xi$: vol of vol
- $\rho$: correlation (biasanya negative → leverage effect)

**Heston closed-form** (via characteristic function / Fourier):
$$C = S_0 P_1 - K e^{-rT} P_2$$
$$P_j = \frac{1}{2} + \frac{1}{\pi} \int_0^\infty \text{Re}\left[\frac{e^{-i\phi \ln K} f_j(\phi)}{i\phi}\right] d\phi$$

**Feller condition**: $2\kappa\theta > \xi^2$ (variance stays positive a.s.)
- Sering dilanggar di BTC! → variance bisa mencapai zero

**Referensi**: [1004.3299] "Valuation equations for stochastic volatility models"

### 0B.3 Rough Volatility — rBergomi Model ⭐ CRITICAL FOR BITCOIN

**Motivasi**: Volatility proses exhibit power-law decay of autocorrelation → $H < 1/2$ (rough)

**Fractional Brownian Motion** (fBm) dengan Hurst exponent H:
$$\langle B_t^H B_s^H \rangle = \frac{1}{2}(t^{2H} + s^{2H} - |t-s|^{2H})$$
- $H = 1/2$: ordinary Brownian motion (no memory)
- $H > 1/2$: persistent (trending)
- $H < 1/2$: rough/anti-persistent (mean-reverting, rough paths)

**Bergomi (2016) Rough Bergomi Model**:
$$dS_t = S_t \sqrt{v_t} dW_t$$
$$v_t = v_0 \exp\left(\eta \tilde{W}_t^H - \frac{1}{2}\eta^2 t^{2H}\right)$$
$$\tilde{W}_t^H = \sqrt{2H} \int_0^t (t-s)^{H-1/2} dW_s^v$$

dimana $\tilde{W}^H$ adalah Riemann-Liouville fBm (Volterra kernel)

**BTC-specific finding** (Caruso 2026 [2608.27575]):
- Calibrated to 30 Deribit surfaces (May 2022 - March 2025)
- H ≈ 0.01 - 0.06 (extremely rough, near lower bound)
- vs equity: H ≈ 0.10 - 0.15
- Calibration RMSE: 22.83 pp (Hybrid+Mixed) vs 41.76 pp (Cholesky+Euler)

**Simulation schemes**:
1. Cholesky: exact but O(n²) → slow untuk large grids
2. Hybrid Scheme (Bennedsen 2017): fast + accurate, decomposes kernel
3. Mixed Estimator (McCrickerd-Pakkanen 2018): variance reduction

**Referensi**:
- [2608.27575] "Pricing and Calibration of Bitcoin Inverse Options via rBergomi" ⭐
- [1610.08878] "Asymptotics for rough stochastic volatility models"
- [2210.12393] "Rough Hawkes Heston stochastic volatility model"
- [2307.02582] "Estimating roughness exponent from discrete observations"

### 0B.4 Realized Volatility Measures

**Realized Variance** (Barndorff-Nielsen & Shephard):
$$RV_t = \sum_{j=1}^{n} r_{t,j}^2$$
dimana $r_{t,j}$ = intraday return ke-j pada hari t

**Bipower Variation** (robust to jumps):
$$BV_t = \frac{\pi}{2} \sum_{j=2}^{n} |r_{t,j}| |r_{t,j-1}|}$$

**Jump component**: $J_t = \max(RV_t - BV_t, 0)$

**HAR-RV model** (Corsi 2009):
$$RV_{t+1} = c + \beta_D RV_t + \beta_W \overline{RV}_{t}^{(5)} + \beta_M \overline{RV}_{t}^{(22)} + \epsilon_{t+1}$$
- Daily, weekly, monthly components → heterogeneous agents
- Simple tapi sangat powerful out-of-sample

**VOLARE platform** [2602.19732]: open database realized volatility standardized dari HF data

**Referensi**:
- [1912.05228] "Risk of Bitcoin Market: Volatility, Jumps, and Forecasts"

---

## 0C. RISK MEASURES

### 0C.1 VaR vs CVaR

**Value at Risk (VaR)** at confidence level $\alpha$:
$$VaR_\alpha(X) = -\inf\{x : P(X \leq x) > 1-\alpha\} = -F_X^{-1}(1-\alpha)$$

Masalah VaR:
1. Tidak coherent (violates sub-additivity)
2. Blind terhadap severity di tail
3. Optimizing VaR → NP-hard, non-convex

**Expected Shortfall (CVaR / ES)** — rata-rata loss yang melebihi VaR:
$$CVaR_\alpha(X) = -E[X | X \leq -VaR_\alpha(X)] = \frac{1}{1-\alpha} \int_\alpha^1 VaR_u(X) du$$

Properties CVaR:
- Coherent (Artzner 1999): monotone, sub-additive, positive homogeneous, translational invariant
- Convex → tractable optimization
- Informatif tentang extreme loss magnitude

**Student-t CVaR** (closed form):
$$CVaR_\alpha = \mu + \sigma \cdot \frac{f_\nu(t_\nu^{-1}(\alpha))}{1-\alpha} \cdot \frac{\nu + (t_\nu^{-1}(\alpha))^2}{\nu - 1}$$

**EVaR (Entropic Value-at-Risk)** — tighter upper bound, strongly monotone:
$$EVaR_\alpha(X) = \inf_{z>0} \left\{z^{-1} \ln\frac{M_X(z)}{1-\alpha}\right\}$$
**Referensi**: [2608.18022] "EVaR portfolio optimization for tempered stable Lévy processes" — directly applicable ke BTC (heavy tails + jumps)

### 0C.2 Risk Measures Hierarchy

$$VaR_\alpha \leq CVaR_\alpha \leq EVaR_\alpha$$

Semakin kanan = semakin conservative, lebih besar penalti untuk extreme events.

**Untuk crypto**: CVaR minimum, EVaR jika ada tempered stable Lévy returns

**Referensi**:
- [0904.0870] "Risk Measures in Quantitative Finance" — komprehensif
- [1102.5665] "VaR, CVaR with Multivariate Student-T"
- [2608.17481] "Nonparametric VaR+CVaR estimator for high dimensions"
- [2104.09879] "GARCH-UGH: Bias-reduced Extreme VaR"

### 0C.3 Greeks

| Greek | Formula | Intuition |
|-------|---------|-----------|
| Delta $\Delta$ | $\partial C / \partial S$ | Sensitivity ke price |
| Gamma $\Gamma$ | $\partial^2 C / \partial S^2$ | Rate of change of delta |
| Vega $\mathcal{V}$ | $\partial C / \partial \sigma$ | Sensitivity ke vol |
| Theta $\Theta$ | $\partial C / \partial t$ | Time decay |
| Rho $\rho$ | $\partial C / \partial r$ | Sensitivity ke interest rate |

BSM Greeks:
- $\Delta_{call} = N(d_1)$, $\Delta_{put} = N(d_1) - 1$
- $\Gamma = \frac{n(d_1)}{S\sigma\sqrt{T}}$
- $\mathcal{V} = S n(d_1) \sqrt{T}$
- $\Theta_{call} = -\frac{S n(d_1) \sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2)$

**Second-order PnL approximation** (Taylor expansion):
$$dV \approx \Delta \cdot dS + \frac{1}{2}\Gamma \cdot (dS)^2 + \mathcal{V} \cdot d\sigma + \Theta \cdot dt$$

---

## 0D. PORTFOLIO OPTIMIZATION

### 0D.1 Mean-Variance (Markowitz)

$$\min_w \quad w^T \Sigma w$$
$$s.t. \quad w^T \mu = \mu^* \quad \text{(return target)}$$
$$\quad\quad w^T \mathbf{1} = 1 \quad \text{(fully invested)}$$

**Efficient frontier**: semua $(σ_p, μ_p)$ yang optimal.

**Masalah di crypto**:
1. $\Sigma$ tidak stabil → estimation error → poor OOS performance
2. Returns non-normal → mean-variance framework incomplete
3. Concentration risk tinggi

### 0D.2 Mean-CVaR Optimization

Rockafellar & Uryasev (2000) — CVaR dapat diformulasikan sebagai LP:

$$\min_{w, \xi} \quad \xi + \frac{1}{(1-\alpha)T} \sum_{t=1}^T \max(-r_t^T w - \xi, 0)$$
$$s.t. \quad w^T \mathbf{1} = 1, \quad w \geq 0 \quad \text{(optional long-only)}$$

**Referensi**: [1308.2324] "Optimal Dynamic Portfolio with Mean-CVaR Criterion"

### 0D.3 Kelly Criterion

Expected log-growth maximization:
$$f^* = \arg\max_f E[\ln(1 + f \cdot r)]$$

Single asset: $f^* = \frac{\mu}{\sigma^2}$ (untuk normally distributed returns)

**Fractional Kelly**: $f_{practical} = \frac{f^*}{4}$ atau $\frac{f^*}{2}$
- Alasan: estimation error → overbet → ruin
- Rule of thumb untuk crypto: quarter-Kelly (25% dari Kelly optimal)

**Multi-asset Kelly** (Peterson framework):
$$f^* = \Sigma^{-1} \mu$$

**Referensi**:
- [1710.00431] "Kelly's Criterion in Portfolio Optimization: A Decoupled Problem"
- [2109.10814] "Fractional Growth Portfolio Investment" — comprehensive review
- [1806.05293] "Generalized framework for applying Kelly to stock markets"
- [2507.05994] "Beating Best Constant Rebalancing Portfolio: Generalization of Kelly"

### 0D.4 Risk Parity

Equalize risk contribution tiap aset:
$$RC_i = w_i \cdot \frac{(\Sigma w)_i}{w^T \Sigma w} = \frac{1}{n} \quad \forall i$$

**Crypto context**: risk parity dengan CVaR contribution (tidak variance)

### 0D.5 Black-Litterman

Combines:
- Market equilibrium returns: $\Pi = \lambda \Sigma w_{mkt}$
- Investor views: $Q = P\mu + \epsilon$, $\epsilon \sim \mathcal{N}(0, \Omega)$

BL posterior:
$$E[\mu | Q] = [(\tau\Sigma)^{-1} + P^T\Omega^{-1}P]^{-1}[(\tau\Sigma)^{-1}\Pi + P^T\Omega^{-1}Q]$$

**Untuk crypto**: $\Pi$ dari market cap weighting, views dari on-chain signals + macro

---

## 0E. COINTEGRATION & PAIRS TRADING

### 0E.1 Engle-Granger Cointegration

Series $X_t, Y_t$ cointegrated jika:
1. Keduanya $I(1)$ (unit root, non-stationary)
2. Linear combination $Z_t = Y_t - \beta X_t$ adalah $I(0)$ (stationary)

**Test procedure**:
1. ADF test pada $X_t, Y_t$ → confirm $I(1)$
2. Regress $Y_t$ pada $X_t$ → get $\hat{\beta}$, residual $\hat{Z}_t$
3. ADF test pada $\hat{Z}_t$ → confirm $I(0)$

**Johansen test**: multivariate, dapat detect multiple cointegration vectors

### 0E.2 Ornstein-Uhlenbeck Spread Process

Setelah menemukan cointegration, spread $Z_t$ modeled as OU:
$$dZ_t = \kappa(\mu - Z_t)dt + \sigma_Z dW_t$$

Parameters dari MLE atau kalman filter:
- $\kappa$: mean reversion speed (half-life = $\ln 2 / \kappa$)
- $\mu$: long-run mean
- $\sigma_Z$: volatility of spread

**Trading rules** (Bertram optimal [2102.04160]):
- Enter long: $Z_t < \mu - s_e \cdot \sigma_Z$
- Enter short: $Z_t > \mu + s_e \cdot \sigma_Z$
- Exit: $Z_t = \mu \pm s_x \cdot \sigma_Z$
- Optimize $s_e, s_x$ untuk maximize expected profit per unit time

### 0E.3 Kalman Filter (Dynamic β)

β tidak constant → menggunakan Kalman filter untuk time-varying hedge ratio:

State: $\beta_t$ (hedge ratio)
Measurement: $Y_t = \alpha + \beta_t X_t + \epsilon_t$
Transition: $\beta_t = \beta_{t-1} + w_t$ ($w_t$ = process noise)

→ Real-time estimates β_t yang adapt ke changing relationships

**Referensi**:
- [2109.10662] "Dynamic Cointegration-Based Pairs Trading in Cryptocurrency Market" — Engle-Granger, KSS, Johansen di crypto
- [2305.06961] "Copula-Based Trading of Cointegrated Cryptocurrency Pairs" — non-linear copula approach
- [2102.04160] "Bertram's Pairs Trading Strategy with Bounded Risk" — OU optimal thresholds

---

## 0F. KEY MATHEMATICAL RESULTS UNTUK CRYPTO

### Fat Tails — Power Law vs Gaussian

Return distribution Bitcoin: empirical kurtosis >> 3

**Power-law tail**: $P(|r| > x) \sim x^{-\alpha}$ dimana $\alpha \approx 3-4$ untuk BTC
→ Finite variance, infinite higher moments

**q-Gaussian** (Tsallis): $P(r) \propto [1 - (1-q)\beta r^2]^{1/(1-q)}$
→ Recovers Gaussian untuk $q \to 1$, power-law tails untuk $q > 1$

Paper [2608.30219]: q-Gaussian global fit vs power-law tail advantages → aktif debat

### Tempered Stable Lévy Distributions

Combines:
- Power-law tails (like stable) tapi dengan finite moments (tempered)
- Better fit untuk crypto returns vs normal, vs pure stable

$\text{KS fit: tempered stable} > \text{normal inverse gaussian} > \text{Student-t} > \text{normal}$

**Referensi**: [2608.18022] — EVaR optimization specifically for tempered stable returns

---

## PAPER DATABASE — FASE 0

| ID | Judul (singkat) | Relevansi | Priority |
|----|----------------|-----------|----------|
| 2604.06068 | Beyond BSM: Heston, GARCH, Jump Diffusion | Pricing models comparison | ⭐⭐⭐ |
| 2608.27575 | rBergomi Bitcoin Inverse Options | Rough vol di BTC | ⭐⭐⭐ |
| 2608.18022 | EVaR for tempered stable Lévy | Risk measure advanced | ⭐⭐⭐ |
| 2210.12393 | Rough Hawkes Heston SV model | Jump + rough vol combined | ⭐⭐⭐ |
| 1812.09452 | GARCH Evidence Bitcoin HF Data | GARCH BTC empirical | ⭐⭐⭐ |
| 1909.04903 | Estimating BTC vol using GARCH | GARCH variants BTC | ⭐⭐⭐ |
| 2109.10662 | Dynamic Cointegration Pairs Crypto | Pairs trading crypto | ⭐⭐⭐ |
| 2305.06961 | Copula Pairs Trading Crypto | Non-linear cointegration | ⭐⭐ |
| 1710.00431 | Kelly Criterion Portfolio Opt | Kelly multi-asset | ⭐⭐ |
| 2109.10814 | Fractional Growth Portfolio | Fractional Kelly | ⭐⭐ |
| 1102.5665 | VaR CVaR Student-T | Risk measures | ⭐⭐ |
| 1906.03828 | Bayesian GARCH via SMC | GARCH estimation | ⭐⭐ |
| 2104.09879 | GARCH-UGH extreme VaR | Extreme risk GARCH | ⭐⭐ |
| 2305.10678 | Option pricing jump diffusion Lévy | Jump pricing | ⭐⭐ |
| 1610.08878 | Asymptotics rough SV | Rough vol theory | ⭐⭐ |
| 2305.06961 | Copula cointegration crypto | Pairs trading | ⭐⭐ |
| 2102.04160 | Bertram pairs trading OU | OU optimal threshold | ⭐⭐ |
| 1806.05293 | Kelly generalized stock market | Multi-asset Kelly | ⭐ |
| 2507.05994 | Beating best CRP: Kelly generalized | Kelly adaptive | ⭐ |
| 2608.30219 | Power-law vs q-Gaussian tails | Fat tail distributions | ⭐ |

---

*Next: fase0_implementations.py — Python implementasi semua model di atas*
*Then: Fase 1 — Market Structure & Data Pipeline*
