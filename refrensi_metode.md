# 📚 Dokumentasi Referensi Metode & Jurnal Kuantitatif
## Pemetaan Riset Ilmiah terhadap Pipeline Notebook: `bitcoin_quant_forecasting_master_pipeline.ipynb`

Dokumen ini memetakan setiap jurnal ilmiah yang tersimpan di dalam folder `C:\Mirza Personal\crypto quant\jurnal\`, bagian (*section*) implementasinya di Jupyter Notebook, cara kerja matematisnya, dan implikasi empirisnya terhadap hasil *forecast* Bitcoin 10 tahun (2016–2026).

---

## 1. Jurnal Seleksi Fitur & Eliminasi Derau Mikrostruktur
* **File PDF**: `jurnal/2602.00776_explainable_patterns_crypto_microstructure.pdf`
* **Judul**: *Explainable Patterns in Cryptocurrency Microstructure: SHAP Dependence Across Assets*
* **ArXiv ID**: `arXiv:2602.00776` (2026)
* **Dipakai pada IPYNB**: **Section 3 (Feature Expansion & SHAP Permutation Pruning)**

### A. Tentang Apa & Cara Kerjanya:
Jurnal ini meneliti bentuk ketergantungan non-linear (*SHAP dependency shape*) pada mikrostruktur pasar crypto. Penulis membuktikan bahwa menumpuk puluhan indikator teknikal justru memicu *curse of dimensionality* (model pohon menghafal *spurious correlations*).
Cara kerjanya:
1. Hitung kontribusi marjinal fitur menggunakan teknik permutasi: nilai sebuah fitur diacak (*shuffled*), lalu diukur penurunan akurasi model (*out-of-sample loss*).
2. Jika pengacakan fitur tidak menurunkan akurasi, fitur tersebut diklasifikasikan sebagai **derau (*noise*)** dan wajib dibuang.

### B. Implikasi terhadap Hasil Forecast:
* Dari 19 fitur kandidat, **14 fitur berhasil dibuang** (termasuk RSI 14 harian standar, MACD cepat, dan Bollinger Bands).
* Menyisakan **5 Fitur Inti (SHAP-5)**: `rsi_90`, `dist_ema50`, `spread_50_200`, `vol_21`, dan `power_law_res`.
* **Dampak Hasil**: Memangkas fitur sampah menaikkan akurasi out-of-sample di horizon 30 hari secara drastis dari **$52.44\% \to 58.96\%$ ($+6.52\%$)**.

---

## 2. Jurnal Dekomposisi Spektral & Rezim Pasar Adaptif
* **File PDF**: `jurnal/2608.17342_mofe_fourier_neural_operators.pdf`
* **Judul**: *MoFE: A Novel Mixture-of-Experts Framework with Fourier Neural Operators for Cryptocurrency Forecasting*
* **ArXiv ID**: `arXiv:2608.17342` (2026)
* **Dipakai pada IPYNB**: **Section 5 (Deteksi Rezim Pasar HMM 4-State) & Section 9 (Soft-Blend Hybrid)**

### A. Tentang Apa & Cara Kerjanya:
Jurnal ini memecahkan masalah *phase-lag* (keterlambatan sinyal indikator bergerak di belakang harga) pada aset non-stasioner seperti Bitcoin.
Cara kerjanya:
1. Membagi dinamika pasar ke dalam sub-ruang frekuensi dan rezim volatilitas terpisah.
2. Menggunakan arsitektur *Mixture-of-Experts (MoE)*: sub-model yang berbeda diaktifkan tergantung apakah pasar sedang berada di fase tenang (*Low-Vol*) atau fase ekspansi momentum (*High-Vol Expansion*).
3. Diadaptasi ke pipeline kita menggunakan **Gaussian Hidden Markov Model (HMM) 4-State** (`hmmlearn`) untuk mengidentifikasi 4 kondisi psikologis pasar: Capitulation, Accumulation, Momentum, dan Euphoria.

### B. Implikasi terhadap Hasil Forecast:
* Model tidak lagi memaksakan satu aturan kaku di semua kondisi pasar.
* Pada fase *Low-Vol Accumulation*, model memprioritaskan sinyal *mean-reversion* (`dist_ema50`). Pada fase *Momentum Expansion*, model memprioritaskan sinyal *breakout* (`spread_50_200` & `rsi_90`).
* **Dampak Hasil**: Menjadi fondasi model **Soft-Blend Hybrid 30 Hari**, menaikkan akurasi hingga **$62.38\%$** dan Macro F1 ke **$0.5962$**.

---

## 3. Jurnal Rezim Sentimen & Ekstremitas Pasar
* **File PDF**: `jurnal/2602.07018_extremity_premium_sentiment_regimes.pdf`
* **Judul**: *The Extremity Premium: Sentiment Regimes and Adverse Selection in Cryptocurrency Markets*
* **ArXiv ID**: `arXiv:2602.07018` (2026)
* **Dipakai pada IPYNB**: **Section 4 (Integrasi Sentimen Fear & Greed) & Section 6 (Penyeimbangan Asimetris)**

### A. Tentang Apa & Cara Kerjanya:
Jurnal ini menganalisis dataset harian *Alternative.me Crypto Fear & Greed Index* (2018–2026) dan mendokumentasikan fenomena **"Extremity Premium"**.
Cara kerjanya:
1. Membuktikan bahwa hubungan sentimen kerumunan bersifat kurva-U (*U-shaped* non-linear).
2. Pada kondisi netral (F&G 40–60), sentimen tidak memiliki daya prediksi.
3. Namun saat F&G menyentuh zona ekstrem ($<20$ Extreme Fear atau $>80$ Extreme Greed), volatilitas masa depan dan ketidakpastian melonjak tajam melampaui volatilitas historis (*adverse selection premium*).

### B. Implikasi terhadap Hasil Forecast:
* Melahirkan fitur `fng_extreme` dan aturan penalti bobot asimetris $W_{\text{bear}} = 1.3\times - 1.6\times$ saat pasar dilanda ketakutan ekstrem.
* **Dampak Hasil**: Menghilangkan fenomena *long-only bias*, mendongkrak skor **Bear F1 (kemampuan mendeteksi koreksi/crash) dari $0.00 \to 0.5020 - 0.5734$**.

---

## 4. Jurnal Ketidakpastian & Selective Trading (Conformal Prediction)
* **File PDF**: `jurnal/2507.05470_temporal_conformal_prediction_bitcoin.pdf`
* **Judul**: *Temporal Conformal Prediction for Bitcoin Volatility and Uncertainty-Aware Intervals*
* **ArXiv ID**: `arXiv:2507.05470` (2025)
* **Dipakai pada IPYNB**: **Section 7 (Seleksi Sinyal Presisi: Setup A vs Setup B)**

### A. Tentang Apa & Cara Kerjanya:
Jurnal ini membahas *distribution-free uncertainty quantification* untuk Bitcoin time-series.
Cara kerjanya:
1. Alih-alih memaksa model menebak arah setiap hari, hitung jarak keyakinan model terhadap ambang keacakan: $\text{Confidence} = |\hat{P} - 0.5|$.
2. Buang sinyal yang berada di area ketidakpastian tinggi (sinyal ragu-ragu di sekitar $0.5$), dan pasang status **WAIT / CASH**.
3. Eksekusi posisi HANYA jika keyakinan model masuk ke persentil teratas (*Conformal Selective Gating*).

### B. Implikasi terhadap Hasil Forecast:
* **Horizon 7 Hari (Top 20% Gate)**: Akurasi melonjak dari $52.51\% \to \mathbf{56.14\%}$.
* **Horizon 14 Hari (Top 35% Gate)**: Akurasi melonjak dari $55.49\% \to \mathbf{61.50\%}$ ($+6.01\%$) dengan Sharpe Ratio naik dari $0.31 \to \mathbf{0.90}$.
* Mengeliminasi *overtrading* dan melindungi modal dari pasar *choppy / sideways*.

---

## 5. Jurnal Pelabelan Target Finansial & Ambang Profit
* **File PDF**: `jurnal/2608.26174_profit_optimized_thresholds.pdf`
* **Judul**: *Forecasting Economically Significant Bitcoin Moves: A Multi-Scale TCN with Profit-Optimized Thresholds*
* **ArXiv ID**: `arXiv:2608.26174` (2026)
* **Dipakai pada IPYNB**: **Section 8 (Eksperimen Frontier & Dynamic Barriers)**

### A. Tentang Apa & Cara Kerjanya:
Jurnal ini mengkritik kebiasaan pelabelan biner naif ($\text{Return} > 0\%$). Di pasar riil, pergerakan kecil $+0.1\%$ habis termakan biaya transaksi (*fee* $0.1\%$ per putaran) dan *slippage*.
Cara kerjanya:
1. Mengubah target prediksi menjadi pergerakan signifikan secara ekonomi (*Economically Significant Moves*):
   $$y_t = \mathbb{I}\left(\frac{P_{t+H} - P_t}{P_t} > \max\left(5\%, 1.5 \times \text{ATR}_{14}\sqrt{H}\right)\right)$$
2. Ambang batas disesuaikan secara dinamis mengikuti volatilitas pasar riil (*Dynamic Volatility Hurdle*).

### B. Implikasi terhadap Hasil Forecast:
* Nilai area di bawah kurva ROC (*Mean AUC*) melonjak drastis dari **$0.5657 \to \mathbf{0.7929}$**.
* Menggandakan efisiensi risiko: **Sharpe Ratio melonjak dari $1.60 \to \mathbf{2.68}$**, dengan tingkat keberhasilan pergerakan ekonomi besar (*Hit Rate $>5\%$*) mencapai **$34.73\%$**.

---

## 6. Jurnal Likuiditas On-Chain & Arus Bursa
* **File PDF**: `jurnal/2411.06327_onchain_flows_exchange_liquidity.pdf`
* **Judul**: *Return and Volatility Forecasting Using On-Chain Flows in Cryptocurrency Markets*
* **ArXiv ID**: `arXiv:2411.06327` (2024)
* **Dipakai pada IPYNB**: **Section 4 (Arus Likuiditas On-Chain & Institusi)**

### A. Tentang Apa & Cara Kerjanya:
Jurnal ini membuktikan secara empiris bahwa pergerakan cadangan koin di bursa (*exchange reserves*) dan aliran stablecoin memicu efek penipisan likuiditas (*supply shock*).
Cara kerjanya:
1. Penarikan koin dari bursa ke *cold storage* memicu efek kelangkaan pasokan di orderbook.
2. Diadaptasi ke pipeline kita menggunakan **Coinbase Premium Gap** (selisih Coinbase USD institusi AS vs Binance USDT ritel) dan **Spot ETF Net Flow Volume** (arus beli BlackRock IBIT / Fidelity FBTC).

### B. Implikasi terhadap Hasil Forecast:
* Pada horizon 7–14 hari, *Coinbase Premium Gap* menghasilkan Information Coefficient **$\text{IC} = +0.0775$**.
* Pada horizon 30 hari, penyerapan ke *cold storage* menaikkan akurasi arah dari **$56.52\% \to \mathbf{59.07\%}$ ($+2.55\%$)** dan memangkas kesalahan estimasi harga (MAPE) hingga $-5.7\%$.

---

## 7. Metodologi Tambahan: Meta-Labeling & Fractional Differentiation (Marcos López de Prado)
* **Referensi Buku**: *Advances in Financial Machine Learning* (Wiley, 2018)
* **Dipakai pada IPYNB**: **Section 8 (Eksperimen Frontier) & Section 9 (Arsitektur Juara)**

### A. Fractional Differentiation ($d^* = 0.20$):
* Menghitung turunan fraksional binomial tak hingga untuk mencapai stasioneritas tanpa menghapus memori harga.
* **Hasil**: Menjaga **$94.02\%$ korelasi memori harga** dengan status stasioner ADF ($p = 0.0175$), menghasilkan korelasi *mean-reversion* kuat **$\text{IC} = -0.1459$**.

### B. Meta-Labeling (Two-Stage Modeling):
* Model Primer menebak arah (Long/Short), Model Sekunder bertindak sebagai verifikator (*gatekeeper*) yang menilai probabilitas apakah trade tersebut akan profit bersih setelah dipotong *fee*.
* **Hasil**: Melesatkan **Sharpe Ratio 14 hari dari $+0.65 \to \mathbf{+1.16}$ (+78% Sharpe)** dengan Bull F1 tembus **$0.7061$**.

---

## 📊 Matriks Ringkasan Kontribusi Jurnal terhadap Akurasi Final

| Jurnal / Metode | Fokus Pemecahan Masalah | Horizon Paling Berdampak | Skor Akhir yang Dicapai |
|---|---|:---:|:---:|
| **SHAP Microstructure (`2602.00776`)** | Membuang derau 14 indikator sampah | 14 Hari & 30 Hari | Akurasi naik $+6.5\%$ |
| **Conformal Gating (`2507.05470`)** | Selective trading (hanya trade saat yakin) | 7 Hari & 14 Hari | Akurasi **$61.50\%$** (Gate 35%) |
| **Extremity Premium (`2602.07018`)** | Proteksi crash & penyeimbangan Bear F1 | 14 Hari & 30 Hari | Bear F1 naik dari $0.00 \to \mathbf{0.573}$ |
| **HMM & MoFE (`2608.17342`)** | Adaptasi rezim pasar non-stasioner | 30 Hari | Akurasi **$62.38\%$** (Soft-Blend) |
| **Dynamic Vol Hurdle (`2608.26174`)** | Eliminasi transaksi rugi fee | 7 Hari & 14 Hari | AUC melonjak ke **$0.7929$** |
| **Veto-Consensus Makro & SMA 3M** | Validasi silang tren struktural kuartalan | 90 Hari | Rekor tertinggi **$67.61\%$** (Bull F1 $0.755$) |
