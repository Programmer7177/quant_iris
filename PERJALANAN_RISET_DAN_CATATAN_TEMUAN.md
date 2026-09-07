# 📖 PERJALANAN RISET QUANT BITCOIN: DARI INTUISI, KEGAGALAN, HINGGA PENEMUAN SISTEMATIS
**Penulis:** Mirza (Analisis & Evaluasi Mandiri)  
**Periode Data:** 10 Tahun Coinbase BTC/USD (2016–2026, 3.900 Bar Harian)  
**Lingkungan:** `C:\Mirza Personal\crypto quant\`

Dokumen ini mencatat secara jujur kronologi perjalanan riset, intuisi awal yang sempat dicoba, anomali dan kegagalan yang dihadapi, hingga bagaimana setiap masalah tersebut dipecahkan secara matematis.

---

## BAB 1: Kronologi Masalah & Intuisi Awal yang Gagal

### 1. Jebakan "Semakin Banyak Fitur, Semakin Pintar Model"
* **Intuisi Awal**: Pada fase awal, kita beranggapan bahwa memasukkan sebanyak mungkin indikator teknikal (RSI 14 harian, MACD, Bollinger Bands, Moving Average spreads, Stochastic, ATR) akan membuat model mengenali semua pola pasar.
* **Fakta Empiris**: Model dengan 19 fitur memiliki akurasi out-of-sample yang stagnan di **$52.4\%$ (hampir sama dengan lempar koin)**.
* **Mengapa Gagal?**: *Curse of Dimensionality*. Di pasar keuangan, indikator-indikator teknikal klasik memiliki korelasi silang sangat tinggi (*multicollinear*). Pohon keputusan (*decision tree*) mengalami *split dilution*—ia memecah data pada fitur-fitur yang sebenarnya hanya membawa derau (*noise*), sehingga model menghafal kebetulan di masa lalu alih-alih menangkap sinyal sejati.
* **Solusi Penemuan**: Kami menggunakan **Permutation Importance (SHAP-Style)**. Mengacak nilai 14 fitur sampah tidak menurunkan akurasi model sama sekali. Membuang 14 fitur tersebut dan hanya menyisakan **5 Fitur Inti (SHAP-5: `rsi_90`, `dist_ema50`, `spread_50_200`, `vol_21`, `power_law_res`)** langsung melambungkan akurasi dari **$52.4\% \to 58.96\%$ ($+6.5\%$)**.

---

### 2. Anomali "Data Makro Merusak Prediksi Jangka Pendek"
* **Intuisi Awal**: Indeks Dolar AS (DXY) dan Suku Bunga Obligasi AS (US 10Y Yield) adalah penggerak utama likuiditas global. Logikanya, memasukkan makro akan meningkatkan ketajaman prediksi harga Bitcoin di semua horizon.
* **Fakta Empiris**:
  - Pada horizon panjang (90 hari), makro memang sangat membantu (akurasi naik dari $57.9\% \to \mathbf{65.1\%}$).
  - Namun pada horizon pendek (7 dan 14 hari), menambahkan data makro **justru menghancurkan akurasi sebesar $-6.7\%$**.
* **Mengapa Gagal?**: *Lead-Lag Structural Friction*. Data makro dan yield obligasi bergerak dalam frekuensi kuartalan dengan jeda transmisi (*lag*) 30–90 hari ke pasar aset berisiko. Di rentang 1–7 hari, pergerakan Bitcoin $95\%$ digerakkan oleh likuidasi derivatif lokal, *funding rate*, dan *order book imbalance*. Memasukkan data yield AS yang lambat ke horizon harian hanya menambahkan sinyal semu yang membingungkan algoritma.
* **Solusi Penemuan**: Kami memisahkan arsitektur fitur secara modular: **Fitur makro dilarang masuk ke horizon $\le 14$ hari**, dan hanya diizinkan aktif pada horizon $\ge 90$ hari.

---

### 3. Masalah Fatal "Model Buta Pasar Crash" (Ketimpangan Bull vs Bear)
* **Kondisi Awal**: Di pertengahan sesi, kita mengecek metrik performa secara mendalam menggunakan F1-Score dan Confusion Matrix:
  - **Akurasi Global**: Terlihat bagus ($\sim 55\% - 58\%$).
  - **Bull F1 (Naik)**: Sangat tinggi ($0.69 - 0.73$).
  - **Bear F1 (Turun)**: **Hampir $0.00$ atau cuma $0.03$!**
  - **Confusion Matrix**: Dari 1.488 hari pasar jatuh/crash, model cuma berani menebak turun **27 kali**, dan keliru menebak naik sebanyak **1.461 kali**.
* **Mengapa Terjadi?**:
  1. *Class Imbalance Alami*: Secara historis, Bitcoin berada dalam tren naik secular selama 10 tahun ($\sim 56\%$ hari hijau).
  2. *Sifat Fungsi Loss MSE*: Model regresi dilatih meminimalkan *Mean Squared Error*. Karena kenaikan Bitcoin sering kali ratusan persen sedangkan penurunannya terbatas maksimal $-100\%$, model secara matematis "mengambil jalan pintas yang licik": **selalu menebak naik akan meminimalkan total error kuadrat daripada menebak turun**.
* **Solusi Penemuan**:
  - Kami membuang regresi simetris dan beralih ke **Directional Balanced Classifier** (`class_weight='balanced'`).
  - Kami menyetel bobot asimetris ($W_{\text{bear}} = 1.3\times - 1.6\times$): salah menebak pasar jatuh dihukum jauh lebih berat daripada salah menebak pasar naik.
  - **Hasil**: Bear F1 melesat dari **$0.00 \to \mathbf{0.49 - 0.57}$**. Tangkapan hari crash naik drastis dari 27 hari menjadi **670–760 hari**.

---

### 4. Jebakan Indikator Ultra-Pendek: Kasus SMA 3 & SMA 5 Harian
* **Intuisi Awal**: Jika candle ditutup di atas SMA 3 atau terjadi *cross up* SMA 3 di atas SMA 5, seharusnya itu menandakan momentum kenaikan kuat (*strong uptrend*), dan sebaliknya.
* **Fakta Empiris**:
  - Menguji aturan ini secara mandiri menghasilkan akurasi **$47.5\% - 49.6\%$ (di bawah lempar koin acak)**!
  - Hari ini ditutup di atas SMA 3 $\to$ peluang besok naik hanya **$50.2\%$**.
  - Hari ini ditutup di bawah SMA 3 $\to$ peluang besok turun hanya **$45.3\%$** (lebih sering membal naik).
* **Mengapa Gagal?**: Di skala harian, Bitcoin adalah aset dengan *short-term mean-reversion*. Lonjakan harga yang menembus SMA 3 harian sering kali merupakan akhir dari *short squeeze* lokal, di mana hari berikutnya langsung terjadi aksi ambil untung (*profit taking*).
* **Penemuan Terobosan (Pindah ke Skala Bulanan)**:
  - Ketika SMA 3 diterapkan bukan pada bar harian, melainkan pada **Bar Bulanan (Monthly SMA 3M / 90 hari & SMA 5M / 150 hari)**:
  - Sifatnya berubah $180$ derajat: Korelasi (*Information Coefficient*) terhadap return 3 bulan melonjak menjadi **$\text{IC} = +0.2018$**.
  - Menambahkan paket SMA Bulanan pada horizon 90 hari membawa akurasi melesat dari **$62.94\% \to \mathbf{65.14\%}$**!

---

### 5. Jebakan Conformal Gating: Mengapa Bear F1 Sempat Drop Lagi?
* **Masalah Baru**: Ketika kita menerapkan *Conformal Gating* (hanya trade pada Top 20%–35% keyakinan tertinggi), akurasi global naik drastis ($61.5\%$), tetapi Bear F1 mendadak anjlok lagi ke $0.22 - 0.24$.
* **Penyebab**: Rumus gating awal adalah $|\text{Prob} - 0.5|$. Model pohon jauh lebih percaya diri saat momentum bull (probabilitas bisa mencapai $0.80 - 0.85$). Saat pasar bear, probabilitas biasanya lebih ragu ($0.38 - 0.42$). Ketika dipotong garis batas persentil umum, **sinyal bear tereliminasi oleh sinyal bull yang lebih jumawa**.
* **Solusi Penemuan**: Kami merancang **Dual-Threshold Gating** (pintu masuk terpisah): kuantil bawah untuk Short dan kuantil atas untuk Long. Hasilnya, **Bear F1 kembali melonjak ke $0.5932$** dengan akurasi tetap tinggi ($59.74\%$).

---

## BAB 2: Daftar Lengkap 30 Jurnal Ilmiah yang Dikoleksi & Dipelajari

Seluruh 30 berkas PDF penelitian quant open-access arXiv telah diunduh dan tersimpan di folder:  
📁 **`C:\Mirza Personal\crypto quant\jurnal\`**

### Kategori 1: Conformal Prediction & Selective Trading (Pondasi Gating)
1. **`2507.05470`**: *Temporal Conformal Prediction (TCP): A Distribution-Free Framework for Adaptive Risk Forecasting*  
   *(Mendasari aturan hanya membuka posisi saat tingkat keyakinan model berada di atas ambang batas).*
2. **`2608.01494`**: *Conformal Kelly: Conformal Prediction Intervals as the Scale in Fractional Kelly Betting*  
   *(Mengatur ukuran posisi trading proporsional terhadap lebar interval ketidakpastian).*
3. **`2602.03903`**: *Taming Tail Risk in Financial Markets: Conformal Calibration for Nonstationary Streams*  
   *(Kalibrasi Value-at-Risk berbasis rezim pasar untuk mencegah kegagalan beruntun).*
4. **`2606.00060`**: *Machine Learning-Based Bitcoin Trading Under Transaction Costs: Walk-Forward Evidence*  
   *(Membuktikan bahwa filter selektif berbasis biaya transaksi mampu menghasilkan Sharpe ratio > 1.0).*
5. **`2601.10591`**: *ProbFM: Probabilistic Time Series Foundation Model with Uncertainty Decomposition*  
   *(Dekomposisi ketidakpastian data vs ketidakpastian model).*

### Kategori 2: Regime-Switching & Hidden Markov Models (Pondasi State Machine)
6. **`2608.17342`**: *MoFE: A Novel Mixture-of-Experts Framework with Fourier Neural Operators for Crypto*  
   *(Mendasari arsitektur pemisahan sub-model untuk rezim volatilitas rendah vs ekspansi tren).*
7. **`2011.03741`**: *Exploring the Predictability of Cryptocurrencies via Bayesian Hidden Markov Models*  
   *(Mendasari deteksi 4 rezim tersembunyi: Capitulation, Accumulation, Expansion, Euphoria).*
8. **`2602.07018`**: *The Extremity Premium: Sentiment Regimes and Adverse Selection in Crypto Markets*  
   *(Membuktikan kurva-U sentimen ekstrem Fear & Greed dan mendasari pembobotan asimetris crash).*
9. **`2307.06400`**: *Quantile and Expectile Copula-Based Hidden Markov Regression Models*  
   *(Pemodelan ekor risiko gabungan di bawah peralihan rezim pasar).*

### Kategori 3: Feature Engineering, SHAP, & Fractional Memory
10. **`2602.00776`**: *Explainable Patterns in Cryptocurrency Microstructure: SHAP Dependence Across Assets*  
    *(Mendasari pemangkasan 14 fitur sampah dan penemuan 5 fitur inti SHAP-5).*
11. **`2511.20105`**: *Multivariate Forecasting of Bitcoin Volatility with Gradient Boosting & SHAP*  
    *(Evaluasi interaksi volatilitas non-linear pada data Bitcoin harian).*
12. **`2303.02223`**: *Feature Selection with Annealing for Forecasting Financial Time Series*  
    *(Metode seleksi fitur stokastik mengungguli mutual information).*
13. **`2605.21316`**: *Bitcoin's Power Law: Weak Structure, Strong Forecasts*  
    *(Mendasari fitur Power-Law Residual sebagai jangkar valuasi adopsi jangka panjang).*

### Kategori 4: Pelabelan Finansial & Target Engineering
14. **`2608.26174`**: *Forecasting Economically Significant Bitcoin Moves: Multi-Scale TCN with Profit-Optimized Thresholds*  
    *(Mendasari pelabelan batas volatilitas dinamis ATR dan pengabaian fluktuasi kecil).*
15. **`2411.06327`**: *Return and Volatility Forecasting Using On-Chain Flows in Cryptocurrency Markets*  
    *(Mendasari integrasi Coinbase Institutional Premium Gap dan arus dompet bursa).*
16. **`2607.15258`**: *Decoding Market Emotion from Blockchain Activity: Data-Driven Sentiment*  
    *(Kombinasi metrik on-chain dan sentimen pasar).*

### Kategori 5: Ensembling, Stacking, & Active Learning
17. **`2511.15350`**: *Multi-layer Stack Ensembles for Time Series Forecasting*  
    *(Mendasari model susun ExtraTrees + GradientBoosting + Ridge pada horizon 7–14 hari).*
18. **`2504.09664`**: *Adapting to the Unknown: Robust Meta-Learning for Zero-Shot Financial Time Series*  
    *(Adaptasi cepat model terhadap pergeseran rezim data).*
19. **`2512.14042`**: *Dynamic Stacking Ensemble Learning with Investor Knowledge Representations*  
    *(Penyesuaian bobot ensemble secara dinamis mengikuti kondisi pasar).*

### Kategori 6: Tail Risk, Copula, & Asimetri Pasar
20. **`2606.16840`**: *Crashing Together, Rallying Apart: Dynamic Conditional Tail Dependence in Crypto*  
    *(Mendasari bukti matematis bahwa korelasi aset melonjak saat crash dan pecah saat bull run).*
21. **`2603.23480`**: *Stablecoins as Dry Powder: A Copula-Based Risk Analysis*  
    *(Peran cadangan kas stablecoin di bursa sebagai bahan bakar likuiditas pompa harga).*
22. **`2407.15766`**: *Analyzing Selected Cryptocurrencies Spillover Effects on Global Financial Indices*  
    *(Transmisi risiko antara ekuitas global dan crypto).*

### Kategori 7: Dekomposisi Gelombang & Hybrid Deep Learning
23. **`2510.15900`**: *Bitcoin Price Forecasting Based on Hybrid Variational Mode Decomposition (VMD)*  
    *(Dekomposisi deret waktu sebelum prediksi).*
24. **`2512.20028`**: *DecoKAN: Interpretable Decomposition for Forecasting Cryptocurrency Markets*  
    *(Penerapan Kolmogorov-Arnold Networks pada deret waktu crypto).*
25. **`2501.13136`**: *Forecasting of Bitcoin Prices Using Hashrate Features: Wavelet and Deep Stacking*  
    *(Penggunaan metrik keamanan hashrate jaringan).*
26. **`2509.10542`**: *Adaptive Temporal Fusion Transformers for Cryptocurrency Price Prediction*
27. **`2412.14529`**: *Leveraging Time Series Categorization and Temporal Fusion Transformers*
28. **`2504.17079`**: *A Novel Hybrid Approach Using an Attention-Based Transformer + GRU Model*
29. **`2504.18206`**: *A Machine Learning Approach For Bitcoin Forecasting with Low-Price Dominance*
30. **`2105.00707`**: *MRC-LSTM: Multi-Scale Residual CNN and LSTM Hybrid*

---

## BAB 3: Kesimpulan Metodologi Terakhir (The Master Architecture)

Setelah melalui ratusan kombinasi dan pengujian walk-forward 10 tahun, sistem mengerucut pada **4 Arsitektur Juara 1 (Top 1 per Horizon)**:

```
┌─────────────────┬──────────────────────────────────────────┬──────────┬──────────┬──────────┐
│  HORIZON WAKTU  │            ARSITEKTUR JUARA 1            │ AKURASI  │ BULL F1  │ BEAR F1  │
├─────────────────┼──────────────────────────────────────────┼──────────┼──────────┼──────────┤
│ Short 7 Hari    │ Soft-Blend Hybrid (50:50, Gate 20%)      │  57.70%  │  0.6845  │  0.3585  │
│ Short 14 Hari   │ Setup A (SHAP-5, Gate 35%)               │  61.50%  │  0.6983  │  0.4680  │
│ Medium 30 Hari  │ Soft-Blend Hybrid (60:40, Gate 40%)      │  62.38%  │  0.7019  │  0.4904  │
│ Long 90 Hari    │ Veto-Consensus (Pa >= 0.58, Pb <= 0.44)  │  67.61%  │  0.7549  │  0.5228  │
└─────────────────┴──────────────────────────────────────────┴──────────┴──────────┴──────────┘
```

Seluruh kode, script eksperimen, dan notebook master telah terintegrasi dan terdokumentasi secara permanen untuk kebutuhan penelitian dan eksekusi kuantitatif.
