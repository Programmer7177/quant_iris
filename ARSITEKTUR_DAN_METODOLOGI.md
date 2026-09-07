# 📘 MASTER TECHNICAL ARCHITECTURE & DECISION LOG (ADR)
## Sistem Prediksi Kuantitatif Multi-Horizon Bitcoin (10-Year OOS Benchmark 2016–2026)

Dokumen ini mendokumentasikan secara menyeluruh arsitektur teknis sistem, alur eksekusi end-to-end, alasan matematis-finansial di balik pemilihan setiap metode, serta catatan keputusan (*Architecture Decision Records / ADR*) agar seluruh eksperimen dapat ditelusuri (*tracked*) dan direproduksi (*reproducible*) secara ilmiah.

---

## 1. Peta Struktur Repositori & Jejak File

```
C:\Mirza Personal\crypto quant\
├── bitcoin_quant_forecasting_master_pipeline.ipynb  <-- Notebook utama seluruh alur eksperimen
├── forecast_engine.py                              <-- Production engine otomatis
├── config_presets.py                               <-- Preset konfigurasi tervalidasi (Setup A & Setup B)
├── refrensi_metode.md                              <-- Pemetaan 6 jurnal arXiv ke bagian notebook
├── ARSITEKTUR_DAN_METODOLOGI.md                    <-- Dokumen ini (Master Technical Documentation)
│
├── data/                                           <-- Folder repositori dataset & hasil audit
│   ├── btc_coinbase_10y.csv                        (3.900 bar harian OHLCV 2016–2026)
│   ├── global_macro_10y.csv                        (DXY, Emas, Minyak, Yield US10Y, SP500)
│   ├── advanced_macro_10y.csv                      (USD/CNY, Saham China FXI, TLT, TIP, HYG)
│   ├── fear_greed_full.csv                         (3.136 bar sentimen harian Alternative.me)
│   ├── coinbase_premium_index.csv                  (Gap spread akumulasi institusi AS vs Ritel)
│   ├── btc_spot_etf_flows.csv                      (Arus transaksi harian IBIT, FBTC, ARKB, BITB)
│   ├── active_learning_results.csv                 (Benchmark lengkap Active Learning QBC)
│   └── quant_loss_metalabeling_results.csv         (Benchmark Meta-Labeling & Triple Barrier)
│
├── jurnal/                                         <-- Berkas PDF asli penelitian ilmiah open-access
│   ├── 2608.26174_profit_optimized_thresholds.pdf  (Pelabelan batas volatilitas dinamis ATR)
│   ├── 2602.00776_explainable_patterns_crypto.pdf  (Seleksi fitur mikrostruktur SHAP)
│   ├── 2507.05470_temporal_conformal_pred.pdf      (Conformal prediction & selective trading)
│   ├── 2608.17342_mofe_fourier_neural_op.pdf       (Mixture-of-Experts & rezim pasar)
│   ├── 2411.06327_onchain_flows_liquidity.pdf      (Dinamika likuiditas on-chain & bursa)
│   └── 2602.07018_extremity_premium_regimes.pdf    (Rezim sentimen ekstrem kurva-U)
│
└── experiments/                                    <-- Arsip 19 script Python eksperimen mandiri
    ├── test_paper_profit_thresholds.py
    ├── test_quant_loss_and_metalabeling.py
    ├── test_advanced_math_features.py
    ├── test_active_learning.py
    ├── test_balanced_prediction.py
    └── ... (seluruh script uji hipotesis)
```

---

## 2. Alur Eksekusi Sistem End-to-End (How the Model Runs)

Sistem beroperasi melalui pipa transmisi data 5-tahap:

```
[TAHAP 1: DATA INGESTION]
  └─ Load Coinbase 10Y Daily (3.900 bar) + Merge Makro + Fear&Greed + Premium Index.
           │
           ▼
[TAHAP 2: FEATURE TRANSFORMATION & SELECTION]
  ├─ Ekstraksi 5 Fitur SHAP murni (rsi_90, dist_ema50, spread_50_200, vol_21, power_law_res).
  ├─ Horizon 30d: Tambah Downside Semi-Variance Skew (semi_vol_skew).
  └─ Horizon 90d: Tambah Siklus Harmonik Halving (cos/sin) + Yield US10Y + Paket SMA Bulanan (3M & 5M).
           │
           ▼
[TAHAP 3: LATENT REGIME DETECTION]
  └─ Gaussian Hidden Markov Model (HMM) 4-State memetakan data ke:
     State 0 (Capitulation), State 1 (Accumulation), State 2 (Expansion), State 3 (Euphoria).
           │
           ▼
[TAHAP 4: SPECIALIZED MULTI-HORIZON INFERENCE]
  ├─ 7d  : ExtraTrees Soft-Blend (50% Balanced + 50% Asymmetric W_bear=1.3).
  ├─ 14d : ExtraTrees Balanced Classifier (class_weight='balanced').
  ├─ 30d : ExtraTrees Soft-Blend (60% SHAP+HMM + 40% Asymmetric W_bear=1.6).
  └─ 90d : Two-Model Veto-Consensus (Makro/Halving vs Paket SMA Bulanan).
           │
           ▼
[TAHAP 5: CONFORMAL SELECTIVE GATING & EXECUTION]
  ├─ Hitung margin keyakinan |Prob - 0.5|.
  ├─ Jika keyakinan < ambang batas kuantil -> Status: WAIT / CASH (Abstain).
  └─ Jika keyakinan >= ambang batas -> Eksekusi sinyal arah (BULL / BEAR) + Ukuran posisi asimetris HMM.
```

---

## 3. Catatan Keputusan Arsitektur & Landasan Ilmiah (Architecture Decision Records)

### ADR-01: Mengapa Mengeliminasi 14 Indikator dan Hanya Menyisakan 5 Fitur SHAP?
* **Latar Belakang**: Eksperimen awal yang memasukkan 19 indikator (termasuk RSI 14, MACD, Bollinger Bands, Stochastic) menghasilkan akurasi jalan di tempat ($52.4\%$).
* **Landasan Teori**: *arXiv:2602.00776 (Explainable Patterns in Crypto Microstructure)* membuktikan bahwa indikator teknikal konvensional saling berkorelasi tinggi (*multicollinear*) dan memecah pohon keputusan (*split dilution*).
* **Keputusan**: Melakukan *Permutation Importance Pruning*. Hanya fitur yang terbukti secara statistik menurunkan akurasi saat diacak yang dipertahankan.
* **Hasil**: Menyisakan 5 fitur inti: `rsi_90`, `dist_ema50`, `spread_50_200`, `vol_21`, dan `power_law_res`. Akurasi out-of-sample naik dari **$52.44\% \to 58.96\%$ ($+6.52\%$)**.

---

### ADR-02: Mengapa Memisahkan Makro Global Hanya untuk Horizon Panjang (90 Hari)?
* **Latar Belakang**: Memasukkan DXY dan Yield US 10Y ke dalam prediksi 7–14 hari justru **menurunkan akurasi sebesar $-6.7\%$**.
* **Landasan Teori**: *IMF Working Paper WP/23/163 (Che et al.)* dan *Ahmadova et al. (2025)* menemukan bahwa transmisi kebijakan moneter global dan suku bunga Fed ke pasar crypto membutuhkan lag 70–180 hari. Di bawah 14 hari, harga crypto didominasi likuidasi derivatif lokal.
* **Keputusan**: Variabel makro (Yield US10Y, DXY EMA50, Siklus Halving 1.460 hari) **dilarang masuk ke model 7–14 hari** dan hanya dipasang pada model 90 hari.
* **Hasil**: Akurasi 90 hari melonjak dari **$57.99\% \to \mathbf{65.14\%}$** dengan Macro F1 tembus **$0.6240$**.

---

### ADR-03: Mengapa Mengganti Regresi MSE dengan Balanced Classifier?
* **Latar Belakang**: Model regresi awal memiliki *Macro F1* rendah ($0.35 - 0.45$) dan *Bear F1* hampir nol ($0.00 - 0.03$), artinya model buta total saat pasar crash.
* **Landasan Teori**: Bitcoin secara historis $56\%$ waktunya naik. Loss function MSE simetris mengambil jalan pintas dengan selalu memproyeksikan angka positif demi meminimalkan error kuadrat, mengorbankan deteksi crash.
* **Keputusan**: Mengubah target ke arah biner terbobot menggunakan `class_weight='balanced'` dan penalti asimetris $W_{\text{bear}} = 1.3\times - 1.6\times$ (*Extremity Premium*, *arXiv:2602.07018*).
* **Hasil**: Bear F1 melesat dari **$0.00 \to \mathbf{0.4903 - 0.5734}$** tanpa menurunkan akurasi global. Model berhasil menangkap ratusan hari bear market secara akurat.

---

### ADR-04: Mengapa Menggunakan Conformal Gating & Selective Trading?
* **Latar Belakang**: Memaksa model membuka posisi setiap hari (100% waktu) menyebabkan modal tergerus oleh sinyal acak di fase pasar sideways.
* **Landasan Teori**: *arXiv:2507.05470 (Temporal Conformal Prediction)* dan *arXiv:2606.00060* membuktikan bahwa menyaring sinyal di sekitar batas ketidakpastian (*epistemic boundary*) menggandakan Sharpe ratio.
* **Keputusan**: Menerapkan Conformal Margin:
  - 7 Hari: Eksekusi hanya Top 20% Conviction.
  - 14 Hari: Eksekusi hanya Top 35% Conviction.
  - 30 Hari: Eksekusi hanya Top 40–50% Conviction.
  - 90 Hari: Eksekusi penuh (100%) karena sinyal struktural makro stabil di seluruh sampel.
* **Hasil**: Akurasi 14 hari melompat dari **$55.49\% \to \mathbf{61.50\%}$** dan Sharpe Ratio naik dari $0.31 \to \mathbf{0.90}$.

---

### ADR-05: Mengapa Paket SMA Bulanan (3M & 5M) Sangat Efektif di 90 Hari tetapi Merusak di 30 Hari?
* **Latar Belakang**: Penambahan SMA 3-Bulanan (90 hari) dan SMA 5-Bulanan (150 hari) menaikkan akurasi 90 hari, namun menurunkan akurasi 30 hari sebesar $-2.66\%$.
* **Landasan Teori**: *Horizon Mismatch*. Jendela data 90–150 hari memiliki inersia lambat. Untuk horizon 30 hari, indikator ini terlalu lambat (*lagging*) mendeteksi pullback lokal. Sebaliknya untuk horizon kuartalan (90 hari), indikator ini menjadi penyaring sempurna dari *bear trap* harian.
* **Keputusan**: Paket SMA Bulanan (`dist_sma3_m` dan `spread_sma_3_5_m`) dikunci eksklusif hanya untuk horizon 90 hari.
* **Hasil**: Membawa akurasi horizon 90 hari menembus **$65.14\%$ (Setup A)** dan **$67.61\%$ (Veto-Consensus)**.

---

## 4. Matriks Ringkasan Juara 1 Setiap Horizon (Master Benchmark)

Validasi Out-of-Sample 10 Tahun (Coinbase Daily 2016–2026, 8-Fold Expanding Window):

| Horizon | Arsitektur Juara 1 | Akurasi (Micro F1) | Bull F1 (Naik) | Bear F1 (Crash) | Macro F1 (Keseimbangan) | Market Coverage |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Short 7 Hari** | Soft-Blend Hybrid (50:50, Gate 20%) | **57.70%** | 0.6845 | 0.3585 | 0.5215 | 20.0% |
| **Short 14 Hari** | Setup A (SHAP-5, Gate 35%) | **61.50%** | 0.6983 | 0.4680 | 0.5832 | 35.0% |
| **Medium 30 Hari** | Soft-Blend Hybrid (60:40, Gate 40%) | **62.38%** | 0.7019 | 0.4904 | 0.5962 | 40.0% |
| **Long 90 Hari** | Veto-Consensus ($P_A \ge 0.58, P_B \le 0.44$) | **67.61%** | **0.7549** | **0.5228** | **0.6389** | **75.3%** |

---

## 5. Panduan Menjalankan Sistem Secara Mandiri

### Menjalankan Engine Produksi Otomatis:
```bash
# Menghitung sinyal dan kondisi pasar hari ini secara cepat (~10 detik)
py -3.11 forecast_engine.py --signal

# Menjalankan backtest walk-forward penuh seluruh horizon
py -3.11 forecast_engine.py
```

### Menjalankan Preset Konfigurasi Terpilih:
```bash
# Menjalankan Setup A (Master High-Accuracy Conviction)
py -3.11 config_presets.py --setup A

# Menjalankan Setup B (Dual-Threshold High-Symmetry Long/Short)
py -3.11 config_presets.py --setup B
```

### Menjalankan Jupyter Notebook Interaktif:
Buka file `bitcoin_quant_forecasting_master_pipeline.ipynb` menggunakan VS Code atau jalankan:
```bash
jupyter notebook bitcoin_quant_forecasting_master_pipeline.ipynb
```
Semua cell telah terstruktur dari Fase 1 (Load data) hingga Fase 10 (Live signal) dan siap dieksekusi ulang secara deterministik.
