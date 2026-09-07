"""
DOWNLOAD & CATALOG KEY QUANTITATIVE FINANCE PAPERS
Downloads open-access arXiv PDFs directly into C:/Mirza Personal/crypto quant/jurnal/
"""
import os
import urllib.request

JURNAL_DIR = r"C:\Mirza Personal\crypto quant\jurnal"
os.makedirs(JURNAL_DIR, exist_ok=True)

PAPERS = [
    {
        "id": "2608.26174",
        "filename": "2608.26174_profit_optimized_thresholds.pdf",
        "title": "Forecasting Economically Significant Bitcoin Moves: A Multi-Scale TCN with Profit-Optimized Thresholds"
    },
    {
        "id": "2602.07018",
        "filename": "2602.07018_extremity_premium_sentiment_regimes.pdf",
        "title": "The Extremity Premium: Sentiment Regimes and Adverse Selection in Cryptocurrency Markets"
    },
    {
        "id": "2602.00776",
        "filename": "2602.00776_explainable_patterns_crypto_microstructure.pdf",
        "title": "Explainable Patterns in Cryptocurrency Microstructure: SHAP Dependence Across Assets"
    },
    {
        "id": "2608.17342",
        "filename": "2608.17342_mofe_fourier_neural_operators.pdf",
        "title": "MoFE: A Novel Mixture-of-Experts Framework with Fourier Neural Operators for Cryptocurrency Forecasting"
    },
    {
        "id": "2507.05470",
        "filename": "2507.05470_temporal_conformal_prediction_bitcoin.pdf",
        "title": "Temporal Conformal Prediction for Bitcoin Volatility and Uncertainty-Aware Intervals"
    },
    {
        "id": "2411.06327",
        "filename": "2411.06327_onchain_flows_exchange_liquidity.pdf",
        "title": "Return and Volatility Forecasting Using On-Chain Flows in Cryptocurrency Markets"
    }
]

print("=== DOWNLOADING OPEN-ACCESS QUANT RESEARCH PAPERS ===")
for p in PAPERS:
    out_path = os.path.join(JURNAL_DIR, p["filename"])
    url = f"https://export.arxiv.org/pdf/{p['id']}.pdf"
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        print(f"[ALREADY EXISTS] {p['filename']} ({os.path.getsize(out_path) // 1024} KB)")
        continue
    print(f"Downloading [{p['id']}] {p['title']}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(out_path, "wb") as f:
            f.write(resp.read())
        print(f"  -> Saved: {p['filename']} ({os.path.getsize(out_path) // 1024} KB)")
    except Exception as e:
        print(f"  -> Error downloading {p['id']}: {e}")

print("\nDone downloading papers.")
