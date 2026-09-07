import pandas as pd
import numpy as np

df = pd.read_csv("C:/Mirza Personal/crypto quant/data/btc_coinbase_10y.csv", parse_dates=["open_time"]).sort_values("open_time").reset_index(drop=True)
c = df["close"]

df["rsi_90"] = (lambda s: 100 - (100 / (1 + s.diff().clip(lower=0).rolling(90).mean() / ((-s.diff().clip(upper=0)).rolling(90).mean() + 1e-9))))(c)
df["rsi_30"] = (lambda s: 100 - (100 / (1 + s.diff().clip(lower=0).rolling(30).mean() / ((-s.diff().clip(upper=0)).rolling(30).mean() + 1e-9))))(c)

ema_f = c.ewm(span=12, adjust=False).mean()
ema_s = c.ewm(span=26, adjust=False).mean()
df["macd"] = ema_f - ema_s
df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_hist"] = df["macd"] - df["macd_sig"]

df["ema_50"] = c.ewm(span=50, adjust=False).mean()
df["ema_200"] = c.ewm(span=200, adjust=False).mean()

latest = df.iloc[-1]
price = latest["close"]
rsi90 = latest["rsi_90"]
rsi30 = latest["rsi_30"]
macd_bull = latest["macd"] > latest["macd_sig"]
ema50 = latest["ema_50"]
ema200 = latest["ema_200"]

dev = (price - ema50) / ema50

if (price < ema200) and (not macd_bull):
    state = "WAIT"
    rec_exposure = 0.15
elif macd_bull and (price > ema50) and (rsi90 > 52):
    state = "RIDE"
    rec_exposure = 0.85
else:
    state = "ACCUMULATE"
    rec_exposure = float(np.clip(0.50 - dev * 1.5, 0.20, 0.80))

print(f"LATEST DATE: {latest['open_time'].date()}")
print(f"BTC SPOT PRICE: ${price:,.2f}")
print(f"EMA 50: ${ema50:,.2f} | EMA 200: ${ema200:,.2f}")
print(f"RSI 90: {rsi90:.2f} | RSI 30: {rsi30:.2f}")
print(f"MACD 1M Bullish: {macd_bull} (Hist: {latest['macd_hist']:+.2f})")
print(f"Deviasi vs EMA 50: {dev:+.2%}")
print(f"CURRENT REGIME STATE: {state}")
print(f"RECOMMENDED PORTFOLIO EXPOSURE: {rec_exposure:.1%}")

# Calculate immediate buy levels around EMA 50
print("\n--- ADAPTIVE GRID BUY LEVELS (ACCUMULATION) ---")
for pct, label in [(-0.03, "Grid 1 (Dip ringan)"), (-0.07, "Grid 2 (Dip menengah)"), (-0.12, "Grid 3 (Major support)"), (-0.18, "Grid 4 (Deep value)")]:
    target_p = ema50 * (1 + pct)
    print(f"  {label:24s}: ${target_p:,.0f} ({pct:+.1%} vs EMA50)")
