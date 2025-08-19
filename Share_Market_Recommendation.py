import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime, timedelta

st.set_page_config(page_title="Indian Stock Suggestions", layout="wide")

# ----------------------------
# Utility Functions
# ----------------------------
def get_stock_data(symbols, period="6mo"):
    return yf.download(symbols, period=period, interval="1d", group_by="ticker", auto_adjust=True, threads=True)

def compute_indicators(df):
    df["ret_60"] = df["Close"].pct_change(60) * 100
    df["52w_high"] = df["Close"].rolling(252).max()
    df["pct_to_52w_high"] = ((df["52w_high"] - df["Close"]) / df["52w_high"]) * 100
    df["rsi14"] = compute_rsi(df["Close"], 14)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_reason(row, ranking_mode):
    try:
        rsi = row.get("rsi14", None)
        momentum = row.get("ret_60", None)
        pct_to_high = row.get("pct_to_52w_high", None)

        if rsi is not None and rsi < 30:
            return "RSI indicates the stock may be oversold (possible bounce)."
        elif rsi is not None and rsi > 70:
            return "RSI indicates the stock may be overbought (watch for correction)."

        if ranking_mode == "3M Momentum" and momentum is not None:
            return f"Strong positive momentum over last 3 months ({momentum:.2f}%)."
        elif ranking_mode == "Near 52W High" and pct_to_high is not None:
            return f"Trading near 52-week high ({pct_to_high:.2f}% away)."
        elif ranking_mode == "Composite Score":
            return "Selected based on composite scoring (valuation + momentum + RSI)."

        return "Promising stock based on available metrics."
    except Exception as e:
        return f"Reason unavailable (error: {e})"

def plot_sparkline(data):
    fig, ax = plt.subplots(figsize=(4, 1))
    ax.plot(data, color="blue", linewidth=1.5)
    ax.fill_between(range(len(data)), data, data.min(), color="blue", alpha=0.1)
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📈 Indian Stock Suggestion App")

universe = st.selectbox("Select Universe", ["NIFTY 50", "NIFTY 500"])
ranking_mode = st.selectbox("Ranking Mode", ["Composite Score", "3M Momentum", "Near 52W High"])
refresh_interval = st.slider("Auto-refresh interval (minutes)", 1, 30, 5)

# For demo purposes, fixed tickers (subset of NIFTY50)
tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

# Fetch and process
raw_data = get_stock_data(tickers)
processed = {}
for t in tickers:
    df = raw_data[t].copy()
    df = compute_indicators(df)
    processed[t] = df

latest_rows = []
for t, df in processed.items():
    row = df.iloc[-1].copy()
    row["Symbol"] = t
    row["Company Name"] = t.replace(".NS", "")
    latest_rows.append(row)

suggestions = pd.DataFrame(latest_rows)

# Composite score
suggestions["Composite Score"] = (
    suggestions["ret_60"].rank(pct=True) * 0.4 +
    (-suggestions["pct_to_52w_high"]).rank(pct=True) * 0.3 +
    (100 - suggestions["rsi14"]).rank(pct=True) * 0.3
)

# Ranking mode
if ranking_mode == "Composite Score":
    suggestions = suggestions.sort_values("Composite Score", ascending=False)
elif ranking_mode == "3M Momentum":
    suggestions = suggestions.sort_values("ret_60", ascending=False)
elif ranking_mode == "Near 52W High":
    suggestions = suggestions.sort_values("pct_to_52w_high", ascending=True)

st.subheader("📊 Top Stock Suggestions")
st.dataframe(
    suggestions[["Company Name", "Symbol", "Close", "Composite Score", "ret_60", "pct_to_52w_high", "rsi14"]],
    use_container_width=True
)

st.subheader("📌 Detailed Suggestions")
for _, row in suggestions.head(5).iterrows():
    with st.container():
        st.markdown(f"### {row['Company Name']} ({row['Symbol']})")
        st.write(f"**Last Price:** ₹{row['Close']:.2f}")
        st.write(f"**RSI (14):** {row['rsi14']:.2f}")
        st.write(f"**3M Return:** {row['ret_60']:.2f}%")
        st.write(f"**% to 52W High:** {row['pct_to_52w_high']:.2f}%")
        reason = get_reason(row.to_dict(), ranking_mode)
        st.write(f"**Reason:** {reason}")

        # Sparkline
        sparkline_buf = plot_sparkline(processed[row["Symbol"]]["Close"].tail(60))
        st.image(sparkline_buf, use_column_width=False)
