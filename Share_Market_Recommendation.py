import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta

st.set_page_config(page_title="Indian Share Recommendation App", layout="wide")

# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def fetch_stock_data(symbols, period="6mo"):
    data = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period=period)
            if not hist.empty:
                data[sym] = hist
        except Exception as e:
            st.warning(f"Failed to fetch {sym}: {e}")
    return data


def compute_indicators(hist):
    df = hist.copy()
    df["return"] = df["Close"].pct_change()
    df["cum_return"] = (1 + df["return"]).cumprod() - 1
    df["rolling_mean"] = df["Close"].rolling(20).mean()
    df["rolling_std"] = df["Close"].rolling(20).std()
    df["upper_band"] = df["rolling_mean"] + (df["rolling_std"] * 2)
    df["lower_band"] = df["rolling_mean"] - (df["rolling_std"] * 2)

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi14"] = 100 - (100 / (1 + rs))

    return df


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


def plot_sparkline(prices):
    fig, ax = plt.subplots(figsize=(4, 1))
    ax.plot(prices, color="blue")
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf


# --------------------------------------------------
# App Sidebar
# --------------------------------------------------
st.sidebar.header("Configuration")

universe = st.sidebar.radio("Select Universe", ["NIFTY 50", "NIFTY 500"])
ranking_mode = st.sidebar.radio("Ranking Mode", ["Composite Score", "3M Momentum", "Near 52W High"])
refresh_interval = st.sidebar.selectbox("Auto-refresh Interval (seconds)", [0, 30, 60, 120], index=2)

# Symbols for demo
if universe == "NIFTY 50":
    symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
else:
    symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
               "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ASIANPAINT.NS"]

# --------------------------------------------------
# Fetch & Process
# --------------------------------------------------
with st.spinner("Fetching stock data..."):
    data = fetch_stock_data(symbols, period="6mo")

results = []
for sym, hist in data.items():
    df = compute_indicators(hist)
    if df.empty:
        continue

    last = df["Close"].iloc[-1]
    ret_60 = (df["Close"].iloc[-1] / df["Close"].iloc[-60] - 1) * 100 if len(df) > 60 else np.nan
    pct_to_52w_high = (df["Close"].iloc[-1] / df["Close"].max() - 1) * 100
    rsi14 = df["rsi14"].iloc[-1]

    composite = 0
    if not np.isnan(ret_60):
        composite += ret_60
    if not np.isnan(rsi14):
        composite += (50 - abs(rsi14 - 50))
    if not np.isnan(pct_to_52w_high):
        composite += (-pct_to_52w_high)

    results.append({
        "Symbol": sym,
        "Last": last,
        "ret_60": ret_60,
        "pct_to_52w_high": pct_to_52w_high,
        "rsi14": rsi14,
        "Composite Score": composite,
        "History": df
    })

df_res = pd.DataFrame(results)

if ranking_mode == "Composite Score":
    suggestions = df_res.sort_values("Composite Score", ascending=False).head(5)
elif ranking_mode == "3M Momentum":
    suggestions = df_res.sort_values("ret_60", ascending=False).head(5)
elif ranking_mode == "Near 52W High":
    suggestions = df_res.sort_values("pct_to_52w_high", ascending=True).head(5)
else:
    suggestions = df_res.head(5)

# --------------------------------------------------
# Display
# --------------------------------------------------
st.header("📈 Suggested Stocks")

if not suggestions.empty:
    st.dataframe(
        suggestions[["Symbol", "Last", "Composite Score", "ret_60", "pct_to_52w_high", "rsi14"]],
        use_container_width=True  # ✅ fixed deprecation
    )

    st.subheader("Detailed Suggestions")

    for _, row in suggestions.iterrows():
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"### {row['Symbol']}")
            st.markdown(f"**Last Price:** {row['Last']:.2f}")
            st.markdown(f"**Reason:** {get_reason(row.to_dict(), ranking_mode)}")

        with col2:
            spark_buf = plot_sparkline(row["History"]["Close"].tail(60).values)
            st.image(spark_buf, use_container_width=True)  # ✅ fixed deprecation
else:
    st.warning("No stock suggestions available right now.")
